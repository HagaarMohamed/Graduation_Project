"""
train.py — Stage 2
Full training loop for EfficientNet-B4 classifier.

Strategy (2-phase):
  Phase 1 (epochs 1-5)  : Backbone frozen → train head only (fast convergence)
  Phase 2 (epoch 6+)    : Full fine-tuning with lower LR (backbone + head)
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from stage2_classification.model   import HieroglyphClassifier
from stage2_classification.dataset import build_dataloaders, extract_crops
from stage2_classification.utils   import (
    load_config,
    load_class_map,
    save_class_map,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    topk_accuracy,
)


def _one_epoch(
    phase: str,
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scaler,              # AMP GradScaler
) -> tuple[float, float, float]:
    """Run one training or validation epoch. Returns (loss, top1, top5)."""
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    loss_m  = AverageMeter("loss")
    top1_m  = AverageMeter("top1")
    top5_m  = AverageMeter("top5")

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bs     = images.size(0)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            top1 = topk_accuracy(logits, labels, k=1)
            top5 = topk_accuracy(logits, labels, k=5)

            loss_m.update(loss.item(), bs)
            top1_m.update(top1,       bs)
            top5_m.update(top5,       bs)

    return loss_m.avg, top1_m.avg, top5_m.avg


def run_training(resume: bool = False):
    cfg  = load_config()
    c2   = cfg["classification"]
    ds   = cfg["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage2] Device: {device}")

    # ── Class map ────────────────────────────────────────────────────────────
    fixed_yaml  = cfg["paths"]["fixed_yaml"]
    idx_to_code, code_to_idx = load_class_map(fixed_yaml)
    num_classes = len(idx_to_code)
    print(f"[Stage2] Classes: {num_classes}")

    class_map_path = os.path.join(c2["output_dir"], "class_map.json")
    save_class_map(idx_to_code, class_map_path)

    # ── Prepare crops (if not already done) ──────────────────────────────────
    crops_root  = c2["crops_dir"]
    train_split = os.path.join(crops_root, "train")
    if not os.path.exists(train_split):
        print("[Stage2] Crops not found — extracting from dataset...")
        from stage2_classification.utils import load_config as _lc
        extract_crops(
            dataset_root=_get_dataset_root(ds),
            output_dir=crops_root,
        )
    else:
        print(f"[Stage2] Using existing crops at: {crops_root}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        crops_root   = crops_root,
        code_to_idx  = code_to_idx,
        batch_size   = c2["batch_size"],
        imgsz        = c2["imgsz"],
        num_workers  = c2.get("num_workers", 0),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HieroglyphClassifier(
        num_classes = num_classes,
        dropout     = c2.get("dropout", 0.4),
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    # LabelSmoothing helps avoid over-confident predictions on 767 classes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Optimizers ────────────────────────────────────────────────────────────
    freeze_epochs = c2.get("freeze_epochs", 5)
    total_epochs  = c2["epochs"]

    head_optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=c2["lr_head"], weight_decay=1e-4
    )
    full_optimizer = AdamW(
        model.parameters(),
        lr=c2["lr_full"], weight_decay=1e-4
    )

    scheduler = CosineAnnealingLR(
        full_optimizer,
        T_max   = total_epochs - freeze_epochs,
        eta_min = 1e-6,
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_acc    = 0.0
    ckpt_path   = os.path.join(c2["output_dir"], "last.pt")
    best_path   = os.path.join(c2["output_dir"], "best.pt")

    if resume and os.path.exists(ckpt_path):
        start_epoch, best_acc = load_checkpoint(
            ckpt_path, model, full_optimizer, scheduler
        )

    os.makedirs(c2["output_dir"], exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[Stage2] Starting training: {total_epochs} epochs, "
          f"freeze_backbone for first {freeze_epochs}")

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()

        # Phase switching
        if epoch <= freeze_epochs:
            model.freeze_backbone()
            optimizer = head_optimizer
        else:
            model.unfreeze_backbone()
            optimizer = full_optimizer
            scheduler.step()

        # Train
        tr_loss, tr_top1, tr_top5 = _one_epoch(
            "train", model, train_loader, criterion, optimizer, device, scaler
        )
        # Validate
        vl_loss, vl_top1, vl_top5 = _one_epoch(
            "val", model, val_loader, criterion, optimizer, device, scaler
        )

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{total_epochs}] "
            f"| LR {lr_now:.2e} "
            f"| Train loss={tr_loss:.4f} top1={tr_top1:.3f} top5={tr_top5:.3f} "
            f"| Val   loss={vl_loss:.4f} top1={vl_top1:.3f} top5={vl_top5:.3f} "
            f"| {elapsed:.0f}s"
        )

        # Save checkpoints
        state = {
            "epoch"           : epoch,
            "model_state"     : model.state_dict(),
            "optimizer_state" : optimizer.state_dict(),
            "scheduler_state" : scheduler.state_dict(),
            "best_acc"        : best_acc,
        }
        save_checkpoint(state, ckpt_path)

        if vl_top1 > best_acc:
            best_acc = vl_top1
            save_checkpoint(state, best_path)
            print(f"  ★ New best val accuracy: {best_acc:.4f}")

    print(f"\n[Stage2] ✅ Training complete! Best val top-1: {best_acc:.4f}")
    print(f"[Stage2] Best weights: {best_path}")
    return best_path


def _get_dataset_root(ds_cfg: dict) -> str:
    """Locate dataset root from config (local or cached kagglehub)."""
    import glob
    if ds_cfg.get("local_root") and os.path.exists(ds_cfg["local_root"]):
        from stage2_classification.utils import locate_dataset_root
        from stage1_detection.utils import locate_dataset_root as _ldr
        return _ldr(ds_cfg["local_root"])
    # Search kagglehub cache
    matches = glob.glob(
        "/root/.cache/kagglehub/**/data_cleaned.yaml", recursive=True
    ) or glob.glob(
        os.path.expanduser("~/.cache/kagglehub/**/data_cleaned.yaml"),
        recursive=True
    )
    if matches:
        return os.path.dirname(matches[0])
    raise FileNotFoundError(
        "Could not find dataset. Run: python main.py train first."
    )
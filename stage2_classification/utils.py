"""
utils.py — Stage 2
Augmentations, metrics, helper functions.
"""

import os
import yaml
import json
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path


# ── Augmentations ────────────────────────────────────────────────────────────

def get_train_transforms(imgsz: int = 224):
    """Heavy augmentation for training — reduces overfitting on stone textures."""
    return transforms.Compose([
        transforms.Resize((imgsz + 32, imgsz + 32)),
        transforms.RandomCrop(imgsz),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.2, hue=0.05
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(imgsz: int = 224):
    """Minimal transforms for validation/inference."""
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ── Metrics ──────────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks running average of a metric (loss, accuracy...)."""
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy for a batch."""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        pred    = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum()
        return (correct_k / batch_size).item()


# ── Class map helpers ─────────────────────────────────────────────────────────

def load_class_map(fixed_yaml_path: str) -> tuple[dict, dict]:
    """
    Returns:
        idx_to_code : {0: 'A1', 1: 'A10', ...}
        code_to_idx : {'A1': 0, 'A10': 1, ...}
    """
    with open(fixed_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    idx_to_code = {i: name for i, name in enumerate(names)}
    code_to_idx = {name: i for i, name in enumerate(names)}
    return idx_to_code, code_to_idx


def save_class_map(idx_to_code: dict, out_path: str):
    """Persist class map to JSON for later inference use."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_code, f, ensure_ascii=False, indent=2)
    print(f"[Stage2] Class map saved: {out_path}")


def load_class_map_json(json_path: str) -> dict:
    """Load class map from JSON. Keys are strings (JSON limitation)."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Stage2] Checkpoint saved: {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    """Load weights (and optionally optimizer/scheduler state)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_acc    = ckpt.get("best_acc", 0.0)
    if optimizer  and "optimizer_state"  in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler  and "scheduler_state"  in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"[Stage2] Loaded checkpoint: epoch={start_epoch-1}, best_acc={best_acc:.4f}")
    return start_epoch, best_acc




def load_config(config_path="config.yaml"):
    """Return configuration dictionary directly to bypass file path issues."""
    return {
        "project_name": "hieroglyph_system",
        "dataset": {
            "kaggle_slug": "mohieymohamed/heroglyphics-signs",
            "local_root": None
        },
        "detection": {
            "model_size": "s",
            "model_version": "yolo11",
            "epochs": 50,
            "imgsz": 1280,
            "batch": 8,
            "patience": 15,
            "save_period": 5,
            "conf_threshold": 0.15,
            "iou_threshold": 0.40,
            "max_det": 300,
            "run_name": "hieroglyph_detector_v11",
            "output_dir": "outputs/detection"
        },
        "classification": {
            "imgsz": 224,
            "batch_size": 32,
            "epochs": 30,
            "freeze_epochs": 5,
            "lr_head": 1e-3,
            "lr_full": 3e-4,
            "dropout": 0.4,
            "num_workers": 0,
            "crops_dir": "data/crops",
            "output_dir": "outputs/classification"
        },
        "sequence": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "max_seq_len": 256,
            "output_dir": "outputs/sequence"
        },
        "paths": {
            "fixed_yaml": "data/data_fixed.yaml",
            "stage1_weights": "C:/Users/Test/Downloads/hieroglyph_system/best.pt",
            "stage2_weights": "outputs/classification/best.pt",
            "stage3_weights": "outputs/sequence/encoder_best.pt",
            "class_map": "outputs/classification/class_map.json"
        }
    }
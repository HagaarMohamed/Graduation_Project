"""
train.py — Stage 1
Trains (or resumes) a YOLO11 model on the hieroglyph dataset.
"""

import os
import torch
from ultralytics import YOLO

from stage1_detection.utils import (
    load_config,
    download_dataset,
    locate_dataset_root,
    fix_yaml,
    get_latest_checkpoint,
)


def run_training(resume: bool = False):
    """
    Full training pipeline:
      1. Load config
      2. Download (or locate) dataset
      3. Fix YAML paths
      4. Train YOLO11
    """
    cfg = load_config()
    det = cfg["detection"]
    ds  = cfg["dataset"]

    # ── GPU check ─────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[Stage1] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Stage1] WARNING: No GPU found — training on CPU will be very slow.")

    # ── Dataset ───────────────────────────────────────────────
    fixed_yaml = cfg["paths"]["fixed_yaml"]

    if ds.get("local_root") and os.path.exists(str(ds["local_root"])):
        dataset_root = locate_dataset_root(ds["local_root"])
    else:
        dataset_root = download_dataset(ds["kaggle_slug"])

    fix_yaml(dataset_root, fixed_yaml)

    # ── Model tag (YOLO11 or YOLOv8) ─────────────────────────
    version   = det.get("model_version", "yolo11")
    size      = det.get("model_size", "n")
    model_tag = f"{version}{size}.pt"

    # ── Resume or fresh start ─────────────────────────────────
    checkpoint = None
    if resume:
        checkpoint = get_latest_checkpoint(
            det["run_name"], det["output_dir"]
        )
        if checkpoint:
            print(f"[Stage1] Resuming from: {checkpoint}")
        else:
            print("[Stage1] No checkpoint found — starting fresh.")

    # ── Training ──────────────────────────────────────────────
    if checkpoint and resume:
        print(f"[Stage1] Resuming training...")
        model = YOLO(checkpoint)
        model.train(resume=True)
    else:
        print(f"[Stage1] Starting fresh training with: {model_tag}")
        model = YOLO(model_tag)
        model.train(
            data        = fixed_yaml,
            epochs      = int(det["epochs"]),
            imgsz       = int(det.get("imgsz", 640)),
            batch       = int(det["batch"]),
            patience    = int(det["patience"]),
            save        = True,
            save_period = int(det["save_period"]),
            name        = det["run_name"],
            project     = det["output_dir"],
            device      = device,
            workers     = 0,      # 0 is safest on Windows
            verbose     = True,
            iou         = float(det.get("iou_threshold", 0.40)),
        )

    best = os.path.join(
        det["output_dir"], det["run_name"], "weights", "best.pt"
    )
    print(f"\n[Stage1] Training complete!")
    print(f"[Stage1] Best weights: {best}")
    return best
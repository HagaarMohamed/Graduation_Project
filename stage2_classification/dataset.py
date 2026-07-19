"""
dataset.py — Stage 2
Converts the YOLO-format dataset into a crop-based classification dataset.

The YOLO dataset has:
  train/images/  + train/labels/  (each label file: class_id cx cy w h per line)

We crop every bounding box → single-symbol image → feed to EfficientNet-B4.
Crops are cached to disk so we only do this once.
"""

import os
import cv2
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from stage2_classification.utils import (
    get_train_transforms,
    get_val_transforms,
    load_class_map,
)


# ── 1. Crop extraction (one-time preprocessing) ───────────────────────────────

def extract_crops(
    dataset_root: str,
    output_dir: str,
    splits: list[str] = ["train", "valid", "test"],
    min_px: int = 16,
) -> dict[str, int]:
    """
    Reads YOLO label files and saves individual symbol crops.

    Output structure:
        output_dir/
          train/
            A1/   ← Gardiner code (class name)
              img001_box0.jpg
              ...
            A10/
              ...
          valid/
            ...

    Args:
        dataset_root : folder containing train/, valid/, test/
        output_dir   : where to write crops
        splits       : which splits to process
        min_px       : discard crops smaller than min_px×min_px

    Returns:
        dict {split: n_crops}
    """
    # Load class names from YAML
    yaml_path = os.path.join(dataset_root, "data_cleaned.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    totals = {}

    for split in splits:
        img_dir = os.path.join(dataset_root, split, "images")
        lbl_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.exists(img_dir):
            print(f"[Stage2] Skipping {split} (not found)")
            continue

        image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        n_crops = 0
        for img_file in tqdm(image_files, desc=f"[Stage2] Cropping {split}"):
            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(
                lbl_dir,
                Path(img_file).stem + ".txt"
            )
            if not os.path.exists(lbl_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]

            with open(lbl_path, "r") as f:
                lines = f.read().strip().splitlines()

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                if cls_id >= len(names):
                    continue  # skip labels beyond nc (known issue in valid split)

                cx, cy, bw, bh = map(float, parts[1:5])

                # Convert YOLO normalized → pixel coords
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)

                # Clamp to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                crop_w = x2 - x1
                crop_h = y2 - y1
                if crop_w < min_px or crop_h < min_px:
                    continue

                crop = img[y1:y2, x1:x2]
                code = names[cls_id]

                # Save crop
                save_dir = os.path.join(output_dir, split, code)
                os.makedirs(save_dir, exist_ok=True)
                stem = Path(img_file).stem
                save_path = os.path.join(save_dir, f"{stem}_b{i}.jpg")
                cv2.imwrite(save_path, crop)
                n_crops += 1

        totals[split] = n_crops
        print(f"[Stage2] {split}: {n_crops:,} crops extracted ✓")

    return totals


# ── 2. PyTorch Dataset ────────────────────────────────────────────────────────

class HieroglyphCropDataset(Dataset):
    """
    Loads pre-extracted crops from:
        crops_root/
          <split>/
            <GardinerCode>/
              *.jpg
    """

    def __init__(
        self,
        crops_root: str,
        split: str,
        code_to_idx: dict[str, int],
        transform=None,
    ):
        self.transform   = transform
        self.code_to_idx = code_to_idx
        self.samples: list[tuple[str, int]] = []  # (img_path, class_idx)

        split_dir = os.path.join(crops_root, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Crops not found at: {split_dir}\n"
                "Run: python main.py prepare-crops first."
            )

        for code in os.listdir(split_dir):
            idx = code_to_idx.get(code)
            if idx is None:
                continue
            class_dir = os.path.join(split_dir, code)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(class_dir, fname), idx)
                    )

        print(f"[Stage2] {split} dataset: {len(self.samples):,} samples, "
              f"{len(self.code_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── 3. DataLoader factory ─────────────────────────────────────────────────────

def build_dataloaders(
    crops_root: str,
    code_to_idx: dict[str, int],
    batch_size: int = 32,
    imgsz: int = 224,
    num_workers: int = 0,   # 0 is safest on Windows
) -> tuple[DataLoader, DataLoader]:

    train_ds = HieroglyphCropDataset(
        crops_root, "train", code_to_idx,
        transform=get_train_transforms(imgsz)
    )
    val_ds = HieroglyphCropDataset(
        crops_root, "valid", code_to_idx,
        transform=get_val_transforms(imgsz)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
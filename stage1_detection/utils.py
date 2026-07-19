"""
utils.py — Stage 1
Utility functions: dataset download, YAML fixing, path resolution.
"""

import os
import glob
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the global project config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_dataset(slug: str) -> str:
    """
    Download the Kaggle dataset using kagglehub.
    Returns the local root path containing data_cleaned.yaml.

    Prerequisites:
      Place kaggle.json in C:/Users/<you>/.kaggle/kaggle.json
      OR set env vars KAGGLE_USERNAME and KAGGLE_KEY.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError("Run: pip install kagglehub")

    print(f"[Stage1] Downloading dataset: {slug}")
    raw_path = kagglehub.dataset_download(slug)
    print(f"[Stage1] Raw download path: {raw_path}")
    return locate_dataset_root(raw_path)


def locate_dataset_root(search_path: str) -> str:
    """
    Recursively find the folder that directly contains data_cleaned.yaml.
    """
    matches = glob.glob(
        os.path.join(search_path, "**", "data_cleaned.yaml"),
        recursive=True
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find data_cleaned.yaml under: {search_path}\n"
            "Make sure the dataset downloaded correctly."
        )
    root = str(Path(matches[0]).parent)
    print(f"[Stage1] Dataset root: {root}")
    return root


def fix_yaml(dataset_root: str, output_yaml_path: str) -> str:
    """
    Fix relative paths inside data_cleaned.yaml so YOLOv8 can
    find images regardless of where kagglehub stored them.
    Saves a corrected copy to output_yaml_path.
    Returns the path to the fixed YAML.
    """
    original_yaml = os.path.join(dataset_root, "data_cleaned.yaml")
    if not os.path.exists(original_yaml):
        raise FileNotFoundError(f"data_cleaned.yaml not found at: {original_yaml}")

    with open(original_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override paths with absolute references
    cfg["path"]  = dataset_root
    cfg["train"] = "train/images"
    cfg["val"]   = "valid/images"
    cfg["test"]  = "test/images"

    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"[Stage1] Fixed YAML saved to: {output_yaml_path}")
    _verify_splits(dataset_root)
    return output_yaml_path


def _verify_splits(dataset_root: str):
    """Print image counts for each data split."""
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_root, split, "images")
        if os.path.exists(img_dir):
            n = len(os.listdir(img_dir))
            print(f"  [Stage1] {split:6s}: {n:,} images  ✓")
        else:
            print(f"  [Stage1] {split:6s}: NOT FOUND  ✗")


def get_latest_checkpoint(run_name: str, output_dir: str) -> str | None:
    """
    Find the most recently modified last.pt across all run folders
    matching run_name. Returns None if nothing found.
    """
    pattern = os.path.join(output_dir, f"{run_name}*", "weights", "last.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def get_class_names(fixed_yaml_path: str) -> list[str]:
    """Return list of class names (Gardiner codes) from the fixed YAML."""
    with open(fixed_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names
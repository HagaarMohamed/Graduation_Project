"""
detect.py — Stage 1
Runs YOLOv8 inference on a single image and returns structured detections.
Output: list of dicts {code, conf, x, y, w, h, bbox}
"""

import os
import cv2
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO


def _load_det_config() -> dict:
    """Load detection config from project root."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path  = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["detection"]


def run_detection(
    image_path : str,
    weights_path: str,
    conf       : float = None,
    save       : bool  = True,
    output_dir : str   = "outputs/detection/inference",
) -> list[dict]:
    """
    Detect hieroglyphic signs in image_path.
    Returns a list of detections sorted left-to-right (by x center).

    Each detection dict:
        code  : Gardiner code string (e.g. "G17")
        conf  : confidence score 0-1
        x, y  : center coordinates (pixels)
        w, h  : box width / height (pixels)
        bbox  : [x1, y1, x2, y2] absolute pixel coords
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Run training first: python main.py train"
        )

    # ── Load config ───────────────────────────────────────────
    det_cfg       = _load_det_config()
    conf          = conf or float(det_cfg.get("conf_threshold", 0.15))
    iou_threshold = float(det_cfg.get("iou_threshold",  0.40))
    imgsz         = int(det_cfg.get("imgsz",            1280))
    max_det       = int(det_cfg.get("max_det",           300))

    print(f"[Stage1] conf={conf}  iou={iou_threshold}  imgsz={imgsz}  max_det={max_det}")

    model = YOLO(weights_path)

    results = model.predict(
        source       = image_path,
        conf         = conf,
        iou          = iou_threshold,
        imgsz        = imgsz,
        augment      = True,      # Test Time Augmentation — catches more signs
        agnostic_nms = True,      # NMS across all classes — removes duplicates
        max_det      = max_det,
        save         = save,
        project      = output_dir,
        name         = "run",
        exist_ok     = True,
        verbose      = False,
    )

    result = results[0]
    names  = result.names  # {class_id: "GardinerCode"}

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        xywh   = box.xywh[0].tolist()
        xyxy   = box.xyxy[0].tolist()
        detections.append({
            "code" : names[cls_id],
            "conf" : round(float(box.conf[0]), 4),
            "x"    : round(xywh[0], 1),
            "y"    : round(xywh[1], 1),
            "w"    : round(xywh[2], 1),
            "h"    : round(xywh[3], 1),
            "bbox" : [round(v, 1) for v in xyxy],
        })

    # ── Remove duplicates (same sign, same location) ──────────
    detections = _remove_duplicates(detections, iou_threshold=0.5)

    # ── Sort left-to-right ────────────────────────────────────
    detections.sort(key=lambda d: d["x"])

    print(f"[Stage1] Detected {len(detections)} sign(s) in: {os.path.basename(image_path)}")
    for d in detections:
        print(f"  {d['code']:8s}  conf={d['conf']:.2f}  pos=({d['x']:.0f},{d['y']:.0f})")

    return detections


def _remove_duplicates(detections: list, iou_threshold: float = 0.5) -> list:
    """
    Remove duplicate detections of the SAME sign at the SAME location.
    Keeps same sign at DIFFERENT locations (different sentences).
    """
    if not detections:
        return detections

    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0:
            return 0.0
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter)

    sorted_dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    kept = []

    for det in sorted_dets:
        duplicate = False
        for k in kept:
            if det["code"] == k["code"]:
                if iou(det["bbox"], k["bbox"]) > iou_threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(det)

    removed = len(detections) - len(kept)
    if removed > 0:
        print(f"[Stage1] Removed {removed} duplicate(s) → {len(kept)} unique signs")

    return kept


def visualize_detections(image_path: str, detections: list[dict]):
    """
    Draw bounding boxes + Gardiner codes on the image and show it.
    """
    img = cv2.imread(image_path)
    if img is None:
        # fallback for webp or unusual formats
        from PIL import Image
        import numpy as np
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(img)

    colors = plt.cm.tab20.colors
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        w = x2 - x1
        h = y2 - y1
        color = colors[i % len(colors)]

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5,
            f"{d['code']} {d['conf']:.0%}",
            fontsize=7, color="white",
            bbox=dict(facecolor=color, alpha=0.8, pad=1)
        )

    ax.set_title(
        f"Stage 1 — Detected {len(detections)} Hieroglyphic Signs",
        fontsize=13
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()
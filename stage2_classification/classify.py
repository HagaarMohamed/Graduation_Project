"""
classify.py — Stage 2
Inference: takes image crops from Stage 1 detections,
returns refined Gardiner codes with confidence scores.
"""

import os
import torch
import numpy as np
from PIL import Image

from stage2_classification.model import HieroglyphClassifier
from stage2_classification.utils import (
    get_val_transforms,
    load_class_map_json,
    load_config,
)


class HieroglyphClassifierInference:
    """
    Wraps EfficientNet-B4 for single-image and batch inference.
    Usage:
        clf = HieroglyphClassifierInference()
        result = clf.classify_crop(pil_image)
        # → {"code": "G17", "conf": 0.97, "top5": [...]}
    """

    def __init__(
        self,
        weights_path: str = None,
        class_map_path: str = None,
        device: str = None,
    ):
        cfg = load_config()
        c2  = cfg["classification"]

        self.weights_path   = weights_path   or cfg["paths"]["stage2_weights"]
        self.class_map_path = class_map_path or os.path.join(
            c2["output_dir"], "class_map.json"
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.transform = get_val_transforms(c2["imgsz"])

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Stage 2 weights not found: {self.weights_path}\n"
                "Run: python main.py train-clf"
            )
        if not os.path.exists(self.class_map_path):
            raise FileNotFoundError(
                f"Class map not found: {self.class_map_path}\n"
                "Run: python main.py train-clf"
            )

        self.idx_to_code = load_class_map_json(self.class_map_path)
        num_classes      = len(self.idx_to_code)

        self.model = HieroglyphClassifier(num_classes=num_classes).to(self.device)
        ckpt = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[Stage2] Classifier loaded: {num_classes} classes, device={self.device}")

    def classify_crop(self, crop: Image.Image, top_k: int = 5) -> dict:
        """
        Args:
            crop : PIL Image (RGB) — a single hieroglyph crop
            top_k: how many top predictions to return

        Returns:
            {
              "code"  : "G17",          # top-1 Gardiner code
              "conf"  : 0.972,          # top-1 confidence
              "top5"  : [               # list of (code, conf)
                  ("G17", 0.972),
                  ("G18", 0.015),
                  ...
              ]
            }
        """
        img_t = self.transform(crop.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_t)
            probs  = torch.softmax(logits, dim=1)[0]

        top_k_probs, top_k_idxs = probs.topk(top_k)
        top_k_probs = top_k_probs.cpu().tolist()
        top_k_idxs  = top_k_idxs.cpu().tolist()

        top5 = [
            (self.idx_to_code[idx], round(p, 4))
            for idx, p in zip(top_k_idxs, top_k_probs)
        ]

        return {
            "code" : top5[0][0],
            "conf" : top5[0][1],
            "top5" : top5,
        }

    def classify_detections(
        self,
        image_path: str,
        detections: list[dict],
    ) -> list[dict]:
        """
        Takes a full image path + Stage 1 detections,
        crops each bbox, classifies it, and returns enriched detections.

        Each detection gets two new fields:
            clf_code : refined Gardiner code from EfficientNet
            clf_conf : EfficientNet confidence
            clf_top5 : top-5 predictions
        """
        from PIL import Image as PILImage
        import numpy as np

        img = PILImage.open(image_path).convert("RGB")
        W, H = img.size

        enriched = []
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            crop = img.crop((x1, y1, x2, y2))
            clf  = self.classify_crop(crop)

            enriched.append({
                **d,
                "clf_code" : clf["code"],
                "clf_conf" : clf["conf"],
                "clf_top5" : clf["top5"],
            })

        return enriched
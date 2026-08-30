"""
classify.py — Stage 2
Inference: takes image crops from Stage 1 detections,
returns refined Gardiner codes with confidence scores.
"""

import os
import sys
import torch
from PIL import Image

from stage2_classification.model import HieroglyphClassifier
from stage2_classification.utils import (
    get_val_transforms,
    load_class_map_json,
    load_config,
)


class HieroglyphClassifierInference:

    def __init__(
        self,
        weights_path   : str = None,
        class_map_path : str = None,
        device         : str = None,
    ):
        cfg = load_config()
        c2  = cfg["classification"]

        self.weights_path   = weights_path   or cfg["paths"]["stage2_weights"]
        self.class_map_path = class_map_path or os.path.join(
            c2["output_dir"], "class_map.json"
        )
        self.device    = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.transform = get_val_transforms(int(c2["imgsz"]))
        self.model     = None
        self.idx_to_code = {}
        self._load_model()

    def _load_model(self):
        # ── Check files ───────────────────────────────────────
        if not os.path.exists(self.weights_path):
            print(f"[Stage2] WARNING: Weights not found: {self.weights_path}")
            print("[Stage2] FALLBACK MODE: using Stage 1 codes directly.")
            return

        if not os.path.exists(self.class_map_path):
            print(f"[Stage2] WARNING: Class map not found: {self.class_map_path}")
            print("[Stage2] FALLBACK MODE: using Stage 1 codes directly.")
            return

        # ── Load class map ────────────────────────────────────
        self.idx_to_code = load_class_map_json(self.class_map_path)
        num_classes      = len(self.idx_to_code)

        # ── Build model ───────────────────────────────────────
        model = HieroglyphClassifier(num_classes=num_classes).to(self.device)

        # ── Load checkpoint ───────────────────────────────────
        ckpt = torch.load(
            self.weights_path,
            map_location = self.device,
            weights_only = False,
        )

        # Extract state dict regardless of format
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            # Check if it looks like a state dict directly
            first_val = next(iter(ckpt.values()))
            if isinstance(first_val, torch.Tensor):
                state_dict = ckpt
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # ── Load weights ──────────────────────────────────────
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            print(f"[Stage2] ✅ Classifier loaded successfully!")
            print(f"[Stage2] Classes={num_classes}, device={self.device}")
            self.model = model
            self.model.eval()

        except RuntimeError as e:
            print(f"[Stage2] ERROR loading weights: {e}")
            print("[Stage2] FALLBACK MODE: using Stage 1 codes directly.")
            self.model = None

    @property
    def is_fallback(self) -> bool:
        return self.model is None

    def classify_crop(self, crop: Image.Image, top_k: int = 5) -> dict:
        """Classify a single PIL Image crop."""
        img_t = self.transform(
            crop.convert("RGB")
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_t)
            probs  = torch.softmax(logits, dim=1)[0]

        actual_k             = min(top_k, probs.size(0))
        top_k_probs, top_k_idxs = probs.topk(actual_k)
        top_k_probs = top_k_probs.cpu().tolist()
        top_k_idxs  = top_k_idxs.cpu().tolist()

        top5 = [
            (self.idx_to_code[idx], round(p, 4))
            for idx, p in zip(top_k_idxs, top_k_probs)
        ]
        return {"code": top5[0][0], "conf": top5[0][1], "top5": top5}

    def classify_detections(
        self,
        image_path : str,
        detections : list,
    ) -> list:
        """
        Run Stage 2 classification on each detected crop.
        Falls back to Stage 1 codes if model unavailable.
        """
        # ── FALLBACK ──────────────────────────────────────────
        if self.is_fallback:
            print("[Stage2] FALLBACK MODE: copying Stage 1 codes to clf_code")
            return [
                {
                    **d,
                    "clf_code" : d["code"],
                    "clf_conf" : d["conf"],
                    "clf_top5" : [(d["code"], d["conf"])],
                }
                for d in detections
            ]

        # ── NORMAL ────────────────────────────────────────────
        img  = Image.open(image_path).convert("RGB")
        W, H = img.size

        enriched = []
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(W, x2);  y2 = min(H, y2)

            if x2 <= x1 or y2 <= y1:
                enriched.append({
                    **d,
                    "clf_code" : d["code"],
                    "clf_conf" : d["conf"],
                    "clf_top5" : [(d["code"], d["conf"])],
                })
                continue

            crop = img.crop((x1, y1, x2, y2))
            clf  = self.classify_crop(crop)
            enriched.append({
                **d,
                "clf_code" : clf["code"],
                "clf_conf" : clf["conf"],
                "clf_top5" : clf["top5"],
            })

        return enriched
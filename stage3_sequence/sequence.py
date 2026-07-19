"""
sequence.py — Stage 3
Main pipeline: takes Stage 2 enriched detections →
  1. Determines reading order automatically
  2. Tokenizes signs
  3. Runs Transformer Encoder
  4. Groups into words
  5. Returns StructuredSequence ready for Stage 4
"""

import os
import json
from dataclasses import dataclass, field
from PIL import Image

from stage3_sequence.reading_order import order_detections, Detection
from stage3_sequence.embeddings    import GardinerTokenizer
from stage3_sequence.encoder       import SequenceEncoderInference
from stage3_sequence.utils         import load_config, load_code_to_idx


@dataclass
class HieroglyphWord:
    """A group of signs that form one word/phonetic unit."""
    signs      : list = field(default_factory=list)
    sign_types : list = field(default_factory=list)
    confs      : list = field(default_factory=list)
    bboxes     : list = field(default_factory=list)

    @property
    def phonetic_signs(self):
        return [
            s for s, t in zip(self.signs, self.sign_types)
            if t in ("phonogram", "ideogram")
        ]

    @property
    def determinatives(self):
        return [s for s, t in zip(self.signs, self.sign_types)
                if t == "determinative"]

    def __repr__(self):
        return f"Word({'+'.join(self.signs)})"


@dataclass
class StructuredSequence:
    """Complete structured output of Stage 3."""
    direction : str
    words     : list
    all_codes : list
    all_types : list
    all_confs : list
    lines     : list

    def to_dict(self) -> dict:
        return {
            "direction" : self.direction,
            "all_codes" : self.all_codes,
            "all_types" : self.all_types,
            "all_confs" : self.all_confs,
            "lines"     : self.lines,
            "words"     : [
                {
                    "signs"          : w.signs,
                    "sign_types"     : w.sign_types,
                    "phonetic"       : w.phonetic_signs,
                    "determinatives" : w.determinatives,
                }
                for w in self.words
            ],
        }

    def pretty_print(self):
        print("\n" + "=" * 65)
        print(f"  Stage 3 — Structured Sequence  [{self.direction.upper()}]")
        print("=" * 65)
        print(f"\n  {'CODE':10s} {'TYPE':15s} {'CONF':6s}")
        print("  " + "-" * 35)
        for code, stype, conf in zip(self.all_codes, self.all_types, self.all_confs):
            print(f"  {code:10s} {stype:15s} {conf:.2%}")
        print(f"\n  Words ({len(self.words)}):")
        for i, w in enumerate(self.words):
            phonetic = "+".join(w.phonetic_signs) or "(no phonetic signs)"
            dets     = ",".join(w.determinatives) or "—"
            print(f"  [{i+1}] phonetic={phonetic}  det={dets}")
        print()


def _detect_layout(detections: list, img_w: float, img_h: float) -> str:
    """
    Automatically detect reading direction from image dimensions
    and sign positions — no user input needed.

    Returns: "ttb" | "ltr" | "rtl"
    """
    if not detections:
        return "ltr"

    # ── Image aspect ratio ────────────────────────────────────
    # Portrait image (taller than wide) → likely vertical columns
    if img_h > img_w * 1.2:
        print("[Stage3] Portrait image detected → assuming vertical columns (TTB)")
        return "ttb"

    # ── Sign spread analysis ──────────────────────────────────
    xs = [d["x"] for d in detections]
    ys = [d["y"] for d in detections]
    x_range = max(xs) - min(xs) if xs else 1
    y_range = max(ys) - min(ys) if ys else 1

    # If signs spread more vertically → top-to-bottom columns
    if y_range > x_range * 1.5:
        print("[Stage3] Signs spread vertically → TTB column order")
        return "ttb"

    # ── Directional sign voting ───────────────────────────────
    # (from reading_order.py logic)
    from stage3_sequence.reading_order import _vote_direction
    det_objs = [
        Detection(
            code     = d.get("code", ""),
            conf     = d.get("conf", 0.0),
            clf_code = d.get("clf_code", d.get("code", "")),
            clf_conf = d.get("clf_conf", d.get("conf", 0.0)),
            x=d["x"], y=d["y"], w=d["w"], h=d["h"],
            bbox=d["bbox"],
        )
        for d in detections
    ]
    direction = _vote_direction(det_objs)
    print(f"[Stage3] Sign voting result → {direction.upper()}")
    return direction


def _order_ttb(detections: list, img_w: float) -> tuple:
    """
    Order detections for top-to-bottom column reading.
    Groups signs into columns by x position, then sorts each column by y.
    """
    if not detections:
        return [], []

    # Estimate column width from average sign width
    avg_w = sum(d["w"] for d in detections) / len(detections)
    col_width = max(avg_w * 2.5, img_w / 10)

    # Group into columns by x position
    columns = {}
    for d in detections:
        col_idx = int(d["x"] / col_width)
        if col_idx not in columns:
            columns[col_idx] = []
        columns[col_idx].append(d)

    # Sort each column top-to-bottom
    ordered_dets = []
    lines_codes  = []
    for col_idx in sorted(columns.keys()):
        col = sorted(columns[col_idx], key=lambda d: d["y"])
        ordered_dets.extend(col)
        lines_codes.append([d.get("clf_code") or d.get("code") for d in col])

    print(f"[Stage3] Found {len(columns)} column(s)")
    for i, line in enumerate(lines_codes):
        print(f"  Column {i+1}: {' '.join(line)}")

    # Convert to Detection objects
    det_objs = [
        Detection(
            code     = d.get("code", ""),
            conf     = d.get("conf", 0.0),
            clf_code = d.get("clf_code", d.get("code", "")),
            clf_conf = d.get("clf_conf", d.get("conf", 0.0)),
            x=d["x"], y=d["y"], w=d["w"], h=d["h"],
            bbox=d["bbox"],
        )
        for d in ordered_dets
    ]

    return det_objs, lines_codes


def _group_into_words(codes, sign_types, boundaries, confs, bboxes):
    """Split flat sign list into word groups using word_boundary flags."""
    words   = []
    current = HieroglyphWord()

    for code, stype, boundary, conf, bbox in zip(
        codes, sign_types, boundaries, confs, bboxes
    ):
        if boundary and current.signs:
            words.append(current)
            current = HieroglyphWord()
        current.signs.append(code)
        current.sign_types.append(stype)
        current.confs.append(conf)
        current.bboxes.append(bbox)

    if current.signs:
        words.append(current)

    return words


class Stage3Pipeline:
    """
    End-to-end Stage 3 processor.
    Input : list of dicts from Stage 1+2
    Output: StructuredSequence
    """

    def __init__(self, image_path: str = None):
        cfg  = load_config()
        c3   = cfg["sequence"]
        self.cfg = c3

        # Image dimensions for spatial normalization
        self.img_w = self.img_h = 640.0
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                self.img_w, self.img_h = float(img.size[0]), float(img.size[1])
            except Exception:
                pass

        # Class map
        class_map_path = cfg["paths"]["class_map"]
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(
                f"Class map not found: {class_map_path}\n"
                "Run: python main.py train-clf"
            )
        self.code_to_idx = load_code_to_idx(class_map_path)
        vocab_size       = len(self.code_to_idx)

        # Tokenizer
        self.tokenizer = GardinerTokenizer(
            code_to_idx  = self.code_to_idx,
            image_width  = self.img_w,
            image_height = self.img_h,
        )

        # Encoder
        self.encoder = SequenceEncoderInference(
            vocab_size   = vocab_size,
            weights_path = cfg["paths"].get("stage3_weights"),
            d_model      = c3["d_model"],
            nhead        = c3["nhead"],
            num_layers   = c3["num_layers"],
        )

    def process(self, detections: list) -> StructuredSequence:
        """
        Full Stage 3 processing — fully automatic, no user input needed.
        """
        if not detections:
            print("[Stage3] No detections to process.")
            return StructuredSequence(
                direction="ltr", words=[], all_codes=[],
                all_types=[], all_confs=[], lines=[]
            )

        # ── 1. Auto-detect reading direction ─────────────────
        direction = _detect_layout(detections, self.img_w, self.img_h)

        # ── 2. Order signs based on direction ─────────────────
        if direction == "ttb":
            ordered, line_codes = _order_ttb(detections, self.img_w)
        else:
            ordered, direction, raw_lines = order_detections(detections)
            line_codes = [
                [s.clf_code or s.code for s in line]
                for line in raw_lines
            ]

        # ── 3. Tokenize ───────────────────────────────────────
        tokenized = self.tokenizer.encode(ordered)

        # ── 4. Transformer Encoder ────────────────────────────
        encoded = self.encoder.encode_sequence(tokenized)

        # ── 5. Extract per-sign data ──────────────────────────
        codes  = encoded["codes"]
        stypes = encoded["sign_types"]
        wbs    = encoded["word_boundaries"]
        confs  = [
            d.clf_conf if hasattr(d, "clf_conf") else d.conf
            for d in ordered
        ]
        bboxes = [d.bbox for d in ordered]

        # ── 6. Group into words ───────────────────────────────
        words = _group_into_words(codes, stypes, wbs, confs, bboxes)

        seq = StructuredSequence(
            direction = direction,
            words     = words,
            all_codes = codes,
            all_types = stypes,
            all_confs = confs,
            lines     = line_codes,
        )

        seq.pretty_print()
        return seq

    def save(self, seq: StructuredSequence, out_path: str):
        """Save structured sequence to JSON for Stage 4."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(seq.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[Stage3] Sequence saved: {out_path}")
"""
reading_order.py — Stage 3

Hieroglyphic reading order determination:

Ancient Egyptian hieroglyphs follow specific directional rules:
  1. Figures (humans, animals, birds) FACE the beginning of the text.
  2. So if signs face RIGHT  → read LEFT  to RIGHT  (→)
     If signs face LEFT   → read RIGHT to LEFT   (←)
  3. Vertical columns are read top-to-bottom.

We approximate gaze direction using asymmetric bounding box analysis
and a lookup table of known directional signs.

References:
  - Gardiner, "Egyptian Grammar" (1927), §2 Reading Order
  - Collier & Manley, "How to Read Egyptian Hieroglyphs"
"""

import os
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    """Single detection from Stage 1+2."""
    code    : str          # Gardiner code (e.g. "G17")
    conf    : float
    clf_code: str          # EfficientNet-refined code
    clf_conf: float
    x       : float        # x-center (pixels)
    y       : float        # y-center (pixels)
    w       : float
    h       : float
    bbox    : list         # [x1, y1, x2, y2]


# ── Known directional sign table ──────────────────────────────────────────────
# Signs that face LEFT by default (reading → right-to-left when these appear)
SIGNS_FACE_LEFT = {
    "A1","A2","A3","A4","A5","A6","A7","A8",
    "B1","B2","B3","B4","B5","B6","B7",
    "C1","C2","C3","C4","C5",
    "E1","E2","E3","E4","E5","E6","E7","E8","E9","E10",
    "G1","G2","G3","G4","G5","G6","G7","G17","G18","G36",
    "G37","G38","G39","G40","G41","G43",
    "I9","I10",
}

# Signs that face RIGHT by default (reading → left-to-right)
SIGNS_FACE_RIGHT = {
    "A14","A15","A24","A25","A26","A28","A30",
    "D1","D2","D3","D4","D21","D36","D46","D58",
}


def _vote_direction(detections: list[Detection]) -> str:
    """
    Cast votes based on known directional signs.
    Returns: "ltr" | "rtl" | "ttb" (left-to-right, right-to-left, top-to-bottom)
    """
    ltr_votes = 0
    rtl_votes = 0

    for d in detections:
        code = d.clf_code or d.code
        if code in SIGNS_FACE_RIGHT:
            ltr_votes += 1
        elif code in SIGNS_FACE_LEFT:
            rtl_votes += 1

    # If no directional votes, check aspect ratio of bounding box cluster
    if ltr_votes == 0 and rtl_votes == 0:
        return _infer_from_layout(detections)

    return "ltr" if ltr_votes >= rtl_votes else "rtl"


def _infer_from_layout(detections: list[Detection]) -> str:
    """
    Fallback: if signs form a tall column → top-to-bottom,
    else → left-to-right (default Western assumption).
    """
    if not detections:
        return "ltr"

    xs = [d.x for d in detections]
    ys = [d.y for d in detections]

    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    if y_range > x_range * 1.5:
        return "ttb"
    return "ltr"


def _cluster_into_lines(
    detections: list[Detection],
    direction: str,
    line_gap_ratio: float = 0.8,
) -> list[list[Detection]]:
    """
    Group detections into lines (rows for horizontal, columns for vertical).
    line_gap_ratio: gap larger than this × median sign height → new line.
    """
    if not detections:
        return []

    if direction in ("ltr", "rtl"):
        # Group by rows (similar y coordinates)
        sorted_d = sorted(detections, key=lambda d: d.y)
        median_h = np.median([d.h for d in detections])
        threshold = median_h * line_gap_ratio

        lines = [[sorted_d[0]]]
        for d in sorted_d[1:]:
            if abs(d.y - lines[-1][-1].y) < threshold:
                lines[-1].append(d)
            else:
                lines.append([d])

        # Sort each row by x
        for line in lines:
            line.sort(key=lambda d: d.x if direction == "ltr" else -d.x)

    else:  # ttb — group by columns (similar x)
        sorted_d = sorted(detections, key=lambda d: d.x)
        median_w = np.median([d.w for d in detections])
        threshold = median_w * line_gap_ratio

        lines = [[sorted_d[0]]]
        for d in sorted_d[1:]:
            if abs(d.x - lines[-1][-1].x) < threshold:
                lines[-1].append(d)
            else:
                lines.append([d])

        for line in lines:
            line.sort(key=lambda d: d.y)

    return lines


def order_detections(detections: list[dict]) -> tuple[list[Detection], str, list[list]]:
    """
    Main function called by Stage 3 pipeline.

    Args:
        detections : list of dicts from Stage 1+2 (each has code, clf_code, x, y, w, h, bbox)

    Returns:
        ordered    : flat list of Detection objects in reading order
        direction  : "ltr" | "rtl" | "ttb"
        lines      : nested list (lines of signs) for display
    """
    det_objs = [
        Detection(
            code     = d.get("code", ""),
            conf     = d.get("conf", 0.0),
            clf_code = d.get("clf_code", d.get("code", "")),
            clf_conf = d.get("clf_conf", d.get("conf", 0.0)),
            x        = d["x"], y=d["y"],
            w        = d["w"], h=d["h"],
            bbox     = d["bbox"],
        )
        for d in detections
    ]

    direction = _vote_direction(det_objs)
    lines     = _cluster_into_lines(det_objs, direction)
    ordered   = [sign for line in lines for sign in line]

    print(f"[Stage3] Reading direction: {direction.upper()}")
    print(f"[Stage3] Lines detected: {len(lines)}")
    for i, line in enumerate(lines):
        codes = [s.clf_code or s.code for s in line]
        print(f"  Line {i+1}: {' '.join(codes)}")

    return ordered, direction, lines
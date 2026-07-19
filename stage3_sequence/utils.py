"""
utils.py — Stage 3
Shared helpers: config loading, sign metadata, positional utilities.
"""

import os
import json
import yaml
import math
import torch
import numpy as np
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_map(json_path: str) -> dict:
    """Load {idx: GardinerCode} from JSON (produced by Stage 2)."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_code_to_idx(json_path: str) -> dict:
    """Load {GardinerCode: idx} — inverse of class map."""
    idx_to_code = load_class_map(json_path)
    return {v: k for k, v in idx_to_code.items()}


# ── Gardiner sign metadata ────────────────────────────────────────────────────
# Each sign belongs to a category (Gardiner list section A-Z, Aa).
# We'll use this to assign phonetic/semantic type tags.

GARDINER_CATEGORIES = {
    "A": "human_male",
    "B": "human_female",
    "C": "deity",
    "D": "body_part",
    "E": "mammal",
    "F": "mammal_part",
    "G": "bird",
    "H": "bird_part",
    "I": "reptile_amphibian",
    "K": "fish",
    "L": "invertebrate",
    "M": "plant",
    "N": "sky_earth_water",
    "O": "building",
    "P": "ship",
    "Q": "furniture",
    "R": "temple",
    "S": "clothing",
    "T": "warfare",
    "U": "agriculture",
    "V": "rope_basket",
    "W": "vessel",
    "X": "bread_cake",
    "Y": "writing",
    "Z": "stroke_geometric",
    "Aa": "unclassified",
}


def get_sign_category(code: str) -> str:
    """Return the semantic category of a Gardiner code."""
    # Handle two-letter prefixes first (e.g. Aa)
    if code.startswith("Aa"):
        return GARDINER_CATEGORIES.get("Aa", "unknown")
    prefix = ""
    for ch in code:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return GARDINER_CATEGORIES.get(prefix.upper(), "unknown")


def get_category_id(code: str, category_list: list[str]) -> int:
    cat = get_sign_category(code)
    if cat in category_list:
        return category_list.index(cat)
    return len(category_list)  # "unknown" bucket


CATEGORY_LIST = list(GARDINER_CATEGORIES.values())
N_CATEGORIES  = len(CATEGORY_LIST) + 1   # +1 for unknown


# ── Positional encoding (sinusoidal) ─────────────────────────────────────────

def sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Classic sinusoidal positional encoding from 'Attention Is All You Need'.
    Returns tensor of shape (seq_len, d_model).
    """
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) *
        (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe   # (seq_len, d_model)
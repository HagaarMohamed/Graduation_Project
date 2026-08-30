"""
embeddings.py — Stage 3

Maps Gardiner codes to dense vector embeddings combining:
  1. Learned sign embedding (nn.Embedding — trained)
  2. Category embedding (semantic group: bird, body part, ...)
  3. Confidence embedding (how sure Stage 2 was)
  4. Spatial embedding (normalized x, y position in image)
  5. Sinusoidal positional encoding (sequence position)

The trained Stage 3 model was created with max_seq_len=256. Long detection
sequences are therefore truncated consistently in the tokenizer and in the
embedding forward pass before positional encoding is added.
"""

import torch
import torch.nn as nn

from stage3_sequence.utils import (
    N_CATEGORIES,
    CATEGORY_LIST,
    get_category_id,
    sinusoidal_encoding,
)


class GardinerEmbedding(nn.Module):
    """
    Input per sign:
      - sign_idx     : int  (Gardiner class index 0-766)
      - category_idx : int  (semantic category 0-25)
      - conf         : float (Stage 2 classification confidence)
      - norm_x       : float (x-center / image_width)
      - norm_y       : float (y-center / image_height)

    Output: (seq_len, d_model) tensor
    """

    def __init__(
        self,
        vocab_size: int,       # number of Gardiner classes (767)
        d_model: int,          # embedding dimension
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # ── Embedding tables ──────────────────────────────────────────────
        self.sign_embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.category_embed = nn.Embedding(N_CATEGORIES + 1, d_model // 4)

        # Confidence + spatial → continuous features projected to d_model//4
        # Input: [conf, norm_x, norm_y] = 3 floats
        self.spatial_proj = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4),
        )

        # Fuse sign + category + spatial → d_model
        fuse_in = d_model + d_model // 4 + d_model // 4
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Positional encoding (fixed sinusoidal)
        pe = sinusoidal_encoding(max_seq_len, d_model)
        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sign_ids: torch.Tensor,       # (B, L) int
        category_ids: torch.Tensor,   # (B, L) int
        spatials: torch.Tensor,       # (B, L, 3) float [conf, x, y]
    ) -> torch.Tensor:
        """Returns: (B, L, d_model), capped at max_seq_len."""
        # Keep every input tensor aligned. This protects direct model calls
        # even if a caller bypasses GardinerTokenizer.
        max_len = self.pe.size(0)
        if sign_ids.size(1) > max_len:
            sign_ids = sign_ids[:, :max_len]
            category_ids = category_ids[:, :max_len]
            spatials = spatials[:, :max_len, :]

        _, sequence_length = sign_ids.shape

        s_emb = self.sign_embed(sign_ids)
        c_emb = self.category_embed(category_ids)
        sp_emb = self.spatial_proj(spatials)

        fused = self.fuse(
            torch.cat([s_emb, c_emb, sp_emb], dim=-1)
        )

        # `sequence_length` is now guaranteed to be <= positional-encoding size.
        fused = fused + self.pe[:sequence_length].unsqueeze(0)

        return self.dropout(fused)


# ── Tokenizer: converts Detection objects → tensor inputs ────────────────────

class GardinerTokenizer:
    """
    Converts a list of Detection objects into model-ready tensors.

    The first max_seq_len detections are retained after Stage 3 ordering.
    This keeps the human-readable `codes` list aligned with the encoded tensor.
    """

    def __init__(
        self,
        code_to_idx: dict,       # {"G17": 42, ...}
        image_width: float = 1.0,
        image_height: float = 1.0,
        max_seq_len: int = 256,
    ):
        self.code_to_idx = code_to_idx
        self.image_width = image_width
        self.image_height = image_height
        self.max_seq_len = max_seq_len

    def encode(
        self,
        ordered_detections,       # list of Detection objects
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        """
        Returns dict of tensors ready for the Transformer:
          sign_ids     : (1, L) int
          category_ids : (1, L) int
          spatials     : (1, L, 3) float
          codes        : list[str] — for human-readable output
        """
        # The model's positional encoding is trained for 256 positions.
        # Truncate after ordering so the retained symbols preserve reading order.
        ordered_detections = list(ordered_detections)[: self.max_seq_len]

        sign_ids = []
        category_ids = []
        spatials = []
        codes = []

        for detection in ordered_detections:
            code = detection.clf_code or detection.code
            codes.append(code)

            idx = self.code_to_idx.get(code, 0)
            sign_ids.append(idx + 1)    # +1 because 0 is padding

            cat_id = get_category_id(code, CATEGORY_LIST)
            category_ids.append(cat_id)

            norm_x = detection.x / self.image_width
            norm_y = detection.y / self.image_height
            conf = detection.clf_conf if hasattr(detection, "clf_conf") else detection.conf
            spatials.append([conf, norm_x, norm_y])

        sign_t = torch.tensor([sign_ids], dtype=torch.long, device=device)
        cat_t = torch.tensor([category_ids], dtype=torch.long, device=device)
        sp_t = torch.tensor([spatials], dtype=torch.float, device=device)

        return {
            "sign_ids": sign_t,
            "category_ids": cat_t,
            "spatials": sp_t,
            "codes": codes,
        }

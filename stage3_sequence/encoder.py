"""
encoder.py — Stage 3

Transformer Encoder for hieroglyphic sequence understanding.

Architecture:
  GardinerEmbedding
    → TransformerEncoder (N layers of Multi-Head Self-Attention + FFN)
    → Sequence output: (B, L, d_model)

The encoder produces context-aware representations for each sign position.
These representations are passed to Stage 4 for LLM-assisted translation.

We do NOT train a seq2seq model here because we have no parallel
hieroglyph↔English sentence data. Instead the encoder:
  1. Understands sign co-occurrence patterns
  2. Separates determinatives from phonetic signs
  3. Groups signs into likely "word" clusters (via attention weights)
"""

import math
import torch
import torch.nn as nn
from stage3_sequence.embeddings import GardinerEmbedding


class HieroglyphTransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size   : int,
        d_model      : int   = 256,
        nhead        : int   = 8,
        num_layers   : int   = 4,
        dim_feedforward: int = 1024,
        dropout      : float = 0.1,
        max_seq_len  : int   = 256,
    ):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model

        # ── Embedding layer ───────────────────────────────────────────────
        self.embedding = GardinerEmbedding(
            vocab_size  = vocab_size,
            d_model     = d_model,
            max_seq_len = max_seq_len,
            dropout     = dropout,
        )

        # ── Transformer Encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,   # (B, L, d_model) convention
            norm_first      = True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers    = num_layers,
            enable_nested_tensor = False,
        )

        # ── Output heads ──────────────────────────────────────────────────

        # 1. Sign type classifier: phonogram | determinative | ideogram | other
        self.sign_type_head = nn.Linear(d_model, 4)

        # 2. Word boundary predictor: is this sign the START of a new word?
        self.word_boundary_head = nn.Linear(d_model, 2)

        # 3. Pooled sequence representation (for Stage 4 context vector)
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        sign_ids    : torch.Tensor,   # (B, L)
        category_ids: torch.Tensor,   # (B, L)
        spatials    : torch.Tensor,   # (B, L, 3)
        key_padding_mask: torch.Tensor = None,  # (B, L) True = pad
    ) -> dict:
        """
        Returns:
          hidden        : (B, L, d_model)  — per-sign contextual embeddings
          sign_type     : (B, L, 4)        — sign type logits
          word_boundary : (B, L, 2)        — word start logits
          pooled        : (B, d_model)     — sequence-level representation
          attn_weights  : list of (B,H,L,L) — attention per layer (for viz)
        """
        # ── Embed ──────────────────────────────────────────────────────────
        x = self.embedding(sign_ids, category_ids, spatials)  # (B, L, d_model)

        # ── Encode ────────────────────────────────────────────────────────
        hidden = self.encoder(
            src              = x,
            src_key_padding_mask = key_padding_mask,
        )  # (B, L, d_model)

        # ── Output heads ──────────────────────────────────────────────────
        sign_type_logits     = self.sign_type_head(hidden)      # (B, L, 4)
        word_boundary_logits = self.word_boundary_head(hidden)  # (B, L, 2)

        # Mean pool (ignoring padding if mask provided)
        if key_padding_mask is not None:
            mask_f = (~key_padding_mask).float().unsqueeze(-1)  # (B, L, 1)
            pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)   # (B, d_model)

        pooled = self.pool_proj(pooled)

        return {
            "hidden"        : hidden,
            "sign_type"     : sign_type_logits,
            "word_boundary" : word_boundary_logits,
            "pooled"        : pooled,
        }

    @torch.no_grad()
    def extract_attention(
        self,
        sign_ids    : torch.Tensor,
        category_ids: torch.Tensor,
        spatials    : torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Extract attention weight matrices from each encoder layer.
        Returns list of (B, nhead, L, L) tensors — one per layer.
        Useful for visualizing which signs attend to which.
        """
        x = self.embedding(sign_ids, category_ids, spatials)
        attn_list = []

        for layer in self.encoder.layers:
            # Use multi_head_attention_forward directly to capture weights
            attn_out, attn_w = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            attn_list.append(attn_w.detach().cpu())
            # Run full layer forward to update x properly
            x = layer(x)

        return attn_list


# ── Lightweight inference wrapper ─────────────────────────────────────────────

class SequenceEncoderInference:
    """
    Wraps the trained (or fresh) encoder for Stage 3 pipeline inference.
    If no weights exist, runs with random weights (useful for testing pipeline
    end-to-end before training the encoder).
    """

    SIGN_TYPES = ["phonogram", "determinative", "ideogram", "other"]

    def __init__(
        self,
        vocab_size   : int,
        weights_path : str = None,
        d_model      : int = 256,
        nhead        : int = 8,
        num_layers   : int = 4,
        device       : str = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = HieroglyphTransformerEncoder(
            vocab_size = vocab_size,
            d_model    = d_model,
            nhead      = nhead,
            num_layers = num_layers,
        ).to(self.device)

        if weights_path and os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"[Stage3] Encoder weights loaded: {weights_path}")
        else:
            print("[Stage3] No encoder weights found — using untrained encoder.")
            print("         (Stage 3 output will be layout-based only)")

        self.model.eval()

    def encode_sequence(self, tokenized: dict) -> dict:
        """
        Args:
            tokenized: output of GardinerTokenizer.encode()

        Returns enriched dict with:
            sign_types     : list[str]  per-sign predicted type
            word_boundaries: list[bool] True = start of new word
            hidden         : (1, L, d_model) tensor
            pooled         : (1, d_model) tensor
        """
        import torch
        with torch.no_grad():
            out = self.model(
                sign_ids     = tokenized["sign_ids"].to(self.device),
                category_ids = tokenized["category_ids"].to(self.device),
                spatials     = tokenized["spatials"].to(self.device),
            )

        # Sign type prediction
        sign_type_ids = out["sign_type"].argmax(dim=-1)[0].tolist()
        sign_types    = [self.SIGN_TYPES[i] for i in sign_type_ids]

        # Word boundary prediction
        wb_ids  = out["word_boundary"].argmax(dim=-1)[0].tolist()
        word_boundaries = [bool(b) for b in wb_ids]
        # Always mark first sign as word start
        if word_boundaries:
            word_boundaries[0] = True

        return {
            **tokenized,
            "sign_types"     : sign_types,
            "word_boundaries": word_boundaries,
            "hidden"         : out["hidden"],
            "pooled"         : out["pooled"],
        }


import os  # needed by SequenceEncoderInference
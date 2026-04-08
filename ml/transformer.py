"""
ml/transformer.py — DS-CAT: Dual-Stream Cross-Attention Transformer.

Architecture overview (draw this on the whiteboard for the professor):

    ┌─────────────┐     ┌─────────────┐
    │  Temporal    │     │   Surface   │
    │  Sequence    │     │  Snapshot   │
    │ (60×19)      │     │ (400×8)     │
    └──────┬──────┘     └──────┬──────┘
           │                    │
    ┌──────▼──────┐     ┌──────▼──────┐
    │  Linear     │     │  Linear     │
    │  Projection │     │  Projection │
    │  19 → d     │     │  8 → d      │
    └──────┬──────┘     └──────┬──────┘
           │                    │
    ┌──────▼──────┐     ┌──────▼──────┐
    │  + Positional│     │ (no pos     │
    │  Encoding   │     │  encoding)  │
    └──────┬──────┘     └──────┬──────┘
           │                    │
    ┌──────▼──────┐     ┌──────▼──────┐
    │  Transformer│     │  Transformer│
    │  Encoder    │     │  Encoder    │
    │  (L layers) │     │  (L layers) │
    └──────┬──────┘     └──────┬──────┘
           │                    │
           │    ┌───────────┐   │
           └───►│  Cross    │◄──┘
                │  Attention│
                │  (T→S)    │
                └─────┬─────┘
                      │
               ┌──────▼──────┐
               │  Classifier │
               │  Head       │
               └──────┬──────┘
                      │
                   P(fast)

Why this architecture?

  1. Temporal encoder captures HOW the violation evolved — did the TP2
     ratio grow quickly? Is the market trending? Was vol rising?

  2. Surface encoder captures the STRUCTURE of the entire options surface
     at detection time — the model learns which surface shapes predict
     fast reversion (e.g., isolated mispricing vs systematic shift).

  3. Cross-attention is the key innovation: the temporal stream QUERIES
     the surface stream. This lets the model ask "given what just happened
     in the market, which parts of the surface are relevant for predicting
     whether THIS violation will revert?"

  4. The surface has no positional encoding because instruments are
     unordered — unlike time steps, there's no natural sequence. The model
     must learn to attend based on features (moneyness, TTM) not position.

Hyperparameters chosen for Colab T4 (16GB):
  d_model = 64, n_heads = 4, n_layers = 2, dropout = 0.1
  → ~150K parameters (tiny — overfitting is the real risk, not underfitting)
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("ml.transformer")


# ── Positional Encoding (temporal stream only) ────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from Vaswani et al. (2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Only applied to the temporal stream — the surface stream is a SET
    of instruments with no natural ordering.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter — not trained)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ── Cross-Attention Module ────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Temporal queries attend to surface keys/values.

    Q = temporal_out (what the model "asks")
    K = surface_out  (what information is available)
    V = surface_out  (what gets retrieved)

    This is the core of DS-CAT. The intuition:
    "Given the recent market dynamics (temporal), which parts of the
     current options surface (surface) are informative for predicting
     fast reversion?"

    Supports masking: padded surface positions get -inf attention scores,
    so the model never attends to zero-padded instruments.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = self.d_head ** -0.5

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        temporal: torch.Tensor,
        surface: torch.Tensor,
        surface_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            temporal:     (B, T, d_model) — query source
            surface:      (B, S, d_model) — key/value source
            surface_mask: (B, S) — 1 for real instruments, 0 for padding

        Returns:
            (B, T, d_model) — temporal enriched with surface information
        """
        B, T, D = temporal.shape
        S = surface.size(1)

        # Project to Q, K, V
        Q = self.W_q(temporal)   # (B, T, D)
        K = self.W_k(surface)    # (B, S, D)
        V = self.W_v(surface)    # (B, S, D)

        # Reshape for multi-head attention
        # (B, T, D) → (B, n_heads, T, d_head)
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T, S)

        # Mask padded surface positions
        if surface_mask is not None:
            # surface_mask: (B, S) → (B, 1, 1, S) for broadcasting
            mask = surface_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, V)  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        out = self.out_proj(out)

        # Residual + LayerNorm
        return self.norm(temporal + out)


# ── DS-CAT Model ──────────────────────────────────────────────

class DSCAT(nn.Module):
    """
    Dual-Stream Cross-Attention Transformer for TP2 violation classification.

    Input:
        temporal:     (B, seq_len, d_temporal)  — market history
        surface:      (B, n_instr, d_surface)   — option surface snapshot
        surface_mask: (B, n_instr)              — padding mask

    Output:
        logits: (B, 2) — raw scores for [slow/never, fast] classes

    Architecture:
        1. Linear projection of both streams to d_model
        2. Positional encoding on temporal only
        3. Self-attention encoding of both streams (independent)
        4. Cross-attention: temporal queries surface
        5. Pooling: mean of temporal output (over timesteps)
        6. Classification head: MLP → 2 classes
    """

    def __init__(
        self,
        d_temporal: int = 19,
        d_surface: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_cross_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        # ── Input projections ──
        self.temporal_proj = nn.Sequential(
            nn.Linear(d_temporal, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.surface_proj = nn.Sequential(
            nn.Linear(d_surface, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # ── Positional encoding (temporal only) ──
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )

        # ── Self-attention encoders ──
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=n_layers
        )

        surface_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.surface_encoder = nn.TransformerEncoder(
            surface_layer, num_layers=n_layers
        )

        # ── Cross-attention (temporal queries surface) ──
        self.cross_layers = nn.ModuleList([
            CrossAttention(d_model, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # ── Classification head ──
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization — standard for transformers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        temporal: torch.Tensor,
        surface: torch.Tensor,
        surface_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            temporal:     (B, seq_len, d_temporal)
            surface:      (B, n_instr, d_surface)
            surface_mask: (B, n_instr)

        Returns:
            logits: (B, 2)
        """
        # ── 1. Project to d_model ──
        t = self.temporal_proj(temporal)    # (B, T, d_model)
        s = self.surface_proj(surface)      # (B, S, d_model)

        # ── 2. Positional encoding (temporal only) ──
        t = self.pos_encoder(t)

        # ── 3. Surface padding mask for self-attention ──
        # PyTorch TransformerEncoder uses src_key_padding_mask:
        #   True = IGNORE this position (opposite of our mask convention)
        if surface_mask is not None:
            s_padding_mask = (surface_mask == 0)  # (B, S) True=pad
        else:
            s_padding_mask = None

        # ── 4. Self-attention ──
        t = self.temporal_encoder(t)                                # (B, T, d)
        s = self.surface_encoder(s, src_key_padding_mask=s_padding_mask)  # (B, S, d)

        # ── 5. Cross-attention: temporal queries surface ──
        for cross_attn in self.cross_layers:
            t = cross_attn(t, s, surface_mask)   # (B, T, d)

        # ── 6. Pool temporal output ──
        # Mean pooling over all timesteps
        t_pooled = t.mean(dim=1)  # (B, d)

        # ── 7. Classify ──
        logits = self.classifier(t_pooled)  # (B, 2)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model factory ─────────────────────────────────────────────

def build_model(
    d_temporal: int = 19,
    d_surface: int = 8,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    device: str = "auto",
) -> tuple:
    """
    Build DS-CAT model and move to device.

    Returns: (model, device_str)

    For Colab T4:
      d_model=64, n_layers=2, n_heads=4 → ~150K params
      Fits easily in 16GB with batch_size=32
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = DSCAT(
        d_temporal=d_temporal,
        d_surface=d_surface,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    n_params = model.count_parameters()
    log.info(
        f"DS-CAT built: d_model={d_model}, layers={n_layers}, heads={n_heads} | "
        f"{n_params:,} parameters | device={device}"
    )

    return model, device

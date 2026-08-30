from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class EncoderConfig:
    input_dim: int
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    proj_dim: int = 128


class GRUEncoder(nn.Module):
    """Sequence encoder -> single embedding per sequence.

    Uses a BiGRU and returns:
    - token features (optional)
    - pooled sequence embedding
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=int(cfg.input_dim),
            hidden_size=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(cfg.dropout) if int(cfg.num_layers) > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(cfg.dropout)) if float(cfg.dropout) > 0 else nn.Identity()

        # Attention pooling
        self.att = nn.Linear(2 * int(cfg.hidden_dim), 1)

        # Projection head for contrastive learning
        self.proj = nn.Sequential(
            nn.Linear(2 * int(cfg.hidden_dim), 2 * int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(2 * int(cfg.hidden_dim), int(cfg.proj_dim)),
        )

    def forward(
        self, x_padded: torch.Tensor, lengths: torch.Tensor, *, return_tokens: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x_padded: [B, T, F]
            lengths: [B]
        Returns:
            z: [B, proj_dim] (L2-normalized)
            h_tokens: [B, T, 2*hidden_dim] if return_tokens else None
        """
        packed = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        h, _ = pad_packed_sequence(out_packed, batch_first=True)  # [B, T, 2H]
        h = self.dropout(h)

        # Build mask for pooling
        B, T, D = h.shape
        mask = torch.arange(T, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]

        att = self.att(h).squeeze(-1)  # [B, T]
        att = att.masked_fill(~mask, torch.finfo(att.dtype).min)
        w = torch.softmax(att, dim=1).unsqueeze(-1)  # [B, T, 1]
        pooled = (h * w).sum(dim=1)  # [B, 2H]

        z = self.proj(pooled)  # [B, proj_dim]
        z = F.normalize(z, p=2, dim=1)
        return z, (h if return_tokens else None)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE (SimCLR-style) on a batch of positive pairs.

    z1,z2 are [B, D] and assumed L2-normalized.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must match shape, got {tuple(z1.shape)} vs {tuple(z2.shape)}")
    B, D = z1.shape
    if B < 2:
        # not meaningful, but keep differentiable
        return (z1 * 0.0).sum()

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = (z @ z.T) / float(temperature)  # [2B, 2B]

    # mask self-similarity
    eye = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, torch.finfo(sim.dtype).min)

    # positives: i <-> i+B
    pos_idx = torch.arange(2 * B, device=z.device)
    pos = (pos_idx + B) % (2 * B)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -log_prob[torch.arange(2 * B, device=z.device), pos].mean()
    return loss


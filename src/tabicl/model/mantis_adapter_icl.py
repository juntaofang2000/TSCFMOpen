from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor

from tabicl.model.learning import ICLearning
from tabicl.model.mantis_dev.architecture.architecture import Mantis8M


class TokenMLPAdapter(nn.Module):
    """Token-wise adapter mapping Mantis embedding dim -> ICL dim.

    Expects input shape (B, T, D_mantis) and returns (B, T, D_icl).
    """

    def __init__(
        self,
        mantis_dim: int,
        icl_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        mantis_dim = int(mantis_dim)
        icl_dim = int(icl_dim)
        hidden_dim = int(hidden_dim) if hidden_dim is not None else icl_dim

        self.mantis_dim = mantis_dim
        self.icl_dim = icl_dim

        self.net = nn.Sequential(
            nn.LayerNorm(mantis_dim) if use_layernorm else nn.Identity(),
            nn.Linear(mantis_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden_dim, icl_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Adapter expects (B, T, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.mantis_dim:
            raise ValueError(
                f"Adapter last-dim mismatch: expected {self.mantis_dim}, got {x.shape[-1]}"
            )
        return self.net(x)


class MantisAdapterICL(nn.Module):
    """Mantis (frozen) -> Adapter (trainable) -> ICL predictor (frozen).

    Notes:
    - Mantis forward is done under torch.no_grad() by default (since frozen).
    - ICL predictor params are frozen, but forward is NOT wrapped in no_grad so
      gradients can flow into the adapter.
    """

    def __init__(
        self,
        mantis_model: Mantis8M,
        icl_predictor: ICLearning,
        adapter: nn.Module,
        *,
        mantis_seq_len: int = 512,
        mantis_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.mantis_model = mantis_model
        self.icl_predictor = icl_predictor
        self.adapter = adapter

        self.mantis_seq_len = int(mantis_seq_len)
        self.mantis_batch_size = int(mantis_batch_size)

    def freeze_mantis_and_icl(self) -> None:
        for p in self.mantis_model.parameters():
            p.requires_grad_(False)
        for p in self.icl_predictor.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen modules in eval mode (dropout off, etc.)
        self.mantis_model.eval()
        self.icl_predictor.eval()
        return self

    def _pad_or_truncate_seq(self, x: Tensor) -> Tensor:
        """Ensure last dimension equals mantis_seq_len."""
        target = self.mantis_seq_len
        if x.shape[-1] == target:
            return x
        if x.shape[-1] > target:
            return x[..., :target]
        pad = x.new_zeros((*x.shape[:-1], target - x.shape[-1]))
        return torch.cat([x, pad], dim=-1)

    def _encode_with_mantis(self, X: Tensor) -> Tensor:
        """Encode X (B,T,H) -> representations (B,T,D_mantis) using frozen Mantis."""
        if X.ndim != 3:
            raise ValueError(f"Expected X of shape (B,T,H), got {tuple(X.shape)}")

        B, T, H = X.shape
        X = self._pad_or_truncate_seq(X)
        H2 = X.shape[-1]

        x_flat = X.reshape(B * T, 1, H2)
        device = next(self.mantis_model.parameters()).device
        x_flat = x_flat.to(device)

        reps = []
        bs = max(1, self.mantis_batch_size)
        with torch.no_grad():
            for i in range(0, x_flat.shape[0], bs):
                reps.append(self.mantis_model(x_flat[i : i + bs]))
        reps = torch.cat(reps, dim=0)
        return reps.reshape(B, T, -1)

    def forward(self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None) -> Tensor:
        mantis_repr = self._encode_with_mantis(X).to(X.device)
        adapted = self.adapter(mantis_repr)
        return self.icl_predictor(adapted, y_train=y_train)

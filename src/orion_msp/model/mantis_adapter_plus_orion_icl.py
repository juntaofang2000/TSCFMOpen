from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor


class _MantisAdapterPlusOrionICL(nn.Module):
    """Mantis encoder -> Adapter -> OrionMSP ICL predictor.

    This module is meant for *end-to-end training* in the ORION-MSP training loop.

    Expected training-loop contract (matches `orion_msp.train.run.Trainer`):
    - forward(X, y_train, d) returns logits of shape (B, T_test, C)

    Notes
    -----
    - `X` is interpreted as a 1D signal per row, i.e. (B,T,H) is reshaped to
      (B*T, 1, L) with L == mantis_seq_len (pad/truncate should be handled by caller).
    - We chunk the Mantis forward along the flattened batch dimension to
      control memory usage.
    """

    def __init__(
        self,
        *,
        mantis_model: nn.Module,
        adapter: nn.Module,
        icl_predictor: nn.Module,
        mantis_seq_len: int = 512,
        mantis_batch_size: int = 256,
    ) -> None:
        super().__init__()
        self.mantis_model = mantis_model
        self.adapter = adapter
        self.icl_predictor = icl_predictor
        self.mantis_seq_len = int(mantis_seq_len)
        self.mantis_batch_size = int(mantis_batch_size)

        self.max_classes = int(getattr(self.icl_predictor, "max_classes", 0) or 0)

    def _pad_or_truncate(self, X: Tensor) -> Tensor:
        target = self.mantis_seq_len
        if X.shape[-1] == target:
            return X
        if X.shape[-1] > target:
            return X[..., :target]
        pad = X.new_zeros((*X.shape[:-1], target - X.shape[-1]))
        return torch.cat([X, pad], dim=-1)

    def _encode_rows(self, X: Tensor) -> Tensor:
        if X.ndim != 3:
            raise ValueError(f"Expected X to be (B,T,H), got {tuple(X.shape)}")

        X = self._pad_or_truncate(X)
        B, T, L = X.shape

        device = next(self.mantis_model.parameters()).device
        x_flat = X.reshape(B * T, 1, L).to(device)

        outs: list[Tensor] = []
        bs = max(1, int(self.mantis_batch_size))
        for i in range(0, x_flat.shape[0], bs):
            outs.append(self.mantis_model(x_flat[i : i + bs]))
        reps = torch.cat(outs, dim=0)  # (B*T, D)

        return reps.reshape(B, T, -1).to(X.device)

    def forward(self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None) -> Tensor:
        # `d` is ignored: feature-count metadata is not used in this model.
        reps = self._encode_rows(X)
        reps = self.adapter(reps)
        # ICLearning (or compatible) returns (B, T_test, C)
        return self.icl_predictor(reps, y_train=y_train, return_logits=True)

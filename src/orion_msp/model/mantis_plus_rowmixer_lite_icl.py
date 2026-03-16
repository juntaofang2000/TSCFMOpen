from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class _MantisPlusRowMixerLiteICL(nn.Module):
    """Mantis encoder -> RowMixerLiteICL.

    Expected training-loop contract:
    - forward(X, y_train, d) returns logits of shape (B, T_test, C)

    Supported inputs
    ----------------
    - Univariate rows: X is (B, T, L)
    - Multichannel rows: X is (B, T, C, L)

    For multichannel inputs, each channel is encoded independently by the same
    Mantis encoder, then channel embeddings are concatenated along the feature
    dimension before being fed into RowMixerLiteICL.
    """

    def __init__(
        self,
        *,
        mantis_model: nn.Module,
        rowmixer_icl: nn.Module,
        mantis_seq_len: int = 512,
        mantis_batch_size: int = 256,
    ) -> None:
        super().__init__()
        self.mantis_model = mantis_model
        self.rowmixer_icl = rowmixer_icl
        self.mantis_seq_len = int(mantis_seq_len)
        self.mantis_batch_size = int(mantis_batch_size)

        self.max_classes = int(getattr(self.rowmixer_icl, "max_classes", 0) or 0)

    def _pad_or_truncate_last_dim(self, X: Tensor) -> Tensor:
        target = self.mantis_seq_len
        if X.shape[-1] == target:
            return X
        if X.shape[-1] > target:
            return X[..., :target]
        pad = X.new_zeros((*X.shape[:-1], target - X.shape[-1]))
        return torch.cat([X, pad], dim=-1)

    def _encode_with_mantis(self, x_flat: Tensor) -> Tensor:
        # Some wrapped/compiled encoders may expose no parameters on replicas.
        # Fall back to buffers, then input device, to avoid StopIteration on multi-GPU.
        try:
            device = next(self.mantis_model.parameters()).device
        except StopIteration:
            try:
                device = next(self.mantis_model.buffers()).device
            except StopIteration:
                device = x_flat.device
        x_flat = x_flat.to(device)

        outs: list[Tensor] = []
        batch_size = max(1, int(self.mantis_batch_size))
        for i in range(0, x_flat.shape[0], batch_size):
            outs.append(self.mantis_model(x_flat[i : i + batch_size]))
        return torch.cat(outs, dim=0)

    def _encode_rows(self, X: Tensor) -> Tensor:
        if X.ndim == 3:
            X = self._pad_or_truncate_last_dim(X)
            B, T, L = X.shape
            x_flat = X.reshape(B * T, 1, L)
            reps = self._encode_with_mantis(x_flat)
            return reps.reshape(B, T, -1).to(X.device)

        if X.ndim == 4:
            X = self._pad_or_truncate_last_dim(X)
            B, T, C, L = X.shape
            x_flat = X.reshape(B * T * C, 1, L)
            reps = self._encode_with_mantis(x_flat)
            reps = reps.reshape(B, T, C, -1)
            reps = reps.reshape(B, T, C * reps.shape[-1])
            return reps.to(X.device)

        raise ValueError(f"Expected X to be (B,T,L) or (B,T,C,L), got {tuple(X.shape)}")

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config=None,
        **_unused,
    ) -> Tensor:
        # `d` is ignored here; RowMixerLiteICL receives dense concatenated Mantis embeddings.
        reps = self._encode_rows(X)
        return self.rowmixer_icl(
            reps,
            y_train=y_train,
            d=None,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            mgr_config=mgr_config,
        )
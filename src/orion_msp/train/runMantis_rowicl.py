# train/tunMantis_rowicl.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from orion_msp.model.mantis_rowicl import MantisRowICLModel
from orion_msp.train.run import Trainer as _BaseTrainer
from orion_msp.train.train_config import build_parser


DEFAULT_CMAX = 8
DEFAULT_NOISE_STD = 0.01


def make_multichannel(
    Zbase: Tensor,
    Cmax: int,
    noise_std: float,
    *,
    ensure_exact_mean: bool = True,
) -> tuple[Tensor, Tensor]:
    """Create a synthetic multichannel embedding set from a base embedding.

    Parameters
    ----------
    Zbase:
        (B, T, 512) base embeddings.
    Cmax:
        Max number of channels to generate.
    noise_std:
        Stddev of Gaussian noise added per channel.
    ensure_exact_mean:
        If True, enforce zero-mean noise across valid channels per (B,T,*) slice.

    Returns
    -------
    Z:
        (B, T, Cmax, 512)
    channel_mask:
        (B, T, Cmax) bool, True for valid channels.

    Construction
    ------------
    For each sample b, sample ci ~ Uniform{1..Cmax}. For valid channels:
        Z[b,:,c,:] = Zbase[b,:,:] + eps[b,:,c,:]
    Padding channels are exactly 0.

    If ensure_exact_mean=True, eps is centered across the channel dimension over
    the valid channels only, so that mean_c eps == 0.
    """

    if Zbase.ndim != 3 or int(Zbase.shape[-1]) != 512:
        raise ValueError(f"Zbase must be (B,T,512), got {tuple(Zbase.shape)}")

    B, T, D = Zbase.shape
    Cmax_i = int(Cmax)
    if Cmax_i <= 0:
        raise ValueError("Cmax must be positive")

    device = Zbase.device
    dtype = Zbase.dtype

    # Sample valid channel counts per sample.
    ci = torch.randint(1, Cmax_i + 1, (B,), device=device, dtype=torch.long)

    # valid_mask_bc: (B,Cmax)
    ch = torch.arange(Cmax_i, device=device, dtype=torch.long)[None, :]
    valid_mask_bc = ch < ci[:, None]

    # Expand to time: (B,T,Cmax)
    channel_mask = valid_mask_bc[:, None, :].expand(B, T, Cmax_i).to(torch.bool)

    # Noise for all channels, then mask invalid.
    if float(noise_std) == 0.0:
        eps = torch.zeros((B, T, Cmax_i, D), device=device, dtype=dtype)
    else:
        eps = torch.randn((B, T, Cmax_i, D), device=device, dtype=dtype) * float(noise_std)

    valid_mask_btcd = channel_mask.unsqueeze(-1)  # (B,T,Cmax,1)
    eps = eps * valid_mask_btcd.to(dtype)

    if ensure_exact_mean:
        # Center eps across valid channels only:
        # mean = sum(eps) / ci
        denom = ci.to(dtype).view(B, 1, 1, 1).clamp(min=1)
        mean = eps.sum(dim=2, keepdim=True) / denom
        eps = (eps - mean) * valid_mask_btcd.to(dtype)

    Z = (Zbase.unsqueeze(2) + eps) * valid_mask_btcd.to(dtype)
    return Z, channel_mask


class Trainer(_BaseTrainer):
    """Trainer that swaps OrionMSP with MantisRowICLModel, keeping the same prior."""

    def __init__(self, config):
        self._did_shape_assert = False
        super().__init__(config)

    def build_model(self):
        # Build MantisRowICLModel. Default mode is identity512.
        self.model_config = {
            "max_classes": int(self.config.max_classes),
            "mode": "identity512",
        }

        model = MantisRowICLModel(max_classes=int(self.config.max_classes), mode="identity512").to(self.config.device)

        if getattr(self.config, "model_compile", False):
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled.")

        if self.ddp:
            self.model = DDP(
                model,
                device_ids=[self.ddp_local_rank],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

        if self.master_process:
            num_params = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {num_params:,}")

    def align_micro_batch(self, micro_X: Tensor, micro_y: Tensor, micro_d: Tensor, seq_len: int):
        # Keep Orion's sequence alignment behavior.
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]
        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]

        # For MantisRowICL training we do NOT crop by micro_d.max().
        # Instead, force last dim to exactly 512.
        target = 512
        Fdim = int(micro_X.shape[-1])
        if Fdim > target:
            micro_X = micro_X[..., :target]
        elif Fdim < target:
            pad = micro_X.new_zeros(*micro_X.shape[:-1], target - Fdim)
            micro_X = torch.cat([micro_X, pad], dim=-1)

        return micro_X, micro_y

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test = micro_y[:, train_size:]

        # early exit if nothing to predict
        if y_test.numel() == 0:
            return {"ce": 0.0, "accuracy": 0.0}

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        # Construct Z and channel_mask from base (B,T,512)
        Zbase = micro_X
        Cmax = int(getattr(self.config, "mantis_rowicl_cmax", DEFAULT_CMAX))
        noise_std = float(getattr(self.config, "mantis_rowicl_noise_std", DEFAULT_NOISE_STD))
        Z, channel_mask = make_multichannel(Zbase, Cmax, noise_std, ensure_exact_mean=True)

        # Minimal one-time assertions (acts like a unit test on the first micro-batch).
        if not self._did_shape_assert:
            B, T, D = Zbase.shape
            assert Z.shape == (B, T, Cmax, 512), (Z.shape, (B, T, Cmax, 512))
            assert channel_mask.shape == (B, T, Cmax), (channel_mask.shape, (B, T, Cmax))
            assert int(Z.shape[-1]) == 512
            self._did_shape_assert = True

        with self.amp_ctx:
            logits = self.model(
                Z,
                channel_mask,
                y_train,
                train_size=train_size,
                channel_shuffle=True,
                return_logits=True,
            )  # (B, Ttest, C)

            B, Ttest, C = logits.shape
            pred = logits.reshape(-1, C)
            true = y_test.reshape(-1).long()

            # drop any labels outside [0, C-1] (corrupt/padded labels)
            valid = (true >= 0) & (true < C)
            if not torch.all(valid):
                true = true[valid]
                pred = pred[valid]
            if true.numel() == 0:
                return {"ce": 0.0, "accuracy": 0.0}

            loss = F.cross_entropy(pred, true)

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {
                "ce": scaled_loss.item(),
                "accuracy": (pred.argmax(dim=1) == true).float().mean().item() / num_micro_batches,
            }
        return micro_results


if __name__ == "__main__":
    parser = build_parser()
    cfg = parser.parse_args()
    trainer = Trainer(cfg)
    trainer.train()

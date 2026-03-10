# rowmixer_lite_icl.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 复用你已有的 ICLearning（就是 orion/tabicl 的 icl 模块）
from .learning import ICLearning


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class RowMixerLite(nn.Module):
    """
    Row-wise mixer: token = feature patch (contiguous chunks along feature dim).

    Input:
        X: (B, T, H) float, padded
        d: (B,) long/int, real feature count per dataset (<=H), or None

    Output:
        R: (B, T, num_cls * d_model)
    """

    def __init__(
        self,
        *,
        d_model: int = 64,
        patch_size: int = 8,
        num_blocks: int = 2,
        nhead: int = 4,
        num_cls: int = 2,
        num_global: int = 1,
        ff_factor: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        shuffle_p: float = 0.25,   # 训练时对 patch token 做随机打乱，弱化“维度顺序依赖”
    ) -> None:
        super().__init__()
        assert patch_size >= 1
        assert num_cls >= 1

        self.d_model = int(d_model)
        self.patch_size = int(patch_size)
        self.num_cls = int(num_cls)
        self.num_global = int(num_global)
        self.shuffle_p = float(shuffle_p)

        act = activation.lower()
        if act == "gelu":
            act_fn = nn.GELU
        elif act == "relu":
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # patch -> token
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(self.patch_size),
            nn.Linear(self.patch_size, self.d_model),
            act_fn(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.token_ln = nn.LayerNorm(self.d_model)

        # special tokens
        self.cls_tokens = nn.Parameter(torch.zeros(self.num_cls, self.d_model))
        self.global_tokens = nn.Parameter(torch.zeros(self.num_global, self.d_model)) if self.num_global > 0 else None

        # transformer over (CLS/GLOBAL + patch_tokens)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=self.d_model * int(ff_factor),
            dropout=float(dropout),
            activation=act,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_blocks))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_tokens, std=0.02)
        if self.global_tokens is not None:
            nn.init.normal_(self.global_tokens, std=0.02)

    def _make_patch_view(self, X: Tensor) -> Tensor:
        # X: (B,T,H) -> (B,T,P,patch_size)
        B, T, H = X.shape
        p = self.patch_size
        pad = (p - (H % p)) % p
        if pad:
            X = F.pad(X, (0, pad), value=0.0)
        P = X.shape[-1] // p
        return X.view(B, T, P, p)

    def _patch_mask(self, d: Optional[Tensor], H: int, P: int, device) -> Optional[Tensor]:
        """
        Return patch key_padding_mask of shape (B, P), where True means "masked out".
        """
        if d is None:
            return None
        # valid patches per dataset = ceil(d / patch_size)
        p = self.patch_size
        dv = d.to(device=device).long().clamp(min=0, max=H)
        valid_p = torch.div(dv + p - 1, p, rounding_mode="floor")  # (B,)
        idx = torch.arange(P, device=device).view(1, P)  # (1,P)
        # mask True for invalid positions
        mask = idx >= valid_p.view(-1, 1)
        return mask  # (B,P)

    @torch.no_grad()
    def _maybe_shuffle_patches_(self, patches: Tensor, patch_mask: Optional[Tensor]) -> None:
        """
        In-place shuffle along patch dimension, per dataset (same perm for all rows T).
        patches: (B,T,P,D)
        patch_mask: (B,P) True=invalid
        """
        if (not self.training) or (self.shuffle_p <= 0):
            return
        if torch.rand(()) > self.shuffle_p:
            return

        B, T, P, D = patches.shape
        device = patches.device

        if patch_mask is None:
            # shuffle all patches
            for b in range(B):
                perm = torch.randperm(P, device=device)
                patches[b] = patches[b, :, perm, :]
            return

        for b in range(B):
            valid = (~patch_mask[b]).nonzero(as_tuple=False).flatten()
            if valid.numel() <= 1:
                continue
            perm = valid[torch.randperm(valid.numel(), device=device)]
            # 只打乱 valid 区间，invalid 继续保持在尾部
            new_order = torch.cat([perm, (patch_mask[b]).nonzero(as_tuple=False).flatten()], dim=0)
            patches[b] = patches[b, :, new_order, :]

    def forward(self, X: Tensor, d: Optional[Tensor] = None) -> Tensor:
        if X.ndim != 3:
            raise ValueError(f"Expected X (B,T,H), got {tuple(X.shape)}")
        B, T, H = X.shape

        # patchify
        Xp = self._make_patch_view(X)  # (B,T,P,p)
        B, T, P, p = Xp.shape

        # patch mask from d
        pmask = self._patch_mask(d, H=H, P=P, device=Xp.device)  # (B,P) or None

        # patch -> token
        tok = self.patch_proj(Xp)      # (B,T,P,D)
        tok = self.token_ln(tok)

        # shuffle augmentation to kill feature-order reliance
        self._maybe_shuffle_patches_(tok, pmask)

        # flatten rows as batch
        tok = tok.view(B * T, P, self.d_model)  # (BT,P,D)

        # build special tokens
        cls = self.cls_tokens.unsqueeze(0).expand(B * T, -1, -1)  # (BT,num_cls,D)
        if self.num_global > 0:
            g = self.global_tokens.unsqueeze(0).expand(B * T, -1, -1)  # (BT,num_global,D)
            src = torch.cat([cls, g, tok], dim=1)
            prefix = self.num_cls + self.num_global
        else:
            src = torch.cat([cls, tok], dim=1)
            prefix = self.num_cls

        # key padding mask
        if pmask is None:
            kpm = None
        else:
            # expand (B,P) -> (BT,P)
            pm_bt = pmask.unsqueeze(1).expand(B, T, P).reshape(B * T, P)
            # add prefix tokens (never masked)
            kpm = torch.cat(
                [torch.zeros((B * T, prefix), device=src.device, dtype=torch.bool), pm_bt],
                dim=1,
            )  # (BT, prefix+P)

        out = self.encoder(src, src_key_padding_mask=kpm)  # (BT, S, D)

        # take CLS tokens -> row embedding
        cls_out = out[:, : self.num_cls, :]  # (BT,num_cls,D)
        cls_out = cls_out.view(B, T, self.num_cls * self.d_model)  # (B,T,num_cls*D)
        return cls_out


class RowMixerLiteICL(nn.Module):
    """
    Drop-in replacement for OrionMSP forward(X, y_train, d, ...) used by run.py.

    Training mode returns: logits (B, Ttest, max_classes)
    Inference mode returns: logits/probs (B, Ttest, num_classes) via ICLearning
    """

    def __init__(
        self,
        *,
        max_classes: int = 10,
        embed_dim: int = 64,          # 这里 embed_dim 就是 RowMixer 的 d_model
        patch_size: int = 8,
        row_num_blocks: int = 2,
        row_nhead: int = 4,
        row_num_cls: int = 2,
        row_num_global: int = 1,
        icl_num_blocks: int = 6,
        icl_nhead: int = 4,
        ff_factor: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        shuffle_p: float = 0.25,
        # 兼容你原来 config 里可能存在但这里用不到的字段
        **_unused,
    ) -> None:
        super().__init__()
        self.max_classes = int(max_classes)
        self.embed_dim = int(embed_dim)

        self.row_mixer = RowMixerLite(
            d_model=self.embed_dim,
            patch_size=int(patch_size),
            num_blocks=int(row_num_blocks),
            nhead=int(row_nhead),
            num_cls=int(row_num_cls),
            num_global=int(row_num_global),
            ff_factor=int(ff_factor),
            dropout=float(dropout),
            activation=str(activation),
            norm_first=bool(norm_first),
            shuffle_p=float(shuffle_p),
        )

        icl_dim = self.embed_dim * int(row_num_cls)
        self.icl_predictor = ICLearning(
            max_classes=int(max_classes),
            d_model=int(icl_dim),
            num_blocks=int(icl_num_blocks),
            nhead=int(icl_nhead),
            dim_feedforward=int(icl_dim) * int(ff_factor),
            dropout=float(dropout),
            activation=str(activation),
            norm_first=bool(norm_first),
            # 保持接口兼容（即使你不使用 perceiver）
            perc_num_latents=int(_unused.get("perc_num_latents", 16)),
            perc_layers=int(_unused.get("perc_layers", 2)),
        )

    def _train_forward(self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None) -> Tensor:
        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T

        # 参照 OrionMSP：如果 d 退化成全等于 H，就当没传 d :contentReference[oaicite:4]{index=4}
        if d is not None and (d.numel() == 1 or (d == H).all()):
            d = None

        R = self.row_mixer(X, d=d)            # (B,T,icl_dim)
        out = self.icl_predictor(R.clone(), y_train=y_train)  # 训练模式会 out[:,train_size:] :contentReference[oaicite:5]{index=5}
        return out

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config=None,
    ) -> Tensor:
        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T

        # 推理阶段一般 d 不需要；但保留兼容
        if d is not None and (d.numel() == 1 or (d == H).all()):
            d = None

        R = self.row_mixer(X, d=d)
        out = self.icl_predictor(
            R.clone(),
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=float(softmax_temperature),
            mgr_config=mgr_config,
        )
        return out

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
        if self.training:
            return self._train_forward(X, y_train, d=d)
        else:
            return self._inference_forward(
                X,
                y_train,
                d=d,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                mgr_config=mgr_config,
            )

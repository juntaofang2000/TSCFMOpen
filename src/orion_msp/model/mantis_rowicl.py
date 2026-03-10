from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor, nn

from .interaction import RowInteraction
from .learning import ICLearning
from .inference_config import MgrConfig


EmbeddingMode = Literal["identity512", "chunk4x128"]


def _assert_padding_suffix(channel_mask: Tensor, d: Tensor) -> None:
    """Assert that padding channels form a strict suffix for each task.

    RowInteraction's internal padding logic (via `d`) assumes that valid feature/channel
    tokens come first and padding tokens are a *suffix*. If padding is interleaved into
    the middle (e.g., by a naive channel permutation over Cmax), the d-based suffix mask
    will ignore the wrong tokens, leading to severe degradation.

    Parameters
    ----------
    channel_mask:
        Bool tensor of shape (B, T, C). True=valid, False=padding.
    d:
        Long tensor of shape (B,) giving number of valid channels (prefix length).

    Checks (per batch item b)
    ------------------------
    a) Channels [0 : d[b]) must be True for *all* time steps.
    b) Channels [d[b] : C) must be False for *all* time steps.
    """

    if channel_mask.ndim != 3:
        raise AssertionError(f"channel_mask must be (B,T,C), got {tuple(channel_mask.shape)}")
    if d.ndim != 1:
        raise AssertionError(f"d must be (B,), got {tuple(d.shape)}")

    B, T, C = channel_mask.shape
    if d.shape[0] != B:
        raise AssertionError(f"d batch mismatch: B={B} vs d.shape[0]={int(d.shape[0])}")

    m = channel_mask.bool()
    d_long = d.to(device=m.device, dtype=torch.long)

    # per_channel_all: (B,C) whether each channel is consistently valid over time
    per_channel_all_true = m.all(dim=1)
    per_channel_all_false = (~m).all(dim=1)

    for b in range(B):
        db = int(d_long[b].item())
        if not (0 <= db <= C):
            raise AssertionError(f"Invalid d[{b}]={db} for C={C}")

        # Prefix must be all-True across time.
        if db > 0:
            prefix_ok = per_channel_all_true[b, :db]
            if not bool(prefix_ok.all().item()):
                # Find first violating channel/time.
                bad_c = int(torch.nonzero(~prefix_ok, as_tuple=False)[0].item())
                c_idx = bad_c
                bad_t = int(torch.nonzero(~m[b, :, c_idx], as_tuple=False)[0].item())
                raise AssertionError(
                    f"Padding suffix violated (prefix not all-True): b={b}, d={db}, "
                    f"first_bad=(t={bad_t}, c={c_idx}), value={bool(m[b, bad_t, c_idx].item())}"
                )

        # Suffix must be all-False across time.
        if db < C:
            suffix_ok = per_channel_all_false[b, db:]
            if not bool(suffix_ok.all().item()):
                bad_rel = int(torch.nonzero(~suffix_ok, as_tuple=False)[0].item())
                c_idx = db + bad_rel
                bad_t = int(torch.nonzero(m[b, :, c_idx], as_tuple=False)[0].item())
                raise AssertionError(
                    f"Padding suffix violated (suffix not all-False): b={b}, d={db}, "
                    f"first_bad=(t={bad_t}, c={c_idx}), value={bool(m[b, bad_t, c_idx].item())}"
                )


class ChannelSetEmbeddingNoLoss(nn.Module):
    """Convert Mantis channel embeddings into RowInteraction-compatible token sequences.

    Inputs
    ------
    Z: (B, T, C, 512)
        Channel embeddings (e.g., frozen Mantis output). C can be padded.
    channel_mask: (B, T, C)
        Boolean mask. True = valid channel, False = padding.

    Two modes
    ---------
    - identity512:
        embed_dim=512, row_num_cls=1.
        Each channel contributes one 512-d token (optional LayerNorm).

    - chunk4x128:
        embed_dim=128, row_num_cls=4.
        Each 512-d channel vector is reshaped into 4 tokens of dim 128.
        No parameters, no information loss; feature count becomes 4*C.

    Returns
    -------
    x_tokens: (B, T, num_special + F, embed_dim)
        num_special = num_cls + num_global (passed in).
        The first num_special tokens are zeros (placeholders).

    token_mask: (B, T, num_special + F)
        True=keep token, False=padding token.
        All special tokens are True.

    Notes
    -----
    We do not introduce labels here; label injection remains inside `ICLearning`.
    """

    def __init__(
        self,
        *,
        mode: EmbeddingMode = "identity512",
        num_special: int,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        if int(num_special) < 0:
            raise ValueError("num_special must be >= 0")

        self.mode: EmbeddingMode = mode
        self.num_special = int(num_special)

        if self.mode == "identity512":
            self.embed_dim = 512
            self.row_num_cls = 1
        elif self.mode == "chunk4x128":
            self.embed_dim = 128
            self.row_num_cls = 4
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.use_layernorm = bool(use_layernorm)
        self.ln = nn.LayerNorm(self.embed_dim) if self.use_layernorm else nn.Identity()

    def forward(self, Z: Tensor, channel_mask: Tensor) -> tuple[Tensor, Tensor]:
        if Z.ndim != 4:
            raise ValueError(f"Expected Z shape (B,T,C,512), got {tuple(Z.shape)}")
        if channel_mask.ndim != 3:
            raise ValueError(f"Expected channel_mask shape (B,T,C), got {tuple(channel_mask.shape)}")

        B, T, C, D = Z.shape
        if int(D) != 512:
            raise ValueError(f"Expected last dim=512, got {int(D)}")
        if channel_mask.shape[0] != B or channel_mask.shape[1] != T or channel_mask.shape[2] != C:
            raise ValueError("channel_mask must match Z in (B,T,C)")

        channel_mask_bool = channel_mask.bool()

        if self.mode == "identity512":
            feat_tokens = self.ln(Z)  # (B,T,C,512)
            feat_mask = channel_mask_bool  # (B,T,C)
        else:
            # (B,T,C,512) -> (B,T,C,4,128) -> (B,T,4*C,128)
            z = Z.reshape(B, T, C, 4, 128)
            z = z.reshape(B, T, C * 4, 128)
            feat_tokens = self.ln(z)

            # Repeat channel mask for each of the 4 chunks.
            m = channel_mask_bool.unsqueeze(-1).expand(B, T, C, 4)
            feat_mask = m.reshape(B, T, C * 4)

        F = int(feat_tokens.shape[2])

        if self.num_special > 0:
            special = feat_tokens.new_zeros(B, T, self.num_special, self.embed_dim)
            x_tokens = torch.cat([special, feat_tokens], dim=2)
            special_mask = torch.ones(B, T, self.num_special, device=feat_mask.device, dtype=torch.bool)
            token_mask = torch.cat([special_mask, feat_mask], dim=2)
        else:
            x_tokens = feat_tokens
            token_mask = feat_mask

        # x_tokens: (B,T,num_special+F,E)
        # token_mask: (B,T,num_special+F)
        return x_tokens, token_mask


@dataclass
class EnsembleConfig:
    n_perm: int = 4


class MantisRowICLModel(nn.Module):
    """ICL classifier over time-series channel embedding sets.

    Core idea
    ---------
    - Use RowInteraction to compute a learned *delta* over a lossless tokenization of
      the channel embeddings.
    - Add it to a robust base representation (masked mean over channels).

    IMPORTANT
    ---------
    `y_train` must be pre-remapped per dataset/task to contiguous IDs 0..K-1.
    (This is the same assumption used throughout OrionMSP/TabICL ICL pipelines.)

    Padding
    -------
    We accept an explicit `channel_mask` and derive a per-task feature count `d`
    (excluding specials) to let RowInteraction ignore padded tokens.
    """

    def __init__(
        self,
        *,
        max_classes: int = 10,
        num_global: int = 2,
        mode: EmbeddingMode = "identity512",
        dropout: float = 0.0,
        # ICLearning knobs (kept aligned with Orion defaults)
        icl_num_blocks: int = 12,
        icl_nhead: int = 4,
        ff_factor: int = 2,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        perc_num_latents: int = 16,
        perc_layers: int = 2,
        # RowInteraction attention window (must be >= max_F to avoid sparsity loss)
        row_window: int = 1024,
        # inference-time ensemble
        ensemble: EnsembleConfig | None = None,
    ) -> None:
        super().__init__()

        self.max_classes = int(max_classes)
        self.num_global = int(num_global)
        self.mode: EmbeddingMode = mode

        if self.mode == "identity512":
            self.embed_dim = 512
            self.row_num_cls = 1
        elif self.mode == "chunk4x128":
            self.embed_dim = 128
            self.row_num_cls = 4
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.num_special = int(self.row_num_cls + self.num_global)

        self.embedder = ChannelSetEmbeddingNoLoss(mode=self.mode, num_special=self.num_special, use_layernorm=True)

        # RowInteraction: scales=(1,) and contiguous grouping => no mean-pooling information loss.
        self.row_interactor = RowInteraction(
            embed_dim=int(self.embed_dim),
            num_blocks=1,
            nhead=4,
            dim_feedforward=int(self.embed_dim) * int(ff_factor),
            num_cls=int(self.row_num_cls),
            num_global=int(self.num_global),
            rope_base=100000.0,
            dropout=float(dropout),
            activation=activation,
            norm_first=bool(norm_first),
            scales=(1,),
            window=int(row_window),
            num_random=0,
            group_mode="contiguous",
        )

        # Base path is always 512-d.
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.icl = ICLearning(
            max_classes=int(self.max_classes),
            d_model=512,
            num_blocks=int(icl_num_blocks),
            nhead=int(icl_nhead),
            dim_feedforward=512 * int(ff_factor),
            dropout=float(dropout),
            activation=activation,
            norm_first=bool(norm_first),
            perc_num_latents=int(perc_num_latents),
            perc_layers=int(perc_layers),
        )

        self.ensemble = ensemble if ensemble is not None else EnsembleConfig()

    @staticmethod
    def _masked_channel_mean(Z: Tensor, channel_mask: Tensor) -> Tensor:
        """Masked mean over channel dim, returning (B,T,512)."""
        if Z.ndim != 4:
            raise ValueError("Z must be (B,T,C,512)")
        m = channel_mask.bool().unsqueeze(-1)  # (B,T,C,1)
        Z_masked = Z * m.to(Z.dtype)
        denom = m.sum(dim=2).clamp(min=1).to(Z.dtype)  # (B,T,1)
        return Z_masked.sum(dim=2) / denom

    @staticmethod
    def _derive_d(channel_mask: Tensor, *, mode: EmbeddingMode) -> Tensor:
        """Derive per-task feature counts (excluding specials) for RowInteraction.

        RowInteraction expects d: (B,) giving number of valid feature tokens.
        We assume padding is consistent across time; if not, we take the minimum
        over time to stay safe (never treating padding as valid).
        """
        if channel_mask.ndim != 3:
            raise ValueError("channel_mask must be (B,T,C)")

        counts_bt = channel_mask.bool().sum(dim=2)  # (B,T)
        d_channels = counts_bt.min(dim=1).values  # (B,)

        if mode == "identity512":
            return d_channels
        return d_channels * 4

    @staticmethod
    def _maybe_shuffle_channels(Z: Tensor, channel_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Shuffle channels consistently across time for each task.

        CRITICAL: RowInteraction builds its padding mask from `d` by masking a *suffix*
        of feature tokens (i.e., it assumes all valid channels/features come first and
        all padding channels are at the end). Therefore, when we do channel permutation
        augmentation / ensemble, we MUST keep padding channels at the tail; otherwise
        `d` will mask the wrong tokens and training/inference degrades sharply.

        Valid-channel criterion must match `_derive_d`'s implicit assumption:
        a channel is considered valid only if it is valid for *all* time steps.
        (If it is invalid at any time step, treat it as padding and keep it at the end.)
        """

        B, T, C, D = Z.shape
        if C <= 1:
            return Z, channel_mask

        # valid_per_channel: (B,C) where True means the channel is valid for all timesteps.
        valid_per_channel = channel_mask.bool().all(dim=1)
        valid_counts = valid_per_channel.sum(dim=1)  # (B,)

        if bool((valid_counts <= 1).all()):
            return Z, channel_mask

        # Build a per-sample index: [permuted valid channels..., padding channels...]
        # Padding channels are always placed at the end to preserve the d-based suffix mask.
        idx_list: list[Tensor] = []
        for b in range(B):
            valid_idx = torch.nonzero(valid_per_channel[b], as_tuple=False).flatten()
            pad_idx = torch.nonzero(~valid_per_channel[b], as_tuple=False).flatten()

            if valid_idx.numel() <= 1:
                new_idx = torch.cat([valid_idx, pad_idx], dim=0)
            else:
                perm = torch.randperm(int(valid_idx.numel()), device=Z.device)
                new_idx = torch.cat([valid_idx.index_select(0, perm), pad_idx], dim=0)

            # Safety: ensure we have a full permutation of length C.
            if int(new_idx.numel()) != C:
                raise RuntimeError("Internal error: channel permutation size mismatch")
            idx_list.append(new_idx)

        perms = torch.stack(idx_list, dim=0)  # (B,C)

        idx_Z = perms[:, None, :, None].expand(B, T, C, D)
        idx_M = perms[:, None, :].expand(B, T, C)
        Z_shuf = Z.gather(dim=2, index=idx_Z)
        M_shuf = channel_mask.gather(dim=2, index=idx_M)
        return Z_shuf, M_shuf

    def _row_delta(self, x_tokens: Tensor, d: Optional[Tensor]) -> Tensor:
        """Compute RowInteraction output as (B,T,512)."""
        # RowInteraction returns (B,T,row_num_cls*embed_dim) which is always 512 here.
        out = self.row_interactor._aggregate_embeddings(x_tokens, d=d)  # type: ignore[attr-defined]
        if out.shape[-1] != 512:
            raise RuntimeError(f"Expected row_delta dim=512, got {out.shape[-1]}")
        return out

    def _forward_once(
        self,
        Z: Tensor,
        channel_mask: Tensor,
        y_train: Tensor,
        *,
        channel_shuffle: bool,
        debug_checks: bool,
        return_logits: bool,
        softmax_temperature: float,
    ) -> Tensor:
        if debug_checks:
            # Derive channel-count d in the same spirit as `_derive_d` (min over time),
            # and assert that padding is a strict suffix BEFORE any shuffling.
            d_channels_pre = channel_mask.bool().sum(dim=2).min(dim=1).values.to(torch.long)
            _assert_padding_suffix(channel_mask, d_channels_pre)

        if channel_shuffle:
            Z, channel_mask = self._maybe_shuffle_channels(Z, channel_mask)
            if debug_checks:
                d_channels_post = channel_mask.bool().sum(dim=2).min(dim=1).values.to(torch.long)
                _assert_padding_suffix(channel_mask, d_channels_post)

        # 1) tokenization for RowInteraction
        x_tokens, token_mask = self.embedder(Z, channel_mask)

        # 2) derive d from mask (feature counts); drop d if fully dense
        d = self._derive_d(channel_mask, mode=self.mode)
        total_F = (channel_mask.shape[2] if self.mode == "identity512" else channel_mask.shape[2] * 4)
        if bool((d == int(total_F)).all()):
            d = None

        # NOTE: RowInteraction does not currently accept an explicit key_padding_mask;
        # we use d to ignore padded feature tokens.
        if debug_checks:
            # Extra safety: verify suffix property right before RowInteraction.
            d_channels_now = channel_mask.bool().sum(dim=2).min(dim=1).values.to(torch.long)
            _assert_padding_suffix(channel_mask, d_channels_now)
        row_delta = self._row_delta(x_tokens, d=d)

        # 3) robust base: masked mean over channels
        row_base = self._masked_channel_mean(Z, channel_mask)  # (B,T,512)

        # 4) residual combine
        R = row_base + self.beta * row_delta

        # 5) ICL: uses y only on support slice and split-attention mask
        mgr_cfg = None
        if not self.training:
            # IMPORTANT: by default ICLearning's inference manager may choose CUDA
            # if available; to avoid device mismatch (weights vs inputs), we force
            # inference execution on the same device as Z.
            mgr_cfg = MgrConfig(device=str(Z.device))

        out = self.icl(
            R,
            y_train=y_train,
            return_logits=True,  # handle temperature/prob outside for ensembles
            softmax_temperature=float(softmax_temperature),
            mgr_config=mgr_cfg,
            n_classes=int(self.max_classes),
        )

        if return_logits:
            return out
        return torch.softmax(out / float(softmax_temperature), dim=-1)

    def forward(
        self,
        Z: Tensor,  # (B,T,Cmax,512)
        channel_mask: Tensor,  # (B,T,Cmax)
        y_train: Tensor,  # (B,train_size) MUST be remapped to 0..K-1 per dataset
        *,
        channel_shuffle: bool = False,
        debug_checks: bool = False,
        train_size: Optional[int] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        n_perm: Optional[int] = None,
    ) -> Tensor:
        """Forward.

        Requirements
        ------------
        - Sequence order must be: first train_size rows are support, remaining rows are query.
        - `y_train` must already be label-remapped to 0..K-1 per dataset/task.

        Inference ensemble
        ------------------
        In eval mode, optionally average logits over multiple random channel permutations.
        """
        if Z.ndim != 4:
            raise ValueError(f"Expected Z (B,T,C,512), got {tuple(Z.shape)}")
        if channel_mask.ndim != 3:
            raise ValueError(f"Expected channel_mask (B,T,C), got {tuple(channel_mask.shape)}")

        B, T, C, D = Z.shape
        if int(D) != 512:
            raise ValueError(f"Expected last dim 512, got {int(D)}")
        if y_train.ndim != 2 or y_train.shape[0] != B:
            raise ValueError("y_train must be (B,train_size)")

        inferred_train_size = int(y_train.shape[1])
        if train_size is not None and int(train_size) != inferred_train_size:
            raise ValueError("train_size must match y_train.shape[1]")
        if inferred_train_size > int(T):
            raise ValueError("train_size exceeds sequence length")

        if self.training:
            # training-time augmentation
            return self._forward_once(
                Z,
                channel_mask,
                y_train,
                channel_shuffle=bool(channel_shuffle),
                debug_checks=bool(debug_checks),
                return_logits=bool(return_logits),
                softmax_temperature=float(softmax_temperature),
            )

        # eval: ensemble over permutations
        n_perm_eff = int(self.ensemble.n_perm if n_perm is None else n_perm)
        n_perm_eff = max(1, n_perm_eff)

        if n_perm_eff == 1:
            return self._forward_once(
                Z,
                channel_mask,
                y_train,
                channel_shuffle=bool(channel_shuffle),
                debug_checks=bool(debug_checks),
                return_logits=bool(return_logits),
                softmax_temperature=float(softmax_temperature),
            )

        logits_sum: Optional[Tensor] = None
        for _ in range(n_perm_eff):
            logits = self._forward_once(
                Z,
                channel_mask,
                y_train,
                channel_shuffle=True,  # always permute for ensemble member
                debug_checks=bool(debug_checks),
                return_logits=True,
                softmax_temperature=float(softmax_temperature),
            )
            logits_sum = logits if logits_sum is None else (logits_sum + logits)

        avg_logits = logits_sum / float(n_perm_eff)
        if return_logits:
            return avg_logits
        return torch.softmax(avg_logits / float(softmax_temperature), dim=-1)


def _small_unit_test() -> None:
    torch.manual_seed(0)

    # ---- regression: channel_shuffle must not break d-suffix assumption ----
    B, T, Cmax, D = 2, 6, 8, 512
    Z = torch.randn(B, T, Cmax, D)

    channel_mask = torch.zeros(B, T, Cmax, dtype=torch.bool)
    channel_mask[0, :, :8] = True  # sample 0: 8 valid channels
    channel_mask[1, :, :5] = True  # sample 1: 5 valid channels, 3 padding at end

    d = MantisRowICLModel._derive_d(channel_mask, mode="identity512")
    assert torch.equal(d.cpu(), torch.tensor([8, 5], dtype=torch.long)), d
    _assert_padding_suffix(channel_mask, d)

    for _ in range(10):
        Zs, Ms = MantisRowICLModel._maybe_shuffle_channels(Z, channel_mask)
        d_s = MantisRowICLModel._derive_d(Ms, mode="identity512")
        _assert_padding_suffix(Ms, d_s)
        # Ensure d stays unchanged by shuffling.
        assert torch.equal(d_s.cpu(), torch.tensor([8, 5], dtype=torch.long)), d_s

    # Negative test: deliberately scatter a padding channel into the middle.
    bad_mask = channel_mask.clone()
    # swap a valid channel with a padding channel for sample 1
    bad_mask[1, :, 3] = False
    bad_mask[1, :, 6] = True
    bad_d = MantisRowICLModel._derive_d(bad_mask, mode="identity512")
    try:
        _assert_padding_suffix(bad_mask, bad_d)
        raise RuntimeError("Expected _assert_padding_suffix to fail, but it passed")
    except AssertionError:
        pass

    # ---- quick end-to-end sanity (kept minimal) ----
    train_size = 3
    max_classes = 10
    y_train = torch.randint(0, max_classes, (B, train_size), dtype=torch.long)
    for mode in ("identity512", "chunk4x128"):
        model = MantisRowICLModel(max_classes=max_classes, mode=mode)
        model.train()
        out = model(
            Z,
            channel_mask,
            y_train,
            train_size=train_size,
            channel_shuffle=True,
            debug_checks=True,
            return_logits=True,
        )
        assert out.shape == (B, T - train_size, max_classes), (mode, out.shape)

        model.eval()
        out2 = model(
            Z,
            channel_mask,
            y_train,
            train_size=train_size,
            debug_checks=True,
            return_logits=True,
            n_perm=4,
        )
        assert out2.shape == (B, T - train_size, max_classes), (mode, out2.shape)

    print("[OK] mantis_rowicl small unit test passed")


if __name__ == "__main__":
    # Run with:
    #   python -m orion_msp.model.mantis_rowicl
    _small_unit_test()

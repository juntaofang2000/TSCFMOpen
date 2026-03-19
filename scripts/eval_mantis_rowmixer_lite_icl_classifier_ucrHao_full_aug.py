from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


# Ensure we import the local workspace package (repo_root/src)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tabicl.model.mantis_tabicl import build_mantis_encoder  # noqa: E402
from tabicl.prior.data_reader import DataReader  # noqa: E402
from tabicl.sklearn.classifier import MantisICLClassifierV2  # noqa: E402
from tabicl.model.mantis_dev.adapters import VarianceBasedSelector  # noqa: E402

from orion_msp.model.mantis_plus_rowmixer_lite_icl import _MantisPlusRowMixerLiteICL  # noqa: E402
from orion_msp.model.rowmixer_lite_icl import RowMixerLiteICL  # noqa: E402


def _patch_safe_mantis_encode() -> None:
    """Runtime guard for old _encode_with_mantis implementations.

    Some environments may still have an older model file where
    _encode_with_mantis uses next(self.mantis_model.parameters()) directly,
    which can raise StopIteration in DataParallel replicas.
    """

    def _safe_encode_with_mantis(self, x_flat: torch.Tensor) -> torch.Tensor:
        try:
            device = next(self.mantis_model.parameters()).device
        except StopIteration:
            try:
                device = next(self.mantis_model.buffers()).device
            except StopIteration:
                device = x_flat.device

        x_flat = x_flat.to(device)
        outs: list[torch.Tensor] = []
        batch_size = max(1, int(self.mantis_batch_size))
        for i in range(0, x_flat.shape[0], batch_size):
            outs.append(self.mantis_model(x_flat[i : i + batch_size]))
        return torch.cat(outs, dim=0)

    _MantisPlusRowMixerLiteICL._encode_with_mantis = _safe_encode_with_mantis


def _parse_gpu_ids(gpu_ids: str | None) -> list[int]:
    if gpu_ids is None:
        return []
    ids: list[int] = []
    for part in str(gpu_ids).split(","):
        s = part.strip()
        if not s:
            continue
        ids.append(int(s))
    return ids


def _device_label(dev: torch.device) -> str:
    if dev.type != "cuda":
        return str(dev)
    idx = 0 if dev.index is None else int(dev.index)
    return f"cuda:{idx}"


def _print_effective_runtime_config(*, title: str, payload: dict) -> None:
    """Pretty-print a JSON payload for reproducibility/debugging."""
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(payload)
    print(f"\n{title}\n{text}\n")


def _remap_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map labels to contiguous ints [0..K-1] based on train set."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    classes = np.unique(y_train)
    cls_to_id = {c: i for i, c in enumerate(classes.tolist())}

    y_train_m = np.vectorize(cls_to_id.get)(y_train)
    y_test_m = np.vectorize(cls_to_id.get)(y_test)

    if np.any(y_test_m == None):  # noqa: E711
        missing = set(np.unique(y_test)) - set(classes)
        raise ValueError(f"Test labels contain unseen classes: {sorted(missing)}")

    return y_train_m.astype(np.int64), y_test_m.astype(np.int64), classes


def _select_support_indices(y: np.ndarray, support_size: int, seed: int) -> np.ndarray:
    """Pick support indices ensuring all classes appear at least once."""
    rng = np.random.RandomState(int(seed))
    y = np.asarray(y)
    classes = np.unique(y)
    n_classes = int(classes.shape[0])
    support_size = int(support_size)
    if support_size < n_classes:
        support_size = n_classes

    chosen: list[int] = []
    remaining: list[int] = []

    for c in classes:
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        pick = int(rng.choice(idx))
        chosen.append(pick)
        remaining.extend([int(i) for i in idx if int(i) != pick])

    need = support_size - len(chosen)
    if need > 0:
        remaining_np = np.array(remaining, dtype=np.int64)
        if remaining_np.size > 0:
            extra = rng.choice(remaining_np, size=min(need, remaining_np.size), replace=False)
            chosen.extend([int(i) for i in extra])

    return np.array(chosen, dtype=np.int64)


def _select_support_subset_per_class(
    y: np.ndarray,
    *,
    shots_ratio: float,
    seed: int,
    replace: bool = False,
) -> np.ndarray:
    """Pick a class-balanced support subset using per-class ratio.

    For each class c with count N_c in the full train set, sample
    round(N_c * shots_ratio) items (at least 1).
    """
    rng = np.random.RandomState(int(seed))
    y = np.asarray(y)
    classes = np.unique(y)
    shots_ratio = float(shots_ratio)
    if shots_ratio <= 0:
        raise ValueError(f"shots_ratio must be > 0, got {shots_ratio}")

    chosen: list[int] = []
    for c in classes.tolist():
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        shots_per_class = max(1, int(round(float(idx.size) * shots_ratio)))
        do_replace = bool(replace) or idx.size < shots_per_class
        picks = rng.choice(idx, size=shots_per_class, replace=do_replace)
        chosen.extend([int(i) for i in np.asarray(picks, dtype=np.int64).tolist()])

    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def _smooth_last_axis(X: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a light moving-average smoothing along the last axis."""
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return np.asarray(X, dtype=np.float32)
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    flat = np.asarray(X, dtype=np.float32).reshape(-1, X.shape[-1])
    out = np.stack([np.convolve(row, kernel, mode="same") for row in flat], axis=0)
    return out.reshape(X.shape).astype(np.float32, copy=False)


def _resize_last_axis_linear(X: np.ndarray, new_len: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    old_len = int(X.shape[-1])
    if old_len == int(new_len):
        return X.copy()
    xp = np.linspace(0.0, 1.0, old_len, dtype=np.float32)
    xnew = np.linspace(0.0, 1.0, int(new_len), dtype=np.float32)
    flat = X.reshape(-1, old_len)
    out = np.empty((flat.shape[0], int(new_len)), dtype=np.float32)
    for i in range(flat.shape[0]):
        out[i] = np.interp(xnew, xp, flat[i]).astype(np.float32)
    return out.reshape(*X.shape[:-1], int(new_len))


def _apply_timeseries_tta(
    X: np.ndarray,
    *,
    seed: int,
    amp_scale_std: float = 0.0,
    jitter_std: float = 0.0,
    shift_max_ratio: float = 0.0,
    magwarp_std: float = 0.0,
    crop_resize_ratio: float = 0.0,
    smooth_kernel: int = 0,
) -> np.ndarray:
    """Light test-time augmentation for 1D or multichannel time series.

    Supports arrays shaped (N, L) or (N, C, L). The augmentation is intentionally
    mild so that class semantics are preserved.
    """
    rng = np.random.RandomState(int(seed))
    Xv = np.asarray(X, dtype=np.float32).copy()
    if Xv.ndim not in {2, 3}:
        raise ValueError(f"TTA expects 2D or 3D time series, got {Xv.shape}")

    batch_shape = (Xv.shape[0],) + ((1,) * (Xv.ndim - 1))
    if float(amp_scale_std) > 0:
        scales = rng.normal(loc=1.0, scale=float(amp_scale_std), size=batch_shape).astype(np.float32)
        Xv *= scales

    if float(jitter_std) > 0:
        sample_std = np.std(Xv, axis=-1, keepdims=True).astype(np.float32)
        noise = rng.normal(loc=0.0, scale=1.0, size=Xv.shape).astype(np.float32)
        Xv += noise * (float(jitter_std) * (sample_std + 1e-6))

    if float(shift_max_ratio) > 0:
        max_shift = int(round(float(shift_max_ratio) * float(Xv.shape[-1])))
        if max_shift > 0:
            shifts = rng.randint(-max_shift, max_shift + 1, size=Xv.shape[0])
            for i, shift in enumerate(shifts.tolist()):
                Xv[i] = np.roll(Xv[i], int(shift), axis=-1)

    if float(magwarp_std) > 0:
        n_knots = 4
        knot_x = np.linspace(0.0, 1.0, n_knots + 2, dtype=np.float32)
        target_x = np.linspace(0.0, 1.0, Xv.shape[-1], dtype=np.float32)
        flat = Xv.reshape(-1, Xv.shape[-1])
        for i in range(flat.shape[0]):
            knot_y = rng.normal(loc=1.0, scale=float(magwarp_std), size=n_knots + 2).astype(np.float32)
            warp = np.interp(target_x, knot_x, knot_y).astype(np.float32)
            flat[i] *= warp
        Xv = flat.reshape(Xv.shape)

    if float(crop_resize_ratio) > 0:
        L = int(Xv.shape[-1])
        keep_ratio = max(0.8, 1.0 - float(crop_resize_ratio))
        crop_len = max(8, int(round(L * keep_ratio)))
        if crop_len < L:
            starts = rng.randint(0, L - crop_len + 1, size=Xv.shape[0])
            cropped = np.empty((*Xv.shape[:-1], crop_len), dtype=np.float32)
            for i, s in enumerate(starts.tolist()):
                cropped[i] = Xv[i, ..., s : s + crop_len]
            Xv = _resize_last_axis_linear(cropped, L)

    if int(smooth_kernel) > 1:
        Xv = _smooth_last_axis(Xv, int(smooth_kernel))

    return np.asarray(Xv, dtype=np.float32)




def _ensure_2d_timeseries(X: np.ndarray) -> np.ndarray:
    """Coerce X into (N, L). Supports UCR (N,L) and UEA (N,C,L)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X
    if X.ndim == 3:
        if X.shape[1] == 1:
            return X[:, 0, :]
        return X
    raise ValueError(f"Unexpected X shape: {X.shape}")


def _uea_concat_and_sample(
    X: np.ndarray,
    *,
    target_len: int,
    seed: int,
    mode: str = "center",
) -> np.ndarray:
    """Convert UEA (N,C,L) into (N,target_len) by channel-concat then crop/pad."""

    X = np.asarray(X, dtype=np.float32)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")

    if X.ndim == 2:
        X_flat = X
    elif X.ndim == 3:
        N, C, L = X.shape
        X_flat = X.reshape(N, C * L)
    else:
        raise ValueError(f"Unexpected UEA X shape: {X.shape}")

    N, Ltot = X_flat.shape
    if Ltot == target_len:
        return X_flat

    if Ltot < target_len:
        pad = np.zeros((N, target_len - Ltot), dtype=np.float32)
        return np.concatenate([X_flat, pad], axis=1)

    if mode not in {"center", "random"}:
        raise ValueError(f"mode must be 'center' or 'random', got {mode}")
    if mode == "center":
        start = (Ltot - target_len) // 2
        return X_flat[:, start : start + target_len]

    rng = np.random.RandomState(int(seed))
    starts = rng.randint(0, Ltot - target_len + 1, size=N)
    out = np.empty((N, target_len), dtype=np.float32)
    for i, s in enumerate(starts.tolist()):
        out[i] = X_flat[i, s : s + target_len]
    return out


def _ensure_3d_timeseries(X: np.ndarray) -> np.ndarray:
    """Coerce X into (N, C, L)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        return X[None, None, :]
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"Unexpected X shape for 3d series: {X.shape}")


def _uea_crop_pad_per_channel(
    X: np.ndarray,
    *,
    target_len: int,
    seed: int,
    mode: str = "center",
) -> np.ndarray:
    """Crop/pad UEA to (N,C,target_len) without mixing channels."""
    X3 = _ensure_3d_timeseries(X)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")

    N, C, Ltot = X3.shape
    if Ltot == target_len:
        return X3
    if Ltot < target_len:
        pad = np.zeros((N, C, target_len - Ltot), dtype=np.float32)
        return np.concatenate([X3, pad], axis=2)

    if mode not in {"center", "random"}:
        raise ValueError(f"mode must be 'center' or 'random', got {mode}")
    if mode == "center":
        start = (Ltot - target_len) // 2
        return X3[:, :, start : start + target_len]

    rng = np.random.RandomState(int(seed))
    starts = rng.randint(0, Ltot - target_len + 1, size=N)
    out = np.empty((N, C, target_len), dtype=np.float32)
    for i, s in enumerate(starts.tolist()):
        out[i] = X3[i, :, s : s + target_len]
    return out


def _maybe_select_channels_uea(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    *,
    enabled: bool,
    new_num_channels: int | None,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply VarianceBasedSelector on UEA multichannel time series."""
    if not enabled:
        return X_train_raw, X_test_raw
    if new_num_channels is None:
        return X_train_raw, X_test_raw

    X_train_np = _ensure_3d_timeseries(X_train_raw)
    X_test_np = _ensure_3d_timeseries(X_test_raw)

    if X_train_np.ndim != 3:
        return X_train_raw, X_test_raw

    _, c_train, _ = X_train_np.shape
    if c_train <= 1:
        return X_train_np, X_test_np

    k = int(new_num_channels)
    k = max(1, min(k, c_train))
    if k == c_train:
        return X_train_np, X_test_np

    if dataset_name is not None:
        print(f"[VarSelector][UEA] {dataset_name}: channels {c_train} -> {k}")

    selector = VarianceBasedSelector(k)
    selector.fit(X_train_np)
    X_train_sel = selector.transform(X_train_np)
    X_test_sel = selector.transform(X_test_np)
    return X_train_sel, X_test_sel


def _load_model_hparams_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid model_hparams_json: {path}")
    return obj


def _extract_full_state_dict(ckpt_obj: object) -> dict:
    if isinstance(ckpt_obj, dict):
        state_dict = ckpt_obj.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    raise ValueError("Invalid full model checkpoint: expected dict with 'state_dict' or raw state_dict.")


def _build_rowmixer_from_hparams(hparams: dict) -> RowMixerLiteICL:
    cfg = hparams
    # Compatibility: allow training-saved JSON with top-level model_config
    if isinstance(hparams.get("model_config"), dict):
        cfg = hparams["model_config"]

    row_cfg = cfg.get("rowmixer_icl", {})
    if not isinstance(row_cfg, dict) or not row_cfg:
        raise ValueError("Missing 'rowmixer_icl' in model_hparams_json")

    return RowMixerLiteICL(
        max_classes=int(row_cfg.get("max_classes", 10)),
        embed_dim=int(row_cfg.get("embed_dim", 128)),
        patch_size=int(row_cfg.get("patch_size", 8)),
        row_num_blocks=int(row_cfg.get("row_num_blocks", 3)),
        row_nhead=int(row_cfg.get("row_nhead", 8)),
        row_num_cls=int(row_cfg.get("row_num_cls", 4)),
        row_num_global=int(row_cfg.get("row_num_global", 2)),
        icl_num_blocks=int(row_cfg.get("icl_num_blocks", 12)),
        icl_nhead=int(row_cfg.get("icl_nhead", 4)),
        ff_factor=int(row_cfg.get("ff_factor", 2)),
        dropout=float(row_cfg.get("dropout", 0.0)),
        activation=str(row_cfg.get("activation", "gelu")),
        norm_first=bool(row_cfg.get("norm_first", True)),
        shuffle_p=float(row_cfg.get("shuffle_p", 0.25)),
        perc_num_latents=int(row_cfg.get("perc_num_latents", 0)),
        perc_layers=int(row_cfg.get("perc_layers", 0)),
    )


@torch.no_grad()
def _predict_direct_logits(
    model: torch.nn.Module,
    *,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    query_batch_size: int,
    softmax_temperature: float,
) -> np.ndarray:
    """Return query logits using (support + query) ICL episodes."""

    device = next(model.parameters()).device
    X_support_np = np.asarray(X_support, dtype=np.float32)
    X_query_np = np.asarray(X_query, dtype=np.float32)

    X_support_t = torch.from_numpy(X_support_np).to(device)
    y_support_t = torch.from_numpy(np.asarray(y_support, dtype=np.int64)).to(device)

    if X_support_t.dim() not in {2, 3}:
        raise ValueError(f"Unsupported support shape: {tuple(X_support_t.shape)}")
    if X_query_np.ndim != X_support_t.dim():
        raise ValueError(f"Support/query dim mismatch: support={tuple(X_support_t.shape)}, query={tuple(X_query_np.shape)}")

    logits_out: list[np.ndarray] = []
    bs = max(1, int(query_batch_size))
    S = int(X_support_t.shape[0])
    K = int(np.max(np.asarray(y_support, dtype=np.int64))) + 1

    for i in range(0, X_query_np.shape[0], bs):
        q = torch.from_numpy(X_query_np[i : i + bs]).to(device)
        B = int(q.shape[0])

        if X_support_t.dim() == 2:
            support_rep = X_support_t.unsqueeze(0).expand(B, S, X_support_t.shape[-1])
            q_rep = q.unsqueeze(1)
        else:
            support_rep = X_support_t.unsqueeze(0).expand(B, S, X_support_t.shape[-2], X_support_t.shape[-1])
            q_rep = q.unsqueeze(1)

        X_episode = torch.cat([support_rep, q_rep], dim=1)
        y_train = y_support_t.unsqueeze(0).expand(B, S)

        logits = model(
            X_episode,
            y_train=y_train,
            d=None,
            return_logits=True,
            softmax_temperature=float(softmax_temperature),
        )
        logits_out.append(logits[:, 0, :K].detach().cpu().numpy())
        del q, support_rep, q_rep, X_episode, y_train, logits

    return np.concatenate(logits_out, axis=0)


@torch.no_grad()
def _predict_direct(
    model: torch.nn.Module,
    *,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    query_batch_size: int,
    softmax_temperature: float,
) -> np.ndarray:
    logits = _predict_direct_logits(
        model,
        X_support=X_support,
        y_support=y_support,
        X_query=X_query,
        query_batch_size=query_batch_size,
        softmax_temperature=softmax_temperature,
    )
    return np.argmax(logits, axis=-1).astype(np.int64)


def _predict_with_custom_ensemble(
    model: torch.nn.Module,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    query_batch_size: int,
    softmax_temperature: float,
    ensemble_kind: str,
    seed: int,
    support_size: int,
    support_ensemble_members: int,
    support_shots_per_class: float | None,
    support_sample_with_replacement: bool,
    tta_n_views: int,
    tta_apply_to: str,
    tta_amp_scale_std: float,
    tta_jitter_std: float,
    tta_shift_max_ratio: float,
    tta_magwarp_std: float,
    tta_crop_resize_ratio: float,
    tta_smooth_kernel: int,
) -> np.ndarray:
    """Custom ensemble that replaces label-shift augmentation.

    ensemble_kind:
      - support_subset
      - timeseries_tta
      - support_subset_tta
    """
    y_train = np.asarray(y_train, dtype=np.int64)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_query = np.asarray(X_query, dtype=np.float32)

    use_support_subset = ensemble_kind in {"support_subset", "support_subset_tta"}
    use_tta = ensemble_kind in {"timeseries_tta", "support_subset_tta"}

    support_members = max(1, int(support_ensemble_members if use_support_subset else 1))
    tta_views = max(1, int(tta_n_views if use_tta else 1))
    total_members = support_members * tta_views

    avg_logits: np.ndarray | None = None
    member_idx = 0

    for sup_view in range(support_members):
        member_seed = int(seed) + 1009 * sup_view
        if use_support_subset:
            if support_shots_per_class is not None and float(support_shots_per_class) > 0:
                sup_idx = _select_support_subset_per_class(
                    y_train,
                    shots_ratio=float(support_shots_per_class),
                    seed=member_seed,
                    replace=bool(support_sample_with_replacement),
                )
            else:
                sup_idx = _select_support_indices(y_train, support_size=int(support_size), seed=member_seed)
        else:
            sup_idx = np.arange(y_train.shape[0], dtype=np.int64)

        X_sup_base = X_train[sup_idx]
        y_sup = y_train[sup_idx]

        for tta_view in range(tta_views):
            view_seed = int(seed) + 100003 * sup_view + 9176 * tta_view
            X_sup_view = X_sup_base
            X_q_view = X_query

            if use_tta:
                if str(tta_apply_to) in {"support", "both"}:
                    X_sup_view = _apply_timeseries_tta(
                        X_sup_base,
                        seed=view_seed + 17,
                        amp_scale_std=float(tta_amp_scale_std),
                        jitter_std=float(tta_jitter_std),
                        shift_max_ratio=float(tta_shift_max_ratio),
                        magwarp_std=float(tta_magwarp_std),
                        crop_resize_ratio=float(tta_crop_resize_ratio),
                        smooth_kernel=int(tta_smooth_kernel),
                    )
                if str(tta_apply_to) in {"query", "both"}:
                    X_q_view = _apply_timeseries_tta(
                        X_query,
                        seed=view_seed + 31,
                        amp_scale_std=float(tta_amp_scale_std),
                        jitter_std=float(tta_jitter_std),
                        shift_max_ratio=float(tta_shift_max_ratio),
                        magwarp_std=float(tta_magwarp_std),
                        crop_resize_ratio=float(tta_crop_resize_ratio),
                        smooth_kernel=int(tta_smooth_kernel),
                    )

            logits = _predict_direct_logits(
                model,
                X_support=X_sup_view,
                y_support=y_sup,
                X_query=X_q_view,
                query_batch_size=int(query_batch_size),
                softmax_temperature=float(softmax_temperature),
            )

            if avg_logits is None:
                avg_logits = logits
            else:
                avg_logits += logits
            member_idx += 1

    if avg_logits is None or member_idx <= 0:
        raise RuntimeError("Custom ensemble produced no members.")

    avg_logits /= float(member_idx)
    return np.argmax(avg_logits, axis=-1).astype(np.int64)


def main() -> None:
    _patch_safe_mantis_encode()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Mantis+RowMixerLiteICL model on UCR/UEA. "
            "This version loads full _MantisPlusRowMixerLiteICL checkpoint + model_hparams JSON."
        )
    )

    parser.add_argument("--mode", type=str, default="classifier_v2", choices=["direct", "classifier_v2"])

    parser.add_argument(
        "--full_ckpt",
        type=str,
        default="/data0/fangjuntao2025/TIC-FS/checkpoints/mantis_rowmixer_lite_icl_full.pt",
        help="Path to full _MantisPlusRowMixerLiteICL checkpoint (.pt).",
    )
    parser.add_argument(
        "--model_hparams_json",
        type=str,
        default="/data0/fangjuntao2025/TIC-FS/checkpoints/mantis_rowmixer_lite_icl_model_hparams.json",
        help="Path to model hyperparameters JSON used to rebuild the full model.",
    )

    parser.add_argument("--ucr_path", type=str, default="/home/hzf00006536/fjt/CauKer/UCRdata/")
    parser.add_argument("--uea_path", type=str, default="/home/hzf00006536/fjt/CauKer/UEAdata/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--multi_gpu", action="store_true", help="Use DataParallel across multiple CUDA devices.")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated CUDA ids for multi-GPU, e.g. '0,1,2'. Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--mantis_batch_size_override",
        type=int,
        default=None,
        help="Override mantis encoder batch size in full model for lower memory usage.",
    )
    parser.add_argument(
        "--v2_batch_size",
        type=int,
        default=2,
        help="Batch size for MantisICLClassifierV2 ensemble-member forwarding (lower helps avoid OOM).",
    )
    parser.add_argument(
        "--empty_cache_each_dataset",
        action="store_true",
        help="Call torch.cuda.empty_cache() after each dataset.",
    )

    parser.add_argument("--suite", type=str, default="ucr", choices=["ucr", "uea"])
    parser.add_argument("--dataset", type=str, default=None, help="Evaluate a single dataset name")

    parser.add_argument("--n_estimators", type=int, default=32)

    v2_class_shift_group = parser.add_mutually_exclusive_group()
    v2_class_shift_group.add_argument("--v2_class_shift", dest="v2_class_shift", action="store_true", default=True)
    v2_class_shift_group.add_argument("--no_v2_class_shift", dest="v2_class_shift", action="store_false")
    parser.add_argument("--v2_n_augmentations", type=int, default=1)
    parser.add_argument("--v2_crop_rate_lo", type=float, default=0.0)
    parser.add_argument("--v2_crop_rate_hi", type=float, default=0.00)

    parser.add_argument(
        "--inference_ensemble",
        type=str,
        default="original",
        choices=["original", "support_subset", "timeseries_tta", "support_subset_tta"],
        help=(
            "Inference-time ensembling strategy. "
            "'original' keeps the legacy classifier_v2 label-shift ensemble. "
            "The other modes replace label-shift with support-subset and/or light time-series TTA."
        ),
    )
    parser.add_argument("--support_ensemble_members", type=int, default=4)
    parser.add_argument(
        "--support_shots_per_class",
        type=float,
        default=0.0,
        help=(
            "When using support-subset ensemble, each class samples round(class_count * ratio) supports per member "
            "from the full train set. Set <=0 to fallback to --support_size class-cover sampling."
        ),
    )
    parser.add_argument("--support_sample_with_replacement", action="store_true")
    parser.add_argument("--tta_n_views", type=int, default=4)
    parser.add_argument("--tta_apply_to", type=str, default="query", choices=["query", "support", "both"])
    parser.add_argument("--tta_amp_scale_std", type=float, default=0.05)
    parser.add_argument("--tta_jitter_std", type=float, default=0.01)
    parser.add_argument("--tta_shift_max_ratio", type=float, default=0.02)
    parser.add_argument("--tta_magwarp_std", type=float, default=0.03)
    parser.add_argument("--tta_crop_resize_ratio", type=float, default=0.0)
    parser.add_argument("--tta_smooth_kernel", type=int, default=0)

    parser.add_argument("--support_size", type=int, default=128)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--softmax_temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--uea_mode", type=str, default="center", choices=["center", "random"])
    parser.add_argument("--uea_use_var_selector", action="store_true")
    parser.add_argument("--uea_var_num_channels", type=int, default=None)
    parser.add_argument("--uea_fusion", type=str, default="concat", choices=["concat", "sum_embed"])

    args = parser.parse_args()
    device = torch.device(args.device)

    selected_gpu_ids: list[int] = []
    using_multi_gpu = False
    num_parallel_gpus = 1
    if device.type == "cuda" and torch.cuda.is_available():
        requested_ids = _parse_gpu_ids(args.gpu_ids)
        if args.multi_gpu:
            if requested_ids:
                selected_gpu_ids = requested_ids
            else:
                selected_gpu_ids = list(range(torch.cuda.device_count()))
            if len(selected_gpu_ids) >= 2:
                device = torch.device(f"cuda:{selected_gpu_ids[0]}")
                using_multi_gpu = True
                num_parallel_gpus = len(selected_gpu_ids)

        # TensorFloat32 can reduce memory pressure on Ampere+ and usually speeds up inference.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    resolved_v2_batch_size = max(1, int(args.v2_batch_size))
    resolved_query_batch_size = max(1, int(args.query_batch_size))
    if using_multi_gpu:
        # DataParallel splits along batch dim; too-small batch can create empty shards and crash some CUDA kernels.
        if resolved_v2_batch_size < num_parallel_gpus:
            print(
                f"[Warn] v2_batch_size={resolved_v2_batch_size} < num_gpus={num_parallel_gpus}. "
                f"Auto-adjusting to {num_parallel_gpus} for stable multi-GPU execution."
            )
            resolved_v2_batch_size = num_parallel_gpus
        if resolved_query_batch_size < num_parallel_gpus:
            print(
                f"[Warn] query_batch_size={resolved_query_batch_size} < num_gpus={num_parallel_gpus}. "
                f"Auto-adjusting to {num_parallel_gpus} for stable multi-GPU execution."
            )
            resolved_query_batch_size = num_parallel_gpus

    model_hparams = _load_model_hparams_json(str(args.model_hparams_json))
    cfg = model_hparams
    if isinstance(model_hparams.get("model_config"), dict):
        cfg = model_hparams["model_config"]

    mantis_cfg = cfg.get("mantis", {})
    mantis_hidden_dim = int(mantis_cfg.get("hidden_dim", 512))
    mantis_seq_len = int(mantis_cfg.get("seq_len", 512))

    mantis_model = build_mantis_encoder(
        mantis_checkpoint=None,
        device=device,
        hidden_dim=int(mantis_hidden_dim),
        seq_len=int(mantis_seq_len),
        num_patches=int(mantis_cfg.get("num_patches", 32)),
        use_fddm=bool(mantis_cfg.get("use_fddm", False)),
        num_channels=1,
        strict=False,
    )
    for p in mantis_model.parameters():
        p.requires_grad_(False)
    mantis_model.eval()

    rowmixer_icl = _build_rowmixer_from_hparams(model_hparams).to(device)
    for p in rowmixer_icl.parameters():
        p.requires_grad_(False)
    rowmixer_icl.eval()

    resolved_mantis_batch_size = int(mantis_cfg.get("batch_size", 64))
    if args.mantis_batch_size_override is not None:
        resolved_mantis_batch_size = max(1, int(args.mantis_batch_size_override))

    custom_model = _MantisPlusRowMixerLiteICL(
        mantis_model=mantis_model,
        rowmixer_icl=rowmixer_icl,
        mantis_seq_len=int(mantis_seq_len),
        mantis_batch_size=resolved_mantis_batch_size,
    ).to(device)
    for p in custom_model.parameters():
        p.requires_grad_(False)
    custom_model.eval()

    ckpt_obj = torch.load(str(args.full_ckpt), map_location="cpu")
    state_dict = _extract_full_state_dict(ckpt_obj)
    cleaned = {str(k).replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    custom_model.load_state_dict(cleaned, strict=True)

    model_for_infer: torch.nn.Module = custom_model
    if using_multi_gpu:
        model_for_infer = torch.nn.DataParallel(custom_model, device_ids=selected_gpu_ids, output_device=selected_gpu_ids[0])
        # sklearn wrapper accesses model_.max_classes directly.
        setattr(model_for_infer, "max_classes", int(getattr(custom_model, "max_classes", 0)))

    _print_effective_runtime_config(
        title="[Eval][EffectiveRuntimeConfig]",
        payload={
            "cli_args": vars(args),
            "resolved": {
                "full_ckpt": str(args.full_ckpt),
                "model_hparams_json": str(args.model_hparams_json),
                "device": _device_label(device),
                "multi_gpu": bool(using_multi_gpu),
                "gpu_ids": selected_gpu_ids,
                "num_parallel_gpus": int(num_parallel_gpus),
                "mantis": mantis_cfg,
                "mantis_batch_size": int(resolved_mantis_batch_size),
                "v2_batch_size": int(resolved_v2_batch_size),
                "query_batch_size": int(resolved_query_batch_size),
                "rowmixer_icl": cfg.get("rowmixer_icl", {}),
            },
        },
    )

    reader = DataReader(
        UEA_data_path=str(args.uea_path),
        UCR_data_path=str(args.ucr_path),
        transform_ts_size=int(mantis_seq_len),
    )

    if args.dataset is not None:
        dataset_names = [args.dataset]
    else:
        dataset_names = list(reader.dataset_list_uea) if args.suite == "uea" else list(reader.dataset_list_ucr)

    print(f"[Eval] full_ckpt: {args.full_ckpt}")
    print(f"[Eval] model_hparams_json: {args.model_hparams_json}")
    print(f"[Eval] suite: {args.suite}")
    print(f"[Eval] dataset_count: {len(dataset_names)}")
    print(f"[Eval] mode: {args.mode}")

    accs: list[float] = []
    for name in dataset_names:
        try:
            X_tr, y_tr = reader.read_dataset(name, which_set="train")
            X_te, y_te = reader.read_dataset(name, which_set="test")

            if args.suite == "uea":
                X_tr, X_te = _maybe_select_channels_uea(
                    X_tr,
                    X_te,
                    enabled=bool(args.uea_use_var_selector),
                    new_num_channels=(None if args.uea_var_num_channels is None else int(args.uea_var_num_channels)),
                    dataset_name=name,
                )

                if args.uea_fusion == "sum_embed":
                    X_tr_2d = _uea_crop_pad_per_channel(
                        X_tr,
                        target_len=int(mantis_seq_len),
                        seed=int(args.seed),
                        mode=str(args.uea_mode),
                    )
                    X_te_2d = _uea_crop_pad_per_channel(
                        X_te,
                        target_len=int(mantis_seq_len),
                        seed=int(args.seed) + 1,
                        mode=str(args.uea_mode),
                    )
                else:
                    X_tr_2d = _uea_concat_and_sample(
                        X_tr,
                        target_len=int(mantis_seq_len),
                        seed=int(args.seed),
                        mode=str(args.uea_mode),
                    )
                    X_te_2d = _uea_concat_and_sample(
                        X_te,
                        target_len=int(mantis_seq_len),
                        seed=int(args.seed) + 1,
                        mode=str(args.uea_mode),
                    )
            else:
                X_tr_2d = _ensure_2d_timeseries(X_tr)
                X_te_2d = _ensure_2d_timeseries(X_te)
                if X_tr_2d.ndim != 2 or X_te_2d.ndim != 2:
                    raise ValueError(
                        f"Expected univariate (N,L) arrays for UCR; got train {X_tr_2d.shape}, test {X_te_2d.shape}. "
                        f"Did you mean --suite uea ?"
                    )

            y_tr_m, y_te_m, _classes = _remap_labels(y_tr, y_te)

            if args.inference_ensemble == "original":
                if args.mode == "classifier_v2":
                    if args.suite == "uea" and args.uea_fusion == "sum_embed":
                        raise ValueError(
                            "classifier_v2 expects 2D inputs (n_samples, length). "
                            "For UEA multichannel sum_embed, use --mode direct (or set --uea_fusion concat)."
                        )
                    crop_lo = float(args.v2_crop_rate_lo)
                    crop_hi = float(args.v2_crop_rate_hi)
                    if not (0.0 <= crop_lo <= crop_hi < 1.0):
                        raise ValueError(
                            f"Invalid v2 crop range: lo={crop_lo}, hi={crop_hi}. Must satisfy 0 <= lo <= hi < 1."
                        )

                    clf = MantisICLClassifierV2(
                        n_estimators=int(args.n_estimators),
                        class_shift=bool(args.v2_class_shift),
                        crop_rate_range=(crop_lo, crop_hi),
                        n_augmentations=int(args.v2_n_augmentations),
                        softmax_temperature=float(args.softmax_temperature),
                        batch_size=int(resolved_v2_batch_size),
                        device=device,
                        random_state=int(args.seed),
                        verbose=False,
                        model_path=None,
                        allow_auto_download=False,
                        checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
                    )
                    clf.model_ = model_for_infer
                    clf.fit(X_tr_2d, y_tr_m)
                    y_pred = clf.predict(X_te_2d)
                    acc = float(np.mean(y_pred == y_te_m))
                else:
                    sup_idx = _select_support_indices(y_tr_m, support_size=int(args.support_size), seed=int(args.seed))
                    X_sup = X_tr_2d[sup_idx]
                    y_sup = y_tr_m[sup_idx]

                    y_pred = _predict_direct(
                        model_for_infer,
                        X_support=X_sup,
                        y_support=y_sup,
                        X_query=X_te_2d,
                        query_batch_size=int(resolved_query_batch_size),
                        softmax_temperature=float(args.softmax_temperature),
                    )
                    acc = float(np.mean(y_pred == y_te_m))
            else:
                shots_per_class = None if float(args.support_shots_per_class) <= 0 else float(args.support_shots_per_class)
                y_pred = _predict_with_custom_ensemble(
                    model_for_infer,
                    X_train=X_tr_2d,
                    y_train=y_tr_m,
                    X_query=X_te_2d,
                    query_batch_size=int(resolved_query_batch_size),
                    softmax_temperature=float(args.softmax_temperature),
                    ensemble_kind=str(args.inference_ensemble),
                    seed=int(args.seed),
                    support_size=int(args.support_size),
                    support_ensemble_members=int(args.support_ensemble_members),
                    support_shots_per_class=shots_per_class,
                    support_sample_with_replacement=bool(args.support_sample_with_replacement),
                    tta_n_views=int(args.tta_n_views),
                    tta_apply_to=str(args.tta_apply_to),
                    tta_amp_scale_std=float(args.tta_amp_scale_std),
                    tta_jitter_std=float(args.tta_jitter_std),
                    tta_shift_max_ratio=float(args.tta_shift_max_ratio),
                    tta_magwarp_std=float(args.tta_magwarp_std),
                    tta_crop_resize_ratio=float(args.tta_crop_resize_ratio),
                    tta_smooth_kernel=int(args.tta_smooth_kernel),
                )
                acc = float(np.mean(y_pred == y_te_m))

            print(f"{name}: {acc:.4f}")
            accs.append(acc)
        except Exception as e:
            msg = str(e)
            if "out of memory" in msg.lower():
                print(
                    f"{name}: failed: CUDA OOM. Try smaller --v2_batch_size / --query_batch_size / "
                    f"--mantis_batch_size_override, and enable --multi_gpu. Raw error: {e}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"{name}: failed: {e}")
        finally:
            if bool(args.empty_cache_each_dataset) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if accs:
        print(f"\nEvaluated {len(accs)} {args.suite.upper()} datasets | mean accuracy: {float(np.mean(accs)):.4f}")
    else:
        print("No datasets evaluated successfully.")


if __name__ == "__main__":
    main()


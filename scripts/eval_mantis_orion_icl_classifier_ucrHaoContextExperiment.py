from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


# Ensure we import the local workspace package (repo_root/src)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tabicl.model.mantis_adapter_icl import TokenMLPAdapter  # noqa: E402
from tabicl.model.mantis_tabicl import build_mantis_encoder  # noqa: E402
from tabicl.prior.data_reader import DataReader  # noqa: E402
from tabicl.sklearn.classifier import MantisICLClassifierV2  # noqa: E402
from tabicl.model.mantis_dev.adapters import VarianceBasedSelector  # noqa: E402

from orion_msp.model.orion_msp import OrionMSP  # noqa: E402


def _parse_csv_like_list(raw: str | list | tuple | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    # Accept both comma-separated and whitespace-separated formats.
    parts = [p.strip() for p in text.replace(" ", ",").split(",")]
    return [p for p in parts if p]


def _parse_int_multipliers(raw: str | list | tuple | None) -> list[int]:
    items = _parse_csv_like_list(raw)
    out: list[int] = []
    for s in items:
        try:
            v = int(float(s))
        except Exception as exc:
            raise ValueError(f"Invalid ctx multiplier '{s}': {exc}")
        if v <= 0:
            raise ValueError(f"ctx multipliers must be positive, got {v}")
        out.append(int(v))
    if not out:
        out = [1, 5, 10, 15]
    # De-dup while preserving order.
    seen: set[int] = set()
    uniq: list[int] = []
    for v in out:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def _print_effective_runtime_config(*, title: str, payload: dict) -> None:
    """Pretty-print a JSON payload for reproducibility/debugging."""
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        # Fallback to best-effort string representation.
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


def stratified_subset_indices(
    y: np.ndarray,
    frac: float,
    seed: int,
    min_per_class: int = 1,
) -> np.ndarray:
    """Return stratified subset indices sampled without replacement.

    Matches the spirit of TIC-FS `stratified_min1_split_from_test`:
    - First, take `min_per_class` per class (when possible)
    - Then distribute remaining quota proportionally by class counts, using
      fractional parts to allocate leftover slots.
    """
    y = np.asarray(y)
    n = int(len(y))
    if n <= 0:
        raise ValueError("Empty y")

    frac = float(frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0,1], got {frac}")

    min_per_class = int(min_per_class)
    if min_per_class < 0:
        raise ValueError(f"min_per_class must be >= 0, got {min_per_class}")

    classes, counts = np.unique(y, return_counts=True)
    num_classes = int(len(classes))
    if num_classes == 0:
        raise ValueError("No classes in y")

    desired = int(np.round(frac * n))
    desired = max(desired, min_per_class * num_classes)
    desired = min(desired, n)

    rng = np.random.default_rng(int(seed))

    # Step 1: mandatory min_per_class per class.
    chosen: list[int] = []
    remaining_indices: list[np.ndarray] = []
    remaining_labels: list[np.ndarray] = []

    for c in classes:
        idx_c = np.where(y == c)[0]
        if idx_c.size == 0:
            continue
        k0 = min(int(min_per_class), int(idx_c.size))
        if k0 > 0:
            picks = rng.choice(idx_c, size=k0, replace=False).astype(np.int64)
            chosen.extend([int(i) for i in picks.tolist()])
            rest_mask = np.ones(idx_c.shape[0], dtype=bool)
            # idx_c unique by construction.
            rest_mask[np.isin(idx_c, picks)] = False
            rest = idx_c[rest_mask]
        else:
            rest = idx_c
        if rest.size > 0:
            remaining_indices.append(rest.astype(np.int64))
            remaining_labels.append(np.full(rest.shape[0], c, dtype=y.dtype))

    chosen_np = np.asarray(chosen, dtype=np.int64)
    chosen_np = np.unique(chosen_np)

    additional_needed = int(desired - chosen_np.size)
    if additional_needed > 0 and remaining_indices:
        rem_idx = np.concatenate(remaining_indices).astype(np.int64)
        rem_y = np.concatenate(remaining_labels)

        if rem_idx.size > 0:
            rem_classes, rem_counts = np.unique(rem_y, return_counts=True)
            total_rem = int(rem_counts.sum())
            if total_rem > 0:
                expected = (additional_needed * (rem_counts / float(total_rem))).astype(float)
                take = np.floor(expected).astype(int)

                leftover = int(additional_needed - int(take.sum()))
                frac_parts = expected - np.floor(expected)
                order = np.argsort(-frac_parts)
                for j in order.tolist():
                    if leftover <= 0:
                        break
                    if take[j] < rem_counts[j]:
                        take[j] += 1
                        leftover -= 1

                if leftover > 0:
                    avail = rem_counts - take
                    while leftover > 0 and int(avail.sum()) > 0:
                        j = int(np.argmax(avail))
                        take[j] += 1
                        avail[j] -= 1
                        leftover -= 1

                extra: list[np.ndarray] = []
                for c, k in zip(rem_classes.tolist(), take.tolist()):
                    if int(k) <= 0:
                        continue
                    idx_c = rem_idx[rem_y == c]
                    k = min(int(k), int(idx_c.size))
                    if k <= 0:
                        continue
                    extra.append(rng.choice(idx_c, size=k, replace=False).astype(np.int64))

                if extra:
                    extra_np = np.concatenate(extra).astype(np.int64)
                    chosen_np = np.concatenate([chosen_np, extra_np]).astype(np.int64)
                    chosen_np = np.unique(chosen_np)

    # Final: shuffle for cleanliness (deterministic).
    rng.shuffle(chosen_np)
    # after shuffle
    if chosen_np.size < desired:
        # 可选：打印 warning 或直接 raise
        print(f"Warning: only selected {chosen_np.size} samples, less than desired {desired}.")

    return chosen_np.astype(np.int64)


def select_balanced_context_indices(y_pool: np.ndarray, N_ctx: int, seed: int) -> np.ndarray:
    """Class-balanced context sampling (strategy A).

    Ensures at least 1 sample per class when N_ctx < C (auto-bump to C).
    Returns indices into the pool, sampled without replacement.
    """
    y_pool = np.asarray(y_pool)
    n_pool = int(len(y_pool))
    if n_pool <= 0:
        return np.array([], dtype=np.int64)

    classes = np.unique(y_pool)
    C = int(classes.shape[0])
    if C <= 0:
        return np.array([], dtype=np.int64)

    N_ctx = int(N_ctx)
    if N_ctx < C:
        N_ctx = C
    target = min(int(N_ctx), n_pool)

    rng = np.random.default_rng(int(seed))

    n_c = target // C
    r = int(target - C * n_c)

    chosen: list[int] = []
    remaining_global: list[int] = []
    remaining_by_class: dict[int, list[int]] = {}

    for c in classes.tolist():
        idx_c = np.where(y_pool == c)[0].astype(np.int64)
        if idx_c.size == 0:
            remaining_by_class[int(c)] = []
            continue
        k = min(int(n_c), int(idx_c.size))
        if k > 0:
            picks = rng.choice(idx_c, size=k, replace=False)
            chosen.extend([int(i) for i in picks.tolist()])
            rest = idx_c[~np.isin(idx_c, picks)]
        else:
            rest = idx_c
        remaining_by_class[int(c)] = [int(i) for i in rest.tolist()]
        remaining_global.extend([int(i) for i in rest.tolist()])

    # Allocate remainder r: prefer distinct classes with remaining.
    need_more = int(target - len(chosen))
    if need_more > 0:
        candidates = [int(c) for c in classes.tolist() if len(remaining_by_class.get(int(c), [])) > 0]
        if candidates:
            take_cls = min(int(need_more), int(len(candidates)))
            picked_classes = rng.choice(np.array(candidates, dtype=np.int64), size=take_cls, replace=False)
            for c in picked_classes.tolist():
                pool_list = remaining_by_class.get(int(c), [])
                if not pool_list:
                    continue
                j = int(rng.integers(0, len(pool_list)))
                chosen.append(int(pool_list.pop(j)))

    # Final fill: sample from any remaining indices.
    chosen_np = np.asarray(chosen, dtype=np.int64)
    chosen_set = set(int(i) for i in chosen_np.tolist())
    remaining_np = np.array([i for i in remaining_global if i not in chosen_set], dtype=np.int64)

    still = int(target - chosen_np.size)
    if still > 0 and remaining_np.size > 0:
        extra = rng.choice(remaining_np, size=min(still, int(remaining_np.size)), replace=False).astype(np.int64)
        chosen_np = np.concatenate([chosen_np, extra]).astype(np.int64)

    # Safety: ensure correct length and uniqueness.
    chosen_np = np.unique(chosen_np)
    if chosen_np.size > target:
        chosen_np = rng.choice(chosen_np, size=target, replace=False).astype(np.int64)
    elif chosen_np.size < target:
        # Should be rare; fallback to global random fill.
        all_idx = np.arange(n_pool, dtype=np.int64)
        missing = target - int(chosen_np.size)
        candidates = all_idx[~np.isin(all_idx, chosen_np)]
        if candidates.size > 0:
            extra = rng.choice(candidates, size=min(missing, int(candidates.size)), replace=False).astype(np.int64)
            chosen_np = np.concatenate([chosen_np, extra]).astype(np.int64)
        chosen_np = np.unique(chosen_np)

    # Final guarantee: len == min(N_ctx, len(pool)).
    if chosen_np.size != target:
        # As a last resort, truncate.
        chosen_np = chosen_np[:target]
    return chosen_np.astype(np.int64)


def run_context_scaling_for_dataset(
    name: str,
    *,
    X_tr: np.ndarray,
    y_tr_raw: np.ndarray,
    X_te: np.ndarray,
    y_te_raw: np.ndarray,
    model: _MantisAdapterPlusOrionICL,
    query_frac: float,
    query_seed: int,
    ctx_repeats: int,
    ctx_seed_base: int,
    multipliers: list[int],
    ctx_k_factor: int,
    # classifier_v2 params (ensemble + RandomCropResize)
    v2_n_estimators: int,
    v2_class_shift: bool,
    v2_n_augmentations: int,
    v2_crop_rate_lo: float,
    v2_crop_rate_hi: float,
    device: torch.device,
    query_batch_size: int,
    softmax_temperature: float,
    results_out_dir: Path,
) -> dict:
    """Run Context Length Scaling sweep for one dataset."""
    X_tr_2d = _ensure_2d_timeseries(X_tr)
    X_te_2d = _ensure_2d_timeseries(X_te)
    if X_tr_2d.ndim != 2 or X_te_2d.ndim != 2:
        raise ValueError(f"context_scaling expects UCR univariate (N,L); got train {X_tr_2d.shape}, test {X_te_2d.shape}")

    # 1) Fixed query set Q from D^te only.
    q_idx = stratified_subset_indices(y_te_raw, frac=float(query_frac), seed=int(query_seed), min_per_class=1)
    all_te = np.arange(int(len(y_te_raw)), dtype=np.int64)
    mask = np.ones(all_te.shape[0], dtype=bool)
    mask[q_idx] = False
    remaining_idx = all_te[mask]

    # Explicit disjointness assertion.
    inter = np.intersect1d(q_idx, remaining_idx)
    assert inter.size == 0, "Q_idx must be disjoint from remaining_idx"

    X_q = X_te_2d[q_idx]
    y_q_raw = np.asarray(y_te_raw)[q_idx]

    # 2) Context pool P = D^tr ∪ (D^te \ Q).
    X_pool = np.concatenate([X_tr_2d, X_te_2d[remaining_idx]], axis=0)
    y_pool_raw = np.concatenate([np.asarray(y_tr_raw), np.asarray(y_te_raw)[remaining_idx]], axis=0)

    # 3) Remap labels based on pool labels (P defines class ids).
    y_pool_m, y_q_m, _ = _remap_labels(y_pool_raw, y_q_raw)
    C = int(np.unique(y_pool_m).shape[0])
    a = int(ctx_k_factor)
    if a <= 0:
        raise ValueError(f"ctx_k_factor must be a positive integer, got {a}")
    # Base context size
    K = int(a * C)
    if K <= 0:
        raise ValueError(f"Invalid base K computed as a*C: a={a}, C={C}, K={K}")

    print(
        f"\nDataset={name} | C={C} | a={a} | K=aC={K} | Q={int(X_q.shape[0])} | Pool={int(X_pool.shape[0])}"
    )

    # 4) Context budget sweep.
    multipliers = [int(m) for m in multipliers]
    N_ctx_list = [int(m * K) for m in multipliers]

    acc_runs: list[list[float]] = []
    mean_list: list[float] = []
    std_list: list[float] = []

    R = max(1, int(ctx_repeats))
    for m, N_ctx in zip(multipliers, N_ctx_list):
        run_accs: list[float] = []
        for r in range(R):
            seed = int(ctx_seed_base) + int(r)
            idx_ctx = select_balanced_context_indices(y_pool_m, N_ctx=int(N_ctx), seed=seed)
            X_support = X_pool[idx_ctx]
            y_support = y_pool_m[idx_ctx]

            crop_lo = float(v2_crop_rate_lo)
            crop_hi = float(v2_crop_rate_hi)
            if not (0.0 <= crop_lo <= crop_hi < 1.0):
                raise ValueError(
                    f"Invalid v2 crop range: lo={crop_lo}, hi={crop_hi}. Must satisfy 0 <= lo <= hi < 1."
                )

            clf = MantisICLClassifierV2(
                n_estimators=int(v2_n_estimators),
                class_shift=bool(v2_class_shift),
                crop_rate_range=(crop_lo, crop_hi),
                n_augmentations=int(v2_n_augmentations),
                softmax_temperature=float(softmax_temperature),
                device=device,
                # Make ensemble member ordering + crop windows reproducible per repeat.
                random_state=int(seed),
                verbose=False,
                model_path=None,
                allow_auto_download=False,
                checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
            )
            # Critical: avoid loading default checkpoint; use our adapter+orion model.
            clf.model_ = model
            clf.fit(X_support, y_support)
            y_pred = clf.predict(X_q)
            acc = float(np.mean(y_pred == y_q_m))
            run_accs.append(acc)

        acc_runs.append(run_accs)
        mean = float(np.mean(run_accs))
        std = float(np.std(run_accs, ddof=0))
        mean_list.append(mean)
        std_list.append(std)

        if int(m) == 1:
            tag = f"K(=aC={K})"
        else:
            tag = f"{int(m)}K"
        print(f"N_ctx={tag}: mean={mean:.4f}, std={std:.4f}")

    results_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_out_dir / f"results_context_scaling_{name}.json"
    payload = {
        "dataset_name": str(name),
        "C": int(C),
        "a": int(a),
        "K": int(K),
        "query_frac": float(query_frac),
        "query_seed": int(query_seed),
        "inference": {
            "mode": "classifier_v2",
            "n_estimators": int(v2_n_estimators),
            "class_shift": bool(v2_class_shift),
            "n_augmentations": int(v2_n_augmentations),
            "crop_rate_range": [float(v2_crop_rate_lo), float(v2_crop_rate_hi)],
            "softmax_temperature": float(softmax_temperature),
        },
        "multipliers": [int(m) for m in multipliers],
        "N_ctx_list": [int(x) for x in N_ctx_list],
        "Q_size": int(X_q.shape[0]),
        "pool_size": int(X_pool.shape[0]),
        "acc_runs": acc_runs,
        "mean": mean_list,
        "std": std_list,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")
    return payload


def _ensure_2d_timeseries(X: np.ndarray) -> np.ndarray:
    """Coerce X into (N, L). Supports UCR (N,L) and UEA (N,C,L)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X
    if X.ndim == 3:
        if X.shape[1] == 1:
            return X[:, 0, :]
        # NOTE: for UEA multivariate series, we do NOT average channels here.
        # Use `_uea_concat_and_sample()` to concat channels and sample to a fixed length.
        return X
    raise ValueError(f"Unexpected X shape: {X.shape}")


def _uea_concat_and_sample(
    X: np.ndarray,
    *,
    target_len: int,
    seed: int,
    mode: str = "center",
) -> np.ndarray:
    """Convert UEA (N,C,L) into (N,target_len) by channel-concat then crop/pad.

    Spec required by user:
    - For multichannel series, concatenate channels into a single long 1D series.
    - Then sample a length `target_len` window (default 512) and feed to Mantis.
    """

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

    # Ltot > target_len: crop
    if mode not in {"center", "random"}:
        raise ValueError(f"mode must be 'center' or 'random', got {mode}")
    if mode == "center":
        start = (Ltot - target_len) // 2
        return X_flat[:, start : start + target_len]

    rng = np.random.RandomState(int(seed))
    # Deterministic but per-sample different windows
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
    """Crop/pad UEA to (N,C,target_len) without mixing channels.

    Used by uea_fusion='sum_embed': each channel is fed into Mantis, then channel embeddings are summed.
    """
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
    """Apply VarianceBasedSelector on UEA multichannel time series.

    Fits selector on training split only, then transforms train/test.
    Expects input in (N, C, L) (or will be coerced by _ensure_3d_timeseries).
    """
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


def _find_latest_adapter_ckpt(dir_path: str) -> str:
    p = Path(dir_path)
    if not p.is_dir():
        raise FileNotFoundError(f"adapter ckpt dir not found: {dir_path}")

    candidates = list(p.glob("*_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No adapter ckpt files matching '*_epoch*.pt' under {dir_path}")

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _load_adapter_best_args_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise ValueError(f"Failed to read adapter_best_args.json: {path}: {exc}")

    if not isinstance(j, dict):
        return {}
    args = j.get("args")
    return args if isinstance(args, dict) else {}


def _extract_adapter_state_and_meta(ckpt_obj) -> tuple[dict, dict]:
    """Return (adapter_state_dict, ckpt_meta_dict).

    Supports:
    - our common format: {'adapter_state_dict': OrderedDict(...), ...}
    - raw adapter state dict: OrderedDict(...) or {name: tensor, ...}
    """
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("adapter_state_dict"), dict):
        return ckpt_obj["adapter_state_dict"], ckpt_obj

    # Fallback: treat as raw state dict.
    if isinstance(ckpt_obj, dict) and all(isinstance(k, str) for k in ckpt_obj.keys()):
        return ckpt_obj, {}

    raise ValueError("Invalid adapter checkpoint object: expected dict.")


def _infer_adapter_hidden_dim_from_state(adapter_state_dict: dict, *, mantis_dim: int) -> int | None:
    # TokenMLPAdapter: net[1] is Linear(mantis_dim -> hidden_dim)
    for k, v in adapter_state_dict.items():
        if not isinstance(k, str):
            continue
        if getattr(v, "ndim", None) != 2:
            continue
        if v.shape[1] == int(mantis_dim):
            return int(v.shape[0])
    return None


def _infer_adapter_use_layernorm_from_state(adapter_state_dict: dict) -> bool | None:
    has_ln = any(isinstance(k, str) and k.startswith("net.0.") for k in adapter_state_dict.keys())
    return True if has_ln else False


def _load_orion_checkpoint(path: str) -> tuple[OrionMSP, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "config" not in ckpt:
        raise ValueError("OrionMSP checkpoint must be a dict containing 'config'.")

    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        for k in ("model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
    if state_dict is None or not isinstance(state_dict, dict):
        raise ValueError("OrionMSP checkpoint must contain a model state dict ('state_dict' or similar).")

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model = OrionMSP(**ckpt["config"])
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model, ckpt["config"]


class _MantisAdapterPlusOrionICL(nn.Module):
    """Mantis encoder -> Adapter -> OrionMSP icl_predictor.

    Implements the forward signature expected by MantisICLClassifierV2.
    """

    def __init__(
        self,
        *,
        mantis_model: nn.Module,
        adapter: nn.Module,
        icl_predictor: nn.Module,
        mantis_seq_len: int = 512,
        mantis_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.mantis_model = mantis_model
        self.adapter = adapter
        self.icl_predictor = icl_predictor
        self.mantis_seq_len = int(mantis_seq_len)
        self.mantis_batch_size = int(mantis_batch_size)

        self.max_classes = int(getattr(self.icl_predictor, "max_classes", 10))

    def train(self, mode: bool = True):
        super().train(mode)
        # evaluation-only: keep everything in eval
        self.mantis_model.eval()
        self.adapter.eval()
        self.icl_predictor.eval()
        return self

    def _pad_or_truncate(self, X: torch.Tensor) -> torch.Tensor:
        """Pad/truncate last dimension to mantis_seq_len.

        Supports:
        - Univariate: (B, T, L)
        - Multivariate: (B, T, C, L)
        """
        target = self.mantis_seq_len
        if X.shape[-1] == target:
            return X
        if X.shape[-1] > target:
            return X[..., :target]
        pad = X.new_zeros((*X.shape[:-1], target - X.shape[-1]))
        return torch.cat([X, pad], dim=-1)

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        """Encode rows with Mantis.

        - If X is (B,T,L): encodes each row.
        - If X is (B,T,C,L): encodes each channel separately then sums embeddings over C.
        """
        X = self._pad_or_truncate(X)

        device = next(self.mantis_model.parameters()).device
        bs = max(1, int(self.mantis_batch_size))

        if X.dim() == 3:
            B, T, L = X.shape
            x_flat = X.reshape(B * T, 1, L).to(device)
            reps = []
            with torch.no_grad():
                for i in range(0, x_flat.shape[0], bs):
                    reps.append(self.mantis_model(x_flat[i : i + bs]))
            reps = torch.cat(reps, dim=0)
            return reps.reshape(B, T, -1)

        if X.dim() == 4:
            B, T, C, L = X.shape
            x_flat = X.reshape(B * T * C, 1, L).to(device)
            reps = []
            with torch.no_grad():
                for i in range(0, x_flat.shape[0], bs):
                    reps.append(self.mantis_model(x_flat[i : i + bs]))
            reps = torch.cat(reps, dim=0)  # (B*T*C, D)
            reps = reps.reshape(B, T, C, -1).sum(dim=2)  # (B,T,D)
            return reps

        raise ValueError(f"Unexpected X dim for mantis encode: {X.dim()} with shape {tuple(X.shape)}")

    @torch.no_grad()
    def encode_and_adapt(self, X: torch.Tensor) -> torch.Tensor:
        reps = self._encode(X)
        reps = reps.to(X.device)
        return self.adapter(reps)

    def forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        d: torch.Tensor | None = None,
        feature_shuffles=None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config=None,
    ) -> torch.Tensor:
        # d/feature_shuffles/embed_with_test are ignored in this path.
        # NOTE: classifier_v2 runs in eval/no_grad outside; keep this forward lightweight.
        reps = self._encode(X)
        reps = reps.to(X.device)
        reps = self.adapter(reps)

        # Orion ICL accepts mgr_config (MgrConfig). We keep it None by default.
        mgr_config = None
        if inference_config is not None and hasattr(inference_config, "ICL_CONFIG"):
            # `inference_config` here is tabicl.InferenceConfig. Its ICL_CONFIG type is
            # not guaranteed to match orion_msp.model.inference_config.MgrConfig, so we
            # intentionally do not pass it through.
            mgr_config = None

        return self.icl_predictor(
            reps,
            y_train=y_train,
            return_logits=bool(return_logits),
            softmax_temperature=float(softmax_temperature),
            mgr_config=mgr_config,
        )


@torch.no_grad()
def _predict_direct(
    model: _MantisAdapterPlusOrionICL,
    *,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    query_batch_size: int,
    softmax_temperature: float,
) -> np.ndarray:
    """Directly predict labels for X_query using (support + query) ICL tables."""

    device = next(model.adapter.parameters()).device
    X_support_t = torch.from_numpy(X_support.astype(np.float32)).to(device)
    X_query_t = torch.from_numpy(X_query.astype(np.float32)).to(device)
    y_support_t = torch.from_numpy(y_support.astype(np.float32)).to(device)

    # support: (S,L) -> (1,S,L), query: (N,L) -> (B,1,L)
    X_support_t = X_support_t.unsqueeze(0)
    y_support_t = y_support_t.unsqueeze(0)

    support_rep = model.encode_and_adapt(X_support_t)  # (1,S,D)

    preds: list[np.ndarray] = []
    bs = max(1, int(query_batch_size))
    for i in range(0, X_query_t.shape[0], bs):
        q = X_query_t[i : i + bs].unsqueeze(1)  # (B,1,L)
        query_rep = model.encode_and_adapt(q)  # (B,1,D)

        B = query_rep.shape[0]
        reps_all = torch.cat([support_rep.repeat(B, 1, 1), query_rep], dim=1)  # (B,S+1,D)
        y_train = y_support_t.repeat(B, 1)  # (B,S)

        # Orion ICL forward returns logits/probs for test region only in eval mode.
        logits = model.icl_predictor(
            reps_all,
            y_train=y_train,
            return_logits=True,
            softmax_temperature=float(softmax_temperature),
        )
        # logits: (B, 1, num_classes)
        pred = torch.argmax(logits[:, 0, :], dim=-1).detach().cpu().numpy()
        preds.append(pred)

    return np.concatenate(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained adapter (mantis->adapter->OrionMSP icl_predictor) on UCR. "
            "Adapter checkpoint is produced by src/tabicl/train/train_mantis_orion_icl_adapter_only_from_ckpts.py."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="classifier_v2",
        choices=["direct", "classifier_v2"],
        help=(
            "Evaluation mode: 'direct' uses cached support reps; "
            "'classifier_v2' uses MantisICLClassifierV2 (class shift + RandomCropResize augmentation)."
        ),
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="context_scaling",
        choices=["standard", "context_scaling"],
        help="Experiment mode: standard evaluation or context length scaling.",
    )
    parser.add_argument(
        "--query_frac",
        type=float,
        default=0.10,
        help="(context_scaling) Fraction of UCR test used as fixed query set Q.",
    )
    parser.add_argument(
        "--query_seed",
        type=int,
        default=0,
        help="(context_scaling) Seed for sampling fixed query set Q (shared across all contexts/repeats).",
    )
    parser.add_argument(
        "--ctx_repeats",
        type=int,
        default=5,
        help="(context_scaling) Number of repeated context samplings per N_ctx.",
    )
    parser.add_argument(
        "--ctx_seed_base",
        type=int,
        default=1000,
        help="(context_scaling) Seed base for context sampling. Repeat r uses seed=ctx_seed_base+r.",
    )
    parser.add_argument(
        "--ctx_multipliers",
        type=str,
        default="1,5,10,15,20",
        help="(context_scaling) Multipliers m for N_ctx=m*K where K=C. Example: '1,5,10,15'.",
    )
    parser.add_argument(
        "--ctx_k_factor",
        type=int,
        default=10,
        help="(context_scaling) Set base context size K=a*C (a is this factor). Example: --ctx_k_factor 100.",
    )
    parser.add_argument(
        "--datasets_context_scaling",
        type=str,
        default="Crop,ElectricDevices",
        help="(context_scaling) Default datasets when --dataset is not provided.",
    )

    parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default="checkpoints/best_adapter.pt",
        help="Path to adapter checkpoint (.pt) saved by the training script.",
    )
    parser.add_argument(
        "--adapter_ckpt_dir",
        type=str,
        default="/home/hzf00006536/fjt/tabicl-main/tabicl-main/src/tabicl/checkpoints/mantis_orion_icl_adapter_only",
        help="Directory to auto-pick the latest '*_epoch*.pt' if --adapter_ckpt is not provided.",
    )

    parser.add_argument(
        "--adapter_best_args_json",
        type=str,
        default=None,
        help=(
            "Optional path to adapter_best_args.json (produced by tuning script). "
            "If omitted, will auto-detect a file named 'adapter_best_args.json' in the same directory as --adapter_ckpt."
        ),
    )
    parser.add_argument(
        "--mantis_ckpt",
        type=str,
        default="checkpoints/CaukerImpro-data100k_emb512_100epochs.pt",
        help="Mantis checkpoint (only used if not present in adapter ckpt).",
    )
    parser.add_argument(
        "--orion_ckpt",
        type=str,
        default="/home/hzf00006536/fjt/Orion-MSP-v1.0.ckpt",
        help="OrionMSP checkpoint (only used if not present in adapter ckpt).",
    )

    parser.add_argument("--ucr_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    parser.add_argument(
        "--uea_path",
        type=str,
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",
        help="Only for DataReader initialization; UCR evaluation does not require UEA datasets.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--suite",
        type=str,
        default="ucr",
        choices=["ucr", "uea"],
        help="Dataset suite to evaluate: UCR (univariate) or UEA (multivariate).",
    )

    parser.add_argument("--dataset", type=str, default=None, help="Evaluate a single UCR dataset name")

    parser.add_argument("--n_estimators", type=int, default=32)

    v2_class_shift_group = parser.add_mutually_exclusive_group()
    v2_class_shift_group.add_argument(
        "--v2_class_shift",
        dest="v2_class_shift",
        action="store_true",
        default=True,
        help="(classifier_v2) Enable class shift ensembling. Default: True.",
    )
    v2_class_shift_group.add_argument(
        "--no_v2_class_shift",
        dest="v2_class_shift",
        action="store_false",
        help="(classifier_v2) Disable class shift ensembling.",
    )
    parser.add_argument(
        "--v2_n_augmentations",
        type=int,
        default=1,
        help="(classifier_v2) Number of RandomCropResize views per estimator.",
    )
    parser.add_argument(
        "--v2_crop_rate_lo",
        type=float,
        default=0.0,
        help="(classifier_v2) RandomCropResize crop_rate lower bound.",
    )
    parser.add_argument(
        "--v2_crop_rate_hi",
        type=float,
        default=0.00,
        help="(classifier_v2) RandomCropResize crop_rate upper bound.",
    )

    parser.add_argument(
        "--support_size",
        type=int,
        default=128,
        help="Number of training samples used as ICL support (auto-bumped to >= #classes).",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=64,
        help="Query batch size (number of test samples per forward).",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=float,
        default=0.9,
        help="Softmax temperature passed to icl_predictor inference.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for support sampling.")

    parser.add_argument(
        "--uea_mode",
        type=str,
        default="center",
        choices=["center", "random"],
        help="(UEA) After channel-concat, how to sample length=mantis_seq_len window.",
    )

    parser.add_argument(
        "--uea_use_var_selector",
        action="store_true",
        help="(UEA) Use VarianceBasedSelector to reduce channels before inference.",
    )
    parser.add_argument(
        "--uea_var_num_channels",
        type=int,
        default=None,
        help="(UEA) Target number of channels after VarianceBasedSelector.",
    )
    parser.add_argument(
        "--uea_fusion",
        type=str,
        default="concat",
        choices=["concat", "sum_embed"],
        help=(
            "(UEA) How to handle multichannel series: "
            "'concat' concatenates channels in time then samples length=mantis_seq_len; "
            "'sum_embed' encodes each channel with Mantis then sums channel embeddings."
        ),
    )

    parser.add_argument("--mantis_hidden_dim", type=int, default=512)
    parser.add_argument("--mantis_seq_len", type=int, default=512)
    parser.add_argument("--mantis_batch_size", type=int, default=512)

    args = parser.parse_args()

    device = torch.device(args.device)

    adapter_ckpt_path = args.adapter_ckpt or _find_latest_adapter_ckpt(args.adapter_ckpt_dir)
    ckpt_obj = torch.load(adapter_ckpt_path, map_location="cpu")
    print("------------------------------------Adapter ckpt loaded from {}".format(adapter_ckpt_path))
    print(adapter_ckpt_path)

    adapter_state_dict, ckpt_meta = _extract_adapter_state_and_meta(ckpt_obj)

    # Optional: load adapter_best_args.json to get the exact tuned adapter config (and ckpt paths).
    best_args_path: str | None
    if args.adapter_best_args_json is not None:
        best_args_path = str(args.adapter_best_args_json)
    else:
        candidate = Path(adapter_ckpt_path).resolve().parent / "adapter_best_args.json"
        best_args_path = str(candidate) if candidate.is_file() else None

    best_args = _load_adapter_best_args_json(best_args_path) if best_args_path is not None else {}
    if best_args_path is not None:
        print(f"[Eval] adapter_best_args_json: {best_args_path} (loaded {len(best_args)} keys)")

    train_args = ckpt_meta.get("args") if isinstance(ckpt_meta.get("args"), dict) else {}

    # Resolve ckpt paths.
    mantis_ckpt = str(best_args.get("mantis_ckpt") or ckpt_meta.get("mantis_ckpt") or args.mantis_ckpt)
    orion_ckpt = str(best_args.get("orion_ckpt") or ckpt_meta.get("orion_ckpt") or args.orion_ckpt)
    if not orion_ckpt:
        raise ValueError("Missing Orion checkpoint. Provide --orion_ckpt or include it in adapter ckpt / adapter_best_args.json.")

    # Load OrionMSP and take icl_predictor only
    orion_model, orion_cfg = _load_orion_checkpoint(orion_ckpt)
    icl_predictor = orion_model.icl_predictor
    for p in icl_predictor.parameters():
        p.requires_grad_(False)
    icl_predictor.to(device)
    icl_predictor.eval()

    embed_dim = int(orion_cfg.get("embed_dim", getattr(orion_model, "embed_dim", 128)))
    row_num_cls = int(orion_cfg.get("row_num_cls", 4))
    icl_dim = int(ckpt_meta.get("icl_dim", embed_dim * row_num_cls))

    # Load mantis encoder
    mantis_model = build_mantis_encoder(
        mantis_checkpoint=Path(mantis_ckpt),
        device=device,
        hidden_dim=int(args.mantis_hidden_dim),
        seq_len=int(args.mantis_seq_len),
    )
    for p in mantis_model.parameters():
        p.requires_grad_(False)
    mantis_model.eval()

    mantis_dim = int(ckpt_meta.get("mantis_dim", getattr(mantis_model, "hidden_dim", int(args.mantis_hidden_dim))))

    # Resolve adapter hyperparams: prefer adapter_best_args.json > ckpt args > defaults.
    ckpt_params = ckpt_meta.get("params") if isinstance(ckpt_meta.get("params"), dict) else {}

    adapter_hidden_dim_val = best_args.get("adapter_hidden_dim")
    if adapter_hidden_dim_val is None:
        adapter_hidden_dim_val = train_args.get("adapter_hidden_dim")
    if adapter_hidden_dim_val is None:
        adapter_hidden_dim_val = ckpt_params.get("adapter_hidden_dim")
    if adapter_hidden_dim_val is None:
        adapter_hidden_dim_val = _infer_adapter_hidden_dim_from_state(adapter_state_dict, mantis_dim=mantis_dim)

    adapter_dropout_val = best_args.get("adapter_dropout")
    if adapter_dropout_val is None:
        adapter_dropout_val = train_args.get("adapter_dropout")
    if adapter_dropout_val is None:
        adapter_dropout_val = ckpt_params.get("adapter_dropout")
    if adapter_dropout_val is None:
        adapter_dropout_val = 0.0

    adapter_no_ln_val = best_args.get("adapter_no_layernorm")
    if adapter_no_ln_val is None:
        adapter_no_ln_val = train_args.get("adapter_no_layernorm")
    if adapter_no_ln_val is None:
        adapter_no_ln_val = ckpt_params.get("adapter_no_layernorm")
    if adapter_no_ln_val is None:
        inferred_ln = _infer_adapter_use_layernorm_from_state(adapter_state_dict)
        adapter_no_ln_val = (not inferred_ln) if inferred_ln is not None else False

    adapter = TokenMLPAdapter(
        mantis_dim=int(mantis_dim),
        icl_dim=int(icl_dim),
        hidden_dim=(None if adapter_hidden_dim_val is None else int(adapter_hidden_dim_val)),
        # hidden_dim= 256,
        dropout=float(adapter_dropout_val),
        use_layernorm=not bool(adapter_no_ln_val),
    )
    adapter.load_state_dict(adapter_state_dict, strict=True)
    adapter.to(device)
    adapter.eval()

    custom_model = _MantisAdapterPlusOrionICL(
        mantis_model=mantis_model,
        adapter=adapter,
        icl_predictor=icl_predictor,
        mantis_seq_len=int(args.mantis_seq_len),
        mantis_batch_size=int(args.mantis_batch_size),
    )
    for p in custom_model.parameters():
        p.requires_grad_(False)
    custom_model.eval()

    reader = DataReader(
        UEA_data_path=str(args.uea_path),
        UCR_data_path=str(args.ucr_path),
        transform_ts_size=int(args.mantis_seq_len),
    )

    if args.experiment == "context_scaling":
        if args.suite != "ucr":
            raise ValueError("context_scaling experiment is only defined for UCR (univariate) datasets.")
        # Force classifier_v2 mode as requested by the experiment spec.
        if args.mode != "classifier_v2":
            print(f"[context_scaling] Forcing --mode classifier_v2 (was {args.mode})")
            args.mode = "classifier_v2"

        if args.dataset is not None:
            dataset_names = [args.dataset]
        else:
            dataset_names = _parse_csv_like_list(args.datasets_context_scaling)
            if not dataset_names:
                dataset_names = ["Crop", "ElectricDevices"]
    else:
        if args.dataset is not None:
            dataset_names = [args.dataset]
        else:
            if args.suite == "uea":
                dataset_names = list(reader.dataset_list_uea)
            else:
                dataset_names = list(reader.dataset_list_ucr)

    print(f"[Eval] adapter_ckpt: {adapter_ckpt_path}")
    print(f"[Eval] mantis_ckpt: {mantis_ckpt}")
    print(f"[Eval] orion_ckpt: {orion_ckpt}")
    print(f"[Eval] suite: {args.suite}")
    print(f"[Eval] dataset_count: {len(dataset_names)}")
    print(f"[Eval] mode: {args.mode}")

    # Print the effective runtime configuration (actual values used after resolving
    # CLI defaults + ckpt meta + adapter_best_args.json + inferred params).
    _print_effective_runtime_config(
        title="[Eval][EffectiveRuntimeConfig]",
        payload={
            "cli_args": vars(args),
            "resolved": {
                "adapter_ckpt_path": str(adapter_ckpt_path),
                "adapter_best_args_json": (None if best_args_path is None else str(best_args_path)),
                "mantis_ckpt": str(mantis_ckpt),
                "orion_ckpt": str(orion_ckpt),
                "mantis_dim": int(mantis_dim),
                "icl_dim": int(icl_dim),
                "adapter_hidden_dim": (None if adapter_hidden_dim_val is None else int(adapter_hidden_dim_val)),
                "adapter_dropout": float(adapter_dropout_val),
                "adapter_use_layernorm": bool(not bool(adapter_no_ln_val)),
                "dataset_names_count": int(len(dataset_names)),
                "dataset_single": (None if args.dataset is None else str(args.dataset)),
            },
            "sources": {
                "ckpt_meta_keys": (sorted(list(ckpt_meta.keys())) if isinstance(ckpt_meta, dict) else None),
                "train_args_keys": (sorted(list(train_args.keys())) if isinstance(train_args, dict) else None),
                "ckpt_params_keys": (sorted(list(ckpt_params.keys())) if isinstance(ckpt_params, dict) else None),
                "best_args_keys": (sorted(list(best_args.keys())) if isinstance(best_args, dict) else None),
            },
        },
    )


    # ============================
    # Dispatch by experiment
    # ============================
    if args.experiment == "context_scaling":
        multipliers = _parse_int_multipliers(args.ctx_multipliers)
        out_dir = _REPO_ROOT / "evaluation_results"

        for name in dataset_names:
            try:
                X_tr, y_tr = reader.read_dataset(name, which_set="train")
                X_te, y_te = reader.read_dataset(name, which_set="test")
                run_context_scaling_for_dataset(
                    name,
                    X_tr=X_tr,
                    y_tr_raw=y_tr,
                    X_te=X_te,
                    y_te_raw=y_te,
                    model=custom_model,
                    query_frac=float(args.query_frac),
                    query_seed=int(args.query_seed),
                    ctx_repeats=int(args.ctx_repeats),
                    ctx_seed_base=int(args.ctx_seed_base),
                    multipliers=multipliers,
                    ctx_k_factor=int(args.ctx_k_factor),
                    v2_n_estimators=int(args.n_estimators),
                    v2_class_shift=bool(args.v2_class_shift),
                    v2_n_augmentations=int(args.v2_n_augmentations),
                    v2_crop_rate_lo=float(args.v2_crop_rate_lo),
                    v2_crop_rate_hi=float(args.v2_crop_rate_hi),
                    device=device,
                    query_batch_size=int(args.query_batch_size),
                    softmax_temperature=float(args.softmax_temperature),
                    results_out_dir=out_dir,
                )
            except Exception as e:
                print(f"{name}: failed: {e}")
        return

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
                    # Keep multichannel and crop/pad per channel to mantis_seq_len.
                    X_tr_2d = _uea_crop_pad_per_channel(
                        X_tr,
                        target_len=int(args.mantis_seq_len),
                        seed=int(args.seed),
                        mode=str(args.uea_mode),
                    )
                    X_te_2d = _uea_crop_pad_per_channel(
                        X_te,
                        target_len=int(args.mantis_seq_len),
                        seed=int(args.seed) + 1,
                        mode=str(args.uea_mode),
                    )
                else:
                    # Backward-compatible: concatenate channels then sample/crop.
                    X_tr_2d = _uea_concat_and_sample(
                        X_tr,
                        target_len=int(args.mantis_seq_len),
                        seed=int(args.seed),
                        mode=str(args.uea_mode),
                    )
                    X_te_2d = _uea_concat_and_sample(
                        X_te,
                        target_len=int(args.mantis_seq_len),
                        seed=int(args.seed) + 1,
                        mode=str(args.uea_mode),
                    )
            else:
                X_tr_2d = _ensure_2d_timeseries(X_tr)
                X_te_2d = _ensure_2d_timeseries(X_te)

                # UCR should already be (N,L); if UEA slips in here, fail loudly.
                if X_tr_2d.ndim != 2 or X_te_2d.ndim != 2:
                    raise ValueError(
                        f"Expected univariate (N,L) arrays for UCR; got train {X_tr_2d.shape}, test {X_te_2d.shape}. "
                        f"Did you mean --suite uea ?"
                    )

            y_tr_m, y_te_m, _classes = _remap_labels(y_tr, y_te)

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
                    device=device,
                    random_state=int(args.seed),
                    verbose=False,
                    model_path=None,
                    allow_auto_download=False,
                    checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
                )
                clf.model_ = custom_model
                clf.fit(X_tr_2d, y_tr_m)
                y_pred = clf.predict(X_te_2d)
                acc = float(np.mean(y_pred == y_te_m))
            else:
                sup_idx = _select_support_indices(y_tr_m, support_size=int(args.support_size), seed=int(args.seed))
                X_sup = X_tr_2d[sup_idx]
                y_sup = y_tr_m[sup_idx]

                y_pred = _predict_direct(
                    custom_model,
                    X_support=X_sup,
                    y_support=y_sup,
                    X_query=X_te_2d,
                    query_batch_size=int(args.query_batch_size),
                    softmax_temperature=float(args.softmax_temperature),
                )
                acc = float(np.mean(y_pred == y_te_m))

            print(f"{name}: {acc:.4f}")
            accs.append(acc)
        except Exception as e:
            print(f"{name}: failed: {e}")

    if accs:
        print(f"\nEvaluated {len(accs)} {args.suite.upper()} datasets | mean accuracy: {float(np.mean(accs)):.4f}")
    else:
        print("No datasets evaluated successfully.")


if __name__ == "__main__":
    main()

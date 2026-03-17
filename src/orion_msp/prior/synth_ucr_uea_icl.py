from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info


def _normalize_probs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= 0.0:
        return np.ones_like(x) / float(len(x))
    return x / s


def _weighted_choice(rng: np.random.Generator, options: list[str], probs: list[float]) -> str:
    p = _normalize_probs(np.asarray(probs, dtype=np.float64))
    idx = int(rng.choice(len(options), p=p))
    return str(options[idx])


def _sample_task_type(rng: np.random.Generator, probs: tuple[float, float, float]) -> str:
    return _weighted_choice(rng, ["ucr_like", "uea_like", "hybrid"], [float(v) for v in probs])


def _sample_difficulty(rng: np.random.Generator, probs: tuple[float, float, float]) -> str:
    return _weighted_choice(rng, ["easy", "medium", "hard"], [float(v) for v in probs])


def _difficulty_scale(difficulty: str) -> float:
    if difficulty == "easy":
        return 0.6
    if difficulty == "medium":
        return 1.0
    return 1.4


def _sample_rule_family(rng: np.random.Generator, task_type: str) -> str:
    if task_type == "ucr_like":
        return _weighted_choice(
            rng,
            ["motif_shape", "motif_polarity", "motif_position", "motif_order", "local_anomaly"],
            [0.28, 0.18, 0.24, 0.18, 0.12],
        )
    if task_type == "uea_like":
        return _weighted_choice(
            rng,
            ["lead_lag", "channel_subset", "channel_motif", "corr_regime", "phase_cross_channel"],
            [0.25, 0.20, 0.20, 0.20, 0.15],
        )
    return _weighted_choice(
        rng,
        ["motif_shape", "motif_position", "lead_lag", "channel_subset"],
        [0.30, 0.20, 0.30, 0.20],
    )


def _sample_class_prior(rng: np.random.Generator, k: int, imbalance: float) -> np.ndarray:
    alpha = np.full((k,), max(0.15, float(imbalance)), dtype=np.float64)
    return rng.dirichlet(alpha)


def _alloc_counts_with_min_one(
    rng: np.random.Generator,
    k: int,
    total: int,
    probs: np.ndarray,
    min_one: bool,
) -> np.ndarray:
    total = int(total)
    if total <= 0:
        return np.zeros((k,), dtype=np.int64)
    if min_one:
        if total < k:
            total = k
        out = np.ones((k,), dtype=np.int64)
        extra = total - k
        if extra > 0:
            out += rng.multinomial(extra, probs).astype(np.int64)
        return out
    return rng.multinomial(total, probs).astype(np.int64)


def _trend_signal(rng: np.random.Generator, l: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(l), dtype=np.float32)
    kind = _weighted_choice(rng, ["zero", "linear", "exp", "step"], [0.20, 0.40, 0.15, 0.25])
    if kind == "zero":
        return np.zeros((l,), dtype=np.float32)
    if kind == "linear":
        a = float(rng.uniform(-0.8, 0.8))
        b = float(rng.uniform(-0.5, 0.5))
        return (a * t + b).astype(np.float32)
    if kind == "exp":
        a = float(rng.uniform(0.2, 1.0))
        b = float(rng.uniform(-1.5, 1.5))
        return (a * np.exp(b * t)).astype(np.float32)
    out = np.zeros((l,), dtype=np.float32)
    pivot = int(rng.integers(max(1, l // 4), max(2, 3 * l // 4)))
    jump = float(rng.uniform(-0.8, 0.8))
    out[pivot:] = jump
    return out


def _periodic_component(rng: np.random.Generator, l: int, terms: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(l), dtype=np.float32)
    out = np.zeros((l,), dtype=np.float32)
    for _ in range(int(terms)):
        freq = float(rng.uniform(1.0, 12.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.uniform(0.05, 0.35))
        out += (amp * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)
    return out


def _ar_component(rng: np.random.Generator, l: int, phi: float, sigma: float) -> np.ndarray:
    out = np.zeros((int(l),), dtype=np.float32)
    eps = rng.normal(0.0, float(sigma), size=int(l)).astype(np.float32)
    for i in range(1, int(l)):
        out[i] = float(phi) * out[i - 1] + eps[i]
    return out


def _colored_noise(rng: np.random.Generator, l: int, sigma: float) -> np.ndarray:
    wn = rng.normal(0.0, float(sigma), size=int(l)).astype(np.float32)
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    return np.convolve(wn, kernel, mode="same").astype(np.float32)


def _build_shared_world(
    rng: np.random.Generator,
    *,
    task_type: str,
    channels: int,
    time_length: int,
) -> tuple[np.ndarray, dict]:
    c = int(channels)
    l = int(time_length)
    family = _weighted_choice(rng, ["gp_like", "ar_like", "piecewise", "periodic", "colored_noise"], [0.25, 0.22, 0.16, 0.22, 0.15])

    if task_type in {"uea_like", "hybrid"} and c > 1:
        latent_dim = int(min(4, max(2, c // 2)))
    else:
        latent_dim = 1

    base_latents = []
    for _ in range(latent_dim):
        if family == "gp_like":
            comp = _periodic_component(rng, l, terms=int(rng.integers(2, 5))) + _colored_noise(rng, l, 0.05)
        elif family == "ar_like":
            comp = _ar_component(rng, l, phi=float(rng.uniform(0.55, 0.95)), sigma=float(rng.uniform(0.02, 0.12)))
        elif family == "piecewise":
            comp = _trend_signal(rng, l) + _ar_component(rng, l, phi=0.35, sigma=0.05)
        elif family == "periodic":
            comp = _periodic_component(rng, l, terms=int(rng.integers(3, 6)))
        else:
            comp = _colored_noise(rng, l, sigma=float(rng.uniform(0.05, 0.2)))
        comp = comp + _trend_signal(rng, l)
        base_latents.append(comp.astype(np.float32))

    z = np.stack(base_latents, axis=0)  # (Z, L)
    mix = rng.normal(0.0, 1.0, size=(c, z.shape[0])).astype(np.float32)
    mix /= (np.linalg.norm(mix, axis=1, keepdims=True) + 1e-6)
    world = (mix @ z).astype(np.float32)  # (C, L)

    meta = {
        "background_family": family,
        "latent_dim": int(z.shape[0]),
    }
    return world, meta


def _motif_wave(rng: np.random.Generator, motif_type: str, length: int, amp: float, polarity: float) -> np.ndarray:
    m = int(max(4, length))
    t = np.linspace(0.0, 1.0, m, dtype=np.float32)
    if motif_type == "spike":
        x = np.exp(-((t - 0.5) ** 2) / 0.01)
    elif motif_type == "bump":
        x = np.exp(-((t - 0.5) ** 2) / 0.04)
    elif motif_type == "chirp":
        x = np.sin(2.0 * np.pi * (2.0 + 8.0 * t) * t)
    elif motif_type == "square":
        x = np.sign(np.sin(2.0 * np.pi * 4.0 * t))
    elif motif_type == "sawtooth":
        x = 2.0 * (t - np.floor(t + 0.5))
    else:
        x = np.exp(-3.0 * t) * np.sin(2.0 * np.pi * 6.0 * t)
    x = x.astype(np.float32)
    return (float(amp) * float(polarity) * x).astype(np.float32)


def _insert_motif(signal: np.ndarray, motif: np.ndarray, pos: int, repeat: int, gap: int) -> np.ndarray:
    out = np.array(signal, copy=True)
    l = int(out.shape[-1])
    m = int(motif.shape[-1])
    for r in range(int(max(1, repeat))):
        st = int(pos + r * (m + max(0, gap)))
        ed = int(min(l, st + m))
        if st >= l:
            break
        out[st:ed] += motif[: ed - st]
    return out


def _piecewise_monotone_warp(rng: np.random.Generator, l: int, strength: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(l), dtype=np.float32)
    k = int(rng.integers(3, 7))
    knots = np.linspace(0.0, 1.0, k, dtype=np.float32)
    noise = rng.normal(0.0, float(0.12 * strength), size=k).astype(np.float32)
    target = np.clip(knots + noise, 0.0, 1.0)
    target[0] = 0.0
    target[-1] = 1.0
    target = np.maximum.accumulate(target)
    target = target / max(1e-6, float(target[-1]))
    return np.interp(t, knots, target).astype(np.float32)


def _apply_time_warp(rng: np.random.Generator, x: np.ndarray, strength: float) -> np.ndarray:
    c, l = x.shape
    grid = _piecewise_monotone_warp(rng, l, strength)
    src = np.linspace(0.0, 1.0, l, dtype=np.float32)
    out = np.empty_like(x)
    for ci in range(c):
        out[ci] = np.interp(src, grid, x[ci]).astype(np.float32)
    return out


def _phase_shift(x: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(x, int(shift), axis=-1)


def _apply_length_artifact(rng: np.random.Generator, x: np.ndarray, strength: float) -> np.ndarray:
    c, l = x.shape
    min_keep = int(max(8, int(l * max(0.35, 1.0 - 0.45 * strength))))
    eff = int(rng.integers(min_keep, l + 1))
    start = int(rng.integers(0, max(1, l - eff + 1)))
    core = x[:, start : start + eff]

    out = np.zeros((c, l), dtype=np.float32)
    mode = _weighted_choice(rng, ["left", "right", "both"], [0.3, 0.3, 0.4])
    if mode == "left":
        out[:, :eff] = core
    elif mode == "right":
        out[:, l - eff :] = core
    else:
        pad_left = int(rng.integers(0, max(1, l - eff + 1)))
        out[:, pad_left : pad_left + eff] = core

    if rng.random() < (0.08 + 0.22 * strength):
        seg = int(rng.integers(max(4, l // 20), max(5, l // 8)))
        st = int(rng.integers(0, max(1, l - seg + 1)))
        out[:, st : st + seg] = 0.0
    return out


def _apply_sensor_noise(rng: np.random.Generator, x: np.ndarray, strength: float, class_dependent_gain: float) -> np.ndarray:
    out = np.array(x, copy=True)
    c, l = out.shape

    sigma = float((0.015 + 0.06 * strength) * (1.0 + class_dependent_gain))
    out += rng.normal(0.0, sigma, size=(c, l)).astype(np.float32)

    if rng.random() < (0.25 + 0.20 * strength):
        out += np.stack([_colored_noise(rng, l, sigma * 0.7) for _ in range(c)], axis=0)

    if rng.random() < (0.10 + 0.20 * strength):
        burst_len = int(rng.integers(max(3, l // 40), max(4, l // 15)))
        st = int(rng.integers(0, max(1, l - burst_len + 1)))
        out[:, st : st + burst_len] += rng.normal(0.0, sigma * 5.0, size=(c, burst_len)).astype(np.float32)

    if rng.random() < (0.10 + 0.15 * strength):
        q = float(rng.choice([0.01, 0.02, 0.05]))
        out = (np.round(out / q) * q).astype(np.float32)

    if rng.random() < (0.10 + 0.15 * strength):
        clip = float(np.quantile(np.abs(out), 0.95))
        out = np.clip(out, -clip, clip)

    return out.astype(np.float32)


def _make_class_prototype(
    rng: np.random.Generator,
    world: np.ndarray,
    *,
    class_idx: int,
    k: int,
    task_type: str,
    rule_family: str,
    difficulty: str,
) -> tuple[np.ndarray, dict]:
    c, l = world.shape
    proto = np.array(world, copy=True)
    diff = _difficulty_scale(difficulty)

    info = {
        "motif_type": None,
        "informative_channels": None,
        "lag": None,
    }

    if task_type == "ucr_like" and c > 1:
        proto[1:] *= 0.65

    if task_type in {"uea_like", "hybrid"}:
        n_info = int(max(1, min(c, rng.integers(1, max(2, c // 2 + 1)))))
        info_ch = rng.choice(np.arange(c), size=n_info, replace=False).astype(int)
    else:
        info_ch = np.array([0], dtype=int)
    info["informative_channels"] = info_ch.tolist()

    motif_type = _weighted_choice(rng, ["spike", "bump", "chirp", "square", "sawtooth", "damped"], [0.18, 0.22, 0.16, 0.16, 0.14, 0.14])
    info["motif_type"] = motif_type

    motif_len = int(rng.integers(max(8, l // 16), max(12, l // 5)))
    amp = float(rng.uniform(0.25, 0.75) / diff)
    if difficulty == "hard":
        amp *= 0.75
    polarity = 1.0 if (class_idx % 2 == 0) else -1.0
    motif = _motif_wave(rng, motif_type, motif_len, amp=amp, polarity=polarity)

    pos_anchor = int((class_idx + 1) * l / (k + 1))
    pos_jitter = int(rng.integers(-max(2, l // 20), max(3, l // 20)))
    pos = int(np.clip(pos_anchor + pos_jitter, 0, max(0, l - motif_len)))

    repeat = 1
    gap = int(rng.integers(max(2, motif_len // 6), max(3, motif_len // 2)))
    if rule_family in {"motif_order", "motif_position"}:
        repeat = int(rng.integers(1, 3))

    for ci in info_ch.tolist():
        proto[ci] = _insert_motif(proto[ci], motif, pos=pos, repeat=repeat, gap=gap)

    if rule_family in {"lead_lag", "phase_cross_channel"} and c > 1:
        lag = int(rng.integers(-max(2, l // 14), max(3, l // 14)))
        info["lag"] = lag
        for ci in info_ch.tolist()[1:]:
            proto[ci] = _phase_shift(proto[ci][None, :], shift=lag)[0]

    # Small class-local delta to keep inter-class margin narrow.
    proto += rng.normal(0.0, 0.015 * diff, size=proto.shape).astype(np.float32)
    return proto.astype(np.float32), info


def _margin_ratio(protos: np.ndarray, class_samples: list[np.ndarray]) -> float:
    k = int(protos.shape[0])
    inter = []
    for i in range(k):
        for j in range(i + 1, k):
            inter.append(float(np.mean((protos[i] - protos[j]) ** 2) ** 0.5))
    inter_m = float(np.mean(inter)) if inter else 0.0

    intra = []
    for arr in class_samples:
        if arr.shape[0] <= 1:
            continue
        std = np.std(arr, axis=0)
        intra.append(float(np.mean(std)))
    intra_m = float(np.mean(intra)) if intra else 1e-6
    return inter_m / max(1e-6, intra_m)


def _episode_quality_ok(protos: np.ndarray, class_samples: list[np.ndarray], difficulty: str) -> bool:
    ratio = _margin_ratio(protos, class_samples)
    if difficulty == "easy":
        return 1.0 <= ratio <= 8.0
    if difficulty == "medium":
        return 0.65 <= ratio <= 5.5
    return 0.45 <= ratio <= 3.8


def generate_episode(
    *,
    task_id: int,
    seed: int,
    k: int,
    seq_len: int,
    train_size: int,
    base_len_choices: tuple[int, ...] = (64, 96, 128, 256, 512),
    min_channels: int = 1,
    max_channels: int = 8,
    fixed_task_type: Optional[str] = None,
    fixed_time_length: Optional[int] = None,
    fixed_channels: Optional[int] = None,
    task_type_probs: tuple[float, float, float] = (0.55, 0.30, 0.15),
    difficulty_probs: tuple[float, float, float] = (0.40, 0.40, 0.20),
    imbalance_alpha: float = 1.3,
) -> dict:
    """Generate one synthetic episode for time-series ICL.

    Returns a dict with fields aligned to synth_cauker_icl style:
    - x_ctx, y_ctx, x_qry, y_qry, x_all, y_all
    - n_ctx, n_qry, K, C, T, task_id, seed, difficulty_level, class_perm
    - task_type, rule_family, nuisance_flags

    Shape semantics:
    - x_*: (N, C, L)
    - y_*: (N,)
    """

    rng = np.random.default_rng(int(seed) + int(task_id) * 1009)
    k = int(max(2, k))
    seq_len = int(max(k + 1, seq_len))
    train_size = int(np.clip(train_size, k, seq_len - 1))

    task_type = str(fixed_task_type) if fixed_task_type is not None else _sample_task_type(rng, task_type_probs)
    difficulty = _sample_difficulty(rng, difficulty_probs)
    difficulty_level = {"easy": 0, "medium": 1, "hard": 2}[difficulty]
    rule_family = _sample_rule_family(rng, task_type)

    base_len = int(fixed_time_length) if fixed_time_length is not None else int(rng.choice(np.array(base_len_choices, dtype=np.int64)))
    if fixed_channels is not None:
        channels = int(max(1, fixed_channels))
    else:
        if task_type == "ucr_like":
            channels = 1
        elif task_type == "uea_like":
            lo = int(max(2, min_channels))
            hi = int(max(lo, max_channels))
            channels = int(rng.integers(lo, hi + 1))
        else:
            lo = int(max(1, min_channels))
            hi = int(max(lo, max_channels))
            channels = int(rng.integers(lo, hi + 1))

    class_prior = _sample_class_prior(rng, k, imbalance=max(0.2, float(imbalance_alpha)))
    ctx_counts = _alloc_counts_with_min_one(rng, k, train_size, class_prior, min_one=True)
    n_qry_total = int(max(0, seq_len - train_size))
    qry_counts = _alloc_counts_with_min_one(rng, k, n_qry_total, class_prior, min_one=False)

    nuisance_flags = {
        "elastic_warp": True,
        "phase_shift": True,
        "length_artifact": True,
        "local_motif": True,
        "sensor_noise": True,
        "channel_relation": bool(channels > 1),
        "margin_control": True,
    }

    world, world_meta = _build_shared_world(
        rng,
        task_type=task_type,
        channels=channels,
        time_length=base_len,
    )

    prototypes = []
    class_infos: list[dict] = []
    for cls in range(k):
        proto, info = _make_class_prototype(
            rng,
            world,
            class_idx=cls,
            k=k,
            task_type=task_type,
            rule_family=rule_family,
            difficulty=difficulty,
        )
        prototypes.append(proto)
        class_infos.append(info)
    prototypes_np = np.stack(prototypes, axis=0).astype(np.float32)  # (K, C, L)

    cls_ctx = []
    cls_qry = []
    for cls in range(k):
        n_ctx = int(ctx_counts[cls])
        n_qry = int(qry_counts[cls])
        n_total = n_ctx + n_qry
        if n_total <= 0:
            cls_ctx.append(np.zeros((0, channels, base_len), dtype=np.float32))
            cls_qry.append(np.zeros((0, channels, base_len), dtype=np.float32))
            continue

        proto = prototypes_np[cls]
        samples = np.repeat(proto[None, :, :], n_total, axis=0).astype(np.float32)

        for i in range(n_total):
            x = samples[i]

            if nuisance_flags["phase_shift"]:
                shift = int(rng.integers(-max(2, base_len // 25), max(3, base_len // 25)))
                x = _phase_shift(x, shift)

            if nuisance_flags["elastic_warp"]:
                x = _apply_time_warp(rng, x, strength=_difficulty_scale(difficulty))

            if nuisance_flags["length_artifact"]:
                x = _apply_length_artifact(rng, x, strength=_difficulty_scale(difficulty))

            if nuisance_flags["sensor_noise"]:
                class_gain = 0.15 * (cls / max(1, k - 1))
                x = _apply_sensor_noise(rng, x, strength=_difficulty_scale(difficulty), class_dependent_gain=class_gain)

            # UEA-like: make distractor channels noisier than informative channels.
            if channels > 1:
                info_ch = np.array(class_infos[cls]["informative_channels"], dtype=np.int64)
                all_ch = np.arange(channels, dtype=np.int64)
                distr = np.setdiff1d(all_ch, info_ch)
                if distr.size > 0:
                    x[distr] += rng.normal(0.0, 0.08 * _difficulty_scale(difficulty), size=(distr.size, base_len)).astype(np.float32)

            samples[i] = x.astype(np.float32)

        cls_ctx.append(samples[:n_ctx])
        cls_qry.append(samples[n_ctx:])

    # Light filter: reject too-trivial or too-impossible episodes.
    all_class_samples = [
        np.concatenate([cls_ctx[i], cls_qry[i]], axis=0) if (cls_ctx[i].shape[0] + cls_qry[i].shape[0]) > 0 else np.zeros((1, channels, base_len), dtype=np.float32)
        for i in range(k)
    ]
    if not _episode_quality_ok(prototypes_np, all_class_samples, difficulty=difficulty):
        # Degrade gracefully by blending prototypes toward shared world (narrow margin).
        prototypes_np = (0.7 * prototypes_np + 0.3 * world[None, :, :]).astype(np.float32)

    x_ctx = np.concatenate(cls_ctx, axis=0).astype(np.float32)
    y_ctx = np.concatenate([np.full((arr.shape[0],), i, dtype=np.int64) for i, arr in enumerate(cls_ctx)], axis=0)
    if n_qry_total > 0:
        x_qry = np.concatenate(cls_qry, axis=0).astype(np.float32)
        y_qry = np.concatenate([np.full((arr.shape[0],), i, dtype=np.int64) for i, arr in enumerate(cls_qry)], axis=0)
    else:
        x_qry = np.zeros((0, channels, base_len), dtype=np.float32)
        y_qry = np.zeros((0,), dtype=np.int64)

    p_ctx = rng.permutation(x_ctx.shape[0])
    x_ctx, y_ctx = x_ctx[p_ctx], y_ctx[p_ctx]
    if x_qry.shape[0] > 0:
        p_qry = rng.permutation(x_qry.shape[0])
        x_qry, y_qry = x_qry[p_qry], y_qry[p_qry]

    # Context/query class validity.
    ctx_classes = set(np.unique(y_ctx).tolist())
    qry_classes = set(np.unique(y_qry).tolist())
    if len(ctx_classes) != k:
        raise RuntimeError("Invalid episode: context does not cover all active classes")
    if not qry_classes.issubset(ctx_classes):
        raise RuntimeError("Invalid episode: query contains unseen classes")

    class_perm = rng.permutation(k).astype(np.int64)
    y_ctx = class_perm[y_ctx]
    y_qry = class_perm[y_qry]

    x_all = np.concatenate([x_ctx, x_qry], axis=0).astype(np.float32)
    y_all = np.concatenate([y_ctx, y_qry], axis=0).astype(np.int64)

    return {
        "x_ctx": x_ctx,
        "y_ctx": y_ctx,
        "x_qry": x_qry,
        "y_qry": y_qry,
        "x_all": x_all,
        "y_all": y_all,
        "n_ctx": int(x_ctx.shape[0]),
        "n_qry": int(x_qry.shape[0]),
        "K": int(k),
        "C": int(channels),
        "T": int(base_len),
        "task_id": int(task_id),
        "seed": int(seed),
        "difficulty_level": int(difficulty_level),
        "class_perm": class_perm,
        "task_type": str(task_type),
        "rule_family": str(rule_family),
        "nuisance_flags": nuisance_flags,
        "world_meta": world_meta,
    }


def generate_episode_from_payload(payload: Dict[str, object]) -> dict:
    return generate_episode(
        task_id=int(payload["task_id"]),
        seed=int(payload["ep_seed"]),
        k=int(payload["K"]),
        seq_len=int(payload["seq_len"]),
        train_size=int(payload["train_size"]),
        base_len_choices=tuple(int(v) for v in payload.get("base_len_choices", (64, 96, 128, 256, 512))),
        min_channels=int(payload.get("min_channels", 1)),
        max_channels=int(payload.get("max_channels", 8)),
        fixed_task_type=(None if payload.get("fixed_task_type", None) is None else str(payload.get("fixed_task_type"))),
        fixed_time_length=(None if payload.get("fixed_time_length", None) is None else int(payload.get("fixed_time_length"))),
        fixed_channels=(None if payload.get("fixed_channels", None) is None else int(payload.get("fixed_channels"))),
        task_type_probs=tuple(float(v) for v in payload.get("task_type_probs", (0.55, 0.30, 0.15))),
        difficulty_probs=tuple(float(v) for v in payload.get("difficulty_probs", (0.40, 0.40, 0.20))),
        imbalance_alpha=float(payload.get("imbalance_alpha", 1.3)),
    )


class SynthUCRUEAICLDataset(IterableDataset):
    def __init__(
        self,
        *,
        num_tasks_per_epoch: int,
        base_seed: int,
        k: int,
        seq_len: int,
        train_size: int,
        base_len_choices: tuple[int, ...] = (64, 96, 128, 256, 512),
        min_channels: int = 1,
        max_channels: int = 8,
        task_type_probs: tuple[float, float, float] = (0.55, 0.30, 0.15),
        difficulty_probs: tuple[float, float, float] = (0.40, 0.40, 0.20),
        imbalance_alpha: float = 1.3,
        level_sampler: Optional[Callable[[int, np.random.Generator], int]] = None,
    ):
        super().__init__()
        self.num_tasks_per_epoch = int(num_tasks_per_epoch)
        self.base_seed = int(base_seed)
        self.k = int(k)
        self.seq_len = int(seq_len)
        self.train_size = int(train_size)
        self.base_len_choices = tuple(int(v) for v in base_len_choices)
        self.min_channels = int(min_channels)
        self.max_channels = int(max_channels)
        self.task_type_probs = tuple(float(v) for v in task_type_probs)
        self.difficulty_probs = tuple(float(v) for v in difficulty_probs)
        self.imbalance_alpha = float(imbalance_alpha)
        self.level_sampler = level_sampler

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start = 0
            stride = 1
            worker_seed = self.base_seed
        else:
            start = int(worker.id)
            stride = int(worker.num_workers)
            worker_seed = int(self.base_seed + worker.id * 1_000_003)

        rng = np.random.default_rng(worker_seed)
        produced = 0
        task_id = start
        while produced < self.num_tasks_per_epoch:
            ep_seed = int(self.base_seed + task_id * 97 + rng.integers(0, 10_000))
            episode = generate_episode(
                task_id=task_id,
                seed=ep_seed,
                k=self.k,
                seq_len=self.seq_len,
                train_size=self.train_size,
                base_len_choices=self.base_len_choices,
                min_channels=self.min_channels,
                max_channels=self.max_channels,
                task_type_probs=self.task_type_probs,
                difficulty_probs=self.difficulty_probs,
                imbalance_alpha=self.imbalance_alpha,
            )
            yield episode
            produced += 1
            task_id += stride


def sanity_check_episode() -> None:
    ep = generate_episode(
        task_id=0,
        seed=42,
        k=4,
        seq_len=40,
        train_size=16,
        base_len_choices=(64, 96, 128),
        min_channels=1,
        max_channels=6,
    )
    print("[Sanity][UCRUEA] x_ctx:", ep["x_ctx"].shape, ep["x_ctx"].dtype)
    print("[Sanity][UCRUEA] y_ctx:", ep["y_ctx"].shape, ep["y_ctx"].dtype)
    print("[Sanity][UCRUEA] x_qry:", ep["x_qry"].shape, ep["x_qry"].dtype)
    print("[Sanity][UCRUEA] y_qry:", ep["y_qry"].shape, ep["y_qry"].dtype)
    print("[Sanity][UCRUEA] x_all:", ep["x_all"].shape)
    print("[Sanity][UCRUEA] task_type/rule:", ep["task_type"], ep["rule_family"])

    uniq_ctx = set(np.unique(ep["y_ctx"]).tolist())
    uniq_qry = set(np.unique(ep["y_qry"]).tolist())
    if not uniq_qry.issubset(uniq_ctx):
        raise RuntimeError("Sanity failed: query classes are not subset of context classes")
    print("[Sanity][UCRUEA] PASS: query classes subset of context classes")


if __name__ == "__main__":
    sanity_check_episode()

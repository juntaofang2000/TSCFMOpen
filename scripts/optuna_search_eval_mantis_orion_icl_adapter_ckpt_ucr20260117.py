#!/usr/bin/env python3
"""Optuna search for *evaluation* hyperparameters on UCR test mean accuracy.

You said:
- `--mode classifier_v2` performs better than `direct`
- `--v2_class_shift` is useful

This script therefore searches the hyperparameters that control
`MantisICLClassifierV2` inference-time ensembling/augmentation, while keeping the
trained adapter checkpoint fixed.

Objective
---------
Maximize average accuracy on UCR *test* split.

Notes
-----
- This uses UCR test accuracy as Optuna objective, so it can overfit the test
  set. This is exactly what you requested.
- Each trial evaluates multiple datasets; pruning is supported to stop bad trials
  early.

Typical usage
-------------
python scripts/optuna_search_eval_mantis_orion_icl_adapter_ckpt_ucr.py \
  --adapter_ckpt /data0/fangjuntao2025/tabicl-main/checkpoints/mantis_orion_icl_adapter_only/adapter_XXXX_epochY.pt \
  --ucr_path /data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/ \
  --device cuda:0 \
  --n_trials 50 --timeout_sec 0 \
  --study_name eval_orion_icl_ucr_test --storage sqlite:///optuna_eval_orion_icl_ucr_test.db

If optuna is missing:
  pip install -e .[optuna]
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _require_optuna():
    try:
        import optuna  # type: ignore

        return optuna
    except Exception as e:
        raise RuntimeError(
            "Optuna is required for this script. Install it with: pip install -e .[optuna]\n"
            f"Original import error: {e}"
        ) from e


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device_from_arg(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and device.index is not None:
        try:
            # Use index (int) to avoid ambiguity across torch versions.
            torch.cuda.set_device(int(device.index))
        except RuntimeError as e:
            if _is_cuda_oom(e):
                # This typically means the target GPU is already fully occupied
                # (or CUDA is in a bad state after a previous crash).
                raise RuntimeError(
                    f"Failed to select CUDA device {device} due to OOM. "
                    "This usually means the GPU has no free memory (or CUDA is left in an error state). "
                    "Try: pick another --device (e.g. cuda:1), stop other GPU processes (nvidia-smi), "
                    "or set CUDA_VISIBLE_DEVICES to a free GPU."
                ) from e
            raise
    return device


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc)
    return "CUDA out of memory" in msg or "CUBLAS_STATUS_ALLOC_FAILED" in msg


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_data_reader_root(path_str: str) -> str:
    """Normalize DataReader root paths.

    DataReader currently concatenates strings like `UEA_data_path + "UEA/"`.
    So the root path must end with '/'.
    """

    s = str(path_str)
    return s if s.endswith("/") else (s + "/")


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
    # TokenMLPAdapter: net[0] is LayerNorm when enabled, otherwise Identity (no params).
    has_ln = any(isinstance(k, str) and k.startswith("net.0.") for k in adapter_state_dict.keys())
    return True if has_ln else False


def _remap_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    return y_train_m.astype(np.int64), y_test_m.astype(np.int64)


def _ensure_2d_timeseries(X: np.ndarray) -> np.ndarray:
    """Coerce X into (N, L). UCR should already be (N,L)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X
    if X.ndim == 3 and X.shape[1] == 1:
        return X[:, 0, :]
    raise ValueError(f"Expected univariate time-series (N,L) or (N,1,L); got {X.shape}")


class _MantisAdapterPlusOrionICL(nn.Module):
    """Mantis encoder -> Adapter -> OrionMSP icl_predictor.

    Forward signature is compatible with `MantisICLClassifierV2`.
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
        self.mantis_model.eval()
        self.adapter.eval()
        self.icl_predictor.eval()
        return self

    def _pad_or_truncate(self, X: torch.Tensor) -> torch.Tensor:
        target = self.mantis_seq_len
        if X.shape[-1] == target:
            return X
        if X.shape[-1] > target:
            return X[..., :target]
        pad = X.new_zeros((*X.shape[:-1], target - X.shape[-1]))
        return torch.cat([X, pad], dim=-1)

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        """Encode rows with Mantis.

        `MantisICLClassifierV2` calls this model with inputs shaped:
        - X: (B, N, L)
        """
        X = self._pad_or_truncate(X)
        device = next(self.mantis_model.parameters()).device
        bs = max(1, int(self.mantis_batch_size))

        if X.dim() != 3:
            raise ValueError(f"Unexpected X dim for mantis encode: {X.dim()} with shape {tuple(X.shape)}")

        B, N, L = X.shape
        x_flat = X.reshape(B * N, 1, L).to(device)

        reps: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, x_flat.shape[0], bs):
                reps.append(self.mantis_model(x_flat[i : i + bs]))
        reps_t = torch.cat(reps, dim=0)  # (B*N,D)
        return reps_t.reshape(B, N, -1)

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
        _ = d, feature_shuffles, embed_with_test, inference_config
        reps = self._encode(X)
        reps = reps.to(X.device)
        reps = self.adapter(reps)
        return self.icl_predictor(
            reps,
            y_train=y_train,
            return_logits=bool(return_logits),
            softmax_temperature=float(softmax_temperature),
            mgr_config=None,
        )


@dataclass
class SearchSpace:
    n_estimators_range: tuple[int, int] = (1, 40)
    n_augmentations_range: tuple[int, int] = (1, 8)
    crop_hi_range: tuple[float, float] = (0.0, 0.25)
    # softmax_temperature is fixed in this script (see objective)


def _parse_devices(devices_csv: str) -> list[str]:
    items = [d.strip() for d in str(devices_csv).split(",") if d.strip()]
    if not items:
        raise ValueError("--devices must be a comma-separated list like 'cuda:0,cuda:1'")
    return items


def _spawn_workers(*, optuna, args: argparse.Namespace, devices: list[str]) -> int:
    """Spawn one worker process per device.

    Uses shared Optuna storage and study_name so trials are coordinated.
    """

    if not args.storage:
        raise ValueError("Multi-GPU mode requires --storage (e.g. sqlite:///optuna.db)")

    # IMPORTANT: initialize Optuna RDB schema + study in the master process first.
    # When multiple workers start concurrently against a fresh sqlite file,
    # SQLAlchemy's create_all() can race and one worker may crash with:
    #   sqlite3.OperationalError: table studies already exists
    # Pre-creating here avoids the TOCTOU DDL race.
    optuna.create_study(
        study_name=str(args.study_name),
        direction="maximize",
        storage=str(args.storage),
        load_if_exists=True,
    )

    n_workers = len(devices)

    # Determine per-worker trial budget
    if int(args.total_trials) > 0:
        total = int(args.total_trials)
        per = (total + n_workers - 1) // n_workers
    else:
        per = int(args.n_trials)

    script_path = str(Path(__file__).resolve())
    procs: list[subprocess.Popen] = []

    print(f"[MultiGPU] devices={devices} workers={n_workers} n_trials_per_worker={per}", flush=True)

    for wid, dev in enumerate(devices):
        cmd = [sys.executable, script_path]

        # Rebuild argv from args but override specific fields for worker
        # Keep it simple: forward the important ones explicitly.
        if args.adapter_ckpt is not None:
            cmd += ["--adapter_ckpt", str(args.adapter_ckpt)]
        cmd += ["--adapter_ckpt_dir", str(args.adapter_ckpt_dir)]

        if args.mantis_ckpt is not None:
            cmd += ["--mantis_ckpt", str(args.mantis_ckpt)]
        if args.orion_ckpt is not None:
            cmd += ["--orion_ckpt", str(args.orion_ckpt)]

        cmd += ["--ucr_path", str(args.ucr_path)]
        cmd += ["--uea_path", str(args.uea_path)]
        cmd += ["--device", str(dev)]
        cmd += ["--seed", str(int(args.seed))]
        cmd += ["--mantis_hidden_dim", str(int(args.mantis_hidden_dim))]
        cmd += ["--mantis_seq_len", str(int(args.mantis_seq_len))]
        cmd += ["--mantis_batch_size", str(int(args.mantis_batch_size))]

        if args.dataset is not None:
            cmd += ["--dataset", str(args.dataset)]
        cmd += ["--limit_eval_datasets", str(int(args.limit_eval_datasets))]
        if bool(args.shuffle_datasets):
            cmd += ["--shuffle_datasets"]

        if bool(args.force_class_shift):
            cmd += ["--force_class_shift"]
        cmd += ["--v2_member_batch_size", str(int(args.v2_member_batch_size))]

        if bool(args.search_random_state):
            cmd += [
                "--search_random_state",
                "--random_state_lo",
                str(int(args.random_state_lo)),
                "--random_state_hi",
                str(int(args.random_state_hi)),
            ]

        cmd += ["--study_name", str(args.study_name)]
        cmd += ["--storage", str(args.storage)]
        cmd += ["--n_trials", str(int(per))]
        cmd += ["--timeout_sec", str(int(args.timeout_sec))]
        cmd += ["--sampler", str(args.sampler)]
        cmd += ["--pruner", str(args.pruner)]
        cmd += ["--out_dir", str(args.out_dir)]

        cmd += ["--is_worker", "--worker_id", str(int(wid))]

        print(f"[MultiGPU] spawn worker{wid} on {dev}: {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1])))

    # Wait all
    exit_codes = [p.wait() for p in procs]
    bad = [c for c in exit_codes if int(c) != 0]
    if bad:
        print(f"[MultiGPU] workers exit codes: {exit_codes}", flush=True)
        return 1
    return 0


def main() -> None:
    optuna = _require_optuna()

    parser = argparse.ArgumentParser(
        description=(
            "Optuna search over evaluation-time hyperparameters for "
            "scripts/eval_mantis_orion_icl_adapter_ckpt_ucr.py (classifier_v2) on UCR test mean acc."
        )
    )

    # Checkpoints / data
    parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default="/home/hzf00006536/fjt/tabicl-main/tabicl-main/src/tabicl/checkpoints/mantis_orion_icl_adapter_only",
        help="Adapter checkpoint (.pt) saved by training script.",
    )
    parser.add_argument(
        "--adapter_ckpt_dir",
        type=str,
        default=str(Path("checkpoints") / "mantis_orion_icl_adapter_only"),
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
        default="/home/hzf00006536/fjt/tabicl-main/tabicl-main/src/tabicl/checkpoints/mantis512checkpoints/CaukerImpro-data100k_emb512_100epochs.pt",
        help="Override Mantis checkpoint (if adapter ckpt does not contain mantis_ckpt).",
    )
    parser.add_argument(
        "--orion_ckpt",
        type=str,
        default="/home/hzf00006536/fjt/Orion-MSP-v1.0.ckpt",
        help="Override Orion checkpoint (if adapter ckpt does not contain orion_ckpt).",
    )

    parser.add_argument("--ucr_path", type=str, required=True)
    parser.add_argument(
        "--uea_path",
        type=str,
        default=str(Path(".") / "UEAData"),
        help="Only for DataReader initialization; UCR evaluation does not require UEA datasets.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated device list for multi-GPU parallel search, e.g. 'cuda:0,cuda:1'. "
            "If set, this process becomes a master that spawns one worker per device (requires --storage)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    # Mantis
    parser.add_argument("--mantis_hidden_dim", type=int, default=512)
    parser.add_argument("--mantis_seq_len", type=int, default=512)
    parser.add_argument("--mantis_batch_size", type=int, default=64)

    # Dataset scope
    parser.add_argument("--dataset", type=str, default=None, help="Evaluate a single UCR dataset name")
    parser.add_argument(
        "--limit_eval_datasets",
        type=int,
        default=0,
        help="Limit number of UCR datasets for speed (0=all).",
    )
    parser.add_argument(
        "--shuffle_datasets",
        action="store_true",
        help="Shuffle dataset order before applying --limit_eval_datasets.",
    )

    # Fixed choices: you said these are good
    parser.add_argument(
        "--force_class_shift",
        action="store_true",
        help="Force v2_class_shift=True (default: True).",
    )
    parser.set_defaults(force_class_shift=True)

    # MantisICLClassifierV2 batching (speed only)
    parser.add_argument(
        "--v2_member_batch_size",
        type=int,
        default=8,
        help="MantisICLClassifierV2.batch_size (ensemble members per forward chunk).",
    )

    # Optional: search random_state (can improve score but may overfit)
    parser.add_argument(
        "--search_random_state",
        action="store_true",
        help="Include MantisICLClassifierV2.random_state in Optuna search (default: off).",
    )
    parser.add_argument(
        "--random_state_lo",
        type=int,
        default=0,
        help="Lower bound for random_state search (inclusive).",
    )
    parser.add_argument(
        "--random_state_hi",
        type=int,
        default=1000,
        help="Upper bound for random_state search (inclusive).",
    )

    # Optuna
    parser.add_argument("--study_name", type=str, default=f"eval_orion_icl_ucr_test_{_now_tag()}")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna.db. If omitted, uses in-memory study.",
    )
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument(
        "--total_trials",
        type=int,
        default=0,
        help=(
            "Total trials across all GPUs in multi-GPU mode (0=disabled). "
            "If >0, each worker runs ceil(total_trials / n_workers) trials."
        ),
    )
    parser.add_argument("--timeout_sec", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"])
    parser.add_argument("--pruner", type=str, default="median", choices=["none", "median"])

    # Output
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("checkpoints") / "optuna_eval_mantis_orion_icl_ucr_test"),
    )

    # Internal: worker mode
    parser.add_argument("--is_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker_id", type=int, default=0, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Multi-GPU master: spawn workers and exit.
    if args.devices is not None and not bool(args.is_worker):
        devices = _parse_devices(str(args.devices))
        raise SystemExit(_spawn_workers(optuna=optuna, args=args, devices=devices))

    # Ensure we can import local package
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from tabicl.model.mantis_adapter_icl import TokenMLPAdapter
    from tabicl.model.mantis_tabicl import build_mantis_encoder
    from tabicl.prior.data_reader import DataReader
    from tabicl.sklearn.classifier import MantisICLClassifierV2
    from tabicl.train.train_mantis_orion_icl_adapter_only_from_ckpts import _load_orion_checkpoint

    out_dir = Path(args.out_dir)
    _safe_mkdir(out_dir)

    wid = int(getattr(args, "worker_id", 0))
    run_tag = f"{_now_tag()}_w{wid}" if bool(getattr(args, "is_worker", False)) else _now_tag()
    with open(out_dir / f"run_{run_tag}_meta.json", "w", encoding="utf-8") as f:
        json.dump({"argv": sys.argv, "args": vars(args)}, f, indent=2, ensure_ascii=False)

    _set_seed(int(args.seed))
    device = _device_from_arg(args.device)

    adapter_ckpt_path = (
        str(args.adapter_ckpt) if args.adapter_ckpt else _find_latest_adapter_ckpt(str(args.adapter_ckpt_dir))
    )
    adapter_ckpt = torch.load(adapter_ckpt_path, map_location="cpu")
    if not isinstance(adapter_ckpt, dict) or "adapter_state_dict" not in adapter_ckpt:
        raise ValueError(f"Invalid adapter checkpoint: {adapter_ckpt_path}")

    # Optional: read adapter_best_args.json for tuned adapter hyperparams + ckpt paths.
    if args.adapter_best_args_json is not None:
        best_args_path = str(args.adapter_best_args_json)
    else:
        candidate = Path(adapter_ckpt_path).resolve().parent / "adapter_best_args.json"
        best_args_path = str(candidate) if candidate.is_file() else None

    best_args = _load_adapter_best_args_json(best_args_path) if best_args_path is not None else {}
    if best_args_path is not None:
        print(f"[Search] adapter_best_args_json: {best_args_path} (loaded {len(best_args)} keys)")

    train_args = adapter_ckpt.get("args") if isinstance(adapter_ckpt.get("args"), dict) else {}
    ckpt_params = adapter_ckpt.get("params") if isinstance(adapter_ckpt.get("params"), dict) else {}

    # Prefer adapter_best_args.json > ckpt dict > CLI.
    mantis_ckpt = str(best_args.get("mantis_ckpt") or adapter_ckpt.get("mantis_ckpt") or args.mantis_ckpt)
    orion_ckpt = str(best_args.get("orion_ckpt") or adapter_ckpt.get("orion_ckpt") or args.orion_ckpt)
    if not mantis_ckpt:
        raise ValueError("Missing mantis checkpoint. Provide --mantis_ckpt or include it in adapter ckpt / adapter_best_args.json.")
    if not orion_ckpt:
        raise ValueError("Missing orion checkpoint. Provide --orion_ckpt or include it in adapter ckpt / adapter_best_args.json.")

    # Load OrionMSP and take icl_predictor only
    orion_model, orion_cfg, _raw = _load_orion_checkpoint(orion_ckpt)
    icl_predictor = orion_model.icl_predictor
    for p in icl_predictor.parameters():
        p.requires_grad_(False)
    icl_predictor.to(device)
    icl_predictor.eval()

    embed_dim = int(orion_cfg.get("embed_dim", getattr(orion_model, "embed_dim", 128)))
    row_num_cls = int(orion_cfg.get("row_num_cls", 4))
    icl_dim = int(adapter_ckpt.get("icl_dim", embed_dim * row_num_cls))

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

    mantis_dim = int(adapter_ckpt.get("mantis_dim", getattr(mantis_model, "hidden_dim", int(args.mantis_hidden_dim))))

    adapter_state_dict = adapter_ckpt["adapter_state_dict"]

    # Resolve adapter hyperparams: prefer adapter_best_args.json > ckpt args > ckpt params > infer from weights.
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
        dropout=float(adapter_dropout_val),
        use_layernorm=not bool(adapter_no_ln_val),
    )
    adapter.load_state_dict(adapter_state_dict, strict=True)
    for p in adapter.parameters():
        p.requires_grad_(False)
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

    # DataReader expects roots that end with '/', then appends 'UEA/' and 'UCR/'.
    uea_root = _as_data_reader_root(str(args.uea_path))
    ucr_root = _as_data_reader_root(str(args.ucr_path))

    # If user is only doing UCR search and doesn't have UEA data handy,
    # ensure an empty '<uea_root>/UEA/' exists so DataReader can initialize.
    uea_dir = Path(uea_root) / "UEA"
    if not uea_dir.exists():
        _safe_mkdir(uea_dir)

    reader = DataReader(
        UEA_data_path=str(uea_root),
        UCR_data_path=str(ucr_root),
        transform_ts_size=int(args.mantis_seq_len),
    )

    if args.dataset is not None:
        dataset_names = [str(args.dataset)]
    else:
        dataset_names = sorted([str(n) for n in list(reader.dataset_list_ucr)])
        if bool(args.shuffle_datasets):
            rng = random.Random(int(args.seed))
            rng.shuffle(dataset_names)
        if int(args.limit_eval_datasets) > 0:
            dataset_names = dataset_names[: int(args.limit_eval_datasets)]

    if not dataset_names:
        raise RuntimeError("No datasets selected for evaluation")

    print(f"[Search] adapter_ckpt={adapter_ckpt_path}")
    print(f"[Search] mantis_ckpt={mantis_ckpt}")
    print(f"[Search] orion_ckpt={orion_ckpt}")
    print(f"[Search] datasets={len(dataset_names)} device={device}")

    space = SearchSpace()

    def _make_sampler():
        # Make sampler seed worker-specific to reduce duplicate suggestions across workers.
        sampler_seed = int(args.seed) + 10000 * int(getattr(args, "worker_id", 0))
        if args.sampler == "random":
            return optuna.samplers.RandomSampler(seed=int(sampler_seed))
        return optuna.samplers.TPESampler(seed=int(sampler_seed))

    def _make_pruner():
        if args.pruner == "none":
            return optuna.pruners.NopPruner()
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=str(args.study_name),
        direction="maximize",
        sampler=_make_sampler(),
        pruner=_make_pruner(),
        storage=str(args.storage) if args.storage else None,
        load_if_exists=True,
    )

    def objective(trial) -> float:
        t0 = time.time()

        # Deterministic per-trial randomness (crop sampling / class-shift ordering)
        # Optionally searched to further improve score.
        if bool(args.search_random_state):
            rs_lo = int(args.random_state_lo)
            rs_hi = int(args.random_state_hi)
            if rs_hi < rs_lo:
                raise ValueError(f"Invalid random_state range: lo={rs_lo} hi={rs_hi}")
            trial_seed = int(trial.suggest_int("random_state", rs_lo, rs_hi))
        else:
            trial_seed = int(args.seed) + int(trial.number)

        # Constraint: n_estimators * n_augmentations must be <= 32
        # Sample n_estimators first, then cap n_augmentations accordingly.
        n_estimators = int(
            trial.suggest_int(
                "n_estimators",
                int(space.n_estimators_range[0]),
                int(min(space.n_estimators_range[1], 32)),
            )
        )
        max_aug = 32 // max(1, int(n_estimators))
        max_aug = int(min(max_aug, int(space.n_augmentations_range[1])))
        min_aug = int(space.n_augmentations_range[0])
        if max_aug < min_aug:
            # Should not happen due to n_estimators <= 32, but keep safe.
            raise RuntimeError(f"Invalid aug cap: n_estimators={n_estimators} => max_aug={max_aug}")
        n_aug = int(trial.suggest_int("v2_n_augmentations", min_aug, max_aug))

        # Only consider 2 decimal places for crop rates.
        # Use Optuna discrete step of 0.01.
        crop_hi = float(
            trial.suggest_float(
                "v2_crop_rate_hi",
                float(space.crop_hi_range[0]),
                float(space.crop_hi_range[1]),
                step=0.01,
            )
        )
        crop_lo = float(trial.suggest_float("v2_crop_rate_lo", 0.0, float(crop_hi), step=0.01))

        # Fixed temperature (not searched)
        softmax_temperature = 0.90

        class_shift = bool(args.force_class_shift)

        member_batch_size = int(args.v2_member_batch_size)
        oom_reduced = False
        oom_count = 0

        accs: list[float] = []
        for i, name in enumerate(dataset_names):
            try:
                X_tr, y_tr = reader.read_dataset(name, which_set="train")
                X_te, y_te = reader.read_dataset(name, which_set="test")

                X_tr_2d = _ensure_2d_timeseries(X_tr)
                X_te_2d = _ensure_2d_timeseries(X_te)

                y_tr_m, y_te_m = _remap_labels(y_tr, y_te)

                # If CUDA OOM happens, progressively reduce v2_member_batch_size and retry.
                while True:
                    try:
                        clf = MantisICLClassifierV2(
                            n_estimators=int(n_estimators),
                            class_shift=bool(class_shift),
                            crop_rate_range=(float(crop_lo), float(crop_hi)),
                            n_augmentations=int(n_aug),
                            softmax_temperature=float(softmax_temperature),
                            average_logits=True,
                            use_hierarchical=True,
                            use_amp=True,
                            batch_size=int(member_batch_size),
                            model_path=None,
                            allow_auto_download=False,
                            checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
                            device=device,
                            random_state=int(trial_seed),
                            verbose=False,
                        )
                        clf.model_ = custom_model

                        clf.fit(X_tr_2d, y_tr_m)
                        y_pred = clf.predict(X_te_2d)
                        acc = float(np.mean(np.asarray(y_pred) == np.asarray(y_te_m)))
                        accs.append(acc)
                        break
                    except Exception as e:
                        if _is_cuda_oom(e) and int(member_batch_size) > 1:
                            oom_count += 1
                            old_bs = int(member_batch_size)
                            member_batch_size = max(1, old_bs // 2)
                            oom_reduced = True
                            print(
                                f"[Trial {trial.number}] {name}: CUDA OOM -> reduce v2_member_batch_size {old_bs} -> {member_batch_size} and retry",
                                flush=True,
                            )
                            _cleanup_cuda()
                            continue
                        raise

                mean_so_far = float(np.mean(accs))
                trial.report(mean_so_far, step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned at dataset {i+1}/{len(dataset_names)} mean={mean_so_far:.4f}")

            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"[Trial {trial.number}] {name}: failed: {e}")
                # Non-OOM failures count as 0 for this dataset.
                accs.append(0.0)

        score = float(np.mean(accs)) if accs else 0.0

        trial.set_user_attr(
            "eval_summary",
            {
                "datasets": int(len(dataset_names)),
                "score": float(score),
                "elapsed_sec": float(time.time() - t0),
            },
        )

        trial.set_user_attr(
            "oom_handling",
            {
                "v2_member_batch_size_initial": int(args.v2_member_batch_size),
                "v2_member_batch_size_final": int(member_batch_size),
                "oom_reduced": bool(oom_reduced),
                "oom_count": int(oom_count),
            },
        )

        _cleanup_cuda()
        return float(score)

    def _callback(study, trial) -> None:
        if trial.state.name != "COMPLETE":
            return
        best = study.best_trial

        best_member_bs = int(args.v2_member_batch_size)
        try:
            best_member_bs = int(best.user_attrs.get("oom_handling", {}).get("v2_member_batch_size_final", best_member_bs))
        except Exception:
            best_member_bs = int(args.v2_member_batch_size)

        out = {
            "study_name": str(study.study_name),
            "best_value": float(best.value),
            "best_params": dict(best.params),
            "best_trial_number": int(best.number),
            "best_v2_member_batch_size": int(best_member_bs),
        }
        with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        # Seed that reproduces augmentation/class-shift randomness used by best trial
        if bool(args.search_random_state) and "random_state" in best.params:
            best_seed = int(best.params["random_state"])
        else:
            best_seed = int(args.seed) + int(best.number)

        cmd = [
            "python",
            "scripts/eval_mantis_orion_icl_adapter_ckpt_ucr.py",
            "--mode",
            "classifier_v2",
            "--adapter_ckpt",
            str(adapter_ckpt_path),
            "--ucr_path",
            str(args.ucr_path),
            "--device",
            str(args.device),
            "--n_estimators",
            str(best.params.get("n_estimators")),
            "--v2_n_augmentations",
            str(best.params.get("v2_n_augmentations")),
            "--v2_crop_rate_lo",
            str(best.params.get("v2_crop_rate_lo")),
            "--v2_crop_rate_hi",
            str(best.params.get("v2_crop_rate_hi")),
            "--v2_member_batch_size",
            str(int(best_member_bs)),
            "--softmax_temperature",
            "0.90",
            "--seed",
            str(best_seed),
        ]
        if bool(args.force_class_shift):
            cmd.append("--v2_class_shift")

        with open(out_dir / "best_eval_command.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n")

    timeout = None if int(args.timeout_sec) <= 0 else int(args.timeout_sec)
    study.optimize(objective, n_trials=int(args.n_trials), timeout=timeout, callbacks=[_callback])

    print("\n[Done]")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    print(f"Saved: {out_dir / 'best_params.json'}")
    print(f"Saved: {out_dir / 'best_eval_command.txt'}")


if __name__ == "__main__":
    main()

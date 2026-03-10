from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# Ensure we import the local workspace package (repo_root/code/src)
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from orion_msp.prior.dataset import PriorDataset
from orion_msp.train.train_config import build_parser
from tabicl.model.mantis_tabicl import build_mantis_encoder, encode_with_mantis


def _prior_worker_init_fn(_worker_id: int) -> None:
    allow_cupy_in_worker = str(os.getenv("ORION_CUPY_IN_WORKER", "1")).lower() in {"1", "true", "yes", "on"}
    if not allow_cupy_in_worker:
        return
    try:
        import cupy as cp

        device_idx = int(os.getenv("ORION_CUPY_DEVICE", os.getenv("LOCAL_RANK", "0")))
        cp.cuda.Device(device_idx).use()
        if int(os.getenv("RANK", "0")) == 0 and int(_worker_id) == 0:
            print(f"[rank0] prior worker init: CuPy device bound to {device_idx}", flush=True)
    except Exception:
        pass


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pad_or_truncate_features(x: Tensor, target_dim: int) -> Tensor:
    if x.shape[-1] > target_dim:
        return x[..., :target_dim]
    if x.shape[-1] < target_dim:
        pad = x.new_zeros(*x.shape[:-1], target_dim - x.shape[-1])
        return torch.cat([x, pad], dim=-1)
    return x


def _build_dataloader(config) -> Iterable[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    dataset = PriorDataset(
        batch_size=config.batch_size,
        batch_size_per_gp=config.batch_size_per_gp,
        min_features=config.min_features,
        max_features=config.max_features,
        max_classes=config.max_classes,
        min_seq_len=config.min_seq_len,
        max_seq_len=config.max_seq_len,
        log_seq_len=config.log_seq_len,
        seq_len_per_gp=config.seq_len_per_gp,
        min_train_size=config.min_train_size,
        max_train_size=config.max_train_size,
        replay_small=config.replay_small,
        prior_type=config.prior_type,
        icl_k=getattr(config, "icl_k", 5),
        icl_time_length=getattr(config, "icl_time_length", 512),
        icl_num_features=getattr(config, "icl_num_features", 6),
        icl_single_channel=getattr(config, "icl_single_channel", False),
        icl_level=getattr(config, "icl_level", 0),
        icl_num_nodes=getattr(config, "icl_num_nodes", 18),
        icl_max_parents=getattr(config, "icl_max_parents", 5),
        icl_max_lag=getattr(config, "icl_max_lag", 5),
        icl_feature_mode=getattr(config, "icl_feature_mode", "mean"),
        icl_base_seed=getattr(config, "icl_base_seed", 42),
        icl_episode_workers=getattr(config, "icl_episode_workers", 1),
        icl_pool_backend=getattr(config, "icl_pool_backend", "thread"),
        icl_program_pool_size=getattr(config, "icl_program_pool_size", 0),
        icl_show_progress=getattr(config, "icl_show_progress", False),
        device=config.prior_device,
        n_jobs=1,
    )

    prior_num_workers = max(0, int(getattr(config, "prior_num_workers", 1)))
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "shuffle": False,
        "num_workers": prior_num_workers,
        "pin_memory": True if config.prior_device == "cpu" else False,
    }
    if prior_num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = max(2, int(getattr(config, "prior_prefetch_factor", 4)))
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["multiprocessing_context"] = "spawn"
        dataloader_kwargs["worker_init_fn"] = _prior_worker_init_fn

    return DataLoader(**dataloader_kwargs)


@torch.no_grad()
def _encode_batch(
    mantis: nn.Module,
    x_batch: Tensor,
    *,
    mantis_seq_len: int,
    mantis_batch_size: int,
    device: torch.device,
) -> Tensor:
    x_batch = _pad_or_truncate_features(x_batch, mantis_seq_len)
    B, T, H = x_batch.shape
    x_flat = x_batch.reshape(B * T, H)
    z_flat = encode_with_mantis(mantis, x_flat, batch_size=mantis_batch_size, device=device)
    return z_flat.reshape(B, T, -1)


def _train_eval_linear(
    z_train: Tensor,
    y_train: Tensor,
    z_test: Tensor,
    y_test: Tensor,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> float:
    set_global_seed(seed)

    classes = torch.unique(y_train).sort().values
    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    ytr = torch.tensor([class_to_idx[int(c)] for c in y_train.tolist()], dtype=torch.long)
    yte = torch.tensor([class_to_idx.get(int(c), -1) for c in y_test.tolist()], dtype=torch.long)

    if (yte < 0).any():
        return 0.0

    ztr = z_train.float().to(device)
    zte = z_test.float().to(device)
    ytr = ytr.to(device)
    yte = yte.to(device)

    model = nn.Linear(ztr.shape[-1], int(classes.numel())).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    n = ztr.shape[0]
    bs = max(1, int(batch_size))
    model.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad(set_to_none=True)
            logits = model(ztr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(zte).argmax(dim=-1)
        acc = (preds == yte).float().mean().item()
    return float(acc)


def _train_eval_sklearn(
    z_train: Tensor,
    y_train: Tensor,
    z_test: Tensor,
    y_test: Tensor,
    *,
    seed: int,
    classifier: str,
) -> float:
    set_global_seed(seed)

    ztr = z_train.detach().cpu().numpy().astype(np.float32, copy=False)
    zte = z_test.detach().cpu().numpy().astype(np.float32, copy=False)
    ytr = y_train.detach().cpu().numpy().astype(np.int64, copy=False)
    yte = y_test.detach().cpu().numpy().astype(np.int64, copy=False)

    if classifier == "logreg":
        model = LogisticRegression(max_iter=1000, solver="saga", random_state=seed)
    elif classifier == "svm":
        model = LinearSVC(max_iter=1000, random_state=seed)
    elif classifier == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    model.fit(ztr, ytr)
    preds = model.predict(zte)
    return float((preds == yte).mean())


def _extend_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--mantis_checkpoint", type=str, required=True)
    parser.add_argument("--mantis_seq_len", type=int, default=512)
    parser.add_argument("--mantis_hidden_dim", type=int, default=512)
    parser.add_argument("--mantis_num_patches", type=int, default=32)
    parser.add_argument("--mantis_use_fddm", action="store_true")
    parser.add_argument("--mantis_num_channels", type=int, default=1)
    parser.add_argument("--mantis_batch_size", type=int, default=512)

    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--eval_epochs", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_lr", type=float, default=1e-3)
    parser.add_argument("--eval_weight_decay", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--eval_classifier",
        type=str,
        default="linear",
        choices=["linear", "logreg", "svm", "knn"],
        help="Classifier head used per synthetic dataset.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    parser = _extend_parser(parser)
    cfg = parser.parse_args()

    if cfg.prior_type != "cauker_icl":
        raise ValueError("This script is intended for --prior_type=cauker_icl")
    if bool(cfg.seq_len_per_gp):
        raise ValueError("seq_len_per_gp is not supported in this evaluation script")

    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        os.environ.setdefault("ORION_CUPY_DEVICE", "0")

    set_global_seed(int(cfg.eval_seed))

    mantis = build_mantis_encoder(
        mantis_checkpoint=cfg.mantis_checkpoint,
        device=device,
        hidden_dim=int(cfg.mantis_hidden_dim),
        seq_len=int(cfg.mantis_seq_len),
        num_patches=int(cfg.mantis_num_patches),
        use_fddm=bool(cfg.mantis_use_fddm),
        num_channels=int(cfg.mantis_num_channels),
        strict=False,
    )
    mantis.eval()

    dataloader = _build_dataloader(cfg)
    iterator = iter(dataloader)

    accs: list[float] = []
    for step in range(int(cfg.num_batches)):
        batch = next(iterator)
        x, y, _d, seq_lens, train_sizes = batch
        if len(torch.unique(seq_lens)) > 1:
            raise ValueError("All datasets in the batch must share the same seq_len")

        x = x.to(device)
        y = y.to(device)
        z = _encode_batch(
            mantis,
            x,
            mantis_seq_len=int(cfg.mantis_seq_len),
            mantis_batch_size=int(cfg.mantis_batch_size),
            device=device,
        )

        B, T, _ = z.shape
        batch_accs = []
        for i in range(B):
            train_size = int(train_sizes[i].item())
            train_size = max(1, min(train_size, T - 1))
            z_train = z[i, :train_size]
            y_train = y[i, :train_size]
            z_test = z[i, train_size:]
            y_test = y[i, train_size:]
            if y_test.numel() == 0:
                continue
            if cfg.eval_classifier == "linear":
                acc = _train_eval_linear(
                    z_train,
                    y_train,
                    z_test,
                    y_test,
                    device=device,
                    seed=int(cfg.eval_seed),
                    epochs=int(cfg.eval_epochs),
                    batch_size=int(cfg.eval_batch_size),
                    lr=float(cfg.eval_lr),
                    weight_decay=float(cfg.eval_weight_decay),
                )
            else:
                acc = _train_eval_sklearn(
                    z_train,
                    y_train,
                    z_test,
                    y_test,
                    seed=int(cfg.eval_seed),
                    classifier=str(cfg.eval_classifier),
                )
            batch_accs.append(acc)

        if batch_accs:
            mean_acc = float(np.mean(batch_accs))
            accs.append(mean_acc)
        else:
            mean_acc = 0.0

        if (step + 1) % int(cfg.log_every) == 0:
            overall = float(np.mean(accs)) if accs else 0.0
            print(f"Step {step + 1}: batch_acc={mean_acc:.4f} overall_mean={overall:.4f}")

    overall = float(np.mean(accs)) if accs else 0.0
    print(f"Done. Mean accuracy over {len(accs)} batches: {overall:.4f}")


if __name__ == "__main__":
    main()

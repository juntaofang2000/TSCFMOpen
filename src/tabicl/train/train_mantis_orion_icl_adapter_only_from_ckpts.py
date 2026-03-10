"""Train an adapter between a frozen Mantis encoder and a frozen OrionMSP ICL module.

This script mirrors the dataset-as-task meta-training logic from:
  src/tabicl/train/train_mantis_icl_adapter_only_from_ckpts.py

The only intentional change is the ICL module source:
- Original: TabICL checkpoint's icl_predictor
- Here:     OrionMSP checkpoint's icl_predictor

Model stack:
  Mantis encoder (from --mantis_ckpt) -> TokenMLPAdapter (trainable) ->
  OrionMSP checkpoint's icl_predictor (frozen)

Notes:
- Each dataset is treated as one task.
- Support set is sampled from train split; query is the remaining train split only (no test leakage).
- Labels are remapped to contiguous IDs based on support classes.
- Training restricts each task to at most `icl_predictor.max_classes` classes.
- UEA multivariate series are converted to single-channel by averaging channels.

Expected Orion checkpoint format:
- A dict containing at least a `config` and a model state dict under one of:
  `state_dict`, `model_state_dict`, or `model`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim


# Ensure we import the local workspace package (repo_root/src)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tabicl.model.mantis_adapter_icl import TokenMLPAdapter  # noqa: E402
from tabicl.model.mantis_tabicl import build_mantis_encoder  # noqa: E402
from tabicl.prior.data_reader import DataReader  # noqa: E402

from orion_msp.model.orion_msp import OrionMSP  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_2d_timeseries(X: np.ndarray) -> np.ndarray:
    """Coerce X into (N, L). Supports UCR (N,L) and UEA (N,C,L)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X
    if X.ndim == 3:
        if X.shape[1] == 1:
            return X[:, 0, :]
        return X.mean(axis=1)
    raise ValueError(f"Unexpected X shape: {X.shape}")


def _resize_series_2d(X: np.ndarray, target_len: int) -> torch.Tensor:
    """Resize (N,L) -> torch (N,target_len) using linear interpolation."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] == target_len:
        return torch.from_numpy(X).float()

    x = torch.from_numpy(X).float().unsqueeze(1)  # (N,1,L)
    x_resized = torch.nn.functional.interpolate(x, size=int(target_len), mode="linear", align_corners=False)
    return x_resized.squeeze(1)


def _sample_support_indices_stratified(y: torch.Tensor, n_support: int) -> torch.Tensor:
    """Prefer covering as many classes as possible, then fill randomly."""
    y = y.detach().cpu().long()
    n = int(y.numel())
    if n_support >= n:
        return torch.arange(n, dtype=torch.long)

    classes = torch.unique(y)
    if classes.numel() == 0:
        return torch.randperm(n)[:n_support]

    classes = classes[torch.randperm(classes.numel())]
    picked: list[int] = []
    for cls in classes.tolist():
        idxs = torch.nonzero(y == int(cls), as_tuple=False).flatten()
        if idxs.numel() == 0:
            continue
        pick = idxs[torch.randint(0, idxs.numel(), (1,)).item()]
        picked.append(int(pick.item()))
        if len(picked) >= n_support:
            break

    picked_idx = torch.tensor(picked, dtype=torch.long)
    if picked_idx.numel() < n_support:
        mask = torch.ones(n, dtype=torch.bool)
        mask[picked_idx] = False
        remaining = torch.nonzero(mask, as_tuple=False).flatten()
        extra = remaining[torch.randperm(remaining.numel())[: (n_support - picked_idx.numel())]]
        picked_idx = torch.cat([picked_idx, extra], dim=0)

    picked_idx = picked_idx[torch.randperm(picked_idx.numel())]
    return picked_idx


def augment_batch(
    X: torch.Tensor,
    y_support: torch.Tensor,
    y_query: torch.Tensor,
    *,
    device: torch.device,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Safe-by-default augmentation: only cyclic class shift."""
    shift = int(torch.randint(0, n_classes, (1,), device=device).item())
    y_support = (y_support + shift) % n_classes
    y_query = (y_query + shift) % n_classes
    return X, y_support, y_query, shift


class MantisAdapterOrionICLOnly(nn.Module):
    def __init__(
        self,
        *,
        mantis_model: nn.Module,
        icl_predictor: nn.Module,
        adapter: nn.Module,
        mantis_seq_len: int = 512,
        mantis_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.mantis_model = mantis_model
        self.icl_predictor = icl_predictor
        self.adapter = adapter
        self.mantis_seq_len = int(mantis_seq_len)
        self.mantis_batch_size = int(mantis_batch_size)

        self.max_classes = int(getattr(self.icl_predictor, "max_classes", 10))

    def freeze_mantis_and_icl(self) -> None:
        for p in self.mantis_model.parameters():
            p.requires_grad_(False)
        for p in self.icl_predictor.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.mantis_model.eval()
        # Keep ICL in train mode to use differentiable forward path.
        self.icl_predictor.train(mode)
        return self

    def _pad_or_truncate(self, X: torch.Tensor) -> torch.Tensor:
        target = self.mantis_seq_len
        if X.shape[-1] == target:
            return X
        if X.shape[-1] > target:
            return X[..., :target]
        pad = X.new_zeros((*X.shape[:-1], target - X.shape[-1]))
        return torch.cat([X, pad], dim=-1)

    def _encode_with_mantis(self, X: torch.Tensor) -> torch.Tensor:
        """Encode X (B,T,H) -> (B,T,D_mantis) with frozen Mantis."""
        if X.ndim != 3:
            raise ValueError(f"Expected X of shape (B,T,H), got {tuple(X.shape)}")

        B, T, _H = X.shape
        X = self._pad_or_truncate(X)
        H2 = X.shape[-1]

        x_flat = X.reshape(B * T, 1, H2)
        device = next(self.mantis_model.parameters()).device
        x_flat = x_flat.to(device)

        reps: list[torch.Tensor] = []
        bs = max(1, int(self.mantis_batch_size))
        with torch.no_grad():
            for i in range(0, x_flat.shape[0], bs):
                reps.append(self.mantis_model(x_flat[i : i + bs]))
        reps_cat = torch.cat(reps, dim=0)
        return reps_cat.reshape(B, T, -1)

    def get_adapter_output(self, X: torch.Tensor) -> torch.Tensor:
        """Return adapted representations (B,T,D_icl) with grad through adapter only."""
        mantis_repr = self._encode_with_mantis(X).to(X.device)
        return self.adapter(mantis_repr)

    def forward(self, X: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        adapted = self.get_adapter_output(X)
        return self.icl_predictor(adapted, y_train=y_train)


def _load_orion_checkpoint(path: str) -> tuple[OrionMSP, dict, dict]:
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
    return model, ckpt["config"], ckpt


def _prepare_meta_tasks(model: MantisAdapterOrionICLOnly, batch_datasets, device: torch.device, args):
    """Build a batch of tasks and compute adapter outputs (connected to adapter graph)."""
    min_train_len = min(int(d[0].size(0)) for d in batch_datasets)
    n_support = min(int(args.train_size), int(min_train_len))
    if n_support < 1:
        return None

    X_seq_list = []
    y_sup_mapped_list = []
    y_qry_mapped_list = []
    valid_mask_list = []

    for X_train, y_train in batch_datasets:
        task_device = y_train.device
        max_classes = int(getattr(model, "max_classes", 10))

        classes_all = torch.unique(y_train)
        if classes_all.numel() == 0:
            continue

        k = int(min(max_classes, int(classes_all.numel()), int(n_support)))
        if k < 2:
            continue

        if classes_all.numel() > k:
            perm = torch.randperm(classes_all.numel())
            classes_keep = classes_all[perm[:k]]
        else:
            classes_keep = classes_all

        keep_mask_train = torch.isin(y_train, classes_keep)
        train_keep_idx = torch.nonzero(keep_mask_train, as_tuple=False).flatten()
        if int(train_keep_idx.numel()) < int(n_support):
            continue

        y_train_keep = y_train[train_keep_idx]
        support_local = _sample_support_indices_stratified(y_train_keep, n_support=int(n_support))
        support_idx = train_keep_idx[support_local.to(train_keep_idx.device)]

        X_sup = X_train[support_idx]
        y_sup = y_train[support_idx]

        support_classes = torch.unique(y_sup)
        keep_mask_train_sup = torch.isin(y_train, support_classes)

        support_mask = torch.ones(y_train.size(0), dtype=torch.bool, device=task_device)
        support_mask[support_idx] = False
        query_train_idx = torch.nonzero(support_mask & keep_mask_train_sup, as_tuple=False).flatten()

        # IMPORTANT: avoid data leakage. Query is built from the remaining TRAIN split only.
        X_qry = X_train[query_train_idx]
        y_qry = y_train[query_train_idx]

        if X_qry.size(0) < 1:
            continue

        X_seq = torch.cat([X_sup, X_qry], dim=0)
        X_seq_list.append(X_seq)

        unique_classes, inverse_indices = torch.unique(y_sup, return_inverse=True)
        y_sup_mapped = inverse_indices.to(device)

        max_label = int(max(y_sup.max().item(), y_qry.max().item()))
        mapper = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
        mapper[unique_classes.to(device)] = torch.arange(len(unique_classes), device=device)

        y_qry_mapped = mapper[y_qry.to(device)]
        valid_mask = y_qry_mapped != -1

        y_qry_mapped_safe = y_qry_mapped.clone()
        y_qry_mapped_safe[~valid_mask] = 0

        y_sup_mapped_list.append(y_sup_mapped)
        y_qry_mapped_list.append(y_qry_mapped_safe)
        valid_mask_list.append(valid_mask)

    if not X_seq_list:
        return None

    min_len = min(int(x.size(0)) for x in X_seq_list)
    target_len = min(int(min_len), int(args.max_icl_len))
    if target_len <= n_support:
        return None

    adapter_out_list = []
    y_sup_batch_list = []
    y_qry_batch_list = []
    mask_batch_list = []

    for i in range(len(X_seq_list)):
        x_item = X_seq_list[i][:target_len].to(device)
        emb = model.get_adapter_output(x_item.unsqueeze(0))  # (1,L,D)
        adapter_out_list.append(emb.squeeze(0))

        y_sup_batch_list.append(y_sup_mapped_list[i])

        qry_len = target_len - n_support
        y_qry_batch_list.append(y_qry_mapped_list[i][:qry_len])
        mask_batch_list.append(valid_mask_list[i][:qry_len])

    adapter_out = torch.stack(adapter_out_list, dim=0)  # (B,L,D)
    return adapter_out, y_sup_batch_list, y_qry_batch_list, mask_batch_list, n_support


def train_step(model: MantisAdapterOrionICLOnly, optimizer, criterion, batch_datasets, device: torch.device, args) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    prepared = _prepare_meta_tasks(model, batch_datasets, device, args)
    if prepared is None:
        return 0.0

    adapter_out, y_sup_batch_list, y_qry_batch_list, mask_batch_list, n_support = prepared

    total_loss = 0.0
    denom = 0

    grad_adapter_out = torch.zeros_like(adapter_out)

    for i in range(adapter_out.size(0)):
        emb = adapter_out[i]  # (L,D)
        emb_leaf = emb.detach().requires_grad_(True)

        y_sup = y_sup_batch_list[i]
        y_qry = y_qry_batch_list[i]
        mask = mask_batch_list[i]

        n_classes = int(y_sup.max().item()) + 1
        if n_classes < 2:
            continue
        if n_classes > int(getattr(model, "max_classes", 10)):
            continue

        grad_sum = torch.zeros_like(emb_leaf)
        used_views = 0

        for _ in range(int(args.n_augmentations)):
            X_aug, y_sup_aug, y_qry_aug, _shift = augment_batch(
                emb_leaf.unsqueeze(0),
                y_sup.unsqueeze(0),
                y_qry.unsqueeze(0),
                device=device,
                n_classes=n_classes,
            )

            # Orion ICL also does an in-place update on its R input.
            X_in = X_aug.clone()
            y_sup_in = y_sup_aug
            y_qry_in = y_qry_aug
            mask_in = mask

            if not mask_in.any():
                continue

            logits = model.icl_predictor(X_in, y_train=y_sup_in)

            qry_len = int(y_qry_in.size(1))
            logits_qry = logits[:, -qry_len:, :] if logits.size(1) == X_in.size(1) else logits

            logits_flat = logits_qry.reshape(-1, logits_qry.size(-1))
            y_flat = y_qry_in.reshape(-1)
            mask_flat = mask_in.reshape(-1)
            if not mask_flat.any():
                continue

            y_sel = y_flat[mask_flat]
            C = int(logits_flat.size(-1))
            y_min = int(y_sel.min().item())
            y_max = int(y_sel.max().item())
            if y_min < 0 or y_max >= C:
                continue

            loss = criterion(logits_flat[mask_flat], y_sel)

            g = torch.autograd.grad(loss, emb_leaf, retain_graph=False, allow_unused=False)[0]
            grad_sum += g.detach()

            total_loss += float(loss.item())
            denom += 1
            used_views += 1

        if used_views > 0:
            grad_adapter_out[i] = grad_sum / float(used_views)

    adapter_out.backward(grad_adapter_out)
    optimizer.step()

    if denom == 0:
        return 0.0
    return float(total_loss / float(denom))


def _load_dataset_tensors(reader: DataReader, name: str, *, seq_len: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    try:
        X_tr, y_tr = reader.read_dataset(name, which_set="train")
    except Exception:
        return None

    X_tr_2d = _ensure_2d_timeseries(X_tr)

    X_tr_t = _resize_series_2d(X_tr_2d, target_len=int(seq_len))

    y_tr_t = torch.from_numpy(np.asarray(y_tr)).long()

    return X_tr_t, y_tr_t


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Train adapter-only between Mantis encoder and OrionMSP ICL module (icl_predictor). "
            "Uses dataset-as-task meta-training on UCR/UEA."
        )
    )

    p.add_argument(
        "--mantis_ckpt",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/mantis512checkpoints/CaukerImpro-data100k_emb512_100epochs.pt", # /data0/fangjuntao2025/CauKer/CauKerOrign/checkpointMantis512GateCaukerTimeLag/best_epoch.pth
    )
    p.add_argument(
        "--orion_ckpt",
        type=str,
        default="/data0/fangjuntao2025/Orion-MSP-v1.0.ckpt",
        help="Path to an OrionMSP checkpoint (dict with 'config' and 'state_dict').",
    )

    p.add_argument("--ucr_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    p.add_argument("--uea_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/")
    p.add_argument("--use_uea", action="store_true", help="Include UEA datasets during training.")

    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--meta_batch_size", type=int, default=8, help="Number of datasets per training step")
    p.add_argument("--train_size", type=int, default=256, help="Support size per task")
    p.add_argument("--max_icl_len", type=int, default=512, help="Max total (support+query) length per task")
    p.add_argument("--n_augmentations", type=int, default=5, help="Number of augmented views per task")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--mantis_hidden_dim", type=int, default=512)
    p.add_argument("--mantis_seq_len", type=int, default=512)
    p.add_argument("--mantis_batch_size", type=int, default=64)

    p.add_argument("--adapter_hidden_dim", type=int, default=None)
    p.add_argument("--adapter_dropout", type=float, default=0.0)
    p.add_argument("--adapter_no_layernorm", action="store_true")

    p.add_argument(
        "--ckpt_dir",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/checkpoints/mantis_orion_icl_adapter_only",
    )
    p.add_argument("--ckpt_prefix", type=str, default="adapter")
    p.add_argument("--save_last", action="store_true")

    args = p.parse_args()

    set_seed(int(args.seed))

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    # Load OrionMSP ckpt and extract ICL predictor only
    orion_model, orion_cfg, _raw_ckpt = _load_orion_checkpoint(str(args.orion_ckpt))
    icl_predictor = orion_model.icl_predictor
    for p_ in icl_predictor.parameters():
        p_.requires_grad_(False)
    icl_predictor.eval()

    # Infer icl_dim from orion config
    embed_dim = int(orion_cfg.get("embed_dim", getattr(orion_model, "embed_dim", 128)))
    row_num_cls = int(orion_cfg.get("row_num_cls", 4))
    icl_dim = int(embed_dim * row_num_cls)

    # Load mantis
    mantis_model = build_mantis_encoder(
        mantis_checkpoint=Path(str(args.mantis_ckpt)),
        device=device,
        hidden_dim=int(args.mantis_hidden_dim),
        seq_len=int(args.mantis_seq_len),
    )
    for p_ in mantis_model.parameters():
        p_.requires_grad_(False)
    mantis_model.eval()

    mantis_dim = int(getattr(mantis_model, "hidden_dim", int(args.mantis_hidden_dim)))

    adapter = TokenMLPAdapter(
        mantis_dim=mantis_dim,
        icl_dim=int(icl_dim),
        hidden_dim=None if args.adapter_hidden_dim is None else int(args.adapter_hidden_dim),
        dropout=float(args.adapter_dropout),
        use_layernorm=not bool(args.adapter_no_layernorm),
    ).to(device)

    model = MantisAdapterOrionICLOnly(
        mantis_model=mantis_model,
        icl_predictor=icl_predictor,
        adapter=adapter,
        mantis_seq_len=int(args.mantis_seq_len),
        mantis_batch_size=int(args.mantis_batch_size),
    ).to(device)
    model.freeze_mantis_and_icl()

    total_params = sum(p_.numel() for p_ in model.parameters())
    trainable_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[Model] total_params={total_params} trainable(adapter)={trainable_params}")
    print(f"[Model] icl_dim={icl_dim} mantis_dim={mantis_dim} max_classes={model.max_classes}")

    reader = DataReader(UEA_data_path=str(args.uea_path), UCR_data_path=str(args.ucr_path))
    dataset_names = list(reader.dataset_list_ucr)
    if bool(args.use_uea):
        dataset_names = list(reader.dataset_list_ucr) + list(reader.dataset_list_uea)
    dataset_names = sorted(dataset_names)

    optimizer = optim.AdamW(model.adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    criterion = nn.CrossEntropyLoss()

    os.makedirs(str(args.ckpt_dir), exist_ok=True)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_base = f"{args.ckpt_prefix}_{run_tag}" if args.ckpt_prefix else run_tag

    # Save args for reproducibility
    try:
        args_log_path = os.path.join(str(args.ckpt_dir), f"{ckpt_base}_args.json")
        with open(args_log_path, "w") as f:
            json.dump({"argv": sys.argv, "args": vars(args)}, f, indent=2, ensure_ascii=False)
        print(f"[Run] Saved args: {args_log_path}")
    except Exception as e:
        print(f"[Run][warn] Failed to save args json: {e}")

    for epoch in range(int(args.epochs)):
        random.shuffle(dataset_names)

        losses: list[float] = []
        for start in range(0, len(dataset_names), int(args.meta_batch_size)):
            chunk = dataset_names[start : start + int(args.meta_batch_size)]
            batch = []
            for name in chunk:
                loaded = _load_dataset_tensors(reader, name, seq_len=int(args.mantis_seq_len))
                if loaded is None:
                    continue
                X_tr, y_tr = loaded
                batch.append((X_tr.to(device), y_tr.to(device)))

            if len(batch) < 1:
                continue

            loss = train_step(model, optimizer, criterion, batch, device, args)
            losses.append(float(loss))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[Epoch {epoch+1}/{int(args.epochs)}] mean_loss={mean_loss:.6f} steps={len(losses)}")

        ckpt_path = os.path.join(str(args.ckpt_dir), f"{ckpt_base}_epoch{epoch+1}.pt")
        if bool(args.save_last) or epoch == int(args.epochs) - 1:
            torch.save(
                {
                    "adapter_state_dict": model.adapter.state_dict(),
                    "mantis_ckpt": str(args.mantis_ckpt),
                    "orion_ckpt": str(args.orion_ckpt),
                    "orion_config": orion_cfg,
                    "icl_dim": int(icl_dim),
                    "mantis_dim": int(mantis_dim),
                    "epoch": int(epoch + 1),
                    "mean_loss": float(mean_loss),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[CKPT] saved: {ckpt_path}")


if __name__ == "__main__":
    main()

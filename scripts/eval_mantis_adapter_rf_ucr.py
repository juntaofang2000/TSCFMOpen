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


def _print_effective_runtime_config(*, title: str, payload: dict) -> None:
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(payload)
    print(f"\n{title}\n{text}\n")


def _remap_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _ensure_2d_timeseries(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X
    if X.ndim == 3 and X.shape[1] == 1:
        return X[:, 0, :]
    raise ValueError(f"Expected UCR X with shape (N,L) (or (N,1,L)); got {X.shape}")


def _pad_or_truncate_2d(X: np.ndarray, *, target_len: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")

    if X.ndim != 2:
        raise ValueError(f"Expected 2D X, got {X.shape}")

    n, l = X.shape
    if l == target_len:
        return X
    if l > target_len:
        return X[:, :target_len]

    pad = np.zeros((n, target_len - l), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)


def _extract_adapter_state_and_meta(ckpt_obj) -> tuple[dict, dict]:
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("adapter_state_dict"), dict):
        return ckpt_obj["adapter_state_dict"], ckpt_obj

    if isinstance(ckpt_obj, dict) and all(isinstance(k, str) for k in ckpt_obj.keys()):
        return ckpt_obj, {}

    raise ValueError("Invalid adapter checkpoint object: expected dict.")


def _infer_dims_from_adapter_state(adapter_state_dict: dict) -> tuple[int | None, int | None]:
    """Infer (mantis_dim, icl_dim) from TokenMLPAdapter state dict.

    TokenMLPAdapter (common):
      net.1.weight: (hidden_dim, mantis_dim)
      net.3.weight: (icl_dim, hidden_dim)

    We only need mantis_dim + icl_dim.
    """
    mantis_dim = None
    icl_dim = None

    w1 = adapter_state_dict.get("net.1.weight")
    if hasattr(w1, "ndim") and w1.ndim == 2:
        mantis_dim = int(w1.shape[1])

    w2 = adapter_state_dict.get("net.3.weight")
    if hasattr(w2, "ndim") and w2.ndim == 2:
        icl_dim = int(w2.shape[0])

    return mantis_dim, icl_dim


def _infer_adapter_hidden_dim_from_state(adapter_state_dict: dict, *, mantis_dim: int) -> int | None:
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


class _MantisPlusAdapterEmbedder(nn.Module):
    """Encode UCR samples into embeddings via Mantis -> Adapter."""

    def __init__(
        self,
        *,
        mantis_model: nn.Module,
        adapter: nn.Module,
        mantis_seq_len: int,
        mantis_batch_size: int,
    ) -> None:
        super().__init__()
        self.mantis_model = mantis_model
        self.adapter = adapter
        self.mantis_seq_len = int(mantis_seq_len)
        self.mantis_batch_size = int(mantis_batch_size)

    def train(self, mode: bool = True):
        super().train(mode)
        self.mantis_model.eval()
        self.adapter.eval()
        return self

    @torch.no_grad()
    def embed_2d(self, X2d: np.ndarray, *, device: torch.device) -> np.ndarray:
        """X2d: (N, L) -> (N, D_out)."""
        X2d = np.asarray(X2d, dtype=np.float32)
        if X2d.ndim != 2:
            raise ValueError(f"Expected X2d (N,L), got {X2d.shape}")

        # (N,L) -> (N,1,L)
        X = torch.from_numpy(X2d).unsqueeze(1).to(device)

        bs = max(1, int(self.mantis_batch_size))
        reps_out = []

        for i in range(0, X.shape[0], bs):
            x = X[i : i + bs]
            # mantis: (B,1,L) -> (B, mantis_dim)
            rep = self.mantis_model(x)
            # adapter expects (B,T,D)
            rep = rep.unsqueeze(1)
            emb = self.adapter(rep).squeeze(1)
            reps_out.append(emb.detach().cpu())

        return torch.cat(reps_out, dim=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Mantis+Adapter embeddings with a per-dataset RandomForest on UCR. "
            "For each UCR dataset: embed train/test, train RF on train embeddings, evaluate on test, then report mean accuracy."
        )
    )

    parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default="checkpoints/best_adapter.pt",
        help="Path to adapter checkpoint (.pt).",
    )
    parser.add_argument(
        "--mantis_ckpt",
        type=str,
        default="checkpoints/CaukerImpro-data100k_emb512_100epochs.pt",
        help="Mantis checkpoint path.",
    )

    parser.add_argument("--ucr_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    parser.add_argument(
        "--uea_path",
        type=str,
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",
        help="Only for DataReader initialization; not used for evaluation.",
    )

    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--dataset", type=str, default=None, help="Evaluate a single UCR dataset name")

    parser.add_argument("--mantis_hidden_dim", type=int, default=512)
    parser.add_argument("--mantis_seq_len", type=int, default=512)
    parser.add_argument("--mantis_batch_size", type=int, default=512)

    parser.add_argument("--rf_n_estimators", type=int, default=100)
    parser.add_argument("--rf_max_depth", type=int, default=None)
    parser.add_argument("--rf_min_samples_split", type=int, default=2)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)
    parser.add_argument("--rf_n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # local import so script can still be imported without sklearn
    from sklearn.ensemble import RandomForestClassifier

    device = torch.device(args.device)

    adapter_ckpt_path = str(args.adapter_ckpt)
    ckpt_obj = torch.load(adapter_ckpt_path, map_location="cpu")
    print("------------------------------------Adapter ckpt loaded from {}".format(adapter_ckpt_path))

    adapter_state_dict, ckpt_meta = _extract_adapter_state_and_meta(ckpt_obj)

    mantis_ckpt = str(ckpt_meta.get("mantis_ckpt") or args.mantis_ckpt)

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

    # Resolve mantis_dim + icl_dim for adapter construction
    inferred_mantis_dim, inferred_icl_dim = _infer_dims_from_adapter_state(adapter_state_dict)

    mantis_dim = int(ckpt_meta.get("mantis_dim") or getattr(mantis_model, "hidden_dim", 0) or 0)
    if inferred_mantis_dim is not None:
        mantis_dim = int(inferred_mantis_dim)

    icl_dim = ckpt_meta.get("icl_dim")
    if icl_dim is None and inferred_icl_dim is not None:
        icl_dim = int(inferred_icl_dim)
    if icl_dim is None:
        raise ValueError("Unable to infer icl_dim from checkpoint; please re-save adapter ckpt with 'icl_dim' in meta.")

    train_args = ckpt_meta.get("args") if isinstance(ckpt_meta.get("args"), dict) else {}
    ckpt_params = ckpt_meta.get("params") if isinstance(ckpt_meta.get("params"), dict) else {}

    adapter_hidden_dim_val = train_args.get("adapter_hidden_dim")
    if adapter_hidden_dim_val is None:
        adapter_hidden_dim_val = ckpt_params.get("adapter_hidden_dim")
    if adapter_hidden_dim_val is None:
        adapter_hidden_dim_val = _infer_adapter_hidden_dim_from_state(adapter_state_dict, mantis_dim=mantis_dim)

    adapter_dropout_val = train_args.get("adapter_dropout")
    if adapter_dropout_val is None:
        adapter_dropout_val = ckpt_params.get("adapter_dropout")
    if adapter_dropout_val is None:
        adapter_dropout_val = 0.0

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

    embedder = _MantisPlusAdapterEmbedder(
        mantis_model=mantis_model,
        adapter=adapter,
        mantis_seq_len=int(args.mantis_seq_len),
        mantis_batch_size=int(args.mantis_batch_size),
    ).eval()

    reader = DataReader(
        UEA_data_path=str(args.uea_path),
        UCR_data_path=str(args.ucr_path),
        transform_ts_size=int(args.mantis_seq_len),
    )

    if args.dataset is not None:
        dataset_names = [args.dataset]
    else:
        dataset_names = list(reader.dataset_list_ucr)

    _print_effective_runtime_config(
        title="[EvalRF][EffectiveRuntimeConfig]",
        payload={
            "cli_args": vars(args),
            "resolved": {
                "adapter_ckpt": adapter_ckpt_path,
                "mantis_ckpt": mantis_ckpt,
                "mantis_dim": int(mantis_dim),
                "icl_dim": int(icl_dim),
                "adapter_hidden_dim": (None if adapter_hidden_dim_val is None else int(adapter_hidden_dim_val)),
                "adapter_dropout": float(adapter_dropout_val),
                "adapter_use_layernorm": bool(not bool(adapter_no_ln_val)),
                "dataset_count": int(len(dataset_names)),
            },
        },
    )

    accs: list[float] = []
    for name in dataset_names:
        try:
            X_tr, y_tr = reader.read_dataset(name, which_set="train")
            X_te, y_te = reader.read_dataset(name, which_set="test")

            X_tr = _pad_or_truncate_2d(_ensure_2d_timeseries(X_tr), target_len=int(args.mantis_seq_len))
            X_te = _pad_or_truncate_2d(_ensure_2d_timeseries(X_te), target_len=int(args.mantis_seq_len))

            y_tr_m, y_te_m, _ = _remap_labels(y_tr, y_te)

            # Embed with mantis+adapter
            Z_tr = embedder.embed_2d(X_tr, device=device)
            Z_te = embedder.embed_2d(X_te, device=device)

            # rf = RandomForestClassifier(
            #     n_estimators=int(args.rf_n_estimators),
            #     max_depth=(None if args.rf_max_depth in (None, -1) else int(args.rf_max_depth)),
            #     min_samples_split=int(args.rf_min_samples_split),
            #     min_samples_leaf=int(args.rf_min_samples_leaf),
            #     random_state=int(args.seed),
            #     n_jobs=int(args.rf_n_jobs),
            # )
            rf = RandomForestClassifier(n_estimators=100, random_state=int(args.seed), n_jobs=-1)
            rf.fit(Z_tr, y_tr_m)
            y_pred = rf.predict(Z_te)
            acc = float(np.mean(y_pred == y_te_m))

            print(f"{name}: {acc:.4f}")
            accs.append(acc)
        except Exception as exc:
            print(f"{name}: failed: {exc}")

    if accs:
        print(f"\nEvaluated {len(accs)} UCR datasets | mean accuracy: {float(np.mean(accs)):.4f}")
    else:
        print("No datasets evaluated successfully.")


if __name__ == "__main__":
    main()

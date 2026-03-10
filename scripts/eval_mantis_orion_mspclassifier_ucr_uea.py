from __future__ import annotations

import argparse
import os
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
from tabicl.model.mantis_dev.adapters import VarianceBasedSelector  # noqa: E402

from orion_msp.sklearn.classifier import OrionMSPClassifier  # noqa: E402


class _CachedOrionMSPClassifier(OrionMSPClassifier):
    """Avoid re-loading the large Orion checkpoint on every .fit()."""

    def _load_model(self):  # type: ignore[override]
        if hasattr(self, "model_") and getattr(self, "model_", None) is not None:
            return
        return super()._load_model()


def _maybe_var_select_multichannel(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    enabled: bool,
    new_num_channels: int | None,
    dataset_name: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally apply VarianceBasedSelector on multichannel time-series.

    Fits selector on training split only, then transforms train/test.
    Expected input shape: (N, C, L). If not 3D or C<=1, returns unchanged.
    """

    if (not enabled) or (new_num_channels is None):
        return X_train, X_test

    X_train_np = np.asarray(X_train)
    X_test_np = np.asarray(X_test)
    if X_train_np.ndim != 3:
        return X_train_np, X_test_np

    _, c_train, _ = X_train_np.shape
    if c_train <= 1:
        return X_train_np, X_test_np

    k = int(new_num_channels)
    k = max(1, min(k, c_train))
    if k == c_train:
        return X_train_np, X_test_np

    if dataset_name:
        print(f"[VarSelector] {dataset_name}: channels {c_train} -> {k}")

    selector = VarianceBasedSelector(k)
    selector.fit(X_train_np)
    return selector.transform(X_train_np), selector.transform(X_test_np)


def _reduce_to_univariate(X: np.ndarray) -> np.ndarray:
    """Coerce time-series array to (N, L) for Mantis.

    - (N,L): keep
    - (N,1,L): squeeze channel
    - (N,C,L): mean over channels
    """

    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X
    if X.ndim == 3:
        if X.shape[1] == 1:
            return X[:, 0, :]
        return X.mean(axis=1)
    raise ValueError(f"Unexpected X shape: {X.shape}")


@torch.no_grad()
def _encode_with_mantis(
    mantis_model: torch.nn.Module,
    X: np.ndarray,
    *,
    device: torch.device,
    mantis_seq_len: int,
    batch_size: int,
) -> np.ndarray:
    """Encode time-series using frozen Mantis.

    Supported shapes
    ----------------
    - Univariate: (N, L) -> (N, D)
    - Multichannel: (N, C, L) -> (N, C*D) via per-channel encoding + concat

    Notes
    -----
    This function intentionally does NOT call `_reduce_to_univariate()`.
    For multichannel data we preserve channel information by concatenating
    the per-channel embeddings.
    """

    X_np = np.asarray(X, dtype=np.float32)
    if X_np.ndim == 2:
        n, L = X_np.shape
        C = 1
        X_flat = X_np  # (N,L)
    elif X_np.ndim == 3:
        n, C, L = X_np.shape
        X_flat = X_np.reshape(n * C, L)  # (N*C,L)
    else:
        raise ValueError(f"Unexpected X shape: {X_np.shape}")

    # Ensure length matches mantis_seq_len (DataReader usually already enforces this).
    target = int(mantis_seq_len)
    if L > target:
        X_flat = X_flat[:, :target]
    elif L < target:
        pad = np.zeros((X_flat.shape[0], target - L), dtype=np.float32)
        X_flat = np.concatenate([X_flat, pad], axis=1)

    x_t = torch.from_numpy(X_flat.astype(np.float32)).to(device)
    x_t = x_t.unsqueeze(1)  # (N*C,1,L) or (N,1,L)

    bs = max(1, int(batch_size))
    reps: list[np.ndarray] = []
    for i in range(0, int(x_t.shape[0]), bs):
        out = mantis_model(x_t[i : i + bs])
        reps.append(out.detach().float().cpu().numpy())
    flat_emb = np.concatenate(reps, axis=0)  # (N*C,D) or (N,D)

    if C == 1:
        return flat_emb

    # (N*C,D) -> (N,C,D) -> (N,C*D)
    D = int(flat_emb.shape[1])
    return flat_emb.reshape(int(n), int(C), D).reshape(int(n), int(C) * D)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate Mantis + OrionMSPClassifier on UCR/UEA time-series datasets. "
            "Pipeline: time-series -> Mantis encoder -> OrionMSPClassifier (tabular)."
        )
    )

    p.add_argument("--ucr-path", type=str, required=True)
    p.add_argument("--uea-path", type=str, required=True)
    p.add_argument("--suite", type=str, default="all", choices=["ucr", "uea", "all"])
    p.add_argument("--dataset", type=str, default=None, help="Evaluate a single dataset name")

    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument(
        "--orion-ckpt",
        type=str,
        default="/data0/fangjuntao2025/Orion-MSP-v1.0.ckpt",
        help="Path to OrionMSP checkpoint (must contain 'config' and 'state_dict').",
    )

    # OrionMSPClassifier knobs
    p.add_argument("--n-estimators", type=int, default=8)
    p.add_argument("--softmax-temperature", type=float, default=0.9)
    p.add_argument("--feat-shuffle-method", type=str, default="latin")
    p.add_argument("--no-class-shift", action="store_true")
    p.add_argument("--no-hierarchical", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--orion-batch-size", type=int, default=8)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--verbose", action="store_true")

    # Mantis knobs
    p.add_argument(
        "--mantis-ckpt",
        type=str,
        default="/data0/fangjuntao2025/TIC-FS/code/checkpoints/CaukerImpro-data100k_emb512_100epochs.pt",  #/data0/fangjuntao2025/tabicl-main/mantis512checkpoints/CaukerImpro-data100k_emb512_100epochs.pt
    )
    p.add_argument("--mantis-hidden-dim", type=int, default=512)
    p.add_argument("--mantis-seq-len", type=int, default=512)
    p.add_argument("--mantis-batch-size", type=int, default=128)

    # Optional variance selector (UEA / multichannel)
    p.add_argument("--use-var-selector", action="store_true")
    p.add_argument("--var-num-channels", type=int, default=None)

    p.add_argument("--results-dir", type=str, default="evaluation_results")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device(args.device)

    # DataReader resizes/crops time series to transform_ts_size.
    reader = DataReader(
        UEA_data_path=str(args.uea_path),
        UCR_data_path=str(args.ucr_path),
        transform_ts_size=int(args.mantis_seq_len),
        log_processing=True,
    )

    if args.dataset is not None:
        dataset_names = [str(args.dataset)]
    else:
        names = []
        if args.suite in {"ucr", "all"}:
            names.extend(list(reader.dataset_list_ucr))
        if args.suite in {"uea", "all"}:
            names.extend(list(reader.dataset_list_uea))
        dataset_names = sorted(names)

    # Load Mantis once
    mantis_model = build_mantis_encoder(
        mantis_checkpoint=Path(str(args.mantis_ckpt)),
        device=device,
        hidden_dim=int(args.mantis_hidden_dim),
        seq_len=int(args.mantis_seq_len),
    )
    for p in mantis_model.parameters():
        p.requires_grad_(False)
    mantis_model.eval()

    # Load OrionMSPClassifier once (checkpoint cached by subclass)
    clf = _CachedOrionMSPClassifier(
        n_estimators=int(args.n_estimators),
        feat_shuffle_method=str(args.feat_shuffle_method),
        class_shift=not bool(args.no_class_shift),
        softmax_temperature=float(args.softmax_temperature),
        average_logits=True,
        use_hierarchical=not bool(args.no_hierarchical),
        use_amp=not bool(args.no_amp),
        batch_size=int(args.orion_batch_size),
        model_path=str(args.orion_ckpt),
        allow_auto_download=False,
        checkpoint_version="Orion-MSP-v1.0.ckpt",
        device=device,
        random_state=int(args.random_state),
        verbose=bool(args.verbose),
    )

    results_dir = Path(str(args.results_dir))
    results_dir.mkdir(parents=True, exist_ok=True)

    ucr_set = set(reader.dataset_list_ucr)
    uea_set = set(reader.dataset_list_uea)

    results: list[tuple[str, str, float]] = []

    print(f"[Eval] suite={args.suite} datasets={len(dataset_names)} device={device}")
    print(f"[Eval] mantis_ckpt={args.mantis_ckpt}")
    print(f"[Eval] orion_ckpt={args.orion_ckpt}")

    for name in dataset_names:
        try:
            base_name = str(name).split(":", 1)[0]
            if base_name in ucr_set:
                group = "ucr"
            elif base_name in uea_set:
                group = "uea"
            else:
                group = "unknown"

            X_train, y_train = reader.read_dataset(name, which_set="train")
            X_test, y_test = reader.read_dataset(name, which_set="test")

            # Optional: variance-based channel selection BEFORE mantis encoding.
            if bool(args.use_var_selector):
                X_train, X_test = _maybe_var_select_multichannel(
                    X_train,
                    X_test,
                    enabled=True,
                    new_num_channels=(None if args.var_num_channels is None else int(args.var_num_channels)),
                    dataset_name=name,
                )

            X_train_emb = _encode_with_mantis(
                mantis_model,
                X_train,
                device=device,
                mantis_seq_len=int(args.mantis_seq_len),
                batch_size=int(args.mantis_batch_size),
            )
            X_test_emb = _encode_with_mantis(
                mantis_model,
                X_test,
                device=device,
                mantis_seq_len=int(args.mantis_seq_len),
                batch_size=int(args.mantis_batch_size),
            )

            clf.fit(X_train_emb, y_train)
            y_pred = clf.predict(X_test_emb)
            acc = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))

            print(f"{name} ({group}): {acc:.4f}")
            results.append((name, group, acc))
        except Exception as e:
            print(f"{name}: failed: {e}")

    if results:
        overall_mean = float(np.mean([acc for _, _, acc in results]))
        ucr_accs = [acc for _, g, acc in results if g == "ucr"]
        uea_accs = [acc for _, g, acc in results if g == "uea"]
        ucr_mean = (float(np.mean(ucr_accs)) if ucr_accs else float("nan"))
        uea_mean = (float(np.mean(uea_accs)) if uea_accs else float("nan"))

        print(f"\nEvaluated {len(results)} datasets")
        if ucr_accs:
            print(f"UCR: {len(ucr_accs)} | mean accuracy: {ucr_mean:.4f}")
        if uea_accs:
            print(f"UEA: {len(uea_accs)} | mean accuracy: {uea_mean:.4f}")
        print(f"Overall | mean accuracy: {overall_mean:.4f}")

        detailed_path = results_dir / "mantis_orion_mspclassifier_detailed.txt"
        summary_path = results_dir / "mantis_orion_mspclassifier_summary.txt"

        with open(detailed_path, "w", encoding="utf-8") as f:
            for n, g, a in results:
                f.write(f"{n}\t{g}\t{a:.6f}\n")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Total datasets: {len(results)}\n")
            f.write(f"UCR datasets: {len(ucr_accs)}\n")
            f.write(f"UEA datasets: {len(uea_accs)}\n")
            if ucr_accs:
                f.write(f"UCR mean accuracy: {ucr_mean:.6f}\n")
            if uea_accs:
                f.write(f"UEA mean accuracy: {uea_mean:.6f}\n")
            f.write(f"Overall mean accuracy: {overall_mean:.6f}\n")

        print(f"[Saved] {detailed_path}")
        print(f"[Saved] {summary_path}")
    else:
        print("No datasets evaluated successfully.")


if __name__ == "__main__":
    main()

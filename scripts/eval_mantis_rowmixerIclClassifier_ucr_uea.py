# 数据增强只用于embeding 后
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
from tabicl.model.mantis_dev.adapters import VarianceBasedSelector  # noqa: E402

from orion_msp.model.mantis_plus_rowmixer_lite_icl import _MantisPlusRowMixerLiteICL  # noqa: E402
from orion_msp.sklearn.classifier import RowMixerLiteICLClassifier  # noqa: E402


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


def _print_json(title: str, payload: dict) -> None:
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(payload)
    print(f"\n{title}\n{text}\n")


def _load_rowmixer_config_from_ckpt(ckpt_path: str) -> dict:
    """Load rowmixer config from checkpoint for reproducible full-model export."""
    try:
        ckpt_obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")

    if not isinstance(ckpt_obj, dict):
        raise ValueError(f"Invalid rowmixer checkpoint object: {type(ckpt_obj)}")
    cfg = ckpt_obj.get("config")
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("RowMixer checkpoint must contain a non-empty 'config' dict.")
    return dict(cfg)


def _export_full_model_bundle(
    *,
    mantis_model: torch.nn.Module,
    rowmixer_model: torch.nn.Module,
    mantis_ckpt: str,
    mantis_hidden_dim: int,
    mantis_seq_len: int,
    mantis_batch_size: int,
    rowmixer_ckpt: str,
    rowmixer_cfg: dict,
    full_ckpt_out: str,
    full_hparams_json_out: str,
) -> tuple[Path, Path]:
    """Export full Mantis+RowMixer model and aligned hparams JSON.

    Output format is compatible with scripts/eval_mantis_rowmixer_lite_icl_classifier_ucrHao_full.py.
    """

    full_model = _MantisPlusRowMixerLiteICL(
        mantis_model=mantis_model,
        rowmixer_icl=rowmixer_model,
        mantis_seq_len=int(mantis_seq_len),
        mantis_batch_size=int(mantis_batch_size),
    )
    full_model.eval()

    full_ckpt_path = Path(str(full_ckpt_out)).expanduser()
    full_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict_cpu = {k: v.detach().cpu() for k, v in full_model.state_dict().items()}
    torch.save(
        {
            "model": "_MantisPlusRowMixerLiteICL",
            "state_dict": state_dict_cpu,
            "meta": {
                "mantis_ckpt": str(mantis_ckpt),
                "rowmixer_ckpt": str(rowmixer_ckpt),
            },
        },
        str(full_ckpt_path),
    )

    mantis_cfg = {
        "hidden_dim": int(mantis_hidden_dim),
        "seq_len": int(mantis_seq_len),
        "batch_size": int(mantis_batch_size),
    }
    full_hparams = {
        "model": "_MantisPlusRowMixerLiteICL",
        "model_config": {
            "mantis": mantis_cfg,
            "rowmixer_icl": dict(rowmixer_cfg),
        },
        # Keep top-level fields for easier inspection / compatibility.
        "mantis": mantis_cfg,
        "rowmixer_icl": dict(rowmixer_cfg),
    }

    full_hparams_path = Path(str(full_hparams_json_out)).expanduser()
    full_hparams_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_hparams_path, "w", encoding="utf-8") as f:
        json.dump(full_hparams, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)

    return full_ckpt_path, full_hparams_path


def _collect_rowmixer_hparams(*, model: torch.nn.Module, ckpt_path: str) -> dict:
    row_mixer = getattr(model, "row_mixer", None)
    icl_predictor = getattr(model, "icl_predictor", None)

    # RowMixerLite details
    rowmixer_encoder_layers = None
    rowmixer_nhead = None
    rowmixer_ff_dim = None
    rowmixer_dropout = None
    rowmixer_norm_first = None
    rowmixer_activation = None

    if row_mixer is not None and hasattr(row_mixer, "encoder"):
        layers = getattr(row_mixer.encoder, "layers", None)
        if layers is not None:
            rowmixer_encoder_layers = int(len(layers))
            if len(layers) > 0:
                first = layers[0]
                if hasattr(first, "self_attn"):
                    rowmixer_nhead = int(getattr(first.self_attn, "num_heads", 0))
                if hasattr(first, "linear1"):
                    rowmixer_ff_dim = int(getattr(first.linear1, "out_features", 0))
                rowmixer_dropout_val = getattr(first, "dropout", 0.0)
                if isinstance(rowmixer_dropout_val, torch.nn.Dropout):
                    rowmixer_dropout = float(rowmixer_dropout_val.p)
                else:
                    rowmixer_dropout = float(rowmixer_dropout_val)
                rowmixer_norm_first = bool(getattr(first, "norm_first", False))
                act = getattr(first, "activation", None)
                rowmixer_activation = getattr(act, "__name__", str(act)) if act is not None else None

    patch_proj_layers = []
    patch_proj_linears = []
    if row_mixer is not None and hasattr(row_mixer, "patch_proj"):
        for layer in row_mixer.patch_proj:
            patch_proj_layers.append(layer.__class__.__name__)
            if isinstance(layer, torch.nn.Linear):
                patch_proj_linears.append(
                    {
                        "in_features": int(layer.in_features),
                        "out_features": int(layer.out_features),
                        "bias": bool(layer.bias is not None),
                    }
                )

    # ICL predictor details
    tf_icl_blocks = None
    icl_nhead = None
    icl_ff_dim = None
    icl_dropout = None
    icl_norm_first = None
    icl_activation = None
    if icl_predictor is not None and hasattr(icl_predictor, "tf_icl"):
        blocks = getattr(icl_predictor.tf_icl, "blocks", None)
        if blocks is not None:
            tf_icl_blocks = int(len(blocks))
            if len(blocks) > 0:
                b0 = blocks[0]
                if hasattr(b0, "attn"):
                    icl_nhead = int(getattr(b0.attn, "num_heads", 0))
                if hasattr(b0, "linear1"):
                    icl_ff_dim = int(getattr(b0.linear1, "out_features", 0))
                icl_dropout_val = getattr(b0, "dropout", 0.0)
                if isinstance(icl_dropout_val, torch.nn.Dropout):
                    icl_dropout = float(icl_dropout_val.p)
                else:
                    icl_dropout = float(icl_dropout_val)
                icl_norm_first = bool(getattr(b0, "norm_first", False))
                act = getattr(b0, "activation", None)
                icl_activation = getattr(act, "__name__", str(act)) if act is not None else None

    memory = getattr(icl_predictor, "memory", None) if icl_predictor is not None else None
    mem_write_layers = None
    mem_read_layers = None
    mem_num_latents = None
    if memory is not None:
        mem_write_layers = int(len(getattr(memory, "write_layers", [])))
        mem_read_layers = int(len(getattr(memory, "read_layers", [])))
        mem_num_latents = int(getattr(memory, "num_latents", 0))

    decoder_linears = []
    if icl_predictor is not None and hasattr(icl_predictor, "decoder"):
        for layer in icl_predictor.decoder:
            if isinstance(layer, torch.nn.Linear):
                decoder_linears.append(
                    {
                        "in_features": int(layer.in_features),
                        "out_features": int(layer.out_features),
                        "bias": bool(layer.bias is not None),
                    }
                )

    return {
        "model": "RowMixerLiteICL",
        "ckpt": str(ckpt_path),
        "structure": {
            "repr": str(model),
            "row_mixer": (None if row_mixer is None else str(row_mixer)),
            "icl_predictor": (None if icl_predictor is None else str(icl_predictor)),
        },
        "row_mixer": {
            "d_model": (None if row_mixer is None else int(getattr(row_mixer, "d_model", 0))),
            "patch_size": (None if row_mixer is None else int(getattr(row_mixer, "patch_size", 0))),
            "num_cls": (None if row_mixer is None else int(getattr(row_mixer, "num_cls", 0))),
            "num_global": (None if row_mixer is None else int(getattr(row_mixer, "num_global", 0))),
            "shuffle_p": (None if row_mixer is None else float(getattr(row_mixer, "shuffle_p", 0.0))),
            "encoder": {
                "num_layers": rowmixer_encoder_layers,
                "nhead": rowmixer_nhead,
                "ff_dim": rowmixer_ff_dim,
                "dropout": rowmixer_dropout,
                "norm_first": rowmixer_norm_first,
                "activation": rowmixer_activation,
            },
            "patch_proj": {
                "layer_types": patch_proj_layers,
                "linear_layers": patch_proj_linears,
            },
            "special_tokens": {
                "cls_tokens_shape": (None if row_mixer is None else list(getattr(row_mixer, "cls_tokens").shape)),
                "global_tokens_shape": (
                    None
                    if row_mixer is None or getattr(row_mixer, "global_tokens", None) is None
                    else list(row_mixer.global_tokens.shape)
                ),
            },
        },
        "icl_predictor": {
            "max_classes": (None if icl_predictor is None else int(getattr(icl_predictor, "max_classes", 0))),
            "tf_icl_blocks": tf_icl_blocks,
            "nhead": icl_nhead,
            "ff_dim": icl_ff_dim,
            "dropout": icl_dropout,
            "norm_first": icl_norm_first,
            "activation": icl_activation,
            "decoder_linears": decoder_linears,
            "memory": {
                "num_latents": mem_num_latents,
                "write_layers": mem_write_layers,
                "read_layers": mem_read_layers,
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate Mantis + RowMixerLiteICLClassifier on UCR/UEA time-series datasets. "
            "Pipeline: time-series -> Mantis encoder -> RowMixerLiteICLClassifier (tabular)."
        )
    )

    p.add_argument("--ucr-path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    p.add_argument("--uea-path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/")
    p.add_argument("--suite", type=str, default="ucr", choices=["ucr", "uea", "all"])
    p.add_argument("--dataset", type=str, default=None, help="Evaluate a single dataset name")

    p.add_argument("--device", type=str, default="cuda:0")

    # RowMixerLiteICLClassifier knobs
    p.add_argument(
        "--rowmixer-ckpt",
        type=str,
        required=True,
        help="Path to RowMixerLiteICL checkpoint (must contain 'config' and 'state_dict').",
    )
    p.add_argument("--n-estimators", type=int, default=32)
    p.add_argument("--softmax-temperature", type=float, default=0.9)
    p.add_argument("--feat-shuffle-method", type=str, default="latin")
    p.add_argument("--no-class-shift", action="store_true")
    p.add_argument("--no-hierarchical", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--rowmixer-batch-size", type=int, default=8)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--verbose", action="store_true")

    # Mantis knobs
    p.add_argument(
        "--mantis-ckpt",
        type=str,
        default="/data0/fangjuntao2025/TIC-FS/code/checkpoints/CaukerImpro-data100k_emb512_100epochs.pt",
    )
    p.add_argument("--mantis-hidden-dim", type=int, default=512)
    p.add_argument("--mantis-seq-len", type=int, default=512)
    p.add_argument("--mantis-batch-size", type=int, default=128)

    # Optional variance selector (UEA / multichannel)
    p.add_argument("--use-var-selector", action="store_true")
    p.add_argument("--var-num-channels", type=int, default=None)

    p.add_argument("--results-dir", type=str, default="evaluation_results")
    p.add_argument(
        "--rowmixer-hparams-json",
        type=str,
        default="evaluation_results/rowmixer_lite_icl_hparams.json",
        help="Path to save RowMixerLiteICL hyperparameters JSON.",
    )
    p.add_argument(
        "--export-full-model-ckpt",
        type=str,
        default="evaluation_results/mantis_rowmixer_lite_icl_full.pt",
        help="Path to save full Mantis+RowMixer model checkpoint for full evaluation script.",
    )
    p.add_argument(
        "--export-full-model-hparams-json",
        type=str,
        default="evaluation_results/mantis_rowmixer_lite_icl_model_hparams.json",
        help="Path to save full-model hyperparameters JSON for full evaluation script.",
    )
    p.add_argument(
        "--no-export-full-model",
        action="store_true",
        help="Disable exporting full Mantis+RowMixer checkpoint/hparams.",
    )

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

    # Load RowMixerLiteICLClassifier once
    clf = RowMixerLiteICLClassifier(
        n_estimators=int(args.n_estimators),
        feat_shuffle_method=str(args.feat_shuffle_method),
        class_shift=not bool(args.no_class_shift),
        softmax_temperature=float(args.softmax_temperature),
        average_logits=True,
        use_hierarchical=not bool(args.no_hierarchical),
        use_amp=not bool(args.no_amp),
        batch_size=int(args.rowmixer_batch_size),
        model_path=str(args.rowmixer_ckpt),
        device=device,
        random_state=int(args.random_state),
        verbose=bool(args.verbose),
    )

    # Force-load checkpoint to expose model structure/hparams early
    clf._load_model()
    model_hparams = _collect_rowmixer_hparams(model=clf.model_, ckpt_path=str(args.rowmixer_ckpt))
    _print_json(title="[RowMixerLiteICL][ModelHyperparams]", payload=model_hparams)

    rowmixer_cfg = _load_rowmixer_config_from_ckpt(str(args.rowmixer_ckpt))

    if not bool(args.no_export_full_model):
        full_ckpt_path, full_hparams_path = _export_full_model_bundle(
            mantis_model=mantis_model,
            rowmixer_model=clf.model_,
            mantis_ckpt=str(args.mantis_ckpt),
            mantis_hidden_dim=int(args.mantis_hidden_dim),
            mantis_seq_len=int(args.mantis_seq_len),
            mantis_batch_size=int(args.mantis_batch_size),
            rowmixer_ckpt=str(args.rowmixer_ckpt),
            rowmixer_cfg=rowmixer_cfg,
            full_ckpt_out=str(args.export_full_model_ckpt),
            full_hparams_json_out=str(args.export_full_model_hparams_json),
        )
        print(f"[Saved] {full_ckpt_path}")
        print(f"[Saved] {full_hparams_path}")

    # Save hyperparams to JSON
    if hasattr(args, "rowmixer_hparams_json") and args.rowmixer_hparams_json:
        out_path = Path(str(args.rowmixer_hparams_json)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_hparams, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        print(f"[Saved] {out_path}")

    results_dir = Path(str(args.results_dir))
    results_dir.mkdir(parents=True, exist_ok=True)

    ucr_set = set(reader.dataset_list_ucr)
    uea_set = set(reader.dataset_list_uea)

    results: list[tuple[str, str, float]] = []

    print(f"[Eval] suite={args.suite} datasets={len(dataset_names)} device={device}")
    print(f"[Eval] mantis_ckpt={args.mantis_ckpt}")
    print(f"[Eval] rowmixer_ckpt={args.rowmixer_ckpt}")

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

        detailed_path = results_dir / "mantis_rowmixer_icl_detailed.txt"
        summary_path = results_dir / "mantis_rowmixer_icl_summary.txt"

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

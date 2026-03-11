# train/runMantis_adapter_plus_orion_icl.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP


# Ensure we import the local workspace package (repo_root/code/src)
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_DEFAULT_TICFM_PRETRAINED_CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "checkpoints" / "mantis_orion_icl_full.pt"
)

from orion_msp.model.learning import ICLearning
from orion_msp.model.mantis_adapter_plus_orion_icl import _MantisAdapterPlusOrionICL
from orion_msp.train.run import Trainer as _BaseTrainer
from orion_msp.train.train_config import build_parser


from tabicl.model.mantis_adapter_icl import TokenMLPAdapter
from tabicl.model.mantis_tabicl import build_mantis_encoder, encode_with_mantis


def _extend_parser(parser):
    # Mantis encoder
    parser.add_argument("--mantis_seq_len", type=int, default=512)
    parser.add_argument("--mantis_hidden_dim", type=int, default=512)
    parser.add_argument("--mantis_num_patches", type=int, default=32)
    parser.add_argument("--mantis_use_fddm", action="store_true")
    parser.add_argument("--mantis_num_channels", type=int, default=1)
    parser.add_argument("--mantis_batch_size", type=int, default=512, help="Chunk size for mantis forward over (B*T) rows")

    # Optional Mantis eval during training
    parser.add_argument("--mantis_eval_checkpoint", type=str, default=None, help="Path or dir for pretrained Mantis encoder")
    parser.add_argument("--mantis_eval_every", type=int, default=100, help="Evaluate Mantis every N steps (rank0 only)")
    parser.add_argument(
        "--mantis_eval_classifier",
        type=str,
        default="linear",
        choices=["linear", "logreg", "svm", "knn"],
        help="Classifier head for Mantis eval on synthetic data.",
    )
    parser.add_argument("--mantis_eval_epochs", type=int, default=10)
    parser.add_argument("--mantis_eval_batch_size", type=int, default=256)
    parser.add_argument("--mantis_eval_lr", type=float, default=1e-3)
    parser.add_argument("--mantis_eval_weight_decay", type=float, default=0.0)

    # Optional reference model eval during training
    parser.add_argument(
        "--ref_eval_checkpoint",
        type=str,
        default=None,
        help="Checkpoint for reference model evaluation on the same synthetic data.",
    )
    parser.add_argument(
        "--ref_eval_every",
        type=int,
        default=100,
        help="Evaluate reference model every N steps (rank0 only)",
    )

    # Adapter
    parser.add_argument("--adapter_hidden_dim", type=int, default=1024)
    parser.add_argument("--adapter_dropout", type=float, default=0.11824302592075059)
    parser.add_argument("--adapter_no_ln", action="store_true", help="Disable adapter LayerNorm")

    # ICL dim override
    parser.add_argument(
        "--icl_dim",
        type=int,
        default=None,
        help="Override ICL token dim. Default: embed_dim * row_num_cls (same as OrionMSP ICL).",
    )
    parser.add_argument(
        "--model_max_classes",
        type=int,
        default=10,
        help="Model head output classes (ICL predictor). Decoupled from prior --max_classes.",
    )

    # Full-model pretrained TIC-FM init for end-to-end finetuning.
    parser.add_argument(
        "--init_from_ticfm_pretrained",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Initialize _MantisAdapterPlusOrionICL from a pretrained TIC-FM full-model checkpoint, "
            "then continue end-to-end training with a fresh optimizer/scheduler state."
        ),
    )
    parser.add_argument(
        "--ticfm_pretrained_checkpoint",
        type=str,
        default=str(_DEFAULT_TICFM_PRETRAINED_CHECKPOINT),
        help="Path to the pretrained TIC-FM full-model checkpoint used for initialization.",
    )
    parser.add_argument(
        "--ticfm_pretrained_strict",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use strict=True when loading the TIC-FM pretrained checkpoint.",
    )

    # DDP knobs
    parser.add_argument(
        "--ddp_find_unused_parameters",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable DDP unused-parameter detection (slower). Default is disabled to fail fast when any trainable "
            "parameter does not receive gradients. Use this only for debugging/temporary compatibility."
        ),
    )
    parser.add_argument(
        "--debug_print_missing_grads",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Print trainable parameter names with grad=None after the last micro-batch backward of each step. "
            "Useful to identify parameters that will trigger DDP reduction errors."
        ),
    )

    parser.add_argument(
        "--force_flash_sdpa",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Force Flash SDPA by disabling other SDPA backends. This will error if Flash is unsupported "
            "for the current dtype/shape/device."
        ),
    )

    # ICL defaults from model.json -> orion.icl_predictor.config
    parser.set_defaults(
        embed_dim=128,
        icl_num_blocks=12,
        icl_nhead=4,
        ff_factor=2,
        dropout=0.0,
        norm_first=True,
        perc_num_latents=32,
        perc_layers=2,
    )

    return parser


class Trainer(_BaseTrainer):
    """Trainer that swaps the backbone with end-to-end `_MantisAdapterPlusOrionICL`.

    Data: uses the same PriorDataset / SCM synthetic tasks as the base trainer.
    """

    def __init__(self, config):
        self._did_shape_assert = False
        super().__init__(config)
        self._configure_mantis_eval()
        self._configure_ref_eval()

    def _configure_mantis_eval(self) -> None:
        self.mantis_eval_enabled = bool(getattr(self.config, "mantis_eval_checkpoint", None))
        self.mantis_eval_model = None
        if not self.mantis_eval_enabled:
            return
        if not self.master_process:
            return

        self.mantis_eval_model = build_mantis_encoder(
            mantis_checkpoint=getattr(self.config, "mantis_eval_checkpoint", None),
            device=self.config.device,
            hidden_dim=int(getattr(self.config, "mantis_hidden_dim", 512)),
            seq_len=int(getattr(self.config, "mantis_seq_len", 512)),
            num_patches=int(getattr(self.config, "mantis_num_patches", 32)),
            use_fddm=bool(getattr(self.config, "mantis_use_fddm", False)),
            num_channels=int(getattr(self.config, "mantis_num_channels", 1)),
            strict=False,
        )
        self.mantis_eval_model.eval()

    def _configure_ref_eval(self) -> None:
        self.ref_eval_enabled = bool(getattr(self.config, "ref_eval_checkpoint", None))
        self.ref_eval_model = None
        if not self.ref_eval_enabled:
            return
        if not self.master_process:
            return

        ckpt_path = str(getattr(self.config, "ref_eval_checkpoint", ""))
        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
        model_cfg = ckpt_obj.get("config") if isinstance(ckpt_obj, dict) else None

        if isinstance(model_cfg, dict) and {"mantis", "adapter", "icl_predictor"}.issubset(model_cfg.keys()):
            mantis_cfg = model_cfg["mantis"]
            adapter_cfg = model_cfg["adapter"]
            icl_cfg = model_cfg["icl_predictor"]

            mantis_model = build_mantis_encoder(
                mantis_checkpoint=mantis_cfg.get("ckpt", None),
                device=self.config.device,
                hidden_dim=int(mantis_cfg.get("hidden_dim", 512)),
                seq_len=int(mantis_cfg.get("seq_len", 512)),
                num_patches=int(mantis_cfg.get("num_patches", 32)),
                use_fddm=bool(mantis_cfg.get("use_fddm", False)),
                num_channels=int(mantis_cfg.get("num_channels", 1)),
                strict=False,
            )

            adapter = TokenMLPAdapter(
                mantis_dim=int(adapter_cfg.get("mantis_dim", mantis_cfg.get("hidden_dim", 512))),
                icl_dim=int(adapter_cfg.get("icl_dim", int(self.config.embed_dim) * int(self.config.row_num_cls))),
                hidden_dim=adapter_cfg.get("hidden_dim", None),
                dropout=float(adapter_cfg.get("dropout", 0.0)),
                use_layernorm=bool(adapter_cfg.get("use_layernorm", True)),
            ).to(self.config.device)

            icl_predictor = ICLearning(
                max_classes=int(icl_cfg.get("max_classes", self.config.max_classes)),
                d_model=int(icl_cfg.get("d_model", adapter_cfg.get("icl_dim", 512))),
                num_blocks=int(icl_cfg.get("num_blocks", self.config.icl_num_blocks)),
                nhead=int(icl_cfg.get("nhead", self.config.icl_nhead)),
                dim_feedforward=int(icl_cfg.get("dim_feedforward", int(self.config.ff_factor) * int(icl_cfg.get("d_model", 512)))),
                dropout=float(icl_cfg.get("dropout", self.config.dropout)),
                activation=str(icl_cfg.get("activation", self.config.activation)),
                norm_first=bool(icl_cfg.get("norm_first", self.config.norm_first)),
                perc_num_latents=int(icl_cfg.get("perc_num_latents", self.config.perc_num_latents)),
                perc_layers=int(icl_cfg.get("perc_layers", self.config.perc_layers)),
            ).to(self.config.device)

            mantis_seq_len = int(mantis_cfg.get("seq_len", 512))
            mantis_batch_size = int(mantis_cfg.get("batch_size", getattr(self.config, "mantis_batch_size", 256)))
        else:
            mantis_seq_len = int(getattr(self.config, "mantis_seq_len", 512))
            mantis_batch_size = int(getattr(self.config, "mantis_batch_size", 256))
            mantis_model = build_mantis_encoder(
                mantis_checkpoint=None,
                device=self.config.device,
                hidden_dim=int(getattr(self.config, "mantis_hidden_dim", 512)),
                seq_len=mantis_seq_len,
                num_patches=int(getattr(self.config, "mantis_num_patches", 32)),
                use_fddm=bool(getattr(self.config, "mantis_use_fddm", False)),
                num_channels=int(getattr(self.config, "mantis_num_channels", 1)),
                strict=False,
            )

            adapter = TokenMLPAdapter(
                mantis_dim=int(getattr(self.config, "mantis_hidden_dim", 512)),
                icl_dim=int(getattr(self.config, "icl_dim", 0) or (int(self.config.embed_dim) * int(self.config.row_num_cls))),
                hidden_dim=(None if getattr(self.config, "adapter_hidden_dim", None) is None else int(self.config.adapter_hidden_dim)),
                dropout=float(getattr(self.config, "adapter_dropout", 0.0)),
                use_layernorm=bool(not bool(getattr(self.config, "adapter_no_ln", False))),
            ).to(self.config.device)

            icl_predictor = ICLearning(
                max_classes=int(getattr(self.config, "model_max_classes", self.config.max_classes)),
                d_model=int(getattr(self.config, "icl_dim", 0) or (int(self.config.embed_dim) * int(self.config.row_num_cls))),
                num_blocks=int(self.config.icl_num_blocks),
                nhead=int(self.config.icl_nhead),
                dim_feedforward=int(self.config.ff_factor) * int(getattr(self.config, "icl_dim", 0) or (int(self.config.embed_dim) * int(self.config.row_num_cls))),
                dropout=float(self.config.dropout),
                activation=str(self.config.activation),
                norm_first=bool(self.config.norm_first),
                perc_num_latents=int(self.config.perc_num_latents),
                perc_layers=int(self.config.perc_layers),
            ).to(self.config.device)

        ref_model = _MantisAdapterPlusOrionICL(
            mantis_model=mantis_model,
            adapter=adapter,
            icl_predictor=icl_predictor,
            mantis_seq_len=mantis_seq_len,
            mantis_batch_size=mantis_batch_size,
        ).to(self.config.device)

        if isinstance(ckpt_obj, dict):
            state_dict = ckpt_obj.get("state_dict") or ckpt_obj.get("model_state_dict") or ckpt_obj.get("model") or ckpt_obj
        else:
            state_dict = ckpt_obj
        cleaned = {str(k).replace("module.", ""): v for k, v in state_dict.items()}
        ref_model.load_state_dict(cleaned, strict=False)
        ref_model.eval()
        self.ref_eval_model = ref_model

    @staticmethod
    def _extract_state_dict(ckpt_obj: object) -> dict[str, Tensor]:
        if isinstance(ckpt_obj, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = ckpt_obj.get(key)
                if isinstance(value, dict):
                    return {str(k).replace("module.", ""): v for k, v in value.items()}
            if all(isinstance(k, str) for k in ckpt_obj.keys()):
                return {str(k).replace("module.", ""): v for k, v in ckpt_obj.items()}
        raise ValueError("Unsupported TIC-FM checkpoint format; expected a dict or state_dict.")

    def _maybe_init_from_ticfm_pretrained(self, model: torch.nn.Module) -> None:
        if not bool(getattr(self.config, "init_from_ticfm_pretrained", False)):
            return

        ckpt_path = Path(str(getattr(self.config, "ticfm_pretrained_checkpoint", ""))).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"TIC-FM pretrained checkpoint not found: {ckpt_path}")

        ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = self._extract_state_dict(ckpt_obj)
        incompatible = model.load_state_dict(
            state_dict,
            strict=bool(getattr(self.config, "ticfm_pretrained_strict", True)),
        )

        if self.master_process:
            print(f"Initialized model from TIC-FM checkpoint: {ckpt_path}")
            if incompatible is not None:
                if getattr(incompatible, "missing_keys", None):
                    print(f"  Missing keys: {len(incompatible.missing_keys)}")
                if getattr(incompatible, "unexpected_keys", None):
                    print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")

    @staticmethod
    def _pad_or_truncate_features(x: Tensor, target_dim: int) -> Tensor:
        if x.shape[-1] > target_dim:
            return x[..., :target_dim]
        if x.shape[-1] < target_dim:
            pad = x.new_zeros(*x.shape[:-1], target_dim - x.shape[-1])
            return torch.cat([x, pad], dim=-1)
        return x

    @staticmethod
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

        model = torch.nn.Linear(ztr.shape[-1], int(classes.numel())).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        n = ztr.shape[0]
        bs = max(1, int(batch_size))
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

        model.train()
        for _ in range(int(epochs)):
            perm = torch.randperm(n, generator=g, device=device)
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

    @staticmethod
    def _train_eval_sklearn(
        z_train: Tensor,
        y_train: Tensor,
        z_test: Tensor,
        y_test: Tensor,
        *,
        seed: int,
        classifier: str,
    ) -> float:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC

        ztr = z_train.detach().cpu().numpy().astype("float32", copy=False)
        zte = z_test.detach().cpu().numpy().astype("float32", copy=False)
        ytr = y_train.detach().cpu().numpy().astype("int64", copy=False)
        yte = y_test.detach().cpu().numpy().astype("int64", copy=False)

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

    def _eval_mantis_on_batch(self, batch) -> float | None:
        if not self.mantis_eval_enabled or self.mantis_eval_model is None:
            return None
        if not self.master_process:
            return None

        x, y, _d, seq_lens, train_sizes = batch
        if len(torch.unique(seq_lens)) > 1:
            return None

        mantis_seq_len = int(getattr(self.config, "mantis_seq_len", 512))
        mantis_bs = int(getattr(self.config, "mantis_batch_size", 512))

        x = x.to(self.config.device)
        y = y.to(self.config.device)
        x = self._pad_or_truncate_features(x, mantis_seq_len)

        B, T, H = x.shape
        x_flat = x.reshape(B * T, H)
        with torch.no_grad():
            z_flat = encode_with_mantis(
                self.mantis_eval_model,
                x_flat,
                batch_size=mantis_bs,
                device=self.config.device,
            )
        z = z_flat.reshape(B, T, -1)

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

            if str(getattr(self.config, "mantis_eval_classifier", "linear")) == "linear":
                acc = self._train_eval_linear(
                    z_train,
                    y_train,
                    z_test,
                    y_test,
                    device=torch.device(self.config.device),
                    seed=int(getattr(self.config, "torch_seed", 42)),
                    epochs=int(getattr(self.config, "mantis_eval_epochs", 10)),
                    batch_size=int(getattr(self.config, "mantis_eval_batch_size", 256)),
                    lr=float(getattr(self.config, "mantis_eval_lr", 1e-3)),
                    weight_decay=float(getattr(self.config, "mantis_eval_weight_decay", 0.0)),
                )
            else:
                acc = self._train_eval_sklearn(
                    z_train,
                    y_train,
                    z_test,
                    y_test,
                    seed=int(getattr(self.config, "torch_seed", 42)),
                    classifier=str(getattr(self.config, "mantis_eval_classifier", "linear")),
                )
            batch_accs.append(acc)

        if not batch_accs:
            return None
        return float(sum(batch_accs) / len(batch_accs))

    def _eval_ref_model_on_batch(self, batch) -> float | None:
        if not self.ref_eval_enabled or self.ref_eval_model is None:
            return None
        if not self.master_process:
            return None

        x, y, _d, seq_lens, train_sizes = batch
        if len(torch.unique(seq_lens)) > 1:
            return None

        mantis_seq_len = int(getattr(self.config, "mantis_seq_len", 512))
        x = x.to(self.config.device)
        y = y.to(self.config.device)
        x = self._pad_or_truncate_features(x, mantis_seq_len)

        batch_accs = []
        B, T, _ = x.shape
        with torch.no_grad():
            for i in range(B):
                train_size = int(train_sizes[i].item())
                train_size = max(1, min(train_size, T - 1))
                x_i = x[i : i + 1]
                y_i = y[i : i + 1]
                y_train = y_i[:, :train_size]
                y_test = y_i[:, train_size:]
                if y_test.numel() == 0:
                    continue

                logits = self.ref_eval_model(x_i, y_train, None)
                _, ttest, cmax = logits.shape
                pred = logits.reshape(-1, cmax)
                true = y_test.reshape(-1).long()
                valid = (true >= 0) & (true < cmax)
                if not torch.all(valid):
                    true = true[valid]
                    pred = pred[valid]
                if true.numel() == 0:
                    continue

                active_idx = torch.unique(y_train).sort().values.long()
                if active_idx.numel() > 0:
                    pred = pred.index_select(dim=-1, index=active_idx)
                    if active_idx.min().item() == 0 and active_idx.max().item() == active_idx.numel() - 1:
                        true_active = true
                    else:
                        mapper = torch.full((cmax,), -1, dtype=torch.long, device=active_idx.device)
                        mapper[active_idx] = torch.arange(active_idx.numel(), device=active_idx.device)
                        true_active = mapper[true]
                        keep = true_active >= 0
                        pred = pred[keep]
                        true_active = true_active[keep]
                else:
                    true_active = true

                if pred.numel() == 0 or true_active.numel() == 0:
                    continue
                acc = (pred.argmax(dim=1) == true_active).float().mean().item()
                batch_accs.append(acc)

        if not batch_accs:
            return None
        return float(sum(batch_accs) / len(batch_accs))

    def _freeze_known_unused_mantis_params(self, mantis_model) -> int:
        """Freeze known dead branches in Mantis architecture.

        In current Mantis8M implementation, when `pre_training=False` the `prj`
        head is defined but not used in forward. Keeping it trainable causes DDP
        to report unused parameters.
        """
        frozen = 0

        pre_training = bool(getattr(mantis_model, "pre_training", False))
        prj = getattr(mantis_model, "prj", None)
        if (not pre_training) and (prj is not None):
            for p in prj.parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    frozen += p.numel()

        return frozen

    def _debug_log_missing_grads(self):
        if not bool(getattr(self.config, "debug_print_missing_grads", False)):
            return

        missing_names: list[str] = []
        for name, p in self.raw_model.named_parameters():
            if p.requires_grad and p.grad is None:
                missing_names.append(name)

        if not missing_names:
            return

        rank = int(getattr(self, "ddp_rank", 0))
        show_n = min(64, len(missing_names))
        shown = "\n  - ".join(missing_names[:show_n])
        suffix = "" if len(missing_names) <= show_n else f"\n  ... and {len(missing_names) - show_n} more"
        print(
            f"[Rank {rank}] Missing gradients for {len(missing_names)} trainable params at step {self.curr_step}:\n"
            f"  - {shown}{suffix}"
        )

    def build_model(self):
        prior_type = str(getattr(self.config, "prior_type", ""))
        if prior_type not in {"mlp_scm", "tree_scm", "mix_scm", "cauker_icl"}:
            raise ValueError(
                f"This trainer is intended for SCM synthetic data; got prior_type={prior_type!r}. "
                "Use --prior_type mlp_scm|tree_scm|mix_scm|cauker_icl."
            )

        model_max_classes = int(getattr(self.config, "model_max_classes", 10))
        prior_max_classes = int(getattr(self.config, "max_classes", model_max_classes))
        if model_max_classes < 1:
            raise ValueError(f"model_max_classes must be >= 1, got {model_max_classes}")
        if prior_type == "cauker_icl":
            eff_prior_upper = int(min(prior_max_classes, int(getattr(self.config, "icl_k", prior_max_classes))))
        else:
            eff_prior_upper = prior_max_classes
        if eff_prior_upper > model_max_classes:
            raise ValueError(
                "Incompatible class settings: prior may generate more classes than model head supports. "
                f"prior upper bound={eff_prior_upper}, model_max_classes={model_max_classes}. "
                "Set --model_max_classes >= effective prior classes, or reduce --max_classes/--icl_k."
            )

        mantis_seq_len = int(getattr(self.config, "mantis_seq_len", 512))
        mantis_hidden_dim = int(getattr(self.config, "mantis_hidden_dim", 512))
        icl_dim = int(getattr(self.config, "icl_dim", 0) or (int(self.config.embed_dim) * int(self.config.row_num_cls)))

        mantis_model = build_mantis_encoder(
            mantis_checkpoint=None,
            device=self.config.device,
            hidden_dim=mantis_hidden_dim,
            seq_len=mantis_seq_len,
            num_patches=int(getattr(self.config, "mantis_num_patches", 32)),
            use_fddm=bool(getattr(self.config, "mantis_use_fddm", False)),
            num_channels=int(getattr(self.config, "mantis_num_channels", 1)),
            strict=False,
        )
        # build_mantis_encoder defaults to eval(); for end-to-end training we enable train mode
        mantis_model.train()
        frozen_numel = self._freeze_known_unused_mantis_params(mantis_model)
        if self.master_process and frozen_numel > 0:
            print(f"Froze known-unused Mantis params: {frozen_numel:,} (pre_training projector)")

        adapter = TokenMLPAdapter(
            mantis_dim=int(mantis_hidden_dim),
            icl_dim=int(icl_dim),
            hidden_dim=(None if getattr(self.config, "adapter_hidden_dim", None) is None else int(self.config.adapter_hidden_dim)),
            dropout=float(getattr(self.config, "adapter_dropout", 0.0)),
            use_layernorm=bool(not bool(getattr(self.config, "adapter_no_ln", False))),
        ).to(self.config.device)

        icl_predictor = ICLearning(
            max_classes=int(model_max_classes),
            d_model=int(icl_dim),
            num_blocks=int(self.config.icl_num_blocks),
            nhead=int(self.config.icl_nhead),
            dim_feedforward=int(self.config.ff_factor) * int(icl_dim),
            dropout=float(self.config.dropout),
            activation=str(self.config.activation),
            norm_first=bool(self.config.norm_first),
            perc_num_latents=int(self.config.perc_num_latents),
            perc_layers=int(self.config.perc_layers),
        ).to(self.config.device)

        model = _MantisAdapterPlusOrionICL(
            mantis_model=mantis_model,
            adapter=adapter,
            icl_predictor=icl_predictor,
            mantis_seq_len=mantis_seq_len,
            mantis_batch_size=int(getattr(self.config, "mantis_batch_size", 256)),
        ).to(self.config.device)

        self._maybe_init_from_ticfm_pretrained(model)

        self.model_config = {
            "model": "_MantisAdapterPlusOrionICL",
            "mantis": {
                "ckpt": None,
                "seq_len": mantis_seq_len,
                "hidden_dim": mantis_hidden_dim,
                "num_patches": int(getattr(self.config, "mantis_num_patches", 32)),
                "use_fddm": bool(getattr(self.config, "mantis_use_fddm", False)),
                "num_channels": int(getattr(self.config, "mantis_num_channels", 1)),
                "batch_size": int(getattr(self.config, "mantis_batch_size", 256)),
            },
            "adapter": {
                "mantis_dim": mantis_hidden_dim,
                "icl_dim": icl_dim,
                "hidden_dim": getattr(self.config, "adapter_hidden_dim", None),
                "dropout": float(getattr(self.config, "adapter_dropout", 0.0)),
                "use_layernorm": bool(not bool(getattr(self.config, "adapter_no_ln", False))),
            },
            "icl_predictor": {
                "max_classes": int(model_max_classes),
                "d_model": int(icl_dim),
                "num_blocks": int(self.config.icl_num_blocks),
                "nhead": int(self.config.icl_nhead),
                "dim_feedforward": int(self.config.ff_factor) * int(icl_dim),
                "dropout": float(self.config.dropout),
                "activation": str(self.config.activation),
                "norm_first": bool(self.config.norm_first),
                "perc_num_latents": int(self.config.perc_num_latents),
                "perc_layers": int(self.config.perc_layers),
            },
            "pretrained_init": {
                "enabled": bool(getattr(self.config, "init_from_ticfm_pretrained", False)),
                "checkpoint": (
                    str(getattr(self.config, "ticfm_pretrained_checkpoint", ""))
                    if bool(getattr(self.config, "init_from_ticfm_pretrained", False))
                    else None
                ),
                "strict": bool(getattr(self.config, "ticfm_pretrained_strict", True)),
            },
        }

        if getattr(self.config, "model_compile", False):
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled.")

        if self.ddp:
            find_unused = bool(getattr(self.config, "ddp_find_unused_parameters", False))
            self.model = DDP(
                model,
                device_ids=[self.ddp_local_rank],
                broadcast_buffers=False,
                find_unused_parameters=find_unused,
            )
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

        if self.master_process:
            num_params = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {num_params:,}")

    def run_batch(self, batch):
        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
        results = super().run_batch(batch)

        sync_needed = False
        if self.mantis_eval_enabled:
            eval_every = int(getattr(self.config, "mantis_eval_every", 0))
            step_idx = int(self.curr_step) + 1
            if eval_every > 0 and (step_idx % eval_every == 0):
                acc = None
                if self.master_process:
                    acc = self._eval_mantis_on_batch(batch)
                    if acc is not None:
                        results["mantis_eval_acc"] = acc
                        print(f"[rank0] step={step_idx} mantis_eval_acc={acc:.4f}", flush=True)
                sync_needed = True

        if self.ref_eval_enabled:
            ref_every = int(getattr(self.config, "ref_eval_every", 0))
            step_idx = int(self.curr_step) + 1
            if ref_every > 0 and (step_idx % ref_every == 0):
                acc = None
                if self.master_process:
                    acc = self._eval_ref_model_on_batch(batch)
                    if acc is not None:
                        results["ref_eval_acc"] = acc
                        print(f"[rank0] step={step_idx} ref_eval_acc={acc:.4f}", flush=True)
                sync_needed = True

        if self.ddp and sync_needed:
            torch.distributed.barrier()

        return results

    def align_micro_batch(self, micro_X: Tensor, micro_y: Tensor, micro_d: Tensor, seq_len: int):
        # Keep base sequence alignment behavior.
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]
        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]

        # For Mantis-based encoding we interpret feature dim as a 1D signal.
        # Do NOT crop by micro_d.max(); instead pad/truncate to mantis_seq_len.
        target = int(getattr(self.config, "mantis_seq_len", 512))
        Fdim = int(micro_X.shape[-1])
        if Fdim > target:
            micro_X = micro_X[..., :target]
        elif Fdim < target:
            pad = micro_X.new_zeros(*micro_X.shape[:-1], target - Fdim)
            micro_X = torch.cat([micro_X, pad], dim=-1)

        return micro_X, micro_y

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)

        y_train = micro_y[:, :train_size].long()
        y_test = micro_y[:, train_size:].long()

        if y_test.numel() == 0:
            return {"ce": 0.0, "accuracy": 0.0}

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            logits = self.model(micro_X, y_train, None)  # (B, Ttest, C)
            B, Ttest, C = logits.shape
            pred = logits.reshape(-1, C)
            true = y_test.reshape(-1).long()

            valid = (true >= 0) & (true < C)
            if not torch.all(valid):
                true = true[valid]
                pred = pred[valid]
            if true.numel() == 0:
                return {"ce": 0.0, "accuracy": 0.0}

            loss = F.cross_entropy(pred, true)

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        if micro_batch_idx == num_micro_batches - 1:
            self._debug_log_missing_grads()

        # Active-class accuracy (query logits sliced to active classes)
        logits_acc = logits.reshape(-1, C)
        true_acc = y_test.reshape(-1).long()
        valid_acc = (true_acc >= 0) & (true_acc < C)
        if valid_acc.any():
            logits_acc = logits_acc[valid_acc]
            true_acc = true_acc[valid_acc]
        else:
            logits_acc = logits_acc[:0]
            true_acc = true_acc[:0]

        active_idx = torch.unique(y_train).sort().values.long()
        K = int(active_idx.numel())
        if K > 0 and logits_acc.numel() > 0:
            logits_active = logits_acc.index_select(dim=-1, index=active_idx)
            if active_idx.numel() == K and active_idx.min().item() == 0 and active_idx.max().item() == K - 1:
                true_active = true_acc
                valid_active = torch.ones_like(true_active, dtype=torch.bool)
            else:
                mapper = torch.full((C,), -1, dtype=torch.long, device=active_idx.device)
                mapper[active_idx] = torch.arange(K, device=active_idx.device)
                true_active = mapper[true_acc]
                valid_active = true_active >= 0
            if valid_active.any():
                logits_active = logits_active[valid_active]
                true_active = true_active[valid_active]
            else:
                logits_active = logits_active[:0]
                true_active = true_active[:0]
        else:
            logits_active = logits_acc
            true_active = true_acc

        with torch.no_grad():
            if logits_active.numel() > 0 and true_active.numel() > 0:
                acc_val = (logits_active.argmax(dim=1) == true_active).float().mean().item()
            else:
                acc_val = 0.0

            micro_results = {
                "ce": scaled_loss.item(),
                "accuracy": acc_val / num_micro_batches,
            }
        return micro_results


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = build_parser()
    parser = _extend_parser(parser)
    cfg = parser.parse_args()
    if bool(getattr(cfg, "force_flash_sdpa", False)) and torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            print("Forced SDPA backend: flash only")
        except Exception as exc:
            print(f"Failed to force flash SDPA backend: {exc}")
    # Log SDPA backend settings and priority order for runtime verification.
    if torch.cuda.is_available():
        try:
            flash = torch.backends.cuda.flash_sdp_enabled()
            mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
            math = torch.backends.cuda.math_sdp_enabled()
            cudnn = torch.backends.cuda.cudnn_sdp_enabled()
            priority = list(torch._C._get_sdp_priority_order())
            backend_names = {
                0: "math",
                1: "flash",
                2: "mem_efficient",
                3: "cudnn",
                4: "overrideable",
            }
            priority_named = [backend_names.get(int(v), str(v)) for v in priority]
            print(
                "SDPA backends - flash: %s, mem_efficient: %s, math: %s, cudnn: %s"
                % (flash, mem_eff, math, cudnn)
            )
            print("SDPA priority order: %s" % (" -> ".join(priority_named)))
        except Exception as exc:
            print(f"SDPA backend query failed: {exc}")
    try:
        import cupy as cp

        device_count = int(cp.cuda.runtime.getDeviceCount())
        device_id = int(cp.cuda.runtime.getDevice())
        print(f"CuPy available. CUDA devices: {device_count}, current: {device_id}")
    except Exception as exc:
        print(f"CuPy unavailable or no CUDA device: {exc}")
    trainer = Trainer(cfg)
    trainer.train()

# train/runMantis_adapter_plus_orion_icl.py
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


# Ensure we import the local workspace package (repo_root/code/src)
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_DEFAULT_TICFM_PRETRAINED_CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "checkpoints" / "mantis_orion_icl_full.pt"
)

from orion_msp.model.learning import ICLearning
from orion_msp.model.mantis_adapter_plus_orion_icl import _MantisAdapterPlusOrionICL
from orion_msp.prior.dataset import PriorDataset
from orion_msp.prior.genload import LoadPriorDataset
from orion_msp.train.run import Trainer as _BaseTrainer
from orion_msp.train.run import _prior_worker_init_fn
from orion_msp.train.train_config import build_parser


from tabicl.model.mantis_adapter_icl import TokenMLPAdapter
from tabicl.model.mantis_tabicl import build_mantis_encoder, encode_with_mantis


def _extend_parser(parser):
    # cauker_icl task replay / episode pool
    parser.add_argument("--icl_task_pool_size", type=int, default=0, help="Replay task-pool size. 0 disables task replay.")
    parser.add_argument("--icl_replay_prob", type=float, default=0.0, help="Probability of replaying an old task from the pool.")
    parser.add_argument(
        "--icl_replay_warmup_steps",
        type=int,
        default=0,
        help="If > 0, linearly schedule replay probability during the first N batch steps.",
    )
    parser.add_argument(
        "--icl_replay_prob_start",
        type=float,
        default=-1.0,
        help="Warmup start replay probability. If < 0, falls back to --icl_replay_prob.",
    )
    parser.add_argument(
        "--icl_replay_prob_end",
        type=float,
        default=-1.0,
        help="Warmup end replay probability. If < 0, falls back to --icl_replay_prob.",
    )
    parser.add_argument(
        "--icl_task_pool_mode",
        type=str,
        default="episode",
        choices=["episode", "payload"],
        help="Replay full episodes or just payload/task configs.",
    )
    parser.add_argument(
        "--icl_pool_replace",
        type=str,
        default="fifo",
        choices=["fifo", "random"],
        help="Replacement policy once the replay pool is full.",
    )
    parser.add_argument(
        "--icl_replay_debug_every",
        type=int,
        default=100,
        help="Print replay hit/pool stats every N prior batches on rank0/worker0.",
    )

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

    # Overfit-one-episode debug mode
    parser.add_argument(
        "--overfit_one_episode",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Debug mode: cache one episode and train on it for N steps.",
    )
    parser.add_argument("--overfit_steps", type=int, default=300, help="Number of optimizer steps in overfit mode.")
    parser.add_argument(
        "--overfit_seed",
        type=int,
        default=1234,
        help="Seed used to generate the cached episode (rank0 generates then broadcasts under DDP).",
    )
    parser.add_argument(
        "--overfit_freeze_prior",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, do not call the prior again after caching the first episode.",
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

    def configure_prior(self):
        if self.config.prior_dir is None:
            dataset = PriorDataset(
                batch_size=self.config.batch_size,
                batch_size_per_gp=self.config.batch_size_per_gp,
                min_features=self.config.min_features,
                max_features=self.config.max_features,
                max_classes=self.config.max_classes,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                log_seq_len=self.config.log_seq_len,
                seq_len_per_gp=self.config.seq_len_per_gp,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
                replay_small=self.config.replay_small,
                prior_type=self.config.prior_type,
                icl_k=getattr(self.config, "icl_k", 5),
                icl_time_length=getattr(self.config, "icl_time_length", 512),
                icl_num_features=getattr(self.config, "icl_num_features", 6),
                icl_single_channel=getattr(self.config, "icl_single_channel", False),
                icl_level=getattr(self.config, "icl_level", 0),
                icl_num_nodes=getattr(self.config, "icl_num_nodes", 18),
                icl_max_parents=getattr(self.config, "icl_max_parents", 5),
                icl_max_lag=getattr(self.config, "icl_max_lag", 5),
                icl_feature_mode=getattr(self.config, "icl_feature_mode", "mean"),
                icl_base_seed=getattr(self.config, "icl_base_seed", 42),
                icl_episode_workers=getattr(self.config, "icl_episode_workers", 1),
                icl_pool_backend=getattr(self.config, "icl_pool_backend", "thread"),
                icl_program_pool_size=getattr(self.config, "icl_program_pool_size", 0),
                icl_task_pool_size=getattr(self.config, "icl_task_pool_size", 0),
                icl_replay_prob=getattr(self.config, "icl_replay_prob", 0.0),
                icl_replay_warmup_steps=getattr(self.config, "icl_replay_warmup_steps", 0),
                icl_replay_prob_start=getattr(self.config, "icl_replay_prob_start", -1.0),
                icl_replay_prob_end=getattr(self.config, "icl_replay_prob_end", -1.0),
                icl_task_pool_mode=getattr(self.config, "icl_task_pool_mode", "episode"),
                icl_pool_replace=getattr(self.config, "icl_pool_replace", "fifo"),
                icl_replay_debug_every=getattr(self.config, "icl_replay_debug_every", 100),
                icl_show_progress=getattr(self.config, "icl_show_progress", False),
                device=self.config.prior_device,
                n_jobs=1,
            )
        else:
            dataset = LoadPriorDataset(
                data_dir=self.config.prior_dir,
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

        if self.master_process:
            print(dataset)
        prior_num_workers = max(0, int(getattr(self.config, "prior_num_workers", 1)))
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": None,
            "shuffle": False,
            "num_workers": prior_num_workers,
            "pin_memory": True if self.config.prior_device == "cpu" else False,
        }
        if prior_num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = max(2, int(getattr(self.config, "prior_prefetch_factor", 4)))
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["multiprocessing_context"] = "spawn"
            dataloader_kwargs["worker_init_fn"] = _prior_worker_init_fn

        self.dataloader = DataLoader(**dataloader_kwargs)

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

    @staticmethod
    def _set_global_seed(seed: int) -> None:
        try:
            import numpy as _np

            _np.random.seed(int(seed))
        except Exception:
            pass
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _broadcast_cached_batch(self, batch_tensors: list[Tensor]) -> list[Tensor]:
        """DDP: broadcast a list of tensors from rank0 to all ranks.

        Uses one broadcast_object_list for metadata, then dist.broadcast per tensor.
        """
        if not self.ddp:
            return batch_tensors
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return batch_tensors

        rank = int(getattr(self, "ddp_rank", 0))
        device = torch.device(self.config.device)

        meta: list[dict] | None
        if rank == 0:
            meta = [
                {
                    "shape": tuple(t.shape),
                    "dtype": str(t.dtype),
                    "requires_grad": bool(t.requires_grad),
                }
                for t in batch_tensors
            ]
        else:
            meta = None

        obj_list = [meta]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        meta = obj_list[0]
        assert isinstance(meta, list) and all(isinstance(m, dict) for m in meta), "Invalid broadcast meta"

        # Map dtype string back to torch.dtype
        str_to_dtype = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.int8": torch.int8,
            "torch.uint8": torch.uint8,
            "torch.int16": torch.int16,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "torch.bool": torch.bool,
        }

        out: list[Tensor] = []
        for i, m in enumerate(meta):
            shape = tuple(m["shape"])
            dtype = str_to_dtype.get(str(m["dtype"]), None)
            if dtype is None:
                raise ValueError(f"Unsupported dtype in broadcast meta: {m['dtype']}")

            if rank == 0:
                t = batch_tensors[i].to(device=device)
            else:
                t = torch.empty(shape, dtype=dtype, device=device)

            torch.distributed.broadcast(t, src=0)
            out.append(t)

        return out

    def make_cached_episode(self, seed: int) -> list[Tensor]:
        """Generate one episode batch and cache it (prefer GPU) for overfit debugging."""
        # DDP: only rank0 generates; others receive via broadcast.
        if self.ddp and torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(getattr(self, "ddp_rank", 0))
            if rank != 0:
                return self._broadcast_cached_batch([])

        # Force deterministic episode generation as much as possible.
        self._set_global_seed(int(seed))

        # Try to also pin CaukerICLPrior seed/cursor if present.
        ds = getattr(self.dataloader, "dataset", None)
        if ds is not None:
            try:
                if hasattr(ds, "icl_base_seed"):
                    ds.icl_base_seed = int(seed)
                if hasattr(ds, "_task_cursor"):
                    ds._task_cursor = 0
            except Exception:
                pass

        iterator = iter(self.dataloader)
        batch = next(iterator)
        # Normalize nested tensors to padded tensors (same as base run_batch).
        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
        # Move to training device once to avoid per-step H2D copies.
        device = torch.device(self.config.device)
        batch_dev = [t.to(device=device, non_blocking=True) for t in batch]

        # DDP: ensure all ranks share the exact same cached episode.
        batch_dev = self._broadcast_cached_batch(batch_dev)
        return batch_dev

    @staticmethod
    def _grad_norm_from_params(params: list[Tensor]) -> float:
        sq_sum = None
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            # ensure float for stable accumulation
            if not torch.is_floating_point(g):
                continue
            v = g.float().pow(2).sum()
            sq_sum = v if sq_sum is None else (sq_sum + v)
        if sq_sum is None:
            return 0.0
        return float(torch.sqrt(sq_sum).item())

    def _pick_param_to_track(self) -> tuple[str, Tensor] | None:
        # Prefer a parameter in the ICL predictor, otherwise first trainable param.
        candidates: list[tuple[str, Tensor]] = []
        for name, p in self.raw_model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2:
                continue
            candidates.append((name, p))
        if not candidates:
            return None

        for name, p in candidates:
            if "icl_predictor" in name:
                return name, p
        return candidates[0]

    def train_step_on_batch(self, batch: list[Tensor]) -> dict[str, float]:
        """One optimizer step on a provided batch. Returns metrics for debugging."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Ensure non-nested, and keep everything on device.
        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        # Split into micro-batches (same logic as base Trainer.run_batch).
        splits = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch]
        all_micros = list(zip(*splits))

        valid_micros = []
        for mb in all_micros:
            _, _, _, micro_seq_len, micro_train_size = mb
            seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
            if seq_len > train_size:
                valid_micros.append(mb)

        num_micro_batches = len(valid_micros)
        if num_micro_batches == 0:
            self.scheduler.step()
            return {
                "ce": 0.0,
                "accuracy": 0.0,
                "grad_norm": 0.0,
                "grad_norm_encoder": 0.0,
                "grad_norm_adapter": 0.0,
                "grad_norm_icl": 0.0,
                "lr": float(self.scheduler.get_last_lr()[0]),
            }

        results = {"ce": 0.0, "accuracy": 0.0}
        failed = 0
        for i, micro in enumerate(valid_micros):
            try:
                res = self.run_micro_batch(micro, i, num_micro_batches)
                for k, v in res.items():
                    results[k] = results.get(k, 0.0) + float(v)
            except torch.cuda.OutOfMemoryError:
                if self.master_process:
                    print(f"[overfit] OOM in micro-batch {i+1}/{num_micro_batches} at step {self.curr_step}. Skipping.")
                torch.cuda.empty_cache()
                failed += 1
                continue
            except FloatingPointError:
                if self.master_process:
                    print(f"[overfit] Non-finite loss in micro-batch {i+1}/{num_micro_batches} at step {self.curr_step}. Skipping.")
                failed += 1
                continue

        if failed / max(1, len(valid_micros)) > 0.1:
            raise RuntimeError("Too many failed micro-batches in overfit mode.")

        # Unscale grads before measuring/optional clipping.
        if bool(getattr(self, "amp", False)):
            try:
                self.scaler.unscale_(self.optimizer)
            except Exception:
                pass

        # Grad norms: total + key modules.
        total_gn = self._grad_norm_from_params([p for p in self.raw_model.parameters() if p.requires_grad])
        enc = getattr(self.raw_model, "mantis_model", None)
        adapter = getattr(self.raw_model, "adapter", None)
        icl = getattr(self.raw_model, "icl_predictor", None)
        gn_enc = self._grad_norm_from_params([p for p in enc.parameters() if p.requires_grad]) if enc is not None else 0.0
        gn_adp = self._grad_norm_from_params([p for p in adapter.parameters() if p.requires_grad]) if adapter is not None else 0.0
        gn_icl = self._grad_norm_from_params([p for p in icl.parameters() if p.requires_grad]) if icl is not None else 0.0

        # Optional clipping (mirror base logic).
        if self.config.gradient_clipping > 0:
            total_norm_clip = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            if not torch.isfinite(total_norm_clip):
                if self.master_process:
                    print(f"[overfit] Non-finite grad norm at step {self.curr_step}; skipping optimizer step.")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.scheduler.step()
                results.update(
                    {
                        "grad_norm": float(total_gn),
                        "grad_norm_encoder": float(gn_enc),
                        "grad_norm_adapter": float(gn_adp),
                        "grad_norm_icl": float(gn_icl),
                        "lr": float(self.scheduler.get_last_lr()[0]),
                    }
                )
                return results

        # Step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        results.update(
            {
                "grad_norm": float(total_gn),
                "grad_norm_encoder": float(gn_enc),
                "grad_norm_adapter": float(gn_adp),
                "grad_norm_icl": float(gn_icl),
                "lr": float(self.scheduler.get_last_lr()[0]),
            }
        )
        return results

    def train_overfit_one_episode(self) -> None:
        """Run overfit-one-episode debug loop and exit."""
        overfit_steps = int(getattr(self.config, "overfit_steps", 300))
        seed = int(getattr(self.config, "overfit_seed", 1234))
        freeze_prior = bool(getattr(self.config, "overfit_freeze_prior", True))

        if self.master_process:
            print(
                f"[overfit] mode=one_episode steps={overfit_steps} seed={seed} freeze_prior={freeze_prior} ddp={self.ddp}",
                flush=True,
            )

        cached = self.make_cached_episode(seed)
        self._overfit_cached_batch = cached

        tracked = self._pick_param_to_track()
        track_name = None
        track_prev = None
        if tracked is not None:
            track_name, track_param = tracked
            track_prev = track_param.detach().float().cpu().clone()
            if self.master_process:
                print(f"[overfit] tracking param_delta on: {track_name}", flush=True)
        else:
            if self.master_process:
                print("[overfit] warning: no trainable parameter found for param_delta tracking", flush=True)

        delta_every = 50
        last_ce: list[float] = []
        last_acc: list[float] = []
        last_gn: list[float] = []
        last_delta: list[float] = []

        # Compute and print episode-level K + Nq_valid from the first micro-batch (no forward needed).
        try:
            X0, y0, _d0, _sl0, ts0 = cached
            train_size0 = int(ts0[0].item())
            y_train0 = y0[: self.config.micro_batch_size, :train_size0].long()
            y_test0 = y0[: self.config.micro_batch_size, train_size0:].long()
            # C here is model head size (model_max_classes). We conservatively infer it from config.
            Cmax = int(getattr(self.config, "model_max_classes", getattr(self.config, "max_classes", 0)))
            active_idx0 = torch.unique(y_train0).sort().values.long()
            active_idx0 = active_idx0[(active_idx0 >= 0) & (active_idx0 < Cmax)]
            K0 = int(active_idx0.numel())
            valid0 = (y_test0.reshape(-1) >= 0) & (y_test0.reshape(-1) < Cmax)
            Nq_valid0 = int(valid0.long().sum().item())
            if self.master_process:
                print(f"[overfit] episode debug: K={K0} active_idx={active_idx0.detach().cpu().tolist()} Nq_valid={Nq_valid0}", flush=True)
        except Exception:
            pass

        start_time = time.time()
        for i in range(overfit_steps):
            # Optionally regenerate each step (slow path). Default is cached reuse.
            if not freeze_prior:
                batch = self.make_cached_episode(seed)
            else:
                batch = cached

            metrics = self.train_step_on_batch(batch)
            ce = float(metrics.get("ce", 0.0))
            acc = float(metrics.get("accuracy", 0.0))
            gn = float(metrics.get("grad_norm", 0.0))

            param_delta = float("nan")
            if track_name is not None and track_prev is not None and ((i + 1) % delta_every == 0 or i == 0):
                p_now = dict(self.raw_model.named_parameters()).get(track_name, None)
                if p_now is not None:
                    now_cpu = p_now.detach().float().cpu()
                    param_delta = float(torch.norm(now_cpu - track_prev).item())
                    track_prev = now_cpu.clone()
                    last_delta.append(param_delta)

            # Keep last-10 stats
            last_ce.append(ce)
            last_acc.append(acc)
            last_gn.append(gn)
            if len(last_ce) > 10:
                last_ce.pop(0)
                last_acc.pop(0)
                last_gn.pop(0)

            if self.master_process:
                lr = float(metrics.get("lr", 0.0))
                msg = (
                    f"[overfit] step={i+1}/{overfit_steps} ce={ce:.4f} acc={acc:.4f} "
                    f"gn={gn:.3e} (enc={metrics.get('grad_norm_encoder', 0.0):.3e}, "
                    f"adp={metrics.get('grad_norm_adapter', 0.0):.3e}, icl={metrics.get('grad_norm_icl', 0.0):.3e}) "
                    f"lr={lr:.3e}"
                )
                if (i + 1) % delta_every == 0 or i == 0:
                    msg += f" param_delta={param_delta:.3e}"
                print(msg, flush=True)

            self.curr_step += 1

        # Summary + diagnosis
        mean_ce = float(sum(last_ce) / max(1, len(last_ce)))
        mean_acc = float(sum(last_acc) / max(1, len(last_acc)))
        mean_gn = float(sum(last_gn) / max(1, len(last_gn)))
        elapsed = time.time() - start_time

        if self.master_process:
            print(
                f"[overfit] done in {elapsed:.1f}s. last10_mean_ce={mean_ce:.4f} last10_mean_acc={mean_acc:.4f} last10_mean_gn={mean_gn:.3e}",
                flush=True,
            )
            success = (mean_acc >= 0.9) or (mean_ce < 0.8)
            if success:
                print("[overfit] SUCCESS: model can overfit this fixed episode.", flush=True)
            else:
                print("[overfit] FAIL: acc/ce did not improve enough on a fixed episode.", flush=True)
                print("[overfit] Diagnosis hints:", flush=True)
                print("  - optimizer.step not happening / scaler skipping steps", flush=True)
                print("  - grads are ~0 (frozen params, detached loss, wrong requires_grad)", flush=True)
                print("  - loss not connected to logits (masking/indexing bug)", flush=True)
                print("  - episode construction prevents learning (query labels mismatch / context leakage issues)", flush=True)
                print(
                    f"  - observed: mean_grad_norm={mean_gn:.3e}, last_param_delta_samples={last_delta[-5:] if last_delta else []}",
                    flush=True,
                )

        # Make sure all ranks exit together under DDP.
        if self.ddp and torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
        raise SystemExit(0)

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

            # Flatten + valid-filter (must match accuracy branch semantics).
            logits_acc = logits.reshape(-1, C)
            true_acc = y_test.reshape(-1).long()
            valid = (true_acc >= 0) & (true_acc < C)
            if not torch.all(valid):
                logits_acc = logits_acc[valid]
                true_acc = true_acc[valid]
            if true_acc.numel() == 0:
                return {"ce": 0.0, "accuracy": 0.0}

            # Active classes come from the context (y_train).
            active_idx = torch.unique(y_train).sort().values.long()
            active_idx = active_idx[(active_idx >= 0) & (active_idx < C)]
            K = int(active_idx.numel())
            if K == 0:
                raise ValueError(
                    f"Empty active classes (K=0) after filtering to [0, C). "
                    f"C={C}, step={int(self.curr_step) + 1}, train_size={train_size}."
                )

            logits_active = logits_acc.index_select(dim=-1, index=active_idx)
            if active_idx.min().item() == 0 and active_idx.max().item() == K - 1:
                true_active = true_acc
            else:
                mapper = torch.full((C,), -1, dtype=torch.long, device=active_idx.device)
                mapper[active_idx] = torch.arange(K, device=active_idx.device)
                true_active = mapper[true_acc]

                # Default behavior (方案A): fail fast if any query label is missing in context.
                if not torch.all(true_active >= 0):
                    missing_labels = torch.unique(true_acc[true_active < 0]).detach().cpu().tolist()
                    active_labels = active_idx.detach().cpu().tolist()
                    step_idx = int(self.curr_step) + 1
                    if self.master_process:
                        print(
                            "[active-loss] Found query labels not in active_idx; episode may be malformed. "
                            f"step={step_idx} train_size={train_size} C={C} K={K}",
                            flush=True,
                        )
                        print(f"  active_idx: {active_labels}", flush=True)
                        print(f"  missing_labels: {missing_labels}", flush=True)
                        try:
                            diff = sorted(set(missing_labels) - set(active_labels))
                            print(f"  missing_labels \\ active_idx: {diff}", flush=True)
                        except Exception:
                            pass
                    raise ValueError(
                        "Query labels not present in context active classes. "
                        f"missing_labels={missing_labels}, active_idx={active_labels}, "
                        f"step={step_idx}, train_size={train_size}"
                    )

            # Debug log (DDP-safe): gate prints to master + periodic steps.
            debug_every = int(getattr(self.config, "debug_active_loss_every", 100))
            step_idx = int(self.curr_step) + 1
            if self.master_process and debug_every > 0 and (step_idx % debug_every == 0):
                ta_min = int(true_active.min().item()) if true_active.numel() else None
                ta_max = int(true_active.max().item()) if true_active.numel() else None
                active_list = active_idx.detach().cpu().tolist()
                if len(active_list) > 64:
                    active_list = active_list[:64] + ["..."]
                print(
                    "[active-loss] "
                    f"step={step_idx} K={K} active_idx={active_list} "
                    f"logits_active={tuple(logits_active.shape)} true_active_min/max={ta_min}/{ta_max}",
                    flush=True,
                )

            # Loss aligned to active classes (context K classes).
            loss = F.cross_entropy(logits_active, true_active)

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        if micro_batch_idx == num_micro_batches - 1:
            self._debug_log_missing_grads()

        # Accuracy: reuse the exact same active-class tensors as the loss.
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
    if bool(getattr(cfg, "overfit_one_episode", False)):
        trainer.train_overfit_one_episode()
    trainer.train()

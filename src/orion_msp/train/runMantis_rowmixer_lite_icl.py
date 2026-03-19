from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP


_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from orion_msp.model.mantis_plus_rowmixer_lite_icl import _MantisPlusRowMixerLiteICL
from orion_msp.model.rowmixer_lite_icl import RowMixerLiteICL
from orion_msp.train.run import Trainer as _BaseTrainer
from orion_msp.train.runMantis_adapter_plus_orion_icl import Trainer as _SharedTrainer
from orion_msp.train.runMantis_adapter_plus_orion_icl import _extend_parser
from orion_msp.train.optim import Muon, get_scheduler
from orion_msp.train.train_config import build_parser
from tabicl.model.mantis_adapter_icl import TokenMLPAdapter
from tabicl.model.mantis_tabicl import build_mantis_encoder

from orion_msp.model.learning import ICLearning
from orion_msp.model.mantis_adapter_plus_orion_icl import _MantisAdapterPlusOrionICL


def should_use_adamw(name: str, p: torch.nn.Parameter) -> bool:
    """Route non-2D, IO heads, and label-injection parameters to AdamW fallback."""
    lname = name.lower()

    # 1) All non-2D tensors go to AdamW.
    if p.ndim != 2:
        return True

    # 2) Input-facing layers go to AdamW.
    input_keywords = [
        "patch_embed",
        "token_embed",
        "input_proj",
        "in_proj",
        "conv",
        "embed_tokens",
        "embedding",
    ]
    if any(k in lname for k in input_keywords):
        return True

    # 3) Label / one-hot injection layers go to AdamW.
    label_keywords = [
        "label",
        "onehot",
        "one_hot",
        "ey",
        "target_embed",
        "embedtae",
    ]
    if any(k in lname for k in label_keywords):
        return True

    # 4) Output-facing layers go to AdamW.
    output_keywords = [
        "head",
        "classifier",
        "predictor",
        "out_proj",
        "lm_head",
        "output",
    ]
    if any(k in lname for k in output_keywords):
        return True

    return False


def build_muon_optimizer(model, lr: float, wd: float = 0.1):
    """Build Muon+AdamW mixed optimizer from model named parameters."""
    muon_params = []
    adamw_params = []
    muon_names = []
    adamw_names = []

    module_counts = {
        "mantis_model": {"muon": 0, "adamw": 0},
        "rowmixer_lite": {"muon": 0, "adamw": 0},
        "icl_predictor": {"muon": 0, "adamw": 0},
        "other": {"muon": 0, "adamw": 0},
    }

    def _bucket(param_name: str) -> str:
        lname = param_name.lower()
        if lname.startswith("mantis_model."):
            return "mantis_model"
        if "icl_predictor" in lname:
            return "icl_predictor"
        if lname.startswith("rowmixer_icl."):
            return "rowmixer_lite"
        return "other"

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if should_use_adamw(name, p):
            adamw_params.append(p)
            adamw_names.append(name)
            module_counts[_bucket(name)]["adamw"] += 1
        else:
            muon_params.append(p)
            muon_names.append(name)
            module_counts[_bucket(name)]["muon"] += 1

    print(f"[Muon] num_muon_tensors={len(muon_params)} num_adamw_tensors={len(adamw_params)}")
    print(f"[Muon sample params] {muon_names[:20]}")
    print(f"[AdamW sample params] {adamw_names[:20]}")

    for module_name in ("mantis_model", "rowmixer_lite", "icl_predictor", "other"):
        c = module_counts[module_name]
        print(f"[Muon group] {module_name}: muon={c['muon']} adamw={c['adamw']}")

    label_keywords = ("label", "onehot", "one_hot", "ey", "target_embed", "embedtae")
    output_keywords = ("head", "classifier", "predictor", "out_proj", "lm_head", "output")
    label_adamw = [n for n in adamw_names if any(k in n.lower() for k in label_keywords)]
    adapter_muon = [n for n in muon_names if "adapter" in n.lower()]
    icl_muon = [n for n in muon_names if "icl_predictor" in n.lower() and n.lower().endswith("weight")]
    output_adamw = [n for n in adamw_names if any(k in n.lower() for k in output_keywords)]

    print(f"[Muon check] label/one-hot -> AdamW: {len(label_adamw)} sample={label_adamw[:8]}")
    print(f"[Muon check] adapter hidden -> Muon: {len(adapter_muon)} sample={adapter_muon[:8]}")
    print(f"[Muon check] icl_predictor 2D -> Muon: {len(icl_muon)} sample={icl_muon[:8]}")
    print(f"[Muon check] predictor/head/output -> AdamW: {len(output_adamw)} sample={output_adamw[:8]}")

    if not muon_params:
        print("[Muon][WARN] Muon parameter group is empty.")
    if not adamw_params:
        print("[Muon][WARN] AdamW parameter group is empty.")

    optimizer = Muon(
        lr=lr,
        wd=wd,
        muon_params=muon_params,
        adamw_params=adamw_params,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    )
    return optimizer


class Trainer(_SharedTrainer):
    """Reuse the Mantis end-to-end training loop, but swap the predictor to RowMixerLiteICL."""

    def __init__(self, config):
        self._did_shape_assert = False
        _BaseTrainer.__init__(self, config)
        self._configure_mantis_eval()
        self._configure_ref_eval()

    def configure_optimizer(self):
        target_model = self.raw_model if hasattr(self, "raw_model") else self.model
        self.optimizer = build_muon_optimizer(
            model=target_model,
            lr=float(self.config.lr),
            wd=0.1,
        )
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def align_micro_batch(self, micro_X: Tensor, micro_y: Tensor, micro_d: Tensor, seq_len: int):
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]
        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]

        target = int(getattr(self.config, "mantis_seq_len", 512))
        feat_dim = int(micro_X.shape[-1])
        if feat_dim > target:
            micro_X = micro_X[..., :target]
        elif feat_dim < target:
            pad = micro_X.new_zeros(*micro_X.shape[:-1], target - feat_dim)
            micro_X = torch.cat([micro_X, pad], dim=-1)
        return micro_X, micro_y

    @staticmethod
    def _clean_state_dict(state_dict: dict) -> dict:
        cleaned = {}
        for key, value in state_dict.items():
            name = str(key)
            if name.startswith("module."):
                name = name[len("module.") :]
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod.") :]
            cleaned[name] = value
        return cleaned

    @staticmethod
    def _extract_state_dict(ckpt_obj: object) -> dict:
        if isinstance(ckpt_obj, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = ckpt_obj.get(key)
                if isinstance(value, dict):
                    return value
            if all(isinstance(k, str) for k in ckpt_obj.keys()):
                return ckpt_obj
        raise ValueError("Unsupported checkpoint format; expected a dict or state_dict.")

    def _load_rowmixer_init_checkpoint(self, rowmixer_icl: RowMixerLiteICL) -> None:
        ckpt_path = str(getattr(self.config, "rowmixer_init_checkpoint", "") or "").strip()
        if not ckpt_path:
            return

        ckpt_file = Path(ckpt_path).expanduser()
        if not ckpt_file.is_absolute():
            ckpt_file = Path.cwd() / ckpt_file
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"RowMixerLiteICL init checkpoint not found: {ckpt_file}")

        ckpt_obj = torch.load(str(ckpt_file), map_location="cpu")
        full_state = self._clean_state_dict(self._extract_state_dict(ckpt_obj))
        module_state = rowmixer_icl.state_dict()

        candidate_prefixes = (
            "rowmixer_icl.",
            "model.rowmixer_icl.",
            "raw_model.rowmixer_icl.",
            "module.rowmixer_icl.",
            "_orig_mod.rowmixer_icl.",
        )

        state_to_load = {k: v for k, v in full_state.items() if k in module_state}
        if not state_to_load:
            for prefix in candidate_prefixes:
                stripped = {
                    k[len(prefix) :]: v
                    for k, v in full_state.items()
                    if k.startswith(prefix) and k[len(prefix) :] in module_state
                }
                if stripped:
                    state_to_load = stripped
                    break

        if not state_to_load:
            raise RuntimeError(
                "No RowMixerLiteICL weights matched the checkpoint. "
                f"checkpoint={ckpt_file}"
            )

        incompatible = rowmixer_icl.load_state_dict(
            state_to_load,
            strict=bool(getattr(self.config, "rowmixer_init_strict", False)),
        )
        if self.master_process:
            print(f"Initialized RowMixerLiteICL from checkpoint: {ckpt_file}")
            print(f"  Loaded tensors: {len(state_to_load)}")
            if incompatible is not None:
                if getattr(incompatible, "missing_keys", None):
                    print(f"  Missing keys: {len(incompatible.missing_keys)}")
                if getattr(incompatible, "unexpected_keys", None):
                    print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")

    def _build_single_channel_mantis(self, *, mantis_checkpoint=None, hidden_dim: int, seq_len: int, num_patches: int, use_fddm: bool):
        return build_mantis_encoder(
            mantis_checkpoint=mantis_checkpoint,
            device=self.config.device,
            hidden_dim=int(hidden_dim),
            seq_len=int(seq_len),
            num_patches=int(num_patches),
            use_fddm=bool(use_fddm),
            num_channels=1,
            strict=False,
        )

    def _encode_rows_with_mantis_model(
        self,
        mantis_model,
        X: Tensor,
        *,
        mantis_seq_len: int | None = None,
        mantis_batch_size: int | None = None,
    ) -> Tensor:
        target_len = int(mantis_seq_len or getattr(self.config, "mantis_seq_len", 512))
        batch_size = int(mantis_batch_size or getattr(self.config, "mantis_batch_size", 256))
        batch_size = max(1, batch_size)

        X = self._pad_or_truncate_features(X.to(self.config.device), target_len)

        if X.ndim == 3:
            B, T, L = X.shape
            x_flat = X.reshape(B * T, 1, L)
            multichannel = False
            channels = 1
        elif X.ndim == 4:
            B, T, channels, L = X.shape
            x_flat = X.reshape(B * T * channels, 1, L)
            multichannel = True
        else:
            raise ValueError(f"Expected X to be (B,T,L) or (B,T,C,L), got {tuple(X.shape)}")

        outs = []
        for i in range(0, int(x_flat.shape[0]), batch_size):
            outs.append(mantis_model(x_flat[i : i + batch_size]))
        reps = torch.cat(outs, dim=0)

        if not multichannel:
            return reps.reshape(B, T, -1)

        reps = reps.reshape(B, T, channels, -1)
        return reps.reshape(B, T, channels * reps.shape[-1])

    def _configure_mantis_eval(self) -> None:
        self.mantis_eval_enabled = bool(getattr(self.config, "mantis_eval_checkpoint", None))
        self.mantis_eval_model = None
        if not self.mantis_eval_enabled:
            return
        if not self.master_process:
            return

        self.mantis_eval_model = self._build_single_channel_mantis(
            mantis_checkpoint=getattr(self.config, "mantis_eval_checkpoint", None),
            hidden_dim=int(getattr(self.config, "mantis_hidden_dim", 512)),
            seq_len=int(getattr(self.config, "mantis_seq_len", 512)),
            num_patches=int(getattr(self.config, "mantis_num_patches", 32)),
            use_fddm=bool(getattr(self.config, "mantis_use_fddm", False)),
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

        if isinstance(model_cfg, dict) and "rowmixer_icl" in model_cfg and "mantis" in model_cfg:
            mantis_cfg = dict(model_cfg.get("mantis", {}))
            row_cfg = dict(model_cfg.get("rowmixer_icl", {}))
            mantis_seq_len = int(mantis_cfg.get("seq_len", getattr(self.config, "mantis_seq_len", 512)))
            mantis_batch_size = int(mantis_cfg.get("batch_size", getattr(self.config, "mantis_batch_size", 256)))
            mantis_model = self._build_single_channel_mantis(
                mantis_checkpoint=mantis_cfg.get("ckpt", None),
                hidden_dim=int(mantis_cfg.get("hidden_dim", 512)),
                seq_len=mantis_seq_len,
                num_patches=int(mantis_cfg.get("num_patches", 32)),
                use_fddm=bool(mantis_cfg.get("use_fddm", False)),
            )
            rowmixer_icl = RowMixerLiteICL(**row_cfg).to(self.config.device)
            ref_model = _MantisPlusRowMixerLiteICL(
                mantis_model=mantis_model,
                rowmixer_icl=rowmixer_icl,
                mantis_seq_len=mantis_seq_len,
                mantis_batch_size=mantis_batch_size,
            ).to(self.config.device)
        elif isinstance(model_cfg, dict) and {"mantis", "adapter", "icl_predictor"}.issubset(model_cfg.keys()):
            mantis_cfg = model_cfg["mantis"]
            adapter_cfg = model_cfg["adapter"]
            icl_cfg = model_cfg["icl_predictor"]

            mantis_model = self._build_single_channel_mantis(
                mantis_checkpoint=mantis_cfg.get("ckpt", None),
                hidden_dim=int(mantis_cfg.get("hidden_dim", 512)),
                seq_len=int(mantis_cfg.get("seq_len", 512)),
                num_patches=int(mantis_cfg.get("num_patches", 32)),
                use_fddm=bool(mantis_cfg.get("use_fddm", False)),
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
            ref_model = _MantisAdapterPlusOrionICL(
                mantis_model=mantis_model,
                adapter=adapter,
                icl_predictor=icl_predictor,
                mantis_seq_len=int(mantis_cfg.get("seq_len", 512)),
                mantis_batch_size=int(mantis_cfg.get("batch_size", getattr(self.config, "mantis_batch_size", 256))),
            ).to(self.config.device)
        else:
            mantis_seq_len = int(getattr(self.config, "mantis_seq_len", 512))
            mantis_batch_size = int(getattr(self.config, "mantis_batch_size", 256))
            mantis_model = self._build_single_channel_mantis(
                mantis_checkpoint=None,
                hidden_dim=int(getattr(self.config, "mantis_hidden_dim", 512)),
                seq_len=mantis_seq_len,
                num_patches=int(getattr(self.config, "mantis_num_patches", 32)),
                use_fddm=bool(getattr(self.config, "mantis_use_fddm", False)),
            )
            rowmixer_icl = RowMixerLiteICL(
                max_classes=int(getattr(self.config, "model_max_classes", self.config.max_classes)),
                embed_dim=int(self.config.embed_dim),
                patch_size=int(getattr(self.config, "rowmixer_patch_size", 8)),
                row_num_blocks=int(self.config.row_num_blocks),
                row_nhead=int(self.config.row_nhead),
                row_num_cls=int(self.config.row_num_cls),
                row_num_global=int(self.config.row_num_global),
                icl_num_blocks=int(self.config.icl_num_blocks),
                icl_nhead=int(self.config.icl_nhead),
                ff_factor=int(self.config.ff_factor),
                dropout=float(self.config.dropout),
                activation=str(self.config.activation),
                norm_first=bool(self.config.norm_first),
                shuffle_p=float(getattr(self.config, "rowmixer_shuffle_p", 0.25)),
                perc_num_latents=int(self.config.perc_num_latents),
                perc_layers=int(self.config.perc_layers),
            ).to(self.config.device)
            ref_model = _MantisPlusRowMixerLiteICL(
                mantis_model=mantis_model,
                rowmixer_icl=rowmixer_icl,
                mantis_seq_len=mantis_seq_len,
                mantis_batch_size=mantis_batch_size,
            ).to(self.config.device)

        if isinstance(ckpt_obj, dict):
            state_dict = ckpt_obj.get("state_dict") or ckpt_obj.get("model_state_dict") or ckpt_obj.get("model") or ckpt_obj
        else:
            state_dict = ckpt_obj
        ref_model.load_state_dict(self._clean_state_dict(state_dict), strict=False)
        ref_model.eval()
        self.ref_eval_model = ref_model

    def _eval_mantis_on_batch(self, batch) -> float | None:
        if not self.mantis_eval_enabled or self.mantis_eval_model is None:
            return None
        if not self.master_process:
            return None

        x, y, _d, seq_lens, train_sizes = batch
        if len(torch.unique(seq_lens)) > 1:
            return None

        x = x.to(self.config.device)
        y = y.to(self.config.device)
        with torch.no_grad():
            z = self._encode_rows_with_mantis_model(
                self.mantis_eval_model,
                x,
                mantis_seq_len=int(getattr(self.config, "mantis_seq_len", 512)),
                mantis_batch_size=int(getattr(self.config, "mantis_batch_size", 512)),
            )

        batch_accs = []
        B, T = z.shape[:2]
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

        x = x.to(self.config.device)
        y = y.to(self.config.device)

        batch_accs = []
        B, T = x.shape[:2]
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
                _, _, cmax = logits.shape
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

    def build_model(self):
        prior_type = str(getattr(self.config, "prior_type", ""))
        if prior_type not in {"mlp_scm", "tree_scm", "mix_scm", "cauker_icl", "ucr_uea_icl"}:
            raise ValueError(
                f"This trainer is intended for SCM synthetic data; got prior_type={prior_type!r}. "
                "Use --prior_type mlp_scm|tree_scm|mix_scm|cauker_icl|ucr_uea_icl."
            )

        model_max_classes = int(getattr(self.config, "model_max_classes", 10))
        prior_max_classes = int(getattr(self.config, "max_classes", model_max_classes))
        if model_max_classes < 1:
            raise ValueError(f"model_max_classes must be >= 1, got {model_max_classes}")
        if prior_type in {"cauker_icl", "ucr_uea_icl"}:
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
        mantis_init_checkpoint = getattr(self.config, "mantis_init_checkpoint", None)
        if mantis_init_checkpoint is not None:
            mantis_init_checkpoint = str(mantis_init_checkpoint).strip() or None

        mantis_model = build_mantis_encoder(
            mantis_checkpoint=mantis_init_checkpoint,
            device=self.config.device,
            hidden_dim=mantis_hidden_dim,
            seq_len=mantis_seq_len,
            num_patches=int(getattr(self.config, "mantis_num_patches", 32)),
            use_fddm=bool(getattr(self.config, "mantis_use_fddm", False)),
            num_channels=1,
            strict=False,
        )
        mantis_model.train()
        frozen_numel = self._freeze_known_unused_mantis_params(mantis_model)
        if self.master_process and frozen_numel > 0:
            print(f"Froze known-unused Mantis params: {frozen_numel:,} (pre_training projector)")

        rowmixer_icl = RowMixerLiteICL(
            max_classes=int(model_max_classes),
            embed_dim=int(self.config.embed_dim),
            patch_size=int(getattr(self.config, "rowmixer_patch_size", 8)),
            row_num_blocks=int(self.config.row_num_blocks),
            row_nhead=int(self.config.row_nhead),
            row_num_cls=int(self.config.row_num_cls),
            row_num_global=int(self.config.row_num_global),
            icl_num_blocks=int(self.config.icl_num_blocks),
            icl_nhead=int(self.config.icl_nhead),
            ff_factor=int(self.config.ff_factor),
            dropout=float(self.config.dropout),
            activation=str(self.config.activation),
            norm_first=bool(self.config.norm_first),
            shuffle_p=float(getattr(self.config, "rowmixer_shuffle_p", 0.25)),
            perc_num_latents=int(self.config.perc_num_latents),
            perc_layers=int(self.config.perc_layers),
        ).to(self.config.device)
        self._load_rowmixer_init_checkpoint(rowmixer_icl)

        model = _MantisPlusRowMixerLiteICL(
            mantis_model=mantis_model,
            rowmixer_icl=rowmixer_icl,
            mantis_seq_len=mantis_seq_len,
            mantis_batch_size=int(getattr(self.config, "mantis_batch_size", 256)),
        ).to(self.config.device)

        self.model_config = {
            "model": "_MantisPlusRowMixerLiteICL",
            "mantis": {
                "ckpt": mantis_init_checkpoint,
                "seq_len": mantis_seq_len,
                "hidden_dim": mantis_hidden_dim,
                "num_patches": int(getattr(self.config, "mantis_num_patches", 32)),
                "use_fddm": bool(getattr(self.config, "mantis_use_fddm", False)),
                "num_channels": 1,
                "batch_size": int(getattr(self.config, "mantis_batch_size", 256)),
                "per_channel_concat": True,
            },
            "rowmixer_icl": {
                "init_ckpt": (str(getattr(self.config, "rowmixer_init_checkpoint", "") or "") or None),
                "max_classes": int(model_max_classes),
                "embed_dim": int(self.config.embed_dim),
                "patch_size": int(getattr(self.config, "rowmixer_patch_size", 8)),
                "row_num_blocks": int(self.config.row_num_blocks),
                "row_nhead": int(self.config.row_nhead),
                "row_num_cls": int(self.config.row_num_cls),
                "row_num_global": int(self.config.row_num_global),
                "icl_num_blocks": int(self.config.icl_num_blocks),
                "icl_nhead": int(self.config.icl_nhead),
                "ff_factor": int(self.config.ff_factor),
                "dropout": float(self.config.dropout),
                "activation": str(self.config.activation),
                "norm_first": bool(self.config.norm_first),
                "shuffle_p": float(getattr(self.config, "rowmixer_shuffle_p", 0.25)),
                "perc_num_latents": int(self.config.perc_num_latents),
                "perc_layers": int(self.config.perc_layers),
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


def _extend_rowmixer_init_parser(parser):
    parser.add_argument(
        "--mantis_init_checkpoint",
        type=str,
        default=None,
        help="Path to mantis checkpoint used to initialize the trainable mantis encoder.",
    )
    parser.add_argument(
        "--rowmixer_init_checkpoint",
        type=str,
        default=None,
        help="Path to RowMixerLiteICL checkpoint used to initialize rowmixer_icl before training.",
    )
    parser.add_argument(
        "--rowmixer_init_strict",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use strict=True when loading --rowmixer_init_checkpoint.",
    )
    return parser


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = build_parser()
    parser = _extend_parser(parser)
    parser = _extend_rowmixer_init_parser(parser)
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
# train/train_config.py
"""Argument parser for ORION-MSP training."""

import argparse


def str2bool(value):
    return value.lower() == "true"


def train_size_type(value):
    value = float(value)
    if 0 < value < 1:
        return value
    elif value.is_integer():
        return int(value)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be either an integer (absolute position) or a float in (0,1)."
        )


def int_list(value):
    if isinstance(value, list):
        return [int(v) for v in value]
    s = str(value).replace(",", " ").split()
    try:
        out = [int(x) for x in s]
        if len(out) == 0:
            raise ValueError
        return out
    except Exception:
        raise argparse.ArgumentTypeError("Expected a list of integers, e.g. '1,4,8' or '1 4 8'.")


def build_parser():
    parser = argparse.ArgumentParser()

    # WandB
    parser.add_argument("--wandb_log", default=False, type=str2bool)
    parser.add_argument("--wandb_project", type=str, default="ORION-MSP")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--wandb_mode", default="offline", type=str)

    # Training config
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dtype", default="float32", type=str)
    parser.add_argument("--np_seed", type=int, default=42)
    parser.add_argument("--torch_seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=60000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=8)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine_warmup")
    parser.add_argument("--warmup_proportion", type=float, default=0.2)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cosine_num_cycles", type=int, default=1)
    parser.add_argument("--cosine_amplitude_decay", type=float, default=1.0)
    parser.add_argument("--cosine_lr_end", type=float, default=0.0)
    parser.add_argument("--poly_decay_lr_end", type=float, default=1e-7)
    parser.add_argument("--poly_decay_power", type=float, default=1.0)
    
    ##### ADDED FOR STABILITY #######
    parser.add_argument("--ignore_index", type=int, default=-100,
                    help="Label value to ignore in CE/masked tokens.")
    #################################
    
    
    # Prior data
    parser.add_argument("--prior_dir", type=str, default=None)
    parser.add_argument("--load_prior_start", type=int, default=0)
    parser.add_argument("--delete_after_load", default=False, type=str2bool)
    parser.add_argument("--batch_size_per_gp", type=int, default=4)
    parser.add_argument("--min_features", type=int, default=5)
    parser.add_argument("--max_features", type=int, default=100)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--min_seq_len", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--log_seq_len", default=False, type=str2bool)
    parser.add_argument("--seq_len_per_gp", default=False, type=str2bool)
    parser.add_argument("--min_train_size", type=train_size_type, default=0.5)
    parser.add_argument("--max_train_size", type=train_size_type, default=0.9)
    parser.add_argument("--replay_small", default=False, type=str2bool)
    parser.add_argument("--prior_type", default="mix_scm", type=str)
    parser.add_argument("--prior_device", default="cpu", type=str)
    parser.add_argument(
        "--prior_num_workers",
        type=int,
        default=1,
        help="DataLoader worker count for online prior generation.",
    )
    parser.add_argument(
        "--prior_prefetch_factor",
        type=int,
        default=4,
        help="DataLoader prefetch factor per worker for online prior generation.",
    )
    parser.add_argument(
        "--log_prior_batch_time",
        default=False,
        type=str2bool,
        help="If true, print rank0 prior(data generation) time per batch.",
    )
    parser.add_argument(
        "--log_prior_batch_time_every",
        type=int,
        default=1,
        help="Print prior batch time every N steps when --log_prior_batch_time=true.",
    )

    # Episode prior knobs (used when --prior_type=cauker_icl)
    parser.add_argument("--icl_k", type=int, default=5)
    parser.add_argument("--icl_time_length", type=int, default=512)
    parser.add_argument("--icl_num_features", type=int, default=6)
    parser.add_argument(
        "--icl_single_channel",
        default=False,
        type=str2bool,
        help="If true, force cauker_icl prior to generate single-channel samples (num_features=1).",
    )
    parser.add_argument("--icl_level", type=int, default=0)
    parser.add_argument("--icl_num_nodes", type=int, default=18)
    parser.add_argument("--icl_max_parents", type=int, default=5)
    parser.add_argument("--icl_max_lag", type=int, default=5)
    parser.add_argument(
        "--icl_feature_mode",
        type=str,
        default="mean",
        choices=["mean", "first", "flatten"],
        help="How to map multichannel episode samples (C,T) to trainer feature vector H.",
    )
    parser.add_argument("--icl_base_seed", type=int, default=42)
    parser.add_argument(
        "--icl_episode_workers",
        type=int,
        default=1,
        help="Parallel workers inside one cauker_icl batch generation call.",
    )
    parser.add_argument(
        "--icl_pool_backend",
        type=str,
        default="thread",
        choices=["serial", "thread", "process"],
        help="Parallel backend used inside cauker_icl get_batch.",
    )
    parser.add_argument(
        "--icl_program_pool_size",
        type=int,
        default=0,
        help="If >0, reuse class programs from a finite pool to increase cache hits and speed.",
    )
    parser.add_argument(
        "--icl_show_progress",
        default=False,
        type=str2bool,
        help="If true, show a temporary tqdm bar for each cauker_icl batch generation on rank0.",
    )

    # Model architecture flags
    parser.add_argument("--amp", default=True, type=str2bool)
    parser.add_argument("--model_compile", default=False, type=str2bool)

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="orion_msp",
        choices=["orion_msp", "rowmixer_lite_icl"],
        help="Backbone to train. Does not change prior data generation.",
    )

    # RowMixerLiteICL knobs (only used when --model=rowmixer_lite_icl)
    parser.add_argument(
        "--rowmixer_patch_size",
        type=int,
        default=8,
        help="RowMixerLite patch size along feature dim (token count ~= ceil(H/patch_size)).",
    )
    parser.add_argument(
        "--rowmixer_shuffle_p",
        type=float,
        default=0.25,
        help="Training-time probability to shuffle valid patch tokens (regularization).",
    )

    # Column emb
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--col_num_blocks", type=int, default=3)
    parser.add_argument("--col_nhead", type=int, default=4)
    parser.add_argument("--col_num_inds", type=int, default=128)
    parser.add_argument("--freeze_col", default=False, type=str2bool)

    # Multi-Scale Sparse Row interactor
    parser.add_argument("--row_num_blocks", type=int, default=3)
    parser.add_argument("--row_nhead", type=int, default=8)
    parser.add_argument("--row_num_cls", type=int, default=4)
    parser.add_argument("--row_rope_base", type=float, default=100000)
    parser.add_argument("--freeze_row", default=False, type=str2bool)
    parser.add_argument("--row_num_global", type=int, default=2)
    parser.add_argument("--row_scales", type=int_list, default=[1, 4, 8])
    parser.add_argument("--row_window", type=int, default=4)
    parser.add_argument("--row_num_random", type=int, default=0)
    parser.add_argument(
        "--row_group_mode",
        type=str,
        default="contiguous",
        choices=["contiguous", "pma"],
        help="Feature grouping per scale in TFrow: contiguous mean or PMA learned pooling.",
    )

    # Perceiver memory
    parser.add_argument("--perc_num_latents", type=int, default=16)
    parser.add_argument("--perc_layers", type=int, default=2)
    
    # ICL
    parser.add_argument("--icl_num_blocks", type=int, default=12)
    parser.add_argument("--icl_nhead", type=int, default=4)
    parser.add_argument("--freeze_icl", default=False, type=str2bool)

    # Shared
    parser.add_argument("--ff_factor", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--norm_first", default=True, type=str2bool)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--save_temp_every", default=50, type=int)
    parser.add_argument("--save_perm_every", default=1000, type=int)
    parser.add_argument("--max_checkpoints", type=int, default=5)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--only_load_model", default=False, type=str2bool)

    return parser

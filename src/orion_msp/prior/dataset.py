"""
The module offers a flexible framework for creating diverse, realistic tabular datasets
with controlled properties, which can be used for training and evaluating in-context
learning models. Key features include:

- Controlled feature relationships and causal structures via multiple generation methods
- Customizable feature distributions with mixed continuous and categorical variables
- Flexible train/test splits optimized for in-context learning evaluation
- Batch generation capabilities with hierarchical parameter sharing
- Memory-efficient handling of variable-length datasets

The main class is PriorDataset, which provides an iterable interface for generating
an infinite stream of synthetic datasets with diverse characteristics.
"""

from __future__ import annotations

import os
import sys
import math
import atexit
import warnings
import multiprocessing as mp
import queue as py_queue
from multiprocessing.pool import Pool as _MPPool
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Union, Optional, Any

import numpy as np
from scipy.stats import loguniform
import joblib

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset, get_worker_info
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

from .mlp_scm import MLPSCM

from .hp_sampling import HpSamplerList
from .reg2cls import Reg2Cls
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


def _cauker_generate_one(payload: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int]:
    from .synth_cauker_icl import render_program
    from .synth_cauker_icl import sample_program

    task_id = int(payload["task_id"])
    seed = int(payload["ep_seed"])
    K = int(payload["K"])
    train_size = int(payload["train_size"])
    seq_len = int(payload["seq_len"])
    n_qry_total = int(max(0, seq_len - train_size))

    rng = np.random.default_rng(seed + task_id * 1009)

    if K <= 0:
        raise ValueError("K must be positive")
    if train_size < K:
        raise ValueError(f"train_size ({train_size}) must be >= K ({K}) to guarantee >=1 context sample per class")

    ctx_counts = np.ones(K, dtype=np.int64)
    ctx_extra = int(train_size - K)
    if ctx_extra > 0:
        probs = rng.dirichlet(np.ones(K, dtype=np.float64))
        ctx_counts += rng.multinomial(ctx_extra, probs).astype(np.int64)

    if n_qry_total > 0:
        probs_q = rng.dirichlet(np.ones(K, dtype=np.float64))
        qry_counts = rng.multinomial(n_qry_total, probs_q).astype(np.int64)
    else:
        qry_counts = np.zeros(K, dtype=np.int64)

    x_ctx_list = []
    y_ctx_list = []
    x_qry_list = []
    y_qry_list = []

    pool_size = int(payload.get("program_pool_size", 0))
    for k in range(K):
        if pool_size > 0:
            pool_slot = int((task_id * K + k) % pool_size)
            class_seed = int(seed + pool_slot * 10_007)
        else:
            class_seed = int(seed + task_id * 100_003 + k * 10_007)

        program = sample_program(
            level=int(payload["level"]),
            time_length=int(payload["time_length"]),
            num_nodes=int(payload["num_nodes"]),
            max_parents=int(payload["max_parents"]),
            max_lag=int(payload["max_lag"]),
            seed=int(class_seed),
        )

        n_ctx_k = int(ctx_counts[k])
        n_qry_k = int(qry_counts[k])
        n_total_k = n_ctx_k + n_qry_k
        if n_total_k <= 0:
            continue

        xk = render_program(
            program=program,
            n=n_total_k,
            num_features=int(payload["num_features"]),
            seed=int(class_seed + 777),
        )

        perm = rng.permutation(n_total_k)
        xk = xk[perm]
        x_ctx_k = xk[:n_ctx_k]
        x_qry_k = xk[n_ctx_k:]

        if n_ctx_k > 0:
            x_ctx_list.append(x_ctx_k)
            y_ctx_list.append(np.full((n_ctx_k,), k, dtype=np.int64))
        if n_qry_k > 0:
            x_qry_list.append(x_qry_k)
            y_qry_list.append(np.full((n_qry_k,), k, dtype=np.int64))

    x_ctx = np.concatenate(x_ctx_list, axis=0).astype(np.float32)
    y_ctx = np.concatenate(y_ctx_list, axis=0).astype(np.int64)
    if n_qry_total > 0:
        x_qry = np.concatenate(x_qry_list, axis=0).astype(np.float32)
        y_qry = np.concatenate(y_qry_list, axis=0).astype(np.int64)
    else:
        x_qry = np.zeros((0, int(payload["num_features"]), int(payload["time_length"])), dtype=np.float32)
        y_qry = np.zeros((0,), dtype=np.int64)

    p_ctx = rng.permutation(x_ctx.shape[0])
    x_ctx, y_ctx = x_ctx[p_ctx], y_ctx[p_ctx]
    if x_qry.shape[0] > 0:
        p_qry = rng.permutation(x_qry.shape[0])
        x_qry, y_qry = x_qry[p_qry], y_qry[p_qry]

    ctx_classes = set(np.unique(y_ctx).tolist())
    if len(ctx_classes) != K:
        raise RuntimeError("Invalid episode: context does not contain all classes")

    x_all = np.concatenate([x_ctx, x_qry], axis=0)
    y_all = np.concatenate([y_ctx, y_qry], axis=0)

    # Keep full multichannel time series: (N, C, L).
    return x_all.astype(np.float32, copy=False), y_all.astype(np.int64, copy=False), train_size


class Prior:
    """
    Abstract base class for dataset prior generators.

    Defines the interface and common functionality for different types of
    synthetic dataset generators.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
    ):
        self.batch_size = batch_size

        assert min_features <= max_features, "Invalid feature range"
        self.min_features = min_features
        self.max_features = max_features

        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len

        self.validate_train_size_range(min_train_size, max_train_size)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = replay_small

    @staticmethod
    def validate_train_size_range(min_train_size: Union[int, float], max_train_size: Union[int, float]) -> None:
        """
        Checks if the training size range is valid.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size (position or ratio)

        max_train_size : int|float
            Maximum training size (position or ratio)

        Raises
        ------
        AssertionError
            If training size range is invalid
        ValueError
            If training size types are mismatched or invalid
        """
        # Check for numeric types only
        if not isinstance(min_train_size, (int, float)) or not isinstance(max_train_size, (int, float)):
            raise TypeError("Training sizes must be int or float")

        # Check for valid ranges based on type
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert 0 < min_train_size < max_train_size, "0 < min_train_size < max_train_size"
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            assert 0 < min_train_size < max_train_size < 1, "0 < min_train_size < max_train_size < 1"
        else:
            raise ValueError("Both training sizes must be of the same type (int or float)")

    @staticmethod
    def sample_seq_len(
        min_seq_len: Optional[int], max_seq_len: int, log: bool = False, replay_small: bool = False
    ) -> int:
        """
        Selects a random sequence length within the specified range.

        This method provides flexible sampling strategies for dataset sizes, including
        occasional re-sampling of smaller sequence lengths for better training diversity.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, returns max_seq_len directly.

        max_seq_len : int
            Maximum sequence length

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes

        replay_small : bool, default=False
            If True, occasionally sample smaller sequence lengths with
            specific distributions to ensure model robustness on smaller datasets

        Returns
        -------
        int
            The sampled sequence length
        """
        if min_seq_len is None:
            return max_seq_len

        if log:
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len)

        if replay_small:
            p = np.random.random()
            if p < 0.05:
                return np.random.randint(200, 1000)
            elif p < 0.3:
                return int(loguniform.rvs(1000, 10000))
            else:
                return seq_len
        else:
            return seq_len

    @staticmethod
    def sample_train_size(min_train_size: Union[int, float], max_train_size: Union[int, float], seq_len: int) -> int:
        """
        Selects a random training size within the specified range.

        This method handles both absolute position and fractional ratio approaches
        for determining the training/test split point.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        max_train_size : int|float
            Maximum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        seq_len : int
            Total sequence length

        Returns
        -------
        int
            The sampled training size position

        Raises
        ------
        ValueError
            If training size range has incompatible types
        """
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            train_size = np.random.randint(min_train_size, max_train_size)
        elif isinstance(min_train_size, float) and isinstance(min_train_size, float):
            train_size = np.random.uniform(min_train_size, max_train_size)
            train_size = int(seq_len * train_size)
        else:
            raise ValueError("Invalid training size range.")
        return train_size

    @staticmethod
    def adjust_max_features(seq_len: int, max_features: int) -> int:
        """
        Adjusts the maximum number of features based on the sequence length.

        This method implements an adaptive feature limit that scales inversely
        with sequence length. Longer sequences are restricted to fewer features
        to prevent memory issues and excessive computation times while still
        maintaining dataset diversity and learning difficulty.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of samples)

        max_features : int
            Original maximum number of features

        Returns
        -------
        int
            Adjusted maximum number of features, ensuring computational feasibility
        """
        if seq_len <= 10240:
            return min(100, max_features)
        elif 10240 < seq_len <= 20000:
            return min(80, max_features)
        elif 20000 < seq_len <= 30000:
            return min(60, max_features)
        elif 30000 < seq_len <= 40000:
            return min(40, max_features)
        elif 40000 < seq_len <= 50000:
            return min(30, max_features)
        elif 50000 < seq_len <= 60000:
            return min(20, max_features)
        elif 60000 < seq_len <= 65000:
            return min(15, max_features)
        else:
            return 10

    @staticmethod
    def delete_unique_features(X: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Removes features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant features
        to improve model training efficiency and stability. The removed features are
        replaced with zero padding to maintain tensor dimensions.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H) where:
            - B is batch size
            - T is sequence length
            - H is feature dimensionality

        d : Tensor
            Number of features per dataset of shape (B,), indicating how many
            features are actually used in each dataset (rest is padding)

        Returns
        -------
        tuple
            (X_new, d_new) where:
            - X_new is the filtered tensor with non-informative features removed
            - d_new is the updated feature count per dataset
        """

        def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor]:
            """Filters features with only one unique value from a single dataset."""
            num_features = xi.shape[-1]
            # Only consider actual features (up to di, ignoring padding)
            xi = xi[:, :di]
            # Identify features with more than one unique value (informative features)
            unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
            di_new = sum(unique_mask)
            # Create new tensor with only informative features, padding the rest
            xi_new = F.pad(xi[:, unique_mask], pad=(0, num_features - di_new), mode="constant", value=0)
            return xi_new, torch.tensor(di_new, device=xi.device)

        # Process each dataset in the batch independently
        filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
        X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

        return X_new, d_new

    @staticmethod
    def sanity_check(X: Tensor, y: Tensor, train_size: int, n_attempts: int = 10, min_classes: int = 2) -> bool:
        """
        Verifies that both train and test sets contain all classes.

        For in-context learning to work properly, we need both the train and test
        sets to contain examples from all classes. This method checks this condition
        and attempts to fix invalid splits by randomly permuting the data.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H)

        y : Tensor
            Target labels tensor of shape (B, T)

        train_size : int
            Position to split the data into train and test sets

        n_attempts : int, default=10
            Number of random permutations to try for fixing invalid splits

        min_classes : int, default=2
            Minimum number of classes required in both train and test sets

        Returns
        -------
        bool
            True if all datasets have valid splits, False otherwise
        """

        def is_valid_split(yi: Tensor) -> bool:
            """Check if a single dataset has a valid train/test split."""
            # Guard against invalid train_size
            if train_size <= 0 or train_size >= yi.shape[0]:
                return False

            # A valid split requires both train and test sets to have the same classes
            # and at least min_classes different classes must be present
            unique_tr = torch.unique(yi[:train_size])
            unique_te = torch.unique(yi[train_size:])
            return set(unique_tr.tolist()) == set(unique_te.tolist()) and len(unique_tr) >= min_classes

        # Check each dataset in the batch
        for i, (xi, yi) in enumerate(zip(X, y)):
            if is_valid_split(yi):
                continue

            # If the dataset has an invalid split, try to fix it with random permutations
            succeeded = False
            for _ in range(n_attempts):
                # Generate a random permutation of the samples
                perm = torch.randperm(yi.shape[0])
                yi_perm = yi[perm]
                xi_perm = xi[perm]
                # Check if the permutation results in a valid split
                if is_valid_split(yi_perm):
                    X[i], y[i] = xi_perm, yi_perm
                    succeeded = True
                    break

            if not succeeded:  # No valid split was found after all attempts
                return False

        return True


class SCMPrior(Prior):
    """
    Generates synthetic datasets using Structural Causal Models (SCM).

    The data generation process follows a hierarchical structure:
    1. Generate a list of parameters for each dataset, respecting group/subgroup sharing.
    2. Process the parameter list to generate datasets, applying necessary transformations and checks.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', or 'mix_scm'
        'mix_scm' randomly selects between 'mlp_scm' and 'tree_scm' based on probabilities.

    fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed structural configuration parameters

    sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
        )

        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.seq_len_per_gp = seq_len_per_gp
        self.prior_type = prior_type
        self.fixed_hp = fixed_hp
        self.sampled_hp = sampled_hp
        self.n_jobs = n_jobs
        self.num_threads_per_generate = num_threads_per_generate
        self.device = device

    def hp_sampling(self) -> Dict[str, Any]:
        """
        Sample hyperparameters for dataset generation.

        Returns
        -------
        dict
            Dictionary with sampled hyperparameters merged with fixed ones
        """
        hp_sampler = HpSamplerList(self.sampled_hp, device=self.device)
        return hp_sampler.sample()

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generates a single valid dataset based on the provided parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, device, etc.

        Returns
        -------
        tuple
            (X, y, d) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Labels tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
        """

        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            # Lazy import: tree_scm may depend on optional packages (e.g. xgboost).
            from .tree_scm import TreeSCM

            prior_cls = TreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        while True:
            X, y = prior_cls(**params)()
            X, y = Reg2Cls(params)(X, y)

            # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

            # Only keep valid datasets with sufficient features and balanced classes
            X, d = self.delete_unique_features(X, d)
            if (d > 0).all() and self.sanity_check(X, y, params["train_size"]):
                return X.squeeze(0), y.squeeze(0), d.squeeze(0)

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size

        Returns
        -------
        X : Tensor or NestedTensor
            Features tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
            If seq_len_per_gp=True, returns a NestedTensor.

        y : Tensor or NestedTensor
            Labels tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len).
            If seq_len_per_gp=True, returns a NestedTensor.

        d : Tensor
            Number of active features per dataset after filtering, shape (batch_size,)

        seq_lens : Tensor
            Sequence length for each dataset, shape (batch_size,)

        train_sizes : Tensor
            Position for train/test split for each dataset, shape (batch_size,)
        """
        batch_size = batch_size or self.batch_size

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None
        global_train_size = None

        # Determine global seq_len/train_size if not per-group
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
            )
            global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            group_sampled_hp = self.hp_sampling()
            # If per-group, sample seq_len and train_size for this group. Otherwise, use global ones
            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
                )
                gp_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len)
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_features = self.max_features

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
                if actual_subgp_size <= 0:
                    break

                # Subgroups share prior type, number of features, and sampled HPs
                subgp_prior_type = self.get_prior()
                subgp_num_features = round(np.random.uniform(self.min_features, gp_max_features))
                subgp_sampled_hp = {k: v() if callable(v) else v for k, v in group_sampled_hp.items()}

                # Generate parameters for each dataset in this subgroup
                for ds_idx in range(actual_subgp_size):
                    # Each dataset has its own number of classes
                    if np.random.random() > 0.5:
                        ds_num_classes = np.random.randint(2, self.max_classes + 1)
                    else:
                        ds_num_classes = 2

                    # Create parameters dictionary for this dataset
                    params = {
                        **self.fixed_hp,  # Fixed HPs
                        "seq_len": gp_seq_len,
                        "train_size": gp_train_size,
                        # If per-gp setting, use adjusted max features for this group because we use nested tensors
                        # If not per-gp setting, use global max features to fix size for concatenation
                        "max_features": gp_max_features if self.seq_len_per_gp else self.max_features,
                        **subgp_sampled_hp,  # sampled HPs for this group
                        "prior_type": subgp_prior_type,
                        "num_features": subgp_num_features,
                        "num_classes": ds_num_classes,
                        "device": self.device,
                    }
                    param_list.append(params)

        # Use joblib to generate datasets in parallel.
        # Note: the 'loky' backend does not support nested parallelism during DDP, whereas the 'threading' backend does.
        # However, 'threading' does not respect `inner_max_num_threads`.
        # Therefore, we stick with the 'loky' backend for parallelism, but this requires generating
        # the prior datasets separately from the training process and loading them from disk,
        # rather than generating them on-the-fly.
        if self.n_jobs > 1 and self.device == "cpu":
            with joblib.parallel_config(
                n_jobs=self.n_jobs, backend="loky", inner_max_num_threads=self.num_threads_per_generate
            ):
                results = joblib.Parallel()(joblib.delayed(self.generate_dataset)(params) for params in param_list)
        else:
            results = [self.generate_dataset(params) for params in param_list]

        X_list, y_list, d_list = zip(*results)

        # Combine Results
        if self.seq_len_per_gp:
            # Use nested tensors for variable sequence lengths
            X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
            y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
        else:
            # Stack into regular tensors for fixed sequence length
            X = torch.stack(X_list).to(self.device)  # (B, T, H)
            y = torch.stack(y_list).to(self.device)  # (B, T)

        # Metadata (always regular tensors)
        d = torch.stack(d_list).to(self.device)  # Actual number of features after filtering out constant ones
        seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list], device=self.device, dtype=torch.long
        )

        return X, y, d, seq_lens, train_sizes

    def get_prior(self) -> str:
        """
        Determine which prior type to use for generation.

        For 'mix_scm' prior type, randomly selects between available priors
        based on configured probabilities.

        Returns
        -------
        str
            The selected prior type name
        """
        if self.prior_type == "mix_scm":
            return np.random.choice(["mlp_scm", "tree_scm"], p=self.fixed_hp.get("mix_probas", [0.7, 0.3]))
        else:
            return self.prior_type


class DummyPrior(Prior):
    """This class creates purely random data. This is useful for testing and debugging
    without the computational overhead of SCM-based generation.

    Parameters
    ----------
    batch_size : int, default=256
        Number of datasets to generate

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    device : str, default="cpu"
        Computation device
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
        )
        self.device = device

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of random datasets for testing purposes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override, if None, uses self.batch_size

        Returns
        -------
        X : Tensor
            Features tensor of shape (batch_size, seq_len, max_features).
            Contains random Gaussian values for all features.

        y : Tensor
            Labels tensor of shape (batch_size, seq_len).
            Contains randomly assigned class labels.

        d : Tensor
            Number of features per dataset of shape (batch_size,).
            Always set to max_features for DummyPrior.

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).
            All datasets share the same sequence length.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
            All datasets share the same split position.
        """

        batch_size = batch_size or self.batch_size
        seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)
        train_size = self.sample_train_size(self.min_train_size, self.max_train_size, seq_len)

        X = torch.randn(batch_size, seq_len, self.max_features, device=self.device)

        num_classes = np.random.randint(2, self.max_classes + 1)
        y = torch.randint(0, num_classes, (batch_size, seq_len), device=self.device)

        d = torch.full((batch_size,), self.max_features, device=self.device)
        seq_lens = torch.full((batch_size,), seq_len, device=self.device)
        train_sizes = torch.full((batch_size,), train_size, device=self.device)

        return X, y, d, seq_lens, train_sizes


class CaukerICLPrior(Prior):
    """Episode-based time-series classification prior for ICL training.

    Internally samples labeled episodes from `synth_cauker_icl.generate_episode`,
    then converts each episode to the trainer contract:
      - X: (seq_len, feature_dim)
      - y: (seq_len,)
      - train_size: split point between context and query
    """

    def __init__(
        self,
        batch_size: int = 256,
        max_classes: int = 10,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        max_seq_len: int = 1024,
        prior_device: str = "cpu",
        icl_k: int = 5,
        icl_time_length: int = 512,
        icl_num_features: int = 6,
        icl_single_channel: bool = False,
        icl_level: int = 0,
        icl_num_nodes: int = 18,
        icl_max_parents: int = 5,
        icl_max_lag: int = 5,
        icl_feature_mode: str = "mean",
        icl_base_seed: int = 42,
        icl_episode_workers: int = 1,
        icl_pool_backend: str = "thread",
        icl_program_pool_size: int = 0,
        icl_task_pool_size: int = 0,
        icl_replay_prob: float = 0.0,
        icl_replay_warmup_steps: int = 0,
        icl_replay_prob_start: float = -1.0,
        icl_replay_prob_end: float = -1.0,
        icl_task_pool_mode: str = "episode",
        icl_pool_replace: str = "fifo",
        icl_replay_debug_every: int = 100,
        icl_show_progress: bool = False,
        replay_log_queue: Any = None,
    ):
        super().__init__(
            batch_size=batch_size,
            max_classes=max_classes,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            max_seq_len=max_seq_len,
        )
        self.device = prior_device
        self.icl_k = int(icl_k)
        self.icl_time_length = int(icl_time_length)
        self.icl_num_features = int(icl_num_features)
        self.icl_single_channel = bool(icl_single_channel)
        self.icl_level = int(icl_level)
        self.icl_num_nodes = int(icl_num_nodes)
        self.icl_max_parents = int(icl_max_parents)
        self.icl_max_lag = int(icl_max_lag)
        self.icl_feature_mode = str(icl_feature_mode)
        self.icl_base_seed = int(icl_base_seed)
        self.icl_episode_workers = max(1, int(icl_episode_workers))
        self.icl_pool_backend = str(icl_pool_backend)
        self.icl_program_pool_size = max(0, int(icl_program_pool_size))
        self.icl_task_pool_size = max(0, int(icl_task_pool_size))
        self.icl_replay_prob = float(icl_replay_prob)
        self.icl_replay_warmup_steps = max(0, int(icl_replay_warmup_steps))
        self.icl_replay_prob_start = float(icl_replay_prob_start)
        self.icl_replay_prob_end = float(icl_replay_prob_end)
        self.icl_task_pool_mode = str(icl_task_pool_mode)
        self.icl_pool_replace = str(icl_pool_replace)
        self.icl_replay_debug_every = max(0, int(icl_replay_debug_every))
        self.icl_show_progress = bool(icl_show_progress)
        self._replay_log_queue = replay_log_queue
        if self.icl_task_pool_mode not in {"episode", "payload"}:
            raise ValueError(f"Unknown icl_task_pool_mode: {self.icl_task_pool_mode}")
        if self.icl_pool_replace not in {"fifo", "random"}:
            raise ValueError(f"Unknown icl_pool_replace: {self.icl_pool_replace}")

        scheduled_probs = [max(0.0, self.icl_replay_prob)]
        if self.icl_replay_prob_start >= 0.0:
            scheduled_probs.append(self.icl_replay_prob_start)
        if self.icl_replay_prob_end >= 0.0:
            scheduled_probs.append(self.icl_replay_prob_end)
        self._replay_enabled = self.icl_task_pool_size > 0 and max(scheduled_probs) > 0.0

        self._task_cursor = 0
        self._batch_cursor = 0
        self._executor = None
        # Replay pool is intentionally local to each rank / DataLoader worker process.
        # We do not synchronize pool state across workers to avoid cross-process coordination.
        self._episode_pool: list[dict[str, Any]] = []
        self._pool_insert_ptr = 0
        self._replay_stats: dict[str, float] = {
            "hits": 0.0,
            "sampled": 0.0,
            "new": 0.0,
            "batches": 0.0,
        }
        self._rng: Optional[np.random.Generator] = None
        self._rng_seed: Optional[int] = None
        self._active_replay_prob: Optional[float] = None
        atexit.register(self._shutdown_executor)
        if int(os.environ.get("RANK", "0")) == 0:
            approx_episode_bytes = self._estimate_episode_bytes()
            approx_pool_mib = (approx_episode_bytes * self.icl_task_pool_size) / float(1024**2)
            print(
                "[rank0][replay-config] "
                f"task_pool_size={self.icl_task_pool_size} replay_prob={self.icl_replay_prob:.3f} "
                f"replay_prob_start={self.icl_replay_prob_start:.3f} replay_prob_end={self.icl_replay_prob_end:.3f} "
                f"warmup_steps={self.icl_replay_warmup_steps} pool_mode={self.icl_task_pool_mode} "
                f"replace={self.icl_pool_replace} program_pool_size={self.icl_program_pool_size} "
                f"approx_episode_mem={approx_episode_bytes / float(1024**2):.2f}MiB "
                f"approx_pool_mem={approx_pool_mib:.2f}MiB",
                flush=True,
            )

    def _shutdown_executor(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            if isinstance(executor, _MPPool):
                try:
                    executor.terminate()
                    executor.join()
                except Exception:
                    pass
            else:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            self._executor = None

    def _get_executor(self):
        if self.icl_episode_workers <= 1:
            return None
        if self._executor is not None:
            return self._executor

        backend = self.icl_pool_backend
        if backend not in {"serial", "thread", "process"}:
            backend = "thread"

        if backend == "serial":
            return None

        if backend == "process" and get_worker_info() is not None:
            backend = "thread"

        if backend == "process":
            try:
                self._executor = mp.get_context("spawn").Pool(processes=self.icl_episode_workers)
                return self._executor
            except Exception:
                self._executor = ThreadPoolExecutor(max_workers=self.icl_episode_workers)
                return self._executor

        self._executor = ThreadPoolExecutor(max_workers=self.icl_episode_workers)
        return self._executor

    def _estimate_episode_bytes(self) -> int:
        # For cauker_icl, x_tab is effectively (seq_len, time_length) float32 after feature reduction,
        # and y is (seq_len,) int64. Keep this estimate simple so users can size the episode pool.
        seq_len = int(self.max_seq_len)
        feature_dim = int(self.icl_time_length)
        return seq_len * feature_dim * 4 + seq_len * 8

    def _ensure_rng(self) -> np.random.Generator:
        if self._rng is not None:
            return self._rng
        worker = get_worker_info()
        worker_offset = 0 if worker is None else int(worker.id) + 1
        rank_offset = int(os.environ.get("RANK", "0"))
        seed = int(self.icl_base_seed + rank_offset * 100_003 + worker_offset * 1_000_003)
        self._rng_seed = seed
        self._rng = np.random.default_rng(seed)
        return self._rng

    def _is_replay_log_owner(self) -> bool:
        if int(os.environ.get("RANK", "0")) != 0:
            return False
        worker = get_worker_info()
        return worker is None or int(worker.id) == 0

    def _emit_replay_log(self, message: str) -> None:
        queue = getattr(self, "_replay_log_queue", None)
        if queue is not None:
            try:
                queue.put_nowait(str(message))
                return
            except Exception:
                pass
        print(message, flush=True)

    def _current_replay_prob(self) -> float:
        if not self._replay_enabled:
            return 0.0
        fixed = float(self.icl_replay_prob)
        if self.icl_replay_warmup_steps <= 0:
            return float(np.clip(fixed, 0.0, 1.0))
        if self.icl_replay_prob_start < 0.0 or self.icl_replay_prob_end < 0.0:
            return float(np.clip(fixed, 0.0, 1.0))
        progress = min(1.0, max(0.0, float(self._batch_cursor) / float(self.icl_replay_warmup_steps)))
        p = self.icl_replay_prob_start + (self.icl_replay_prob_end - self.icl_replay_prob_start) * progress
        return float(np.clip(p, 0.0, 1.0))

    def _should_replay(self) -> bool:
        if not self._replay_enabled or not self._episode_pool:
            return False
        rng = self._ensure_rng()
        replay_prob = self._active_replay_prob if self._active_replay_prob is not None else self._current_replay_prob()
        return bool(rng.random() < replay_prob)

    def _sample_from_pool(self, rng: np.random.Generator) -> Optional[dict[str, Any]]:
        if not self._episode_pool:
            return None
        idx = int(rng.integers(0, len(self._episode_pool)))
        item = self._episode_pool[idx]
        if self.icl_task_pool_mode == "payload":
            return {"mode": "payload", "payload": dict(item["payload"])}
        return {
            "mode": "episode",
            "payload": dict(item["payload"]),
            "x_tab": np.array(item["x_tab"], copy=True),
            "y_tab": np.array(item["y_tab"], copy=True),
            "n_ctx": int(item["n_ctx"]),
        }

    def _insert_into_pool(self, item: dict[str, Any]) -> None:
        if not self._replay_enabled or self.icl_task_pool_size <= 0:
            return
        if len(self._episode_pool) < self.icl_task_pool_size:
            self._episode_pool.append(item)
            return
        if self.icl_pool_replace == "random":
            rng = self._ensure_rng()
            idx = int(rng.integers(0, len(self._episode_pool)))
            self._episode_pool[idx] = item
            return
        idx = int(self._pool_insert_ptr % self.icl_task_pool_size)
        self._episode_pool[idx] = item
        self._pool_insert_ptr = (self._pool_insert_ptr + 1) % self.icl_task_pool_size

    def _make_pool_item(self, result: Tuple[np.ndarray, np.ndarray, int], payload: Dict[str, Any]) -> dict[str, Any]:
        if self.icl_task_pool_mode == "payload":
            return {"mode": "payload", "payload": dict(payload)}
        x_tab, y_tab, n_ctx = result
        return {
            "mode": "episode",
            "payload": dict(payload),
            "x_tab": np.array(x_tab, copy=True),
            "y_tab": np.array(y_tab, copy=True),
            "n_ctx": int(n_ctx),
        }

    def _materialize_pool_item(self, item: dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int]:
        if item["mode"] == "payload":
            return _cauker_generate_one(dict(item["payload"]))
        return (
            np.array(item["x_tab"], copy=True),
            np.array(item["y_tab"], copy=True),
            int(item["n_ctx"]),
        )

    def _generate_payloads(self, payloads: list[Dict[str, Any]]) -> list[Tuple[np.ndarray, np.ndarray, int]]:
        if not payloads:
            return []

        executor = self._get_executor()
        show_progress = bool(self.icl_show_progress) and self._is_replay_log_owner()
        if executor is None or len(payloads) <= 1:
            if show_progress and _tqdm is not None:
                iterator = (_cauker_generate_one(p) for p in payloads)
                return list(
                    _tqdm(
                        iterator,
                        total=len(payloads),
                        desc="cauker batch gen",
                        leave=False,
                        dynamic_ncols=True,
                    )
                )
            return [_cauker_generate_one(p) for p in payloads]

        if isinstance(executor, _MPPool):
            if show_progress and _tqdm is not None:
                iterator = executor.imap(_cauker_generate_one, payloads)
                return list(
                    _tqdm(
                        iterator,
                        total=len(payloads),
                        desc="cauker batch gen",
                        leave=False,
                        dynamic_ncols=True,
                    )
                )
            return list(executor.map(_cauker_generate_one, payloads))

        if show_progress and _tqdm is not None:
            futures = [executor.submit(_cauker_generate_one, p) for p in payloads]
            ordered: list[Optional[Tuple[np.ndarray, np.ndarray, int]]] = [None] * len(futures)
            future_to_idx = {fut: idx for idx, fut in enumerate(futures)}
            for fut in _tqdm(
                as_completed(futures),
                total=len(futures),
                desc="cauker batch gen",
                leave=False,
                dynamic_ncols=True,
            ):
                ordered[future_to_idx[fut]] = fut.result()
            return [r for r in ordered if r is not None]
        return list(executor.map(_cauker_generate_one, payloads))

    def _build_payload(self, task_id: int, train_size: int, seq_len: int, k_eff: int) -> Dict[str, Any]:
        ep_num_features = 1 if self.icl_single_channel else self.icl_num_features
        return {
            "task_id": int(task_id),
            "ep_seed": int(self.icl_base_seed + task_id * 131),
            "K": int(k_eff),
            "train_size": int(train_size),
            "seq_len": int(seq_len),
            "time_length": int(self.icl_time_length),
            "num_features": int(ep_num_features),
            "level": int(self.icl_level),
            "num_nodes": int(self.icl_num_nodes),
            "max_parents": int(self.icl_max_parents),
            "max_lag": int(self.icl_max_lag),
            "feature_mode": str(self.icl_feature_mode),
            # program_pool_size only reuses per-class program seeds. It is distinct from
            # the task pool below, which replays full episode results or full payloads.
            "program_pool_size": int(self.icl_program_pool_size),
        }

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = batch_size or self.batch_size
        rng = self._ensure_rng()
        batch_replay_prob = self._current_replay_prob()
        self._active_replay_prob = batch_replay_prob

        seq_len = int(self.max_seq_len)
        train_size = int(self.sample_train_size(self.min_train_size, self.max_train_size, seq_len))
        train_size = max(1, min(train_size, seq_len - 1))
        k_eff = max(1, min(int(min(self.icl_k, self.max_classes)), train_size))

        results: list[Optional[Tuple[np.ndarray, np.ndarray, int]]] = [None] * batch_size
        new_payloads: list[Dict[str, Any]] = []
        new_indices: list[int] = []
        batch_hits = 0

        for i in range(batch_size):
            if self._should_replay():
                pooled = self._sample_from_pool(rng)
                if pooled is not None:
                    results[i] = self._materialize_pool_item(pooled)
                    batch_hits += 1
                    continue

            task_id = int(self._task_cursor + i)
            payload = self._build_payload(task_id, train_size, seq_len, k_eff)
            new_payloads.append(payload)
            new_indices.append(i)

        new_results = self._generate_payloads(new_payloads)
        for idx, payload, result in zip(new_indices, new_payloads, new_results):
            results[idx] = result
            self._insert_into_pool(self._make_pool_item(result, payload))

        if any(r is None for r in results):
            raise RuntimeError("CaukerICLPrior produced an incomplete batch while mixing replay and new tasks.")

        materialized = [r for r in results if r is not None]

        x_tabs, y_tabs, n_ctxs = zip(*materialized)
        x_np = np.stack(x_tabs, axis=0)
        y_np = np.stack(y_tabs, axis=0)

        X = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        feature_dim = int(x_np.shape[-1])
        seq_len = int(x_np.shape[1])
        d = torch.full((batch_size,), feature_dim, device=self.device, dtype=torch.long)
        seq_lens = torch.full((batch_size,), seq_len, device=self.device, dtype=torch.long)
        train_sizes = torch.tensor(n_ctxs, device=self.device, dtype=torch.long)

        self._task_cursor += batch_size
        self._batch_cursor += 1
        self._replay_stats["hits"] += float(batch_hits)
        self._replay_stats["sampled"] += float(batch_size)
        self._replay_stats["new"] += float(len(new_payloads))
        self._replay_stats["batches"] += 1.0
        self._active_replay_prob = None

        if self._is_replay_log_owner() and self.icl_replay_debug_every > 0 and self._batch_cursor % self.icl_replay_debug_every == 0:
            cum_hit = self._replay_stats["hits"] / max(1.0, self._replay_stats["sampled"])
            hit_ratio = float(batch_hits) / max(1, batch_size)
            worker = get_worker_info()
            worker_id = -1 if worker is None else int(worker.id)
            self._emit_replay_log(
                "[rank0][replay] "
                f"worker={worker_id} batch={self._batch_cursor} task_cursor={self._task_cursor} p={batch_replay_prob:.3f} "
                f"pool={len(self._episode_pool)} hit={batch_hits}/{batch_size} new={len(new_payloads)}/{batch_size} "
                f"({1.0 - hit_ratio:.3f}) cum_hit={cum_hit:.3f}"
            )
        return X, y, d, seq_lens, train_sizes


class PriorDataset(IterableDataset):
    """
    Main dataset class that provides an infinite iterator over synthetic tabular datasets.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', 'mix_scm', 'cauker_icl', or 'dummy'

        1. SCM-based: Structural causal models with complex feature relationships
         - 'mlp_scm': MLP-based causal models
         - 'tree_scm': Tree-based causal models
         - 'mix_scm': Probabilistic mix of the above models
         - 'cauker_icl': Program-based labeled episode generator for time-series classification

        2. Dummy: Randomly generated datasets for debugging

    scm_fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed parameters for SCM-based priors

    scm_sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors)

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        icl_k: int = 5,
        icl_time_length: int = 512,
        icl_num_features: int = 6,
        icl_single_channel: bool = False,
        icl_level: int = 0,
        icl_num_nodes: int = 18,
        icl_max_parents: int = 5,
        icl_max_lag: int = 5,
        icl_feature_mode: str = "mean",
        icl_base_seed: int = 42,
        icl_episode_workers: int = 1,
        icl_pool_backend: str = "thread",
        icl_program_pool_size: int = 0,
        icl_task_pool_size: int = 0,
        icl_replay_prob: float = 0.0,
        icl_replay_warmup_steps: int = 0,
        icl_replay_prob_start: float = -1.0,
        icl_replay_prob_end: float = -1.0,
        icl_task_pool_mode: str = "episode",
        icl_pool_replace: str = "fifo",
        icl_replay_debug_every: int = 100,
        icl_show_progress: bool = False,
        replay_log_queue: Any = None,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        if prior_type == "dummy":
            self.prior = DummyPrior(
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                device=device,
            )
        elif prior_type in ["mlp_scm", "tree_scm", "mix_scm"]:
            self.prior = SCMPrior(
                batch_size=batch_size,
                batch_size_per_gp=batch_size_per_gp,
                batch_size_per_subgp=batch_size_per_subgp,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                seq_len_per_gp=seq_len_per_gp,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                replay_small=replay_small,
                prior_type=prior_type,
                fixed_hp=scm_fixed_hp,
                sampled_hp=scm_sampled_hp,
                n_jobs=n_jobs,
                num_threads_per_generate=num_threads_per_generate,
                device=device,
            )
        elif prior_type == "cauker_icl":
            self.prior = CaukerICLPrior(
                batch_size=batch_size,
                max_classes=max_classes,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                max_seq_len=max_seq_len,
                prior_device=device,
                icl_k=icl_k,
                icl_time_length=icl_time_length,
                icl_num_features=icl_num_features,
                icl_single_channel=icl_single_channel,
                icl_level=icl_level,
                icl_num_nodes=icl_num_nodes,
                icl_max_parents=icl_max_parents,
                icl_max_lag=icl_max_lag,
                icl_feature_mode=icl_feature_mode,
                icl_base_seed=icl_base_seed,
                icl_episode_workers=icl_episode_workers,
                icl_pool_backend=icl_pool_backend,
                icl_program_pool_size=icl_program_pool_size,
                icl_task_pool_size=icl_task_pool_size,
                icl_replay_prob=icl_replay_prob,
                icl_replay_warmup_steps=icl_replay_warmup_steps,
                icl_replay_prob_start=icl_replay_prob_start,
                icl_replay_prob_end=icl_replay_prob_end,
                icl_task_pool_mode=icl_task_pool_mode,
                icl_pool_replace=icl_pool_replace,
                icl_replay_debug_every=icl_replay_debug_every,
                icl_show_progress=icl_show_progress,
                replay_log_queue=replay_log_queue,
            )
        else:
            raise ValueError(
                f"Unknown prior type '{prior_type}'. Available options: 'mlp_scm', 'tree_scm', 'mix_scm', 'cauker_icl', or 'dummy'."
            )

        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.prior_type = prior_type

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of (batch_size, seq_len, max_features).

        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class labels of (batch_size, seq_len).

        d : Tensor
            Number of active features per dataset of shape (batch_size,).

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
        """
        return self.prior.get_batch(batch_size)

    def __iter__(self) -> "PriorDataset":
        """
        Returns an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator
        """
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns the next batch from the iterator. Since this is an infinite
        iterator, it never raises StopIteration and instead continuously generates
        new synthetic data batches.
        """
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        return (
            f"PriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max classes: {self.max_classes}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  device: {self.device}\n"
            f")"
        )


class DisablePrinting:
    """Context manager to temporarily suppress printed output."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout

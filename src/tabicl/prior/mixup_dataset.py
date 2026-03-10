"""Mixup-based synthetic dataset stream for MantisICL training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

import torch
from torch.utils.data import IterableDataset

from tabicl.data_io.synth_icl import MultiClassMixupDataset, _tensor_list_collator

# Conservative defaults aligned with the MultiClassMixupDataset signature
DEFAULT_MIXUP_CONFIG: Dict[str, float] = {
    "n_bit": 8,
    "n_step": 120,
    "max_class": 5,
    "mix_alpha": 1.0,
    "augment_cap": 2,
}


def load_mixup_config(config_path: Optional[str]) -> Dict[str, float]:
    """Load JSON config if provided; return empty dict otherwise."""

    if not config_path:
        return {}

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mixup config file not found: {path}")

    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Mixup config file must contain a JSON object")
    return data


class MixupPriorDataset(IterableDataset):
    """Wrap MultiClassMixupDataset to emit TabICL-style batches."""

    def __init__(
        self,
        batch_size: int,
        mixup_config: Optional[Dict[str, float]] = None,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1000,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.mixup_config = {**DEFAULT_MIXUP_CONFIG, **(mixup_config or {})}
        
        if min_seq_len is not None:
            self.mixup_config['min_n_step'] = min_seq_len
            self.mixup_config['max_n_step'] = max_seq_len
        else:
            self.mixup_config['n_step'] = max_seq_len

        self.mixup_config['min_train_size'] = min_train_size
        self.mixup_config['max_train_size'] = max_train_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        dataset = MultiClassMixupDataset(self.mixup_config)
        data_iter = iter(dataset)

        while True:
            try:
                samples = [next(data_iter) for _ in range(self.batch_size)]
            except StopIteration:
                data_iter = iter(dataset)
                continue
            collated = _tensor_list_collator(samples)
            yield self._convert_batch(collated)

    @staticmethod
    def _active_lengths(mask: torch.Tensor) -> int:
        """Return the number of non-masked entries (shared across batch)."""

        valid_counts = (~mask).sum(dim=1)
        min_count = int(valid_counts.min().item())
        return min_count

    def _convert_batch(self, batch: Dict[str, torch.Tensor]):
        """Convert synth batch into (X, y, d, seq_len, train_size)."""

        x_train = batch["x_train"].squeeze(2)  # (B, T_train, L)
        x_test = batch["x_test"].squeeze(2)  # (B, T_test, L)
        y_train = batch["y_train"]
        y_test = batch["y_test"]

        train_len = self._active_lengths(batch["mask_train"])
        test_len = self._active_lengths(batch["mask_test"])

        x_train = x_train[:, :train_len, :]
        x_test = x_test[:, :test_len, :]
        y_train = y_train[:, :train_len]
        y_test = y_test[:, :test_len]

        X = torch.cat([x_train, x_test], dim=1).contiguous()
        y = torch.cat([y_train, y_test], dim=1).contiguous()

        batch_size = X.shape[0]
        feature_dim = X.shape[-1]
        seq_len = X.shape[1]

        d = torch.full((batch_size,), feature_dim, dtype=torch.long)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
        train_sizes = torch.full((batch_size,), train_len, dtype=torch.long)

        return X.float(), y.long(), d, seq_lens, train_sizes

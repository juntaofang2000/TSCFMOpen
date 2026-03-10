from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def _extract_state_dict(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    """Best-effort extraction of a PyTorch state_dict from a checkpoint object."""
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            val = ckpt_obj.get(key)
            if isinstance(val, dict):
                return {str(k).replace("module.", ""): v for k, v in val.items()}
        # Some checkpoints are already a state-dict mapping.
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return {str(k).replace("module.", ""): v for k, v in ckpt_obj.items()}
    raise ValueError("Unsupported mantis checkpoint format; expected a dict or state_dict.")


def build_mantis_encoder(
    *,
    mantis_checkpoint: str | Path | None,
    device: torch.device | str | None = None,
    hidden_dim: int = 512,
    seq_len: int = 512,
    num_patches: int = 32,
    use_fddm: bool = False,
    num_channels: int = 1,
    strict: bool = False,
) -> nn.Module:
    """Build a Mantis encoder and optionally load a checkpoint.

    This repo uses the Mantis implementation under `tabicl.model.mantis_dev`.
    The returned module accepts input shaped `(B, C, L)` where `L == seq_len`.
    """

    dev = torch.device(device) if device is not None else torch.device("cpu")

    # Import lazily: mantis_dev pulls in heavier deps (einops, huggingface_hub, etc.).
    from tabicl.model.mantis_dev.architecture.architecture import Mantis8M, Mantis8MWithFDDM

    if use_fddm:
        model: nn.Module = Mantis8MWithFDDM(
            seq_len=int(seq_len),
            hidden_dim=int(hidden_dim),
            num_patches=int(num_patches),
            num_channels=int(num_channels),
            device=str(dev),
            pre_training=False,
        )
    else:
        model = Mantis8M(
            seq_len=int(seq_len),
            hidden_dim=int(hidden_dim),
            num_patches=int(num_patches),
            device=str(dev),
            pre_training=False,
        )

    ckpt_path = Path(mantis_checkpoint) if mantis_checkpoint is not None else None
    if ckpt_path is not None:
        if ckpt_path.is_dir():
            model = model.from_pretrained(str(ckpt_path))
            print(f"[MantisTabICL] Loaded pretrained Mantis encoder from {ckpt_path}")
        elif ckpt_path.is_file():
            ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
            state_dict = _extract_state_dict(ckpt_obj)
            model.load_state_dict(state_dict, strict=bool(strict))
        else:
            raise FileNotFoundError(f"Mantis checkpoint not found: {ckpt_path}")

    model.to(dev)
    model.eval()
    return model


@torch.no_grad()
def encode_with_mantis(
    model: nn.Module,
    x: torch.Tensor,
    *,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode a batch of time series with a Mantis encoder.

    - `x` supports `(N, L)`, `(N, 1, L)` or `(N, C, L)`.
    - returns `(N, D)` where `D == model.hidden_dim`.
    """

    if x.dim() == 2:
        x = x[:, None, :]
    if x.dim() != 3:
        raise ValueError(f"Expected x of shape (N,L) or (N,C,L); got {tuple(x.shape)}")

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    x = x.to(dev)

    outs: list[torch.Tensor] = []
    bs = max(1, int(batch_size))
    for i in range(0, x.shape[0], bs):
        outs.append(model(x[i : i + bs]))
    return torch.cat(outs, dim=0)

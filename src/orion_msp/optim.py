from __future__ import annotations

import math
from typing import Iterable, Optional

import torch


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the nearest orthogonal matrix using Newton-Schulz iterations.

    This helper is used by Muon to orthogonalize 2D updates. The iteration runs in
    float32 for numerical stability and casts back to the original dtype.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(matrix.shape)}")

    orig_dtype = matrix.dtype
    x = matrix.float()

    # Normalize to keep the iteration numerically stable.
    x = x / (x.norm() + eps)

    transposed = False
    if x.shape[0] < x.shape[1]:
        x = x.t()
        transposed = True

    for _ in range(int(max(1, steps))):
        xxt = x @ x.t()
        x = 1.5 * x - 0.5 * (xxt @ x)

    if transposed:
        x = x.t()
    return x.to(dtype=orig_dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer with AdamW fallback parameter group.

    Parameters listed in ``muon_params`` are optimized with Muon, while
    ``adamw_params`` use AdamW-style moment updates.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.1,
        muon_params: Optional[Iterable[torch.nn.Parameter]] = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params: Optional[Iterable[torch.nn.Parameter]] = None,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []

        defaults = dict(
            lr=float(lr),
            wd=float(wd),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            adamw_betas=tuple(adamw_betas),
            adamw_eps=float(adamw_eps),
        )
        super().__init__(muon_params + adamw_params, defaults)

        for p in muon_params:
            if p.ndim != 2:
                raise ValueError(f"Muon expects 2D tensors, got ndim={p.ndim}")
            self.state[p]["use_muon"] = True

        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @staticmethod
    def _adjust_lr_for_muon(lr: float, param_shape: torch.Size) -> float:
        a, b = int(param_shape[0]), int(param_shape[1])
        return float(lr) * (0.2 * math.sqrt(max(a, b)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["wd"])
            momentum = float(group["momentum"])
            beta1, beta2 = group["adamw_betas"]
            eps = float(group["adamw_eps"])

            # Muon branch.
            for p in [pp for pp in group["params"] if self.state[pp]["use_muon"]]:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_eff = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                u = zeropower_via_newtonschulz5(g_eff, steps=group["ns_steps"])
                adjusted_lr = self._adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            # AdamW fallback branch.
            for p in [pp for pp in group["params"] if not self.state[pp]["use_muon"]]:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)

                state["step"] += 1
                step = state["step"]
                m1 = state["moment1"]
                m2 = state["moment2"]
                m1.lerp_(g, 1 - beta1)
                m2.lerp_(g.square(), 1 - beta2)

                g_hat = m1 / (eps + m2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / math.sqrt(bias_correction2)

                p.data.mul_(1 - lr * wd)
                p.data.add_(g_hat, alpha=-lr / scale)

        return loss


__all__ = ["Muon", "zeropower_via_newtonschulz5"]

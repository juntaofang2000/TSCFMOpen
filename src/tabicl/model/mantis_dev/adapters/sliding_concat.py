import torch
import torch.nn.functional as F


class SlidingWindowChannelConcat:
    """Sliding window-based channel concatenation to unify multivariate time series to univariate.

    This implements the method you described:

    Given X in R^{B x C x L} and target sequence length T:
    - Window size: W = floor(T / C)
    - Remainder:   R = T % C
    - Stride:      S = floor(W / 2) (at least 1)
    - For each window k: X^(k) = X[:, :, start:end] in R^{B x C x W}
      where start = k*S, end = start+W
    - Flatten channels inside each window:
      X~^(k) = reshape(X^(k), [B, 1, C*W])
    - Pad zeros at the end to reach length T (pad length = R)

    Output formats:
    - return_format='stack': (B, K, T)
    - return_format='concat': (B, 1, K*T)
    - return_format='flatten_batch': (B*K, 1, T)

    Notes:
    - This method assumes T >= C so that W >= 1 and C*W <= T.
      If T < C, the method cannot fit one timestep from all channels into length T
      without dropping channels, so a ValueError is raised.
    - If L < W, X is zero-padded on the time axis to length W and K=1.

    Parameters
    ----------
    target_len : int
        Target univariate sequence length T per window.
    stride : int | None
        Stride S. If None, use max(1, W//2).
    return_format : str
        One of {'stack','concat','flatten_batch'}.
    """

    def __init__(self, target_len: int, stride: int | None = None, return_format: str = "stack"):
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")
        if return_format not in {"stack", "concat", "flatten_batch"}:
            raise ValueError(f"return_format must be one of stack/concat/flatten_batch, got {return_format}")
        if stride is not None and stride <= 0:
            raise ValueError(f"stride must be positive if provided, got {stride}")

        self.target_len = int(target_len)
        self.stride = stride
        self.return_format = return_format

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x[None, None, :]
        if x.dim() == 2:
            return x[:, None, :]
        return x

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform X of shape (B, C, L) into univariate windows."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        x = self._ensure_3d(x)
        if x.dim() != 3:
            raise ValueError(f"Expected X to be 3D (B,C,L); got shape {tuple(x.shape)}")

        b, c, l = x.shape
        t = self.target_len

        if t < c:
            raise ValueError(
                f"target_len T must be >= num_channels C for this method. Got T={t}, C={c}."
            )

        w = t // c
        r = t % c
        if w < 1:
            raise ValueError(f"Computed window size W is <1 (T={t}, C={c}).")

        s = self.stride if self.stride is not None else max(1, w // 2)

        # If sequence is shorter than one window, pad time axis so we can take K=1.
        if l < w:
            pad_len = w - l
            # pad last dimension (time): (left, right)
            x = F.pad(x, (0, pad_len))
            l = w

        # Compute window start indices
        starts = list(range(0, l - w + 1, s))
        if not starts:
            starts = [0]

        windows = []
        for start in starts:
            end = start + w
            xk = x[:, :, start:end]  # (B, C, W)
            # Flatten channels within the window: (B, C*W)
            flat = xk.reshape(b, c * w)
            if r > 0:
                flat = F.pad(flat, (0, r))  # (B, T)
            # Keep univariate channel dim
            windows.append(flat.unsqueeze(1))  # (B, 1, T)

        out = torch.stack(windows, dim=1)  # (B, K, 1, T)

        if self.return_format == "stack":
            return out.squeeze(2)  # (B, K, T)

        if self.return_format == "flatten_batch":
            b, k, _, t = out.shape
            return out.reshape(b * k, 1, t)  # (B*K, 1, T)

        # concat: concatenate windows along time
        # (B, K, 1, T) -> (B, 1, K*T)
        out = out.permute(0, 2, 1, 3).reshape(b, 1, -1)
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

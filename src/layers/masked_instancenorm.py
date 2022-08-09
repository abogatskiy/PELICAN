from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.instancenorm import _InstanceNorm


# Masked Instance Normalization

def masked_instance_norm(input: Tensor, mask: Tensor, weight: Optional[Tensor], bias: Optional[Tensor],
                         running_mean: Optional[Tensor], running_var: Optional[Tensor], use_input_stats: bool,
                         momentum: float, eps: float = 1e-5) -> Tensor:
    r"""Applies Masked Instance Normalization for each channel in each data sample in a batch.

    See :class:`~MaskedInstanceNorm1d`, :class:`~MaskedInstanceNorm2d`, :class:`~MaskedInstanceNorm3d` for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')

    shape = input.shape
    b, c = shape[:2]

    num_dims = len(shape[2:])
    _dims = tuple(range(-num_dims, 0))
    _slice = (...,) + (None,) * num_dims

    running_mean_ = running_mean[None, :].repeat(b, 1) if running_mean is not None else None
    running_var_ = running_var[None, :].repeat(b, 1) if running_mean is not None else None

    if use_input_stats:
        lengths = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / lengths  # (N, C)
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / lengths  # (N, C)

        if running_mean is not None:
            running_mean_.mul_(1 - momentum).add_(momentum * mean.detach())
            running_mean.copy_(running_mean_.view(b, c).mean(0, keepdim=False))
        if running_var is not None:
            running_var_.mul_(1 - momentum).add_(momentum * var.detach())
            running_var.copy_(running_var_.view(b, c).mean(0, keepdim=False))
    else:
        mean, var = running_mean_.view(b, c), running_var_.view(b, c)

    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)  # (N, C, ...)

    if weight is not None and bias is not None:
        out = out * weight[None, :][_slice] + bias[None, :][_slice]

    return out


class _MaskedInstanceNorm(_InstanceNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        super(_MaskedInstanceNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)

        if mask is None:
            return F.instance_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps
            )
        else:
            return masked_instance_norm(
                input, mask, self.weight, self.bias, self.running_mean, self.running_var,
                self.training or not self.track_running_stats, self.momentum, self.eps
            )


class MaskedInstanceNorm1d(torch.nn.InstanceNorm1d, _MaskedInstanceNorm):
    r"""Applies Instance Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.InstanceNorm1d` for details.

    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = False, track_running_stats: bool = False, device=None, dtype=None) -> None:
        super(MaskedInstanceNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype)


class MaskedInstanceNorm2d(torch.nn.InstanceNorm2d, _MaskedInstanceNorm):
    r"""Applies Instance Normalization over a masked 4D input
    (a mini-batch of 2D inputs with additional channel dimension).

    See documentation of :class:`~torch.nn.InstanceNorm2d` for details.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Mask: :math:`(N, 1, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = False, track_running_stats: bool = False, device=None, dtype=None) -> None:
        super(MaskedInstanceNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype)


class MaskedInstanceNorm3d(torch.nn.InstanceNorm3d, _MaskedInstanceNorm):
    r"""Applies Instance Normalization over a masked 5D input
    (a mini-batch of 3D inputs with additional channel dimension).

    See documentation of :class:`~torch.nn.InstanceNorm3d` for details.

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Mask: :math:`(N, 1, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = False, track_running_stats: bool = False, device=None, dtype=None) -> None:
        super(MaskedInstanceNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype)

"""Generic building blocks: activations, MLPs and the per-element message network."""
import torch
import torch.nn as nn

from .norms import MaskedBatchNorm1d, MaskedBatchNorm2d


class ATan(nn.Module):
    def forward(self, x):
        return torch.atan(x)


_ACTIVATIONS = {
    'leakyrelu': lambda: nn.LeakyReLU(negative_slope=0.01),
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'selu': nn.SELU,
    'gelu': nn.GELU,
    'elu': nn.ELU,
    'celu': nn.CELU,
    'silu': nn.SiLU,
    'sigmoid': nn.Sigmoid,
    'logsigmoid': nn.LogSigmoid,
    'atan': ATan,
    'soft': nn.Softsign,
    'tanh': nn.Tanh,
    'identity': nn.Identity,
}


def get_activation(name):
    """A fresh activation module (fresh instances matter for parametric ones like PReLU)."""
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {', '.join(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')


def mlp(channels, activation='leakyrelu', activate_last=False, dropout=0.0, batchnorm=False):
    """
    nn.Sequential of Linear layers over the last axis, channels = [in, hidden..., out].
    batchnorm inserts BatchNorm1d after each hidden Linear (only for [B,C] inputs).
    """
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        last = i == len(channels) - 2
        if batchnorm and not last:
            layers.append(nn.BatchNorm1d(channels[i + 1]))
        if not last or activate_last:
            layers.append(get_activation(activation))
            if dropout > 0 and not last:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MessageNet(nn.Module):
    """
    Element-wise MLP with every layer activated, applied to particle features [B,N,C] (order 1) or pair
    features [B,N,N,C] (order 2), then masked batch normalization. A single-entry `channels` is the identity.
    """
    def __init__(self, channels, order=1, activation='leakyrelu', batchnorm=True, masked=True):
        super().__init__()
        self.order, self.masked = order, masked
        self.net = mlp(channels, activation, activate_last=True)
        self.norm = None
        if batchnorm:
            if masked:
                self.norm = (MaskedBatchNorm1d if order == 1 else MaskedBatchNorm2d)(channels[-1])
            else:
                self.norm = (nn.BatchNorm1d if order == 1 else nn.BatchNorm2d)(channels[-1])

    def forward(self, x, mask):
        """mask: bool [B,N] (order 1) or [B,N,N] (order 2)."""
        m = mask.unsqueeze(-1)
        x = self.net(x) * m
        if self.norm is not None:
            if self.masked:
                x = self.norm(x, m)
            else:  # torch BatchNorm wants channels in dim 1
                x = self.norm(x.movedim(-1, 1)).movedim(1, -1) * m
        return x

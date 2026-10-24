"""
Input encoder: the pairwise invariants [B,N,N,K] (see symmetry.pair_invariants) are mixed by a linear map into
`out_dim` channels, then each channel goes through a learnable power function (base^a - 1)/a with a = alpha^2 + 1e-6,
which approaches log(base) as a -> 0:
  'slog'  (default): base = 1 + |x|, times sign(x)
  'angle' (IRC-safe inputs, which are angles 2(1 - cos theta) >= 0): base = 1e-5 + |x|
"""
import torch
import torch.nn as nn


def powerlog(x, alphas, mode):
    a = 1e-6 + alphas ** 2
    if mode == 'slog':
        return torch.expm1(a * torch.log1p(x.abs())) / a * x.sign()
    if mode == 'angle':
        return torch.expm1(a * (1e-5 + x.abs()).log()) / a
    raise ValueError(f"Unknown encoder mode '{mode}'")


def embedding_init(stabilizer, spurions, in_dim, out_dim):
    """
    Initial weight [out_dim, in_dim] of the linear embedding: every output channel starts as the dot product (the last
    input channel), or for the lab-frame stabilizers '1' and '1_0' with --method input, as the Minkowski product
    assembled from the component products.
    """
    if stabilizer == 'so13' or spurions:
        return torch.ones(out_dim, in_dim)
    if stabilizer in ('1', '1_0'):
        return torch.ones(out_dim, in_dim) - torch.tensor([[0., 2., 2., 2.]])
    w = torch.zeros(out_dim, in_dim)
    w[:, -1] = 1.
    return w


class InputEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, init_weight, mode='slog'):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(init_weight.clone())                 # not an nn.Linear, so init_weights leaves it alone
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.alphas = nn.Parameter(torch.linspace(0.05, 0.5, out_dim))
        self.out_dim = out_dim

    def forward(self, x, mask):
        """x: [B,N,N,in_dim] pairwise invariants, mask: bool [B,N,N]. Returns [B,N,N,out_dim]."""
        x = powerlog(torch.nn.functional.linear(x, self.weight, self.bias), self.alphas, self.mode)
        return torch.where(mask.unsqueeze(-1), x, torch.zeros((), dtype=x.dtype, device=x.device))

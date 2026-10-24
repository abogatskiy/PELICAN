import math
import pytest
import torch

from pelican.models import PelicanModel

torch.set_default_dtype(torch.float64)


def make_jets(B=6, N=24, n_min=4, seed=0, dtype=torch.float64):
    """
    Synthetic massless jets: constituents spread ~0.3 rad around a random axis, Pareto-like energies,
    random multiplicities in [n_min, N], zero-padded. Returns a batch dict like the data pipeline produces.
    """
    g = torch.Generator().manual_seed(seed)
    P = torch.zeros(B, N, 4, dtype=dtype)
    for b in range(B):
        n = int(torch.randint(n_min, N + 1, (1,), generator=g))
        eta, phi = 3 * torch.rand(1, generator=g) - 1.5, 2 * math.pi * torch.rand(1, generator=g)
        axis = torch.tensor([math.cos(phi) / math.cosh(eta), math.sin(phi) / math.cosh(eta), math.tanh(eta)], dtype=dtype)
        d = axis + 0.3 * torch.randn(n, 3, generator=g, dtype=dtype)
        d = d / d.norm(dim=1, keepdim=True)
        E = 5.0 * torch.rand(n, generator=g, dtype=dtype).pow(-1.2)        # heavy-tailed energies
        E, _ = E.sort(descending=True)
        P[b, :n, 0] = E
        P[b, :n, 1:] = E.unsqueeze(-1) * d
    mask = P[..., 0] != 0
    return {'Pmu': P, 'particle_mask': mask, 'Nobj': mask.sum(1),
            'is_signal': torch.randint(0, 2, (B,), generator=g), 'truth_Pmu': P.sum(1)}


SMALL = dict(num_channels_m=[[12], [12]], num_channels_eq=[8, 8], num_channels_m_out=[12, 8], num_channels_out=[8],
             num_channels_scalar=4, agg='M', agg_out='M', nobj_avg=10, batchnorm=True)


def build(task='classification', stabilizer='so2', method='spurions', irc_safe=False, nobj=24, **kw):
    tk = dict(SMALL)
    if irc_safe:
        tk.update(agg='s', agg_out='s')
    tk.update(kw)
    torch.manual_seed(0)
    m = PelicanModel(task=task, stabilizer=stabilizer, method=method, irc_safe=irc_safe, **tk).double()
    # One training-mode pass so batchnorm running stats are non-trivial, then evaluate deterministically.
    m.train(); m(make_jets(B=8, seed=123, N=nobj)); m.eval()
    return m


@pytest.fixture
def jets():
    return make_jets()

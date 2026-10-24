"""
Checks that Eq2to2 and Eq1to2 give the same outputs and gradients with and without materializing the equivariant basis
(fast_basis=True uses eops_2_to_2_mix / eops_1_to_2_mix, fast_basis=False the stacked eops_2_to_2 / eops_1_to_2 ops).
Run with `python -m pytest tests` or `python tests/test_fast_basis.py` from the repository root.
"""
import copy
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.layers.perm_equiv_models import Eq2to2, Eq1to2

B, N, C_IN, C_OUT = 5, 9, 4, 3
DTYPE = torch.float64


def make_inputs(rank, seed):
    g = torch.Generator().manual_seed(seed)
    nobj = torch.tensor([N, N - 2, 3, 1, N - 4]).view(B, 1)
    particle_mask = torch.arange(N).view(1, N) < nobj
    edge_mask = (particle_mask.unsqueeze(1) & particle_mask.unsqueeze(2)).unsqueeze(-1).to(DTYPE)
    if rank == 2:
        x = torch.randn(B, N, N, C_IN, generator=g, dtype=DTYPE) * edge_mask
    else:
        x = torch.randn(B, N, C_IN, generator=g, dtype=DTYPE) * particle_mask.unsqueeze(-1)
    irc_weight = torch.rand(B, N, generator=g, dtype=DTYPE) + 0.1
    return x, edge_mask, nobj, irc_weight


def randomize_alphas(layer, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for alpha in layer.alphas:
            if alpha is not None:
                sign = torch.randint(0, 2, alpha.shape, generator=g).to(DTYPE) * 2 - 1
                alpha.copy_(sign * (0.2 + torch.rand(alpha.shape, generator=g, dtype=DTYPE)))


def compare(fast, slow, x, mask, nobj, irc_weight):
    assert fast.fast_basis and not slow.fast_basis
    outputs, grads = [], []
    for layer in [fast, slow]:
        inp = x.clone().requires_grad_(True)
        out = layer(inp, mask=mask, nobj=nobj, irc_weight=irc_weight)
        out.pow(2).sum().backward()
        outputs.append(out.detach())
        grads.append([inp.grad] + [p.grad for p in layer.parameters()])
    assert torch.allclose(outputs[0], outputs[1], atol=1e-10, rtol=0), (outputs[0] - outputs[1]).abs().max()
    for g_fast, g_slow in zip(*grads):
        assert (g_fast is None) == (g_slow is None)
        if g_fast is not None:
            assert torch.allclose(g_fast, g_slow, atol=1e-10, rtol=0), (g_fast - g_slow).abs().max()


def check_eq2to2(config, factorize, use_weight, activate_lin, folklore=False, seed=0):
    torch.manual_seed(seed)
    fast = Eq2to2(C_IN, C_OUT, config=config, factorize=factorize, activate_lin=activate_lin, folklore=folklore, dtype=DTYPE)
    randomize_alphas(fast, seed)
    with torch.no_grad():
        fast.bias.normal_()
        fast.diag_bias.normal_()
    slow = copy.deepcopy(fast)
    slow.fast_basis = False
    x, mask, nobj, irc_weight = make_inputs(2, seed)
    compare(fast, slow, x, mask, nobj, irc_weight if use_weight else None)


def check_eq1to2(config, factorize, use_weight, activate_lin, seed=0):
    torch.manual_seed(seed)
    fast = Eq1to2(C_IN, C_OUT, config=config, factorize=factorize, activate_lin=activate_lin, dtype=DTYPE)
    randomize_alphas(fast, seed)
    with torch.no_grad():
        fast.bias.normal_()
    slow = copy.deepcopy(fast)
    slow.fast_basis = False
    x, mask, nobj, irc_weight = make_inputs(1, seed)
    compare(fast, slow, x, mask, nobj, irc_weight if use_weight else None)


def test_eq2to2():
    for seed, (config, factorize, use_weight, activate_lin) in enumerate(itertools.product(
            ['s', 'm', 'x', 'n', 'M', 'S', 'X', 'N'], [True, False], [False, True], [True, False])):
        check_eq2to2(config, factorize, use_weight, activate_lin, seed=seed)


def test_eq2to2_folklore():
    for seed, (config, use_weight) in enumerate(itertools.product(['s', 'M'], [False, True])):
        check_eq2to2(config, True, use_weight, True, folklore=True, seed=seed)


def test_eq1to2():
    for seed, (config, factorize, use_weight, activate_lin) in enumerate(itertools.product(
            ['s', 'm', 'x', 'n', 'M', 'S', 'X', 'N'], [True, False], [False, True], [True, False])):
        check_eq1to2(config, factorize, use_weight, activate_lin, seed=seed)


def test_state_dict_unchanged():
    expected = {'alphas.0', 'coefs00', 'coefs01', 'coefs10', 'coefs11', 'bias', 'diag_bias'}
    assert set(Eq2to2(C_IN, C_OUT, config='M', factorize=True).state_dict().keys()) == expected
    assert set(Eq2to2(C_IN, C_OUT, config='s', factorize=True).state_dict().keys()) == expected - {'alphas.0'}


def test_multi_letter_config_rejected():
    for layer in [Eq2to2, Eq1to2]:
        for config in ['sM', '', 'q']:
            try:
                layer(C_IN, C_OUT, config=config)
            except ValueError:
                continue
            raise AssertionError(f'{layer.__name__} accepted config {config!r}')


if __name__ == '__main__':
    test_eq2to2()
    test_eq2to2_folklore()
    test_eq1to2()
    test_state_dict_unchanged()
    test_multi_letter_config_rejected()
    print('All tests passed')

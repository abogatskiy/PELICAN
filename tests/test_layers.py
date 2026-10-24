"""Layer-level checks, including regressions for bugs in the pre-package code."""
import pytest
import torch

from pelican.nn import EqLayer, MessageNet, get_activation, powerlog

ORDERS = [(1, 2), (2, 0), (2, 1), (2, 2)]     # the equivariant maps PELICAN uses


@pytest.mark.parametrize('agg', ['x', 'n'])
def test_max_min_ignore_padding(agg):
    """Max/min aggregations used to include the zero padding (e.g. max of all-negative features was 0)."""
    torch.manual_seed(0)
    layer = EqLayer(2, 0, 3, 2, agg=agg, activate_lin=False).double()
    x = torch.randn(2, 5, 5, 3, dtype=torch.float64) - (3 if agg == 'x' else -3)   # all negative (max) / positive (min)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    x = x * (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
    nobj = mask.sum(1)
    out = layer(x, mask, nobj)
    out_trimmed = layer(x[:1, :3, :3], mask[:1, :3], nobj[:1])
    assert torch.allclose(out[:1], out_trimmed)


@pytest.mark.parametrize('orders', ORDERS)
@pytest.mark.parametrize('agg', list('smxnSMXN'))
def test_eq_layer_permutation_equivariance(orders, agg):
    i, o = orders
    torch.manual_seed(0)
    layer = EqLayer(i, o, 3, 4, agg=agg).double()
    N = 6
    x = torch.randn(2, *([N] * i), 3, dtype=torch.float64)
    mask = torch.ones(2, N, dtype=torch.bool)
    nobj = mask.sum(1)
    perm = torch.randperm(N)
    xp = x[:, perm] if i == 1 else x[:, perm][:, :, perm]
    out, outp = layer(x, mask, nobj), layer(xp, mask, nobj)
    expected = out if o == 0 else (out[:, perm] if o == 1 else out[:, perm][:, :, perm])
    assert torch.allclose(outp, expected, atol=1e-10)


def test_eq_layer_rejects_multi_letter_aggregation():
    with pytest.raises(ValueError):
        EqLayer(2, 2, 3, 3, agg='sM')


@pytest.mark.parametrize('orders,scaled', [((1, 2), 5), ((2, 0), 2), ((2, 1), 5), ((2, 2), 10)])
def test_scaled_operations(orders, scaled):
    """PELICAN scales all ops of its 1->2 and 2->1 layers by (nobj/nobj_avg)^alpha, but not the 5 free ops of 2->2."""
    assert EqLayer(*orders, 3, 4, agg='M').alphas.shape == (1, 3, scaled)
    assert EqLayer(*orders, 3, 4, agg='m').alphas is None


def test_message_net_identity_when_no_hidden_layers():
    net = MessageNet([5], batchnorm=False)
    x = torch.randn(2, 4, 5)
    mask = torch.ones(2, 4, dtype=torch.bool)
    assert torch.equal(net(x, mask), x)


def test_prelu_instances_are_independent():
    assert get_activation('prelu') is not get_activation('prelu')


@pytest.mark.parametrize('mode', ['slog', 'angle'])
def test_powerlog_is_accurate_in_float32_for_small_alpha(mode):
    """(base^a - 1)/a used to lose all precision in float32 as a -> 0; it is computed with expm1 now."""
    x = torch.tensor([1e-6, 1e-3, 0.1, 3., 1e3], dtype=torch.float64)
    alphas = torch.tensor(1e-3, dtype=torch.float64)
    ref = powerlog(x, alphas, mode)
    got = powerlog(x.float(), alphas.float(), mode).double()
    assert ((got - ref).abs() / ref.abs()).max() < 1e-4
    base = 1 + x if mode == 'slog' else 1e-5 + x
    assert torch.allclose(ref, base.log(), rtol=1e-2)            # approaches log(base) as alpha -> 0


@pytest.mark.parametrize('orders', [(1, 2), (2, 2)])
@pytest.mark.parametrize('agg', ['s', 'M', 'x', 'N'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('factorize', [False, True])
def test_pair_output_fast_path_matches_materialized(orders, agg, weighted, factorize):
    torch.manual_seed(0)
    layer = EqLayer(*orders, 3, 4, agg=agg, factorize=factorize).double()
    if layer.alphas is not None:
        layer.alphas.data.uniform_(-0.5, 0.5)
    layer.bias.data.normal_()
    if layer.diag_bias is not None:
        layer.diag_bias.data.normal_()
    N = 7
    mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3], dtype=torch.bool)
    x = torch.randn(2, *([N] * orders[0]), 3, dtype=torch.float64)
    x = x * (mask.unsqueeze(-1) if orders[0] == 1 else (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1))
    weight = torch.rand(2, N, dtype=torch.float64) * mask if weighted else None
    fast = layer(x, mask, mask.sum(1), weight)
    layer.activate_agg = True                                # forces the materialized path...
    layer.activation = torch.nn.Identity()                   # ...with an identity activation, so both compute the same map
    act_lin, layer.activate_lin = layer.activate_lin, False
    slow = layer(x, mask, mask.sum(1), weight)
    if act_lin:
        slow = torch.nn.functional.leaky_relu(slow, 0.01) * (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
    assert torch.allclose(fast, slow, atol=1e-10)

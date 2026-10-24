"""End-to-end symmetry properties of the model for every task and symmetry-breaking method."""
import pytest
import torch

from pelican.lorentz import stabilizer_algebra, random_group_element
from conftest import make_jets, build

COMBOS = [('classification', 'spurions'), ('classification', 'input'), ('regression', 'spurions'), ('regression', 'input')]


def transformed(batch, Lam):
    b = dict(batch)
    b['Pmu'] = batch['Pmu'] @ Lam.T
    return b


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def randomized_embedding(m):
    """The embedding initially reads only the dot products; random weights make the stabilizer-specific inputs matter."""
    with torch.no_grad():
        m.encoder.weight.copy_(torch.randn(m.encoder.weight.shape, generator=torch.Generator().manual_seed(3)))
    return m


SE2_INPUT_REGRESSION = pytest.mark.xfail(strict=True, reason='pre-package bug: with --method input, the se2 regression '
                                         'prediction combines the momentum components in a non-covariant way')


@pytest.mark.parametrize('task,method', COMBOS)
@pytest.mark.parametrize('stabilizer', ['so13', 'so3', 'so12', 'se2', 'so2', 'so2_0', 'R', '1', '11'])
def test_lorentz_symmetry(task, method, stabilizer, request):
    if (task, method, stabilizer) == ('regression', 'input', 'se2'):
        request.applymarker(SE2_INPUT_REGRESSION)
    m = randomized_embedding(build(task, stabilizer, method))
    b = make_jets()
    out = m(b)['predict']
    assert torch.isfinite(out).all()
    alg = stabilizer_algebra(m.symmetry_vectors)
    for seed in range(2):
        Lam = random_group_element(alg, 0.7, torch.Generator().manual_seed(seed))
        out2 = m(transformed(b, Lam))['predict']
        expected = out if task == 'classification' else out @ Lam.T     # regression output is covariant
        # Dot products of boosted, nearly collinear constituents cancel in the lab frame, hence the tolerance
        assert rel(out2, expected) < 1e-6


@pytest.mark.parametrize('task,method', COMBOS)
def test_broken_generators_change_outputs(task, method):
    """Sanity check of the test above: a generic Lorentz transformation must change the output of an so2 model."""
    m = randomized_embedding(build(task, 'so2', method))
    b = make_jets()
    Lam = random_group_element(stabilizer_algebra([]), 0.7, torch.Generator().manual_seed(0))
    out = m(b)['predict']
    expected = out if task == 'classification' else out @ Lam.T
    assert rel(m(transformed(b, Lam))['predict'], expected) > 1e-4


@pytest.mark.parametrize('task,method', COMBOS)
def test_permutation_invariance(task, method):
    m = build(task, 'so2', method)
    b = make_jets()
    out = m(b)['predict']
    g = torch.Generator().manual_seed(1)
    P = b['Pmu'].clone()
    for i, n in enumerate(b['Nobj'].tolist()):
        P[i, :n] = P[i, torch.randperm(n, generator=g)]
    assert rel(m({**b, 'Pmu': P})['predict'], out) < 1e-9


@pytest.mark.parametrize('task,method', COMBOS)
def test_padding_and_batch_independence(task, method):
    m = build(task, 'so2', method, nobj=30)
    b = make_jets(N=24)
    out = m(b)['predict']
    # extra zero-padding slots must not matter
    pad = lambda t, k: torch.cat([t, t.new_zeros(t.shape[0], k, *t.shape[2:])], 1)
    padded = {**b, 'Pmu': pad(b['Pmu'], 6), 'particle_mask': pad(b['particle_mask'], 6)}
    assert rel(m(padded)['predict'], out) < 1e-9
    # each event's output must not depend on the rest of the batch (eval mode)
    single = m({k: v[2:3] for k, v in b.items()})['predict']
    assert rel(single, out[2:3]) < 1e-9


def test_particle_scalars_keep_symmetry_and_padding_independence():
    """Per-particle features (e.g. --dataset jc) enter through the 1->2 lift and are masked like everything else."""
    m = build('classification', 'so2', num_scalars=3)
    b = make_jets()
    g = torch.Generator().manual_seed(2)
    b['scalars'] = torch.randn(*b['Pmu'].shape[:2], 3, generator=g) * b['particle_mask'].unsqueeze(-1)
    out = m(b)['predict']
    Lam = random_group_element(stabilizer_algebra(m.symmetry_vectors), 0.7, torch.Generator().manual_seed(0))
    assert rel(m(transformed(b, Lam))['predict'], out) < 1e-7
    pad = lambda t, k: torch.cat([t, t.new_zeros(t.shape[0], k, *t.shape[2:])], 1)
    padded = {**b, 'Pmu': pad(b['Pmu'], 5), 'particle_mask': pad(b['particle_mask'], 5), 'scalars': pad(b['scalars'], 5)}
    assert rel(m(padded)['predict'], out) < 1e-9


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_float32(task):
    m = build(task).float()
    b = make_jets()
    out = m({**b, 'Pmu': b['Pmu'].float()})['predict']
    assert out.dtype == torch.float32 and torch.isfinite(out).all()


def test_regression_shape_multiple_targets():
    m = build('regression', num_targets=3)
    out = m(make_jets())
    assert out['predict'].shape == (6, 3, 4) and out['weights'].shape == (6, 24 + 2, 3, 1)   # 2 spurions for so2


def test_classification_without_output_mlp():
    m = build('classification', mlp_out=False)
    assert m.head is None and m(make_jets())['predict'].shape == (6, 2)


@pytest.mark.parametrize('kwargs', [dict(task='regression', mlp_out=False), dict(method='input', stabilizer='1', irc_safe=True),
                                    dict(method='other'), dict(agg='sM'), dict(stabilizer='so4')])
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        build(**kwargs)

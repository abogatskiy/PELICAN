"""
IRC safety: zero-momentum insertion (IR) leaves IRC-safe models unchanged. PELICAN's IRC-safe mode is only
approximately collinear safe (its diagonal operations, diagonal bias and input nonlinearity see a collinear split),
which the strict xfail below documents.
"""
import pytest
import torch

from conftest import make_jets, build


def ir_insert(b):
    """Enable a padding slot (zero momentum) as a real particle in every event."""
    mask = b['particle_mask'].clone()
    for i, n in enumerate(b['Nobj'].tolist()):
        mask[i, n] = True
    return {**b, 'particle_mask': mask, 'Nobj': mask.sum(1)}


def c_split(b, frac=0.3):
    """Split the hardest constituent p into frac*p and (1-frac)*p, using a padding slot."""
    P, mask = b['Pmu'].clone(), b['particle_mask'].clone()
    for i, n in enumerate(b['Nobj'].tolist()):
        P[i, n] = (1 - frac) * P[i, 0]
        P[i, 0] = frac * P[i, 0]
        mask[i, n] = True
    return {**b, 'Pmu': P, 'particle_mask': mask, 'Nobj': mask.sum(1)}


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def with_free_slots(b, n_free=4):
    b = {**b, 'Pmu': b['Pmu'][:, :20], 'particle_mask': b['particle_mask'][:, :20]}   # guarantee free slots
    B = b['Pmu'].shape[0]
    return {**b, 'Pmu': torch.cat([b['Pmu'], torch.zeros(B, n_free, 4)], 1),
            'particle_mask': torch.cat([b['particle_mask'], torch.zeros(B, n_free, dtype=torch.bool)], 1),
            'Nobj': b['particle_mask'].sum(1)}


IRC_MODELS = [('spurions', 'so2', 's'), ('spurions', 'R', 's'), ('input', '11', 's'), ('input', 'so13', 's'),
              ('spurions', 'so2', 'm')]      # 'm' uses the SoftDrop multiplicity instead of the particle count


@pytest.mark.parametrize('task', ['classification', 'regression'])
@pytest.mark.parametrize('method,stabilizer,agg', IRC_MODELS)
def test_ir_safe(task, method, stabilizer, agg):
    m = build(task, stabilizer, method, irc_safe=True, agg=agg, agg_out=agg)
    b = with_free_slots(make_jets(N=24, n_min=4))
    out = m(b)['predict']
    assert rel(m(ir_insert(b))['predict'], out) < 1e-9


@pytest.mark.xfail(strict=True, reason='PELICAN IRC-safe models are only approximately collinear safe')
@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_collinear_safe(task):
    m = build(task, 'so2', irc_safe=True)
    b = with_free_slots(make_jets(N=24, n_min=4))
    assert rel(m(c_split(b))['predict'], m(b)['predict']) < 1e-9


def test_non_irc_model_is_not_ir_safe():
    """Sanity check that the IR test can fail."""
    m = build('classification', 'so2')
    b = with_free_slots(make_jets(N=24, n_min=4))
    assert rel(m(ir_insert(b))['predict'], m(b)['predict']) > 1e-6

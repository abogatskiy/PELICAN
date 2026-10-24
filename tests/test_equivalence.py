"""
TEMPORARY: the package's PelicanModel computes exactly what the pre-package models in src/ compute, given the same
weights (converted with pelican.legacy.convert_state_dict). Delete together with src/.
"""
import logging
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
logging.disable(logging.CRITICAL)
from src.models import PELICANClassifier, PELICANRegression          # noqa: E402
from pelican.models import PelicanModel                            # noqa: E402
from pelican.legacy import convert_state_dict                      # noqa: E402
from pelican.data.collate import _jetclass_scalars                 # noqa: E402
from conftest import make_jets                                     # noqa: E402

NCS, M, EQ, M_OUT, OUT = 4, [[12], [12]], [8, 8], [12, 8], [8]


def make_pair(task='classification', stabilizer='so2', method='spurions', irc_safe=False, agg='M', agg_out='M',
              factorize=True, batchnorm=True, mlp_out=True, activate_last=True, activate_agg=False, drop_rate=0.0,
              num_targets=1, num_classes=2, dataset='', seed=0):
    old_kw = dict(stabilizer=stabilizer, method=method, activate_agg=activate_agg, activation='leakyrelu', config=agg,
                  config_out=agg_out, average_nobj=10, factorize=factorize, masked=True, mlp_out=mlp_out,
                  mlp_out_activate_last=activate_last, irc_safe=irc_safe, dropout=drop_rate > 0, drop_rate=drop_rate,
                  drop_rate_out=drop_rate, batchnorm='b' if batchnorm else None, dataset=dataset, dtype=torch.float64)
    torch.manual_seed(seed)
    if task == 'classification':
        old = PELICANClassifier(2, NCS, M, EQ, OUT, M_OUT, num_classes=num_classes, **old_kw)
    else:
        old = PELICANRegression(2, NCS, M, EQ, OUT, M_OUT, num_targets, **old_kw)
    old = old.double()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():                                   # random weights everywhere, incl. alphas and batchnorm stats
        for name, t in list(old.named_parameters()) + list(old.named_buffers()):
            if 'num_batches_tracked' in name:
                continue
            if 'running_var' in name:
                t.copy_(0.5 + torch.rand(t.shape, generator=g, dtype=t.dtype))
            elif 'alphas' in name:
                t.copy_(torch.rand(t.shape, generator=g, dtype=t.dtype) - 0.5)
            else:
                t.copy_(0.4 * torch.randn(t.shape, generator=g, dtype=t.dtype) + (1. if name.endswith('normlayer.weight') else 0.))
    new = PelicanModel(task=task, num_classes=num_classes, num_targets=num_targets, stabilizer=stabilizer, method=method,
                       irc_safe=irc_safe, num_scalars=12 if dataset == 'jc' else 0, num_channels_scalar=NCS,
                       num_channels_m=M, num_channels_eq=EQ, num_channels_m_out=M_OUT, num_channels_out=OUT, agg=agg,
                       agg_out=agg_out, factorize=factorize, batchnorm=batchnorm, dropout=drop_rate,
                       dropout_out=drop_rate, nobj_avg=10, mlp_out=mlp_out, mlp_out_activate_last=activate_last,
                       activate_agg=activate_agg).double()
    new.load_state_dict(convert_state_dict(old.state_dict()), strict=True)
    return old, new


def batch(B=6, N=16, n_min=4, seed=0, jc=False):
    b = make_jets(B=B, N=N, n_min=n_min, seed=seed)
    b['edge_mask'] = b['particle_mask'].unsqueeze(1) & b['particle_mask'].unsqueeze(2)
    if jc:
        g = torch.Generator().manual_seed(seed)
        b['part_charge'] = torch.randint(-1, 2, (B, N), generator=g)
        for k in ['part_isChargedHadron', 'part_isElectron', 'part_isMuon', 'part_isNeutralHadron', 'part_isPhoton']:
            b[k] = torch.randint(0, 2, (B, N), generator=g).double()
        for k in ['d0val', 'd0err', 'dzval', 'dzerr']:
            b[k] = torch.randn(B, N, generator=g)
        b['scalars'] = _jetclass_scalars(b) * b['particle_mask'].unsqueeze(-1)    # what the package's collate_fn adds
    return b


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()


def compare(old, new, b, train=False):
    old.train(train), new.train(train)
    old_b = {k: v for k, v in b.items() if k != 'scalars'}                # the old model builds its own JetClass scalars
    out_old, out_new = old(old_b), new(b)
    assert rel(out_new['predict'], out_old['predict']) < 1e-10
    if 'weights' in out_old:
        assert rel(out_new['weights'], out_old['weights']) < 1e-10
    if train:
        (out_old['predict'] ** 2).sum().backward()
        (out_new['predict'] ** 2).sum().backward()
        grads_old = convert_state_dict({k: p.grad for k, p in old.named_parameters() if p.grad is not None})
        for k, p in new.named_parameters():
            assert k in grads_old, k
            assert rel(p.grad, grads_old[k]) < 1e-9, k
        buffers_old = convert_state_dict(dict(old.named_buffers()))
        for k, t in new.named_buffers():
            assert torch.equal(t, buffers_old[k]) if t.dtype == torch.long else rel(t, buffers_old[k]) < 1e-10, k


STABILIZERS = ['so13', 'so3', 'so12', 'se2', 'so2', 'so2_0', 'R', '1', '1_0', '11', '11_0']
CASES = [dict(task=t, method=m, stabilizer=s) for t in ('classification', 'regression') for m in ('spurions', 'input')
         for s in STABILIZERS if not (m == 'spurions' and s == 'so13')]                # that combination crashes in src/
CASES += [dict(task=t, agg=a, agg_out=ao) for t in ('classification', 'regression')
          for a, ao in [('s', 's'), ('m', 'm'), ('x', 'n'), ('n', 'x'), ('S', 'M'), ('X', 'S'), ('N', 'N'), ('M', 'x')]]
CASES += [dict(task=t, irc_safe=True, agg='s', agg_out='s', method=m, stabilizer=s)
          for t in ('classification', 'regression') for m, s in [('spurions', 'so2'), ('spurions', 'R'), ('input', '11'), ('input', 'so13')]]
# (IRC-safe models with a mean aggregation, which use the SoftDrop multiplicity, crash in src/ and are not compared.)
CASES += [dict(task=t, factorize=False) for t in ('classification', 'regression')]
CASES += [dict(task=t, batchnorm=False) for t in ('classification', 'regression')]
CASES += [dict(task=t, activate_agg=True) for t in ('classification', 'regression')]
CASES += [dict(task=t, activate_last=False) for t in ('classification', 'regression')]
CASES += [dict(task='classification', mlp_out=False), dict(task='classification', num_classes=5),
          dict(task='regression', num_targets=3), dict(task='regression', num_targets=2, method='input', stabilizer='so2')]


@pytest.mark.parametrize('case', CASES, ids=lambda c: '-'.join(f'{k}={v}' for k, v in c.items()))
def test_matches_old_model(case):
    old, new = make_pair(**case)
    compare(old, new, batch())                      # eval mode
    compare(old, new, batch(seed=1), train=True)    # train mode: outputs, gradients and batchnorm statistics


def test_matches_old_model_with_dropout_in_eval_mode():
    old, new = make_pair(drop_rate=0.3)
    compare(old, new, batch())


@pytest.mark.parametrize('task', ['classification'])
def test_matches_old_model_jetclass_without_padding(task):
    """JetClass scalars: the old code did not mask them, so they only agree on events without padding."""
    old, new = make_pair(task=task, dataset='jc')
    compare(old, new, batch(N=10, n_min=10, jc=True))

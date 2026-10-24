"""
MPS (Apple GPU) support. PELICAN computes lab-frame dot products in the model dtype, so float32 results are sensitive
to rounding; MPS results must be as close to a float64 reference as the CPU float32 results are.
"""
import copy
import pytest
import torch

from pelican.train.trainer import _to
from conftest import make_jets, build

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS not available')


@mps
@pytest.mark.parametrize('task,method', [('classification', 'spurions'), ('regression', 'spurions'),
                                         ('classification', 'input'), ('regression', 'input')])
def test_mps_matches_cpu(task, method):
    ref = build(task, method=method)
    cpu = copy.deepcopy(ref).float()
    gpu = copy.deepcopy(cpu).to('mps')
    batch = make_jets(B=16)                                   # float64 host batch, as read from HDF5
    out_ref = ref(batch)['predict']
    out_cpu = cpu(_to(batch, torch.device('cpu'), torch.float32))['predict'].double()
    out_gpu = gpu(_to(batch, torch.device('mps'), torch.float32))['predict'].cpu().double()
    rel = lambda a: ((a - out_ref).abs().max() / out_ref.abs().max()).item()
    assert rel(out_gpu) < max(2 * rel(out_cpu), 1e-4)


@mps
def test_mps_training_step():
    m = build().float().to('mps')
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    batch = _to(make_jets(B=16), torch.device('mps'), torch.float32)
    loss = torch.nn.functional.cross_entropy(m(batch)['predict'], batch['is_signal'].long())
    loss.backward()
    opt.step()
    assert torch.isfinite(loss) and all(torch.isfinite(p).all() for p in m.parameters())

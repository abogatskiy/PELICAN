"""Quick symmetry checks of a model on one real batch (pelican train --self-test)."""
import logging
import torch

from .lorentz import stabilizer_algebra, random_group_element

logger = logging.getLogger('pelican')


@torch.no_grad()
def run_selftest(model, batch, task):
    net = model.module if hasattr(model, 'module') else model
    was_training = net.training
    net.eval()
    out = net(batch)['predict']
    scale = out.abs().max().clamp_min(1e-12)
    results = {}

    P, mask = batch['Pmu'].clone(), batch['particle_mask']
    for i, n in enumerate(mask.sum(1).tolist()):
        P[i, :n] = P[i, torch.randperm(n)]
    results['permutation'] = ((net({**batch, 'Pmu': P})['predict'] - out).abs().max() / scale).item()

    alg = stabilizer_algebra(net.stabilizer.spurions).to(batch['Pmu'].device)
    Lam = random_group_element(alg.cpu(), 0.5).to(batch['Pmu'].device, batch['Pmu'].dtype)
    out2 = net({**batch, 'Pmu': batch['Pmu'] @ Lam.T})['predict']
    expected = out if task == 'classification' else out @ Lam.T.to(out.dtype)
    results[f'{net.stabilizer.name} transformation'] = ((out2 - expected).abs().max() / scale).item()

    for name, err in results.items():
        level = logging.INFO if err < 1e-3 else logging.WARNING
        logger.log(level, f'self-test: {name} invariance error {err:.1e}')
    net.train(was_training)
    return results

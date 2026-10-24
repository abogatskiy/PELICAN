"""Batch collation: pad events to a common length, scale momenta, and build masks and per-particle scalars."""
import torch
import torch.nn.functional as F


def _stack(values, nobj=None):
    v0 = values[0]
    if not torch.is_tensor(v0):
        return torch.tensor(values)
    if v0.dim() == 0:
        return torch.stack(values)
    if nobj is not None and nobj > 0:
        values = [v[:nobj] for v in values]
    return torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)


def _jetclass_scalars(batch):
    """JetClass per-particle features: charge one-hot (3), particle-ID flags (5) and impact parameters (4)."""
    charge = F.one_hot((batch['part_charge'].long() + 1).clamp(0, 2), 3)
    flags = [batch[k] for k in ('part_isChargedHadron', 'part_isElectron', 'part_isMuon', 'part_isNeutralHadron', 'part_isPhoton',
                                'd0val', 'd0err', 'dzval', 'dzerr')]
    return torch.cat([charge.to(torch.float64), torch.stack(flags, -1).to(torch.float64)], -1)


DATASET_PRESETS = {'': (None, 0), 'jc': (_jetclass_scalars, 12)}   # name -> (scalars builder, number of scalars)


def collate_fn(events, scale=1.0, nobj=None, dataset=''):
    """
    Keys whose first axis matches the number of particle slots of Pmu are per-particle and get truncated to the
    first `nobj` particles; all other keys (labels, one-hot targets, target 4-vectors) are stacked as they are.
    """
    n_slots = events[0]['Pmu'].shape[0]
    batch = {}
    for k in events[0].keys():
        v0 = events[0][k]
        per_particle = torch.is_tensor(v0) and v0.dim() > 0 and v0.shape[0] == n_slots
        batch[k] = _stack([e[k] for e in events], nobj if per_particle else None)
    batch['Pmu'] = batch['Pmu'] * scale
    mask = batch['Pmu'][..., 0] != 0
    batch['particle_mask'] = mask
    builder, _ = DATASET_PRESETS[dataset]
    if builder is not None:
        batch['scalars'] = builder(batch) * mask.unsqueeze(-1)
    return batch

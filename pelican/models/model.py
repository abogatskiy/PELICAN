"""
The PELICAN model: (spurions) -> pairwise invariants -> input encoder -> rank-2 equivariant blocks -> output.

Classification returns logits [B, num_classes]. Regression is Lorentz covariant: the network produces per-particle
weights w_i (invariant), and the prediction is sum_i w_i p_i over all particles (including spurions), shape
[B, num_targets, 4] (squeezed to [B,4] for a single target).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..symmetry import get_stabilizer, pair_invariants, symmetry_vectors, dot4
from ..softdrop import CATree, SDMultiplicity
from ..nn import EqLayer, EqNet, MessageNet, InputEncoder, embedding_init, init_weights, mlp


class PelicanModel(nn.Module):
    def __init__(self, task='classification', num_classes=2, num_targets=1, stabilizer='so2', method='spurions',
                 irc_safe=False, num_scalars=0, scale=1.0, num_channels_scalar=10, num_channels_m=([60],) * 5,
                 num_channels_eq=(35,) * 5, num_channels_m_out=(60, 35), num_channels_out=(60,), agg='M', agg_out='M',
                 activation='leakyrelu', activate_agg=False, activate_lin=True, activate_agg_out=True,
                 activate_lin_out=False, factorize=True, batchnorm=True, masked=True, dropout=0.0, dropout_out=0.0,
                 nobj_avg=49, mlp_out=True, mlp_out_activate_last=True):
        super().__init__()
        if method not in ('spurions', 'input'):
            raise ValueError(f"Unknown symmetry-breaking method '{method}' (choose spurions or input)")
        if task == 'regression' and not mlp_out:
            raise ValueError('Regression needs the output MLP (--mlp-out)')
        self.task, self.scale, self.irc_safe = task, scale, irc_safe
        self.num_targets = num_targets
        self.stabilizer = get_stabilizer(stabilizer)
        self.method = method
        self.symmetry_vectors = symmetry_vectors(self.stabilizer, method)     # their stabilizer is the model's symmetry group
        self.spurions = method == 'spurions' and self.stabilizer.num_spurions > 0
        # With spurions the inputs are plain dot products; otherwise the stabilizer decides which products enter.
        self.invariants = 'so13' if method == 'spurions' else stabilizer
        if irc_safe and self.invariants in ('1', '1_0'):
            raise ValueError(f"IRC-safe mode needs dot products; stabilizer '{stabilizer}' with --method input has none (use '11' or '11_0')")
        # The SoftDrop multiplicity replaces nobj in IRC-safe models with a mean aggregation (slow).
        self.softdrop_nobj = irc_safe and 'm' in (agg.lower(), agg_out.lower())
        pair_dim = 1 if self.invariants == 'so13' else self.stabilizer.pair_dim
        num_scalars = num_scalars + (1 + self.stabilizer.num_spurions if self.spurions else 0)

        num_channels_m = [list(c) for c in num_channels_m]
        width = num_channels_m[0][0] if num_channels_m and num_channels_m[0] else num_channels_eq[0]
        embedding_dim = width
        if num_scalars > 0:      # particle scalars are lifted to num_channels_scalar pair channels next to the embedding
            if width <= num_channels_scalar:
                raise ValueError(f'The first block width ({width}) must exceed num_channels_scalar ({num_channels_scalar}) when there are particle scalars')
            embedding_dim = width - num_channels_scalar

        self.encoder = InputEncoder(pair_dim, embedding_dim, embedding_init(self.invariants, self.spurions, pair_dim, embedding_dim),
                                    mode='angle' if irc_safe else 'slog')
        self.body = EqNet(2, list(num_channels_eq) + [num_channels_m_out[0]], num_channels_m, agg=agg, activate_agg=activate_agg,
                          activate_lin=activate_lin, activation=activation, dropout=dropout, batchnorm=batchnorm, masked=masked,
                          factorize=factorize, nobj_avg=nobj_avg)
        # PELICAN's final message layer always uses masked batch norm
        self.message_out = MessageNet(list(num_channels_m_out), order=2, activation=activation, batchnorm=batchnorm, masked=True)
        n_out = num_classes if task == 'classification' else num_targets * (1 if self.spurions else min(4, pair_dim))
        pool_dim = num_channels_out[0] if mlp_out else n_out
        self.pool = EqLayer(2, 0 if task == 'classification' else 1, num_channels_m_out[-1], pool_dim, agg=agg_out,
                            activate_agg=activate_agg_out, activate_lin=activate_lin_out, activation=activation,
                            factorize=factorize and task == 'regression', nobj_avg=nobj_avg)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout_out = nn.Dropout(dropout_out) if dropout_out > 0 else nn.Identity()
        self.head = mlp(list(num_channels_out) + [n_out], activation, activate_last=mlp_out_activate_last) if mlp_out else None
        self.apply(init_weights)
        # Particle scalars (spurion labels, JetClass features) are lifted to pairs with the output aggregation
        self.lift = EqLayer(1, 2, num_scalars, num_channels_scalar, agg=agg_out, activate_agg=False, activate_lin=True,
                            activation=activation, factorize=False, nobj_avg=nobj_avg) if num_scalars > 0 else None

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, batch):
        p = batch['Pmu'].to(self.dtype)
        mask = batch['particle_mask'].bool()
        scalars = batch['scalars'].to(self.dtype) if 'scalars' in batch else None
        if self.spurions:
            p, mask, scalars = self._add_spurions(p, mask, scalars)
        nobj = mask.sum(1)
        pairs = mask.unsqueeze(1) & mask.unsqueeze(2)

        inputs, weight = pair_invariants(p, self.invariants, self.irc_safe)
        if self.softdrop_nobj:
            with torch.no_grad():
                dots = dot4(p.unsqueeze(1), p.unsqueeze(2)).cpu()
                nobj = SDMultiplicity(CATree(dots, nobj.unsqueeze(-1).cpu(), ycut=1000, eps=10e-8)).to(nobj.device)
        x = self.encoder(inputs, pairs)
        if self.lift is not None:
            x = torch.cat([x, self.lift(scalars, mask, nobj, weight)], -1)

        x = self.body(x, mask, nobj, weight)
        x = self.dropout(self.message_out(x, pairs))
        x = self.dropout_out(self.pool(x, mask, nobj, weight))
        out = self.head(x) if self.head is not None else x
        if self.task == 'classification':
            return {'predict': out}
        weights = (out * mask.unsqueeze(-1)).view(out.shape[:2] + (self.num_targets, -1))
        return {'predict': self._regression_prediction(p, weights), 'weights': weights}

    def _add_spurions(self, p, mask, scalars):
        """Prepend the spurions as particles, labelled by a one-hot type (0 = constituent, k = k-th spurion)."""
        B = p.shape[0]
        S = self.stabilizer.spurion_tensor(p.device, p.dtype)
        ns = S.shape[0]
        p = torch.cat([S.expand(B, -1, -1), p], 1)
        mask = torch.cat([torch.ones(B, ns, dtype=torch.bool, device=mask.device), mask], 1)
        labels = 1 - mask.long()
        labels[:, :ns] = torch.arange(1, ns + 1, device=mask.device)
        onehot = (F.one_hot(labels, 1 + ns) * mask.unsqueeze(-1)).to(p.dtype)
        if scalars is not None:
            scalars = torch.cat([torch.cat([scalars.new_zeros(B, ns, scalars.shape[-1]), scalars], 1), onehot], -1)
        else:
            scalars = onehot
        return p, mask, scalars

    def _regression_prediction(self, p, weights):
        """sum_i w_i p_i. With --method input, each weight multiplies a stabilizer-invariant part of p_i."""
        dtype, device = p.dtype, p.device
        p = p.unsqueeze(-2)
        if self.method == 'input':
            name = self.stabilizer.name
            if name == 'so3':     # L_x, L_y, L_z
                p = p * torch.tensor([[[[1, 0, 0, 0], [0, 1, 1, 1]]]], dtype=dtype, device=device)
            elif name == 'so12':  # K_x, K_y, L_z
                p = p * torch.tensor([[[[1, 1, 1, 0], [0, 0, 0, 1]]]], dtype=dtype, device=device)
            elif name == 'se2':   # L_z, K_x-L_y, K_y+L_x (stabilizer of [1,0,0,1])
                p = (p.unsqueeze(-2) @ torch.tensor([[[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
                                                        [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0]]]]],
                                                      dtype=dtype, device=device)).squeeze(-2)
            elif name in ('so2', 'so2_0'):  # L_z
                p = p * torch.tensor([[[[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]]]], dtype=dtype, device=device)
            elif name == 'R':     # K_z
                p = p * torch.tensor([[[[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]]], dtype=dtype, device=device)
            elif name in ('11', '11_0'):
                p = p * torch.tensor([[[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]], dtype=dtype, device=device)
        pred = (p.unsqueeze(-3) * weights.unsqueeze(-1)).sum((1, 3)) / self.scale
        return pred.squeeze(-2) if self.num_targets == 1 else pred

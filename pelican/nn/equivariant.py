"""
Permutation-equivariant linear layers between particle tensors of order 0 (event), 1 (particles) and 2 (pairs),
after Maron et al. Each layer forms the basis of equivariant operations for one aggregation,
mixes them with learnable coefficients, adds a bias and applies an activation.

Aggregation letters: s=sum (divided by nobj_avg), m=mean, x=max, n=min. Upper case (S, M, X, N) additionally
multiplies each scaled operation by a learnable power (nobj/nobj_avg)^alpha.

With a per-particle `weight` (IRC-safe mode) the aggregated inputs are multiplied by the weights first; sums keep
their 1/nobj_avg normalization (PELICAN's convention).
"""
import math
import numpy as np
import torch
import torch.nn as nn

from .layers import get_activation, MessageNet

AGGREGATIONS = 'smxnSMXN'


def _aggregate(x, mask, dims, kind, nobj, nobj_avg):
    """x: [B,C,*S] (already multiplied by weights if any), mask: bool broadcastable to x. Reduces over dims."""
    shape = [-1] + [1] * (x.dim() - 1 - len(dims))
    n = nobj.view(shape)
    if kind == 's':
        return (x * mask).sum(dims) / nobj_avg ** len(dims)
    if kind == 'm':
        return (x * mask).sum(dims) / n ** len(dims)
    if kind == 'x':
        out = x.masked_fill(~mask, -math.inf).amax(dims)
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out)) - n.log()
    if kind == 'n':
        out = x.masked_fill(~mask, math.inf).amin(dims)
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out)) + n.log()
    raise ValueError(f'Unknown aggregation {kind}')


def _ops_1to0(x, m, agg, w):            # x [B,C,N] -> [B,C,1]
    return agg(x * w, m, (2,)).unsqueeze(2)


def _ops_1to1(x, m, agg, w):            # -> [B,C,2,N]
    N = x.shape[-1]
    return torch.stack([x, agg(x * w, m, (2,)).unsqueeze(-1).expand(-1, -1, N)], 2)


def _ops_1to2(x, m, agg, w):            # -> [B,C,5,N,N]
    N = x.shape[-1]
    s = agg(x * w, m, (2,)).unsqueeze(-1)                       # [B,C,1]
    return torch.stack([torch.diag_embed(x),
                        x.unsqueeze(2).expand(-1, -1, N, -1),
                        x.unsqueeze(3).expand(-1, -1, -1, N),
                        torch.diag_embed(s.expand(-1, -1, N)),
                        s.unsqueeze(-1).expand(-1, -1, N, N)], 2)


def _pair_parts(x, m, agg, w):
    """Aggregates of a pair tensor x [B,C,N,N]: diagonal, row sums, column sums, diagonal sum, total."""
    diag = torch.diagonal(x, dim1=-2, dim2=-1)                  # [B,C,N]
    md = torch.diagonal(m, dim1=-2, dim2=-1)                    # [B,1,N]
    wr, wc = w.unsqueeze(-2), w.unsqueeze(-1)                   # weights over columns / rows
    rows = agg(x * wr, m, (3,))                                 # [B,C,N]
    cols = agg(x * wc, m, (2,))
    sdiag = agg(diag * w, md, (2,))                             # [B,C]
    total = agg(x * wr * wc, m, (2, 3))                         # [B,C]
    return diag, rows, cols, sdiag, total


def _ops_2to0(x, m, agg, w):            # -> [B,C,2]
    _, _, _, sdiag, total = _pair_parts(x, m, agg, w)
    return torch.stack([total, sdiag], 2)


def _ops_2to1(x, m, agg, w):            # -> [B,C,5,N]
    N = x.shape[-1]
    diag, rows, cols, sdiag, total = _pair_parts(x, m, agg, w)
    return torch.stack([diag, rows, cols, sdiag.unsqueeze(-1).expand(-1, -1, N), total.unsqueeze(-1).expand(-1, -1, N)], 2)


def _ops_2to2(x, m, agg, w):            # -> [B,C,15,N,N]  (the 15-element basis of PELICAN)
    N = x.shape[-1]
    diag, rows, cols, sdiag, total = _pair_parts(x, m, agg, w)
    sdiag, total = sdiag.unsqueeze(-1), total.unsqueeze(-1)
    ops = [x, x.transpose(2, 3), torch.diag_embed(diag),
           diag.unsqueeze(2).expand(-1, -1, N, -1), diag.unsqueeze(3).expand(-1, -1, -1, N),
           torch.diag_embed(cols), cols.unsqueeze(2).expand(-1, -1, N, -1), cols.unsqueeze(3).expand(-1, -1, -1, N),
           torch.diag_embed(rows), rows.unsqueeze(2).expand(-1, -1, N, -1), rows.unsqueeze(3).expand(-1, -1, -1, N),
           sdiag.unsqueeze(-1).expand(-1, -1, N, N), torch.diag_embed(sdiag.expand(-1, -1, N)),
           total.unsqueeze(-1).expand(-1, -1, N, N), torch.diag_embed(total.expand(-1, -1, N))]
    return torch.stack(ops, 2)


# (in_order, out_order) -> (ops function, number of unscaled ops at the front, total basis size).
# PELICAN's 1->2 and 2->1 layers apply the nobj^alpha scaling to all of their operations, including the
# aggregation-free ones; its 2->2 layer leaves the five aggregation-free operations unscaled.
_SPECS = {(1, 0): (_ops_1to0, 0, 1), (1, 1): (_ops_1to1, 1, 2), (1, 2): (_ops_1to2, 0, 5),
          (2, 0): (_ops_2to0, 0, 2), (2, 1): (_ops_2to1, 0, 5), (2, 2): (_ops_2to2, 5, 15)}
# For layers producing pairs: how each basis element depends on the output indices (i, j), and which part of the
# input it carries. 'full' = an N x N tensor, 'row' = a vector over i, 'col' = a vector over j, 'diag' = a vector
# on the diagonal, 'const' / 'diagconst' = per-event constants everywhere / on the diagonal. Mixing channels before
# broadcasting avoids materializing the whole [B,C,K,N,N] basis (see EqLayer._pair_output).
_PAIR_STRUCTURE = {
    (1, 2): [('diag', 'x'), ('col', 'x'), ('row', 'x'), ('diagconst', 's'), ('const', 's')],
    (2, 2): [('full', 'x'), ('fullT', 'x'), ('diag', 'diag'), ('col', 'diag'), ('row', 'diag'),
             ('diag', 'cols'), ('col', 'cols'), ('row', 'cols'), ('diag', 'rows'), ('col', 'rows'), ('row', 'rows'),
             ('const', 'sdiag'), ('diagconst', 'sdiag'), ('const', 'total'), ('diagconst', 'total')],
}


class EqLayer(nn.Module):
    """Equivariant linear map from order-`in_order` to order-`out_order` tensors (channels last)."""
    def __init__(self, in_order, out_order, in_dim, out_dim, agg='M', activate_agg=False, activate_lin=True,
                 activation='leakyrelu', factorize=True, nobj_avg=49):
        super().__init__()
        if len(agg) != 1 or agg not in AGGREGATIONS:
            raise ValueError(f"Aggregation must be one letter from '{AGGREGATIONS}', got '{agg}'")
        self.in_order, self.out_order = in_order, out_order
        self.ops_fn, self.n_free, K = _SPECS[(in_order, out_order)]
        self.agg, self.nobj_avg = agg, nobj_avg
        self.activate_agg, self.activate_lin = activate_agg, activate_lin
        self.activation = get_activation(activation)
        self.factorize = factorize
        # PELICAN initializes the scaling exponents of its pair-to-pair and pooling layers randomly, the others at 0
        init = torch.rand if (in_order, out_order) in ((2, 2), (2, 0)) else torch.zeros
        self.alphas = nn.Parameter(init(1, in_dim, K - self.n_free)) if agg.isupper() else None
        if factorize:
            self.coefs00 = nn.Parameter(torch.randn(in_dim, K) * np.sqrt(1. / K))
            self.coefs01 = nn.Parameter(torch.randn(out_dim, K) * np.sqrt(1. / K))
            self.coefs10 = nn.Parameter(torch.randn(in_dim, out_dim) * np.sqrt(1. / in_dim))
            self.coefs11 = nn.Parameter(torch.randn(in_dim, out_dim) * np.sqrt(1. / in_dim))
        else:
            gain = 4. if (in_order, out_order) == (2, 0) else 2.
            self.coefs = nn.Parameter(torch.randn(in_dim, out_dim, K) * np.sqrt(gain / (in_dim * K)))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.diag_bias = nn.Parameter(torch.zeros(out_dim)) if (in_order, out_order) == (2, 2) else None

    def coefficients(self):
        if self.factorize:
            return self.coefs00.unsqueeze(1) * self.coefs10.unsqueeze(-1) + self.coefs01.unsqueeze(0) * self.coefs11.unsqueeze(-1)
        return self.coefs

    def forward(self, x, mask, nobj, weight=None):
        """
        x: [B,N,C] or [B,N,N,C]; mask: bool [B,N]; nobj: [B] particle counts; weight: optional [B,N].
        Returns [B,C'], [B,N,C'] or [B,N,N,C'].
        """
        kind = self.agg.lower()
        if self.in_order == 1:
            xt, m = x.permute(0, 2, 1), mask.unsqueeze(1)                                   # [B,C,N], [B,1,N]
            w = weight.unsqueeze(1) if weight is not None else torch.ones_like(m, dtype=x.dtype)
        else:
            xt, m = x.permute(0, 3, 1, 2), (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(1)
            w = weight.unsqueeze(1) if weight is not None else torch.ones_like(mask, dtype=x.dtype).unsqueeze(1)
        nobj = nobj.to(x.dtype).clamp_min(1)
        agg = lambda t, mm, dims: _aggregate(t, mm, dims, kind, nobj, self.nobj_avg)
        if self.out_order == 2 and not self.activate_agg:
            out = self._pair_output(xt, m, agg, w, nobj) + self.bias
        else:
            out = self._materialized_output(xt, m, agg, w, nobj) + self.bias
        if self.diag_bias is not None:
            out = out + torch.diag_embed(self.diag_bias.expand(out.shape[1], -1).T).permute(1, 2, 0)
        if self.activate_lin:
            out = self.activation(out)
        if self.out_order == 1:
            out = out * mask.unsqueeze(-1)
        elif self.out_order == 2:
            out = out * (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
        return out

    def _alpha_mult(self, nobj, k):
        """(nobj/nobj_avg)^alpha for basis element k (scaled elements only), shape [B,C]; None if unscaled."""
        if self.alphas is None or k < self.n_free:
            return None
        return (nobj / self.nobj_avg).view(-1, 1) ** self.alphas[0, :, k - self.n_free]

    def _materialized_output(self, xt, m, agg, w, nobj):
        ops = self.ops_fn(xt, m, agg, w)                                                    # [B,C,K,*S]
        if self.alphas is not None:
            mult = (nobj / self.nobj_avg).view([-1, 1, 1] + [1] * (ops.dim() - 3)) ** self.alphas.view(self.alphas.shape + (1,) * (ops.dim() - 3))
            ops = torch.cat([ops[:, :, :self.n_free], ops[:, :, self.n_free:] * mult], 2)
        if self.activate_agg:
            ops = self.activation(ops)
        return torch.einsum('dsb,ndb...->n...s', self.coefficients(), ops)

    def _pair_output(self, xt, m, agg, w, nobj):
        """Same result as _materialized_output for pair outputs, without building the [B,C,K,N,N] basis."""
        if self.in_order == 1:
            parts = {'x': xt, 's': agg(xt * w, m, (2,))}
        else:
            diag, rows, cols, sdiag, total = _pair_parts(xt, m, agg, w)
            parts = {'x': xt, 'diag': diag, 'rows': rows, 'cols': cols, 'sdiag': sdiag, 'total': total}
        structure = _PAIR_STRUCTURE[(self.in_order, 2)]
        W = self.coefficients()                                                              # [C,C',K]
        B, N = xt.shape[0], xt.shape[-1]
        acc = {}
        for k, (kind, src) in enumerate(structure):
            v, mult = parts[src], self._alpha_mult(nobj, k)
            if mult is not None:
                v = v * (mult if v.dim() == 2 else mult.unsqueeze(-1))
            if kind == 'full':
                term = torch.einsum('bdij,ds->bijs', v, W[:, :, k])
            elif kind == 'fullT':
                term = torch.einsum('bdji,ds->bijs', v, W[:, :, k])
            elif kind in ('row', 'col', 'diag'):
                term = torch.einsum('bdn,ds->bns', v, W[:, :, k])
            else:
                term = torch.einsum('bd,ds->bs', v, W[:, :, k])
            key = 'full' if kind == 'fullT' else kind
            acc[key] = acc[key] + term if key in acc else term
        out = acc.get('full', xt.new_zeros(B, N, N, W.shape[1]))
        if 'row' in acc:
            out = out + acc['row'].unsqueeze(2)
        if 'col' in acc:
            out = out + acc['col'].unsqueeze(1)
        if 'const' in acc:
            out = out + acc['const'][:, None, None, :]
        diag = acc.get('diag')
        if 'diagconst' in acc:
            dc = acc['diagconst'].unsqueeze(1).expand(-1, N, -1)
            diag = dc if diag is None else diag + dc
        if diag is not None:
            out = out + torch.diag_embed(diag.transpose(1, 2)).permute(0, 2, 3, 1)
        return out


class EqNet(nn.Module):
    """
    Stack of [MessageNet -> dropout -> EqLayer(order -> order)] blocks.
    num_channels[i] is the input width of block i's EqLayer (num_channels[-1] is the final output width);
    num_channels_m[i] lists the hidden widths of block i's MessageNet (may be empty).
    """
    def __init__(self, order, num_channels, num_channels_m, agg='M', activate_agg=False, activate_lin=True,
                 activation='leakyrelu', dropout=0.0, batchnorm=True, masked=True, factorize=True, nobj_avg=49):
        super().__init__()
        L = len(num_channels) - 1
        if len(num_channels_m) != L:
            raise ValueError(f'num_channels_m has {len(num_channels_m)} entries but there are {L} equivariant blocks')
        self.in_dim = num_channels_m[0][0] if num_channels_m[0] else num_channels[0]
        out_dims = [num_channels_m[i + 1][0] if num_channels_m[i + 1] else num_channels[i + 1] for i in range(L - 1)] + [num_channels[-1]]
        self.messages = nn.ModuleList([MessageNet(list(num_channels_m[i]) + [num_channels[i]], order=order, activation=activation,
                                                  batchnorm=batchnorm, masked=masked) for i in range(L)])
        self.eqs = nn.ModuleList([EqLayer(order, order, num_channels[i], out_dims[i], agg=agg, activate_agg=activate_agg,
                                          activate_lin=activate_lin, activation=activation, factorize=factorize, nobj_avg=nobj_avg)
                                  for i in range(L)])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.order = order

    def forward(self, x, mask, nobj, weight=None):
        m = mask if self.order == 1 else (mask.unsqueeze(1) & mask.unsqueeze(2))
        for msg, eq in zip(self.messages, self.eqs):
            x = msg(x, m)
            if self.dropout is not None:
                x = self.dropout(x)
            x = eq(x, mask, nobj, weight)
        return x

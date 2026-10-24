# Based on https://github.com/horacepan/permeqlayers/blob/main/equivariant_layers.py

import pdb
import torch
import torch.nn as nn

def check_shape(x, shape):
    assert len(x.shape) == len(shape)
    for xs, s in zip(x.shape, shape):
        assert xs == s

def masked_mean(x, nobj, dim=None, keepdims=False):
    x = torch.sum(x, dim=dim, keepdims=keepdims)
    if type(dim)!=int:
        nobj = nobj**(len(dim))
    nobj = nobj.view([-1]+[1,]*(len(x.shape)-1))
    x = x / nobj
    return x

def masked_amax(x, nobj, dim=None, keepdims=False):
    x = torch.amax(x, dim=dim, keepdims=keepdims)
    x = x - nobj.log().view([-1]+[1,]*(len(x.shape)-1))
    return x

def masked_amin(x, nobj, dim=None, keepdims=False):
    x = torch.amin(x, dim=dim, keepdims=keepdims)
    x = x + nobj.log().view([-1]+[1,]*(len(x.shape)-1))
    return x

def masked_var(x, nobj, dim=None, keepdims=False):
    var = (masked_mean((x - masked_mean(x, nobj, dim, keepdims=True))**2, nobj, dim, keepdims))
    return var

def masked_sum(x, N, dim=None, keepdims=False):
    if type(dim)!=int:
        N = N**(len(dim))
    return x.sum(dim=dim, keepdims=keepdims) / N

def eops_1_to_1(inputs, normalize=False):
    """inputs: Tensor of shape (Batch, Channel, Num_Particle), aggregation over the last dimension"""
    inputs = inputs.permute(0, 2, 1)
    dim = inputs.shape[-1]
    sums = inputs.sum(dim=-1, keepdim=True) / dim
    op1 = inputs
    op2 = sums.expand(-1, -1, dim)
    return torch.stack([op1, op2], dim=2)

def get_aggregation_fn(aggregation, nobj, nobj_avg):
    if aggregation == 'mean':
        aggregation_fn = masked_mean
    elif aggregation == 'max':
        aggregation_fn = masked_amax
    elif aggregation == 'min':
        aggregation_fn = masked_amin
    elif aggregation == 'var':
        aggregation_fn = masked_var
    elif aggregation == 'sum':
        aggregation_fn = masked_sum
        nobj = nobj_avg
    return aggregation_fn, nobj

def eops_1_to_2_aggregates(inputs, nobj=None, nobj_avg=49, aggregation='mean', weight=None):
    '''
    inputs: B x C x N tensor (already permuted)
    Returns the aggregate B x C x 1 that eops_1_to_2 broadcasts into its basis operations.
    '''
    aggregation_fn, nobj = get_aggregation_fn(aggregation, nobj, nobj_avg)

    if weight is not None:
        sum_all = aggregation_fn(inputs * weight.unsqueeze(1), nobj, dim=2, keepdims=True) # B x C x 1
    else:
        sum_all = aggregation_fn(inputs, nobj, dim=2, keepdims=True) # B x C x 1
    return sum_all

def eops_1_to_2(inputs, nobj=None, nobj_avg=49, aggregation='mean', weight=None):
    '''
    inputs: B x N x C tensor (event, particle, channel)
    '''
    inputs = inputs.permute(0, 2, 1)
    B, C, N = inputs.shape

    sum_all = eops_1_to_2_aggregates(inputs, nobj, nobj_avg, aggregation, weight) # B x C x 1

    op1 = torch.diag_embed(inputs)
    op2 = inputs.unsqueeze(2).expand(-1, -1, N, -1)
    op3 = inputs.unsqueeze(3).expand(-1, -1, -1, N)
    op4 = torch.diag_embed(sum_all.expand(-1, -1, N))
    op5 = sum_all.unsqueeze(3).expand(-1, -1, N, N)
    return torch.stack([op1, op2, op3, op4, op5], dim=2)

def eops_1_to_2_mix(inputs, coefs, nobj=None, nobj_avg=49, aggregation='mean', weight=None, mult=None):
    '''
    Equivalent to torch.einsum('dsb,ndbij->nijs', coefs, eops_1_to_2(inputs, ...) * mult), without materializing the basis.
    inputs: B x N x C, coefs: C x S x 5, mult: None or B x C x 5 multiplier of each basis operation.
    Returns: B x N x N x S
    '''
    inputs = inputs.permute(0, 2, 1)
    sum_all = eops_1_to_2_aggregates(inputs, nobj, nobj_avg, aggregation, weight) # B x C x 1

    # ops 1, 2, 3 are x placed on the diagonal, at [i,j] = x[j], at [i,j] = x[i]; ops 4, 5 are sum_all on the diagonal and everywhere
    vecs = inputs.unsqueeze(2).expand(-1, -1, 3, -1) # B x C x 3 x N
    consts = sum_all.expand(-1, -1, 2) # B x C x 2
    if mult is not None:
        vecs = vecs * mult[:, :, :3, None]
        consts = consts * mult[:, :, 3:]
    vecs = torch.einsum('bdkn,dsk->bkns', vecs, coefs[:, :, :3]) # B x 3 x N x S
    consts = torch.einsum('bdk,dsk->bks', consts, coefs[:, :, 3:]) # B x 2 x S

    output = (vecs[:, 2] + consts[:, 1].unsqueeze(1)).unsqueeze(2) + vecs[:, 1].unsqueeze(1)
    output.diagonal(dim1=1, dim2=2).add_((vecs[:, 0] + consts[:, 0].unsqueeze(1)).transpose(1, 2))
    return output

def eops_2_to_0(inputs, nobj=None, nobj_avg=49, aggregation='mean', weight=None):
    inputs = inputs.permute(0, 3, 1, 2)
    B, C, N, N = inputs.shape

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
    aggregation_fn, nobj = get_aggregation_fn(aggregation, nobj, nobj_avg)

    if weight is not None:
        sum_diag_part = aggregation_fn(diag_part * weight.unsqueeze(1), nobj, dim=2) # B x C x 1
        weight_rows = weight.unsqueeze(1).unsqueeze(2)
        weight_cols = weight.unsqueeze(1).unsqueeze(3)
        sum_all = aggregation_fn(inputs * weight_rows * weight_cols, nobj, dim=(2,3)) # B x C
    else:
        sum_diag_part = aggregation_fn(diag_part, nobj, dim=2)
        sum_all = aggregation_fn(inputs, nobj, dim=(2,3)) # B x C

    op1 = sum_all
    op2 = sum_diag_part
    ops = [op1, op2]
    return torch.stack(ops, dim=2)

def eops_2_to_1(inputs, nobj=None, nobj_avg=49, aggregation='mean', weight=None):
    inputs = inputs.permute(0, 3, 1, 2)
    B, C, N, N = inputs.shape

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
    aggregation_fn, nobj = get_aggregation_fn(aggregation, nobj, nobj_avg)

    if weight is not None:
        weight_rows = weight.unsqueeze(1).unsqueeze(2)
        weight_cols = weight.unsqueeze(1).unsqueeze(3)
        sum_diag_part = aggregation_fn(diag_part * weight.unsqueeze(1), nobj, dim=2, keepdims=True)
        sum_rows = aggregation_fn(inputs * weight_rows, nobj, dim=3)
        sum_cols = aggregation_fn(inputs * weight_cols, nobj, dim=2)
        sum_all = aggregation_fn(inputs * weight_rows * weight_cols, nobj, dim=(2,3)) # B x C
    else:
        sum_diag_part = aggregation_fn(diag_part, nobj, dim=2, keepdims=True)
        sum_rows = aggregation_fn(inputs, nobj, dim=3)
        sum_cols = aggregation_fn(inputs, nobj, dim=2)
        sum_all = aggregation_fn(inputs, nobj, dim=(2,3)) # B x C

    op1 = diag_part 
    op2 = sum_rows
    op3 = sum_cols
    op4 = sum_diag_part.expand(-1, -1, N)
    op5 = sum_all.unsqueeze(2).expand(-1, -1, N)
    ops = [op1, op2, op3, op4, op5]
    return torch.stack(ops, dim=2)

# def eops_2_to_1_sym(inputs, normalize=False):
#     N, D, m, m = inputs.shape
#     dim = inputs.shape[-1]

#     diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
#     sum_diag_part = diag_part.sum(dim=2, keepdims=True) / dim
#     sum_rows = inputs.sum(dim=3) / dim
#     sum_all = inputs.sum(dim=(2,3)) / dim**2

#     op1 = diag_part
#     op2 = sum_diag_part.expand(-1, -1, dim)
#     op3 = sum_rows
#     op4 = sum_all.unsqueeze(2).expand(-1, -1, dim)
#     ops = [op1, op2, op3, op4]
#     return torch.stack([op1, op2, op3, op4], dim=2)

# def eops_2_to_2_sym(inputs, mask=None):
#     inputs = inputs.permute(0, 3, 1, 2)
#     N, D, m, m = inputs.shape
#     dim = inputs.shape[-1]

#     diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1) # N x D x m
#     sum_diag_part = diag_part.sum(dim=2, keepdims=True) / dim # N x D x 1
#     sum_rows = inputs.sum(dim=3) / dim # N x D x m
#     # sum_cols = inputs.sum(dim=2) # N x D x m
#     sum_all = inputs.sum(dim=(2,3)) / dim**2 # N x D

#     ops = [None] * (7 + 1)
#     ops[5]  = torch.diag_embed(diag_part) # N x D x m x m
#     ops[2]  = torch.diag_embed(sum_diag_part.expand(-1, -1, dim))
#     ops[3]  = torch.diag_embed(sum_rows)
#     #ops[4]  = torch.diag_embed(sum_rows)
#     # ops[4]  = torch.diag_embed(sum_cols)
#     ops[4]  = torch.diag_embed(sum_all.unsqueeze(-1).expand(-1, -1, dim))
#     # ops[6]  = sum_cols.unsqueeze(3).expand(-1, -1, -1, dim)
#     # ops[7]  = sum_rows.unsqueeze(3).expand(-1, -1, -1, dim)
#     # ops[8]  = sum_cols.unsqueeze(2).expand(-1, -1, dim, -1)
#     # ops[9]  = sum_rows.unsqueeze(2).expand(-1, -1, dim, -1)

#     ops[1] = inputs
#     # ops[11] = torch.transpose(inputs, 2, 3)
#     # ops[12] = diag_part.unsqueeze(3).expand(-1, -1, -1, dim)
#     # ops[13] = diag_part.unsqueeze(2).expand(-1, -1, dim, -1)
#     ops[6] = sum_diag_part.unsqueeze(3).expand(-1, -1, dim, dim)
#     ops[7] = sum_all.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dim, dim)
#     return torch.stack(ops[1:], dim=2)


def eops_2_to_2_aggregates(inputs, nobj=None, nobj_avg=49, aggregation='mean', weight=None, folklore=False):
    '''
    inputs: B x C x N x N tensor (already permuted)
    Returns the tensors that eops_2_to_2 broadcasts into its basis operations.
    '''
    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1) # B x C x N
    aggregation_fn, nobj = get_aggregation_fn(aggregation, nobj, nobj_avg)

    # sum_diag_part = aggregation_fn(diag_part, nobj, dim=2, keepdims=True) # B x C x 1
    if weight is not None:
        weight_rows = weight.unsqueeze(1).unsqueeze(2)
        weight_cols = weight.unsqueeze(1).unsqueeze(3)
        sum_diag_part = aggregation_fn(diag_part * weight.unsqueeze(1), nobj, dim=2, keepdims=True) # B x C x 1
        sum_rows = aggregation_fn(inputs * weight_rows, nobj, dim=3) # B x C x N
        sum_cols = aggregation_fn(inputs * weight_cols, nobj, dim=2) # B x C x N
        sum_all = aggregation_fn(inputs * weight_rows * weight_cols, nobj, dim=(2,3)) # B x C
    else:
        sum_diag_part = aggregation_fn(diag_part, nobj, dim=2, keepdims=True) # B x C x 1
        sum_rows = aggregation_fn(inputs, nobj, dim=3) # B x C x N
        sum_cols = aggregation_fn(inputs, nobj, dim=2) # B x C x N
        sum_all = aggregation_fn(inputs, nobj, dim=(2,3)) # B x C

    aggregates = {'x': inputs, 'diag_part': diag_part, 'sum_rows': sum_rows, 'sum_cols': sum_cols,
                  'sum_diag_part': sum_diag_part, 'sum_all': sum_all}
    if folklore:
        aggregates['folklore'] = aggregation_fn(torch.nn.LeakyReLU()(inputs.unsqueeze(-2) + inputs.unsqueeze(-3).permute(0,1,2,4,3)), nobj, dim=-1)
    return aggregates

def eops_2_to_2(inputs, nobj=None, nobj_avg=49, aggregation='mean', weight=None, folklore=False):
    inputs = inputs.permute(0, 3, 1, 2)
    B, C, N, N = inputs.shape

    aggregates = eops_2_to_2_aggregates(inputs, nobj, nobj_avg, aggregation, weight, folklore)
    diag_part = aggregates['diag_part']
    sum_rows = aggregates['sum_rows']
    sum_cols = aggregates['sum_cols']
    sum_diag_part = aggregates['sum_diag_part']
    sum_all = aggregates['sum_all']

    ops = [None] * (17 if folklore else 16)

    ops[1] = inputs
    ops[2] = torch.transpose(inputs, 2, 3)
    ops[3] = torch.diag_embed(diag_part) # B x C x N x N
    ops[4] = diag_part.unsqueeze(2).expand(-1, -1, N, -1)
    ops[5] = diag_part.unsqueeze(3).expand(-1, -1, -1, N)

    ops[6]  = torch.diag_embed(sum_cols)
    ops[7]  = sum_cols.unsqueeze(2).expand(-1, -1, N, -1)
    ops[8]  = sum_cols.unsqueeze(3).expand(-1, -1, -1, N)
    ops[9]  = torch.diag_embed(sum_rows)
    ops[10] = sum_rows.unsqueeze(2).expand(-1, -1, N, -1)
    ops[11] = sum_rows.unsqueeze(3).expand(-1, -1, -1, N)
    ops[12] = sum_diag_part.unsqueeze(3).expand(-1, -1, N, N)
    ops[13] = torch.diag_embed(sum_diag_part.expand(-1, -1, N))

    ops[14] = sum_all.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, N)
    ops[15] = torch.diag_embed(sum_all.unsqueeze(-1).expand(-1, -1, N))

    if folklore:
        ops[16] = aggregates['folklore']

    return torch.stack(ops[1:], dim=2)

def eops_2_to_2_mix(inputs, coefs, nobj=None, nobj_avg=49, aggregation='mean', weight=None, mult=None, folklore=False):
    '''
    Equivalent to torch.einsum('dsb,ndbij->nijs', coefs, ops), where ops = eops_2_to_2(inputs, ...) with ops 6 and up multiplied by mult,
    but only the N x N operations (1, 2 and the folklore op 16) are materialized:
    all others are mixed over channels first and broadcast afterwards.
    inputs: B x N x N x C, coefs: C x S x K, mult: None or B x C x (K-5)
    Returns: B x N x N x S
    '''
    inputs = inputs.permute(0, 3, 1, 2)
    aggregates = eops_2_to_2_aggregates(inputs, nobj, nobj_avg, aggregation, weight, folklore)

    # ops 3-11: diag_part, sum_cols, sum_rows, each placed on the diagonal, at [i,j] = v[j], and at [i,j] = v[i]
    vecs = torch.stack([aggregates['diag_part']] * 3 + [aggregates['sum_cols']] * 3 + [aggregates['sum_rows']] * 3, dim=2) # B x C x 9 x N
    # ops 12-15: sum_diag_part, sum_all, each placed everywhere and on the diagonal
    consts = torch.stack([aggregates['sum_diag_part'].squeeze(-1)] * 2 + [aggregates['sum_all']] * 2, dim=2) # B x C x 4
    if mult is not None:
        vecs = torch.cat([vecs[:, :, :3], vecs[:, :, 3:] * mult[:, :, :6, None]], dim=2)
        consts = consts * mult[:, :, 6:10]
    vecs = torch.einsum('bdkn,dsk->bkns', vecs, coefs[:, :, 2:11]).unflatten(1, (3, 3)).sum(1) # B x 3 x N x S: diagonal, v[j], v[i]
    consts = torch.einsum('bdk,dsk->bks', consts, coefs[:, :, 11:15]).unflatten(1, (2, 2)).sum(1) # B x 2 x S: everywhere, diagonal

    output = torch.einsum('bdij,ds->bijs', inputs, coefs[:, :, 0]) + torch.einsum('bdji,ds->bijs', inputs, coefs[:, :, 1])
    if folklore:
        folk = aggregates['folklore'] if mult is None else aggregates['folklore'] * mult[:, :, 10, None, None]
        output = output + torch.einsum('bdij,ds->bijs', folk, coefs[:, :, 15])
    output = output + (vecs[:, 2] + consts[:, 0].unsqueeze(1)).unsqueeze(2) + vecs[:, 1].unsqueeze(1)
    output.diagonal(dim1=1, dim2=2).add_((vecs[:, 0] + consts[:, 1].unsqueeze(1)).transpose(1, 2))
    return output

# def eset_ops_1_to_3(inputs):
#     N, D, m = inputs.shape
#     dim = inputs.shape[-1]
#     sum_all = inputs.sum(dim=-1)
#     x = inputs
#     ops = [None] * 5

#     ops[1] = sum_all.view(N, D, 1, 1, 1).expand(-1, -1, dim, dim, dim) / m
#     ops[2] = x.view(N, D, m, 1, 1).expand(-1, -1, -1, m, m)
#     ops[3] = x.view(N, D, 1, m, 1).expand(-1, -1, m, -1, m)
#     ops[4] = x.view(N, D, 1, 1, m).expand(-1, -1, m, m, -1)
#     return torch.stack(ops[1:], dim=2)

# def eset_ops_3_to_3(inputs, normalize=True):
#     N, D, m, m, m = inputs.shape
#     dim = inputs.shape[-1]

#     # only care about marginalizing over each of the individual indices
#     sum_all = inputs.sum(dim=(-1, -2, -3))
#     sum_c1 = inputs.sum(dim=-1)
#     sum_c2 = inputs.sum(dim=-2)
#     sum_c3 = inputs.sum(dim=-3)

#     sum_c12 = inputs.sum(dim=(-1, -2))
#     sum_c13 = inputs.sum(dim=(-1, -3))
#     sum_c23 = inputs.sum(dim=(-2, -3))

#     # dont really care about diagonal
#     ops = [None] * 20
#     ops[1] = sum_all.view(N, D, 1, 1, 1).expand(-1, -1, dim, dim, dim) / (m * m * m)

#     ops[2] = sum_c1.unsqueeze(-1).expand(-1, -1, -1, -1, m) / (m)
#     ops[3] = sum_c1.unsqueeze(-2).expand(-1, -1, -1, m, -1) / (m)
#     ops[4] = sum_c1.unsqueeze(-3).expand(-1, -1, m, -1, -1) / (m)

#     ops[5] = sum_c2.unsqueeze(-1).expand(-1, -1, -1, -1, m) / (m)
#     ops[6] = sum_c2.unsqueeze(-2).expand(-1, -1, -1, m, -1) / (m)
#     ops[7] = sum_c2.unsqueeze(-3).expand(-1, -1, m, -1, -1) / (m)

#     ops[8]  = sum_c3.unsqueeze(-1).expand(-1, -1, -1, -1, m) / m
#     ops[9]  = sum_c3.unsqueeze(-2).expand(-1, -1, -1, m, -1) / m
#     ops[10] = sum_c3.unsqueeze(-3).expand(-1, -1, m, -1, -1) / m

#     ops[11] = sum_c12.view(N, D, m, 1, 1).expand(-1, -1, -1, m, m) / (m*m)
#     ops[12] = sum_c12.view(N, D, 1, m, 1).expand(-1, -1, m, -1, m) / (m*m)
#     ops[13] = sum_c12.view(N, D, 1, 1, m).expand(-1, -1, m, m, -1) / (m*m)

#     ops[14] = sum_c13.view(N, D, m, 1, 1).expand(-1, -1, -1, m, m) / (m*m)
#     ops[15] = sum_c13.view(N, D, 1, m, 1).expand(-1, -1, m, -1, m) / (m*m)
#     ops[16] = sum_c13.view(N, D, 1, 1, m).expand(-1, -1, m, m, -1) / (m*m)

#     ops[17] = sum_c23.view(N, D, m, 1, 1).expand(-1, -1, -1, m, m) / (m*m)
#     ops[18] = sum_c23.view(N, D, 1, m, 1).expand(-1, -1, m, -1, m) / (m*m)
#     ops[19] = sum_c23.view(N, D, 1, 1, m).expand(-1, -1, m, m, -1) / (m*m)

#     #if normalize:
#     #    ops[1] = torch.divide(ops[1], dim * dim * dim)
#     #    for d in range(2, 11):
#     #        ops[d] = torch.divide(ops[d], dim)

#     #    for d in range(11, 20):
#     #        ops[d] = torch.divide(ops[d], dim * dim)

#     return torch.stack(ops[1:], dim=2)


# def eset_ops_4_to_4(inputs, normalize=False):
#     N, D, m, _, _, _ = inputs.shape
#     sum_all = inputs.sum(dim=(-1, -2, -3, -4))
#     c1, c2, c3 = [], [], []

#     # collapse 1 dim
#     sum_c1 = inputs.sum(dim=-1)
#     sum_c2 = inputs.sum(dim=-2)
#     sum_c3 = inputs.sum(dim=-3)
#     sum_c4 = inputs.sum(dim=-4)
#     c1s = [sum_c1, sum_c2, sum_c3, sum_c4]

#     # collapse 2
#     sum_c12 = inputs.sum(dim=(-1, -2))
#     sum_c13 = inputs.sum(dim=(-1, -3))
#     sum_c14 = inputs.sum(dim=(-1, -4))
#     sum_c23 = inputs.sum(dim=(-2, -3))
#     sum_c24 = inputs.sum(dim=(-2, -4))
#     sum_c34 = inputs.sum(dim=(-3, -4))
#     c2s = [sum_c12, sum_c13, sum_c14, sum_c23, sum_c24, sum_c34]

#     # collapse 3
#     sum_c123 = inputs.sum(dim=(-1, -2, -3))
#     sum_c124 = inputs.sum(dim=(-1, -2, -4))
#     sum_c134 = inputs.sum(dim=(-1, -3, -4))
#     sum_c234 = inputs.sum(dim=(-2, -3, -4))
#     c3s = [sum_c123, sum_c124, sum_c134, sum_c234]

#     # broadcast
#     ops = []
#     ops.append(sum_all.view(N, D, 1, 1, 1, 1).expand(-1, -1, m, m, m, m))

#     # broadcast collapsed 1 dims
#     for c1 in c1s:
#         ops.append(c1.view(N, D, m, m, m, 1).expand(-1, -1, -1, -1, -1, m))
#         ops.append(c1.view(N, D, m, m, 1, m).expand(-1, -1, -1, -1, m, -1))
#         ops.append(c1.view(N, D, m, 1, m, m).expand(-1, -1, -1, m, -1, -1))
#         ops.append(c1.view(N, D, 1, m, m, m).expand(-1, -1, m, -1, -1, -1))

#     for c2 in c2s:
#         ops.append(c2.view(N, D, m, m, 1, 1).expand(-1, -1, -1, -1, m, m))
#         ops.append(c2.view(N, D, m, 1, m, 1).expand(-1, -1, -1, m, -1, m))
#         ops.append(c2.view(N, D, 1, m, m, 1).expand(-1, -1, m, -1, -1, m))
#         ops.append(c2.view(N, D, m, 1, 1, m).expand(-1, -1, -1, m, m, -1))
#         ops.append(c2.view(N, D, 1, m, 1, m).expand(-1, -1, m, -1, m, -1))
#         ops.append(c2.view(N, D, 1, 1, m, m).expand(-1, -1, m, m, -1, -1))

#     for c3 in c3s:
#         ops.append(c3.view(N, D, m, 1, 1, 1).expand(-1, -1, -1, m, m, m))
#         ops.append(c3.view(N, D, 1, m, 1, 1).expand(-1, -1, m, -1, m, m))
#         ops.append(c3.view(N, D, 1, 1, m, 1).expand(-1, -1, m, m, -1, m))
#         ops.append(c3.view(N, D, 1, 1, 1, m).expand(-1, -1, m, m, m, -1))

#     return torch.stack(ops, dim=2)

# if __name__ == '__main__':
#     N = 32
#     D = 16
#     m = 2
#     x = torch.rand(N, D, m)
#     x = torch.rand(N, D, m)
#     x2 = torch.rand(N, D, m, m)
#     x3 = torch.rand(N, D, m, m, m)
#     x4 = torch.rand(N, D, m, m, m, m)
#     o = eops_1_to_1(x)
#     print('1->1 okay')
#     o2 = eops_1_to_2(x)
#     print('1->2 okay')
#     o1 = eops_2_to_1(x2)
#     print('2->1 okay')
#     o22 = eops_2_to_2(x2)
#     print('2->2 okay')

#     t33 = eset_ops_3_to_3(x3)
#     print(t33.shape)

#     t44 = eset_ops_4_to_4(x4)
#     print(t44.shape)

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
        nobj = nobj**len(dim)
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

def masked_sum(x, nobj, dim=None, keepdims=False):
    return x.mean(dim=dim, keepdims=keepdims)

def eops_1_to_1(inputs, normalize=False):
    """inputs: Tensor of shape (Batch, Channel, Num_Atom), aggregation over the last dimension"""
    inputs = inputs.permute(0, 2, 1)
    dim = inputs.shape[-1]
    sums = inputs.sum(dim=-1, keepdim=True) / dim
    op1 = inputs
    op2 = sums.expand(-1, -1, dim)
    return torch.stack([op1, op2], dim=2)

# def eops_1_to_2(inputs, normalize=False):
#     '''
#     inputs: N x D x m tensor
#     '''
#     N, D, m = inputs.shape
#     dim = inputs.shape[-1]
#     sum_all = inputs.sum(dim=2, keepdim=True) # N x D x 1

#     op1 = torch.diag_embed(inputs)
#     op2 = torch.diag_embed(sum_all.expand(-1, -1, dim))
#     op3 = inputs.unsqueeze(2).expand(-1, -1, dim, -1)
#     op4 = inputs.unsqueeze(3).expand(-1, -1, -1, dim)
#     op5 = sum_all.unsqueeze(3).expand(-1, -1, dim, dim)
#     return torch.stack([op1, op2, op3, op4, op5], dim=2)

def eops_2_to_0(inputs, normalize=False):
    N, D, m, m = inputs.shape
    dim = inputs.shape[-1]

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
    sum_diag_part = diag_part.sum(dim=2) / dim
    sum_all = inputs.sum(dim=(2,3)) / dim**2

    op1 = sum_all
    op2 = sum_diag_part
    ops = [op1, op2]
    return torch.stack(ops, dim=2)

def eops_2_to_1(inputs, normalize=False):
    N, D, m, m = inputs.shape
    dim = inputs.shape[-1]

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
    sum_diag_part = diag_part.sum(dim=2, keepdims=True) / dim
    sum_rows = inputs.sum(dim=3) / dim
    sum_cols = inputs.sum(dim=2) / dim
    sum_all = inputs.sum(dim=(2,3)) / dim**2

    op1 = diag_part
    op2 = sum_diag_part.expand(-1, -1, dim)
    op3 = sum_rows
    op4 = sum_cols
    op5 = sum_all.unsqueeze(2).expand(-1, -1, dim)
    ops = [op1, op2, op3, op4, op5]
    return torch.stack(ops, dim=2)

def eops_2_to_1_sym(inputs, normalize=False):
    N, D, m, m = inputs.shape
    dim = inputs.shape[-1]

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
    sum_diag_part = diag_part.sum(dim=2, keepdims=True) / dim
    sum_rows = inputs.sum(dim=3) / dim
    sum_all = inputs.sum(dim=(2,3)) / dim**2

    op1 = diag_part
    op2 = sum_diag_part.expand(-1, -1, dim)
    op3 = sum_rows
    op4 = sum_all.unsqueeze(2).expand(-1, -1, dim)
    ops = [op1, op2, op3, op4]
    return torch.stack([op1, op2, op3, op4], dim=2)

def eops_2_to_2_sym(inputs, mask=None):
    inputs = inputs.permute(0, 3, 1, 2)
    N, D, m, m = inputs.shape
    dim = inputs.shape[-1]

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1) # N x D x m
    sum_diag_part = diag_part.sum(dim=2, keepdims=True) / dim # N x D x 1
    sum_rows = inputs.sum(dim=3) / dim # N x D x m
    # sum_cols = inputs.sum(dim=2) # N x D x m
    sum_all = inputs.sum(dim=(2,3)) / dim**2 # N x D

    ops = [None] * (7 + 1)
    ops[5]  = torch.diag_embed(diag_part) # N x D x m x m
    ops[2]  = torch.diag_embed(sum_diag_part.expand(-1, -1, dim))
    ops[3]  = torch.diag_embed(sum_rows)
    #ops[4]  = torch.diag_embed(sum_rows)
    # ops[4]  = torch.diag_embed(sum_cols)
    ops[4]  = torch.diag_embed(sum_all.unsqueeze(-1).expand(-1, -1, dim))
    # ops[6]  = sum_cols.unsqueeze(3).expand(-1, -1, -1, dim)
    # ops[7]  = sum_rows.unsqueeze(3).expand(-1, -1, -1, dim)
    # ops[8]  = sum_cols.unsqueeze(2).expand(-1, -1, dim, -1)
    # ops[9]  = sum_rows.unsqueeze(2).expand(-1, -1, dim, -1)

    ops[1] = inputs
    # ops[11] = torch.transpose(inputs, 2, 3)
    # ops[12] = diag_part.unsqueeze(3).expand(-1, -1, -1, dim)
    # ops[13] = diag_part.unsqueeze(2).expand(-1, -1, dim, -1)
    ops[6] = sum_diag_part.unsqueeze(3).expand(-1, -1, dim, dim)
    ops[7] = sum_all.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dim, dim)
    return torch.stack(ops[1:], dim=2)

def eops_2_to_2(inputs, nobj=None, aggregation='mean', skip_order_zero=False):
    inputs = inputs.permute(0, 3, 1, 2)
    N, D, m, m = inputs.shape
    dim = inputs.shape[-1]

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1) # N x D x m
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

    sum_diag_part = aggregation_fn(diag_part, nobj, dim=2, keepdims=True) # N x D x 1
    sum_rows = aggregation_fn(inputs, nobj, dim=3) # N x D x m
    sum_cols = aggregation_fn(inputs, nobj, dim=2) # N x D x m
    sum_all = aggregation_fn(inputs, nobj, dim=(2,3)) # N x D

    ops = [None] * (15 + 1)

    if not skip_order_zero:
        ops[1] = inputs
        ops[2] = torch.transpose(inputs, 2, 3)
        ops[3]  = torch.diag_embed(diag_part) # N x D x m x m
        ops[4] = diag_part.unsqueeze(3).expand(-1, -1, -1, dim)
        ops[5] = diag_part.unsqueeze(2).expand(-1, -1, dim, -1)

    ops[6]  = torch.diag_embed(sum_diag_part.expand(-1, -1, dim))
    ops[7]  = torch.diag_embed(sum_rows)
    ops[8]  = torch.diag_embed(sum_cols)
    ops[9]  = sum_cols.unsqueeze(3).expand(-1, -1, -1, dim)
    ops[10]  = sum_rows.unsqueeze(3).expand(-1, -1, -1, dim)
    ops[11]  = sum_cols.unsqueeze(2).expand(-1, -1, dim, -1)
    ops[12]  = sum_rows.unsqueeze(2).expand(-1, -1, dim, -1)
    ops[13] = sum_diag_part.unsqueeze(3).expand(-1, -1, dim, dim)

    ops[14]  = torch.diag_embed(sum_all.unsqueeze(-1).expand(-1, -1, dim))
    ops[15] = sum_all.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dim, dim)

    if skip_order_zero:
        ops = torch.stack(ops[6:], dim=2)
    else:
        ops = torch.stack(ops[1:], dim=2)

    return ops

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

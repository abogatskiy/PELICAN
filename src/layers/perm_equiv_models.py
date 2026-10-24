import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .perm_equiv_layers import eops_1_to_2, eops_2_to_2, eops_2_to_1, eops_2_to_0, eops_1_to_2_mix, eops_2_to_2_mix #, eset_ops_3_to_3, eset_ops_4_to_4, eset_ops_1_to_3, eops_1_to_2
from .generic_layers import get_activation_fn, MessageNet
from .masked_batchnorm import MaskedBatchNorm3d

# class Eq1to1(nn.Module):
#     def __init__(self, in_dim, out_dim, ops_func=None, activation = 'leakyrelu', device=torch.device('cpu'), dtype=torch.float):
#         super(Eq1to1, self).__init__()
#         self.basis_dim = 2
#         self.out_dim = out_dim
#         self.in_dim = in_dim
#         self.activation_fn = get_activation_fn(activation)
#         self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim * self.basis_dim + out_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
#         self.bias = nn.Parameter(torch.zeros(1, 1, out_dim, device=device, dtype=dtype))
#         if ops_func is None:
#             self.ops_func = eops_1_to_1
#         else:
#             self.ops_func = ops_func

#     def forward(self, inputs, mask=None):
#         ops = self.activation_fn(self.ops_func(inputs))
#         output = torch.einsum('dsb, ndbi->nis', self.coefs, ops)
#         output = output + self.bias
#         if mask is not None:
#             output = output * mask
#         return output

AGGREGATIONS = {'s': 'sum', 'm': 'mean', 'x': 'max', 'n': 'min'}

def parse_config(config):
    '''
    config is a single letter from smxnSMXN choosing the aggregation (sum, mean, max, min).
    Upper case additionally scales the aggregated basis operations by (nobj/average_nobj)^alpha with trainable alphas.
    Returns the aggregation name and whether it is scaled.
    '''
    if len(config) != 1 or config.lower() not in AGGREGATIONS:
        raise ValueError("config must be a single letter from smxnSMXN", config)
    return AGGREGATIONS[config.lower()], config.isupper()

def nobj_multiplier(nobj, alphas, average_nobj):
    mult = (nobj).view([-1] + [1] * (alphas.dim() - 1))**alphas
    return mult / (average_nobj**alphas)

def mixing_coefs(layer):
    if layer.factorize:
        return layer.coefs00.unsqueeze(1) * layer.coefs10.unsqueeze(-1) + layer.coefs01.unsqueeze(0) * layer.coefs11.unsqueeze(-1)
    return layer.coefs

# In all layers below, self.alphas is a list with one entry (None for lower case configs),
# which keeps the parameter name alphas.0 that existing checkpoints use.

class Eq2to0(nn.Module):
    def __init__(self, in_dim, out_dim, activate_agg=False, activate_lin=True, activation = 'leakyrelu', config='s', factorize=True, average_nobj=49, device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to0, self).__init__()
        self.device = device
        self.dtype = dtype
        self.activate_agg = activate_agg
        self.activate_lin = activate_lin
        self.activation_fn = get_activation_fn(activation)
        self.config = config
        self.aggregation, self.scaled = parse_config(config)
        self.factorize = factorize

        self.average_nobj = average_nobj                 # 50 is the mean number of particles per event in the toptag dataset; ADJUST FOR YOUR DATASET
        self.basis_dim = 2
        self.alphas = nn.ParameterList([nn.Parameter(torch.rand(1, in_dim, 2, device=device, dtype=dtype)) if self.scaled else None])

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.ops_func = eops_2_to_0
        if factorize:
            self.coefs00 = nn.Parameter(torch.normal(0, np.sqrt(1. / self.basis_dim), (in_dim, self.basis_dim), device=device, dtype=dtype))
            self.coefs01 = nn.Parameter(torch.normal(0, np.sqrt(1. / self.basis_dim), (out_dim, self.basis_dim), device=device, dtype=dtype))            
            self.coefs10 = nn.Parameter(torch.normal(0, np.sqrt(2. / in_dim), (in_dim, out_dim), device=device, dtype=dtype))
            self.coefs11 = nn.Parameter(torch.normal(0, np.sqrt(2. / in_dim), (in_dim, out_dim), device=device, dtype=dtype))
        else:
            self.coefs = nn.Parameter(torch.normal(0, np.sqrt(4./(in_dim * self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        
        self.bias = nn.Parameter(torch.zeros(1, out_dim, device=device, dtype=dtype))
        self.to(device=device, dtype=dtype)

    def forward(self, inputs, mask=None, nobj=None, irc_weight=None):
        '''
        inputs: N x D x m x m
        Returns: N x D x m
        '''
        ops = self.ops_func(inputs, nobj=nobj, nobj_avg=self.average_nobj, aggregation=self.aggregation, weight=irc_weight)
        if self.scaled:
            ops = ops * nobj_multiplier(nobj, self.alphas[0], self.average_nobj)

        if self.activate_agg:
            ops = self.activation_fn(ops)

        output = torch.einsum('dsb,ndb->ns', mixing_coefs(self), ops)

        output = output + self.bias

        if self.activate_lin:
            output = self.activation_fn(output)

        if mask is not None:
            output = output * mask
        return output

class Eq1to2(nn.Module):
    def __init__(self, in_dim, out_dim, activate_agg=False, activate_lin=True, activation = 'leakyrelu', config='s', factorize=False, average_nobj=49, device=torch.device('cpu'), dtype=torch.float):
        super(Eq1to2, self).__init__()
        self.basis_dim = 5
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.activate_agg = activate_agg
        self.activate_lin = activate_lin
        self.activation_fn = get_activation_fn(activation)
        self.config = config
        self.aggregation, self.scaled = parse_config(config)
        self.factorize = factorize

        self.average_nobj = average_nobj                 # 50 is the mean number of particles per event in the toptag dataset; ADJUST FOR YOUR DATASET
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(1, in_dim, 5, 1, 1, device=device, dtype=dtype)) if self.scaled else None])

        self.ops_func = eops_1_to_2
        # Mix the basis without materializing it (see eops_1_to_2_mix) whenever that gives the same result
        self.fast_basis = True

        if factorize:
            self.coefs00 = nn.Parameter(torch.normal(0, np.sqrt(1. / self.basis_dim), (in_dim, self.basis_dim), device=device, dtype=dtype))
            self.coefs01 = nn.Parameter(torch.normal(0, np.sqrt(1. / self.basis_dim), (out_dim, self.basis_dim), device=device, dtype=dtype))            
            self.coefs10 = nn.Parameter(torch.normal(0, np.sqrt(1. / in_dim), (in_dim, out_dim), device=device, dtype=dtype))   # Replace 1. with 2. when using ELU/GELU, leave 1. for LeakyReLU
            self.coefs11 = nn.Parameter(torch.normal(0, np.sqrt(1. / in_dim), (in_dim, out_dim), device=device, dtype=dtype))   # Replace 1. with 2. when using ELU/GELU, leave 1. for LeakyReLU
        else:
            self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2./(in_dim * self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        
        self.bias = nn.Parameter(torch.zeros(out_dim, device=device, dtype=dtype))
        self.to(device=device, dtype=dtype)

    def forward(self, inputs, mask=None, nobj=None, softmask_ir=None, irc_weight=None):
        '''
        inputs: B x N x C
        Returns: B x N x N x C
        '''
        coefs = mixing_coefs(self)
        mult = nobj_multiplier(nobj, self.alphas[0], self.average_nobj) if self.scaled else None # B x C x 5 x 1 x 1

        if self.fast_basis and not self.activate_agg and softmask_ir is None:
            output = eops_1_to_2_mix(inputs, coefs, nobj, self.average_nobj, aggregation=self.aggregation, weight=irc_weight,
                                     mult=None if mult is None else mult[..., 0, 0])
        else:
            ops = self.ops_func(inputs, nobj, self.average_nobj, aggregation=self.aggregation, weight=irc_weight)
            if mult is not None:
                ops = ops * mult
            if softmask_ir is not None:
                ops = torch.cat([ops[:,:,:3], ops[:,:,3:] * softmask_ir], dim=2)
            if self.activate_agg:
                ops = self.activation_fn(ops)
            output = torch.einsum('dsb,ndbij->nijs', coefs, ops)

        output = output + self.bias.view(1,1,-1)

        if self.activate_lin:
            output = self.activation_fn(output)

        if mask is not None:
            output = output * mask
        return output



class Eq2to1(nn.Module):
    def __init__(self, in_dim, out_dim, activate_agg=False, activate_lin=True, activation = 'leakyrelu', config='s', factorize=False, average_nobj=49, device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to1, self).__init__()
        self.basis_dim = 5
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.activate_agg = activate_agg
        self.activate_lin = activate_lin
        self.activation_fn = get_activation_fn(activation)
        self.config = config
        self.aggregation, self.scaled = parse_config(config)
        self.factorize = factorize

        self.average_nobj = average_nobj                 # 50 is the mean number of particles per event in the toptag dataset; ADJUST FOR YOUR DATASET
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(1, in_dim, 5, 1, device=device, dtype=dtype)) if self.scaled else None])

        self.ops_func = eops_2_to_1

        if factorize:
            self.coefs00 = nn.Parameter(torch.normal(0, np.sqrt(1. / self.basis_dim), (in_dim, self.basis_dim), device=device, dtype=dtype))
            self.coefs01 = nn.Parameter(torch.normal(0, np.sqrt(1. / self.basis_dim), (out_dim, self.basis_dim), device=device, dtype=dtype))            
            self.coefs10 = nn.Parameter(torch.normal(0, np.sqrt(1. / in_dim), (in_dim, out_dim), device=device, dtype=dtype))   # Replace 1. with 2. when using ELU/GELU, leave 1. for LeakyReLU
            self.coefs11 = nn.Parameter(torch.normal(0, np.sqrt(1. / in_dim), (in_dim, out_dim), device=device, dtype=dtype))   # Replace 1. with 2. when using ELU/GELU, leave 1. for LeakyReLU
        else:
            self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2./(in_dim * self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        
        self.bias = nn.Parameter(torch.zeros(out_dim, device=device, dtype=dtype))
        self.to(device=device, dtype=dtype)

    def forward(self, inputs, mask=None, nobj=None, softmask_ir=None, irc_weight=None):
        '''
        inputs: B x N x N x C
        Returns: B x N x C
        '''
        ops = self.ops_func(inputs, nobj, self.average_nobj, aggregation=self.aggregation, weight=irc_weight)
        if self.scaled:
            ops = ops * nobj_multiplier(nobj, self.alphas[0], self.average_nobj)
        if softmask_ir is not None:
            ops = torch.cat([ops[:,:,:3], ops[:,:,3:] * softmask_ir], dim=2)

        if self.activate_agg:
            ops = self.activation_fn(ops)

        output = torch.einsum('dsb,ndbi->nis', mixing_coefs(self), ops)

        output = output + self.bias.view(1,1,-1)

        if self.activate_lin:
            output = self.activation_fn(output)

        if mask is not None:
            output = output * mask
        return output

class Eq2to2(nn.Module):
    def __init__(self, in_dim, out_dim, ops_func=None, activate_agg=False, activate_lin=True, activation = 'leakyrelu', config='s', factorize=True, folklore=False, average_nobj=49, device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to2, self).__init__()
        self.device = device
        self.dtype = dtype
        self.activate_agg = activate_agg
        self.activate_lin = activate_lin
        self.activation_fn = get_activation_fn(activation)
        self.config = config
        self.aggregation, self.scaled = parse_config(config)
        self.factorize=factorize
        self.folklore = folklore

        self.average_nobj = average_nobj
        self.basis_dim = 16 if folklore else 15

        # The first 5 basis operations don't aggregate, so only the remaining ones get scaled
        self.alphas = nn.ParameterList([nn.Parameter(torch.rand(1, in_dim, self.basis_dim - 5, 1, 1, device=device, dtype=dtype)) if self.scaled else None])

        self.out_dim = out_dim
        self.in_dim = in_dim
        if factorize:
            self.coefs00 = nn.Parameter(torch.normal(0, np.sqrt(1. / self.basis_dim), (in_dim, self.basis_dim), device=device, dtype=dtype))
            self.coefs01 = nn.Parameter(torch.normal(0, np.sqrt(1. / self.basis_dim), (out_dim, self.basis_dim), device=device, dtype=dtype))            
            self.coefs10 = nn.Parameter(torch.normal(0, np.sqrt(1. / in_dim), (in_dim, out_dim), device=device, dtype=dtype))   # Replace 1. with 2. when using ELU/GELU, leave 1. for LeakyReLU
            self.coefs11 = nn.Parameter(torch.normal(0, np.sqrt(1. / in_dim), (in_dim, out_dim), device=device, dtype=dtype))   # Replace 1. with 2. when using ELU/GELU, leave 1. for LeakyReLU
        else:
            self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2./(in_dim * self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_dim, device=device, dtype=dtype))
        self.diag_bias = nn.Parameter(torch.zeros(out_dim, device=device, dtype=dtype))

        if ops_func is None:
            self.ops_func = eops_2_to_2
        else:
            self.ops_func = ops_func
        # Mix the basis without materializing it (see eops_2_to_2_mix) whenever that gives the same result
        self.fast_basis = ops_func is None

        self.to(device=device, dtype=dtype)

    def forward(self, inputs, mask=None, nobj=None, softmask_ir=None, irc_weight=None):
        coefs = mixing_coefs(self)
        mult = nobj_multiplier(nobj, self.alphas[0], self.average_nobj) if self.scaled else None # B x C x (basis_dim-5) x 1 x 1

        if self.fast_basis and not self.activate_agg and softmask_ir is None:
            output = eops_2_to_2_mix(inputs, coefs, nobj, self.average_nobj, aggregation=self.aggregation, weight=irc_weight,
                                     mult=None if mult is None else mult[..., 0, 0], folklore=self.folklore)
        else:
            ops = self.ops_func(inputs, nobj, self.average_nobj, aggregation=self.aggregation, weight=irc_weight, folklore=self.folklore)
            if mult is not None:
                ops = torch.cat([ops[:, :, :5], ops[:, :, 5:] * mult], dim=2)
            if softmask_ir is not None:
                s = ops.shape
                ops = ops * torch.cat([torch.ones([s[0],1,3,s[-2],s[-1]], device=ops.device),
                                       softmask_ir.expand([s[0],1,12,s[-2],s[-1]])], 2)
            if self.activate_agg:
                ops = self.activation_fn(ops)
            output = torch.einsum('dsb,ndbij->nijs', coefs, ops)

        diag_eye = torch.eye(inputs.shape[1], device=self.device, dtype=self.dtype).unsqueeze(0).unsqueeze(-1)
        diag_bias = diag_eye.multiply(self.diag_bias.view(1,1,1,-1))
        output = output + self.bias.view(1,1,1,-1) + diag_bias

        if self.activate_lin:
            output = self.activation_fn(output)

        if mask is not None:
            output = output * mask
        return output

# class Net1to1(nn.Module):
#     def __init__(self, num_channels, ops_func=None, activation='leakyrelu', batchnorm=None, device=torch.device('cpu'), dtype=torch.float):
#         super(Net1to1, self).__init__()
#         self.eq_layers = nn.ModuleList([Eq1to1(num_channels[i], num_channels[i + 1], ops_func, activation, device=device, dtype=dtype) for i in range(len(num_channels) - 1)])
#         self.message_layers = nn.ModuleList(([MessageNet(num_ch, activation=activation, batchnorm=batchnorm, device=device, dtype=dtype) for num_ch in num_channels[1:]]))
#         self.to(device=device, dtype=dtype)

#     def forward(self, x, mask=None):
#         for (layer, message) in zip(self.eq_layers, self.message_layers):
#             x = message(layer(x, mask), mask)
#         return x

class Net2to2(nn.Module):
    def __init__(self, num_channels, num_channels_m, ops_func=None, activate_agg=False, activate_lin=True,
                 activation='leakyrelu', dropout=True, drop_rate=0.25, batchnorm=None,
                 config='s', average_nobj=49, factorize=False, masked=True, device=torch.device('cpu'), dtype=torch.float):
        super(Net2to2, self).__init__()
        
        self.masked = masked
        self.num_channels = num_channels
        self.num_channels_message = num_channels_m
        self.activate_agg = activate_agg
        self.activate_lin = activate_lin
        self.batchnorm = batchnorm
        num_layers = len(num_channels) - 1
        self.in_dim = num_channels_m[0][0] if len(num_channels_m[0]) > 0 else num_channels[0]

        eq_out_dims = [num_channels_m[i+1][0] if len(num_channels_m[i+1]) > 0 else num_channels[i+1] for i in range(num_layers-1)] + [num_channels[-1]]

        self.dropout = dropout
        if dropout:
            self.dropout_layer = nn.Dropout(drop_rate)

        self.message_layers = nn.ModuleList(([MessageNet(num_channels_m[i]+[num_channels[i],], activation=activation, batchnorm=batchnorm, masked=masked, device=device, dtype=dtype) for i in range(num_layers)]))        
        self.eq_layers = nn.ModuleList([Eq2to2(num_channels[i], eq_out_dims[i], ops_func, activate_agg=activate_agg, activate_lin=activate_lin, activation=activation, config=config, average_nobj=average_nobj, factorize=factorize, device=device, dtype=dtype) for i in range(num_layers)])
        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None, nobj=None, softmask_ir=None, irc_weight=None):
        '''
        x: N x m x m x in_dim
        Returns: N x m x m x out_dim
        '''
        assert (x.shape[-1] == self.in_dim), "Input dimension of Net2to2 doesn't match the dimension of the input tensor"

        for agg, msg in zip(self.eq_layers, self.message_layers):
            x = msg(x, mask)
            if self.dropout: x = self.dropout_layer(x.permute(0,3,1,2)).permute(0,2,3,1)
            x = agg(x, mask, nobj, irc_weight=irc_weight)
            # if self.dropout: x = self.dropout_layer(x.permute(0,3,1,2)).permute(0,2,3,1)
        return x

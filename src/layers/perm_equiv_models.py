import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .perm_equiv_layers import eops_1_to_2, eops_2_to_2, eops_2_to_1, eops_2_to_0 #, eset_ops_3_to_3, eset_ops_4_to_4, eset_ops_1_to_3, eops_1_to_2
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

class Eq2to0(nn.Module):
    def __init__(self, in_dim, out_dim, activate_agg=False, activate_lin=True, activation = 'leakyrelu', config='s', factorize=True, average_nobj=49, device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to0, self).__init__()
        self.device = device
        self.dtype = dtype
        self.activate_agg = activate_agg
        self.activate_lin = activate_lin
        self.activation_fn = get_activation_fn(activation)
        self.config = config
        self.factorize = factorize

        self.average_nobj = average_nobj                 # 50 is the mean number of particles per event in the toptag dataset; ADJUST FOR YOUR DATASET
        self.basis_dim = 2 * len(config)
        self.alphas = nn.ParameterList([None] * len(config))
        for i, char in enumerate(config):
            if char in ['M', 'X', 'N']:
                self.alphas[i] = nn.Parameter(torch.rand(1, in_dim, 2, device=device, dtype=dtype))
            elif char=='S':
                self.alphas[i] = nn.Parameter(torch.rand(1, in_dim, 2, device=device, dtype=dtype))

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
        d = {'s': 'sum', 'm': 'mean', 'x': 'max', 'n': 'min'}

        ops = []
        for i, char in enumerate(self.config):
            if char in ['s', 'm', 'x', 'n']:
                op = self.ops_func(inputs, nobj=nobj, nobj_avg=self.average_nobj, aggregation=d[char], weight=irc_weight)
            elif char in ['S', 'M', 'X', 'N']:
                op = self.ops_func(inputs, nobj=nobj, nobj_avg=self.average_nobj, aggregation=d[char.lower()], weight=irc_weight)
                mult = (nobj).view([-1,1,1])**self.alphas[i]
                mult = mult / (self.average_nobj** self.alphas[i])
                op = op * mult            
            else:
                raise ValueError("args.config must consist of the following letters: smxnSMXN", self.config)
            ops.append(op)


        ops = torch.cat(ops, dim=2)

        if self.activate_agg:
            ops = self.activation_fn(ops)

        if self.factorize:
            coefs = self.coefs00.unsqueeze(1) * self.coefs10.unsqueeze(-1) + self.coefs01.unsqueeze(0) * self.coefs11.unsqueeze(-1)
        else:
            coefs = self.coefs

        output = torch.einsum('dsb,ndb->ns', coefs, ops)

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
        self.factorize = factorize

        self.average_nobj = average_nobj                 # 50 is the mean number of particles per event in the toptag dataset; ADJUST FOR YOUR DATASET
        self.alphas = nn.ParameterList([None] * len(config))
        for i, char in enumerate(config):
            if char in ['M', 'X', 'N']:
                self.alphas[i] = nn.Parameter(torch.zeros(1, in_dim, 5, 1, 1, device=device, dtype=dtype))
            elif char=='S':
                self.alphas[i] = nn.Parameter(torch.zeros(1, in_dim, 5, 1, 1, device=device, dtype=dtype))

        self.ops_func = eops_1_to_2

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
        d = {'s': 'sum', 'm': 'mean', 'x': 'max', 'n': 'min'}

        ops = []
        for i, char in enumerate(self.config):
            if char in ['s', 'm', 'x', 'n']:
                op = self.ops_func(inputs, nobj, self.average_nobj, aggregation=d[char], weight=irc_weight)
            elif char in ['S', 'M', 'X', 'N']:
                op = self.ops_func(inputs, nobj, self.average_nobj, aggregation=d[char.lower()], weight=irc_weight)
                mult = (nobj).view([-1,1,1,1,1])**self.alphas[i]
                mult = mult / (self.average_nobj** self.alphas[i])
                op = op * mult
            else:
                raise ValueError("args.config must consist of the following letters: smxnSMXN", self.config)
            if softmask_ir is not None:
                op = torch.cat([op[:,:,:3], op[:,:,3:] * softmask_ir], dim=2)
            ops.append(op)
        ops = torch.cat(ops, dim=2)

        if self.activate_agg:
            ops = self.activation_fn(ops)

        if self.factorize:
            coefs = self.coefs00.unsqueeze(1) * self.coefs10.unsqueeze(-1) + self.coefs01.unsqueeze(0) * self.coefs11.unsqueeze(-1)
        else:
            coefs = self.coefs

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
        self.factorize = factorize

        self.average_nobj = average_nobj                 # 50 is the mean number of particles per event in the toptag dataset; ADJUST FOR YOUR DATASET
        self.alphas = nn.ParameterList([None] * len(config))
        for i, char in enumerate(config):
            if char in ['M', 'X', 'N']:
                self.alphas[i] = nn.Parameter(torch.zeros(1, in_dim, 5, 1, device=device, dtype=dtype))
            elif char=='S':
                self.alphas[i] = nn.Parameter(torch.zeros(1, in_dim, 5, 1, device=device, dtype=dtype))

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
        d = {'s': 'sum', 'm': 'mean', 'x': 'max', 'n': 'min'}

        ops = []
        for i, char in enumerate(self.config):
            if char in ['s', 'm', 'x', 'n']:
                op = self.ops_func(inputs, nobj, self.average_nobj, aggregation=d[char], weight=irc_weight)
            elif char in ['S', 'M', 'X', 'N']:
                op = self.ops_func(inputs, nobj, self.average_nobj, aggregation=d[char.lower()], weight=irc_weight)
                mult = (nobj).view([-1,1,1,1])**self.alphas[i]
                mult = mult / (self.average_nobj** self.alphas[i])
                op = op * mult
            else:
                raise ValueError("args.config must consist of the following letters: smxnSMXN", self.config)
            if softmask_ir is not None:
                op = torch.cat([op[:,:,:3], op[:,:,3:] * softmask_ir], dim=2)
            ops.append(op)
        ops = torch.cat(ops, dim=2)

        if self.activate_agg:
            ops = self.activation_fn(ops)

        if self.factorize:
            coefs = self.coefs00.unsqueeze(1) * self.coefs10.unsqueeze(-1) + self.coefs01.unsqueeze(0) * self.coefs11.unsqueeze(-1)
        else:
            coefs = self.coefs

        output = torch.einsum('dsb,ndbi->nis', coefs, ops)

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
        self.factorize=factorize
        self.folklore = folklore

        self.average_nobj = average_nobj
        self.basis_dim = (16 if folklore else 15) + (11 if folklore else 10) * (len(config) - 1)
        # self.basis_dim = 6

        self.alphas = nn.ParameterList([None] * len(config))
        self.dummy_alphas = torch.zeros(1, in_dim, 5, 1, 1, device=device, dtype=dtype)
        # countM = 0
        for i, char in enumerate(config):
            if char in ['M', 'X', 'N']:
                self.alphas[i] = nn.Parameter(torch.rand(1, in_dim, 11 if folklore else 10,  1, 1, device=device, dtype=dtype))
            elif char=='S':
                self.alphas[i] = nn.Parameter(torch.rand(1, in_dim, 11 if folklore else 10,  1, 1, device=device, dtype=dtype))

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

        self.to(device=device, dtype=dtype)

    def forward(self, inputs, mask=None, nobj=None, softmask_ir=None, irc_weight=None):

        d = {'s': 'sum', 'm': 'mean', 'x': 'max', 'n': 'min'}

        ops=[]
        for i, char in enumerate(self.config):
            if char.lower() in ['s', 'm', 'x', 'n']:
                op = self.ops_func(inputs, nobj, self.average_nobj, aggregation=d[char.lower()], weight=irc_weight, skip_order_zero=False if i==0 else True, folklore = self.folklore)
                if char in ['S', 'M', 'X', 'N']:
                    if i==0:
                        alphas = torch.cat([self.dummy_alphas, self.alphas[0]], dim=2)
                    else:
                        alphas = self.alphas[i]
                    mult = (nobj).view([-1,1,1,1,1])**alphas
                    mult = mult / (self.average_nobj**alphas)
                    op = op * mult
            else:
                raise ValueError("args.config must consist of the following letters: smxnSMXN", self.config)
            if softmask_ir is not None:
                s = op.shape
                softmask_ir = torch.cat([
                                        torch.ones([s[0],1,3,s[-2],s[-1]], device=op.device), 
                                        softmask_ir.expand([s[0],1,12,s[-2],s[-1]])
                                        ], 2)
                op = op*softmask_ir
            ops.append(op)

        ops = torch.cat(ops, dim=2)

        if self.activate_agg:
            ops = self.activation_fn(ops)

        if self.factorize:
            coefs = self.coefs00.unsqueeze(1) * self.coefs10.unsqueeze(-1) + self.coefs01.unsqueeze(0) * self.coefs11.unsqueeze(-1)
        else:
            coefs = self.coefs
        
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models import MLP
# from lgn.nn.perm_equiv_layers import ops_2_to_1, ops_1_to_1,eops_2_to_2, set_ops_3_to_3, set_ops_4_to_4, ops_1_to_2
from .perm_equiv_layers import eops_1_to_1, eops_2_to_1_sym, eops_2_to_2_sym, eops_2_to_2, eops_2_to_1, eops_2_to_0 #, eset_ops_3_to_3, eset_ops_4_to_4, eset_ops_1_to_3, eops_1_to_2
from .generic_layers import get_activation_fn, MessageNet, BasicMLP


class Eq1to1(nn.Module):
    def __init__(self, in_dim, out_dim, ops_func=None, activation = 'leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(Eq1to1, self).__init__()
        self.basis_dim = 2
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.activation_fn = get_activation_fn(activation)
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim + out_dim + self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(1, 1, out_dim, device=device, dtype=dtype))
        if ops_func is None:
            self.ops_func = eops_1_to_1
        else:
            self.ops_func = ops_func

    def forward(self, inputs, mask=None):
        ops = self.activation_fn(self.ops_func(inputs))
        output = torch.einsum('dsb, ndbi->nis', self.coefs, ops)
        output = output + self.bias
        if mask is not None:
            output = output * mask
        return output

class Eq2to0(nn.Module):
    def __init__(self, in_dim, out_dim, activation = 'leakyrelu', sym=False, device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to0, self).__init__()
        self.basis_dim = 2
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.activation_fn = get_activation_fn(activation)
        self.ops_func = eops_2_to_0
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim + out_dim + self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(1, out_dim, device=device, dtype=dtype))

    def forward(self, inputs, mask=None):
        '''
        inputs: N x D x m x m
        Returns: N x D x m
        '''
        inputs = inputs.permute(0, 3, 1, 2)
        ops = self.activation_fn(self.ops_func(inputs))
        output = torch.einsum('dsb,ndb->ns', self.coefs, ops)
        output = output + self.bias
        if mask is not None:
            output = output * mask
        return output

class Eq2to1(nn.Module):
    def __init__(self, in_dim, out_dim, activation = 'leakyrelu', sym=False, device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to1, self).__init__()
        self.basis_dim = 4 if sym else 5
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.activation_fn = get_activation_fn(activation)
        self.ops_func = eops_2_to_1_sym if sym else eops_2_to_1
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim + out_dim + self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1, device=device, dtype=dtype))

    def forward(self, inputs, mask=None):
        '''
        inputs: N x D x m x m
        Returns: N x D x m
        '''
        inputs = inputs.permute(0, 3, 1, 2)
        # ops = self.activation_fn(self.ops_func(inputs))
        output = torch.einsum('dsb,ndbi->nsi', self.coefs, ops)
        output = output + self.bias
        if mask is not None:
            output = output.permute(0, 2, 1) * mask
        return output

class Eq2to2(nn.Module):
    def __init__(self, in_dim, out_dim, ops_func=None, activation = 'leakyrelu', sym=False, device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to2, self).__init__()
        self.device = device
        self.dtype = dtype
        self.activation_fn = get_activation_fn(activation)
        self.basis_dim = (7 if sym else 15) * 1

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim + out_dim + self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, out_dim, device=device, dtype=dtype))
        self.diag_bias = nn.Parameter(torch.zeros(1, 1, 1, out_dim, device=device, dtype=dtype))
        self.diag_eyes = {}

        self.diag_eye = None #torch.eye(n).unsqueeze(0).unsqueeze(0).to(device)
        if ops_func is None:
            self.ops_func = eops_2_to_2_sym if sym else eops_2_to_2
        else:
            self.ops_func = ops_func

    def forward(self, inputs, mask=None, nobj=None):

        # ops = self.ops_func(inputs, nobj, aggregation='mean') * ((1+nobj).log().view([-1,1,1,1,1]) / 3.845)
        ops = self.ops_func(inputs, nobj, aggregation='sum')
        # ops0 = [self.ops_func(inputs, nobj, aggregation=agg) for agg in ['mean','max','min']]
        # ops1 = [y for x in ops0 for y in [x, x* ((1+nobj).log().view([-1,1,1,1,1]) / 3.845)]] #, x / ((1+nobj).log().view([-1,1,1,1,1]) / 3.845)
        # ops = torch.cat(ops1, dim=2)
        # ops = ops0[0] * ((1+nobj).log().view([-1,1,1,1,1]) / 3.845)
        # ops = torch.cat(ops, dim=2)

        # ops = self.activation_fn(ops)
        output = torch.einsum('dsb,ndbij->nijs', self.coefs, ops)

        diag_eye = torch.eye(inputs.shape[1], device=self.device, dtype=self.dtype).unsqueeze(0).unsqueeze(-1)
        diag_bias = diag_eye.multiply(self.diag_bias)

        output = output + self.bias + diag_bias
        output = self.activation_fn(output)

        if mask is not None:
            output = output * mask
        return output

class Net1to1(nn.Module):
    def __init__(self, num_channels, ops_func=None, activation='leakyrelu', batchnorm=None, device=torch.device('cpu'), dtype=torch.float):
        super(Net1to1, self).__init__()
        self.eq_layers = nn.ModuleList([Eq1to1(num_channels[i], num_channels[i + 1], ops_func, activation, device=device, dtype=dtype) for i in range(len(num_channels) - 1)])
        self.message_layers = nn.ModuleList(([MessageNet(num_ch, activation=activation, batchnorm=batchnorm, device=device, dtype=dtype) for num_ch in num_channels[1:]]))

    def forward(self, x, mask=None):
        for (layer, message) in zip(self.eq_layers, self.message_layers):
            x = message(layer(x, mask), mask)
        return x

class Net2to2(nn.Module):
    def __init__(self, num_channels, num_channels_message, ops_func=None, message_depth=2, activation='leakyrelu', batchnorm=None, sym=False, device=torch.device('cpu'), dtype=torch.float):
        super(Net2to2, self).__init__()
        self.message_depth = message_depth
        self.num_channels = num_channels
        self.num_channels_message = num_channels_message
        num_layers = len(num_channels)-1
        self.eq_layers = nn.ModuleList([Eq2to2(num_channels[i], num_channels[i+1], ops_func, activation, sym=sym, device=device, dtype=dtype) for i in range(num_layers)])
        # self.significance = nn.ModuleList([BasicMLP([num_channels[i+1], 1], activation='sigmoid', device=device, dtype=dtype) for i in range(num_layers)])
        if message_depth > 0:
            self.message_layers = nn.ModuleList(([MessageNet([num_channels[i],]+num_channels_message+[num_channels[i],], depth=message_depth, activation=activation, batchnorm=batchnorm, device=device, dtype=dtype) for i in range(num_layers)]))

    def forward(self, x, mask=None, nobj=None):
        '''
        x: N x d x m x m
        Returns: N x m x m x out_dim
        '''
        if self.message_depth > 0:
            for layer, message in zip(self.eq_layers, self.message_layers):
                # x = sig(x) * x
                x = layer(message(x, mask), mask, nobj)
        else:
            for layer in self.eq_layers:
                # x = sig(x) * x
                x = layer(x, mask)
        return x

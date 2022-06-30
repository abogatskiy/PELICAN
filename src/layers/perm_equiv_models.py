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
        self.to(device=device, dtype=dtype)

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
        self.to(device=device, dtype=dtype)

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
    def __init__(self, in_dim, out_dim, ops_func=None, activate_agg=False, activate_lin=True, activation = 'leakyrelu', sym=False, config='s', device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to2, self).__init__()
        self.device = device
        self.dtype = dtype
        self.activate_agg = activate_agg
        self.activate_lin = activate_lin
        self.activation_fn = get_activation_fn(activation)
        self.basis_dim = (7 if sym else 15) * len(config)

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
        self.config = config
        self.to(device=device, dtype=dtype)

    def forward(self, inputs, mask=None, nobj=None):

        d = {'s': 'sum', 'm': 'mean', 'x': 'max', 'n': 'min'}
        ops = [self.ops_func(inputs, nobj, aggregation=d[char]) for char in self.config if char in ['s', 'm', 'x', 'n']]
        ops = ops+[self.ops_func(inputs, nobj, aggregation=d[char.lower()]) * ((1+nobj).log().view([-1,1,1,1,1]) / 3.845) for char in self.config if char in ['S', 'M', 'X', 'N']]
        ops = torch.cat(ops, dim=2)
        
        if self.activate_agg:
            ops = self.activation_fn(ops)

        output = torch.einsum('dsb,ndbij->nijs', self.coefs, ops)

        diag_eye = torch.eye(inputs.shape[1], device=self.device, dtype=self.dtype).unsqueeze(0).unsqueeze(-1)
        diag_bias = diag_eye.multiply(self.diag_bias)

        output = output + self.bias + diag_bias

        if self.activate_lin:
            output = self.activation_fn(output)

        if mask is not None:
            output = output * mask
        return output

class Net1to1(nn.Module):
    def __init__(self, num_channels, ops_func=None, activation='leakyrelu', batchnorm=None, device=torch.device('cpu'), dtype=torch.float):
        super(Net1to1, self).__init__()
        self.eq_layers = nn.ModuleList([Eq1to1(num_channels[i], num_channels[i + 1], ops_func, activation, device=device, dtype=dtype) for i in range(len(num_channels) - 1)])
        self.message_layers = nn.ModuleList(([MessageNet(num_ch, activation=activation, batchnorm=batchnorm, device=device, dtype=dtype) for num_ch in num_channels[1:]]))
        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None):
        for (layer, message) in zip(self.eq_layers, self.message_layers):
            x = message(layer(x, mask), mask)
        return x

class Net2to2(nn.Module):
    def __init__(self, num_channels, num_channels_m, ops_func=None, activate_agg=False, activate_lin=True, activation='leakyrelu', batchnorm=None, sig=False, sym=False, config='s', device=torch.device('cpu'), dtype=torch.float):
        super(Net2to2, self).__init__()
        
        self.sig = sig
        self.num_channels = num_channels
        self.num_channels_message = num_channels_m
        num_layers = len(num_channels) - 1
        # self.eq_layers = nn.ModuleList([Eq2to2(num_channels[i], num_channels[i+1], ops_func, activate_agg=activate_agg, activate_lin=activate_lin, activation=activation, sym=sym, config=config, device=device, dtype=dtype) for i in range(num_layers)])
        self.in_dim = num_channels_m[0][0] if len(num_channels_m[0]) > 0 else num_channels[0]

        eq_out_dims = [num_channels_m[i+1][0] if len(num_channels_m[i+1]) > 0 else num_channels[i+1] for i in range(num_layers-1)] + [num_channels[-1]]

        self.message_layers = nn.ModuleList(([MessageNet(num_channels_m[i]+[num_channels[i],], activation=activation, batchnorm=batchnorm, device=device, dtype=dtype) for i in range(num_layers)]))        
        if sig: 
            self.attention = nn.ModuleList([nn.Linear(num_channels[i], 1, device=device, dtype=dtype) for i in range(num_layers)])
            self.normlayers = nn.ModuleList([nn.LayerNorm(num_channels[i], device=device, dtype=dtype) for i in range(num_layers)])
        self.eq_layers = nn.ModuleList([Eq2to2(num_channels[i], eq_out_dims[i], ops_func, activate_agg=activate_agg, activate_lin=activate_lin, activation=activation, sym=sym, config=config, device=device, dtype=dtype) for i in range(num_layers)])
        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None, nobj=None):
        '''
        x: N x d x m x m
        Returns: N x m x m x out_dim
        '''

        assert (x.shape[-1] == self.in_dim), "Input dimension of Net2to2 doesn't match the dimension of the input tensor"
        if self.sig: 
            for layer, message, sig, normlayer in zip(self.eq_layers, self.message_layers, self.attention, self.normlayers):
                m = message(x, mask)        # form messages at each of the NxN nodes
                y = sig(x)                  # compute the dot product with the attention vector over the channel dim
                # yy = torch.exp(y - y.amax(dim=(1,2), keepdim=True)) * mask  # apply softmax over NxN particles taking into account the mask (normalized in the next line)
                # ms = yy / yy.sum(dim=(1,2), keepdim=True)  
                ms = y.sigmoid() * mask
                z = normlayer(ms * m)       # apply LayerNorm, i.e. normalize over the channel dimension
                x = layer(z, mask, nobj)   # apply the permutation-equivariant layer
        else:
            for layer, message in zip(self.eq_layers, self.message_layers):
                x = layer(message(x, mask), mask, nobj)
        return x

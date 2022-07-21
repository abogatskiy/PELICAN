import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models import MLP
# from lgn.nn.perm_equiv_layers import ops_2_to_1, ops_1_to_1,eops_2_to_2, set_ops_3_to_3, set_ops_4_to_4, ops_1_to_2
from .perm_equiv_layers import eops_1_to_1, eops_2_to_2, eops_2_to_1, eops_2_to_0 #, eset_ops_3_to_3, eset_ops_4_to_4, eset_ops_1_to_3, eops_1_to_2
from .generic_layers import get_activation_fn, MessageNet, BasicMLP


class Eq1to1(nn.Module):
    def __init__(self, in_dim, out_dim, ops_func=None, activation = 'leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(Eq1to1, self).__init__()
        self.basis_dim = 2
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.activation_fn = get_activation_fn(activation)
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim * self.basis_dim + out_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
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
    def __init__(self, in_dim, out_dim, activate_agg=False, activate_lin=True, activation = 'leakyrelu', ir_safe=False, config='s', device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to0, self).__init__()
        self.device = device
        self.dtype = dtype
        self.activate_agg = activate_agg
        self.activate_lin = activate_lin
        self.activation_fn = get_activation_fn(activation)
        self.ir_safe = ir_safe
        self.config = config

        self.average_nobj = 49                 # 50 is the mean number of particles per event in the toptag dataset; ADJUST FOR YOUR DATASET
        self.basis_dim = 2 * len(config)
        self.alphas = nn.ParameterList([None] * len(config))
        # self.betas = nn.ParameterList([None] * len(config))
        self.betas = [None] * len(config)
        # countM = 0
        for i, char in enumerate(config):
            if char in ['M', 'X', 'N']:
                self.alphas[i] = nn.Parameter(torch.zeros(1, 1, 2, device=device, dtype=dtype))
                self.betas[i] = torch.zeros([1, 1, 2], device=device, dtype=dtype)
            elif char=='S':
                self.alphas[i] = nn.Parameter(torch.zeros(1, in_dim, 2, device=device, dtype=dtype))
                self.betas[i] = torch.zeros([1, 1, 2], device=device, dtype=dtype)
                # self.betas[i] = torch.cat([(100/self.average_nobj)    * torch.ones( 1, 1, 1, device=device, dtype=dtype),
                #                            (100/self.average_nobj)**2 * torch.ones( 1, 1, 1, device=device, dtype=dtype)], dim=2).log()
            # elif char == 'M':
            #     countM += 1
            #     if countM > 1:
            #         # self.betas[i] = 1 # nn.Parameter(torch.randn( 1, 1, 2, device=device, dtype=dtype))
            #         self.betas[i] = torch.zeros( 1, 1, 2, device=device, dtype=dtype)
            #         self.alphas[i] = nn.Parameter(torch.zeros(1, in_dim, 2, device=device, dtype=dtype))
            #     else:
            #         # self.betas[i] = nn.Parameter(torch.cat([(self.average_nobj/128)    * torch.ones( 1, 1, 1, device=device, dtype=dtype),
            #         #                                         (self.average_nobj/128)**2 * torch.ones( 1, 1, 1, device=device, dtype=dtype)], dim=2).log())
            #         self.betas[i] = torch.cat([(self.average_nobj/128)    * torch.ones( 1, 1, 1, device=device, dtype=dtype),
            #                                                (self.average_nobj/128)**2 * torch.ones( 1, 1, 1, device=device, dtype=dtype)], dim=2).log()
            #         self.alphas[i] = nn.Parameter(torch.cat([torch.ones(    1, in_dim, 1, device=device, dtype=dtype),
            #                                                 2 * torch.ones(1, in_dim, 1, device=device, dtype=dtype)], dim=2))

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.ops_func = eops_2_to_0
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2./(in_dim * self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        if not ir_safe:
            self.bias = nn.Parameter(torch.zeros(1, out_dim, device=device, dtype=dtype))
        self.to(device=device, dtype=dtype)

    def forward(self, inputs, mask=None, nobj=None):
        '''
        inputs: N x D x m x m
        Returns: N x D x m
        '''
        d = {'s': 'sum', 'm': 'mean', 'x': 'max', 'n': 'min'}

        ops = []
        for i, char in enumerate(self.config):
            if char in ['s', 'm', 'x', 'n']:
                ops.append(self.ops_func(inputs, nobj=nobj, aggregation=d[char]))
            elif char in ['S', 'M', 'X', 'N']:
                ops.append(self.ops_func(inputs, nobj=nobj, aggregation=d[char.lower()]))
                mult = self.betas[i].exp() * (nobj).view([-1,1,1])**self.alphas[i]
                mult = mult / (self.average_nobj** self.alphas[i])
                ops[i] = ops[i] * mult            
            else:
                raise ValueError("args.config must consist of the following letters: smxnSMXN", self.config)

        ops = torch.cat(ops, dim=2)

        if self.activate_agg:
            ops = self.activation_fn(ops)

        output = torch.einsum('dsb,ndb->ns', self.coefs, ops)
        if not self.ir_safe:
            output = output + self.bias

        if self.activate_lin:
            output = self.activation_fn(output)

        if mask is not None:
            output = output * mask
        return output

class Eq2to1(nn.Module):
    def __init__(self, in_dim, out_dim, activate_agg=False, activate_lin=True, activation = 'leakyrelu', ir_safe=False, config='s', device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to1, self).__init__()
        self.basis_dim = 5
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.activate_agg = activate_agg
        self.activate_lin = activate_lin
        self.activation_fn = get_activation_fn(activation)
        self.ir_safe = ir_safe
        self.config = config

        self.average_nobj = 49                 # 50 is the mean number of particles per event in the toptag dataset; ADJUST FOR YOUR DATASET
        self.alphas = nn.ParameterList([None] * len(config))
        for i, char in enumerate(config):
            if char in ['M', 'X', 'N']:
                self.alphas[i] = nn.Parameter(torch.zeros(1, 1, 2, device=device, dtype=dtype))
            elif char=='S':
                self.alphas[i] = nn.Parameter(torch.zeros(1, in_dim, 2, device=device, dtype=dtype))

        self.ops_func = eops_2_to_1
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2./(in_dim * self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(1, 1, out_dim, device=device, dtype=dtype))
        self.to(device=device, dtype=dtype)

    def forward(self, inputs, mask=None, nobj=None):
        '''
        inputs: N x D x m x m
        Returns: N x D x m
        '''
        d = {'s': 'sum', 'm': 'mean', 'x': 'max', 'n': 'min'}

        ops = []
        for i, char in enumerate(self.config):
            if char in ['s', 'm', 'x', 'n']:
                ops.append(self.ops_func(inputs, nobj=nobj, aggregation=d[char]))
            elif char in ['S', 'M', 'X', 'N']:
                ops.append(self.ops_func(inputs, nobj=nobj, aggregation=d[char.lower()]))
                mult = (nobj).view([-1,1,1])**self.alphas[i]
                mult = mult / (self.average_nobj** self.alphas[i])
                ops[i] = ops[i] * mult            
            else:
                raise ValueError("args.config must consist of the following letters: smxnSMXN", self.config)

        ops = torch.cat(ops, dim=2)

        if self.activate_agg:
            ops = self.activation_fn(ops)

        output = torch.einsum('dsb,ndbi->nis', self.coefs, ops)

        if not self.ir_safe:
            output = output + self.bias

        if self.activate_lin:
            output = self.activation_fn(output)

        if mask is not None:
            output = output * mask
        return output

class Eq2to2(nn.Module):
    def __init__(self, in_dim, out_dim, ops_func=None, activate_agg=False, activate_lin=True, activation = 'leakyrelu', ir_safe = False, config='s', device=torch.device('cpu'), dtype=torch.float):
        super(Eq2to2, self).__init__()
        self.device = device
        self.dtype = dtype
        self.activate_agg = activate_agg
        self.activate_lin = activate_lin
        self.activation_fn = get_activation_fn(activation)
        self.ir_safe = ir_safe
        self.config = config

        self.average_nobj = 49                 # 50 is the mean number of particles per event in the toptag dataset; ADJUST FOR YOUR DATASET
        self.basis_dim = 15 + 10 * (len(config) - 1)

        self.alphas = nn.ParameterList([None] * len(config))
        # self.betas = nn.ParameterList([None] * len(config))
        self.dummy_alphas = torch.zeros(1, in_dim, 5, 1, 1, device=device, dtype=dtype)
        # countM = 0
        for i, char in enumerate(config):
            if char in ['M', 'X', 'N']:
                self.alphas[i] = nn.Parameter(torch.zeros(1, in_dim, 10,  1, 1, device=device, dtype=dtype))
                # self.betas[i] = nn.Parameter(torch.zeros([1, 1, 10, 1, 1], device=device, dtype=dtype))
            elif char=='S':
                self.alphas[i] = nn.Parameter(torch.zeros(1, in_dim, 10,  1, 1, device=device, dtype=dtype))
                # self.betas[i] = torch.cat([(100/self.average_nobj)    * torch.ones( 1, 1, 8,  1, 1, device=device, dtype=dtype),
                #                            (100/self.average_nobj)**2 * torch.ones( 1, 1, 2,  1, 1, device=device, dtype=dtype)], dim=2).log()
            # elif char == 'M':
            #     countM += 1
            #     if countM > 1:
            #         # self.betas[i] = nn.Parameter(torch.randn( 1, 1, 10, 1, 1, device=device, dtype=dtype))
            #         self.betas[i] = torch.zeros( 1, 1, 10, 1, 1, device=device, dtype=dtype)                
            #         self.alphas[i] = nn.Parameter(torch.zeros(1, in_dim, 10, 1, 1, device=device, dtype=dtype))
            #     else:
            #         # self.betas[i] = nn.Parameter(torch.cat([(self.average_nobj/128)    * torch.ones( 1, 1, 8,  1, 1, device=device, dtype=dtype),
            #         #                                         (self.average_nobj/128)**2 * torch.ones( 1, 1, 2,  1, 1, device=device, dtype=dtype)], dim=2).log())
            #         # self.betas[i] = torch.cat([(self.average_nobj/128)    * torch.ones( 1, 1, 8,  1, 1, device=device, dtype=dtype),
            #         #                            (self.average_nobj/128)**2 * torch.ones( 1, 1, 2,  1, 1, device=device, dtype=dtype)], dim=2).log()
            #         self.betas[i] = torch.zeros( 1, 1, 10, 1, 1, device=device, dtype=dtype)                
            #         self.alphas[i] = nn.Parameter(torch.cat([torch.ones(   1, in_dim, 8,  1, 1, device=device, dtype=dtype),
            #                                                 2 * torch.ones(1, in_dim, 2,  1, 1, device=device, dtype=dtype)], dim=2))

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2./(in_dim * self.basis_dim)), (in_dim, out_dim, self.basis_dim), device=device, dtype=dtype))
        if not ir_safe:
            self.bias = nn.Parameter(torch.zeros(1, 1, 1, out_dim, device=device, dtype=dtype))
            self.diag_bias = nn.Parameter(torch.zeros(1, 1, 1, out_dim, device=device, dtype=dtype))

        self.diag_eye = None #torch.eye(n).unsqueeze(0).unsqueeze(0).to(device)
        if ops_func is None:
            self.ops_func = eops_2_to_2
        else:
            self.ops_func = ops_func

        self.to(device=device, dtype=dtype)

    def forward(self, inputs, mask=None, nobj=None):

        d = {'s': 'sum', 'm': 'mean', 'x': 'max', 'n': 'min'}

        # if self.config == 'learn':
        #     ops = self.ops_func(inputs, nobj, aggregation='mean')
        #     alphas = torch.cat([self.dummy_alphas,self.alphas],dim=2)
        #     mult = (1+nobj).view([-1,1,1,1,1])**alphas
        #     mult = mult / (50**alphas)                # 50 is the mean number of particles per event in the toptag dataset; ADJUST FOR YOUR DATASET
        #     ops = ops * mult
        # else:    
        #     ops = [self.ops_func(inputs, nobj, aggregation=d[self.config[0]])]
        #     ops = ops + [self.ops_func(inputs, nobj, aggregation=d[char], skip_order_zero=True) for char in self.config[1:] if char in ['s', 'm', 'x', 'n']]
        #     ops = ops+[self.ops_func(inputs, nobj, aggregation=d[char.lower()]) * ((1+nobj).log().view([-1,1,1,1,1]) / 3.845) for char in self.config if char in ['S', 'M', 'X', 'N']]
        #     # ops = ops+[self.ops_func(inputs, nobj, aggregation=d[char.lower()]) * ((1+nobj).log().view([-1,1,1,1,1]) / 3.845) for char in self.config if char in ['S', 'M', 'X', 'N']]
        #     ops = torch.cat(ops, dim=2)

        for i, char in enumerate(self.config):
            if char in ['s', 'm', 'x', 'n']:
                if i==0:
                    ops = [self.ops_func(inputs, nobj, aggregation=d[char])]
                else:
                    ops.append(self.ops_func(inputs, nobj, aggregation=d[char], skip_order_zero=True))
            elif char in ['S', 'M', 'X', 'N']:
                if i==0:
                    ops = [self.ops_func(inputs, nobj, aggregation=d[char.lower()])]
                    alphas = torch.cat([self.dummy_alphas, self.alphas[0]], dim=2)
                else:
                    ops.append(self.ops_func(inputs, nobj, aggregation=d[char.lower()], skip_order_zero=True))
                    alphas = self.alphas[i]
                mult = (nobj).view([-1,1,1,1,1])**alphas
                mult = mult / (self.average_nobj**alphas)
                ops[i] = ops[i] * mult
            else:
                raise ValueError("args.config must consist of the following letters: smxnSMXN", self.config)

        ops = torch.cat(ops, dim=2)

        if self.activate_agg:
            ops = self.activation_fn(ops)

        output = torch.einsum('dsb,ndbij->nijs', self.coefs, ops)

        if not self.ir_safe:
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
    def __init__(self, num_channels, num_channels_m, ops_func=None, activate_agg=False, activate_lin=True, activation='leakyrelu', batchnorm=None, sig=False, ir_safe=False, config='s', device=torch.device('cpu'), dtype=torch.float):
        super(Net2to2, self).__init__()
        
        self.sig = sig
        self.num_channels = num_channels
        self.num_channels_message = num_channels_m
        # self.softmax = nn.LogSoftmax(dim=-1)
        num_layers = len(num_channels) - 1
        # self.eq_layers = nn.ModuleList([Eq2to2(num_channels[i], num_channels[i+1], ops_func, activate_agg=activate_agg, activate_lin=activate_lin, activation=activation, config=config, device=device, dtype=dtype) for i in range(num_layers)])
        self.in_dim = num_channels_m[0][0] if len(num_channels_m[0]) > 0 else num_channels[0]

        eq_out_dims = [num_channels_m[i+1][0] if len(num_channels_m[i+1]) > 0 else num_channels[i+1] for i in range(num_layers-1)] + [num_channels[-1]]

        self.message_layers = nn.ModuleList(([MessageNet(num_channels_m[i]+[num_channels[i],], activation=activation, batchnorm=batchnorm, ir_safe=ir_safe, device=device, dtype=dtype) for i in range(num_layers)]))        
        if sig: 
            self.attention = nn.ModuleList([nn.Linear(num_channels[i], 1, bias=False, device=device, dtype=dtype) for i in range(num_layers)])
            self.normlayers = nn.ModuleList([nn.LayerNorm(num_channels[i], device=device, dtype=dtype) for i in range(num_layers)])
        self.eq_layers = nn.ModuleList([Eq2to2(num_channels[i], eq_out_dims[i], ops_func, activate_agg=activate_agg, activate_lin=activate_lin, activation=activation, ir_safe=ir_safe, config=config, device=device, dtype=dtype) for i in range(num_layers)])
        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None, nobj=None):
        '''
        x: N x m x m x in_dim
        Returns: N x m x m x out_dim
        '''

        assert (x.shape[-1] == self.in_dim), "Input dimension of Net2to2 doesn't match the dimension of the input tensor"
        B = x.shape[0]

        if self.sig: 
            for layer, message, sig, normlayer in zip(self.eq_layers, self.message_layers, self.attention, self.normlayers):
                m = message(x, mask)        # form messages at each of the NxN nodes
                y = sig(m)                  # compute the dot product with the attention vector over the channel dim
                # ms = torch.exp(self.softmax(y.view(B,-1))).view_as(y) * mask
                # ms = ms / ms.sum(dim=(1,2), keepdim=True)
                # z = normlayer(ms * m)       # apply LayerNorm, i.e. normalize over the channel dimension
                ms = y.sigmoid() * mask
                z = ms * m
                x = layer(z, mask, nobj)   # apply the permutation-equivariant layer
        else:
            for layer, message in zip(self.eq_layers, self.message_layers):
                x = message(x, mask)
                x = layer(x, mask, nobj)
        return x

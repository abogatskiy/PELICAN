import torch
import torch.nn as nn

import logging

from .lorentz_metric import normsq4, dot4
from ..layers import BasicMLP, get_activation_fn, Net1to1, Net2to2, Eq2to1, Eq2to0, MessageNet

class PELICANClassifier(nn.Module):
    """
    Permutation Invariant, Lorentz Invariant/Covariant Awesome Network
    """
    def __init__(self, num_channels0, num_channels_m, num_channels1, num_channels2,
                 activate_agg=False, activate_lin=True, activation='leakyrelu', add_beams=True, sym=False, config='s',
                 scale=1, ir_safe=False, dropout = False, batchnorm=None,
                 device=torch.device('cpu'), dtype=None, cg_dict=None):

        logging.info('Initializing network!')

        num_channels0 = expand_var_list(num_channels0)
        num_channels_m = expand_var_list(num_channels_m)
        num_channels1 = expand_var_list(num_channels1)
        num_channels2 = expand_var_list(num_channels2)

        # logging.info('num_channels0: {}'.format(num_channels0))
        # logging.info('num_channelsm: {}'.format(num_channels_m))
        # logging.info('num_channels1: {}'.format(num_channels1))
        # logging.info('num_channels2: {}'.format(num_channels2))

        super().__init__()
        self.device, self.dtype = device, dtype
        self.num_channels0 = num_channels0
        self.num_channels_m = num_channels_m
        self.num_channels1 = num_channels1
        self.num_channels2 = num_channels2
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.scale = scale
        self.add_beams = add_beams
        self.ir_safe = ir_safe
        if add_beams:
            num_scalars_in = 3
        else:
            num_scalars_in = 1

        if dropout:
            self.dropout_layer = torch.nn.Dropout(0.2)

        # self.mlp0 = BasicMLP([num_scalars_in] + num_channels0 + [num_channels1[0]], activation = activation, ir_safe=ir_safe, dropout = dropout, batchnorm = False, device=device, dtype=dtype)
        # self.mlp_mass = BasicMLP([num_scalars_in] + num_channels0 + [num_channels1[0]], activation = activation, ir_safe=ir_safe, dropout = dropout, batchnorm = False, device=device, dtype=dtype)
        # self.net2to2 = Net2to2(num_channels1, activation = activation, batchnorm = batchnorm, sym=sym, device = device, dtype = dtype)
        # self.eq2to1 = Eq2to1(num_channels1[-1], num_channels2[0], activation = activation, sym=sym, device = device, dtype = dtype)
        # self.message = MessageNet(num_channels2[0], activation=activation,  batchnorm = batchnorm, device=device, dtype=dtype)
        # self.net1to1 = Net1to1(num_channels2, activation = activation,  batchnorm = batchnorm, device = device, dtype = dtype)
        # self.mlp_out = BasicMLP([num_channels2[-1], 15] + [2], activation=activation, ir_safe=ir_safe, dropout = dropout, batchnorm = False, device=device, dtype=dtype)

        self.message = False
        if len(num_channels_m) > 0:
            if len(num_channels_m[0]) > 0:
                num_channels_m[0] = [num_scalars_in] + num_channels_m[0] 
            else:
                self.message = True
                self.input_layer = nn.Linear(num_scalars_in, num_channels1[0], bias = not ir_safe, device = device, dtype = dtype)
  
        self.net2to2 = Net2to2(num_channels1, num_channels_m, activate_agg=activate_agg, activate_lin=activate_lin, activation = activation, batchnorm = batchnorm, sym=sym, config=config, device = device, dtype = dtype)
        self.eq2to0 = Eq2to0(num_channels1[-1], num_channels2[0], activation = activation, device = device, dtype = dtype)
        self.mlp_out = BasicMLP(num_channels2 + [2], activation=activation, ir_safe=ir_safe, dropout = dropout, batchnorm = False, device=device, dtype=dtype)

        logging.info('_________________________\n')
        for n, p in self.named_parameters(): logging.info(f'{"Parameter: " + n:<80} {p.shape}')
        logging.info('Model initialized. Number of parameters: {}'.format(sum(p.nelement() for p in self.parameters())))
        logging.info('_________________________\n')

    def forward(self, data, covariance_test=False):
        """
        Runs a forward pass of the network.

        Parameters
        ----------
        data : :obj:`dict`
            Dictionary of data to pass to the network.
        covariance_test : :obj:`bool`, optional
            If true, returns all of the atom-level representations twice.

        Returns
        -------
        prediction : :obj:`torch.Tensor`
            The output of the layer
        """
        # Get and prepare the data
        atom_scalars, atom_mask, edge_mask, event_momenta = self.prepare_input(data)

        # Calculate spherical harmonics and radial functions
        num_atom = atom_mask.shape[1]
        nobj = atom_mask.sum(-1, keepdim=True)
        dot_products = dot4(event_momenta.unsqueeze(1), event_momenta.unsqueeze(2)).unsqueeze(-1)
        if self.add_beams:
            inputs = torch.cat([dot_products, atom_scalars], dim=-1)
        else:
            inputs = dot_products
        
        # inputs = (10e-03+inputs).abs().log()/2 * edge_mask.unsqueeze(-1) # Add a logarithmic rescaling function before MLP to soften the heavy tails in inputs

        # Simplest version with only 2->2 and 2->0 layers

        if self.message:
            inputs_log = self.input_layer(inputs) * edge_mask.unsqueeze(-1)

        act1 = self.net2to2(inputs, mask=edge_mask.unsqueeze(-1), nobj=nobj)
        act2 = self.eq2to0(act1)
        if self.dropout:
            act2 = self.dropout_layer(act2)
        prediction = self.mlp_out(act2)




        # Verion with 2->1 and 1->1 layers
        # D = torch.ones((num_atom,num_atom), dtype=self.dtype, device=self.device) - torch.eye(num_atom, dtype=self.dtype, device=self.device)
        # D_mask = D.unsqueeze(0).bool() * edge_mask  # Mask to consistently exclude the diagonal (mass features) from mlp0
        # mass_features = torch.permute(torch.diagonal(inputs, dim1 = 1, dim2 = 2), (0, 2, 1))
        # mass_features = self.mlp_mass(mass_features, mask=atom_mask.unsqueeze(-1))        
        # act1 = self.mlp0(inputs * D_mask.unsqueeze(-1), mask=D_mask.unsqueeze(-1))
        # act1[:, range(num_atom), range(num_atom), :] = mass_features
        # act2 = self.net2to2(act1, mask=edge_mask.unsqueeze(-1))
        # act3 = self.eq2to1(act2, mask=atom_mask.unsqueeze(-1))
        # act4 = self.message(act3, mask=atom_mask.unsqueeze(-1))
        # act5 = self.net1to1(act4, mask=atom_mask.unsqueeze(-1))
        # if self.dropout:
        #     act4 = self.dropout_layer(act5)
        # prediction = self.mlp_out(act5.mean(dim=1))





        if covariance_test:
            return prediction, [inputs, act1, act2]
        else:
            return prediction

    def prepare_input(self, data):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        atom_scalars : :obj:`torch.Tensor`
            Tensor of scalars for each atom.
        atom_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        atom_ps: :obj:`torch.Tensor`
            Positions of the atoms
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        """
        device, dtype = self.device, self.dtype

        atom_ps = data['Pmu'].to(device, dtype)

        data['Pmu'].requires_grad_(True)
        atom_mask = data['atom_mask'].to(device, torch.bool)
        edge_mask = data['edge_mask'].to(device, torch.bool)

        if 'scalars' in data.keys():
            scalars = data['scalars'].to(device, dtype)
        else:
            # scalars = torch.ones_like(atom_ps[:, :, 0]).unsqueeze(-1)
            scalars = normsq4(atom_ps).abs().sqrt().unsqueeze(-1)
        return scalars, atom_mask, edge_mask, atom_ps

def expand_var_list(var):
    if type(var) is list:
        var_list = var
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list

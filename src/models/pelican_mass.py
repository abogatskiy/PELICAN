import torch
import torch.nn as nn

import logging

from .lorentz_metric import normsq4, dot4
from ..layers import BasicMLP, get_activation_fn, Net1to1, Net2to2, Eq2to1, Eq2to0, MessageNet, InputEncoder
from ..trainer import init_weights

class PELICANMass(nn.Module):
    """
    Permutation Invariant, Lorentz Invariant/Covariant Awesome Network
    """
    def __init__(self, num_channels_m, num_channels1, num_channels2, num_channels_m_out,
                 activate_agg=False, activate_lin=True, activation='leakyrelu', add_beams=True, sig=False, config1='s', config2='s', average_nobj=20, factorize=False, masked=True, softmasked=True,
                 activate_agg2=True, activate_lin2=False, mlp_out=True,
                 scale=1, ir_safe=False, dropout = False, drop_rate=0.1, drop_rate_out=0.1, batchnorm=None,
                 device=torch.device('cpu'), dtype=None, cg_dict=None):
        super().__init__()

        logging.info('Initializing network!')

        # num_channels0 = expand_var_list(num_channels0)
        num_channels_m = expand_var_list(num_channels_m)
        num_channels1 = expand_var_list(num_channels1)
        num_channels2 = expand_var_list(num_channels2)

        self.device, self.dtype = device, dtype
        self.num_channels_m = num_channels_m
        self.num_channels1 = num_channels1
        self.num_channels2 = num_channels2
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.scale = scale
        self.add_beams = add_beams
        self.ir_safe = ir_safe
        self.mlp_out = mlp_out
        self.average_nobj = average_nobj
        self.factorize = factorize
        self.masked = masked
        self.softmasked = softmasked

        if dropout:
            self.dropout_layer = nn.Dropout(drop_rate)
            self.dropout_layer_out = nn.Dropout(drop_rate_out)

        if (len(num_channels_m) > 0) and (len(num_channels_m[0]) > 0):
            embedding_dim = self.num_channels_m[0][0]
        else:
            embedding_dim = self.num_channels1[0]
        if add_beams: 
            assert embedding_dim > 2, f"num_channels_m[0][0] or num-channels1[0] has to be at least 3 when using --add_beams but got {embedding_dim}"
            embedding_dim -= 2

        self.input_encoder = InputEncoder(embedding_dim, device = device, dtype = dtype)
        self.input_mix_and_norm = MessageNet([embedding_dim], activation=activation, ir_safe=ir_safe, batchnorm=batchnorm, device=device, dtype=dtype)

        self.net2to2 = Net2to2(num_channels1 + [num_channels_m_out[0]], num_channels_m, activate_agg=activate_agg, activate_lin=activate_lin, activation = activation, dropout=dropout, drop_rate=drop_rate, batchnorm = batchnorm, sig=sig, ir_safe=ir_safe, average_nobj=average_nobj, config=config1, factorize=factorize, masked=masked, device = device, dtype = dtype)
        self.message_layer = MessageNet(num_channels_m_out, activation=activation, ir_safe=ir_safe, batchnorm=batchnorm, device=device, dtype=dtype)       
        self.eq2to0 = Eq2to0(num_channels_m_out[-1], num_channels2[0] if mlp_out else 1,  activate_agg=activate_agg2, activate_lin=activate_lin2, activation = activation, ir_safe=ir_safe, average_nobj=average_nobj, config=config2, device = device, dtype = dtype)
        if mlp_out:
            self.mlp_out = BasicMLP(self.num_channels2 + [1], activation=activation, ir_safe=ir_safe, dropout = False, batchnorm = False, device=device, dtype=dtype)

        self.apply(init_weights)

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
            If true, returns several intermediate tensors as well.

        Returns
        -------
        prediction : :obj:`torch.Tensor`
            The output of the layer has shape [Batch, 4], i.e. one 4-vector per event.
        """
        # Get and prepare the data
        particle_scalars, particle_mask, edge_mask, event_momenta, label = self.prepare_input(data)

        # Calculate spherical harmonics and radial functions
        num_particle = particle_mask.shape[1]
        nobj = particle_mask.sum(-1, keepdim=True)
        dot_products = dot4(event_momenta.unsqueeze(1), event_momenta.unsqueeze(2))
        inputs = self.input_encoder(dot_products, mask=edge_mask.unsqueeze(-1))
        inputs = self.input_mix_and_norm(inputs, mask=edge_mask.unsqueeze(-1))
        
        if self.add_beams:
            inputs = torch.cat([inputs, particle_scalars], dim=-1)

        act1 = self.net2to2(inputs, mask=edge_mask.unsqueeze(-1), nobj=nobj, softmask=softmask.unsqueeze(1).unsqueeze(2) if self.softmasked else None)

        act2 = self.message_layer(act1, mask=edge_mask.unsqueeze(-1))

        if self.dropout:
            act2 = self.dropout_layer(act2)

        if self.softmasked:
            act2 = act2 * softmask.unsqueeze(-1)

        act3 = self.eq2to0(act2, nobj=nobj)
        
        if self.dropout:
            act3 = self.dropout_layer_out(act3)

        prediction = self.mlp_out(act3)[:,0] / self.scale

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
        particle_scalars : :obj:`torch.Tensor`
            Tensor of scalars for each particle.
        particle_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        particle_ps: :obj:`torch.Tensor`
            Positions of the particles
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        """
        device, dtype = self.device, self.dtype

        particle_ps = data['Pmu'].to(device, dtype)

        data['Pmu'].requires_grad_(True)
        particle_mask = data['particle_mask'].to(device, torch.bool)
        edge_mask = data['edge_mask'].to(device, torch.bool)

        if 'scalars' in data.keys():
            scalars = data['scalars'].to(device, dtype)
        else:
            # scalars = torch.ones_like(particle_ps[:, :, 0]).unsqueeze(-1)
            scalars = normsq4(particle_ps).abs().sqrt().unsqueeze(-1)
        return scalars, particle_mask, edge_mask, particle_ps, data['is_signal']


def expand_var_list(var):
    if type(var) is list:
        var_list = var
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list

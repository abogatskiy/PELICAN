import torch
import torch.nn as nn

import logging

from .lorentz_metric import normsq4, dot4, CATree, SDMultiplicity
from ..layers import BasicMLP, get_activation_fn, Net2to2, Eq2to1, SoftMask, MessageNet, InputEncoder, eops_2_to_2
from ..trainer import init_weights

class PELICANRegression(nn.Module):
    """
    Permutation Invariant, Lorentz Invariant/Covariant Awesome Network
    """
    def __init__(self, num_channels_m, num_channels1, num_channels2, num_channels_m_out, num_targets,
                 activate_agg=False, activate_lin=True, activation='leakyrelu', add_beams=True, config1='s', config2='s', average_nobj=20, factorize=False, masked=True,
                 activate_agg2=True, activate_lin2=False, mlp_out=True,
                 scale=1, irc_safe=False, dropout = False, drop_rate=0.1, drop_rate_out=0.1, batchnorm=None,
                 device=torch.device('cpu'), dtype=None):
        super().__init__()

        logging.info('Initializing network!')

        num_channels_m = expand_var_list(num_channels_m)
        num_channels1 = expand_var_list(num_channels1)
        num_channels2 = expand_var_list(num_channels2)

        self.device, self.dtype = device, dtype
        self.num_channels_m = num_channels_m
        self.num_channels1 = num_channels1
        self.num_channels2 = num_channels2
        self.num_targets = num_targets
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.scale = scale
        self.add_beams = add_beams
        self.irc_safe = irc_safe
        self.mlp_out = mlp_out
        self.config1 = config1
        self.config2 = config2
        self.average_nobj = average_nobj
        self.factorize = factorize
        self.masked = masked

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

        if irc_safe:
            self.softmask = SoftMask(device=device,dtype=dtype)

        # The input stack applies an encoding function
        self.input_encoder = InputEncoder(embedding_dim, device = device, dtype = dtype)

        self.net2to2 = Net2to2(num_channels1 + [num_channels_m_out[0]], num_channels_m, activate_agg=activate_agg, activate_lin=activate_lin, activation = activation, dropout=dropout, drop_rate=drop_rate, batchnorm = batchnorm, config=config1, average_nobj=average_nobj, factorize=factorize, masked=masked, device = device, dtype = dtype)
        self.message_layer = MessageNet(num_channels_m_out, activation=activation, batchnorm=batchnorm, device=device, dtype=dtype)       
        self.eq2to1 = Eq2to1(num_channels_m_out[-1], num_channels2[0] if mlp_out else num_targets,  activate_agg=activate_agg2, activate_lin=activate_lin2, activation = activation, average_nobj=average_nobj, config=config2, factorize=factorize, device = device, dtype = dtype)

        if mlp_out:
            self.mlp_out_1 = BasicMLP(self.num_channels2 + [num_targets], activation=activation, dropout = False, batchnorm = False, device=device, dtype=dtype)

        self.apply(init_weights)

        logging.info('_________________________\n')
        for n, p in self.named_parameters(): logging.info(f'{"Parameter: " + n:<80} {p.shape}')
        logging.info('Model initialized. Number of parameters: {}'.format(sum(p.nelement() for p in self.parameters())))
        logging.info('_________________________\n')

    def forward(self, data, covariance_test=False):
        """
        Runs a forward pass of the network.
        """
        # Get and prepare the data
        particle_scalars, particle_mask, edge_mask, event_momenta = self.prepare_input(data)
        dot_products = dot4(event_momenta.unsqueeze(1), event_momenta.unsqueeze(2))
        inputs = dot_products.unsqueeze(-1)

        # regular multiplicity
        nobj = particle_mask.sum(-1, keepdim=True)

        if self.irc_safe:
            # Define the C-safe weight proportional to fractional constituent energy in jet frame (Lorentz-invariant and adds up to 1)
            softmask_irc = self.softmask(dot_products, mode='c')
            # Replace input dot products with 2*(1-cos(theta_ij)) where theta is the pairwise angle in jet frame (assuming massless particles)
            eps = 1e-12
            energies = ((dot_products.sum(1).unsqueeze(1) * dot_products.sum(1).unsqueeze(2)) / dot_products.sum((1, 2), keepdim=True)).unsqueeze(-1)
            inputs1 = inputs.clone()
            inputs = 2 * inputs1 / (eps + energies)  # inputs = 2*(1-cos(theta_ij))
            if ('m' in self.config1 or 'M' in self.config1) or ('m' in self.config2 or 'M' in self.config2):
                nobj = SDMultiplicity(CATree(dot_products, nobj, ycut=1000, eps=eps)).unsqueeze(-1).to(device=nobj.device)
        
        # The first nonlinearity is the input encoder, which applies functions of the form ((1+x)^alpha-1)/alpha with trainable alphas.
        # In the C-safe case, this is still fine because inputs depends only on relative angles
        inputs = self.input_encoder(inputs, mask=edge_mask.unsqueeze(-1), mode='angle' if self.irc_safe else 'log')

        # If beams are included, then at this point we also append the scalar channels that contain particle labels.
        if self.add_beams:
            inputs = torch.cat([inputs, particle_scalars], dim=-1)

        # Apply the sequence of PELICAN equivariant 2->2 blocks with the IRC weighting.
        act1 = self.net2to2(inputs, mask = edge_mask.unsqueeze(-1), nobj=nobj,
                            softmask_irc = softmask_irc if self.irc_safe else None)

        # The last equivariant 2->1 block is constructed here by hand: message layer, dropout, and Eq2to1.
        act2 = self.message_layer(act1, mask=edge_mask.unsqueeze(-1))
        if self.dropout:
            act2 = self.dropout_layer(act2)
        act3 = self.eq2to1(act2, mask=particle_mask.unsqueeze(-1), nobj=nobj,
                           softmask_irc = softmask_irc if self.irc_safe else None)

        # The output layer applies dropout and an MLP.
        if self.dropout:
            act3 = self.dropout_layer_out(act3)
        PELICAN_weights =  self.mlp_out_1(act3, mask=particle_mask.unsqueeze(-1))
        prediction = (event_momenta.unsqueeze(-2) * PELICAN_weights.unsqueeze(-1)).sum(1) / self.scale
        prediction = prediction.squeeze(-2)

        if covariance_test:
            return {'predict': prediction, 'weights': PELICAN_weights}, [inputs, act1, act2, act3]
        else:
            return {'predict': prediction, 'weights': PELICAN_weights}

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
            Tensor of scalars for each Particle.
        particle_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        particle_ps: :obj:`torch.Tensor`
            Positions of the Particles
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
            # scalars = torch.ones_like(Particle_ps[:, :, 0]).unsqueeze(-1)
            scalars = normsq4(particle_ps).abs().sqrt().unsqueeze(-1)
        return scalars, particle_mask, edge_mask, particle_ps


def expand_var_list(var):
    if type(var) is list:
        var_list = var
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list

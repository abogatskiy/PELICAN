import torch
import torch.nn as nn

import logging

from .lorentz_metric import normsq4, dot4, CATree, SDMultiplicity
from ..layers import BasicMLP, Net2to2, Eq2to1, Eq1to2, SoftMask, MessageNet, InputEncoder, GInvariants
from ..trainer import init_weights

class PELICANRegression(nn.Module):
    """
    Permutation Invariant, Lorentz Invariant/Covariant Awesome Network
    """
    def __init__(self,  rank1_width_multiplier, num_channels_scalar, num_channels_m, num_channels_2to2, num_channels_out, num_channels_m_out, num_targets,
                 stabilizer='so13', activate_agg_in=False, activate_lin_in=True,
                 activate_agg=False, activate_lin=True, activation='leakyrelu', add_beams=True, read_pid=False, config='s', config_out='s', average_nobj=20, factorize=True, masked=True,
                 activate_agg_out=True, activate_lin_out=False, mlp_out=True,
                 scale=1, irc_safe=False, dropout = False, drop_rate=0.1, drop_rate_out=0.1, batchnorm=None,
                 device=torch.device('cpu'), dtype=None):
        super().__init__()

        logging.info('Initializing network!')

        num_channels_m = expand_var_list(num_channels_m)
        num_channels_2to2 = expand_var_list(num_channels_2to2)
        num_channels_out = expand_var_list(num_channels_out)

        self.device, self.dtype = device, dtype
        self.rank1_width_multiplier = rank1_width_multiplier
        self.num_channels_scalar = num_channels_scalar
        self.num_channels_m = num_channels_m
        self.num_channels_2to2 = num_channels_2to2
        self.num_channels_m_out = num_channels_m_out
        self.num_channels_out = num_channels_out
        self.num_targets = num_targets
        self.stabilizer = stabilizer
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.scale = scale
        self.add_beams = add_beams
        self.irc_safe = irc_safe
        self.mlp_out = mlp_out
        self.config = config
        self.config_out = config_out
        self.average_nobj = average_nobj
        self.factorize = factorize
        self.masked = masked

        if dropout:
            self.dropout_layer = nn.Dropout(drop_rate)
            self.dropout_layer_out = nn.Dropout(drop_rate_out)

        if (len(num_channels_m) > 0) and (len(num_channels_m[0]) > 0):
            embedding_dim = self.num_channels_m[0][0]
        else:
            embedding_dim = self.num_channels_2to2[0]

        self.ginvariants = GInvariants(stabilizer=stabilizer, irc_safe=irc_safe)
        self.rank1_dim = self.ginvariants.rank1_dim
        self.rank2_dim = self.ginvariants.rank2_dim

        if read_pid:
            self.num_scalars = 14
        elif add_beams:
            self.num_scalars = 2
        else:
            self.num_scalars = 0
            
        if self.num_scalars > 0 or self.rank1_dim > 0: 
            if self.rank2_dim > 0:
                assert embedding_dim > num_channels_scalar, f"num_channels_m[0][0] has to be at least {num_channels_scalar + 1} because you enabled --add_beams or --read-pid but got {embedding_dim}"
                embedding_dim -= num_channels_scalar

        if irc_safe:
            self.softmask = SoftMask(device=device,dtype=dtype)

        # The input stack applies an encoding function
        rank1_width_multiplier = 1 # each scalar will produce this many channels
        self.input_encoder = InputEncoder(rank1_width_multiplier, embedding_dim, rank1_in_dim = self.rank1_dim, rank2_in_dim=self.rank2_dim, device = device, dtype = dtype)
        # If there are scalars (like beam labels or PIDs) we promote them using an equivariant 1->2 layer and then concatenate them to the embedded dot products
        if self.num_scalars > 0:
            eq1to2_in_dim = self.num_scalars + self.input_encoder.rank1_out_dim*self.input_encoder.rank1_in_dim
            eq2to2_out_dim = num_channels_scalar if self.rank2_dim > 0 else embedding_dim
            self.eq1to2 = Eq1to2(eq1to2_in_dim, eq2to2_out_dim, activate_agg=activate_agg_in, activate_lin=activate_lin_in, activation = activation, average_nobj=average_nobj, config=config_out, factorize=False, device = device, dtype = dtype)

        # This is the main part of the network -- a sequence of permutation-equivariant 2->2 blocks
        # Each 2->2 block consists of a component-wise messaging layer that mixes channels, followed by the equivariant aggegration over particle indices
        self.net2to2 = Net2to2(num_channels_2to2 + [num_channels_m_out[0]], num_channels_m, activate_agg=activate_agg, activate_lin=activate_lin, activation = activation, dropout=dropout, drop_rate=drop_rate, batchnorm = batchnorm, config=config, average_nobj=average_nobj, factorize=factorize, masked=masked, device = device, dtype = dtype)
        
        # The final equivariant block is 2->1 and is defined here manually as a messaging layer followed by the 2->1 aggregation layer
        self.msg_2to1 = MessageNet(num_channels_m_out, activation=activation, batchnorm=batchnorm, device=device, dtype=dtype)       
        self.agg_2to1 = Eq2to1(num_channels_m_out[-1], num_channels_out[0] if mlp_out else num_targets,  activate_agg=activate_agg_out, activate_lin=activate_lin_out, activation = activation, average_nobj=average_nobj, config=config_out, factorize=factorize, device = device, dtype = dtype)

        # We have produced one feature vector per each of the N particles, and now we apply an MLP to each of those to get the final PELICAN weights
        if mlp_out:
            self.mlp_out_1 = BasicMLP(self.num_channels_out + [num_targets], activation=activation, dropout = False, batchnorm = False, device=device, dtype=dtype)

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
        rank1_inputs, dot_products, irc_weight = self.ginvariants(event_momenta)

        # regular multiplicity
        nobj = particle_mask.sum(-1, keepdim=True)

        if self.irc_safe:
            if ('m' in self.config or 'M' in self.config) or ('m' in self.config_out or 'M' in self.config_out):
                nobj = SDMultiplicity(CATree(dot_products, nobj, ycut=1000, eps=10e-8)).unsqueeze(-1).to(device=nobj.device)
        
        # The first nonlinearity is the input encoder, which applies functions of the form ((1+x)^alpha-1)/alpha with trainable alphas.
        # In the C-safe case, this is still fine because inputs depends only on relative angles
        rank1_inputs, rank2_inputs = self.input_encoder(rank1_inputs, dot_products, rank1_mask=particle_mask.unsqueeze(-1) ,rank2_mask=edge_mask.unsqueeze(-1), mode='angle' if self.irc_safe else 'log')

        # First concatenate particle scalar inputs with rank 1 momentum features (if any)
        if self.num_scalars > 0:
            if rank1_inputs is None:
                rank1_inputs = particle_scalars
            else:
                rank1_inputs = torch.cat([particle_scalars, rank1_inputs], dim=-1)
        # Now promore all rank 1 data to rank 2 using Eq1to2
        if rank1_inputs is not None:
            rank2_particle_scalars = self.eq1to2(rank1_inputs, mask=edge_mask.unsqueeze(-1), nobj=nobj, irc_weight = irc_weight if self.irc_safe else None)
        # Concatenate all rank 2 data together
        if rank2_inputs is None:
            inputs = rank2_particle_scalars
        elif rank2_particle_scalars is None:
            inputs = rank2_inputs
        else:
            inputs = torch.cat([rank2_inputs, rank2_particle_scalars], dim=-1)

        # Apply the sequence of PELICAN equivariant 2->2 blocks with the IRC weighting.
        act1 = self.net2to2(inputs, mask = edge_mask.unsqueeze(-1), nobj=nobj,
                            irc_weight = irc_weight if self.irc_safe else None)

        # The last equivariant 2->1 block is constructed here by hand: message layer, dropout, and Eq2to1.
        act2 = self.msg_2to1(act1, mask=edge_mask.unsqueeze(-1))
        if self.dropout:
            act2 = self.dropout_layer(act2)
        act3 = self.agg_2to1(act2, mask=particle_mask.unsqueeze(-1), nobj=nobj,
                           irc_weight = irc_weight if self.irc_safe else None)

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

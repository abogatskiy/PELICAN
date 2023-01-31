import torch
import torch.nn as nn

import logging

from .lorentz_metric import normsq4, dot4
from ..layers import BasicMLP, get_activation_fn, Net1to1, Net2to2, Eq2to2, Eq2to1, Eq2to0, MessageNet, InputEncoder, SoftMask, eops_2_to_2
from ..trainer import init_weights

class PELICANClassifier(nn.Module):
    """
    Permutation Invariant, Lorentz Invariant/Covariant Aggregator Network
    """
    def __init__(self, num_channels_m, num_channels1, num_channels2, num_channels_m_out,
                 activate_agg=False, activate_lin=True, activation='leakyrelu', add_beams=True, config1='s', config2='s', average_nobj=49, factorize=False, masked=True, softmasked=True,
                 activate_agg2=True, activate_lin2=False, mlp_out=True,
                 scale=1, ir_safe=False, c_safe=False, dropout = False, drop_rate=0.1, drop_rate_out=0.1, batchnorm=None,
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
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.scale = scale
        self.add_beams = add_beams
        self.ir_safe = ir_safe
        self.c_safe = c_safe
        self.mlp_out = mlp_out
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
            assert embedding_dim > 2, f"num_channels_m[0][0] has to be at least 3 when using --add_beams but got {embedding_dim}"
            embedding_dim -= 2

        if ir_safe or c_safe:
            self.softmask_layer = SoftMask(device=device,dtype=dtype)

        if c_safe:
            self.c_safe_eq_layer = Eq2to2(3 if add_beams else 1, embedding_dim, eops_2_to_2, activate_agg=False, activate_lin=False, activation=activation, ir_safe=True, config='s', average_nobj=average_nobj, factorize=factorize, device=device, dtype=dtype)
        
        # The input stack applies an encoding function
        self.input_encoder = InputEncoder(embedding_dim, device = device, dtype = dtype)
        # then a BatchNorm layer (messily implemented by calling MessageNet with zero layers):
        self.input_mix_and_norm = MessageNet([embedding_dim], activation=activation, ir_safe=ir_safe, batchnorm=batchnorm, device=device, dtype=dtype)

        self.net2to2 = Net2to2(num_channels1 + [num_channels_m_out[0]], num_channels_m, activate_agg=activate_agg, activate_lin=activate_lin, activation = activation, dropout=dropout, drop_rate=drop_rate, batchnorm = batchnorm, ir_safe=ir_safe, config=config1, average_nobj=average_nobj, factorize=factorize, masked=masked, device = device, dtype = dtype)
        self.message_layer = MessageNet(num_channels_m_out, activation=activation, ir_safe=ir_safe, batchnorm=batchnorm, device=device, dtype=dtype)       
        self.eq2to0 = Eq2to0(num_channels_m_out[-1], num_channels2[0] if mlp_out else 2, activate_agg=activate_agg2, activate_lin=activate_lin2, activation = activation, ir_safe=ir_safe, config=config2, factorize=False, average_nobj=average_nobj, device = device, dtype = dtype)
        if mlp_out:
            self.mlp_out = BasicMLP(self.num_channels2 + [2], activation=activation, ir_safe=ir_safe, dropout = False, batchnorm = False, device=device, dtype=dtype)

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
            The output of the model with 2 classification weights per event (intended for CrossEntropyLoss). 
            The class predicted by the classifier (0 or 1) is given by output.argmax(dim=1).
        """
        # Get and prepare the data
        particle_scalars, particle_mask, edge_mask, event_momenta = self.prepare_input(data)

        # Calculate spherical harmonics and radial functions
        nobj = particle_mask.sum(-1, keepdim=True)
        dot_products = dot4(event_momenta.unsqueeze(1), event_momenta.unsqueeze(2))
        inputs = dot_products.unsqueeze(-1)


        if self.c_safe:
            # Define a softmask that zeroes out rows and columns that correspond to massless inputs
            softmask_c = self.softmask_layer(dot_products, mask=edge_mask, mode='c')
            if self.ir_safe:
                # In case of IRC-safety, the IR softmask will be the same as the C softmask.
                softmask_ir = softmask_c
                # Prior to any nonlinearities, apply one equivariant layer to the matrix of dot products with the C-mask.
                # This ensures that massless inputs survive in later layers only as part of sums over all momenta. 
                # The original entries for those inputs get killed by the C-safe softmask. This is done only once, so the C-mask is not needed again.
                # To avoid multiplying any entries by the mask twice, we use the softmask_irc option.
                inputs = self.c_safe_eq_layer(inputs, softmask_irc=softmask_c.unsqueeze(1).unsqueeze(2))
            else:       
                # In case of only C-safety, apply the same equivariant layer with the C-safe mask.  
                inputs = self.c_safe_eq_layer(inputs, softmask_c=softmask_c.unsqueeze(1).unsqueeze(2))
        elif self.ir_safe:
            # In case of only IR-safety, define a more lenient softmask that vanishes on a given
            #  row/column corresponding to an input particle p if p*J vanishes, where J is the total jet momentum.
            # This makes sure that rows and columns corresponding to soft inputs remain soft throughout the network
            # but without the extreme rigidity of the C-safe mask, which relies on accurate information about non-zero masses.
            softmask_ir = self.softmask_layer(dot_products, mask=edge_mask, mode='ir')

        # The first nonlinearity is the input encoder, which applies functions of the form ((1+x)^alpha-1)/alpha with trainable alphas.
        inputs = self.input_encoder(inputs, mask=edge_mask.unsqueeze(-1))
        # Now apply a BatchNorm2D (remember to set --batchnorm=False if you need IR or C-safety)
        inputs = self.input_mix_and_norm(inputs, mask=edge_mask.unsqueeze(-1))

        # If beams are included, then at this point we also append the scalar channels that contain particle labels.
        if self.add_beams:
            inputs = torch.cat([inputs, particle_scalars], dim=-1)

        # Apply the sequence of PELICAN equivariant 2->2 blocks with the IR mask.
        act1 = self.net2to2(inputs, mask=edge_mask.unsqueeze(-1), nobj=nobj, softmask_ir=softmask_ir.unsqueeze(1).unsqueeze(2) if self.ir_safe else None)

        # The last equivariant 2->0 block is constructed here by hand: message layer, dropout, and Eq2to0.
        act2 = self.message_layer(act1, mask=edge_mask.unsqueeze(-1))
        if self.dropout:
            act2 = self.dropout_layer(act2)
        act3 = self.eq2to0(act2, nobj=nobj)

        # The output layer applies dropout and an MLP.
        if self.dropout:
            act3 = self.dropout_layer_out(act3)
        if self.mlp_out:
            prediction = self.mlp_out(act3)
        else:
            prediction = act3

        if torch.isnan(prediction).any():
            logging.info(f"inputs: {torch.isnan(inputs).any()}")
            logging.info(f"act1: {torch.isnan(act1).any()}")
            logging.info(f"act2: {torch.isnan(act2).any()}")
            logging.info(f"prediction: {torch.isnan(prediction).any()}")
        assert not torch.isnan(prediction).any(), "There are NaN entries in the output! Evaluation terminated."

        if covariance_test:
            return {'predict': prediction}, [inputs, act1, act2, act3]
        else:
            return {'predict': prediction}

    def prepare_input(self, data):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        scalars : :obj:`torch.Tensor`
            Tensor of scalars for each particle.
        particle_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        particle_ps: :obj:`torch.Tensor`
            4-momenta of the particles
        """
        device, dtype = self.device, self.dtype

        particle_ps = data['Pmu'].to(device, dtype)

        data['Pmu'].requires_grad_(True)
        particle_mask = data['particle_mask'].to(device, torch.bool)
        edge_mask = data['edge_mask'].to(device, torch.bool)

        if 'scalars' in data.keys():
            scalars = data['scalars'].to(device, dtype)
        else:
            scalars = normsq4(particle_ps).abs().sqrt().unsqueeze(-1)
        return scalars, particle_mask, edge_mask, particle_ps

def expand_var_list(var):
    if type(var) is list:
        var_list = var
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list

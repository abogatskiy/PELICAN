import torch
import torch.nn as nn

import logging

from .lorentz_metric import normsq4, dot4, CATree, SDMultiplicity
from ..layers import BasicMLP, Net2to2, Eq2to1, Eq1to2, MessageNet, InputEncoder, GInvariants, MyLinear
from ..trainer import init_weights

class PELICANRegression(nn.Module):
    """
    Permutation Invariant, Lorentz Invariant/Covariant Aggregator Network
    """
    def __init__(self,  rank1_width_multiplier, num_channels_scalar, num_channels_m, num_channels_2to2, num_channels_out, num_channels_m_out, num_targets,
                 stabilizer='so13',  method='spurions', activate_agg_in=False, activate_lin_in=True,
                 activate_agg=False, activate_lin=True, activation='leakyrelu', config='s', config_out='s', average_nobj=20, factorize=True, masked=True,
                 activate_agg_out=True, activate_lin_out=False, mlp_out=True,
                 scale=1, irc_safe=False, dropout = False, drop_rate=0.1, drop_rate_out=0.1, batchnorm=None,  
                 dataset='', device=torch.device('cpu'), dtype=None):
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
        self.method = method
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.scale = scale
        self.irc_safe = irc_safe
        self.mlp_out = mlp_out
        self.config = config
        self.config_out = config_out
        self.average_nobj = average_nobj
        self.factorize = factorize
        self.masked = masked
        self.dataset = dataset

        if dropout:
            self.dropout_layer = nn.Dropout(drop_rate)
            self.dropout_layer_out = nn.Dropout(drop_rate_out)

        if method.startswith('s'):
            self.ginvariants = GInvariants(stabilizer='so13', irc_safe=irc_safe)
        else:
            self.ginvariants = GInvariants(stabilizer=stabilizer, irc_safe=irc_safe)
        self.rank1_dim = self.ginvariants.rank1_dim
        self.rank2_dim = self.ginvariants.rank2_dim

        self.num_scalars = 1 + self.num_spurions() + {'qg': 12, 'jc': 12, '': 0}[dataset]

        if (len(num_channels_m) > 0) and (len(num_channels_m[0]) > 0):
            embedding_dim = self.num_channels_m[0][0]
        else:
            embedding_dim = self.num_channels_2to2[0]

        # rank2_dim_multiplier  =1
        if self.num_scalars > 0 or self.rank1_dim > 0: 
            if self.rank2_dim > 0:
                assert embedding_dim > num_channels_scalar, f"num_channels_m[0][0] has to be at least {num_channels_scalar + 1}, because you have particle scalars, but got {embedding_dim}"
                # rank2_dim_multiplier = (embedding_dim - num_channels_scalar)//self.rank2_dim
                embedding_dim = embedding_dim - num_channels_scalar
        
        # The input stack applies an encoding function
        rank1_dim_multiplier = 1 # each scalar will produce this many channels

        if stabilizer == 'so13' or method.startswith('s'):
            weights = torch.ones((embedding_dim, self.rank2_dim), device=device, dtype=dtype)
        elif stabilizer in ['1','1_0']:
            weights = torch.ones((embedding_dim, self.rank2_dim), device=device, dtype=dtype) - torch.tensor([[0,2,2,2]], device=device, dtype=dtype)
        elif stabilizer in ['so3','so12','se2']:
            weights = torch.zeros((embedding_dim, self.rank2_dim), device=device, dtype=dtype) + torch.tensor([[0,1]], device=device, dtype=dtype)
        elif stabilizer in ['so2','so2_0','R']:
            weights = torch.zeros((embedding_dim, self.rank2_dim), device=device, dtype=dtype) + torch.tensor([[0,0,1]], device=device, dtype=dtype)
        elif stabilizer in ['11','11_0']:
            weights = torch.zeros((embedding_dim, self.rank2_dim), device=device, dtype=dtype) + torch.tensor([[0,0,0,0,1]], device=device, dtype=dtype)
        
        self.linear = MyLinear(self.rank2_dim, embedding_dim, weights, device=device, dtype=dtype)

        mode = 'angle' if self.irc_safe else 'slog'
        self.input_encoder = InputEncoder(rank1_dim_multiplier, embedding_dim, 
                                          rank1_in_dim = self.rank1_dim, rank2_in_dim=self.rank2_dim, 
                                          mode=mode, device = device, dtype = dtype)
        
        # This is the main part of the network -- a sequence of permutation-equivariant 2->2 blocks
        # Each 2->2 block consists of a component-wise messaging layer that mixes channels, followed by the equivariant aggegration over particle indices
        self.net2to2 = Net2to2(num_channels_2to2 + [num_channels_m_out[0]], num_channels_m, activate_agg=activate_agg, activate_lin=activate_lin, activation = activation, dropout=dropout, drop_rate=drop_rate, batchnorm = batchnorm, config=config, average_nobj=average_nobj, factorize=factorize, masked=masked, device = device, dtype = dtype)
        
        # The final equivariant block is 2->1 and is defined here manually as a messaging layer followed by the 2->1 aggregation layer
        self.msg_2to1 = MessageNet(num_channels_m_out, activation=activation, batchnorm=batchnorm, device=device, dtype=dtype)       
        self.agg_2to1 = Eq2to1(num_channels_m_out[-1], num_channels_out[0] if mlp_out else num_targets,  activate_agg=activate_agg_out, activate_lin=activate_lin_out, activation = activation, average_nobj=average_nobj, config=config_out, factorize=factorize, device = device, dtype = dtype)

        # We have produced one feature vector per each of the N particles, and now we apply an MLP to each of those to get the final PELICAN weights
        if mlp_out:
            self.mlp_out_1 = BasicMLP(self.num_channels_out + [num_targets*(1 if self.method.startswith('s') else min(4,self.rank2_dim))], 
                                      activation=activation, dropout = False, batchnorm = False, device=device, dtype=dtype)

        self.apply(init_weights)

        # If there are scalars (like beam labels or PIDs) we promote them using an equivariant 1->2 layer and then concatenate them to the embedded dot products
        if self.num_scalars > 0:
            eq1to2_in_dim = self.num_scalars + rank1_dim_multiplier*self.rank1_dim
            # eq2to2_out_dim = (embedding_dim - self.rank2_dim * rank2_dim_multiplier) if self.rank2_dim > 0 else embedding_dim
            self.eq1to2 = Eq1to2(eq1to2_in_dim, num_channels_scalar, 
                                 activate_agg=activate_agg_in, activate_lin=activate_lin_in, 
                                 activation = activation, average_nobj=average_nobj, 
                                 config=config_out, factorize=False, device = device, dtype = dtype)
            self.eq1to2.apply(init_weights)

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
        rank1_inputs, rank2_inputs, irc_weight = self.ginvariants(event_momenta)

        # regular multiplicity
        nobj = particle_mask.sum(-1, keepdim=True)

        if self.irc_safe:
            if ('m' in self.config or 'M' in self.config) or ('m' in self.config_out or 'M' in self.config_out):
                nobj = SDMultiplicity(CATree(rank2_inputs, nobj, ycut=1000, eps=10e-8)).unsqueeze(-1).to(device=nobj.device)
        
        # The first nonlinearity is the input encoder, which applies functions of the form ((1+x)^alpha-1)/alpha with trainable alphas.
        # In the C-safe case, this is still fine because inputs depends only on relative angles
        rank2_inputs = self.linear(rank2_inputs)
        rank1_inputs, rank2_inputs = self.input_encoder(rank1_inputs, rank2_inputs, 
                                                        rank1_mask=particle_mask.unsqueeze(-1) ,rank2_mask=edge_mask.unsqueeze(-1))

        inputs = self.apply_eq1to2(particle_scalars, rank1_inputs, rank2_inputs, edge_mask, nobj, irc_weight)

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
        PELICAN_weights = self.mlp_out_1(act3, mask=particle_mask.unsqueeze(-1))
        PELICAN_weights = PELICAN_weights.view(PELICAN_weights.shape[:2]+(self.num_targets,min(4,self.rank2_dim),))
        prediction = self.regression_prediction(event_momenta, PELICAN_weights)

        check_nan = torch.isnan(prediction).any()
        if check_nan:
            logging.info(torch.isnan(act1).sum(1,2,3))
            logging.info(f"inputs has NaNs: {torch.isnan(inputs).any()}")
            logging.info(f"rank1_inputs: {torch.isnan(rank1_inputs).any()}")
            logging.info(f"rank2_inputs: {torch.isnan(rank2_inputs).any()}")
            logging.info(f"act1 has NaNs: {torch.isnan(act1).any()}")
            logging.info(f"act2 has NaNs: {torch.isnan(act2).any()}")
            logging.info(f"prediction has NaNs: {check_nan}")
        assert not check_nan, "There are NaN entries in the output! Evaluation terminated."
        if covariance_test:
            return {'predict': prediction, 'weights': PELICAN_weights}, [inputs, act1, act2, act3]
        else:
            return {'predict': prediction, 'weights': PELICAN_weights}

    def regression_prediction(self, event_momenta, PELICAN_weights):
        dtype, device = self.dtype, self.device
        event_momenta = event_momenta.unsqueeze(-2)
        if self.method.startswith('i'):
            if self.stabilizer=='so3':  # L_x, L_y, L_z
                event_momenta = event_momenta * torch.tensor([[[[1,0,0,0],[0,1,1,1]]]],dtype=dtype,device=device)
            elif self.stabilizer=='so12': # K_x, K_y, L_z
                event_momenta = event_momenta * torch.tensor([[[[1,1,1,0],[0,0,0,1]]]],dtype=dtype,device=device)
            elif self.stabilizer=='se2':  # L_z, K_x-L_y, K_y+L_x (stabilizer of [1,0,0,1] )
                event_momenta = (event_momenta.unsqueeze(-2) @ 
                                 torch.tensor([[[[[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,1]],
                                               [[1,0,0,-1],[0,1,0,0],[0,0,1,0],[-1,0,0,0]]]]],
                                               dtype=dtype,device=device)).squeeze(-2)
            elif self.stabilizer in ['so2','so2_0']:  # L_z
                event_momenta = event_momenta * torch.tensor([[[[1,0,0,0],[0,1,1,0],[0,0,0,1]]]],dtype=dtype,device=device)
            elif self.stabilizer=='R':    # K_z
                event_momenta = event_momenta * torch.tensor([[[[1,0,0,1],[0,1,0,0],[0,0,1,0]]]],dtype=dtype,device=device)
            elif self.stabilizer in ['11','11_0']:   # 0
                event_momenta = event_momenta * torch.tensor([[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]],dtype=dtype,device=device)
        prediction = (event_momenta.unsqueeze(-3) * PELICAN_weights.unsqueeze(-1)).sum((1,3)) / self.scale
        prediction = prediction.squeeze(-2) # in case there is only one target vector, remove that dimension
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
        scalars : :obj:`torch.Tensor`
            Tensor of scalars for each particle.
        particle_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        event_momenta: :obj:`torch.Tensor`
            4-momenta of the particles
        """
        device, dtype = self.device, self.dtype

        if self.method.startswith('s'):
            data = self.add_spurions(data)

        event_momenta = data['Pmu'].to(device, dtype)
        # event_momenta.requires_grad_(True)
        particle_mask = data['particle_mask'].to(device, torch.bool)
        edge_mask = data['edge_mask'].to(device, torch.bool)

        if 'scalars' in data.keys():
            scalars = data['scalars'].to(device, dtype)
        else:
            scalars = None
        return scalars, particle_mask, edge_mask, event_momenta

    def num_spurions(self):
        stabilizer = self.stabilizer
        if stabilizer == 'so13':
            return 0
        elif stabilizer in ['so3','so12','se2']:
            return 1
        elif stabilizer in ['so2','so2_0','R']:
            return 2
        elif stabilizer in ['1','11','1_0','11_0']:
            return 4

    def add_spurions(self, data):
        stabilizer = self.stabilizer
        if stabilizer == 'so13':
            return data
        device, dtype = data['Pmu'].device, data['Pmu'].dtype
        batch_size = len(data['Nobj'])
        if stabilizer == 'so3':
            spurions = torch.tensor([[[1,0,0,0]]], dtype=dtype, device=device)
        elif stabilizer == 'so12':
            spurions = torch.tensor([[[0,0,0,1]]], dtype=dtype, device=device)
        elif stabilizer == 'se2':
            spurions = torch.tensor([[[1,0,0,-1]]], dtype=dtype, device=device)
        elif stabilizer == 'so2':
            spurions = torch.tensor([[[1,0,0,0],[0,0,0,1]]], dtype=dtype, device=device)
        elif stabilizer == 'so2_0':
            spurions = torch.tensor([[[1,0,0,1],[1,0,0,-1]]], dtype=dtype, device=device)
        elif stabilizer == 'R':
            spurions = torch.tensor([[[0,1,0,0],[0,0,1,0]]], dtype=dtype, device=device)
        elif stabilizer in ['1','11']:
            spurions = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]], dtype=dtype, device=device)
        elif stabilizer in ['1_0','11_0']:
            spurions = torch.tensor([[[1,0,0,1],[1,1,0,0],[1,0,1,0],[1,0,0,-1]]], dtype=dtype, device=device)
        spurions = spurions.expand((batch_size, -1, -1))

        num_spurions = spurions.shape[1]
        particle_mask = torch.cat((torch.ones(batch_size, num_spurions).bool().to(device=device), data['particle_mask']),dim=-1)
        edge_mask = particle_mask.unsqueeze(1) * particle_mask.unsqueeze(2)

        data['Pmu'] = torch.cat([spurions, data['Pmu']], 1)
        data['particle_mask'] = particle_mask
        data['edge_mask'] = edge_mask
        if 'scalars' in data.keys():
            data['scalars'] = torch.cat([torch.zeros((batch_size, num_spurions, data['scalars'].shape[2]), device=device, dtype=data['scalars'].dtype), data['scalars']], dim=1)
        else:
            labels = 1 - particle_mask.long()
            labels[:, :num_spurions] = torch.arange(1,num_spurions+1)
            data['scalars'] = onehot(labels, num_classes=1+num_spurions, mask=particle_mask.unsqueeze(-1))
        return data


    def apply_eq1to2(self, particle_scalars, rank1_inputs, rank2_inputs, edge_mask, nobj, irc_weight):
        # First concatenate particle scalar inputs with rank 1 momentum features (if any)
        if self.num_scalars > 0:
            if rank1_inputs is None:
                rank1_inputs = particle_scalars
            else:
                rank1_inputs = torch.cat([particle_scalars, rank1_inputs], dim=-1)
        # Now promore all rank 1 data to rank 2 using Eq1to2
        if rank1_inputs is not None:
            rank2_particle_scalars = self.eq1to2(rank1_inputs, mask=edge_mask.unsqueeze(-1), nobj=nobj, irc_weight = irc_weight if self.irc_safe else None)
        else:
            rank2_particle_scalars = None
        # Concatenate all rank 2 data together
        if rank2_inputs is None:
            inputs = rank2_particle_scalars
        elif rank2_particle_scalars is None:
            inputs = rank2_inputs
        else:
            inputs = torch.cat([rank2_inputs, rank2_particle_scalars], dim=-1)
        return inputs

def expand_var_list(var):
    if type(var) is list:
        var_list = var
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list

def onehot(x, num_classes=2, mask=None):
    x = torch.nn.functional.one_hot(x, num_classes=num_classes)
    zero = torch.tensor(0, device=x.device, dtype=torch.long)
    if mask is not None:
        x = torch.where(mask, x, zero)
    return x
import torch
import math
import torch.nn as nn
from .masked_batchnorm import MaskedBatchNorm1d, MaskedBatchNorm2d
from .masked_instancenorm import MaskedInstanceNorm2d, MaskedInstanceNorm3d
# from ..models.lorentz_metric import dot4, dot3, dot2, dot12, dot11



class MyLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, weight = None, bias: bool = True,
                 rand_bias=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        if weight is None:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            self.reset_weight()
            self.reset_bias()
        else:
            self.weight = nn.Parameter(weight)
            if rand_bias:
                self.reset_bias()
            elif self.bias is not None:
                nn.init.zeros_(self.bias)

    def reset_weight(self) -> None:
        nn.init.kaiming_normal_(self.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def reset_bias(self) -> None:
        if self.bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out) if fan_out > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class BasicMLP(nn.Module):
    """
    Multilayer perceptron used in various locations.  Operates only on the last axis of the data.
    If num_channels has length 2, this becomes a linear layer.

    NB: the ir_safe flag is a vestige of an old implementation of IR-safety, currently unused.
    """

    def __init__(self, num_channels, activation='leakyrelu', ir_safe=False, batchnorm=False, dropout=False, drop_rate=0.25, device=torch.device('cpu'), dtype=torch.float):
        super(BasicMLP, self).__init__()

        self.num_channels = num_channels
        self.num_in = num_channels[0]
        self.num_out = num_channels[-1]
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(self.num_in, num_channels[1], bias=not ir_safe))

        self.num_hidden = len(num_channels) - 2

        for i in range(self.num_hidden - 1):
            self.linear.append(nn.Linear(num_channels[1+i], num_channels[2+i], bias=not ir_safe))
        if self.num_hidden > 0:
            self.linear.append(nn.Linear(num_channels[-2], self.num_out, bias=not ir_safe))

        activation_fn = get_activation_fn(activation)

        self.activations = nn.ModuleList()
        for i in range(self.num_hidden + 1):
            self.activations.append(activation_fn)
        
        if batchnorm: self.batchnormlayer = nn.BatchNorm2d(self.num_out)
        if dropout: self.dropoutlayer = nn.Dropout(drop_rate)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None):
        # Standard MLP. Loop over a linear layer followed by a non-linear activation

        for (lin, activation) in zip(self.linear, self.activations):
            x = activation(lin(x))

        # Use Batch Normalization for Deep Learning. 
        # This should be done before applying the mask to avoid skewing the values.
        if self.batchnorm: 
            if len(x.shape)==3:
                x = self.batchnormlayer(x.unsqueeze(-1).permute(0,2,1,3)).permute(0,2,1,3).squeeze(-1)
            elif len(x.shape)==4:
                x = self.batchnormlayer(x.permute(0,3,1,2)).permute(0,2,3,1)

        # Apply Dropout, which independently zeroes every component of x with a given probability.
        if self.dropout: x = self.dropoutlayer(x)

        # If mask is included, mask the output
        if mask is not None:
            x = torch.where(mask, x, self.zero)

        return x

    def scale_weights(self, scale):
        self.linear[-1].weight *= scale
        if self.linear[-1].bias is not None:
            self.linear[-1].bias *= scale

class MessageNet(nn.Module):
    """
    Multilayer perceptron used in message forming for message passing.  Operates only on the last axis of the data.
    If num_channels has length 2, this becomes a linear layer.
    """

    def __init__(self, num_channels, depth=1, activation='leakyrelu', ir_safe=False, batchnorm = None, masked=True, device=torch.device('cpu'), dtype=torch.float):
        super().__init__()

        self.num_channels = num_channels
        self.batchnorm = batchnorm
        self.masked = masked

        if type(num_channels) not in [list, tuple]:
            num_channels = [num_channels,] * (depth + 1)
        depth = len(num_channels) - 1
        
        self.linear = nn.ModuleList([nn.Linear(num_channels[i], num_channels[i+1], bias=not ir_safe) for i in range(depth)])

        activation_fn = get_activation_fn(activation)
        self.activations = nn.ModuleList([activation_fn for i in range(depth)])
        
        # if depth == 0: self.batchnorm = False
        if self.batchnorm == True:
            self.batchnorm = 'b'
        if self.batchnorm:
            if self.batchnorm.startswith('b'):
                if masked:
                    self.normlayer = MaskedBatchNorm2d(num_channels[-1], device=device, dtype=dtype)
                else:
                    self.normlayer = nn.BatchNorm2d(num_channels[-1], device=device, dtype=dtype)
            elif self.batchnorm.startswith('i'):
                if masked:
                    self.normlayer = MaskedInstanceNorm2d(num_channels[-1], device=device, dtype=dtype)
                else:
                    self.normlayer = nn.InstanceNorm2d(num_channels[-1], device=device, dtype=dtype)
            elif self.batchnorm.startswith('l'):
                if masked:
                    self.normlayer = MaskedInstanceNorm3d(1, device=device, dtype=dtype)
                else:
                    self.normlayer = nn.InstanceNorm3d(1, device=device, dtype=dtype)
            else:
                self.batchnorm = False

        self.zero = torch.tensor(0, device=device, dtype=dtype)
        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None):
        # Standard MLP. Loop over a linear layer followed by a non-linear activation

        for (lin, activation) in zip(self.linear, self.activations):
            x = activation(lin(x))

        # If mask is included, mask the output
        if mask is not None:
            x = torch.where(mask, x, self.zero)

        if self.batchnorm: 
            if self.batchnorm.startswith('b') or self.batchnorm.startswith('i'):
                if len(x.shape)==3:
                    if self.masked:
                        x = self.normlayer(x.unsqueeze(1), mask).squeeze(1)
                    else:
                        x = self.normlayer(x.unsqueeze(-1).permute(0,2,1,3)).permute(0,2,1,3).squeeze(-1)
                elif len(x.shape)==4:
                    if self.masked:
                        x = self.normlayer(x, mask)
                    else:
                        x = self.normlayer(x.permute(0,3,1,2)).permute(0,2,3,1)
            elif self.batchnorm.startswith('l'):
                if len(x.shape)==3:
                    if self.masked:
                        x = self.normlayer(x.unsqueeze(1).unsqueeze(1), mask.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
                    else:
                        x = self.normlayer(x.unsqueeze(1)).squeeze(-1)
                elif len(x.shape)==4:
                    if self.masked:
                        x = self.normlayer(x.unsqueeze(1), mask.expand(x.shape).unsqueeze(1)).squeeze(1)
                    else:
                        x = self.normlayer(x.unsqueeze(1)).squeeze(1)

        return x

    def scale_weights(self, scale):
        self.linear[-1].weight *= scale
        if self.linear[-1].bias is not None:
            self.linear[-1].bias *= scale


class InputEncoder(nn.Module):
    def __init__(self, rank1_dim_multiplier, rank2_dim_multiplier, rank1_in_dim = 0, rank2_in_dim = 1, mode = 'log', device=torch.device('cpu'), dtype=torch.float):
        super().__init__()

        self.rank1_in_dim = rank1_in_dim
        self.rank2_in_dim = rank2_in_dim
        self.rank1_dim_multiplier = rank1_dim_multiplier if rank1_in_dim else 0
        self.rank2_dim_multiplier = rank2_dim_multiplier if rank2_in_dim else 0
        self.mode = mode
        if rank1_in_dim > 0:
            self.rank1_alphas = nn.Parameter(torch.linspace(0.05, 0.5, rank1_dim_multiplier, device=device, dtype=dtype).unsqueeze(-1).repeat((1, rank1_in_dim)).t())
        if rank2_in_dim > 0:
            # rank2_dim is never higher than 1 for us, so these alphas are defined with that in mind
            # self.rank2_alphas = nn.Parameter(torch.linspace(0.05, 0.5, rank2_dim_multiplier, device=device, dtype=dtype).unsqueeze(-1).repeat((1, rank2_in_dim)).t())
            self.rank2_alphas = nn.Parameter(torch.linspace(0.05, 0.5, rank2_dim_multiplier, device=device, dtype=dtype))
        # self.alphas = nn.Parameter(0.5 * torch.rand(1, 1, 1, out_dim, device=device, dtype=dtype))
        # self.betas = nn.Parameter(torch.randn(1, 1, 1, out_dim, device=device, dtype=dtype))
        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, rank1_inputs, dot_products, rank1_mask=None, rank2_mask=None):        
        if self.rank1_in_dim > 0:
            s = rank1_inputs.shape[:-1]
            l = len(s)
            rank1_alphas = self.rank1_alphas.view([1,]*l+[self.rank1_in_dim, self.rank1_dim_multiplier])
            rank1_out = fn(rank1_inputs.unsqueeze(-1), rank1_alphas, self.mode).reshape(s+(self.rank1_in_dim*self.rank1_dim_multiplier,))
        else:
            rank1_out = None
        if self.rank2_in_dim > 0: 
            s = dot_products.shape[:-1]
            l = len(s)
            # rank2_alphas = self.rank2_alphas.view([1,]*l+[self.rank2_in_dim, self.rank2_dim_multiplier])
            rank2_alphas = self.rank2_alphas.view([1,]*l+[self.rank2_dim_multiplier])
            rank2_out = fn(dot_products, rank2_alphas, self.mode)
        else:
            rank2_out = None

        if rank1_mask is not None and self.rank1_in_dim > 0:
            rank1_out = torch.where(rank1_mask, rank1_out, self.zero)
        if rank2_mask is not None and self.rank2_in_dim > 0: 
            rank2_out = torch.where(rank2_mask, rank2_out, self.zero)
        return rank1_out, rank2_out
        
class GInvariants(nn.Module):
    def __init__(self, stabilizer='so13', irc_safe=False):
        super().__init__()

        dict_rank1 = {'so13': 0, 'so3': 0, 'so12': 0, 'se2': 0, 'so2':   0, 'R': 0, '1':   0, '11':   0,
                                                                'so2_0': 0,         '1_0': 0, '11_0': 0}
        dict_rank2 = {'so13': 1, 'so3': 2, 'so12': 2, 'se2': 2, 'so2':   3, 'R': 3, '1':   4, '11':   5,
                                                                'so2_0': 3,         '1_0': 4, '11_0': 5}

        self.stabilizer = stabilizer
        self.irc_safe = irc_safe
        self.rank1_dim = dict_rank1[stabilizer]
        self.rank2_dim = dict_rank2[stabilizer]

    def forward(self, event_momenta):
        dtype, device = event_momenta.dtype, event_momenta.device
        # event_momenta = event_momenta.unsqueeze(1)
        if self.stabilizer == '1':
            rank1 = event_momenta
            rank2 = rank1.unsqueeze(1)*rank1.unsqueeze(2)
            rank1 = None
        elif self.stabilizer == '1_0':
            rank1 = event_momenta @ torch.tensor([[1,0,0,1],[1,-1,0,0],[1,0,-1,0],[1,0,0,-1]],dtype=dtype,device=device).t()
            rank2 = rank1.unsqueeze(1)*rank1.unsqueeze(2)
            rank1 = None
        else:
            dot_products = dot4(event_momenta.unsqueeze(1), event_momenta.unsqueeze(2)).unsqueeze(-1)
            if self.stabilizer=='so13':   # L_x, L_y, L_z, K_x, K_y, K_z
                rank1 = None
            elif self.stabilizer=='so3':  # L_x, L_y, L_z
                rank1 = event_momenta[...,[0]] # Energy
                # rank2 = dot3(event_momenta, event_momenta).unsqueeze(-1)
            elif self.stabilizer=='so12': # K_x, K_y, L_z
                rank1 = event_momenta[...,[-1]] #p_z
                # rank2 = dot12(event_momenta, event_momenta).unsqueeze(-1)
            elif self.stabilizer=='se2':  # L_z, K_x-L_y, K_y+L_x
                rank1 = (event_momenta[...,[0]] - event_momenta[...,[-1]]) #E - p_z
            elif self.stabilizer=='so2':  # L_z
                rank1 = (event_momenta[...,[0, 3]]) # E, p_z
            elif self.stabilizer=='so2_0':# L_z
                rank1 = (event_momenta[...,[0, 3]]) @ torch.tensor([[1,1],[1,-1]],dtype=dtype,device=device) # E ± p_z
                # rank2 = dot2(event_momenta, event_momenta).unsqueeze(-1)            
            elif self.stabilizer=='R':    # K_z
                rank1 = (event_momenta[...,[1, 2]]) # p_x, p_y
                # rank2 = dot11(event_momenta, event_momenta).unsqueeze(-1)  
            elif self.stabilizer=='11':   # 0
                rank1 = event_momenta
            elif self.stabilizer=='11_0':   # 0
                rank1 = (event_momenta) @ torch.tensor([[1,0,0,1],[1,1,0,0],[1,0,1,0],[1,0,0,-1]],dtype=dtype,device=device).t()

            if rank1 == None:
                rank2 = dot_products
            else:
                rank2 = torch.cat([rank1.unsqueeze(1)*rank1.unsqueeze(2), dot_products], dim=-1)

        # TODO: make irc_safe option work with rank1 inputs
        if self.irc_safe:
            if self.rank1_dim > 0:
                raise NotImplementedError
            # Define the C-safe weight proportional to fractional constituent energy in jet frame (Lorentz-invariant and adds up to 1)
            # Replace input dot products with 2*(1-cos(theta_ij)) where theta is the pairwise angle in jet frame (assuming massless particles)
            eps = 1e-12
            irc_weight = dot_products.sum(1).squeeze(-1) / dot_products.sum((1,2))
            energies = ((dot_products.sum(1).unsqueeze(1) * dot_products.sum(1).unsqueeze(2)) / dot_products.sum((1, 2), keepdim=True))
            inputs1 = rank2.clone()
            rank2 = 2 * inputs1 / (eps + energies)  # 2*(1-cos(theta_ij)) in massless case

        return rank1, rank2, irc_weight

class SoftMask(nn.Module):
    """
    Multilayer perceptron used in various locations.  Operates only on the last axis of the data.
    If num_channels has length 2, this becomes a linear layer.
    """

    def __init__(self, device=torch.device('cpu'), dtype=torch.float):
        super(SoftMask, self).__init__()

        self.zero = torch.tensor(0, device=device, dtype=dtype)

        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None, mode=''):
        
        if mode == 'c':
            x = x.sum(dim=1) / x.sum(dim=(1,2)).unsqueeze(-1) # computes energy fractions Epsilon_i in jet frame
        
        if mode=='ir':
            mag = x.sum(dim=1) * 0.001
            x = torch.clamp(mag.unsqueeze(-1) * mag.unsqueeze(-2), min=-1., max=1.)

        if mode=='ir1d':
            mag = x.sum(dim=1) * 0.001
            x = torch.clamp(mag, min=-1., max=1.)

        # If mask is included, mask the output
        if mask is not None:
            x = torch.where(mask, x, self.zero)

        return x


def get_activation_fn(activation):
    activation = activation.lower()
    if activation == 'leakyrelu':
        activation_fn = nn.LeakyReLU(negative_slope=0.01)
    elif activation == 'relu':
        activation_fn = nn.ReLU()
    elif activation == 'prelu':
        activation_fn = nn.PReLU()    
    elif activation == 'selu':
        activation_fn = nn.SELU()
    elif activation == 'gelu':
        activation_fn = nn.GELU()
    elif activation == 'elu':
        activation_fn = nn.ELU()
    elif activation == 'celu':
        activation_fn = nn.CELU(alpha=1.)
    elif activation == 'sigmoid':
        activation_fn = nn.Sigmoid()
    elif activation == 'logsigmoid':
        activation_fn = nn.LogSigmoid()
    elif activation == 'atan':
        activation_fn = ATan()
    elif activation == 'silu':
        activation_fn = SiLU()
    elif activation == 'soft':
        activation_fn = nn.Softsign()
    elif activation == 'tanh':
        activation_fn = nn.Tanh()   
    elif activation == 'identity':
        activation_fn = nn.Identity()
    else:
        raise ValueError('Activation function {} not implemented!'.format(activation))
    return activation_fn


class ATan(torch.nn.Module):
   
    def forward(self, input):
        return torch.atan(input)

# simply define a silu function
def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return silu(input) # simply apply already implemented SiLU
    


def dot4(p1, p2):
    # Quick hack to calculate the dot products of the four-vectors
    # The last dimension of the input gets eaten up
    # Broadcasts over other dimensions
    prod = p1 * p2
    return 2 * prod[..., 0] - prod.sum(dim=-1)

def dot3(p1, p2):
    # Dot product of the spatial parts
    prod = p1[...,1:] * p2[...,1:]
    return prod.sum(dim=-1)

def dot2(p1, p2):
    # Dot product of the xy parts (pT)
    prod = p1[...,1:3] * p2[...,1:3]
    return prod.sum(dim=-1)

def dot12(p1, p2):
    # 2+1 invariant dot product
    prod = p1[...,:3] * p2[...,:3]
    return 2 * prod[..., 0] - prod.sum(dim=-1)

def dot11(p1, p2):
    # 1+1 invariant dot product
    prod = p1[...,[0,-1]] * p2[...,[0,-1]]
    return 2 * prod[..., 0] - prod.sum(dim=-1)

def fn(x, alphas, mode):
    if mode=='log':
        x = ((1 + x).abs().pow(1e-6 + alphas ** 2) - 1) / (1e-6 + alphas ** 2)
    elif mode=='slog':
        x = ((1 + x.abs()).pow(1e-6 + alphas ** 2) - 1) / (1e-6 + alphas ** 2) * x.sign()
    elif mode=='angle':
        x = ((1e-5 + x.abs()).pow(1e-6 + alphas ** 2) - 1) / (1e-6 + alphas ** 2)
    elif mode=='arcsinh':
        x = (x * 2 * alphas).arcsinh() / (1e-6 + alphas.abs())
    return x

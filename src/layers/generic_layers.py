import torch
import torch.nn as nn
from .masked_batchnorm import MaskedBatchNorm1d, MaskedBatchNorm2d
from .masked_instancenorm import MaskedInstanceNorm2d, MaskedInstanceNorm3d
class BasicMLP(nn.Module):
    """
    Multilayer perceptron used in various locations.  Operates only on the last axis of the data.
    If num_channels has length 2, this becomes a linear layer.
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

    def __init__(self, num_channels, depth=2, activation='leakyrelu', ir_safe=False, batchnorm = None, masked=True, device=torch.device('cpu'), dtype=torch.float):
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
        
        if depth == 0: self.batchnorm = False
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
    def __init__(self, out_dim, device=torch.device('cpu'), dtype=torch.float):
        super().__init__()

        self.to(device=device, dtype=dtype)
        # self.alphas = nn.Parameter(torch.linspace(0.01, 1.05, out_dim, device=device, dtype=dtype).view(1, 1, 1, out_dim))
        self.alphas = nn.Parameter(0.5 * torch.rand(1, 1, 1, out_dim, device=device, dtype=dtype))
        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, x, mask=None):

        # x = (1. + x.unsqueeze(-1)).abs().pow(1e-6 + self.alphas) - 1.

        x = ((1. + x.unsqueeze(-1)).abs().pow(1e-6 + self.alphas ** 2) - 1.) / (1e-6 + self.alphas ** 2)
        # x = x.unsqueeze(-1) * self.alphas

        # x = (1e-2 + x).abs().log()/2  # Add a logarithmic rescaling function before MLP to soften the heavy tails in inputs
        
        # x = x.arcsinh()

        if mask is not None:
            x = torch.where(mask, x, self.zero)

        return x


class SoftMask(nn.Module):
    """
    Multilayer perceptron used in various locations.  Operates only on the last axis of the data.
    If num_channels has length 2, this becomes a linear layer.
    """

    def __init__(self, device=torch.device('cpu'), dtype=torch.float):
        super(SoftMask, self).__init__()

        self.beta = nn.Parameter(torch.tensor(1., device=device, dtype=dtype))
        self.mu = 1.# nn.Parameter(torch.tensor(2., device=device, dtype=dtype))

        self.zero = torch.tensor(0, device=device, dtype=dtype)

        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None):
        # x = ((x-self.mu)*self.beta).sigmoid()

        x = (self.beta * x).tanh() ** self.mu

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
        activation_fn = nn.Tanhshrink()   
    elif activation == 'identity':
        activation_fn = lambda x: x
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
import torch
import torch.nn.functional as F
import numpy as np
from math import sqrt

def batch_stack_general(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Unlike :batch_stack:, this will automatically stack scalars, vectors,
    and matrices. It will also automatically convert Numpy Arrays to
    Torch Tensors.

    Parameters
    ----------
    props : list or tuple of Pytorch Tensors, Numpy ndarrays, ints or floats.
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if type(props[0]) in [int, float]:
        # If batch is list of floats or ints, just create a new Torch Tensor.
        return torch.tensor(props)

    if type(props[0]) is np.ndarray:
        # Convert numpy arrays to tensors
        props = [torch.from_numpy(prop) for prop in props]

    shapes = [prop.shape for prop in props]

    if all(shapes[0] == shape for shape in shapes):
        # If all shapes are the same, stack along dim=0
        return torch.stack(props)

    elif all(shapes[0][1:] == shape[1:] for shape in shapes):
        # If shapes differ only along first axis, use the RNN pad_sequence to pad/stack.
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)

    elif all((shapes[0][2:] == shape[2:]) for shape in shapes):
        # If shapes differ along the first two axes, (shuch as a matrix),
        # pad/stack first two axes

        # Ensure that input features are matrices
        assert all((shape[0] == shape[1]) for shape in shapes), 'For batch stacking matrices, first two indices must match for every data point'

        max_particles = max([len(p) for p in props])
        max_shape = (len(props), max_particles, max_particles) + props[0].shape[2:]
        padded_tensor = torch.zeros(max_shape, dtype=props[0].dtype, device=props[0].device)

        for idx, prop in enumerate(props):
            this_particles = len(prop)
            padded_tensor[idx, :this_particles, :this_particles] = prop

        return padded_tensor
    else:
        ValueError('Input tensors must have the same shape on all but at most the first two axes!')




def batch_stack(props, edge_mat=False, nobj=None):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack
    edge_mat : bool
        The included tensor refers to edge properties, and therefore needs
        to be stacked/padded along two axes instead of just one.

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if nobj is not None and nobj < 0:
        nobj = None
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    elif not edge_mat:
        props = [p[:nobj, ...] for p in props]
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
    else:
        max_particles = max([len(p) for p in props])
        max_shape = (len(props), max_particles, max_particles) + props[0].shape[2:]
        padded_tensor = torch.zeros(max_shape, dtype=props[0].dtype, device=props[0].device)

        for idx, prop in enumerate(props):
            this_particles = len(prop)
            padded_tensor[idx, :this_particles, :this_particles] = prop

        return padded_tensor


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


def collate_fn(data, scale=1., nobj=None, edge_features=[], add_beams=False, beam_mass=1, read_pid=False):
    """
    Collation function that collates datapoints into the batch format for lgn

    Parameters
    ----------
    data : list of datapoints
        The data to be collated.
    edge_features : list of strings
        Keys of properties that correspond to edge features, and therefore are
        matrices of shapes (num_particles, num_particles), which when forming a batch
        need to be padded along the first two axes instead of just the first one.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """
    common_keys = data[0].keys()
    # common_keys = set.intersection(*[set(d.keys()) for d in data]) # Uncomment if different data files have different sets of keys
    data = {prop: batch_stack([mol[prop] for mol in data], nobj=nobj) for prop in common_keys}
    device = data['Pmu'].device
    dtype = data['Pmu'].dtype
    zero = torch.tensor(0.)
    # to_keep = batch['Nobj'].to(torch.uint8)
    s = data['Pmu'].shape
    particle_mask = torch.cat((torch.ones(s[0],2).bool().to(device=device), data['Pmu'][...,0] != 0.),dim=-1)

    # p3s = data['Pmu'][:,:,1:4]
    # Es = data['Pmu'][:,:,[0]]
    # p3s = F.normalize(p3s, dim=-1) * Es
    # data['Pmu'] = torch.cat([Es,p3s],dim=-1)

    edge_mask = particle_mask.unsqueeze(1) * particle_mask.unsqueeze(2)

    if read_pid:
        assert 'pdgid' in data.keys(), "Need the key pdgid in your data before using read_pid"
        
    if add_beams:
        p = 1
        beams = torch.tensor([[[sqrt(p**2+beam_mass**2),0,0,p], [sqrt(p**2+beam_mass**2),0,0,-p]]], dtype=data['Pmu'].dtype, device=data['Pmu'].device).expand(s[0], 2, 4)
        data['Pmu'] = torch.cat([beams, data['Pmu'] * scale], dim=1)
        data['Nobj'] = data['Nobj'] + 2
        if read_pid:
            num_classes=14
            beams_pdg = torch.tensor([[2212, 2212]], dtype=torch.long, device=data['Pmu'].device).expand(s[0], 2)
            data['pdgid'] = torch.cat([beams_pdg, data['pdgid'].to(dtype=torch.long)], dim=1)
        else:
            num_classes=2
            data['pdgid'] = torch.cat([2212 * torch.ones(s[0], 2, dtype=torch.long, device=data['Pmu'].device),
                                       torch.zeros(s[0], s[1], dtype=torch.long, device=data['Pmu'].device)], dim=1)
    else:
        data['Pmu'] = data['Pmu'] * scale

    # labels = torch.cat((torch.ones(s[0], 2), torch.zeros(s[0], s[1])), dim=1)
    # labelsi = labels.unsqueeze(1).expand(s[0], s[1]+2, s[1]+2)
    # labelsj = labels.unsqueeze(2).expand(s[0], s[1]+2, s[1]+2)
    # labels = torch.where(edge_mask.unsqueeze(-1), torch.stack([labelsi, labelsj], dim=-1), zero)

    particle_mask = data['Pmu'][...,0] != 0.
    edge_mask = particle_mask.unsqueeze(1) * particle_mask.unsqueeze(2)

    if read_pid or add_beams:
        if 'scalars' not in data.keys():
            data['scalars'] = pdg_onehot(data['pdgid'], num_classes=num_classes, mask=particle_mask.unsqueeze(-1))
        else:
            data['scalars'] = torch.cat([data['scalars'], pdg_onehot(data['pdgid'], num_classes=num_classes, mask=particle_mask.unsqueeze(-1))], dim=-1)

    data['particle_mask'] = particle_mask.bool()
    data['edge_mask'] = edge_mask.bool()

    return data

def pdg_onehot(x, num_classes=14, mask=None):
    if num_classes==14:
        x = 0*(x==22) + 1*(x==211) + 2*(x==-211) + 3*(x==321) + 4*(x==-321) + 5*(x==130) + 6*(x==2112) + 7*(x==-2112) + 8*(x==2212) + 9*(x==-2212) + 10*(x==11) + 11*(x==-11) + 12*(x==13) + 13*(x==-13)
    elif num_classes==2:
        x = 0*(x!=2212) + 1*(x==2212)
    x = torch.nn.functional.one_hot(x, num_classes=num_classes)
    zero = torch.tensor(0, device=x.device, dtype=torch.long)
    if mask is not None:
        x = torch.where(mask, x, zero)
    
    return x

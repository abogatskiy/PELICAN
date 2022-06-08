import torch

def normsq4(p):
    # Quick hack to calculate the norms of the four-vectors
    # The last dimension of the input gets eaten up
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dot4(p1, p2):
    # Quick hack to calculate the dot products of the four-vectors
    # The last dimension of the input gets eaten up
    # Broadcasts over other dimensions
    prod = p1 * p2
    return 2 * prod[..., 0] - prod.sum(dim=-1)
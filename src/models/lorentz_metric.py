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



def SDMultiplicity(jetsbatch, zcut=0.005, thetacut=0., R0 = 0.8, beta = -1.):
    """
    Given a C/A tree produced by CATree(), compute the Lorentz-invariant
    analog of the Soft Drop multiplicity nSD from https://arxiv.org/pdf/1704.06266.pdf
    nSD is the depth of the branching tree along its hard core (i.e. choosing the harder subjet at each branching), until we hit the thetacut.

    Input: a list of events, each event potentially containing several jets:
        jetsbatch=[[jet0,jet1,...],[jet0,jet1,...],...] where each jet is stored 
        as a binary tree with branching info (z, theta) stored at nodes.
    
    Output: torch.Tensor of nSD of only the first jet (jet0) contained in each event.
    """

    def SD_traverse(jet, zcut, thetacut, R0, beta, out):
        # This function outputs a sequence of (z_n, theta_n) similar to https://arxiv.org/pdf/1704.06266.pdf
        # That sequence can be used to define all kinds of IRC-safe observables, but we will only use it for nSD
        if type(jet) is int:
            return out
        else:
            subjet_0, subjet_1, z, thetasq = jet[0][0], jet[0][1], jet[1][0], jet[1][1]
            if thetasq <= thetacut ** 2:
                return out                                                    # if the branching angle is below the cut, terminate
            if z <= zcut * (thetasq / R0 ** 2) ** (beta / 2):
                return SD_traverse(subjet_1, zcut, thetacut, R0, beta, out)   # if the softdrop condition is not satisfied, recurse on the harder subjet
            else:
                out.append((z, thetasq))                                      # otherwise save (z, thetasq), and recurse on the harder subjet
                return SD_traverse(subjet_1, zcut, thetacut, R0, beta, out)

    # Now we traverse the tree and count the number of branchings that satisfy the soft drop condition
    B=len(jetsbatch)
    nSD=[]
    for b in range(B):
        jet=jetsbatch[b][0]
        nSD.append(len(SD_traverse(jet, zcut, thetacut, R0, beta, [])))
    return torch.tensor(nSD, dtype=torch.long)    
            

def CATree(dot_products, nobj, ycut=1, eps=1e-12):
    """
    Primitive and slow implementation of a Lorentz-invariant analog
    of the C/A algorithm from https://arxiv.org/pdf/hep-ph/9803322.pdf
    If the jet frame were to coincide with the lab frame, this would've 
    completely matched the standard C/A for electron-positron collisions.

    Input: batch of matrices of pairwise dot products of the 4-momenta of jet constituents
    Output: for every event, a list of binary branching trees for each jet in the event (if ycut>=1, there will be only one jet)
        Each node has the form ((left_node, right_node), (z, theta)),
        where z is the SoftDrop observable min(E_i,E_j)/(E_i+E_j) and theta is the branching angle.
    """
    B = dot_products.shape[0]
    energies = dot_products.sum(1)
    jetsbatch=[]
    for b in range(B):
        N = nobj[b].item()
        dots = dot_products[b,:N,:N]
        treelist = list(range(N))
        jets =[]
        energy = energies[b, :N]         # Computes E_i*M, where E_i is the jet-frame energy and M is the jet mass (we avoid dividing by M)
        Msq = energy.sum()

        # First we construct the branching tree, which will be located in jets[0] 
        # (if y_cut<1, it can produce several jets with separate trees for each)
        while N > 1:
            thetasq = 2 * dots / (eps + (energy.unsqueeze(0) * energy.unsqueeze(1) / Msq).abs())  # Computes the Lorentz-invariant analog of 2*(1-cos(theta_ij))
            thetasq = thetasq + 100 * torch.eye(N, dtype=thetasq.dtype, device=thetasq.device) # Add a number greater than 4 to the diagonal to avoid it being chosen as the minimum
            (i, j) = unravel_index(torch.argmin(thetasq), thetasq.shape).tolist()  # Find the pair with the smallest angle
            if energy[i] > energy[j]:       # Order the pair so that i has the lower energy
                (i, j) = (j, i)
            y = (energy[i] ** 2) * thetasq[i,j]  # Compute the main test variable
            y_cut = ycut * Msq ** 2
            if y <= y_cut:
                z = min(energy[i], energy[j]) / (energy[i] + energy[j])
                treelist[j] = ((treelist[i], treelist[j]),(z, thetasq[i,j])) # Each node in the tree has the form ((left_node, right_node),(z_ij,theta_ij))
                treelist.pop(i)
                energy[j] = energy[i] + energy[j]
                energy = torch.cat((energy[:i],energy[i+1:]))
                merged_pmu = dots[i] + dots[j]
                dots[j] = merged_pmu
                dots[:,j] = merged_pmu
                dots[j,j] = dots[j,j] + dots[i,j]
                dots = torch.cat((dots[:i], dots[i+1:]))
                dots = torch.cat((dots[:,:i], dots[:,i+1:]),1)
            else:
                jets.append(treelist[i])
                energy = torch.cat((energy[:i],energy[i+1:]))
                dots = torch.cat((dots[:i], dots[i+1:]))
                dots = torch.cat((dots[:,:i], dots[:,i+1:]),1)
            N = N - 1
        jets.append(treelist[0])
        jetsbatch.append(jets)

    return jetsbatch


def unravel_index(
    indices: torch.LongTensor,
    shape: tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord
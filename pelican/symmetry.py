"""
Symmetry breaking.

A *stabilizer* names the subgroup of SO+(1,3) that the model is invariant under. PELICAN realizes it in one of two
ways (`--method`):

'spurions' (default): fixed 4-vectors (e.g. the beam directions) are added to every event as extra particles,
     labelled by a one-hot particle-type scalar. The inputs are then the Lorentz-invariant dot products of all
     particles, and the residual symmetry is the subgroup leaving all spurions fixed.
'input': no extra particles; instead the pairwise inputs include products of the components that the stabilizer
     leaves invariant (e.g. E and p_z for so2), alongside the dot products.
"""
from dataclasses import dataclass
import torch


def dot4(p1, p2):
    """
    Minkowski product as PELICAN has always computed it, 2 E1 E2 - sum_mu p1_mu p2_mu. It equals lorentz.dot4 but rounds
    differently for nearly collinear constituents, and the pairwise inputs are kept bit-compatible with trained models.
    """
    prod = p1 * p2
    return 2 * prod[..., 0] - prod.sum(dim=-1)

_T, _X, _Y, _Z = (1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.)
_NP, _NM = (1., 0., 0., 1.), (1., 0., 0., -1.)


@dataclass(frozen=True)
class Stabilizer:
    name: str
    spurions: tuple   # fixed 4-vectors breaking the symmetry
    pair_dim: int     # number of pairwise input channels with --method input
    description: str

    @property
    def num_spurions(self):
        return len(self.spurions)

    def spurion_tensor(self, device=None, dtype=torch.float64):
        return torch.tensor(self.spurions, device=device, dtype=dtype).reshape(-1, 4)


STABILIZERS = {s.name: s for s in [
    Stabilizer('so13', (), 1, 'full Lorentz invariance'),
    Stabilizer('so3', (_T,), 2, 'rotations (fixes the time axis)'),
    Stabilizer('so12', (_Z,), 2, 'SO(1,2) fixing the beam axis'),
    Stabilizer('se2', (_NM,), 2, 'ISO(2) fixing one light-like beam'),
    Stabilizer('so2', (_T, _Z), 3, 'rotations about the beam axis'),
    Stabilizer('so2_0', (_NP, _NM), 3, 'rotations about the beam axis (light-like beams)'),
    Stabilizer('R', (_X, _Y), 3, 'boosts along the beam axis'),
    Stabilizer('1', (_T, _X, _Y, _Z), 4, 'no symmetry (all lab axes)'),
    Stabilizer('1_0', (_NP, (1., 1., 0., 0.), (1., 0., 1., 0.), _NM), 4, 'no symmetry (light-like axes)'),
    # Same spurions as '1' and '1_0'; with --method input the pairwise inputs also include the dot products.
    Stabilizer('11', (_T, _X, _Y, _Z), 5, 'no symmetry (all lab axes, with dot products)'),
    Stabilizer('11_0', (_NP, (1., 1., 0., 0.), (1., 0., 1., 0.), _NM), 5, 'no symmetry (light-like axes, with dot products)'),
]}


def get_stabilizer(name):
    if name not in STABILIZERS:
        raise ValueError(f"Unknown stabilizer '{name}'. Choose from: {', '.join(STABILIZERS)}")
    return STABILIZERS[name]


def symmetry_vectors(stabilizer, method):
    """
    4-vectors whose common stabilizer is the model's symmetry group: the spurions, except that with --method input
    'se2' uses E - p_z, which fixes (1,0,0,1) rather than its spurion (1,0,0,-1).
    """
    if method == 'input' and stabilizer.name == 'se2':
        return (_NP,)
    return stabilizer.spurions


def _invariant_components(p, name):
    """Per-particle components left invariant by the stabilizer (--method input), or None if there are none."""
    dtype, device = p.dtype, p.device
    if name == 'so3':
        return p[..., [0]]                                                      # E
    if name == 'so12':
        return p[..., [-1]]                                                     # p_z
    if name == 'se2':
        return p[..., [0]] - p[..., [-1]]                                       # E - p_z
    if name == 'so2':
        return p[..., [0, 3]]                                                   # E, p_z
    if name == 'so2_0':
        return p[..., [0, 3]] @ torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device)    # E +- p_z
    if name == 'R':
        return p[..., [1, 2]]                                                   # p_x, p_y
    if name == '11':
        return p
    if name == '11_0':
        return p @ torch.tensor([[1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, -1]], dtype=dtype, device=device).t()
    return None


def pair_invariants(p, stabilizer, irc_safe=False, eps=1e-12):
    """
    Pairwise inputs [B,N,N,K] of constituents p [B,N,4] (padding = 0) for the given stabilizer name (with spurions
    as particles, pass 'so13': dot products only). With irc_safe, also returns the IRC weights [B,N] (each particle's
    jet-frame energy fraction) and replaces the inputs by 2(1 - cos theta_ij) in the jet frame (massless particles).
    """
    if stabilizer == '1':
        rank2, dots = p.unsqueeze(1) * p.unsqueeze(2), None
    elif stabilizer == '1_0':
        q = p @ torch.tensor([[1, 0, 0, 1], [1, -1, 0, 0], [1, 0, -1, 0], [1, 0, 0, -1]], dtype=p.dtype, device=p.device).t()
        rank2, dots = q.unsqueeze(1) * q.unsqueeze(2), None
    else:
        dots = dot4(p.unsqueeze(1), p.unsqueeze(2)).unsqueeze(-1)
        comps = _invariant_components(p, stabilizer)
        rank2 = dots if comps is None else torch.cat([comps.unsqueeze(1) * comps.unsqueeze(2), dots], dim=-1)
    if not irc_safe:
        return rank2, None
    if dots is None:
        raise ValueError(f"IRC-safe mode needs dot products; stabilizer '{stabilizer}' with --method input has none (use '11' or '11_0')")
    weight = dots.sum(1).squeeze(-1) / dots.sum((1, 2))
    energies = (dots.sum(1).unsqueeze(1) * dots.sum(1).unsqueeze(2)) / dots.sum((1, 2), keepdim=True)
    return 2 * rank2 / (eps + energies), weight

"""Minkowski-space helpers. Metric signature (+,-,-,-); 4-vectors are (E, px, py, pz) in the last axis."""
import torch

ETA = torch.diag(torch.tensor([1., -1., -1., -1.], dtype=torch.float64))


def dot4(p1, p2):
    """Minkowski inner product over the last axis (broadcasts over the others)."""
    return p1[..., 0] * p2[..., 0] - (p1[..., 1:] * p2[..., 1:]).sum(-1)


def normsq4(p):
    return dot4(p, p)


def lie_algebra_basis(dtype=torch.float64):
    """The six generators of so(1,3) acting on column 4-vectors: rotations J_x, J_y, J_z, then boosts K_x, K_y, K_z."""
    basis = []
    for i, j in [(2, 3), (3, 1), (1, 2)]:  # rotations in the (y,z), (z,x), (x,y) planes
        g = torch.zeros(4, 4, dtype=dtype)
        g[i, j], g[j, i] = -1., 1.
        basis.append(g)
    for i in [1, 2, 3]:  # boosts along x, y, z
        g = torch.zeros(4, 4, dtype=dtype)
        g[0, i] = g[i, 0] = 1.
        basis.append(g)
    return torch.stack(basis)


def stabilizer_algebra(vectors, dtype=torch.float64):
    """
    Basis of the subalgebra of so(1,3) that annihilates every vector in `vectors` (shape [K,4]),
    i.e. the Lie algebra of the subgroup leaving all of them fixed. Returns a tensor of shape [d,4,4].
    """
    basis = lie_algebra_basis(dtype)
    if len(vectors) == 0:
        return basis
    vectors = torch.as_tensor(vectors, dtype=dtype)
    # Linear map c -> (sum_a c_a X_a) v_k for all k; its null space is the stabilizer subalgebra.
    A = torch.einsum('aij,kj->kia', basis, vectors).reshape(-1, basis.shape[0])
    _, s, vh = torch.linalg.svd(A)
    rank = int((s > 1e-10).sum())
    null = vh[rank:]
    return torch.einsum('na,aij->nij', null, basis)


def random_group_element(algebra, scale=1.0, generator=None):
    """exp of a random element of the given Lie algebra basis [d,4,4]; the identity if the algebra is trivial."""
    if algebra.shape[0] == 0:
        return torch.eye(4, dtype=algebra.dtype)
    c = scale * torch.randn(algebra.shape[0], dtype=algebra.dtype, generator=generator)
    return torch.linalg.matrix_exp(torch.einsum('a,aij->ij', c, algebra))

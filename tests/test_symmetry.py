"""Stabilizers and the pairwise inputs: invariance under each stabilizer's subgroup, and the IRC-safe inputs."""
import pytest
import torch

from pelican.lorentz import stabilizer_algebra, random_group_element
from pelican.symmetry import get_stabilizer, pair_invariants, symmetry_vectors, dot4
from conftest import make_jets

NAMES = ['so13', 'so3', 'so12', 'se2', 'so2', 'so2_0', 'R', '1', '1_0', '11', '11_0']


def test_unknown_stabilizer():
    with pytest.raises(ValueError):
        get_stabilizer('so4')


@pytest.mark.parametrize('name', NAMES)
def test_stabilizer_algebra_dimension(name):
    expected = {'so13': 6, 'so3': 3, 'so12': 3, 'se2': 3, 'so2': 1, 'so2_0': 1, 'R': 1, '1': 0, '1_0': 0, '11': 0, '11_0': 0}
    assert stabilizer_algebra(get_stabilizer(name).spurions).shape[0] == expected[name]


@pytest.mark.parametrize('name', NAMES)
def test_pair_invariants_shape_and_invariance(name):
    """With --method input, the pairwise inputs are invariant under the stabilizer's subgroup."""
    b = make_jets()
    st = get_stabilizer(name)
    inv, weight = pair_invariants(b['Pmu'], name)
    assert inv.shape == b['Pmu'].shape[:2] + b['Pmu'].shape[1:2] + (st.pair_dim,) and weight is None
    alg = stabilizer_algebra(symmetry_vectors(st, 'input'))
    for seed in range(2):
        Lam = random_group_element(alg, 0.7, torch.Generator().manual_seed(seed))
        inv2, _ = pair_invariants(b['Pmu'] @ Lam.T, name)
        assert (inv2 - inv).abs().max() / inv.abs().max() < 1e-10


@pytest.mark.parametrize('name', ['so3', 'so2', 'R', '11'])
def test_spurions_are_fixed_by_the_subgroup(name):
    st = get_stabilizer(name)
    S = st.spurion_tensor()
    for seed in range(2):
        Lam = random_group_element(stabilizer_algebra(st.spurions), 0.7, torch.Generator().manual_seed(seed))
        assert torch.allclose(S @ Lam.T, S, atol=1e-12)


def test_se2_conventions():
    """With --method input, se2 inputs (E - p_z) are invariant under the stabilizer of (1,0,0,1), not of the spurion."""
    st = get_stabilizer('se2')
    assert st.spurions == ((1., 0., 0., -1.),) and symmetry_vectors(st, 'input') == ((1., 0., 0., 1.),)


def test_dot4_matches_minkowski_product():
    b = make_jets()
    p = b['Pmu']
    eta = torch.tensor([1., -1., -1., -1.])
    scale = p[..., 0].unsqueeze(1) * p[..., 0].unsqueeze(2)          # products of nearly collinear momenta cancel
    assert ((dot4(p.unsqueeze(1), p.unsqueeze(2)) - torch.einsum('bid,bjd,d->bij', p, p, eta)).abs() <= 1e-13 * scale).all()


def test_irc_safe_inputs_are_angles_and_weights_energy_fractions():
    """For massless constituents: inputs 2(1 - cos theta_ij) in the jet frame, weights summing to 1."""
    b = make_jets()
    inv, weight = pair_invariants(b['Pmu'], 'so13', irc_safe=True)
    mask = b['particle_mask']
    assert torch.allclose(weight.sum(1), torch.ones(weight.shape[0]))
    assert (weight[~mask] == 0).all() and (weight[mask] > 0).all()
    pairs = mask.unsqueeze(1) & mask.unsqueeze(2)
    assert (inv[pairs] >= -1e-9).all() and (inv[pairs] <= 4 + 1e-9).all()


def test_irc_inputs_need_dot_products():
    with pytest.raises(ValueError):
        pair_invariants(make_jets()['Pmu'], '1', irc_safe=True)

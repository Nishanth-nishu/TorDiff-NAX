"""flexitors.geometry.internal: exact dihedrals, signed volumes, bond lengths (DECISION fix 7; plan_P2 V-E3-7)."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from flexitors.geometry.internal import bond_lengths, dihedral, signed_volume, wrap_angle


def embedded(smiles, seed=7):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0
    return m


def test_dihedral_matches_rdkit():
    m = embedded('CC(O)C(=O)NCC=CC')
    conf = m.GetConformer()
    rng = np.random.default_rng(0)
    quads = []
    for b in m.GetBonds():
        j, k = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ni = [n.GetIdx() for n in m.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        nl = [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
        if ni and nl:
            quads.append((ni[rng.integers(len(ni))], j, k, nl[rng.integers(len(nl))]))
    ours = dihedral(conf.GetPositions(), np.array(quads))
    ref = np.array([rdMolTransforms.GetDihedralRad(conf, *q) for q in quads])
    assert np.allclose(wrap_angle(ours - ref), 0, atol=1e-9)


def test_dihedral_resolves_near_planar_angles():
    # TD's clamped acos cannot resolve |tau| < ~4.5e-3 rad (utils/torsion.py:120); atan2 must
    for tau in (0.0, 1e-6, 1e-4, -1e-4, np.pi - 1e-4, np.pi):
        p = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 1.5], [np.cos(tau), np.sin(tau), 1.5]])
        got = dihedral(p, np.array([[0, 1, 2, 3]]))[0]
        assert abs(wrap_angle(got - tau)) < 1e-12


def test_dihedral_is_batched():
    m = embedded('CCCC')
    p = m.GetConformer().GetPositions()
    batch = np.stack([p, p[::1] + 1.0])
    out = dihedral(batch, np.array([[0, 1, 2, 3]]))
    assert out.shape == (2, 1) and np.isclose(out[0, 0], out[1, 0])


def test_signed_volume_flips_under_reflection():
    m = embedded('C[C@H](N)O')
    p = m.GetConformer().GetPositions()
    c = np.array([[1, 0, 2, 3]])
    v = signed_volume(p, c)[0]
    assert abs(v) > 0.1 and np.sign(signed_volume(-p, c)[0]) == -np.sign(v)


def test_bond_lengths_and_wrap():
    p = np.array([[0.0, 0, 0], [1.5, 0, 0], [1.5, 2.0, 0]])
    assert np.allclose(bond_lengths(p, np.array([[0, 1], [1, 2]])), [1.5, 2.0])
    x = wrap_angle(np.array([-np.pi, np.pi, 3 * np.pi, 0.5]))
    assert np.allclose(x, [np.pi, np.pi, np.pi, 0.5]) and np.all((x > -np.pi) & (x <= np.pi))

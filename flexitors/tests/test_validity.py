"""flexitors.eval.validity (EVAL §3.1; EXP-1 §2.6-§2.8, §5.3 test_constraints; EXP-3 §9.3)."""
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from flexitors.eval.validity import (bond_window_violations, chirality_flips, clash_count, ez_state,
                                     perceived_bonds, stereocentres, validate_conformer)
from flexitors.geometry.torsion_types import structural_stereo_bonds, td_reference_quadruple


def embedded(smiles, seed=3):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(m)
    return m


def with_positions(mol, pos):
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, p.tolist())
    return m


def pos(m):
    return m.GetConformer().GetPositions()


def test_relaxed_conformer_passes_every_check():
    for s in ('C[C@H](O)[C@@H](N)C(=O)O', 'C/C=C/C(=O)NC1CC1', 'c1ccccc1O', 'C1CC2CC1C2'):
        m = embedded(s)
        r = validate_conformer(m, ref=m, stereo_bonds=structural_stereo_bonds(m))
        assert r.valid and r.n_nonplanar == 0, (s, r)


def test_enantiomer_is_flagged():
    m = embedded('C[C@H](O)[C@@H](N)C(=O)O')
    mirror = with_positions(m, -pos(m))
    assert len(stereocentres(m)) == 2
    assert chirality_flips(mirror, m) == 2
    r = validate_conformer(mirror, ref=m)
    assert not r.valid and r.connectivity_ok and r.n_clashes == 0


def test_bond_stretch_breaks_connectivity_and_window():
    m = embedded('CCCO')
    p = pos(m)
    p[3] = p[2] + 1.6 * (p[3] - p[2])  # C2-O3 stretched by 60 %
    bad = with_positions(m, p)
    n_all, n_ring = bond_window_violations(bad, m)
    assert n_all >= 1 and n_ring == 0
    r = validate_conformer(bad, ref=m)
    assert not r.connectivity_ok and not r.valid


def test_ring_opening_is_caught():
    m = embedded('C1CCCCC1')
    p = pos(m)
    ring = Chem.MolFromSmarts('[CH2]1[CH2][CH2][CH2][CH2][CH2]1')
    order = m.GetSubstructMatch(ring)
    a, b = order[0], order[1]
    side = [a] + [n.GetIdx() for n in m.GetAtomWithIdx(a).GetNeighbors() if n.GetAtomicNum() == 1]
    shift = (p[a] - p[b]) / np.linalg.norm(p[a] - p[b]) * 1.5
    p[side] += shift
    r = validate_conformer(with_positions(m, p), ref=m)
    assert not r.rings_closed and r.n_ring_bond_violations >= 1


def test_clash_detected():
    m = embedded('CCCCCC')
    p = pos(m)
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 6]
    p[heavy[5]] = p[heavy[0]] + np.array([0.6, 0.0, 0.0])
    assert clash_count(with_positions(m, p)) >= 1


def test_perceived_bonds_equal_graph_for_relaxed_conformer():
    m = embedded('OCC(=O)N')
    graph = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in m.GetBonds()}
    assert perceived_bonds(m) == graph


def test_ez_flip_detected_and_planarity():
    m = embedded('C/C=C/CC')
    (b,) = structural_stereo_bonds(m)
    match, dev = ez_state(pos(m), [b])
    assert match.all() and dev[0] < 5
    # rotate the side TD rotates by pi: E -> Z, still planar
    c, a, bb, d = td_reference_quadruple(m, b.bond_idx)
    side = _side(m, a, bb)
    p = _rotate(pos(m), a, bb, side, math.pi)
    match, dev = ez_state(p, [b])
    assert not match.any() and dev[0] < 5
    # rotate by pi/2: non-planar
    _, dev = ez_state(_rotate(pos(m), a, bb, side, math.pi / 2), [b])
    assert dev[0] > 80
    r = validate_conformer(with_positions(m, p), ref=m, stereo_bonds=[b])
    assert r.n_ez_mismatches == 1 and not r.valid


def _side(m, a, b):
    seen, stack = {b}, [b]
    while stack:
        u = stack.pop()
        for n in m.GetAtomWithIdx(u).GetNeighbors():
            v = n.GetIdx()
            if v != a and v not in seen:
                seen.add(v)
                stack.append(v)
    return sorted(seen - {b})


def _rotate(p, a, b, side, angle):
    p = p.copy()
    axis = (p[b] - p[a]) / np.linalg.norm(p[b] - p[a])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
    p[side] = (p[side] - p[b]) @ rot.T + p[b]
    return p


def test_no_reference_leaves_reference_counts_unset():
    m = embedded('CCO')
    r = validate_conformer(m)
    assert r.n_bond_violations is None and r.n_chirality_flips is None and r.valid

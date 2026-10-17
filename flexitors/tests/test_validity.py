"""flexitors.eval.validity (EVAL §3.1; EXP-1 §2.6-§2.8, §5.3 test_constraints; EXP-3 §9.3; reviews P1/P2 M1)."""
import math
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from flexitors.eval.validity import (bond_window_mask, chirality_centres, chirality_flip_mask, clash_index,
                                     clash_mask, ez_state, perceived_bonds, validate_conformer)
from flexitors.geometry.torsion_types import structural_stereo_bonds, td_reference_quadruple
from flexitors.td import REPO_ROOT


def embedded(smiles, seed=3, n=1, relax=True):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    cids = list(AllChem.EmbedMultipleConfs(m, n, randomSeed=seed))
    assert len(cids) == n
    if relax:
        AllChem.MMFFOptimizeMoleculeConfs(m)
    return m


def seed_and_relaxed(smiles):
    """an unrelaxed ETKDG seed and its MMFF-relaxed version: same atom labels, different geometry (review P1 N6)"""
    seed = embedded(smiles, relax=False)
    relaxed = Chem.Mol(seed)
    AllChem.MMFFOptimizeMolecule(relaxed)
    return seed, relaxed


def conf(m, cid):
    return Chem.Mol(m, confId=cid)


def with_positions(mol, pos):
    m = Chem.Mol(mol)
    c = m.GetConformer()
    for i, p in enumerate(pos):
        c.SetAtomPosition(i, p.tolist())
    return m


def pos(m):
    return m.GetConformer().GetPositions()


def test_relaxed_conformer_passes_against_its_seed_and_another_conformer():
    for s in ('C[C@H](O)[C@@H](N)C(=O)O', 'C/C=C/C(=O)NC1CC1', 'c1ccccc1O', 'C1CC2CC1C2', 'C12C3C4C1C5C2C3C45'):
        seed, relaxed = seed_and_relaxed(s)
        r = validate_conformer(relaxed, ref=seed, stereo_bonds=structural_stereo_bonds(seed))
        assert r.valid is True and r.n_nonplanar == 0, (s, r)
        m = embedded(s, n=2)  # an independent reference conformer: graph-stereogenic centres only
        r = validate_conformer(conf(m, 1), ref=conf(m, 0), ref_is_seed=False)
        assert r.valid is True, (s, r)


def test_bicyclopentane_extra_perceived_bond_is_explained_by_the_reference():
    """review P2 B1: correct bicyclo[1.1.1]pentanes get one extra perceived (1,3-bridgehead) bond"""
    for s in ('C1C2CC1C2', 'OC12CC(C1)C2', 'N#CC12CC(C1)C2'):
        seed, relaxed = seed_and_relaxed(s)
        r = validate_conformer(relaxed, ref=seed)
        assert r.n_extra_bonds == 1 and r.n_extra_bonds_unexplained == 0 and r.n_missing_bonds == 0
        assert r.valid is True, s


def test_no_reference_is_undetermined():
    m = embedded('C[C@H](N)O')
    r = validate_conformer(m)
    assert r.valid is None and r.n_chirality_flips is None and r.n_bond_window_all is None


def test_enantiomer_is_flagged():
    seed, relaxed = seed_and_relaxed('C[C@H](O)[C@@H](N)C(=O)O')
    mirror = with_positions(relaxed, -pos(relaxed))
    r = validate_conformer(mirror, ref=seed)
    assert r.n_chirality_flips >= 2
    assert validate_conformer(mirror, ref=seed, ref_is_seed=False).n_chirality_flips == 2  # the two real centres and r.valid is False and r.n_missing_bonds == 0 and r.n_clashes == 0


def test_bridgehead_inversion_is_flagged():
    # EXP-1 §2.8: ring-fusion stereo via the signed volume at each bridgehead (degree-3 N in two rings here)
    m = embedded('C1CN2CCC1CC2')
    n_idx = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'N'][0]
    assert n_idx in set(chirality_centres(m)[:, 0])
    p = pos(m)
    flipped = chirality_flip_mask(-p, p, chirality_centres(m))
    assert flipped[list(chirality_centres(m)[:, 0]).index(n_idx)]


def test_bond_stretch_breaks_connectivity_and_window_is_reported_separately():
    m = embedded('CCCO')
    p = pos(m)
    p[3] = p[2] + 1.6 * (p[3] - p[2])  # C2-O3 stretched by 60 %
    r = validate_conformer(with_positions(m, p), ref=m)
    assert r.n_missing_bonds == 1 and r.n_bond_window_all >= 1 and r.valid is False
    q = pos(m)
    q[3] = q[2] + 1.16 * (q[3] - q[2])  # 16 % stretch, still perceived: window flagged, EVAL §3.1 valid
    r = validate_conformer(with_positions(m, q), ref=m)
    if r.n_missing_bonds == 0 and r.n_clashes == 0:
        assert r.n_bond_window_all == 1 and r.valid is True


def test_ring_opening_is_caught():
    m = embedded('C1CCCCC1')
    p = pos(m)
    order = m.GetSubstructMatch(Chem.MolFromSmarts('[CH2]1[CH2][CH2][CH2][CH2][CH2]1'))
    a, b = order[0], order[1]
    side = [a] + [n.GetIdx() for n in m.GetAtomWithIdx(a).GetNeighbors() if n.GetAtomicNum() == 1]
    p[side] += (p[a] - p[b]) / np.linalg.norm(p[a] - p[b]) * 1.5
    r = validate_conformer(with_positions(m, p), ref=m)
    assert not r.rings_closed and r.n_bond_window_ring >= 1 and r.valid is False


def test_clash_detected_and_primitives_are_batched():
    m = embedded('CCCCCC')
    p = pos(m)
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 6]
    q = p.copy()
    q[heavy[5]] = q[heavy[0]] + np.array([0.6, 0.0, 0.0])
    pairs, limits = clash_index(m)
    batch = np.stack([p, q])
    assert clash_mask(batch, pairs, limits).sum(axis=-1).tolist()[0] == 0
    assert clash_mask(batch, pairs, limits).sum(axis=-1)[1] >= 1
    bonds = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()])
    assert bond_window_mask(batch, p, bonds).shape == (2, len(bonds))
    assert chirality_flip_mask(batch, p, chirality_centres(m)).shape == (2, len(chirality_centres(m)))


def test_perceived_bonds_equal_graph_for_relaxed_conformer():
    m = embedded('OCC(=O)N')
    assert perceived_bonds(m) == {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in m.GetBonds()}


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


def test_ez_flip_detected_and_planarity():
    m = embedded('C/C=C/CC')
    (b,) = structural_stereo_bonds(m)
    match, dev = ez_state(pos(m), [b])
    assert match.all() and dev[0] < 5
    _, a, bb, _ = td_reference_quadruple(m, b.bond_idx)
    side = _side(m, a, bb)
    p = _rotate(pos(m), a, bb, side, math.pi)
    match, dev = ez_state(p, [b])
    assert not match.any() and dev[0] < 5
    _, dev = ez_state(_rotate(pos(m), a, bb, side, math.pi / 2), [b])
    assert dev[0] > 80
    r = validate_conformer(with_positions(m, p), ref=m, stereo_bonds=[b])
    assert r.n_ez_mismatches == 1 and r.valid is False


def test_gt_calibration_cli(tmp_path):
    rows, mols = [], {}
    for s in ('C1C2CC1C2', 'CC(O)C(=O)O', 'C1CN2CCC1CC2'):
        m = embedded(s, n=3, seed=5)
        mols[s] = [conf(m, c.GetId()) for c in m.GetConformers()]
        rows.append((s, 3, s))
    pd.DataFrame(rows, columns=['smiles', 'n_conformers', 'corrected_smiles']).to_csv(tmp_path / 't.csv', index=False)
    with open(tmp_path / 't.pkl', 'wb') as f:
        pickle.dump(mols, f)
    subprocess.run([sys.executable, '-m', 'flexitors.eval.validity', '--set', 'toy', str(tmp_path / 't.csv'),
                    str(tmp_path / 't.pkl'), '--out', str(tmp_path / 'cal.csv')], check=True, cwd=REPO_ROOT,
                   capture_output=True)
    D = pd.read_csv(tmp_path / 'cal.csv')
    assert len(D) == 9 and D.valid.all()
    assert 'invalid conformers 0' in (tmp_path / 'cal.txt').read_text()

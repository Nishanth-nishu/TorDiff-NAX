"""
Local unit tests of the round-2 geometry layer (code_plan_2 §9.1, T1-T9). Needs only rdkit + numpy + scipy.
Run:  python -m pytest -q tools/tests/test_lgeom.py      (also run on the cluster to cover RDKit 2022.9.5)
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lgeom  # noqa: E402

SMILES = ['CCOC(=O)CCN', 'CC(O)CC#N', 'OCC1CC1C=O', 'CC(C)C(N)=O', 'FC(F)CCO', 'C[C@H](N)C(=O)O', 'CC1(C)CC1O',
          'O=C1CCCN1', 'C1CC2CC1O2', 'N#CC1=CCCO1', 'CC(F)(F)F', 'C[C@@H]1CC[C@H](O)C1', 'CC=CC(O)C', 'c1ccoc1C',
          'CN(C)C=O', 'OC1COC1', 'CC12CC1C2', 'NC(=O)C1CC1', 'CCC(C)(C)O', 'O=C1NC=CC1']


def embed(smi, n=2, seed=7):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    cids = list(AllChem.EmbedMultipleConfs(m, n, randomSeed=seed))
    assert len(cids) == n, smi
    return m, [lgeom.positions(m, c) for c in cids]


def test_T1_kabsch_matches_alignmol_and_no_reflection():
    m, (X, Y) = embed('CCOC(=O)CCN')
    h = lgeom.heavy_idx(m)
    R = Rot.random(random_state=0).as_matrix()
    Yr = Y @ R.T + np.array([1.0, -2.0, 3.0])
    Yal = lgeom.kabsch_fit(Y, Yr, h)
    assert lgeom.rmsd_on(Y, Yal, h) < 1e-8
    a = lgeom.with_positions(m, X)
    b = lgeom.with_positions(m, Y)
    ref = AllChem.AlignMol(lgeom.REMOVE_HS(Chem.Mol(b)), lgeom.REMOVE_HS(a))
    assert abs(lgeom.rmsd_on(X, lgeom.kabsch_fit(X, Y, h), h) - ref) < 1e-6
    Ym = Y * np.array([-1.0, 1.0, 1.0])  # mirror image: a proper rotation cannot superpose it
    assert lgeom.rmsd_on(Y, lgeom.kabsch_fit(Y, Ym, h), h) > 0.1


def _rotate_methyl(m, X, carbon, deg):
    nb = [n.GetIdx() for n in m.GetAtomWithIdx(carbon).GetNeighbors()]
    hs = [i for i in nb if m.GetAtomWithIdx(i).GetAtomicNum() == 1]
    heavy = [i for i in nb if m.GetAtomWithIdx(i).GetAtomicNum() > 1][0]
    v = X[carbon] - X[heavy]
    R = Rot.from_rotvec(v / np.linalg.norm(v) * np.radians(deg)).as_matrix()
    Y = X.copy()
    Y[hs] = (X[hs] - X[carbon]) @ R.T + X[carbon]
    return Y, hs


def test_T3_terminal_relabelling_fixes_methyl_collapse():
    m, (X, _) = embed('CC(O)CC#N')
    Y, hs = _rotate_methyl(m, X, 0, 120.0)
    mid = 0.5 * (X + Y)
    ch_identity = [np.linalg.norm(mid[h] - mid[0]) for h in hs]
    assert max(ch_identity) < 0.75  # the defect: C-H collapses with the identity map
    res = lgeom.align_pair(m, X, Y)
    mid = 0.5 * (X + res['Y_al'])
    ch = [np.linalg.norm(mid[h] - mid[0]) for h in hs]
    assert min(ch) > 1.05 and res['n_swaps'] == 3
    assert lgeom.pair_check(m, X, res['Y_al'])['pair_ok']
    # idempotent
    res2 = lgeom.align_pair(m, X, res['Y_al'])
    assert res2['n_swaps'] == 0 and np.allclose(res2['Y_al'], res['Y_al'], atol=1e-8)


def test_T3b_no_swap_across_different_bond_order_or_charge():
    m = Chem.AddHs(Chem.MolFromSmiles('C[N+](=O)[O-]'))
    AllChem.EmbedMolecule(m, randomSeed=3)
    X = lgeom.positions(m)
    o = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'O']
    Y = X.copy()
    Y[o] = Y[o[::-1]]
    _, perm, _ = lgeom.permute_terminal(m, X, Y)
    assert all(perm[i] == i for i in o)  # =O and -O[-] are not graph-equivalent, never swapped


def test_T4_interpolation_endpoints_and_monotone_L_error():
    # torsion-matched pairs (as after conformer matching): Y = MMFF relaxation of X, i.e. same basin, different L
    n_ok = 0
    for smi in SMILES[:10]:
        m, (X, _) = embed(smi, seed=11)
        mm = lgeom.with_positions(m, X)
        AllChem.MMFFOptimizeMolecule(mm, mmffVariant='MMFF94s')
        Y = lgeom.positions(mm) @ Rot.random(random_state=1).as_matrix().T
        res = lgeom.align_pair(m, X, Y)
        Yal = res['Y_al']
        assert np.allclose((1 - 0.0) * X + 0.0 * Yal, X) and np.allclose(0.0 * X + 1.0 * Yal, Yal)
        errs = [lgeom.l_subset_errors(m, (1 - lam) * X + lam * Yal, Yal)['angle_rmsd_all_any']
                for lam in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert errs[-1] < 1e-6
        if all(a >= b - 1e-6 for a, b in zip(errs, errs[1:])):
            n_ok += 1
    assert n_ok >= 8  # angle error shrinks monotonically towards the reference for (almost) all pairs


def test_T5_enantiomer_pair_fails_stereo():
    m, (X, _) = embed('C[C@H](N)C(=O)O')
    Xm = X * np.array([-1.0, 1.0, 1.0])
    res = lgeom.align_pair(m, X, Xm)
    chk = lgeom.pair_check(m, X, res['Y_al'])
    assert not chk['stereo_ok'] and not chk['pair_ok'] and chk['reason'] == 'stereo'


def test_T6_noise_statistics_and_common_random_numbers():
    rng = np.random.default_rng([1, 2, 3])
    z = rng.normal(size=(100000, 3))
    assert abs(z.std() - 1.0) < 0.01
    for s in (0.02, 0.04):
        e = s * z
        assert abs(e.std() - s) / s < 0.02
        # bond-length error of two independently jittered endpoints ~ sqrt(2)*sigma (F6, verify_A §3)
        d0 = np.array([1.5, 0, 0])
        d = np.linalg.norm(d0 + e[:50000] - e[50000:], axis=1) - 1.5
        assert abs(np.sqrt((d ** 2).mean()) - np.sqrt(2) * s) / (np.sqrt(2) * s) < 0.05
    assert np.allclose((0.04 * z) / (0.02 * z), 2.0)


def test_T6b_no_stereo_flip_at_sigma_004():
    rng = np.random.default_rng(0)
    flips = 0
    for smi in SMILES:
        m, (X, _) = embed(smi, seed=5)
        Y = X + 0.04 * rng.normal(size=X.shape)
        flips += lgeom.stereo_labels(m, X) != lgeom.stereo_labels(m, Y)
    assert flips == 0


def test_T8_ring_acyclic_partition_is_exact_on_the_kept_part():
    for smi in ['OCC1CC1C=O', 'CC1(C)CC1O', 'C[C@@H]1CC[C@H](O)C1', 'NC(=O)C1CC1', 'O=C1CCCN1']:
        m, (X, Y) = embed(smi, seed=13)
        Yal = lgeom.align_pair(m, X, Y)['Y_al']
        ring = lgeom.set_internal_subset(m, Yal, X)   # A5-ring: GT rings, RDKit acyclic
        acyc = lgeom.set_internal_subset(m, X, Yal)   # A5-acyc: RDKit rings, GT acyclic
        e_ring = lgeom.l_subset_errors(m, ring, Yal)
        e_acyc = lgeom.l_subset_errors(m, acyc, X)
        for k in ('bond_rmsd_all_ring', 'angle_rmsd_all_ring', 'ring_dihedral_rmsd'):
            assert e_ring[k] < 1e-6, (smi, k, e_ring[k])   # ring part of A5-ring == GT exactly
            assert e_acyc[k] < 1e-6, (smi, k, e_acyc[k])   # ring part of A5-acyc == RDKit exactly
        # acyclic part copied (approximately: sequential SetAngle at branching centres)
        assert lgeom.l_subset_errors(m, acyc, Yal)['bond_rmsd_all_acyc'] < 1e-6
        assert lgeom.l_subset_errors(m, acyc, Yal)['angle_rmsd_all_acyc'] < \
            lgeom.l_subset_errors(m, X, Yal)['angle_rmsd_all_acyc'] + 1e-9


def test_T9_graph_signature_detects_renumbering():
    m, (X, _) = embed('CCOC(=O)CCN')
    order = list(range(m.GetNumAtoms()))
    order[0], order[1] = order[1], order[0]
    r = Chem.RenumberAtoms(m, order)
    assert lgeom.same_graph([m, Chem.Mol(m)]) and not lgeom.same_graph([m, r])
    w = lgeom.with_positions(m, X)
    assert w.GetNumConformers() == 1 and np.allclose(lgeom.positions(w), X)


def test_X1_seed_sources_replace_unsafe_pairs():
    ok = [True, False, True, False, False]
    assert lgeom.seed_sources(ok, interpolates=False) == [0, 1, 2, 3, 4]       # lambda 0/1, A5: no filter
    src = lgeom.seed_sources(ok, interpolates=True)
    assert len(src) == 5 and all(ok[s] for s in src) and src[0] == 0 and src[2] == 2
    assert src[1] == 0 and src[3] == 2 and src[4] == 0                            # cycling over the safe pairs
    assert lgeom.seed_sources([False, False], interpolates=True) is None           # no safe pair: dropped
    assert lgeom.seed_sources([False, False], interpolates=False) == [0, 1]


def test_X1_pair_check_grid_reports_failing_lambda():
    m, (X, _) = embed('C[C@H](N)C(=O)O')
    res = lgeom.align_pair(m, X, X * np.array([-1.0, 1.0, 1.0]))
    chk = lgeom.pair_check_grid(m, X, res['Y_al'])
    assert not chk['pair_ok'] and chk['lam'] == 0.25
    m2, (X2, _) = embed('CCOC(=O)CCN')
    ok = lgeom.pair_check_grid(m2, X2, X2.copy())
    assert ok['pair_ok'] and ok['worst_bond_dev'] < 1e-9

"""Regression test of the floor bug at tools/local_structure_analysis.py:302 (EXP-1 §9 item 1, E-JOINT-031, D-003)
and of the round-5 other-half variant (EXP-1 §10.5). Needs rdkit + numpy + pandas + scipy + networkx (no torch).
    python -m pytest -q tools/tests/test_lsa_floor.py
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def lsa(tmp_path_factory):
    """the script parses argv at import: import it once with a minimal --mode test command line"""
    out = tmp_path_factory.mktemp('lsa') / 'o.csv'
    argv = sys.argv
    sys.argv = ['local_structure_analysis.py', '--mode', 'test', '--out', str(out), '--maxiter', '5',
                '--popsize', '5']
    sys.path.insert(0, TOOLS)
    try:
        mod = importlib.import_module('local_structure_analysis')
    finally:
        sys.argv = argv
    return mod


def test_target_never_among_candidates(lsa):
    rng = np.random.default_rng(0)
    for n in range(2, 8):
        for i in range(n):
            row = rng.random(n)
            row[i] = np.inf  # job() sets cost_gg[i, i] = inf
            cand = lsa.floor_candidates(row, i, 3)
            assert i not in cand and len(cand) == min(3, n - 1)
            old = [int(j) for j in np.argsort(row)[:3]]  # the buggy slice of :302
            if n > 3:  # rows with n_gt > 3 are unchanged by the fix (the old slice never reached the inf entry)
                assert cand == old
            else:  # and for n_gt <= 3 the old slice did contain the target itself (the bug)
                assert i in old


def test_allowed_restricts_to_other_half(lsa):
    row = np.array([np.inf, 0.1, 0.2, 0.05, 0.3])
    assert lsa.floor_candidates(row, 0, 3, allowed={1, 3}) == [3, 1]
    assert lsa.floor_candidates(row, 0, 3, allowed=set()) == []


def gt_mols(smiles, n, seed):
    """unrelaxed ETKDG conformers: their local structures differ, so no torsion change maps one onto another"""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    cids = list(AllChem.EmbedMultipleConfs(m, n, randomSeed=seed))
    return [Chem.Mol(m, confId=c) for c in cids]


def test_two_conformer_floor_no_longer_collapses(lsa):
    """the bug: with n_gt = 2 the target itself was a candidate, so floor_gtother ~ 0 (all 769 rows with n_gt <= 3
    had floor_gtother <= 2e-5 A). A rigid bicycle has no heavy torsion, so the floor is the aligned RMSD itself."""
    smi = 'C1CC2CCC1C2'
    confs = gt_mols(smi, 2, 5)
    rows = pd.DataFrame(lsa.job((smi, smi, confs, ({0}, {1}, 2)))).sort_values('gt_idx')
    assert 'error' not in rows
    amap = lsa.heavy_automorphisms(confs[0])
    want = [lsa.heavy_rmsd(Chem.Mol(confs[1 - i]), confs[i], -1, -1, amap) for i in (0, 1)]
    assert min(want) > 1e-3
    assert np.allclose(rows.floor_gtother.values, want, atol=1e-6)
    # the other-half variant with halves {0} / {1} uses exactly the other conformer
    assert np.allclose(rows.floor_gtother_half.values, want, atol=1e-6)
    assert np.allclose(rows.floor_gtother_half_all.values, want, atol=1e-6)


def test_split_misalignment_is_reported(lsa):
    """review P2 S6: split.tsv halves index the cleaned GT list; a different clean count must not be used silently"""
    confs = gt_mols('C1CC2CCC1C2', 2, 5)
    (row,) = lsa.job(('C1CC2CCC1C2', 'C1CC2CCC1C2', confs, ({0, 2}, {1}, 3)))
    assert row['error'].startswith('split_misaligned')

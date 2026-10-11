"""
Cluster (model-path) test of the round-4 split-half runs, on the outputs of a smoke run of
slurm/r4_leakfree_array.sbatch (LIMIT_MOLS=20). Skipped unless R4_SMOKE_RES (= $RES/<model run>) and R4_SPLIT are set:
    R4_SMOKE_RES=... R4_SPLIT=... python -m pytest -q tools/tests/test_leakfree_cluster.py
Checks, per condition: generated conformer i keeps the bond lengths of seed i mod |seeds| of the intended half (so the
target half's own L never enters a split run), 2 x |half| conformers per molecule, eval.pkl rows = the target half,
and that each split run and its own run share targets and conformer counts (paired leak).
"""
import os
import pickle
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import check_split_seeds as css  # noqa: E402

RES, SPLIT = os.environ.get('R4_SMOKE_RES'), os.environ.get('R4_SPLIT')
pytestmark = pytest.mark.skipif(not (RES and SPLIT), reason='needs R4_SMOKE_RES and R4_SPLIT (cluster smoke run)')
CONDS = dict(split_AtoB=('A', 'B'), split_BtoA=('B', 'A'), own_BtoB=('B', 'B'), own_AtoA=('A', 'A'))


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


@pytest.mark.parametrize('cond', list(CONDS))
def test_seed_provenance_and_shapes(cond):
    seed_h, tgt_h = CONDS[cond]
    other_h = 'B' if seed_h == 'A' else 'A'
    d = os.path.join(RES, f'r4lf_{cond}_ORACLE')
    st = css.check(os.path.join(d, 'confs.pkl'), os.path.join(SPLIT, f'seeds_{seed_h}.pkl'),
                   os.path.join(SPLIT, f'seeds_{other_h}.pkl'), os.path.join(SPLIT, f'csv_{tgt_h}.csv'))
    print(cond, st)
    assert st['n_mols'] > 0 and st['max_dev_own'] < 1e-3
    E = load(os.path.join(d, 'eval.pkl'))
    true = load(os.path.join(SPLIT, f'true_{tgt_h}.pkl'))
    for (smi, corr), r in E['results'].items():
        assert r['n_true'] == len(true[smi]) and r['rmsd'].shape == (len(true[smi]), 2 * len(true[smi]))
        assert np.isfinite(r['rmsd']).all()


@pytest.mark.parametrize('split_c,own_c', [('split_AtoB', 'own_BtoB'), ('split_BtoA', 'own_AtoA')])
def test_split_and_own_are_paired(split_c, own_c):
    S = load(os.path.join(RES, f'r4lf_{split_c}_ORACLE', 'eval.pkl'))['results']
    O = load(os.path.join(RES, f'r4lf_{own_c}_ORACLE', 'eval.pkl'))['results']
    assert set(S) == set(O)
    for k in S:
        assert S[k]['rmsd'].shape == O[k]['rmsd'].shape

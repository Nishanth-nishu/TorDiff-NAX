"""Budget-matched leak-free reference ref_half (EXP-1 §3.6 / §10.5, D-001; Q-P1-011 answer): exactness of the subset
expectation, consistency with random draws, and the analysis driver with --ref_subsample (round-4 outputs unchanged).
    python -m pytest -q tools/tests/test_leakfree_budget.py
"""
import filecmp
import itertools
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import analyze_leakfree as alf  # noqa: E402
import make_split_half as msh  # noqa: E402
from test_leakfree import write_eval  # noqa: E402


def brute(M, s, t=None):
    vals = []
    for cols in itertools.combinations(range(M.shape[1]), s):
        m = M[:, cols].min(axis=1)
        vals.append(m if t is None else (m < t).astype(float))
    return np.mean(vals, axis=0)


def test_exact_expectation_equals_enumeration():
    rng = np.random.default_rng(1)
    M = rng.random((4, 8))
    M[1, 3] = M[1, 5]  # ties
    for s in range(1, 9):
        assert np.allclose(alf.subset_min_expectation(M, s), brute(M, s), atol=1e-12)
        for t in (0.2, 0.5, 0.9):
            assert np.allclose(alf.subset_cover_prob(M, s, t), brute(M, s, t), atol=1e-12)


def test_full_subset_is_the_row_minimum_and_nan_propagates():
    M = np.array([[0.3, 0.1, 0.2], [0.5, np.nan, 0.4]])
    e = alf.subset_min_expectation(M, 3)
    assert e[0] == pytest.approx(0.1) and np.isnan(e[1])
    assert np.isnan(alf.subset_cover_prob(M, 2, 0.5)[1])


def test_random_draws_agree_within_three_se():
    rng = np.random.default_rng(2)
    M = rng.gamma(2.0, 0.2, size=(3, 60))
    s = 14
    draws = np.array([M[:, rng.choice(60, s, replace=False)].min(axis=1) for _ in range(20000)])
    se = draws.std(axis=0, ddof=1) / np.sqrt(len(draws))
    assert np.all(np.abs(draws.mean(axis=0) - alf.subset_min_expectation(M, s)) < 3 * se)


def build(tmp_path, rng):
    mols = {'m1': 2, 'm2': 3, 'm3': 5, 'm4': 4}
    split = pd.DataFrame([dict(smiles=m, corrected_smiles=m, n_csv=n + 1, n_clean=n, n_bin=msh.n_bin(n),
                               idx_A=','.join(map(str, msh.split_indices(n)[0])),
                               idx_B=','.join(map(str, msh.split_indices(n)[1]))) for m, n in mols.items()])
    sd = tmp_path / 'split'
    sd.mkdir()
    split.to_csv(sd / 'split.tsv', sep='\t', index=False)
    nn = [dict(corrected_smiles=m, half=h, gt_idx=i, nn_rmsd=0.05 if m == 'm1' else 0.3) for m, n in mols.items()
          for h, idx in zip('AB', msh.split_indices(n)) for i in idx]
    pd.DataFrame(nn).to_csv(sd / 'nn.tsv', sep='\t', index=False)
    res = tmp_path / 'res'
    full = {}
    for fam, seeds in alf.SEEDS.items():
        for s in seeds:
            run = alf.RUN[fam].format(s)
            for c in alf.CONDS:
                mats = {}
                for m, n in mols.items():
                    a, b = msh.split_indices(n)
                    tgt = b if c in ('split_AtoB', 'own_BtoB') else a
                    mats[m] = 0.2 + 0.3 * rng.random((len(tgt), 2 * len(tgt)))
                write_eval(os.path.join(res, run, alf.TAG.format(c), 'eval.pkl'), mats)
            # full-set run: all n_clean targets x 2 n_csv generated conformers
            full[(fam, s)] = {m: 0.1 + 0.4 * rng.random((n, 2 * (n + 1))) for m, n in mols.items()}
            write_eval(os.path.join(res, run, 'steps20_seed0', 'eval.pkl'), full[(fam, s)])
    return res, sd, mols, full


def test_driver_ref_half(tmp_path):
    res, sd, mols, full = build(tmp_path, np.random.default_rng(3))
    out0, out1 = tmp_path / 'out0', tmp_path / 'out1'
    alf.analyse(str(res), str(sd), str(out0), n_boot=200)
    pd.DataFrame(dict(corrected_smiles=list(mols), m=[0, 2, 0, 3])).to_csv(tmp_path / 'strata.csv', index=False)
    alf.analyse(str(res), str(sd), str(out1), n_boot=200, ref_subsample=10, strata=str(tmp_path / 'strata.csv'))
    for f in os.listdir(out0):  # round-4 outputs are unchanged by the option
        if not f.endswith('.gz'):
            assert filecmp.cmp(out0 / f, out1 / f, shallow=False), f
    P = pd.read_csv(out1 / 'budget_per_model.csv')
    C = pd.read_csv(out1 / 'budget_contrasts.csv')
    # ref_half of CR by hand: per seed, per molecule, the two directions with s = 2|H| out of 2 n_csv columns
    want = []
    for s in alf.SEEDS['CR']:
        per = []
        for m, n in mols.items():
            a, b = msh.split_indices(n)
            M = full[('CR', s)][m]
            per.append(np.mean([brute(M[a], 2 * len(a)).mean(), brute(M[b], 2 * len(b)).mean()]))
        want.append(np.mean(per))
    got = P[(P.model == 'CR') & (P.view == 'ref_half') & (P.metric == 'AMR-R')].value.iloc[0]
    assert got == pytest.approx(np.mean(want))
    fam = C[C.family == 'budget']
    assert len(fam) == 8 and fam.p_holm.notna().all()
    assert set(fam.id) == {f'B.{m}.split-ref_half' for m in ('CR', 'B1', 'S3', 'S4')}
    assert set(fam.metric) == {'AMR-R', 'COV-R@0.5'}
    assert (C.family == 'descriptive').sum() == 4 * 3
    h = C[C.family == 'post hoc']
    assert set(h.id) == {f'H.{m}.split-ref_half.{lab}' for m in ('CR', 'B1', 'S3', 'S4') for lab in ('m=0', 'm>0')}
    assert (h.groupby(['id']).n.first().groupby(lambda i: i.split('.')[1]).sum() == 4).all() and h.p_holm.isna().all()
    d = C[C.id == 'B.CR.split_d-ref_half_d']
    assert len(d) == 3 and (d.n == 3).all()  # m1 has no distinct target
    assert d.note.str.contains('post hoc').all()
    assert (out1 / 'budget_mc_check.csv').exists() and (out1 / 'budget_matched_summary.txt').exists()

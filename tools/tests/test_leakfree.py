"""
Local tests of the round-4 leak-free (split-half) oracle test: tools/make_split_half.py, the GT-subset evaluation
(evaluate_confs.py UNCHANGED, fed a subset true_mols pickle), tools/check_split_seeds.py and tools/analyze_leakfree.py.
Needs rdkit + numpy + pandas + tqdm (no torch / PyG). Run from the repo root:
    python -m pytest -q tools/tests/test_leakfree.py
"""
import hashlib
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(TOOLS)
TD = os.path.join(ROOT, 'torsional-diffusion')
sys.path.insert(0, TOOLS)
import analyze_leakfree as alf  # noqa: E402
import check_split_seeds as css  # noqa: E402
import make_split_half as msh  # noqa: E402

# golden pins (LF-normalised sha256 of the committed files at round-3 HEAD 782e7d6): the split-half test must not
# need ANY change to the evaluator or the generator
EVAL_SHA = '67052cadccc365041c44c054add60865d8501d712b831548e9c06698ef09a3f4'
GEN_SHA = '706977cdd6fb5d7bc5da256ff070c5d7dc0a1b643331957ac79afa644e4df451'
SAMPLING_SHA = '412bff7efb168b3f7c4c107d009dc0630b07e7be93b2c80774de2c1dfc302006'

# smiles -> number of GT conformers (CCCO also gets a foreign conformer that clean_confs must drop)
SPEC = {'CCOC(=O)CCN': 5, 'OCC1CC1C=O': 4, 'CC(C)C(N)=O': 1, 'CC1(C)CC1O': 2, 'CCCOC': 3}
MISSING = 'CCN(C)C=O'  # in the csv, not in true_mols


def lf_sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read().replace(b'\r\n', b'\n')).hexdigest()


def fake_gt(smi, n, seed):
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
    for iso in EnumerateStereoisomers(Chem.MolFromSmiles(smi)):
        m = Chem.AddHs(iso)
        cids = list(AllChem.EmbedMultipleConfs(m, n, randomSeed=seed))
        if len(cids) == n:
            break
    # no MMFF: relaxed fake conformers collapse onto shared minima (bond-length-identical duplicates across halves)
    return [Chem.Mol(m, confId=c) for c in cids]


@pytest.fixture(scope='module')
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp('lf')
    rows, true_mols = [], {}
    for i, (s, n) in enumerate(SPEC.items()):
        gts = fake_gt(s, n, 100 + i)
        if s == 'CCCOC':  # a foreign conformer at raw index 1: cleaned index != raw index afterwards
            gts = gts[:1] + fake_gt('CCCCO', 1, 7) + gts[1:]
        true_mols[s] = gts
        rows.append((s, n + 1, s))  # n_csv deliberately != n_clean (the split uses n_clean)
    rows.append((MISSING, 2, MISSING))
    pd.DataFrame(rows, columns=['smiles', 'n_conformers', 'corrected_smiles']).to_csv(d / 'test.csv', index=False)
    with open(d / 'true.pkl', 'wb') as f:
        pickle.dump(true_mols, f)
    cnt = msh.build(str(d / 'test.csv'), str(d / 'true.pkl'), str(d / 'split'))
    return d, true_mols, cnt


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def clean(smi, confs):
    return msh.clean_confs(smi, confs)


def test_split_indices_alternate():
    assert msh.split_indices(2) == ([0], [1])
    assert msh.split_indices(5) == ([0, 2, 4], [1, 3])
    for n in range(2, 40):
        a, b = msh.split_indices(n)
        assert sorted(a + b) == list(range(n)) and not set(a) & set(b)
        assert len(a) == (n + 1) // 2 and len(b) == n // 2
        assert len(a) <= 2 * len(b) and len(b) <= 2 * len(a)  # every other-half conformer is among the cycled seeds


def test_population_and_files(fake):
    d, _, cnt = fake
    true_mols = load(d / 'true.pkl')  # as the builder sees it (RDKit pickles coordinates as float32 by default)
    assert cnt['test_rows'] == 6 and cnt['missing_in_true_mols'] == 1
    assert cnt['excluded_single_conformer'] == 1 and cnt['eligible'] == 4
    sp = d / 'split'
    S = pd.read_csv(sp / 'split.tsv', sep='\t')
    assert set(S.smiles) == {'CCOC(=O)CCN', 'OCC1CC1C=O', 'CC1(C)CC1O', 'CCCOC'}
    for h in 'AB':
        seeds, trues = load(sp / f'seeds_{h}.pkl'), load(sp / f'true_{h}.pkl')
        csv = pd.read_csv(sp / f'csv_{h}.csv')
        assert list(csv.columns) == ['smiles', 'n_conformers', 'corrected_smiles']
        assert set(seeds) == set(trues) == set(csv.smiles) == set(S.smiles)
        for raw, n, corr in csv.values:
            g = clean(corr, true_mols[raw])
            idx = msh.split_indices(len(g))[0 if h == 'A' else 1]
            assert n == len(idx) == len(seeds[raw]) == len(trues[raw])
            for k, i in enumerate(idx):  # exact GT conformers of that half, in cleaned order
                assert np.array_equal(seeds[raw][k].GetConformer().GetPositions(), g[i].GetConformer().GetPositions())
                assert trues[raw][k] is seeds[raw][k] or np.array_equal(
                    trues[raw][k].GetConformer().GetPositions(), g[i].GetConformer().GetPositions())
    # the foreign conformer was dropped before splitting: CCCOC has 3 clean conformers -> A {0, 2}, B {1}
    assert S.set_index('smiles').loc['CCCOC', 'n_clean'] == 3
    nn = pd.read_csv(sp / 'nn.tsv', sep='\t')
    assert len(nn) == int(S.n_clean.sum()) and (nn.nn_rmsd >= 0).all()


def test_evaluator_unchanged_golden():
    """the default evaluator (and the generator / sampler the split-half runs use) are byte-identical to round 3"""
    assert lf_sha(os.path.join(TD, 'evaluate_confs.py')) == EVAL_SHA
    assert lf_sha(os.path.join(TD, 'generate_confs.py')) == GEN_SHA
    assert lf_sha(os.path.join(TD, 'diffusion', 'sampling.py')) == SAMPLING_SHA


def fake_generated(true_mols, factor=2, seed=11):
    preds = {}
    for i, (s, gts) in enumerate(true_mols.items()):
        m = Chem.Mol(clean(s, gts)[0])
        m.RemoveAllConformers()
        cids = list(AllChem.EmbedMultipleConfs(m, factor * len(gts), randomSeed=seed + i))
        preds[s] = [Chem.Mol(m, confId=c) for c in cids]
    return preds


def run_eval(cwd, confs, csv, true, out):
    r = subprocess.run([sys.executable, 'evaluate_confs.py', '--confs', str(confs), '--test_csv', str(csv),
                        '--true_mols', str(true), '--dataset', 'qm9', '--n_workers', '1', '--out_results', str(out)],
                       cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return load(out), r.stdout


def test_subset_evaluation_equals_full_matrix_rows(fake):
    """GT-subset evaluation (subset true_mols pickle, unchanged evaluator) == the rows of the full-GT evaluation"""
    d, true_mols, _ = fake
    preds = fake_generated(true_mols)
    with open(d / 'preds.pkl', 'wb') as f:
        pickle.dump(preds, f)
    full, _ = run_eval(TD, d / 'preds.pkl', d / 'test.csv', d / 'true.pkl', d / 'eval_full.pkl')
    sp = d / 'split'
    S = pd.read_csv(sp / 'split.tsv', sep='\t').set_index('corrected_smiles')
    for h in 'AB':
        sub, out = run_eval(TD, d / 'preds.pkl', sp / f'csv_{h}.csv', sp / f'true_{h}.pkl', d / f'eval_{h}.pkl')
        assert set(k[1] for k in sub['results']) == set(S.index)
        for (smi, corr), r in sub['results'].items():
            idx = [int(x) for x in str(S.loc[corr, f'idx_{h}']).split(',')]
            F = full['results'][(smi, corr)]['rmsd']
            assert r['n_true'] == len(idx) and r['rmsd'].shape == (len(idx), F.shape[1])
            assert np.array_equal(r['rmsd'], F[idx]), (h, corr)
        assert 'n_model_failures=0' in out


def test_check_split_seeds(fake, tmp_path):
    """a cycled 'generation' from half A passes; the same conformers attributed to half B fail"""
    d, true_mols, _ = fake
    sp = d / 'split'
    A, B = load(sp / 'seeds_A.pkl'), load(sp / 'seeds_B.pkl')
    csv = pd.read_csv(sp / 'csv_B.csv')
    confs = {}
    for raw, n, corr in csv.values:  # 2|B| conformers, conformer i = seed i mod |A| with its torsions changed
        out = []
        for i in range(2 * n):
            m = Chem.Mol(A[raw][i % len(A[raw])])
            conf = m.GetConformer()
            for b in m.GetBonds():  # rigid rotation about every acyclic heavy-heavy single bond (as TD does)
                j, k = b.GetBeginAtom(), b.GetEndAtom()
                if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE or j.GetDegree() < 2 or k.GetDegree() < 2:
                    continue
                if j.GetAtomicNum() == 1 or k.GetAtomicNum() == 1:
                    continue
                a0 = [x.GetIdx() for x in j.GetNeighbors() if x.GetIdx() != k.GetIdx()][0]
                d0 = [x.GetIdx() for x in k.GetNeighbors() if x.GetIdx() != j.GetIdx()][0]
                ang = rdMolTransforms.GetDihedralRad(conf, a0, j.GetIdx(), k.GetIdx(), d0)
                rdMolTransforms.SetDihedralRad(conf, a0, j.GetIdx(), k.GetIdx(), d0, ang + 1.0 + 0.3 * i)
            out.append(m)
        confs[corr] = out
    with open(tmp_path / 'confs.pkl', 'wb') as f:
        pickle.dump(confs, f)
    st = css.check(str(tmp_path / 'confs.pkl'), str(sp / 'seeds_A.pkl'), str(sp / 'seeds_B.pkl'), str(sp / 'csv_B.csv'))
    assert st['n_mols'] == len(csv) and st['n_missing'] == 0 and st['n_other_identical'] == 0
    with pytest.raises(AssertionError):
        css.check(str(tmp_path / 'confs.pkl'), str(sp / 'seeds_B.pkl'), str(sp / 'seeds_A.pkl'), str(sp / 'csv_B.csv'))


# ------------------------------------------------------------------------------------------------ analysis driver
def write_eval(path, mats, num_failures=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    res = {(s, s): {'n_true': M.shape[0], 'n_model': M.shape[1], 'rmsd': M} for s, M in mats.items()}
    with open(path, 'wb') as f:
        pickle.dump({'results': res, 'num_failures': num_failures}, f)


def test_analysis_driver(tmp_path):
    rng = np.random.default_rng(0)
    mols = {'m1': 2, 'm2': 3, 'm3': 5, 'm4': 4, 'm5': 2}
    split = pd.DataFrame([dict(smiles=m, corrected_smiles=m, n_csv=n, n_clean=n, n_bin=msh.n_bin(n),
                               idx_A=','.join(map(str, msh.split_indices(n)[0])),
                               idx_B=','.join(map(str, msh.split_indices(n)[1]))) for m, n in mols.items()])
    sd = tmp_path / 'split'
    sd.mkdir()
    split.to_csv(sd / 'split.tsv', sep='\t', index=False)
    nn = [dict(corrected_smiles=m, half=h, gt_idx=i, nn_rmsd=0.05 if m == 'm1' else 0.3) for m, n in mols.items()
          for h, idx in zip('AB', msh.split_indices(n)) for i in idx]
    pd.DataFrame(nn).to_csv(sd / 'nn.tsv', sep='\t', index=False)
    res = tmp_path / 'res'
    base = dict(CR=0.10, B1=0.05, S3=0.06, S4=0.07)  # own-L level; split adds +0.2 for B1/S3/S4, +0.05 for CR
    expect = {}
    for fam, seeds in alf.SEEDS.items():
        for s in seeds:
            run = alf.RUN[fam].format(s)
            for c in alf.CONDS:
                mats = {}
                for m, n in mols.items():
                    a, b = msh.split_indices(n)
                    tgt = b if c in ('split_AtoB', 'own_BtoB') else a
                    lvl = base[fam] + (0 if c.startswith('own') else (0.05 if fam == 'CR' else 0.2))
                    M = lvl + 0.01 * rng.random((len(tgt), 2 * len(tgt)))
                    M[:, 0] = lvl  # row minimum is exactly lvl
                    mats[m] = M
                if fam == 'CR' and s == 1 and c == 'own_AtoA':
                    mats['m4'][0, :] = np.nan  # NaN row -> m4 leaves the universe
                if fam == 'S3' and s == 0 and c == 'split_BtoA':
                    del mats['m5']  # model failure -> m5 leaves the universe
                write_eval(os.path.join(res, run, alf.TAG.format(c), 'eval.pkl'), mats)
            # round-2 full-set own-L reference run of the same checkpoint (all n targets, 2n conformers)
            write_eval(os.path.join(res, run, 'steps20_seed0_gtLcycle', 'eval.pkl'),
                       {m: np.full((n, 2 * n), base[fam] / 2) for m, n in mols.items()})
        expect[fam] = base[fam]
    out = tmp_path / 'out'
    models, C, S = alf.analyse(str(res), str(sd), str(out), n_boot=200)
    get = lambda f, v, m: float(models[(models.model == f) & (models.view == v) & (models.metric == m)].value.iloc[0])
    for fam in base:
        assert get(fam, 'own', 'AMR-R') == pytest.approx(base[fam])
        add = 0.05 if fam == 'CR' else 0.2
        assert get(fam, 'split', 'AMR-R') == pytest.approx(base[fam] + add)
        assert get(fam, 'leak', 'AMR-R') == pytest.approx(-add)
        assert get(fam, 'split', 'COV-R@0.1') == pytest.approx(0.0)
        assert get(fam, 'own', 'COV-R@0.1') == pytest.approx(1.0 if base[fam] < 0.1 else 0.0)
    assert get('COPY', 'split', 'AMR-R') == pytest.approx((0.05 + 0.3 + 0.3) / 3)
    # distinct-target views: m1's only targets have an other-half twin (nn 0.05 < 0.1) -> universe_d = {m2, m3}
    assert get('COPY', 'split_d', 'AMR-R') == pytest.approx(0.3)
    for fam in base:
        assert get(fam, 'split_d', 'AMR-R') == pytest.approx(get(fam, 'split', 'AMR-R'))
        assert int(models[(models.model == fam) & (models.view == 'split_d')].n.iloc[0]) == 2
    assert set(C[C.view == 'split_d'].id) >= {'S.S3-CR.split_d', 'S.B1-CR.split_d', 'S.S4-CR.split_d'}
    for fam in base:
        assert get(fam, 'ref_gtLcycle_full', 'AMR-R') == pytest.approx(base[fam] / 2)
    r = C[(C.id == 'R.S3-CR.ref_gtLcycle_full') & (C.metric == 'AMR-R')].iloc[0]
    assert r.n == 3 and r['diff'] == pytest.approx((0.06 - 0.10) / 2)
    row = C[(C.id == 'P1.S3-CR.split') & (C.metric == 'AMR-R')].iloc[0]
    assert row.n == 3 and row['diff'] == pytest.approx((0.06 + 0.2) - (0.10 + 0.05))
    assert row.ci_lo == pytest.approx(row['diff']) and row.hier_lo == pytest.approx(row['diff'])
    assert C.p_holm.notna().sum() == 4  # 2 primary contrasts x {AMR-R, COV-R@0.5}
    assert set(C[C.family == 'primary'].id) == {'P1.S3-CR.split', 'P2.B1-CR.split'}
    assert (out / 'leakfree_summary.txt').exists() and (out / 'per_mol.csv.gz').exists()


def test_per_mol_distinct_mask(tmp_path):
    M = np.array([[0.2, 0.3, 0.9, 0.9], [0.04, 0.5, 0.9, 0.9]])  # row minima 0.2 and 0.04
    write_eval(str(tmp_path / 'e.pkl'), {'x': M})
    df, _ = alf.per_mol(str(tmp_path / 'e.pkl'), {'x': np.array([True, False])})
    r = df.loc['x']
    assert r['AMR-R'] == pytest.approx(0.12) and r['COV-R@0.1'] == pytest.approx(0.5)
    assert r['AMR-R_d'] == pytest.approx(0.2) and r['COV-R@0.1_d'] == 0.0 and r['COV-R@0.5_d'] == 1.0
    df, _ = alf.per_mol(str(tmp_path / 'e.pkl'), {'x': np.array([False, False])})
    assert np.isnan(df.loc['x', 'AMR-R_d'])

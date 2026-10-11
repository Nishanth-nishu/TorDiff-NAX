"""
Round-4 leak-free true-geometry oracle test (split-half), input builder. ORACLE: everything written here is test-set
ground truth (GT) and is analysis only, never model selection (orchestration/README.md rule 5).

Why (round3/impact.md §3g; round3/panel/score_VALIDITY_r1.md V-2/V-3): the round-1/2 "true L" conditions (gtL,
gtLcycle) put each target conformer's OWN local structure (L) among the seeds, so a true-L-trained model can rebuild
that conformer from its L (conformer-identity leak). The split-half design measures recall against GT conformers whose
own L is NOT among the seeds, with no division by P0 (validity judge V-3, "direct, leak-free design").

Split (deterministic, documented): GT conformers are first filtered exactly as evaluate_confs.py clean_confs does (the
same filter and order as tools/make_seed_pickles.py, i.e. the order of test_gt_seed_confs.pkl and of the eval.pkl rows),
then split ALTERNATING BY CLEANED GT INDEX: half A = indices 0, 2, 4, ...; half B = indices 1, 3, 5, ...
(|A| = ceil(n/2), |B| = floor(n/2)). Alternating rather than random keeps the split free of RNG and balances the halves
over whatever order GEOM stores the conformers in. Molecules with n_clean = 1 (no other conformer exists) and
n_clean = 0 are excluded and counted.

Four conditions per model (generate_confs.py --seed_confs ... --seed_confs_cycle; evaluate_confs.py UNCHANGED, with a
GT-subset true_mols pickle, so the evaluator stays byte-identical):
  split_AtoB : seeds = A's GT conformers (cycled), 2|B| generated (csv_B), recall vs B only (true_B)   own L absent
  split_BtoA : seeds = B, 2|A| generated (csv_A), recall vs A only (true_A)                            own L absent
  own_BtoB   : seeds = B, 2|B| generated (csv_B), recall vs B (true_B)                                  own L present
  own_AtoA   : seeds = A, 2|A| generated (csv_A), recall vs A (true_A)                                  own L present
split-half = mean of the two split directions, own = mean of the two own directions; for each target half the split
and own runs have the same targets and the same number of generated conformers, so leak = own - split is paired.

Outputs (--out_dir):
  seeds_A.pkl / seeds_B.pkl  {raw_smi (csv col 0): [GT conformer mols of that half]}  (format of test_gt_seed_confs.pkl)
  true_A.pkl  / true_B.pkl   {csv `smiles` key: [GT conformer mols of that half]}    (format of test_mols.pkl)
  csv_A.csv   / csv_B.csv    the test csv restricted to eligible molecules, n_conformers = |A| (resp. |B|), so the
                             default 2 x n_conformers of generate_confs.py gives 2|A| (resp. 2|B|) conformers
  split.tsv                  one row per eligible molecule: smiles, corrected_smiles, n_csv, n_clean, idx_A, idx_B
  nn.tsv                     (unless --no_nn) one row per eligible GT conformer: its heavy-atom RMSD to the NEAREST GT
                             conformer of the OTHER half (GetBestRMS on RemoveHs, as evaluate_confs.py). Its mean per
                             molecule is the AMR-R of the model-free "copy the other half" baseline (every other-half
                             conformer is among the 2|half| cycled seeds), i.e. the seed contribution that a split-half
                             model result must beat (score_VALIDITY_r1.md V-8); small values flag near-duplicate
                             conformers across the halves (same basin: own-like L legitimately present)
  population.txt             counts (test rows, missing, no clean GT, single-conformer excluded, eligible, n bins)

Usage (torsional-diffusion repo root, CPU):
  python ../tools/make_split_half.py --test_csv data/QM9/test_smiles.csv --true_mols data/QM9/test_mols.pkl \
      --out_dir data/QM9/round4_splithalf
"""
import os
import pickle
from argparse import ArgumentParser
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

N_BINS = ((2, 3), (4, 10), (11, 30), (31, 10 ** 9))  # round3/impact.md §3g n_true bins


def clean_confs(smi, confs):  # identical to evaluate_confs.py clean_confs and tools/make_seed_pickles.py:38-42
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    return [c for c in confs if Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False) == smi]


def split_indices(n):
    """alternating split of cleaned GT indices 0..n-1: A = even, B = odd"""
    return list(range(0, n, 2)), list(range(1, n, 2))


def n_bin(n):
    for lo, hi in N_BINS:
        if lo <= n <= hi:
            return f'{lo}-{hi}' if hi < 10 ** 9 else f'>{lo - 1}'
    return 'n<2'


def nn_job(item):
    """heavy-atom RMSD of every GT conformer to the nearest GT conformer of the other half (evaluate_confs.py RMSD)"""
    corr, gts, ia, ib = item
    heavy = [Chem.RemoveHs(g) for g in gts]
    out = []
    for half, own, other in (('A', ia, ib), ('B', ib, ia)):
        for i in own:
            r = []
            for j in other:
                try:
                    r.append(AllChem.GetBestRMS(heavy[i], heavy[j]))
                except Exception:
                    r.append(np.nan)
            r = np.asarray(r, float)
            out.append(dict(corrected_smiles=corr, half=half, gt_idx=i,
                            nn_rmsd=float(np.nanmin(r)) if np.isfinite(r).any() else np.nan))
    return out


def build(test_csv, true_mols_path, out_dir, limit_mols=0, nn=True, n_workers=1):
    df = pd.read_csv(test_csv)
    cols = list(df.columns)
    assert cols[:3] == ['smiles', 'n_conformers', 'corrected_smiles'], cols  # generate/evaluate read these positionally
    if limit_mols:
        df = df.iloc[:limit_mols]
    with open(true_mols_path, 'rb') as f:
        true_mols = pickle.load(f)
    os.makedirs(out_dir, exist_ok=True)
    seeds = {'A': {}, 'B': {}}
    trues = {'A': {}, 'B': {}}
    rows = {'A': [], 'B': []}
    split_rows, nn_items = [], []
    cnt = Counter()
    for raw_smi, n_csv, corr in df[['smiles', 'n_conformers', 'corrected_smiles']].values:
        cnt['test_rows'] += 1
        key = raw_smi  # evaluate_confs.py indexes true_mols with the `smiles` column (== csv col 0 == raw_smi)
        if key not in true_mols:
            cnt['missing_in_true_mols'] += 1
            continue
        gts = clean_confs(corr, true_mols[key])
        n = len(gts)
        if n == 0:
            cnt['excluded_no_clean_gt'] += 1
            continue
        if n == 1:
            cnt['excluded_single_conformer'] += 1
            continue
        ia, ib = split_indices(n)
        for h, idx in (('A', ia), ('B', ib)):
            seeds[h][raw_smi] = [gts[i] for i in idx]
            trues[h][key] = [gts[i] for i in idx]
            rows[h].append((raw_smi, len(idx), corr))
        split_rows.append(dict(smiles=raw_smi, corrected_smiles=corr, n_csv=int(n_csv), n_clean=n, n_bin=n_bin(n),
                               idx_A=','.join(map(str, ia)), idx_B=','.join(map(str, ib))))
        nn_items.append((corr, gts, ia, ib))
        cnt['eligible'] += 1
        cnt[f'eligible_nbin_{n_bin(n)}'] += 1
        cnt['gt_confs_eligible'] += n
    for h in 'AB':
        with open(os.path.join(out_dir, f'seeds_{h}.pkl'), 'wb') as f:
            pickle.dump(seeds[h], f)
        with open(os.path.join(out_dir, f'true_{h}.pkl'), 'wb') as f:
            pickle.dump(trues[h], f)
        pd.DataFrame(rows[h], columns=cols[:3]).to_csv(os.path.join(out_dir, f'csv_{h}.csv'), index=False)
        cnt[f'gt_confs_{h}'] = sum(len(v) for v in seeds[h].values())
    pd.DataFrame(split_rows).to_csv(os.path.join(out_dir, 'split.tsv'), sep='\t', index=False)
    lines = ['# make_split_half.py population (ORACLE inputs; split = alternating cleaned GT index, A even / B odd)',
             f'test_csv {test_csv}', f'true_mols {true_mols_path}']
    lines += [f'{k} {v}' for k, v in sorted(cnt.items())]
    if nn and nn_items:
        if n_workers > 1:
            with Pool(n_workers) as p:
                res = p.map(nn_job, nn_items, chunksize=4)
        else:
            res = [nn_job(it) for it in nn_items]
        NN = pd.DataFrame([r for rr in res for r in rr])
        NN.to_csv(os.path.join(out_dir, 'nn.tsv'), sep='\t', index=False)
        x = NN.nn_rmsd.values
        # copy-the-other-half baseline (model-free): per molecule mean over the two halves, then over molecules
        per_mol = NN.groupby(['corrected_smiles', 'half']).nn_rmsd.mean().groupby(level=0).mean()
        lines += [f'nn_targets {len(x)}  nn_nan {int(np.isnan(x).sum())}',
                  f'nn_rmsd_median {np.nanmedian(x):.4f}  nn_rmsd_mean {np.nanmean(x):.4f}',
                  'nn_frac_below ' + ' '.join(f'{t}:{np.mean(x[np.isfinite(x)] < t):.4f}'
                                              for t in (0.05, 0.1, 0.25, 0.5)),
                  f'copy_other_half_baseline_AMR-R {per_mol.mean():.4f} (molecule mean of the half-averaged nn RMSD)']
    with open(os.path.join(out_dir, 'population.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return cnt


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--test_csv', default='data/QM9/test_smiles.csv')
    ap.add_argument('--true_mols', default='data/QM9/test_mols.pkl')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--limit_mols', type=int, default=0)
    ap.add_argument('--no_nn', action='store_true', help='skip the nearest-other-half RMSD diagnostic (nn.tsv)')
    ap.add_argument('--n_workers', type=int, default=1)
    a = ap.parse_args()
    build(a.test_csv, a.true_mols, a.out_dir, a.limit_mols, nn=not a.no_nn, n_workers=a.n_workers)

"""
Per-molecule breakdown of COV/MAT (ablation plan I) and correlation with local-structure error (ablation plan J).

Input : pickle written by the patched evaluate_confs.py (--out_results), optionally the CSV of
        tools/local_structure_analysis.py --mode test.
Output: per-molecule CSV + markdown tables printed to stdout.

Usage (repo root):
  python ../tools/breakdown.py --results ../results/qm9_baseline_eval.pkl --threshold 0.5 \
      --local_csv ../results/local_structure_test.csv --out ../results/qm9_baseline_breakdown.csv
"""
import pickle
from argparse import ArgumentParser

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr

RDLogger.DisableLog('rdApp.*')

parser = ArgumentParser()
parser.add_argument('--results', required=True)
parser.add_argument('--threshold', type=float, default=0.5,
                    help='primary threshold: 0.5 A for QM9 (TD protocol, pre-registered), 0.75 A for DRUGS')
parser.add_argument('--local_csv', default=None)
parser.add_argument('--out', default=None)
args = parser.parse_args()


def rot_bond_counts(smi):
    """(#rotatable bonds as defined by upstream utils/torsion.py:get_transformation_mask, i.e. any non-ring bond
    with >=2 atoms on both sides incl. H rotors and double bonds; #those that move heavy atoms)"""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    G = nx.Graph(); G.add_nodes_from(range(m.GetNumAtoms()))
    G.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds())
    heavy = {a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1}
    n_repo = n_heavy = 0
    for e in list(G.edges()):
        G2 = G.copy(); G2.remove_edge(*e)
        if nx.is_connected(G2):
            continue
        comps = list(nx.connected_components(G2))
        if min(len(c) for c in comps) < 2:
            continue
        n_repo += 1
        n_heavy += all(len(c & heavy) >= 2 for c in comps)
    return n_repo, n_heavy, m.GetNumHeavyAtoms()


with open(args.results, 'rb') as f:
    R = pickle.load(f)
thr = args.threshold
rows = []
for (smi, corrected), res in R['results'].items():
    M = res['rmsd']
    if np.isnan(M).all():
        continue
    n_repo, n_heavy_rot, n_heavy = rot_bond_counts(corrected)
    rows.append(dict(smiles=smi, corrected_smiles=corrected, n_true=res['n_true'], n_model=res['n_model'],
                     n_heavy=n_heavy, n_rot_repo=n_repo, n_rot_heavy=n_heavy_rot,
                     cov_r=float(np.mean(np.nanmin(M, axis=1) < thr)), mat_r=float(np.nanmean(np.nanmin(M, axis=1))),
                     cov_p=float(np.mean(np.nanmin(M, axis=0) < thr)), mat_p=float(np.nanmean(np.nanmin(M, axis=0)))))
df = pd.DataFrame(rows)
print(f"{len(df)} molecules (model failures excluded here: {R.get('num_failures', '?')})  threshold={thr}")
print('SUMMARY (from evaluate):', {k: v for k, v in (R.get('summary') or {}).items() if k != 'sweep'})

# threshold sweep from the raw matrices (failures count as 0 coverage, as in evaluate_confs.py)
nf = int(R.get('num_failures', 0))
print('\n### COV threshold sweep (mean / median over molecules, failures = 0) -- SECONDARY / exploratory; '
      'the pre-registered primary threshold is --threshold (0.5 A for QM9)\n')
print('| thr (A) | COV-R mean | COV-R median | COV-P mean | COV-P median |\n|---|---|---|---|---|')
for t in (0.05, 0.1, 0.125, 0.25, 0.375, 0.5, 0.75):
    cr = [float(np.mean(np.min(r['rmsd'], axis=1) < t)) for r in R['results'].values()] + [0] * nf
    cp = [float(np.mean(np.min(r['rmsd'], axis=0) < t)) for r in R['results'].values()] + [0] * nf
    print(f'| {t:.3f} | {100*np.mean(cr):.2f} | {100*np.median(cr):.2f} | {100*np.mean(cp):.2f} | {100*np.median(cp):.2f} |')


def table(df, col, bins, labels):
    d = df.copy()
    d['bin'] = pd.cut(d[col], bins=bins, labels=labels, right=True)
    g = d.groupby('bin', observed=False).agg(n=('cov_r', 'size'), COV_R=('cov_r', 'mean'), MAT_R=('mat_r', 'mean'),
                                             COV_P=('cov_p', 'mean'), MAT_P=('mat_p', 'mean'))
    g[['COV_R', 'COV_P']] *= 100
    print(f'\n### by {col}\n')
    try:
        print(g.round(3).to_markdown())
    except ImportError:  # tabulate not installed
        print(g.round(3))


table(df, 'n_heavy', [0, 5, 6, 7, 8, 9, 100], ['<=5', '6', '7', '8', '9', '>9'])
table(df, 'n_rot_heavy', [-1, 0, 1, 2, 3, 100], ['0', '1', '2', '3', '4+'])
table(df, 'n_rot_repo', [-1, 1, 2, 3, 4, 6, 100], ['<=1', '2', '3', '4', '5-6', '7+'])
table(df, 'n_true', [0, 1, 3, 10, 30, 10000], ['1', '2-3', '4-10', '11-30', '>30'])

if args.local_csv:
    L = pd.read_csv(args.local_csv)
    L = L[L.get('error').isna()] if 'error' in L else L
    # floor_best_sym (min over all K = 2L seeds, symmetry-aware) is the test-time analogue of AMR-R; floor_rmsd
    # (one-to-one over L seeds, = conformer matching) overestimates it (review/cross_validation.md M1).
    # Older CSVs only have floor_rmsd.
    fcol = 'floor_best_sym' if 'floor_best_sym' in L else 'floor_rmsd'
    print(f'floor column used: {fcol}')
    agg = L.groupby('corrected_smiles').agg(floor_mean=(fcol, 'mean'), floor_max=(fcol, 'max'),
                                            frac_floor_gt_thr=(fcol, lambda x: float(np.mean(x > thr))),
                                            angle_rmsd=('angle_rmsd_assigned', 'mean'),
                                            bond_rmsd=('bond_rmsd_assigned', 'mean'),
                                            stereo_match=('stereo_match', 'mean')).reset_index()
    df = df.merge(agg, on='corrected_smiles', how='left')
    print('\n### local-structure error vs. final error (Spearman rho, p)\n')
    for x in ('floor_mean', 'angle_rmsd', 'bond_rmsd', 'frac_floor_gt_thr'):
        for y in ('mat_r', 'cov_r', 'mat_p'):
            d = df[[x, y]].dropna()
            if len(d) > 3:
                rho, p = spearmanr(d[x], d[y])
                print(f'{x:>18s} vs {y}: rho={rho:+.3f} p={p:.2e} n={len(d)}')
    d = df.dropna(subset=['frac_floor_gt_thr'])
    # Not a hard bound: independent ETKDG draws, heuristic DE, K may differ from the run. Spearman rho above is
    # confounded by size/flexibility/n_true (both floor and AMR grow with them) -- read it with the strata tables and
    # the run-matched floor (local_structure_analysis.py --mode run).
    print(f'\nESTIMATED COV-R ceiling for torsion-only models on RDKit L (1 - mean frac_floor>thr): '
          f'{100 * (1 - d.frac_floor_gt_thr.mean()):.2f}%  vs achieved COV-R {100 * d.cov_r.mean():.2f}%')
    table(df.dropna(subset=['angle_rmsd']), 'angle_rmsd', [0, 1, 2, 3, 4, 6, 180], ['<1', '1-2', '2-3', '3-4', '4-6', '>6'])

if args.out:
    df.to_csv(args.out, index=False)
    print('wrote', args.out)

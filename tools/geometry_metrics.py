"""
Local-geometry error of GENERATED conformers against their matched ground-truth conformers
(recommended local-structure metrics: bond-length / bond-angle MAE, plus heavy-torsion error).

For every molecule in an evaluate_confs.py --out_results pickle (patched evaluator):
  recall side   : each GT conformer l  <-> generated conformer k* = argmin_k RMSD[l, k]
  precision side: each generated conf k <-> GT conformer l*     = argmin_l RMSD[l, k]
and for each pair: heavy-atom bond-length MAE (A), bond-angle MAE (deg), torsion MAE over heavy-atom rotatable-bond
dihedrals (deg, circular; bonds with an sp centre are skipped because their dihedral is ill-defined). The atom
mapping gen->GT is the heavy-atom graph isomorphism with the lowest aligned heavy-atom RMSD (the correspondence
GetBestRMS uses). Units: bond MAE in A, angle/torsion MAE in degrees.

With torsional diffusion the generated bond lengths/angles are EXACTLY those of the ETKDG seed (rigid torsion
updates), so the angle MAE here is the local-structure error that a FlexiTors angle factor must reduce.

Usage (repo root):
  python ../tools/geometry_metrics.py --results ../results/qm9_baseline_s0/R0_base_seed0/eval.pkl \
      --confs ../results/qm9_baseline_s0/R0_base_seed0/confs.pkl --test_csv data/QM9/test_smiles.csv \
      --true_mols data/QM9/test_mols.pkl --out ../results/qm9_baseline_s0/R0_base_seed0/geometry.csv --n_workers 16
"""
import pickle
from argparse import ArgumentParser
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolTransforms

RDLogger.DisableLog('rdApp.*')
parser = ArgumentParser()
parser.add_argument('--results', required=True)
parser.add_argument('--confs', required=True)
parser.add_argument('--test_csv', required=True)
parser.add_argument('--true_mols', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--max_matches', type=int, default=200)
args = parser.parse_args()


def clean_confs(smi, confs):  # identical to evaluate_confs.py
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    return [c for c in confs if Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False) == smi]


def topology(m):
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    angles = [(a.GetIdx(), j.GetIdx(), c.GetIdx()) for j in m.GetAtoms()
              for a in j.GetNeighbors() for c in j.GetNeighbors() if a.GetIdx() < c.GetIdx()]
    G = nx.Graph(); G.add_nodes_from(range(m.GetNumAtoms())); G.add_edges_from(bonds)
    tors = []
    for u, v in bonds:
        if m.GetBondBetweenAtoms(u, v).IsInRing():
            continue
        # [metric-verifier fix] skip bonds with an sp centre (C#C, C#N, allene, and the single bonds next to them):
        # the dihedral a-u-v-d is then ill-conditioned (a-u-v or u-v-d ~180 deg) and a 0.02 A wiggle changes it by
        # tens of degrees (review/tests/test_metrics.py). The rotation across a linear chain is not measured here.
        if Chem.HybridizationType.SP in (m.GetAtomWithIdx(u).GetHybridization(), m.GetAtomWithIdx(v).GetHybridization()):
            continue
        nu = [n for n in G.neighbors(u) if n != v]; nv = [n for n in G.neighbors(v) if n != u]
        if nu and nv:
            tors.append((nu[0], u, v, nv[0]))
    return bonds, angles, tors


def kabsch_rmsd(P, Q):
    """RMSD after optimal proper rotation + translation (what AlignMol/GetBestRMS minimise for a fixed map)."""
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean()))


def values(conf, bonds, angles, tors, amap=None):
    f = (lambda i: i) if amap is None else (lambda i: amap[i])
    bl = np.array([rdMolTransforms.GetBondLength(conf, f(i), f(j)) for i, j in bonds])
    ba = np.array([rdMolTransforms.GetAngleDeg(conf, f(i), f(j), f(k)) for i, j, k in angles])
    td = np.array([rdMolTransforms.GetDihedralDeg(conf, f(a), f(b), f(c), f(d)) for a, b, c, d in tors])
    return bl, ba, td


def job(item):
    smi, corrected, M, gt, gen = item
    rows = []
    try:
        gen_h = [Chem.RemoveHs(m) for m in gen]
        gt_h = [Chem.RemoveHs(m) for m in gt]
        bonds, angles, tors = topology(gen_h[0])
        matches = gt_h[0].GetSubstructMatches(gen_h[0], uniquify=False, useChirality=False,
                                              maxMatches=args.max_matches)
        if not matches:
            return [dict(smiles=smi, error='no_match')]
        gen_vals = [values(m.GetConformer(), bonds, angles, tors) for m in gen_h]
        gt_vals = {}  # (l, match_idx) -> values

        gen_xyz = [m.GetConformer().GetPositions() for m in gen_h]
        gt_xyz = [m.GetConformer().GetPositions() for m in gt_h]

        def pair(l, k):
            # [metric-verifier fix] atom map = the automorphism with the lowest aligned heavy-atom RMSD, i.e. the
            # same correspondence GetBestRMS used to pair (l, k); previously the map minimising the angle MAE was
            # used, which biases the angle MAE low and can give torsion errors of a symmetry-equivalent permutation.
            best, best_r = None, np.inf
            for mi, amap in enumerate(matches):
                r = kabsch_rmsd(gen_xyz[k], gt_xyz[l][list(amap)])
                if r >= best_r - 1e-9:
                    continue
                if (l, mi) not in gt_vals:
                    gt_vals[(l, mi)] = values(gt_h[l].GetConformer(), bonds, angles, tors, amap)
                g = gt_vals[(l, mi)]
                s = gen_vals[k]
                dt = np.abs((s[2] - g[2] + 180) % 360 - 180)  # periodic difference, in [0, 180]
                best_r = r
                best = (float(np.mean(np.abs(s[0] - g[0]))) if len(g[0]) else 0.0,
                        float(np.mean(np.abs(s[1] - g[1]))) if len(g[1]) else 0.0,
                        float(np.mean(dt)) if len(dt) else np.nan,
                        float(np.max(np.abs(s[1] - g[1]))) if len(g[1]) else 0.0)
            return best

        for l in range(M.shape[0]):
            if np.isnan(M[l]).all():
                continue
            k = int(np.nanargmin(M[l]))
            b, a, t, amax = pair(l, k)
            rows.append(dict(smiles=smi, corrected_smiles=corrected, side='recall', gt=l, gen=k, rmsd=float(M[l, k]),
                             bond_mae=b, angle_mae=a, angle_maxerr=amax, torsion_mae=t))
        for k in range(M.shape[1]):
            if np.isnan(M[:, k]).all():
                continue
            l = int(np.nanargmin(M[:, k]))
            b, a, t, amax = pair(l, k)
            rows.append(dict(smiles=smi, corrected_smiles=corrected, side='precision', gt=l, gen=k, rmsd=float(M[l, k]),
                             bond_mae=b, angle_mae=a, angle_maxerr=amax, torsion_mae=t))
    except Exception as e:
        rows.append(dict(smiles=smi, error=repr(e)[:200]))
    return rows


def main():
    R = pickle.load(open(args.results, 'rb'))
    gen_all = pickle.load(open(args.confs, 'rb'))
    true_all = pickle.load(open(args.true_mols, 'rb'))
    items = []
    for (smi, corrected), res in R['results'].items():
        if corrected not in gen_all or smi not in true_all:
            continue
        gt = clean_confs(corrected, true_all[smi])
        if len(gt) != res['rmsd'].shape[0]:
            continue
        items.append((smi, corrected, res['rmsd'], gt, gen_all[corrected]))
    print('molecules:', len(items))
    rows = []
    if args.n_workers > 1:
        with Pool(args.n_workers) as p:
            for r in p.imap_unordered(job, items, chunksize=4):
                rows.extend(r)
    else:
        for it in items:
            rows.extend(job(it))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    for side in ('recall', 'precision'):
        d = df[df.get('side') == side] if 'side' in df else df.iloc[0:0]
        if len(d):
            per_mol = d.groupby('smiles')[['bond_mae', 'angle_mae', 'torsion_mae', 'rmsd']].mean()
            print(f'{side:9s}: per-molecule mean  bond MAE {per_mol.bond_mae.mean():.4f} A | angle MAE '
                  f'{per_mol.angle_mae.mean():.3f} deg | torsion MAE {per_mol.torsion_mae.mean():.2f} deg | matched RMSD '
                  f'{per_mol.rmsd.mean():.4f} A')
    print('wrote', args.out)


if __name__ == '__main__':
    main()

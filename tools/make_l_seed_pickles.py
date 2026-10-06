"""
S1 test-time local-structure (L) seed pickles for generate_confs.py --seed_confs --seed_confs_cycle, with metadata,
population reports and the achieved-L-error table (F5) written in the same pass (vote_1 P2-f: no separate tool).
round2/code_plan_2.md §3.1-3.2 as amended by round2/DECISION.md (D1, D2, defects 3 and 4) and vote_3 P2-4.

Sources (--sources, comma separated), all keyed by raw_smi (csv column 0, as tools/make_seed_pickles.py):
  etkdg   K = 2*n_csv ETKDG seeds embedded from the GT graph (stereo from 3D, as make_seed_pickles.py:69-70), written
          twice: unmodified (L_etkdg2L) and MMFF94s-relaxed from the SAME embeddings (L_etkdg2L_mmff, F2). MMFF is
          wrapped in try/except like diffusion/sampling.py:21-26 (defect 3); a molecule whose MMFF raises is dropped
          from BOTH pickles so their key sets are identical. Non-oracle.
  noise   GT conformer j + sigma * z_{j,r}, per axis on all atoms (F6), r in {0,1} -> 2L seeds ordered j + L*r so that
          under --seed_confs_cycle generated conformer i has source GT (i mod 2L) mod L. z is shared across sigma
          (common random numbers) and drawn from default_rng([seed, mol_idx, j, r]). ORACLE.
  interp  one conformer-matched ETKDG seed per GT conformer, built with the TRAINING recipe of
          standardize_confs.py:71-119 (von Mises Hungarian, DE popsize 15 maxiter 15, H-inclusive objective, and the
          return value of optimize_rotatable_bonds DISCARDED exactly as :106 does: defect 4, so test-time lambda=0 L
          comes from the training-time lambda=0 distribution); then lgeom.align_pair + pair_check (DECISION D1); then
          x_lam = (1-lam) X + lam Y_al for lam in {0, .25, .5, .75, 1} (lam=1 is the V18 pipeline check vs gtLcycle),
          plus the ring/acyclic oracles from the SAME pair (DECISION D2, all-atom acyclic set):
            A5ring = Y_al with every acyclic bond/angle copied from X  (GT rings, RDKit everything else)
            A5acyc = X with every acyclic bond/angle copied from Y_al  (RDKit rings, GT everything else)
          A molecule with ANY failing pair (pair_check, DE error, embed failure) is dropped from the whole family, so
          all lambda / A5 pickles share one population. ORACLE (lambda = 0 included: Hungarian-selected on test GT).

Output (--out_dir): <name>.pkl, <name>.meta.pkl ({raw_smi: [per-seed dict]}), <name>.population.txt, and
l_error.csv (one row per seed: achieved bond/angle/ring-dihedral RMSD to its source GT; F5).

Usage (repo root, CPU):
  python ../tools/make_l_seed_pickles.py --test_csv data/QM9/test_smiles.csv --true_mols data/QM9/test_mols.pkl \
      --out_dir data/QM9/round2_seeds --sources etkdg,noise,interp --n_workers 16
"""
import copy
import os
import pickle
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # torsional-diffusion repo root (utils.standardization)
import lgeom  # noqa: E402

RDLogger.DisableLog('rdApp.*')

parser = ArgumentParser()
parser.add_argument('--test_csv', default='data/QM9/test_smiles.csv')
parser.add_argument('--true_mols', default='data/QM9/test_mols.pkl')
parser.add_argument('--out_dir', required=True)
parser.add_argument('--sources', default='etkdg,noise,interp')
parser.add_argument('--sigmas', type=float, nargs='+', default=[0.01, 0.02, 0.04])
parser.add_argument('--lams', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--popsize', type=int, default=15)   # standardize_confs.py:15
parser.add_argument('--max_iter', type=int, default=15)  # standardize_confs.py:16
parser.add_argument('--limit_mols', type=int, default=0)
parser.add_argument('--n_workers', type=int, default=1)
args = parser.parse_args()
assert args.seed >= 1, 'RDKit randomSeed=0 makes all conformers identical (generate_confs.py:64-67)'


def clean_confs(smi, confs):  # identical to tools/make_seed_pickles.py:38-42 and evaluate_confs.py clean_confs
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    return [c for c in confs if Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False) == smi]


def split_confs(mol):
    return [lgeom.with_positions(mol, lgeom.positions(mol, c.GetId())) for c in mol.GetConformers()]


def err_row(mol, X, R, **kw):
    d = dict(kw)
    d.update(lgeom.l_subset_errors(mol, X, R))
    return d


def build_etkdg(raw_smi, gts, n_csv, mol_idx, cnt):
    base = copy.deepcopy(gts[0])
    Chem.AssignStereochemistryFrom3D(base, confId=base.GetConformer().GetId(), replaceExistingTags=True)
    emb = Chem.Mol(base)
    emb.RemoveAllConformers()
    K = 2 * int(n_csv)
    cids = list(AllChem.EmbedMultipleConfs(emb, numConfs=K, randomSeed=args.seed))
    if len(cids) != K:
        cnt['etkdg_embed_failed'] += 1
        return None
    seeds = split_confs(emb)
    mm = Chem.Mol(emb)
    try:  # defect 3: same guard as diffusion/sampling.py:21-26
        res = AllChem.MMFFOptimizeMoleculeConfs(mm, mmffVariant='MMFF94s')
    except Exception:
        cnt['etkdg_mmff_error'] += 1
        return None
    if len(res) != K:
        cnt['etkdg_mmff_error'] += 1
        return None
    seeds_mm = split_confs(mm)
    gt_stereo = lgeom.stereo_labels(gts[0])
    meta, meta_mm, rows = [], [], []
    # F5 for non-cycle seeds: Hungarian assignment GT conformer j <- seed on heavy angle RMSD
    for tag, ss, mt in (('L_etkdg2L', seeds, meta), ('L_etkdg2L_mmff', seeds_mm, meta_mm)):
        P = [lgeom.positions(s) for s in ss]
        G = [lgeom.positions(g) for g in gts]
        C = np.array([[lgeom.l_subset_errors(base, p, g)['angle_rmsd_heavy_any'] for p in P] for g in G])
        C = np.nan_to_num(C, nan=0.0)
        r_, c_ = linear_sum_assignment(C)
        for j, k in zip(r_, c_):
            rows.append(err_row(base, P[k], G[j], cond=tag, smiles=raw_smi, k=int(k), src_gt_idx=int(j)))
        for k, s in enumerate(ss):
            mt.append(dict(kind=tag, seed=args.seed, stereo_ok=lgeom.stereo_labels(s) == gt_stereo,
                           mmff_not_converged=int(res[k][0]) if tag.endswith('mmff') else None))
    # FIXES X11: count test-time MMFF outcomes per conformer (MMFFOptimizeMoleculeConfs: 0 converged, 1 not converged
    # within maxIters, -1 force-field setup failed; the old sum() let -1 cancel +1) and per molecule
    codes = [int(r[0]) for r in res]
    cnt['etkdg_mmff_confs'] += len(codes)
    cnt['etkdg_mmff_not_converged'] += sum(c == 1 for c in codes)
    cnt['etkdg_mmff_setup_failed'] += sum(c == -1 for c in codes)
    cnt['etkdg_mmff_mols_with_any_failure'] += int(any(c != 0 for c in codes))
    return {'L_etkdg2L': (seeds, meta), 'L_etkdg2L_mmff': (seeds_mm, meta_mm)}, rows


def build_noise(raw_smi, gts, mol_idx, cnt):
    base = gts[0]
    L = len(gts)
    gt_stereo = [lgeom.stereo_labels(g) for g in gts]
    out, rows = {}, []
    Z = {(j, r): np.random.default_rng([args.seed, mol_idx, j, r]).normal(size=(base.GetNumAtoms(), 3))
         for r in (0, 1) for j in range(L)}
    for s in args.sigmas:
        tag = f'L_noise{s:.2f}pa_cyc_ORACLE'
        seeds, meta = [], []
        for r in (0, 1):
            for j in range(L):
                G = lgeom.positions(gts[j])
                X = G + s * Z[(j, r)]
                m = lgeom.with_positions(base, X)
                ok = lgeom.stereo_labels(m) == gt_stereo[j]
                cnt[f'noise{s}_stereo_flip'] += int(not ok)
                seeds.append(m)
                meta.append(dict(kind=tag, src_gt_idx=j, rep=r, sigma=s, seed=args.seed, stereo_ok=ok))
                rows.append(err_row(base, X, G, cond=tag, smiles=raw_smi, k=j + L * r, src_gt_idx=j))
        out[tag] = (seeds, meta)
    return out, rows


def build_interp(raw_smi, gts, mol_idx, cnt):
    """training recipe of standardize_confs.py:71-119 on the test GT conformers (one matched seed per GT conformer).

    FIXES X1 (review_3 S6, review_1): nothing here drops a molecule because of ONE unsafe pair any more.
      * lambda = 0 / 1 and A5ring / A5acyc do not interpolate: every matched pair is used (no pair_ok filter).
      * lambda in (0, 1): pair_ok is checked on the grid {0.25, 0.5, 0.75} (research_check_A I1); a GT conformer
        whose pair is unsafe gets the x_lambda of one of that molecule's SAFE pairs instead (cycling over them, so
        the seed list keeps L entries); the molecule is dropped from those sets only if it has no safe pair.
      * embedding: if ETKDG returns fewer than L conformers, it is retried once with useRandomCoords=True; if still
        short but >= 1, every GT conformer takes its lowest-cost seed (seed reuse allowed; counted as
        interp_seed_reuse) instead of the one-to-one Hungarian assignment.
    Counts per set (kept / replaced / dropped) go to the population report."""
    from utils.standardization import get_torsion_angles, get_von_mises_rms, optimize_rotatable_bonds
    n = len(gts)
    mol_rdkit = copy.deepcopy(gts[0])                                    # standardize_confs.py:73
    rot = get_torsion_angles(mol_rdkit)                                  # :74
    mol_rdkit.RemoveAllConformers()                                      # :78
    cids = list(AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=n, randomSeed=args.seed))  # :79, seeded here
    if len(cids) != n:
        cnt['interp_embed_retry'] += 1
        mol_rdkit.RemoveAllConformers()
        cids = list(AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=n, randomSeed=args.seed, useRandomCoords=True))
    m = len(cids)
    if m == 0:
        cnt['interp_embed_failed'] += 1
        return None
    if rot:
        cost = np.array([[get_von_mises_rms(gts[i], mol_rdkit, rot, cids[j]) for j in range(m)] for i in range(n)])
    else:  # no rotatable bond: no DE in training either (such molecules are not in the training set at all)
        cost = np.array([[AllChem.AlignMol(Chem.Mol(mol_rdkit, confId=cids[j]), gts[i]) for j in range(m)]
                         for i in range(n)])
    if m == n:
        _, col = linear_sum_assignment(cost)                             # :93
    else:
        cnt['interp_seed_reuse'] += 1
        col = cost.argmin(axis=1)
    pairs = []
    for i in range(n):
        single = copy.deepcopy(mol_rdkit)                                # :104
        keep = cids[int(col[i])]
        [single.RemoveConformer(c) for c in cids if c != keep]           # :105
        if rot:
            try:
                # :106 -- return value DISCARDED on purpose (defect 4): the stored conformer carries the torsions of the
                # last DE/polish evaluation, exactly like the rematch training pickles
                optimize_rotatable_bonds(single, gts[i], rot, popsize=args.popsize, maxiter=args.max_iter)
            except Exception:
                cnt['interp_de_error'] += 1
                return None
        X = lgeom.positions(single)
        Y = lgeom.positions(gts[i])
        if not lgeom.same_graph([single, gts[i]]):
            cnt['interp_graph_mismatch'] += 1
            return None
        al = lgeom.align_pair(single, X, Y)
        chk = lgeom.pair_check_grid(single, X, al['Y_al'])
        if not chk['pair_ok']:
            cnt[f'interp_unsafe_pair_{chk["reason"]}'] += 1
        pairs.append((X, al, chk))
    base = gts[0]
    term = lgeom.terminal_atoms(base)
    pair_ok = [p[2]['pair_ok'] for p in pairs]
    safe = [i for i, o in enumerate(pair_ok) if o]
    out, rows = {}, []
    for lam in args.lams:
        tag = f'L_lam{lam:.2f}_cyc_ORACLE'
        srcs = lgeom.seed_sources(pair_ok, interpolates=0.0 < lam < 1.0)  # cycle over the molecule's safe pairs
        if srcs is None:
            cnt[f'{tag}:dropped_no_safe_pair'] += 1
            continue
        seeds, meta = [], []
        n_rep = sum(int(src != i) for i, src in enumerate(srcs))
        for i, src in enumerate(srcs):
            X, al, chk = pairs[src]
            Xl = lgeom.interp_x(X, al['Y_al'], lam, *term)  # same formula as S4 training
            seeds.append(lgeom.with_positions(base, Xl))
            meta.append(dict(kind=tag, slot=i, src_gt_idx=src, replaced=src != i, lam=lam, seed=args.seed,
                             heavy_rmsd=al['heavy_rmsd'], n_swaps=al['n_swaps'], pair_ok=chk['pair_ok'],
                             worst_bond_dev=chk['worst_bond_dev'], min_nonbonded=chk['min_nonbonded']))
            rows.append(err_row(base, Xl, al['Y_al'], cond=tag, smiles=raw_smi, k=i, src_gt_idx=src))
        cnt[f'{tag}:kept'] += 1
        cnt[f'{tag}:replaced_seeds'] += n_rep
        cnt[f'{tag}:mols_with_replacement'] += int(n_rep > 0)
        out[tag] = (seeds, meta)
    for tag, fn in (('L_A5ring_cyc_ORACLE', lambda X, Y: lgeom.set_internal_subset(base, Y, X)),
                    ('L_A5acyc_cyc_ORACLE', lambda X, Y: lgeom.set_internal_subset(base, X, Y))):
        seeds, meta = [], []
        for i, (X, al, chk) in enumerate(pairs):  # no pair_ok filter: A5 does not interpolate (FIXES X1)
            Xa = fn(X, al['Y_al'])
            mm = lgeom.with_positions(base, Xa)
            ok = lgeom.stereo_labels(mm) == lgeom.stereo_labels(base, al['Y_al'])
            cnt[f'{tag}_stereo_flip'] += int(not ok)
            seeds.append(mm)
            meta.append(dict(kind=tag, src_gt_idx=i, seed=args.seed, stereo_ok=ok, pair_ok=chk['pair_ok']))
            rows.append(err_row(base, Xa, al['Y_al'], cond=tag, smiles=raw_smi, k=i, src_gt_idx=i))
        cnt[f'{tag}:kept'] += 1
        out[tag] = (seeds, meta)
    cnt['interp_mols_all_pairs_safe'] += int(len(safe) == n)
    # S4 test-side subset (user ruling 2026-10-06): test molecules with >= 1 pair_ok-safe matched pair
    out['_s4_subset'] = bool(safe)
    return out, rows


def job(item):
    raw_smi, n_csv, mol_idx, gts = item
    cnt, out, rows = Counter(), {}, []
    try:
        if len(Chem.GetMolFrags(gts[0])) > 1:          # rejected at generation anyway (sampling.py:38-40)
            cnt['multifrag'] += 1
            return raw_smi, out, rows, cnt
        if not lgeom.same_graph(gts):                   # code_plan_2 V2: GT conformers must share atom order
            cnt['gt_graph_mismatch'] += 1
            return raw_smi, out, rows, cnt
        srcs = args.sources.split(',')
        for src, fn in (('etkdg', lambda: build_etkdg(raw_smi, gts, n_csv, mol_idx, cnt)),
                        ('noise', lambda: build_noise(raw_smi, gts, mol_idx, cnt)),
                        ('interp', lambda: build_interp(raw_smi, gts, mol_idx, cnt))):
            if src not in srcs:
                continue
            r = fn()
            if r is None:
                continue
            o, rw = r
            if o.pop('_s4_subset', False):
                cnt['s4_subset_member'] += 1
                out['_s4_subset'] = True
            out.update(o)
            rows.extend(rw)
    except Exception as e:
        cnt['exception'] += 1
        print('ERROR', raw_smi, repr(e)[:200], flush=True)
    return raw_smi, out, rows, cnt


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.test_csv)
    cols = list(df.columns)
    with open(args.true_mols, 'rb') as f:
        true_mols = pickle.load(f)
    items, pre = [], Counter()
    for mol_idx, row in enumerate(df.values):
        raw_smi, n_csv, smi = row[0], row[1], row[2]
        key = row[cols.index('smiles')] if 'smiles' in cols else raw_smi
        if key not in true_mols:
            pre['not_in_true_mols'] += 1
            continue
        gts = clean_confs(smi, true_mols[key])
        if not gts:
            pre['no_clean_gt'] += 1
            continue
        items.append((raw_smi, n_csv, mol_idx, gts))
    if args.limit_mols:
        items = items[:args.limit_mols]
    print('molecules:', len(items), dict(pre), flush=True)
    pickles, metas, rows, cnt = defaultdict(dict), defaultdict(dict), [], Counter(pre)
    s4_subset, mol_counts = [], defaultdict(Counter)
    pool = Pool(args.n_workers) if args.n_workers > 1 else None
    it = pool.imap(job, items, chunksize=2) if pool else map(job, items)
    for t, (raw_smi, out, rw, c) in enumerate(it):
        cnt.update(c)
        rows.extend(rw)
        if out.pop('_s4_subset', False):
            s4_subset.append(raw_smi)
        for k_, v_ in c.items():
            if ':' in k_:
                mol_counts[k_.split(':')[0]][k_.split(':')[1]] += v_
        for tag, (seeds, meta) in out.items():
            assert all(s.GetNumConformers() == 1 for s in seeds)
            assert np.all(np.isfinite(np.concatenate([lgeom.positions(s) for s in seeds])))
            pickles[tag][raw_smi] = seeds
            metas[tag][raw_smi] = meta
        if (t + 1) % 100 == 0:
            print(t + 1, dict(cnt), flush=True)
    if pool:
        pool.close()
        pool.join()
    with open(os.path.join(args.out_dir, 'S4_test_subset.txt'), 'w') as f:
        f.write('\n'.join(sorted(s4_subset)) + '\n')
    # twin rule: identical key sets for the MMFF pair (F2)
    if 'L_etkdg2L' in pickles:
        assert set(pickles['L_etkdg2L']) == set(pickles['L_etkdg2L_mmff'])
    # FIXES X1: lambda 0/1 and A5 share one population (all matched molecules); the interpolating lambda sets are a
    # subset (molecules with >= 1 safe pair)
    fam = [t for t in pickles if t in ('L_lam0.00_cyc_ORACLE', 'L_lam1.00_cyc_ORACLE') or t.startswith('L_A5')]
    for t in fam:
        assert set(pickles[t]) == set(pickles[fam[0]]), 'lambda 0/1 and A5 must share one population'
    for t in pickles:
        if t.startswith('L_lam') and fam:
            assert set(pickles[t]) <= set(pickles[fam[0]])
    for tag in pickles:
        with open(os.path.join(args.out_dir, tag + '.pkl'), 'wb') as f:
            pickle.dump(pickles[tag], f)
        with open(os.path.join(args.out_dir, tag + '.meta.pkl'), 'wb') as f:
            pickle.dump(metas[tag], f)
        dropped = sorted(set(it_[0] for it_ in items) - set(pickles[tag]))
        with open(os.path.join(args.out_dir, tag + '.population.txt'), 'w') as f:
            f.write(f'n_molecules {len(pickles[tag])} of {len(items)} candidates; dropped {len(dropped)} '
                    f'({100.0 * len(pickles[tag]) / max(len(items), 1):.1f}% kept)\n')
            if tag in mol_counts:
                f.write('set_counts ' + repr(dict(mol_counts[tag])) + '\n')
            f.write('counts ' + repr(dict(cnt)) + '\n')
            f.write('\n'.join(dropped) + '\n')
        print(f'{tag}: {len(pickles[tag])} molecules', flush=True)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(args.out_dir, 'l_error.csv'), index=False)
    if len(E):
        cols_ = ['bond_rmsd_all_any', 'angle_rmsd_all_any', 'bond_rmsd_heavy_ring', 'angle_rmsd_heavy_ring',
                 'bond_rmsd_heavy_acyc', 'angle_rmsd_heavy_acyc', 'ring_dihedral_rmsd']
        print(E.groupby(['cond', 'smiles'])[cols_].mean().groupby('cond').mean().round(4).to_string())
    print('COUNTS', dict(cnt))


if __name__ == '__main__':
    main()

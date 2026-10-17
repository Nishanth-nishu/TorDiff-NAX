"""
Round-4 leak-free true-geometry oracle test (split-half): analysis driver. ORACLE analysis only (README rule 5).
Everything below (models, conditions, metrics, contrasts, Holm family, universe, bootstrap) is declared IN THIS FILE
before any split-half result exists (commit time = pre-registration), as tools/analyze_round2.py did for round 2.

Design (tools/make_split_half.py; round3/panel/score_VALIDITY_r1.md V-3): molecules with >= 2 clean GT conformers,
halves A (even cleaned GT index) / B (odd). Per model run, four generations (seed 0, 20 steps, --seed_confs_cycle):
  split_AtoB  seeds A, 2|B| confs, recall vs B      own_BtoB  seeds B, 2|B| confs, recall vs B
  split_BtoA  seeds B, 2|A| confs, recall vs A      own_AtoA  seeds A, 2|A| confs, recall vs A
Per molecule:  split = mean(split_AtoB, split_BtoA)   own = mean(own_BtoB, own_AtoA)   leak = own - split
(the two directions are weighted equally, so the dataset mean equals the average of the two direction means).

Endpoints: AMR-R (= MAT-R) mean and COV-R@0.5 (pre-registered primary, user ruling 2026-09-29); COV-R@0.1 (key
secondary). Per molecule and file: rmin = min over generated conformers per target (the eval.pkl row minimum, exactly as
evaluate_confs.py), AMR-R = mean(rmin), COV-R@t = mean(rmin < t) (strict <).
Universe: the success INTERSECTION -- molecules with a finite AMR-R in every file (4 conditions x every run of every
model below). Model failures (molecule absent) and NaN rows are counted per file and reported.
Model aggregate: per molecule, mean over the model's training seeds; then mean over molecules. Per-seed values and the
training-seed SD are reported next to every aggregate.

Pre-declared contrasts (arm - ref, paired by molecule):
  PRIMARY (one Holm family: 2 contrasts x {AMR-R, COV-R@0.5}):
    P1  S3 - CR on split-half         P2  B1 - CR on split-half
  SECONDARY (raw p, no Holm; COV-R@0.1 rows are the key secondary):
    S4 - CR on split-half (S4 with --l_level 1; also on S4's pair_ok test subset if --s4_subset is given, with the
         round-2 LIMITATION note: S4 trains on ~12 % fewer pairs than its controls)
    S3 - CR, B1 - CR, S4 - CR on own (the leaky version, for comparison with the split-half contrast)
    leak(m) = own - split within each model m (paired within model); leak(S3) - leak(CR), leak(B1) - leak(CR)
  DISTINCT-TARGET SENSITIVITY (secondary; declared after the nn diagnostic of make_split_half.py, before any model
    result): 39 % of eligible GT conformers have an other-half GT conformer within 0.1 A heavy-atom RMSD (QM9 GEOM
    near-duplicates in heavy atoms, e.g. H-rotamers), so for them the "other" half still carries an own-like L and
    torsions. Views split_d / own_d / leak_d repeat split / own / leak on targets with nn_rmsd >= NN_MIN = 0.1 A only
    (per direction over those targets; per molecule the mean over directions that have >= 1 such target; same
    targets for split and own, so still paired). Contrasts S3 - CR, B1 - CR, S4 - CR on split_d; leak_d per model.
  REFERENCE (descriptive, same universe): the round-2 full-set runs of the same checkpoints, steps20_seed0 (RDKit L;
    S4 at lambda 0) and steps20_seed0_gtLcycle (all own GT L cycled, 2 x n_csv conformers vs all n targets; S4 at
    lambda 1), re-scored on the split-half universe; S3 - CR and B1 - CR on them (molecule bootstrap) for comparison.
  EXPLORATORY: S3 - B1 on split-half; per n_clean bin (2-3, 4-10, 11-30, >30; round3/impact.md §3g) means;
    model - COPY on split-half for CR, B1, S3, S4, where COPY is the model-free "copy the other half" baseline from
    make_split_half.py nn.tsv (the other half's GT conformers used as predictions; V-8 seed-vs-model contribution).
CIs: (a) molecule bootstrap on the seed-averaged per-molecule differences (tools/paired_compare.py `boot`, 95 %
percentile, two-sided bootstrap p); (b) hierarchical bootstrap -- training seeds resampled with replacement within each
model, then molecules (score_VALIDITY_r1.md V-4) -- 95 % percentile interval.

BUDGET-MATCHED REFERENCE (round 5; EXP-1 §3.6 and §10.5, D-001; pre-declared in the Q-P1-011 answer before any
ref_half value existed; only with --ref_subsample N, default 0 = the round-4 analysis above, outputs unchanged):
  ref_half: the full-set steps20_seed0 eval.pkl of each run (rows = cleaned GT conformers, columns = the 2 n_csv
    generated conformers) scored against half H with a uniformly random subset of s = 2|H| columns, i.e. the
    generated-sample budget of the split condition that targets H. Primary = the EXACT expectation over subsets
    (sorted per-target RMSDs r_(1) <= ... <= r_(N): E[min] = sum_i r_(i) C(N-i, s-1) / C(N, s);
    P(covered at t) = 1 - C(N-c, s) / C(N, s), c = #samples with r < t); N random draws are reported as a
    consistency check. The two directions are averaged as in view().
  BUDGET family (one Holm family of 8): CR, B1, S3, S4 x {AMR-R, COV-R@0.5} for split - ref_half, paired molecule
    bootstrap on seed-averaged differences, hierarchical interval (seeds, then molecules) beside each row.
  Secondary (no Holm): COV-R@0.1, and split_d - ref_half_d on the distinct-target universe.
  Descriptive: split - ref_rdkitL (the published full-budget gaps) on the same universe.
  POST HOC (added after the first result, review P2 S8; with --strata): split - ref_half on molecules with m = 0
    TD-rotatable bonds (TD runs no model there, generate_confs.py:150-153, so the split output is the other half's
    GT conformers for every model) and m > 0. The distinct-target rows stay pre-declared secondary; reading them as
    primary for the coupling question (EXP-1 §11) is post hoc and is labelled so in the outputs.

Usage (on gnode118, after slurm/r4_leakfree_array.sbatch):
  python tools/analyze_leakfree.py --res $RES --split_dir $QM9_DIR/round4_splithalf \
      --s4_subset $QM9_SEEDS2/S4_test_subset.txt --out $RES/analysis_r4_leakfree
"""
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.special import gammaln

# ------------------------------------------------------------------------------------------------ pre-declared
RUN = dict(CR='qm9_CTRL_rematch_100ep_e100_s{}', B1='qm9_B1_train_gtL_e100_s{}', S3='qm9_S3_mixL_p0.5_e100_s{}',
           S4='qm9_S4_lamcond_e100_s{}')
SEEDS = dict(CR=[0, 1, 2], B1=[0, 1, 2], S3=[0, 1], S4=[0, 1, 2])
CONDS = ['split_AtoB', 'split_BtoA', 'own_BtoB', 'own_AtoA']
TAG = 'r4lf_{}_ORACLE'
METRICS = ['AMR-R', 'COV-R@0.5', 'COV-R@0.1']
PRIMARY_METRICS = ['AMR-R', 'COV-R@0.5']
THRS = {'COV-R@0.5': 0.5, 'COV-R@0.1': 0.1}
# (id, family, arm, ref, view) ; view in {split, own, leak}
CONTRASTS = [
    ('P1.S3-CR.split', 'primary', 'S3', 'CR', 'split'),
    ('P2.B1-CR.split', 'primary', 'B1', 'CR', 'split'),
    ('S.S4-CR.split', 'secondary', 'S4', 'CR', 'split'),
    ('S.S3-CR.own', 'secondary', 'S3', 'CR', 'own'),
    ('S.B1-CR.own', 'secondary', 'B1', 'CR', 'own'),
    ('S.S4-CR.own', 'secondary', 'S4', 'CR', 'own'),
    ('S.leakS3-leakCR', 'secondary', 'S3', 'CR', 'leak'),
    ('S.leakB1-leakCR', 'secondary', 'B1', 'CR', 'leak'),
    ('E.S3-B1.split', 'exploratory', 'S3', 'B1', 'split'),
]
N_BINS = ((2, 3), (4, 10), (11, 30), (31, 10 ** 9))
REF_TAGS = {'ref_rdkitL': 'steps20_seed0', 'ref_gtLcycle_full': 'steps20_seed0_gtLcycle'}
NN_MIN = 0.1  # distinct-target sensitivity: other-half nearest GT conformer >= 0.1 A (heavy-atom GetBestRMS)
TGT = {'split_AtoB': 'B', 'split_BtoA': 'A', 'own_BtoB': 'B', 'own_AtoA': 'A'}  # target half of each condition
METRICS_D = [m + '_d' for m in METRICS]
CONTRASTS += [
    ('S.S3-CR.split_d', 'secondary', 'S3', 'CR', 'split_d'),
    ('S.B1-CR.split_d', 'secondary', 'B1', 'CR', 'split_d'),
    ('S.S4-CR.split_d', 'secondary', 'S4', 'CR', 'split_d'),
]


# ------------------------------------------------------------------------------------------------ helpers
def per_mol(path, masks=None):
    """per-molecule recall metrics of one eval.pkl; returns (DataFrame indexed by corrected smiles, info dict).
    masks: {corrected smiles: bool array over the eval rows (= target half, cleaned-index order)} -> also the *_d
    columns over the masked (distinct) targets only, NaN if a molecule has none."""
    with open(path, 'rb') as f:
        R = pickle.load(f)
    rows = {}
    n_nan = 0
    for (smi, corr), res in R['results'].items():
        rmin = np.min(res['rmsd'], axis=1)  # NaN rows propagate exactly as in evaluate_confs.py
        d = {'AMR-R': float(np.mean(rmin)), 'n_targets': int(res['n_true']), 'n_gen': int(res['n_model'])}
        for m, t in THRS.items():
            d[m] = float(np.mean(rmin < t))
        if masks is not None:
            mk = masks.get(corr)
            assert mk is None or len(mk) == len(rmin), f'{path} {corr}: mask {len(mk)} != rows {len(rmin)}'
            r = rmin[mk] if mk is not None else rmin[:0]
            d['AMR-R_d'] = float(np.mean(r)) if len(r) else np.nan
            for m, t in THRS.items():
                d[m + '_d'] = float(np.mean(r < t)) if len(r) else np.nan
        n_nan += int(not np.isfinite(d['AMR-R']))
        rows[corr] = d
    info = dict(num_failures=int(R.get('num_failures', 0)), n_results=len(rows), n_nan=n_nan)
    return pd.DataFrame.from_dict(rows, orient='index'), info


def view(d, v):
    """d: {cond: DataFrame} for one run -> per-molecule split / own / leak frame (METRICS columns); the *_d views
    (distinct targets) average the directions that have >= 1 distinct target"""
    if v.endswith('_d'):
        def avg(a, b):
            x, y = d[a][METRICS_D].set_axis(METRICS, axis=1), d[b][METRICS_D].set_axis(METRICS, axis=1)
            return pd.concat([x, y]).groupby(level=0).mean()  # skips NaN (direction without distinct targets)
        split, own = avg('split_AtoB', 'split_BtoA'), avg('own_BtoB', 'own_AtoA')
        return {'split_d': split, 'own_d': own, 'leak_d': own - split}[v]
    split = (d['split_AtoB'][METRICS] + d['split_BtoA'][METRICS]) / 2
    own = (d['own_BtoB'][METRICS] + d['own_AtoA'][METRICS]) / 2
    return {'split': split, 'own': own, 'leak': own - split}[v]


def boot(diff, rng, n_boot):  # = tools/paired_compare.py boot
    diff = np.asarray(diff, float)
    n = len(diff)
    bs = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))
    return float(diff.mean()), float(lo), float(hi), float(p)


def hier_ci(A, Rf, rng, n_boot):
    """A, Rf: arrays (n_seeds, n_mols) of per-molecule values (ref may be None for a within-model quantity).
    Resample seeds within each model, then molecules (same molecules for arm and ref). 95 % percentile interval."""
    n = A.shape[1]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        a = A[rng.integers(0, len(A), len(A))][:, idx].mean()
        r = Rf[rng.integers(0, len(Rf), len(Rf))][:, idx].mean() if Rf is not None else 0.0
        vals[b] = a - r
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def holm(p):
    p = np.asarray(p, float)
    order, m, adj, run = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def n_bin(n):
    for lo, hi in N_BINS:
        if lo <= n <= hi:
            return f'{lo}-{hi}' if hi < 10 ** 9 else f'>{lo - 1}'
    return 'n<2'


def nn_masks(split_dir):
    """{half: {corrected smiles: bool array (nn_rmsd >= NN_MIN) in eval-row order}} from make_split_half.py nn.tsv"""
    p = os.path.join(split_dir, 'nn.tsv')
    if not os.path.exists(p):
        return None, None
    NN = pd.read_csv(p, sep='\t').sort_values(['corrected_smiles', 'half', 'gt_idx'])
    out = {'A': {}, 'B': {}}
    for (corr, h), g in NN.groupby(['corrected_smiles', 'half']):
        out[h][corr] = g.nn_rmsd.values >= NN_MIN
    return out, NN


def load_all(res, families, masks=None):
    data, info, missing = {}, [], []
    for fam in families:
        for s in SEEDS[fam]:
            run = RUN[fam].format(s)
            d = {}
            for c in CONDS:
                p = os.path.join(res, run, TAG.format(c), 'eval.pkl')
                if not os.path.exists(p):
                    missing.append(p)
                    continue
                d[c], inf = per_mol(p, masks[TGT[c]] if masks else None)
                info.append(dict(model=fam, seed=s, cond=c, **inf))
            if len(d) == len(CONDS):
                data[(fam, s)] = d
    return data, pd.DataFrame(info), missing


def analyse(res, split_dir, out, s4_subset=None, n_boot=5000, seed=0, families=('CR', 'B1', 'S3', 'S4'),
            ref_subsample=0, strata=None):
    os.makedirs(out, exist_ok=True)
    split = pd.read_csv(os.path.join(split_dir, 'split.tsv'), sep='\t').set_index('corrected_smiles')
    masks, NN = nn_masks(split_dir)
    data, info, missing = load_all(res, families, masks)
    lines = ['# Leak-free (split-half) true-geometry oracle test: analysis (ORACLE; tools/analyze_leakfree.py)', '']
    if missing:
        lines += [f'MISSING {p}' for p in missing] + ['']
    fams = [f for f in families if all((f, s) in data for s in SEEDS[f])]
    # universe: finite AMR-R in every file of every complete model
    sets = [set(df.index[np.isfinite(df['AMR-R'])]) for k in data if k[0] in fams for df in data[k].values()]
    universe = sorted(set.intersection(*sets) & set(split.index)) if sets else []
    universe_d = [m for m in universe if masks and (masks['A'].get(m, np.zeros(0, bool)).any()
                                                    or masks['B'].get(m, np.zeros(0, bool)).any())]
    set_d = set(universe_d)
    views = ['split', 'own', 'leak'] + (['split_d', 'own_d', 'leak_d'] if masks else [])
    lines += [f'eligible molecules (split.tsv, n_clean >= 2): {len(split)}',
              f'success intersection over {sum(len(SEEDS[f]) for f in fams)} runs x 4 conditions: {len(universe)}',
              f'distinct-target universe (>= 1 target with other-half nn_rmsd >= {NN_MIN} A): {len(universe_d)}',
              f'models complete: {fams}', '']
    if NN is not None and universe:
        u = NN[NN.corrected_smiles.isin(universe)]
        lines += [f'targets in universe: {len(u)}; with nn_rmsd >= {NN_MIN} A: {int((u.nn_rmsd >= NN_MIN).sum())}', '']
    info.to_csv(os.path.join(out, 'failures.csv'), index=False)
    # per-run, per-view per-molecule frames on the universe
    V = {}
    for (fam, s), d in data.items():
        if fam not in fams:
            continue
        for v in views:
            V[(fam, s, v)] = view(d, v).loc[universe_d if v.endswith('_d') else universe]
        for c in CONDS:
            V[(fam, s, c)] = d[c].loc[universe, METRICS]

    def stack(fam, v, metric, mols=None):
        mols = (universe_d if v.endswith('_d') else universe) if mols is None else mols
        return np.stack([V[(fam, s, v)].loc[mols, metric].values for s in SEEDS[fam]])

    # ---- per-model table
    rows = []
    for fam in fams:
        for v in views + CONDS:
            for m in METRICS:
                X = stack(fam, v, m)
                per_seed = X.mean(axis=1)
                rows.append(dict(model=fam, view=v, metric=m, n=X.shape[1], value=float(X.mean()),
                                 seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan,
                                 per_seed=' / '.join(f'{x:.4f}' for x in per_seed)))
    models = pd.DataFrame(rows)
    models.to_csv(os.path.join(out, 'per_model.csv'), index=False)

    # ---- contrasts
    rng = np.random.default_rng(seed)
    crow = []
    subsets = [('all', universe)]
    if s4_subset and os.path.exists(s4_subset):
        with open(s4_subset) as f:
            sub = {ln.strip().split()[0] for ln in f if ln.strip() and not ln.startswith('#')}
        # S4_test_subset.txt may be keyed by raw or corrected smiles: accept either
        raw = split['smiles'].to_dict()
        s4mols = [m for m in universe if m in sub or raw.get(m) in sub]
        subsets.append(('S4subset', s4mols))
        lines.append(f'S4 pair_ok test subset (round 2): {len(s4mols)} of {len(universe)} universe molecules')
    for cid, family, arm, ref, v in CONTRASTS:
        if arm not in fams or ref not in fams:
            continue
        for sname, mols in subsets:
            if sname == 'S4subset' and 'S4' not in (arm, ref):
                continue
            mv = [x for x in mols if x in set_d] if v.endswith('_d') else mols
            if not mv:
                continue
            for m in METRICS:
                A, Rf = stack(arm, v, m, mv), stack(ref, v, m, mv)
                mean, lo, hi, p = boot(A.mean(0) - Rf.mean(0), rng, n_boot)
                hlo, hhi = hier_ci(A, Rf, rng, n_boot)
                sc = 1.0 if m == 'AMR-R' else 100.0
                crow.append(dict(id=cid + ('@S4subset' if sname != 'all' else ''), family=family, metric=m,
                                 n=len(mv), arm=arm, ref=ref, view=v, arm_value=A.mean() * sc,
                                 ref_value=Rf.mean() * sc, diff=mean * sc, ci_lo=lo * sc, ci_hi=hi * sc, p_boot=p,
                                 hier_lo=hlo * sc, hier_hi=hhi * sc,
                                 note='S4 LIMITATION (trains on pair_ok-safe pairs only)' if 'S4' in (arm, ref) else ''))
    # within-model leak with CI (paired own - split, same model)
    for fam in fams:
        for sfx in [''] + (['_d'] if masks else []):
            for m in METRICS:
                A = stack(fam, 'leak' + sfx, m)
                mean, lo, hi, p = boot(A.mean(0), rng, n_boot)
                hlo, hhi = hier_ci(A, None, rng, n_boot)
                sc = 1.0 if m == 'AMR-R' else 100.0
                crow.append(dict(id=f'S.leak{sfx}.{fam}', family='secondary', metric=m, n=A.shape[1], arm=fam,
                                 ref='-', view='own' + sfx + '-split' + sfx,
                                 arm_value=stack(fam, 'own' + sfx, m).mean() * sc,
                                 ref_value=stack(fam, 'split' + sfx, m).mean() * sc, diff=mean * sc, ci_lo=lo * sc,
                                 ci_hi=hi * sc, p_boot=p, hier_lo=hlo * sc, hier_hi=hhi * sc, note=''))
    # model-free COPY baseline (exploratory): other half's GT conformers as predictions, split views only
    if NN is not None and universe:
        for sfx, U, NNv in (('', universe, NN), ('_d', universe_d, NN[NN.nn_rmsd >= NN_MIN])):
            if not U:
                continue
            g = NNv.groupby(['corrected_smiles', 'half']).nn_rmsd
            cp = {'AMR-R': g.mean()}
            for m, t in THRS.items():
                cp[m] = g.apply(lambda x, t=t: float(np.mean(x.values < t)))
            copy = pd.DataFrame({m: q.groupby(level=0).mean() for m, q in cp.items()}).reindex(U)
            for m in METRICS:
                sc = 1.0 if m == 'AMR-R' else 100.0
                models = pd.concat([models, pd.DataFrame([dict(model='COPY', view='split' + sfx, metric=m, n=len(U),
                                                               value=float(copy[m].mean()), seed_sd=np.nan,
                                                               per_seed='model-free')])], ignore_index=True)
                for fam in fams:
                    A = stack(fam, 'split' + sfx, m)
                    mean, lo, hi, p = boot(A.mean(0) - copy[m].values, rng, n_boot)
                    crow.append(dict(id=f'E.{fam}-COPY.split{sfx}', family='exploratory', metric=m, n=len(U),
                                     arm=fam, ref='COPY', view='split' + sfx, arm_value=A.mean() * sc,
                                     ref_value=copy[m].mean() * sc, diff=mean * sc, ci_lo=lo * sc, ci_hi=hi * sc,
                                     p_boot=p, hier_lo=np.nan, hier_hi=np.nan,
                                     note='COPY = other-half GT conformers as predictions (no model)'))
        models.to_csv(os.path.join(out, 'per_model.csv'), index=False)
    # reference: round-2 full-set runs of the same checkpoints on this universe (descriptive)
    REF = {}
    for v, tag in REF_TAGS.items():
        for fam in fams:
            fr = []
            for s in SEEDS[fam]:
                pth = os.path.join(res, RUN[fam].format(s), tag, 'eval.pkl')
                if os.path.exists(pth):
                    fr.append(per_mol(pth)[0].reindex(universe)[METRICS])
            if len(fr) == len(SEEDS[fam]):
                REF[(fam, v)] = np.stack([f[METRICS].values for f in fr])  # (seeds, mols, metrics)
                X = REF[(fam, v)]
                for k, m in enumerate(METRICS):
                    per_seed = np.nanmean(X[:, :, k], axis=1)
                    models = pd.concat([models, pd.DataFrame([dict(
                        model=fam, view=v, metric=m, n=int(np.isfinite(X[:, :, k]).all(0).sum()),
                        value=float(np.nanmean(X[:, :, k].mean(0))), seed_sd=float(np.std(per_seed, ddof=1))
                        if len(per_seed) > 1 else np.nan, per_seed=' / '.join(f'{x:.4f}' for x in per_seed))])],
                        ignore_index=True)
        for arm in ('S3', 'B1'):
            if (arm, v) in REF and ('CR', v) in REF:
                for k, m in enumerate(METRICS):
                    a, r = REF[(arm, v)][:, :, k].mean(0), REF[('CR', v)][:, :, k].mean(0)
                    ok = np.isfinite(a) & np.isfinite(r)
                    mean, lo, hi, p = boot(a[ok] - r[ok], rng, n_boot)
                    sc = 1.0 if m == 'AMR-R' else 100.0
                    crow.append(dict(id=f'R.{arm}-CR.{v}', family='reference', metric=m, n=int(ok.sum()), arm=arm,
                                     ref='CR', view=v, arm_value=a[ok].mean() * sc, ref_value=r[ok].mean() * sc,
                                     diff=mean * sc, ci_lo=lo * sc, ci_hi=hi * sc, p_boot=p, hier_lo=np.nan,
                                     hier_hi=np.nan, note=f'round-2 {REF_TAGS[v]} re-scored on this universe'))
    models.to_csv(os.path.join(out, 'per_model.csv'), index=False)
    C = pd.DataFrame(crow)
    C['p_holm'] = np.nan
    prim = (C.family == 'primary') & C.metric.isin(PRIMARY_METRICS)
    if prim.any():
        C.loc[prim, 'p_holm'] = holm(C.loc[prim, 'p_boot'].values)
    C.to_csv(os.path.join(out, 'contrasts.csv'), index=False)

    # ---- strata by n_clean bin (exploratory)
    srow = []
    bins = split.loc[universe, 'n_clean'].map(n_bin)
    for fam in fams:
        for b in [f'{lo}-{hi}' if hi < 10 ** 9 else f'>{lo - 1}' for lo, hi in N_BINS]:
            mols = list(bins.index[bins == b])
            if not mols:
                continue
            for v in ('split', 'own', 'leak'):
                srow.append(dict(model=fam, n_bin=b, n=len(mols), view=v,
                                 **{m: float(stack(fam, v, m, mols).mean()) for m in METRICS}))
            md = [x for x in mols if x in set_d]
            for v in [w for w in views if w.endswith('_d') and md]:
                srow.append(dict(model=fam, n_bin=b, n=len(md), view=v,
                                 **{m: float(stack(fam, v, m, md).mean()) for m in METRICS}))
    S = pd.DataFrame(srow)
    S.to_csv(os.path.join(out, 'strata.csv'), index=False)

    # ---- per-molecule dump (seed-averaged; for later re-analysis without the cluster)
    pm = []
    for fam in fams:
        for v in [w for w in views if not w.startswith('leak')] + CONDS:
            X = {m: stack(fam, v, m).mean(0) for m in METRICS}
            pm.append(pd.DataFrame(dict(model=fam, view=v, corrected_smiles=universe_d if v.endswith('_d')
                                        else universe, **X)))
    if pm:
        pd.concat(pm).to_csv(os.path.join(out, 'per_mol.csv.gz'), index=False, compression='gzip')

    # ---- text report
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    piv = models[models.view.isin(views + list(REF_TAGS))].pivot_table(
        index='model', columns=['view', 'metric'], values='value', sort=False)
    lines += ['## Per model (seed-averaged; AMR-R in A, COV-R as fraction)', piv.round(4).to_string(), '',
              '## Per model and direction', models[models.view.isin(CONDS)].pivot_table(
                  index='model', columns=['view', 'metric'], values='value', sort=False).round(4).to_string(), '',
              '## Per-seed values', models[models.view.isin(views)][
                  ['model', 'view', 'metric', 'value', 'seed_sd', 'per_seed']].round(4).to_string(index=False), '',
              '## Contrasts (AMR-R in A, COV in %; diff = arm - ref; p_holm over the PRIMARY family only)',
              C.round(4).to_string(index=False), '',
              '## n_clean strata (exploratory)', S.round(4).to_string(index=False), '',
              '## Files', info.to_string(index=False)]
    txt = '\n'.join(lines)
    with open(os.path.join(out, 'leakfree_summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print(txt)
    if ref_subsample:
        budget_matched(res, split, universe, universe_d, masks, V, fams, out, ref_subsample, n_boot, seed,
                       pd.read_csv(strata) if strata else None)
    return models, C, S


# ------------------------------------------------------------------------------------------------ round 5: ref_half
BUDGET_MODELS = ['CR', 'B1', 'S3', 'S4']


def log_comb(n, k):
    """log C(n, k) elementwise; -inf where k > n or k < 0"""
    n, k = np.broadcast_arrays(np.asarray(n, float), np.asarray(k, float))
    ok = (k >= 0) & (k <= n)
    out = np.full(n.shape, -np.inf)
    out[ok] = gammaln(n[ok] + 1) - gammaln(k[ok] + 1) - gammaln(n[ok] - k[ok] + 1)
    return out


def subset_min_expectation(M, s):
    """exact E[min over a uniformly random s-subset of the columns] for every row of M (n_rows x N); NaN rows -> NaN"""
    N = M.shape[1]
    i = np.arange(1, N + 1)
    w = np.exp(log_comb(N - i, s - 1) - log_comb(N, s))  # P(the subset minimum is the i-th smallest)
    nan = np.isnan(M).any(axis=1)
    with np.errstate(invalid='ignore'):  # inf * 0 in NaN rows, overwritten below
        out = np.sort(np.where(np.isnan(M), np.inf, M), axis=1) @ w
    out[nan] = np.nan
    return out


def subset_cover_prob(M, s, t):
    """exact P(min over a random s-subset < t) per row of M (strict <, as evaluate_confs.py); NaN rows -> NaN"""
    N = M.shape[1]
    c = np.sum(M < t, axis=1)
    out = 1.0 - np.exp(log_comb(N - c, s) - log_comb(N, s))
    out[np.isnan(M).any(axis=1)] = np.nan
    return out


def ref_half_molecule(M, idx, mask, rng, n_draws):
    """budget-matched reference of one molecule. M: full-set RMSD matrix (rows = cleaned GT order); idx: {half: row
    indices}; mask: {half: bool array over that half's rows} (distinct targets) or None.
    Returns (exact metrics dict incl. *_d, list of n_draws MC AMR-R values), both direction-averaged as view()."""
    ex, ex_d, mc = {m: [] for m in METRICS}, {m: [] for m in METRICS}, np.zeros((n_draws, 2))
    for h_i, h in enumerate('AB'):
        rows = M[idx[h]]
        s = 2 * len(idx[h])
        vals = {'AMR-R': subset_min_expectation(rows, s), **{m: subset_cover_prob(rows, s, t) for m, t in THRS.items()}}
        mk = mask[h] if mask is not None else None
        for m in METRICS:
            ex[m].append(float(np.mean(vals[m])))
            if mk is not None and mk.any():
                ex_d[m].append(float(np.mean(vals[m][mk])))
        for d in range(n_draws):
            cols = rng.choice(M.shape[1], s, replace=False)
            mc[d, h_i] = float(np.mean(rows[:, cols].min(axis=1)))
    out = {m: float(np.mean(v)) for m, v in ex.items()}
    out.update({m + '_d': float(np.mean(v)) if v else np.nan for m, v in ex_d.items()})
    return out, list(mc.mean(axis=1))


def load_ref_half(res, fam, seed_, split, universe, masks, rng_mc, n_draws):
    """ref_half per molecule for one run, plus MC-check rows and exclusions (molecule absent, row count != n_clean,
    or 2|H| larger than the number of generated conformers)"""
    with open(os.path.join(res, RUN[fam].format(seed_), REF_TAGS['ref_rdkitL'], 'eval.pkl'), 'rb') as f:
        by_corr = {corr: r['rmsd'] for (smi, corr), r in pickle.load(f)['results'].items()}
    rows, mc_rows, excl = {}, [], []
    for mol in universe:
        M = by_corr.get(mol)
        n_clean = int(split.loc[mol, 'n_clean'])
        idx = {h: [int(x) for x in str(split.loc[mol, f'idx_{h}']).split(',')] for h in 'AB'}
        if M is None:
            reason = 'missing'
        elif M.shape[0] != n_clean:
            reason = f'rows {M.shape[0]} != n_clean {n_clean}'
        elif 2 * max(len(idx['A']), len(idx['B'])) > M.shape[1]:
            reason = f'2|H| > {M.shape[1]} generated'
        else:
            reason = None
        if reason:
            excl.append(dict(model=fam, seed=seed_, mol=mol, reason=reason))
            continue
        mk = {h: masks[h].get(mol, np.zeros(len(idx[h]), bool)) for h in 'AB'} if masks is not None else None
        rows[mol], mc = ref_half_molecule(M, idx, mk, rng_mc, n_draws)
        mc_rows.append(dict(model=fam, seed=seed_, mol=mol, exact=rows[mol]['AMR-R'],
                            **{f'draw{d}': v for d, v in enumerate(mc)}))
    return pd.DataFrame.from_dict(rows, orient='index'), mc_rows, excl


def mc_check_lines(MC, fams):
    """consistency of the exact expectation with the random draws (AMR-R)"""
    dcols = [c for c in MC.columns if c.startswith('draw')]
    MC['mc_mean'] = MC[dcols].mean(axis=1)
    MC['mc_se'] = MC[dcols].std(axis=1, ddof=1) / np.sqrt(len(dcols))
    z = (MC.mc_mean - MC.exact).abs() / MC.mc_se.replace(0, np.nan)
    lines = ['## MC consistency check (AMR-R; exact expectation vs the mean of the random draws)',
             f'per (run, molecule): mean |MC - exact| = {(MC.mc_mean - MC.exact).abs().mean():.5f} A; '
             f'share within 3 MC SE = {float(np.mean(z.fillna(0) <= 3)):.4f}']
    for fam in fams:
        g = MC[MC.model == fam]
        dm = g.groupby('seed')[dcols].mean().mean(axis=0)  # dataset mean per draw, averaged over seeds
        lines.append(f'  {fam}: exact {g.exact.mean():.4f}  mean over draws {dm.mean():.4f}  '
                     f'SD of the dataset mean over draws {dm.std(ddof=1):.4f}')
    return lines + ['']


def budget_matched(res, split, universe, universe_d, masks, V, fams, out, n_draws, n_boot, seed, strata=None):
    """EXP-1 §3.6 / §10.5 budget-matched leak-free reference and the pre-declared BUDGET family (module doc)."""
    rng_mc = np.random.default_rng([seed, 2026])
    rng = np.random.default_rng(seed)
    fams = [f for f in BUDGET_MODELS if f in fams]
    REFH, MC, excl = {}, [], []
    for fam in fams:
        for s_ in SEEDS[fam]:
            REFH[(fam, s_)], mc, ex = load_ref_half(res, fam, s_, split, universe, masks, rng_mc, n_draws)
            MC += mc
            excl += ex
    excl = pd.DataFrame(excl, columns=['model', 'seed', 'mol', 'reason'])
    bad = set(excl.mol)
    U = [m for m in universe if m not in bad]
    Ud = [m for m in universe_d if m not in bad]
    lines = ['# Budget-matched leak-free reference ref_half (ORACLE; EXP-1 §3.6 / §10.5; tools/analyze_leakfree.py)',
             f'split-half universe {len(universe)}; excluded from ref_half (every view and model) {len(bad)}; '
             f'analysed {len(U)}; distinct-target universe {len(Ud)}; MC check draws per half {n_draws}',
             'NOTE: the excluded molecules are not missing at random: they lack a full-set RDKit-L result (ETKDG cannot '
             'embed them) while their GT-L split runs exist; the exclusion is paired across models and views.',
             'NOTE: the MC check is descriptive (10-draw SE estimates); the exact expectation is the primary value.',
             'NOTE: *_d rows are the pre-declared secondary distinct-target view; promoting them to primary for the '
             'coupling reading (EXP-1 §11) is POST HOC.', '']
    MC = pd.DataFrame(MC)
    lines += mc_check_lines(MC, fams)
    MC.to_csv(os.path.join(out, 'budget_mc_check.csv'), index=False)
    excl.to_csv(os.path.join(out, 'budget_excluded.csv'), index=False)

    def stack(fam, col, mols, view_):
        if view_ == 'ref_half':
            return np.stack([REFH[(fam, s_)].loc[mols, col].values for s_ in SEEDS[fam]])
        if view_ == 'ref_rdkitL':
            return np.stack([REFL[(fam, s_)].loc[mols, col].values for s_ in SEEDS[fam]])
        return np.stack([V[(fam, s_, view_)].loc[mols, col].values for s_ in SEEDS[fam]])

    REFL = {(fam, s_): per_mol(os.path.join(res, RUN[fam].format(s_), REF_TAGS['ref_rdkitL'], 'eval.pkl'))[0]
            for fam in fams for s_ in SEEDS[fam]}
    levels, crow = [], []
    for fam in fams:
        for sfx, mols in (('', U), ('_d', Ud)) if masks is not None else (('', U),):
            for m in METRICS:
                A, Rh = stack(fam, m, mols, 'split' + sfx), stack(fam, m + sfx, mols, 'ref_half')
                levels += [dict(model=fam, view='split' + sfx, metric=m, n=len(mols), value=float(A.mean())),
                           dict(model=fam, view='ref_half' + sfx, metric=m, n=len(mols), value=float(Rh.mean()))]
                D = A - Rh  # (seeds, molecules), paired within model
                mean, lo, hi, p = boot(D.mean(0), rng, n_boot)
                hlo, hhi = hier_ci(D, None, rng, n_boot)
                sc = 1.0 if m == 'AMR-R' else 100.0
                crow.append(dict(id=f'B.{fam}.split{sfx}-ref_half{sfx}', metric=m, n=len(mols),
                                 family='budget' if (sfx == '' and m in PRIMARY_METRICS) else 'secondary',
                                 arm_value=A.mean() * sc, ref_value=Rh.mean() * sc, diff=mean * sc, ci_lo=lo * sc,
                                 ci_hi=hi * sc, p_boot=p, hier_lo=hlo * sc, hier_hi=hhi * sc,
                                 note='pre-declared secondary; as primary for coupling (EXP-1 §11): post hoc'
                                 if sfx else ''))
        for m in METRICS:  # descriptive: the published full-budget gap on the same molecules
            A, Rl = stack(fam, m, U, 'split'), stack(fam, m, U, 'ref_rdkitL')
            sc = 1.0 if m == 'AMR-R' else 100.0
            levels.append(dict(model=fam, view='ref_rdkitL', metric=m, n=len(U), value=float(Rl.mean())))
            crow.append(dict(id=f'D.{fam}.split-ref_rdkitL', metric=m, n=len(U), family='descriptive',
                             arm_value=A.mean() * sc, ref_value=Rl.mean() * sc, diff=float((A - Rl).mean()) * sc))
    if strata is not None:  # POST HOC (review P2 S8a), own RNG so the rows above are unchanged
        rng_ph = np.random.default_rng([seed, 8])
        mcol = strata.drop_duplicates('corrected_smiles').set_index('corrected_smiles')['m'].reindex(U)
        for fam in fams:
            for lab, mols in (('m=0', [x for x in U if mcol[x] == 0]), ('m>0', [x for x in U if mcol[x] > 0])):
                for m in PRIMARY_METRICS:
                    D = stack(fam, m, mols, 'split') - stack(fam, m, mols, 'ref_half')
                    mean, lo, hi, p = boot(D.mean(0), rng_ph, n_boot)
                    sc = 1.0 if m == 'AMR-R' else 100.0
                    crow.append(dict(id=f'H.{fam}.split-ref_half.{lab}', metric=m, n=len(mols), family='post hoc',
                                     arm_value=stack(fam, m, mols, 'split').mean() * sc,
                                     ref_value=stack(fam, m, mols, 'ref_half').mean() * sc, diff=mean * sc,
                                     ci_lo=lo * sc, ci_hi=hi * sc, p_boot=p,
                                     note='m = TD-rotatable bonds; added after the result (review P2 S8a)'))
    P, C = pd.DataFrame(levels), pd.DataFrame(crow)
    C['p_holm'] = np.nan
    fb = C.family == 'budget'
    C.loc[fb, 'p_holm'] = holm(C.loc[fb, 'p_boot'].values)
    P.to_csv(os.path.join(out, 'budget_per_model.csv'), index=False)
    C.to_csv(os.path.join(out, 'budget_contrasts.csv'), index=False)
    lines += ['## Levels (AMR-R in A, COV as fraction)',
              P.pivot_table(index='model', columns=['view', 'metric'], values='value', sort=False).round(4)
              .to_string(), '',
              '## Contrasts (AMR-R in A, COV in %; p_holm over the BUDGET family of 8)',
              C.round(4).to_string(index=False)]
    txt = '\n'.join(lines)
    with open(os.path.join(out, 'budget_matched_summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print(txt)
    return P, C


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--split_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--s4_subset', default=None)
    ap.add_argument('--n_boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ref_subsample', type=int, default=0,
                    help='[round5] > 0: also the budget-matched ref_half (EXP-1 §3.6) with this many MC check draws')
    ap.add_argument('--strata', default=None,
                    help='[round5, post hoc] csv with corrected_smiles, m (python -m flexitors.data.strata)')
    a = ap.parse_args()
    analyse(a.res, a.split_dir, a.out, a.s4_subset, a.n_boot, a.seed, ref_subsample=a.ref_subsample,
            strata=a.strata)

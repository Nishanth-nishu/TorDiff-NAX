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
  EXPLORATORY: S3 - B1 on split-half; per n_clean bin (2-3, 4-10, 11-30, >30; round3/impact.md §3g) means;
    model - COPY on split-half for CR, B1, S3, S4, where COPY is the model-free "copy the other half" baseline from
    make_split_half.py nn.tsv (the other half's GT conformers used as predictions; V-8 seed-vs-model contribution).
CIs: (a) molecule bootstrap on the seed-averaged per-molecule differences (tools/paired_compare.py `boot`, 95 %
percentile, two-sided bootstrap p); (b) hierarchical bootstrap -- training seeds resampled with replacement within each
model, then molecules (score_VALIDITY_r1.md V-4) -- 95 % percentile interval.

Usage (on gnode118, after slurm/r4_leakfree_array.sbatch):
  python tools/analyze_leakfree.py --res $RES --split_dir $QM9_DIR/round4_splithalf \
      --s4_subset $QM9_SEEDS2/S4_test_subset.txt --out $RES/analysis_r4_leakfree
"""
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd

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


# ------------------------------------------------------------------------------------------------ helpers
def per_mol(path):
    """per-molecule recall metrics of one eval.pkl; returns (DataFrame indexed by corrected smiles, info dict)"""
    with open(path, 'rb') as f:
        R = pickle.load(f)
    rows = {}
    n_nan = 0
    for (smi, corr), res in R['results'].items():
        rmin = np.min(res['rmsd'], axis=1)  # NaN rows propagate exactly as in evaluate_confs.py
        d = {'AMR-R': float(np.mean(rmin)), 'n_targets': int(res['n_true']), 'n_gen': int(res['n_model'])}
        for m, t in THRS.items():
            d[m] = float(np.mean(rmin < t))
        n_nan += int(not np.isfinite(d['AMR-R']))
        rows[corr] = d
    info = dict(num_failures=int(R.get('num_failures', 0)), n_results=len(rows), n_nan=n_nan)
    return pd.DataFrame.from_dict(rows, orient='index'), info


def view(d, v):
    """d: {cond: DataFrame} for one run -> per-molecule split / own / leak frame (METRICS columns)"""
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


def load_all(res, families):
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
                d[c], inf = per_mol(p)
                info.append(dict(model=fam, seed=s, cond=c, **inf))
            if len(d) == len(CONDS):
                data[(fam, s)] = d
    return data, pd.DataFrame(info), missing


def analyse(res, split_dir, out, s4_subset=None, n_boot=5000, seed=0, families=('CR', 'B1', 'S3', 'S4')):
    os.makedirs(out, exist_ok=True)
    split = pd.read_csv(os.path.join(split_dir, 'split.tsv'), sep='\t').set_index('corrected_smiles')
    data, info, missing = load_all(res, families)
    lines = ['# Leak-free (split-half) true-geometry oracle test: analysis (ORACLE; tools/analyze_leakfree.py)', '']
    if missing:
        lines += [f'MISSING {p}' for p in missing] + ['']
    fams = [f for f in families if all((f, s) in data for s in SEEDS[f])]
    # universe: finite AMR-R in every file of every complete model
    sets = [set(df.index[np.isfinite(df['AMR-R'])]) for k in data if k[0] in fams for df in data[k].values()]
    universe = sorted(set.intersection(*sets) & set(split.index)) if sets else []
    lines += [f'eligible molecules (split.tsv, n_clean >= 2): {len(split)}',
              f'success intersection over {sum(len(SEEDS[f]) for f in fams)} runs x 4 conditions: {len(universe)}',
              f'models complete: {fams}', '']
    info.to_csv(os.path.join(out, 'failures.csv'), index=False)
    # per-run, per-view per-molecule frames on the universe
    V = {}
    for (fam, s), d in data.items():
        if fam not in fams:
            continue
        for v in ('split', 'own', 'leak'):
            V[(fam, s, v)] = view(d, v).loc[universe]
        for c in CONDS:
            V[(fam, s, c)] = d[c].loc[universe, METRICS]

    def stack(fam, v, metric, mols=None):
        mols = universe if mols is None else mols
        return np.stack([V[(fam, s, v)].loc[mols, metric].values for s in SEEDS[fam]])

    # ---- per-model table
    rows = []
    for fam in fams:
        for v in ['split', 'own', 'leak'] + CONDS:
            for m in METRICS:
                X = stack(fam, v, m)
                per_seed = X.mean(axis=1)
                rows.append(dict(model=fam, view=v, metric=m, n=len(universe), value=float(X.mean()),
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
            for m in METRICS:
                A, Rf = stack(arm, v, m, mols), stack(ref, v, m, mols)
                mean, lo, hi, p = boot(A.mean(0) - Rf.mean(0), rng, n_boot)
                hlo, hhi = hier_ci(A, Rf, rng, n_boot)
                sc = 1.0 if m == 'AMR-R' else 100.0
                crow.append(dict(id=cid + ('@S4subset' if sname != 'all' else ''), family=family, metric=m,
                                 n=len(mols), arm=arm, ref=ref, view=v, arm_value=A.mean() * sc,
                                 ref_value=Rf.mean() * sc, diff=mean * sc, ci_lo=lo * sc, ci_hi=hi * sc, p_boot=p,
                                 hier_lo=hlo * sc, hier_hi=hhi * sc,
                                 note='S4 LIMITATION (trains on pair_ok-safe pairs only)' if 'S4' in (arm, ref) else ''))
    # within-model leak with CI (paired own - split, same model)
    for fam in fams:
        for m in METRICS:
            A = stack(fam, 'leak', m)
            mean, lo, hi, p = boot(A.mean(0), rng, n_boot)
            hlo, hhi = hier_ci(A, None, rng, n_boot)
            sc = 1.0 if m == 'AMR-R' else 100.0
            crow.append(dict(id=f'S.leak.{fam}', family='secondary', metric=m, n=len(universe), arm=fam, ref='-',
                             view='own-split', arm_value=stack(fam, 'own', m).mean() * sc,
                             ref_value=stack(fam, 'split', m).mean() * sc, diff=mean * sc, ci_lo=lo * sc,
                             ci_hi=hi * sc, p_boot=p, hier_lo=hlo * sc, hier_hi=hhi * sc, note=''))
    # model-free COPY baseline (exploratory): other half's GT conformers as predictions, split view only
    nn_path = os.path.join(split_dir, 'nn.tsv')
    if os.path.exists(nn_path) and universe:
        NN = pd.read_csv(nn_path, sep='\t')
        g = NN.groupby(['corrected_smiles', 'half']).nn_rmsd
        cp = {'AMR-R': g.mean()}
        for m, t in THRS.items():
            cp[m] = g.apply(lambda x, t=t: float(np.mean(x.values < t)))
        copy = pd.DataFrame({m: s.groupby(level=0).mean() for m, s in cp.items()}).reindex(universe)
        for m in METRICS:
            sc = 1.0 if m == 'AMR-R' else 100.0
            models = pd.concat([models, pd.DataFrame([dict(model='COPY', view='split', metric=m, n=len(universe),
                                                           value=float(copy[m].mean()), seed_sd=np.nan,
                                                           per_seed='model-free')])], ignore_index=True)
            for fam in fams:
                A = stack(fam, 'split', m)
                mean, lo, hi, p = boot(A.mean(0) - copy[m].values, rng, n_boot)
                crow.append(dict(id=f'E.{fam}-COPY.split', family='exploratory', metric=m, n=len(universe), arm=fam,
                                 ref='COPY', view='split', arm_value=A.mean() * sc, ref_value=copy[m].mean() * sc,
                                 diff=mean * sc, ci_lo=lo * sc, ci_hi=hi * sc, p_boot=p, hier_lo=np.nan,
                                 hier_hi=np.nan, note='COPY = other-half GT conformers as predictions (no model)'))
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
    S = pd.DataFrame(srow)
    S.to_csv(os.path.join(out, 'strata.csv'), index=False)

    # ---- per-molecule dump (seed-averaged; for later re-analysis without the cluster)
    pm = []
    for fam in fams:
        for v in ['split', 'own'] + CONDS:
            X = {m: stack(fam, v, m).mean(0) for m in METRICS}
            pm.append(pd.DataFrame(dict(model=fam, view=v, corrected_smiles=universe, **X)))
    if pm:
        pd.concat(pm).to_csv(os.path.join(out, 'per_mol.csv.gz'), index=False, compression='gzip')

    # ---- text report
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    piv = models[models.view.isin(['split', 'own', 'leak'])].pivot_table(
        index='model', columns=['view', 'metric'], values='value', sort=False)
    lines += ['## Per model (seed-averaged; AMR-R in A, COV-R as fraction)', piv.round(4).to_string(), '',
              '## Per model and direction', models[models.view.isin(CONDS)].pivot_table(
                  index='model', columns=['view', 'metric'], values='value', sort=False).round(4).to_string(), '',
              '## Per-seed values', models[models.view.isin(['split', 'own', 'leak'])][
                  ['model', 'view', 'metric', 'value', 'seed_sd', 'per_seed']].round(4).to_string(index=False), '',
              '## Contrasts (AMR-R in A, COV in %; diff = arm - ref; p_holm over the PRIMARY family only)',
              C.round(4).to_string(index=False), '',
              '## n_clean strata (exploratory)', S.round(4).to_string(index=False), '',
              '## Files', info.to_string(index=False)]
    txt = '\n'.join(lines)
    with open(os.path.join(out, 'leakfree_summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print(txt)
    return models, C, S


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--split_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--s4_subset', default=None)
    ap.add_argument('--n_boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    analyse(a.res, a.split_dir, a.out, a.s4_subset, a.n_boot, a.seed)

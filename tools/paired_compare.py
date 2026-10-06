"""
Paired, molecule-level comparison of ablation arms against a reference arm (ablation plan section 1, "Uncertainty";
review/cross_validation.md S1-S3).

Every arm is one or more eval.pkl files written by the patched evaluate_confs.py (--out_results). Several files for
one arm (sampling seeds, or training seeds) are averaged PER MOLECULE first, then molecules are bootstrapped
(paired: the same resampled molecules for the arm and the reference). Reported per arm and metric:
  mean difference (arm - ref), 95% percentile CI, two-sided bootstrap p, Holm-adjusted p over all (arm, primary
  metric) pairs, and -- if the reference has >= 2 replicate files -- the replicate-to-replicate SD of the reference
  aggregate, so that a difference can be compared with seed noise.

Conventions (identical to evaluate_confs.py): COV counts a model failure (molecule absent from `results`) as 0 and
is averaged over the union of evaluated molecules; AMR is compared on molecules with a finite value in BOTH arms
(the n is printed; if arms fail on different molecules -- e.g. GT-L seeds never fail to embed -- the COV and AMR
populations differ, which is why both n are shown).

Usage:
  python tools/paired_compare.py --ref R0=RES/qm9_baseline_s0/R0_base_seed0/eval.pkl,RES/.../R1_base_seed1/eval.pkl \
      --arm A1=RES/qm9_baseline_s0/A1_gtL_model/eval.pkl --arm A2=... --thresholds 0.1 0.25 0.5 --out cmp.md
Primary endpoints (pre-registered, Holm-corrected): MAT-R (= AMR-R) mean and COV-R at 0.5 A (TD QM9 protocol).
The 0.05-0.25 A sweep, medians, precision metrics and strata are secondary / exploratory (not Holm-corrected).
"""
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--ref', required=True, help='NAME=path[,path...]')
parser.add_argument('--arm', action='append', required=True, help='NAME=path[,path...] (repeatable)')
parser.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.05, 0.1, 0.25],
                    help='0.5 = TD QM9 protocol (primary); 0.05-0.25 are secondary/exploratory')
parser.add_argument('--primary', nargs='+', default=['MAT-R', 'COV-R@0.5'])
parser.add_argument('--n_boot', type=int, default=5000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out', default=None)
args = parser.parse_args()


def per_mol(path, thrs):
    with open(path, 'rb') as f:
        R = pickle.load(f)
    rows = {}
    for (smi, corr), res in R['results'].items():
        M = res['rmsd']
        d = {}
        rmin, pmin = np.min(M, axis=1), np.min(M, axis=0)  # NaN rows propagate exactly as in evaluate_confs.py
        d['MAT-R'] = float(np.mean(rmin))
        d['MAT-P'] = float(np.mean(pmin))
        for t in thrs:
            d[f'COV-R@{t}'] = float(np.mean(rmin < t))
            d[f'COV-P@{t}'] = float(np.mean(pmin < t))
        rows[corr] = d
    return pd.DataFrame.from_dict(rows, orient='index')


def load_arm(spec, thrs):
    name, paths = spec.split('=', 1)
    dfs = [per_mol(p, thrs) for p in paths.split(',') if p]
    return name, dfs


def combine(dfs, universe):
    """average over replicate files per molecule; a failure (absent) counts COV 0 and AMR NaN in that replicate"""
    out = []
    for d in dfs:
        d = d.reindex(universe)
        cov = [c for c in d.columns if c.startswith('COV')]
        d[cov] = d[cov].fillna(0.0)
        out.append(d)
    return sum(out) / len(out) if len(out) > 1 else out[0]


def boot(diff, rng, n_boot):
    diff = np.asarray(diff, float)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = diff[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))
    return diff.mean(), lo, hi, p


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


rng = np.random.default_rng(args.seed)
ref_name, ref_dfs = load_arm(args.ref, args.thresholds)
arms = [load_arm(a, args.thresholds) for a in args.arm]
universe = sorted(set().union(*[set(d.index) for d in ref_dfs], *[set(d.index) for _, ds in arms for d in ds]))
ref = combine(ref_dfs, universe)
metrics = list(ref.columns)

rep_sd = {}
if len(ref_dfs) > 1:
    for m in metrics:
        vals = []
        for d in ref_dfs:
            d = combine([d], universe)
            vals.append(np.nanmean(d[m]) if m.startswith('MAT') else d[m].mean())
        rep_sd[m] = float(np.std(vals, ddof=1))

rows = []
for name, dfs in arms:
    a = combine(dfs, universe)
    for m in metrics:
        pair = pd.concat([a[m], ref[m]], axis=1, keys=['a', 'r']).dropna()
        if m.startswith('MAT'):
            pair = pair[np.isfinite(pair.a) & np.isfinite(pair.r)]
        if not len(pair):
            continue
        mean, lo, hi, p = boot(pair.a - pair.r, rng, args.n_boot)
        scale = 1.0 if m.startswith('MAT') else 100.0
        rows.append(dict(arm=name, metric=m, n=len(pair), ref=pair.r.mean() * scale, arm_value=pair.a.mean() * scale,
                         diff=mean * scale, ci_lo=lo * scale, ci_hi=hi * scale, p_boot=p,
                         ref_replicate_sd=rep_sd.get(m, np.nan) * scale, primary=m in args.primary,
                         n_ref_files=len(ref_dfs), n_arm_files=len(dfs)))
res = pd.DataFrame(rows)
prim = res.primary.values
res['p_holm'] = np.nan
if prim.any():
    res.loc[prim, 'p_holm'] = holm(res.loc[prim, 'p_boot'].values)
pd.set_option('display.width', 250)
txt = res.round(4).to_string(index=False)
print(f'reference: {ref_name} ({len(ref_dfs)} file(s)); molecules in universe: {len(universe)}')
print('COV in %, MAT in A; diff = arm - ref; p_holm only over primary endpoints')
print(txt)
if args.out:
    try:
        md = res.round(4).to_markdown(index=False)
    except ImportError:
        md = txt
    with open(args.out, 'w') as f:
        f.write(f'reference: {ref_name}\n\n{md}\n')
    print('wrote', args.out)

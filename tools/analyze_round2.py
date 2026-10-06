"""
Round-2 analysis driver (FIXES X10). Every comparison, Holm family and decision rule is declared IN THIS FILE before any
round-2 result exists (commit time = pre-registration). Run after the round-2 arrays:

  python tools/analyze_round2.py --res $RES --seeds $QM9_SEEDS2 --test_csv $QM9_TEST_CSV --out $RES/analysis_r2

Endpoints (SHORTLIST "Endpoints", unchanged):
  primary   : MAT-R (= AMR-R) mean and COV-R@0.5, TD protocol (union of molecules, model failure = COV 0), paired
              molecule bootstrap, Holm over the PRIMARY family (all primary contrasts x 2 metrics)
  secondary : COV-R@0.1 on the success intersection (molecules evaluated in every file of arm and ref), own Holm family
  also      : failure counts per file; COV-R@0.5 on the intersection (sensitivity); COV-R@0.05 (exploratory, no
              Holm); AMR-P / COV-P (descriptive)
  every row : training-seed SD of the arm and ref aggregates next to the bootstrap CI (research_check_B I7)
  S2 margin : non-inferiority of B6 to CTRL_base on RDKit L, margin 0.004 A AMR-R, judged with a HIERARCHICAL bootstrap
              (training seeds resampled, then molecules), i.e. with seed variance included (research_check_B I7)
  S4        : every S4 row is repeated on S4's own test-molecule subset (molecules with >= 1 pair_ok-safe matched pair,
              $QM9_SEEDS2/S4_test_subset.txt), with the controls re-scored on that subset (user ruling 2026-10-06).
              LIMITATION, printed next to every S4 result: re-scoring removes the test-set difference but not the
              training-set one (S4 trains on ~4 % fewer molecules than its controls; DECISION.md 2026-10-07).
  ORACLE    : S1 / S1.5 / lambda-series rows use test-set GT L and are analysis, never model selection.
  S1 / S4 dose-response is reported against the MEASURED L error (l_error.csv: bond / angle / ring-subset RMSD to GT),
  not against nominal lambda or sigma (research checks A and B).
"""
import os
import pickle
import re
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd

ap = ArgumentParser()
ap.add_argument('--res', required=True)
ap.add_argument('--seeds', required=True, help='$QM9_SEEDS2 (l_error.csv, S4_test_subset.txt)')
ap.add_argument('--test_csv', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--n_boot', type=int, default=5000)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

S = [0, 1, 2]
RUN = dict(CB='qm9_CTRL_base_100ep_e100_s{}', CR='qm9_CTRL_rematch_100ep_e100_s{}', B1='qm9_B1_train_gtL_e100_s{}',
           B6='qm9_B6_jit0.04pa_e100_s{}', B6b='qm9_B6_jit0.02pa_e100_s{}', S3='qm9_S3_mixL_p0.5_e100_s{}',
           B1cap='qm9_B1cap_gtL_e100_s{}', S4='qm9_S4_lamcond_e100_s{}', B3='qm9_B3_match_mmff_e100_s{}')
SEEDS = dict(CB=S, CR=S, B1=S, B6=S, B6b=[0], S3=[0, 1], B1cap=[0], S4=S, B3=S)
RD, GTC = 'steps20_seed0', 'steps20_seed0_gtLcycle'
S1CORE = ['S1_etkdg2L', 'S1_etkdg2L_mmff', 'S1_noise0.02pa_cyc_ORACLE', 'S1_noise0.04pa_cyc_ORACLE',
          'S1_lam0.00_cyc_ORACLE', 'S1_lam0.25_cyc_ORACLE', 'S1_lam0.50_cyc_ORACLE', 'S1_lam0.75_cyc_ORACLE']
S1PANEL = ['S1_etkdg2L', 'S1_etkdg2L_mmff', 'S1_noise0.02pa_cyc_ORACLE', 'S1_noise0.04pa_cyc_ORACLE',
           'S1_lam0.00_cyc_ORACLE', 'S1_lam0.50_cyc_ORACLE']
A5 = ['S1_A5ring_cyc_ORACLE', 'S1_A5acyc_cyc_ORACLE']

# ------------------------------------------------------------------------------------------------ pre-declared contrasts
# (id, family, arm, arm_tag, ref, ref_tag, subset, note). family: primary | explore. subset: None | 'S4'
C = [
    # S2 (B6): target CTRL_base on RDKit L (non-inferiority, margin 0.004 A), oracle superiority on GT L; control B1
    ('S2.rdkit_vs_CB', 'primary', 'B6', RD, 'CB', RD, None, 'non-inferiority margin 0.004 A (hierarchical bootstrap)'),
    ('S2.gtLc_vs_CB_ORACLE', 'primary', 'B6', GTC, 'CB', GTC, None, 'oracle superiority'),
    ('S2.rdkit_vs_B1', 'primary', 'B6', RD, 'B1', RD, None, 'single-factor control'),
    # S3 (mixed L): controls CTRL_rematch (RDKit end, F4) and B1 / B1cap (GT end)
    ('S3.rdkit_vs_CR', 'primary', 'S3', RD, 'CR', RD, None, ''),
    ('S3.gtLc_vs_CR_ORACLE', 'primary', 'S3', GTC, 'CR', GTC, None, ''),
    ('S3.gtLc_vs_B1cap_ORACLE', 'primary', 'S3', GTC, 'B1cap', GTC, None, 'cap-matched GT end'),
    # S4 (lambda-conditioned): lambda 0 on RDKit vs CR; lambda 1 on GT L vs CR / B1cap / B1; all also on the S4 subset
    ('S4.rdkit_l0_vs_CR', 'primary', 'S4', RD, 'CR', RD, None, 'S4 LIMITATION'),
    ('S4.gtLc_l1_vs_CR_ORACLE', 'primary', 'S4', GTC, 'CR', GTC, None, 'S4 LIMITATION'),
    ('S4.gtLc_l1_vs_B1cap_ORACLE', 'primary', 'S4', GTC, 'B1cap', GTC, None, 'S4 LIMITATION'),
    ('S4.rdkit_l0_vs_CR@S4subset', 'primary', 'S4', RD, 'CR', RD, 'S4', 'S4 LIMITATION; controls re-scored on subset'),
    ('S4.gtLc_l1_vs_CR_ORACLE@S4subset', 'primary', 'S4', GTC, 'CR', GTC, 'S4', 'S4 LIMITATION; re-scored'),
    ('S4.gtLc_l1_vs_B1_ORACLE@S4subset', 'primary', 'S4', GTC, 'B1', GTC, 'S4', 'S4 LIMITATION; re-scored'),
    # S5 (B3): CTRL_rematch with and without --pre_mmff
    ('S5.premmff_vs_CRpremmff', 'primary', 'B3', RD, 'CR', 'S5ctrl_pre_mmff', None, 'B3 in-job eval uses --pre_mmff'),
    ('S5.premmff_vs_CR', 'primary', 'B3', RD, 'CR', RD, None, ''),
    ('S5.nommff_vs_CR', 'explore', 'B3', 'S5_nommff', 'CR', RD, None, 'train-only cell'),
    # cap control and exploratory arms
    ('B1cap.gtLc_vs_B1_ORACLE', 'explore', 'B1cap', GTC, 'B1', GTC, None, 'cap / population confound of B1'),
    ('B6b.rdkit_vs_CB', 'explore', 'B6b', RD, 'CB', RD, None, 'sigma 0.02, 1 seed'),
]
# S1 / S1.5 dose-response (ORACLE, analysis only): B1 - CR per condition; S4 lambda_in = lambda_build vs CR on the
# lambda series (S4 subset); trained arms vs CR on their panel conditions
for t in S1CORE + A5:
    C.append((f'S1.B1_vs_CR.{t}', 'explore', 'B1', t, 'CR', t, None, 'ORACLE dose-response'))
for t in S1PANEL:
    for arm in ('B6', 'S3'):
        C.append((f'S1.{arm}_vs_CR.{t}', 'explore', arm, t, 'CR', t, None, 'ORACLE dose-response'))
for lam in ('0.00', '0.50'):
    C.append((f'S4.lam{lam}_vs_CR@S4subset', 'explore', 'S4', f'S4_lam{lam}_l{lam}_cyc_ORACLE', 'CR',
              f'S1_lam{lam}_cyc_ORACLE', 'S4', 'S4 LIMITATION; ORACLE'))
THRS = [0.5, 0.1, 0.05]
PRIMARY_METRICS = ['MAT-R', 'COV-R@0.5']


# ------------------------------------------------------------------------------------------------ helpers
def per_mol(path):
    with open(path, 'rb') as f:
        R = pickle.load(f)
    rows = {}
    for (smi, corr), res in R['results'].items():
        M = res['rmsd']
        rmin, pmin = np.min(M, axis=1), np.min(M, axis=0)
        d = {'MAT-R': float(np.mean(rmin)), 'MAT-P': float(np.mean(pmin))}
        for t in THRS:
            d[f'COV-R@{t}'] = float(np.mean(rmin < t))
            d[f'COV-P@{t}'] = float(np.mean(pmin < t))
        rows[corr] = d
    return pd.DataFrame.from_dict(rows, orient='index'), R.get('num_failures')


def load(name, tag):
    out, fails, missing = [], [], []
    for s in SEEDS[name]:
        p = os.path.join(args.res, RUN[name].format(s), tag, 'eval.pkl')
        if os.path.exists(p):
            d, nf = per_mol(p)
            out.append(d)
            fails.append(nf)
        else:
            missing.append(p)
    return out, fails, missing


def combine(dfs, universe):
    o = []
    for d in dfs:
        d = d.reindex(universe)
        cov = [c for c in d.columns if c.startswith('COV')]
        d[cov] = d[cov].fillna(0.0)  # TD protocol: failure = COV 0
        o.append(d)
    return o


def boot(diff, rng):
    diff = np.asarray(diff, float)
    bs = diff[rng.integers(0, len(diff), size=(args.n_boot, len(diff)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return diff.mean(), lo, hi, min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))


def hier_upper(A, Rf, metric, rng, q=95):
    """one-sided upper q% bound of mean(arm) - mean(ref): resample training seeds, then molecules"""
    mols = A[0].index
    vals = []
    for _ in range(2000):
        a = [A[i] for i in rng.integers(0, len(A), len(A))]
        r = [Rf[i] for i in rng.integers(0, len(Rf), len(Rf))]
        idx = mols[rng.integers(0, len(mols), len(mols))]
        am = np.nanmean(np.mean([x.loc[idx, metric].values for x in a], axis=0))
        rm = np.nanmean(np.mean([x.loc[idx, metric].values for x in r], axis=0))
        vals.append(am - rm)
    return float(np.percentile(vals, q))


def holm(p):
    p = np.asarray(p, float)
    order, m, adj, run = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------------------------------------ main
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed)
df = pd.read_csv(args.test_csv)
raw2corr = dict(zip(df.iloc[:, 0], df.iloc[:, 2]))
sub_path = os.path.join(args.seeds, 'S4_test_subset.txt')
S4SUB = {raw2corr.get(x.strip()) for x in open(sub_path) if x.strip()} if os.path.exists(sub_path) else None
rows, fail_rows = [], []
for cid, fam, arm, atag, ref, rtag, subset, note in C:
    A, fa, ma = load(arm, atag)
    Rf, fr, mr = load(ref, rtag)
    for nm, tg, fl, ms in ((arm, atag, fa, ma), (ref, rtag, fr, mr)):
        fail_rows.append(dict(contrast=cid, model=nm, tag=tg, num_failures=fl, n_missing_files=len(ms)))
    if not A or not Rf:
        rows.append(dict(contrast=cid, family=fam, status='missing', missing=';'.join(ma + mr)[:300], note=note))
        continue
    for universe_kind in ('union', 'intersection'):
        sets = [set(d.index) for d in A + Rf]
        U = sorted(set.union(*sets)) if universe_kind == 'union' else sorted(set.intersection(*sets))
        if subset == 'S4':
            if S4SUB is None:
                rows.append(dict(contrast=cid, family=fam, status='no S4 subset file', note=note))
                break
            U = [u for u in U if u in S4SUB]
        Ac, Rc = combine(A, U), combine(Rf, U)
        a, r = sum(Ac) / len(Ac), sum(Rc) / len(Rc)
        for m in [c for c in a.columns]:
            pair = pd.concat([a[m], r[m]], axis=1, keys=['a', 'r']).dropna()
            if m.startswith('MAT'):
                pair = pair[np.isfinite(pair.a) & np.isfinite(pair.r)]
            if not len(pair):
                continue
            mean, lo, hi, p = boot(pair.a - pair.r, rng)
            agg = lambda X: [np.nanmean(x.loc[pair.index, m]) for x in X]
            sa, sr = agg(Ac), agg(Rc)
            role = 'descriptive'
            if fam == 'primary' and universe_kind == 'union' and m in PRIMARY_METRICS:
                role = 'primary'
            elif fam == 'primary' and universe_kind == 'intersection' and m == 'COV-R@0.1':
                role = 'secondary'
            elif m in ('COV-R@0.05',):
                role = 'exploratory'
            sc = 1.0 if m.startswith('MAT') else 100.0
            row = dict(contrast=cid, family=fam, universe=universe_kind, metric=m, role=role, n=len(pair),
                       arm=arm, arm_tag=atag, ref=ref, ref_tag=rtag, subset=subset or '',
                       arm_value=pair.a.mean() * sc, ref_value=pair.r.mean() * sc, diff=mean * sc, ci_lo=lo * sc,
                       ci_hi=hi * sc, p_boot=p,
                       arm_seed_sd=(np.std(sa, ddof=1) * sc if len(sa) > 1 else np.nan), n_arm_seeds=len(sa),
                       ref_seed_sd=(np.std(sr, ddof=1) * sc if len(sr) > 1 else np.nan), n_ref_seeds=len(sr),
                       note=note, status='ok')
            if cid == 'S2.rdkit_vs_CB' and m == 'MAT-R' and universe_kind == 'union':
                Ai = [x.loc[pair.index] for x in Ac]
                Ri = [x.loc[pair.index] for x in Rc]
                ub = hier_upper(Ai, Ri, 'MAT-R', rng)
                row['noninf_upper95_hier'] = ub
                row['noninferior_0.004'] = bool(ub < 0.004)
            rows.append(row)
R = pd.DataFrame(rows)
for role in ('primary', 'secondary'):
    mask = (R['role'] == role) & R['p_boot'].notna() if 'role' in R and 'p_boot' in R else None
    if mask is not None and mask.any():
        R.loc[mask, f'p_holm_{role}'] = holm(R.loc[mask, 'p_boot'].values)
R.to_csv(os.path.join(args.out, 'contrasts.csv'), index=False)
pd.DataFrame(fail_rows).drop_duplicates(['model', 'tag']).to_csv(os.path.join(args.out, 'failures.csv'), index=False)

# ------------------------------------------------------------------------ dose-response against the MEASURED L error
le_path = os.path.join(args.seeds, 'l_error.csv')
if os.path.exists(le_path) and len(R) and 'metric' in R:
    E = pd.read_csv(le_path)
    cols = ['bond_rmsd_all_any', 'angle_rmsd_all_any', 'bond_rmsd_heavy_ring', 'angle_rmsd_heavy_ring',
            'angle_rmsd_heavy_acyc', 'ring_dihedral_rmsd']
    Lm = E.groupby(['cond', 'smiles'])[cols].mean().groupby('cond').mean()  # per-molecule mean, then over molecules
    Lm.index = [re.sub(r'^L_', 'S1_', i) for i in Lm.index]
    D = R[(R.metric == 'MAT-R') & (R.universe == 'union') & R.contrast.str.startswith('S1.')].copy()
    D['condition'] = D['arm_tag']
    D = D.merge(Lm, left_on='condition', right_index=True, how='left')
    D.to_csv(os.path.join(args.out, 'dose_response_vs_measured_L_error.csv'), index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for k, xcol in enumerate(('angle_rmsd_all_any', 'bond_rmsd_all_any')):
            for arm, g in D.groupby('arm'):
                g = g.dropna(subset=[xcol]).sort_values(xcol)
                ax[k].errorbar(g[xcol], g['diff'], yerr=[g['diff'] - g['ci_lo'], g['ci_hi'] - g['diff']], fmt='o-',
                               label=f'{arm} - CR', capsize=2)
            ax[k].axhline(0, color='grey', lw=0.8)
            ax[k].set_xlabel(f'measured {xcol} vs GT (test seeds)')
            ax[k].set_ylabel('AMR-R difference to CTRL_rematch (A)')
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'dose_response_vs_measured_L_error.png'), dpi=120)
    except Exception as e:  # plotting is optional
        print('plot skipped:', e)

# ------------------------------------------------------------------------ MMFF failure counts (FIXES X11)
mm = []
for name in ('CR', 'B3'):
    for s in SEEDS[name]:
        for tag in ('S5ctrl_pre_mmff', RD):
            p = os.path.join(args.res, RUN[name].format(s), tag, 'generate.log')
            if os.path.exists(p):
                for line in open(p, errors='replace'):
                    if line.startswith('MMFF_PRE'):
                        mm.append(dict(model=RUN[name].format(s), tag=tag, **dict(kv.split('=') for kv in
                                                                                   line.split()[1:])))
pd.DataFrame(mm).to_csv(os.path.join(args.out, 'mmff_pre_failures.csv'), index=False)

with open(os.path.join(args.out, 'README.md'), 'w') as f:
    f.write(__doc__ + '\n\nFiles: contrasts.csv (all rows; primary/secondary Holm columns), failures.csv, '
            'dose_response_vs_measured_L_error.{csv,png}, mmff_pre_failures.csv\n')
print(f'{len(R)} rows -> {args.out}/contrasts.csv')
if 'role' in R:
    show = R[R.role.isin(['primary', 'secondary'])]
    if len(show):
        print(show[['contrast', 'universe', 'metric', 'n', 'diff', 'ci_lo', 'ci_hi', 'arm_seed_sd', 'ref_seed_sd']
                   + [c for c in show.columns if c.startswith('p_holm')]].round(4).to_string(index=False))

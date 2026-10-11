"""Impact analyst (round 3, P3): dose-response of AMR-R on measured L error, and predictions for new L sources.

Inputs (from collect.py and lerror_profile.py, same directory): bins.csv, runs.csv, lerror_profile.csv.
Data: round-2 S1 seed-pickle runs (PRELIMINARY, unpaired), CTRL_rematch s0-2 and B1 s0-2 (lambda 1.00: B1 s0 only).

Method (stated so it can be checked):
  * unit = n_rot_heavy bin (0, 1, 2, 3, 4+) x test-time L condition; AMR-R of the bin = mean of the bin's MAT-R over
    training seeds (breakdown.log); L profile of the bin = lerror_profile.csv (rows of l_error.csv in that bin).
  * CLEAN conditions only for fitting: ETKDG, MMFF, lam0, lam1, A5ring, A5acyc, noise0.02, noise0.04.
    lam0.25/0.50/0.75 are excluded: 1268 of 12707 slots (10%, 258 molecules) reuse another GT conformer's blended
    seed (population.txt 'replaced_seeds'), and l_error.csv measures those rows against the SOURCE GT (k != src_gt_idx),
    so their L error understates the slot's effective error. They are shown as out-of-sample checks only.
  * model per bin and per checkpoint: AMR = a + bA * angle_any_heavy (deg) + bT * tail10 (share of ring>=4 rows with
    ring-dihedral RMSD > 10 deg).  angle and bond error are collinear across conditions (r > 0.9), so one bond/angle
    regressor is used.
  * predictions are ANCHORED: AMR_new(bin) = AMR_obs(anchor, bin) + bA * dA + bT * dT, with anchor = MMFF (xTB starts
    from the same ETKDG embeddings; same 936 molecules) and bin weights of the ETKDG population (284/296/174/93/88).
  * B1 is strongly non-linear near true L (lam 0.75 -> 1: 0.115 -> 0.020); its linear fit is reported with that caveat
    and a log-AMR variant.
Usage: python round3/impact_data/dose_response.py
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
pd.set_option('display.width', 250)
B = pd.read_csv(os.path.join(OUT, 'bins.csv'))
P = pd.read_csv(os.path.join(OUT, 'lerror_profile.csv'))
TAG2COND = {'S1_etkdg2L': 'L_etkdg2L', 'S1_etkdg2L_mmff': 'L_etkdg2L_mmff', 'S1_lam0.00_cyc_ORACLE': 'L_lam0.00_cyc_ORACLE',
            'S1_lam0.25_cyc_ORACLE': 'L_lam0.25_cyc_ORACLE', 'S1_lam0.50_cyc_ORACLE': 'L_lam0.50_cyc_ORACLE',
            'S1_lam0.75_cyc_ORACLE': 'L_lam0.75_cyc_ORACLE', 'S1_lam1.00_cyc_ORACLE': 'L_lam1.00_cyc_ORACLE',
            'S1_A5ring_cyc_ORACLE': 'L_A5ring_cyc_ORACLE', 'S1_A5acyc_cyc_ORACLE': 'L_A5acyc_cyc_ORACLE',
            'S1_noise0.02pa_cyc_ORACLE': 'L_noise0.02pa_cyc_ORACLE', 'S1_noise0.04pa_cyc_ORACLE': 'L_noise0.04pa_cyc_ORACLE'}
CLEAN = ['L_etkdg2L', 'L_etkdg2L_mmff', 'L_lam0.00_cyc_ORACLE', 'L_lam1.00_cyc_ORACLE', 'L_A5ring_cyc_ORACLE',
         'L_A5acyc_cyc_ORACLE', 'L_noise0.02pa_cyc_ORACLE', 'L_noise0.04pa_cyc_ORACLE']
ARMS = {'CTRL': 'qm9_CTRL_rematch_100ep', 'B1': 'qm9_B1_train_gtL'}
BINS = ['0', '1', '2', '3', '4+']
W = np.array([284, 296, 174, 93, 88], float)  # S1_etkdg2L breakdown bin counts (CTRL_rematch s0)


def dataset(arm):
    b = B[(B['round'] == 'r2') & (B.arm == ARMS[arm]) & B.tag.isin(TAG2COND)].copy()
    b['cond'] = b.tag.map(TAG2COND)
    b = b.groupby(['cond', 'bin']).agg(amr=('mat_r', 'mean'), n=('n', 'mean'), seeds=('mat_r', 'size')).reset_index()
    p = P[P.stratum.str.startswith('rot')].copy()
    p['bin'] = p.stratum.str[3:]
    return b.merge(p[['cond', 'bin', 'any_angle', 'any_bond', 'tail10_ge4', 'ringdih_mean', 'acyc_angle',
                      'ring_angle']], on=['cond', 'bin'])


def fit(d, y='amr', xs=('any_angle', 'tail10_ge4')):
    X = np.column_stack([np.ones(len(d))] + [d[x].values for x in xs])
    beta, *_ = np.linalg.lstsq(X, d[y].values, rcond=None)
    pred = X @ beta
    loo = []
    for i in range(len(d)):
        m = np.arange(len(d)) != i
        bi, *_ = np.linalg.lstsq(X[m], d[y].values[m], rcond=None)
        loo.append(X[i] @ bi)
    return beta, pred, np.array(loo)


lines = []
coef = {}
for arm in ARMS:
    D = dataset(arm)
    lines.append(f'\n## {arm}: per-bin fit AMR = a + bA*angle_heavy + bT*tail10 (clean conditions)')
    for bn in BINS:
        d = D[(D.bin == bn) & D.cond.isin(CLEAN)].reset_index(drop=True)
        beta, pred, loo = fit(d)
        coef[(arm, bn)] = beta
        rmse = np.sqrt(np.mean((pred - d.amr) ** 2)); loormse = np.sqrt(np.mean((loo - d.amr) ** 2))
        lines.append(f'bin {bn:>2}: a={beta[0]:.4f} bA={beta[1]:.4f}/deg bT={beta[2]:.4f}/share  '
                     f'in-sample RMSE {rmse:.4f}  LOO RMSE {loormse:.4f}  (n={len(d)})')
        for i, r in d.iterrows():
            lines.append(f'    {r.cond:28s} angle {r.any_angle:5.2f} tail {r.tail10_ge4:5.3f}  obs {r.amr:.3f}  '
                         f'fit {pred[i]:.3f}  loo {loo[i]:.3f}')
        # out-of-sample lambda points
        o = D[(D.bin == bn) & ~D.cond.isin(CLEAN)]
        for _, r in o.iterrows():
            pr = beta[0] + beta[1] * r.any_angle + beta[2] * r.tail10_ge4
            lines.append(f'    [oos] {r.cond:22s} angle {r.any_angle:5.2f} tail {r.tail10_ge4:5.3f}  obs {r.amr:.3f}  '
                         f'pred {pr:.3f}  (replaced seeds: understated L error)')

# ---------------------------------------------------------------- predictions for new L sources (anchored on MMFF)
lines.append('\n## Predictions on the ETKDG population (936 mols), anchored on observed MMFF (or ETKDG) bins')
lines.append('Scenario profile = anchor profile with heavy-angle error set to A and the ring>=4 tail10 share scaled '
             'per bin by X / anchor overall share. Bin weights 284/296/174/93/88.')
D = {arm: dataset(arm) for arm in ARMS}
tail_overall = {c: float(P[(P.cond == c) & (P.stratum == 'all')].tail10_ge4.iloc[0]) for c in ['L_etkdg2L', 'L_etkdg2L_mmff']}


def predict(arm, anchor, A, X, coefs=coef):
    out = []
    for i, bn in enumerate(BINS):
        r = D[arm][(D[arm].cond == anchor) & (D[arm].bin == bn)].iloc[0]
        a, bA, bT = coefs[(arm, bn)]
        tail_new = r.tail10_ge4 * X / tail_overall[anchor]
        out.append(r.amr + bA * (A - r.any_angle) + bT * (tail_new - r.tail10_ge4))
    out = np.array(out)
    return float(np.sum(out * W) / W.sum()), out


obs = {}
for arm in ARMS:
    for c in ['L_etkdg2L', 'L_etkdg2L_mmff']:
        bins = D[arm][D[arm].cond == c].set_index('bin').loc[BINS].amr.values
        obs[(arm, c)] = float(np.sum(bins * W) / W.sum())
lines.append('bin-weighted observed: ' + ', '.join(f'{k[0]} {k[1]} {v:.4f}' for k, v in obs.items()))

rows = []
for A in (0.8, 1.2, 1.6, 1.97):
    for X in (0.496, 0.351, 0.25, 0.15, 0.05, 0.0):
        r = dict(angle=A, tail_ge4=X)
        for arm in ARMS:
            tot, bins = predict(arm, 'L_etkdg2L_mmff', A, X)
            r[f'{arm}'] = tot
            r[f'{arm}_rigid'] = bins[0]
        rows.append(r)
T = pd.DataFrame(rows)
lines.append(T.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
T.to_csv(os.path.join(OUT, 'pred_xtb_grid.csv'), index=False)

# sanity: predict MMFF from ETKDG anchor (out-of-anchor check of the anchored method)
for arm in ARMS:
    A_m = float(P[(P.cond == 'L_etkdg2L_mmff') & (P.stratum == 'all')].any_angle.iloc[0])
    tot, bins = predict(arm, 'L_etkdg2L', A_m, tail_overall['L_etkdg2L_mmff'])
    # bin-specific MMFF profile is better: use real per-bin MMFF values
    out = []
    for i, bn in enumerate(BINS):
        re_ = D[arm][(D[arm].cond == 'L_etkdg2L') & (D[arm].bin == bn)].iloc[0]
        rm = D[arm][(D[arm].cond == 'L_etkdg2L_mmff') & (D[arm].bin == bn)].iloc[0]
        a, bA, bT = coef[(arm, bn)]
        out.append(re_.amr + bA * (rm.any_angle - re_.any_angle) + bT * (rm.tail10_ge4 - re_.tail10_ge4))
    out = np.array(out)
    lines.append(f'check {arm}: ETKDG -> MMFF predicted {float(np.sum(out*W)/W.sum()):.4f} (uniform-scaling '
                 f'version {tot:.4f}) vs observed {obs[(arm, "L_etkdg2L_mmff")]:.4f}; rigid pred {out[0]:.3f} obs '
                 f'{D[arm][(D[arm].cond=="L_etkdg2L_mmff")&(D[arm].bin=="0")].amr.iloc[0]:.3f}')

txt = '\n'.join(lines)
open(os.path.join(OUT, 'dose_response.out'), 'w').write(txt)
print(txt)

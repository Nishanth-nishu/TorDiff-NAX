"""Impact analyst (round 3, P3): two-anchor 'ring-tail mixture' model of AMR-R, validated on the non-oracle ETKDG and
MMFF points it is NOT fitted to, then used to predict CTRL / B1 / S3 on new L sources (xTB, templates, ...).

Model (per n_rot_heavy bin b, per checkpoint):
    AMR_b(X_b) = G_b + (H_b - G_b) * X_b / X_H,b
  G_b = AMR with physically consistent, reference-level L and correct ring puckers: in-job 'gtL' = true L of a RANDOM
        GT conformer of the molecule (round 1; CTRL_base and B1, 3 seeds; not cycled, so no conformer-identity leak).
  H_b = AMR on A5acyc (round 2 S1, ORACLE): RDKit-matched rings, GT acyclic L (ring-tail share X_H,b).
  X_b = share of ring>=4 rows with ring-dihedral RMSD > 10 deg in bin b (lerror_profile.csv).
  i.e. 'everything except the wrong-pucker rings is reference-like'. ETKDG / MMFF are NOT reference-like in acyclic L,
  so the model is EXPECTED to fit them only for a checkpoint that ignores acyclic L error (CTRL), and to fail for B1.
Aggregation: bin weights of the ETKDG population (284/296/174/93/88; S1_etkdg2L breakdown, CTRL_rematch s0).
Population caveat: G comes from 996-molecule in-job runs, H from the 955 lambda/A5 family, X from l_error.csv rows;
  bin-wise reweighting to the ETKDG bins removes most of the population difference (not all).
Usage: python round3/impact_data/mixture_model.py
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
B = pd.read_csv(os.path.join(OUT, 'bins.csv'))
P = pd.read_csv(os.path.join(OUT, 'lerror_profile.csv'))
BINS = ['0', '1', '2', '3', '4+']
W = np.array([284, 296, 174, 93, 88], float)


def bins(arm, tag, rnd):
    b = B[(B['round'] == rnd) & (B.arm == arm) & (B.tag == tag)]
    return b.groupby('bin').mat_r.mean().loc[BINS].values


def tail(cond):
    p = P[(P.cond == cond) & P.stratum.str.startswith('rot')].copy()
    p['bin'] = p.stratum.str[3:]
    return p.set_index('bin').loc[BINS].tail10_ge4.values


def wavg(v):
    return float(np.sum(v * W) / W.sum())


lines = []
XH = tail('L_A5acyc_cyc_ORACLE')
XE, XM = tail('L_etkdg2L'), tail('L_etkdg2L_mmff')
lines.append('tail10 (ring>=4 rows) per bin 0/1/2/3/4+:')
for n, x in (('A5acyc', XH), ('ETKDG', XE), ('MMFF', XM)):
    lines.append(f'  {n:7s} ' + ' '.join(f'{v:.3f}' for v in x) + f'   bin-weighted {wavg(x):.3f}')

models = {
    'CTRL': dict(G=bins('qm9_CTRL_base_100ep', 'steps20_seed0_gtL', 'r1'),
                 H=bins('qm9_CTRL_rematch_100ep', 'S1_A5acyc_cyc_ORACLE', 'r2'),
                 E=bins('qm9_CTRL_rematch_100ep', 'S1_etkdg2L', 'r2'),
                 M=bins('qm9_CTRL_rematch_100ep', 'S1_etkdg2L_mmff', 'r2')),
    'B1': dict(G=bins('qm9_B1_train_gtL', 'steps20_seed0_gtL', 'r1'),
               H=bins('qm9_B1_train_gtL', 'S1_A5acyc_cyc_ORACLE', 'r2'),
               E=bins('qm9_B1_train_gtL', 'S1_etkdg2L', 'r2'),
               M=bins('qm9_B1_train_gtL', 'S1_etkdg2L_mmff', 'r2')),
}


def amr(m, X):
    return m['G'] + (m['H'] - m['G']) * X / XH


lines.append('\n## Validation on the non-oracle points (not used in the fit)')
for k, m in models.items():
    pe, pm = amr(m, XE), amr(m, XM)
    lines.append(f'{k}: anchors G={wavg(m["G"]):.4f} H={wavg(m["H"]):.4f} | ETKDG pred {wavg(pe):.4f} obs '
                 f'{wavg(m["E"]):.4f} | MMFF pred {wavg(pm):.4f} obs {wavg(m["M"]):.4f}')
    lines.append(f'   per bin ETKDG pred/obs: ' + ' '.join(f'{a:.3f}/{b:.3f}' for a, b in zip(pe, m['E'])))
    lines.append(f'   per bin MMFF  pred/obs: ' + ' '.join(f'{a:.3f}/{b:.3f}' for a, b in zip(pm, m['M'])))
    slope = (wavg(m['H']) - wavg(m['G'])) / wavg(XH)
    lines.append(f'   population-level line: AMR = {wavg(m["G"]):.4f} + {slope:.4f} * X (X = bin-weighted tail share)')

# ------------------------------------------------------------------ scenarios: new L sources (reference-like acyclic)
lines.append('\n## Scenario grid: reference-like acyclic L (xTB-type), ring tail X (overall ring>=4 basis, the 49.6% '
             'metric); bin tails scaled from MMFF bins by X / 0.351')
x_mmff_overall = 0.3506
rows = []
for X in (0.496, 0.351, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0):
    Xb = XM * X / x_mmff_overall
    r = dict(tail_X=X)
    for k, m in models.items():
        v = amr(m, Xb)
        r[k] = wavg(v)
        r[f'{k}_rigid'] = v[0]
    r['B1_minus_CTRL'] = r['B1'] - r['CTRL']
    rows.append(r)
T = pd.DataFrame(rows)
lines.append(T.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
T.to_csv(os.path.join(OUT, 'pred_mixture_grid.csv'), index=False)

# literature crossing for B1 / CTRL along the line (C-EVAL-01)
lines.append('\n## Tail share X at which each checkpoint reaches literature AMR-R levels (reference-like acyclic L)')
for k, m in models.items():
    g, h = wavg(m['G']), wavg(m['H'])
    s = (h - g) / wavg(XH)
    for lvl, nm in ((0.073, 'ET-Flow published'), (0.083, 'ET-Flow re-run'), (0.090, 'DMT-B'), (0.103, 'MCF-B'),
                    (0.1525, 'CTRL on MMFF today')):
        x = (lvl - g) / s
        lines.append(f'  {k:4s} reaches {lvl:.3f} ({nm}) at X <= {x:.3f}' if x >= 0 else
                     f'  {k:4s} cannot reach {lvl:.3f} ({nm}): floor {g:.4f}')

# S3 bracket from its in-job endpoints
s3_rd = bins('qm9_S3_mixL_p0.5', 'steps20_seed0', 'r2'); s3_cy = bins('qm9_S3_mixL_p0.5', 'steps20_seed0_gtLcycle', 'r2')
c_cy = bins('qm9_CTRL_rematch_100ep', 'steps20_seed0_gtLcycle', 'r2'); b_cy = bins('qm9_B1_train_gtL', 'steps20_seed0_gtLcycle', 'r1')
lines.append('\n## S3 (seed 0, in-job) endpoints vs CTRL / B1, bin-weighted:')
lines.append(f'  RDKit L: S3 {wavg(s3_rd):.4f} | CTRL_rematch {wavg(bins("qm9_CTRL_rematch_100ep","steps20_seed0","r1")):.4f} '
             f'| B1 {wavg(bins("qm9_B1_train_gtL","steps20_seed0","r1")):.4f}')
lines.append(f'  true L cycled: S3 {wavg(s3_cy):.4f} | CTRL_rematch {wavg(c_cy):.4f} | B1 {wavg(b_cy):.4f}  '
             f'-> S3 sits at fraction {(wavg(s3_cy)-wavg(b_cy))/(wavg(c_cy)-wavg(b_cy)):.2f} of the way from B1 to CTRL')
lines.append('  cycle -> random-GT leak (gtL - gtLcycle, bin-weighted): CTRL_base '
             f'{wavg(bins("qm9_CTRL_base_100ep","steps20_seed0_gtL","r1"))-wavg(bins("qm9_CTRL_base_100ep","steps20_seed0_gtLcycle","r1")):+.4f}'
             f', B1 {wavg(models["B1"]["G"])-wavg(b_cy):+.4f}')
txt = '\n'.join(lines)
open(os.path.join(OUT, 'mixture_model.out'), 'w').write(txt)
print(txt)

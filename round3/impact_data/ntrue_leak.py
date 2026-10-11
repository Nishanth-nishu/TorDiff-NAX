"""Impact analyst (round 3, P3 cross-read): how much conformer-identity leak is in anchor G (random-GT-conformer L),
and what the mixture-model predictions become without it.

Grounder (grounding.md 5.1(d), 5.2.1): round-1 `gtL` draws each generated conformer's L with replacement from the
molecule's OWN GT conformers (diffusion/sampling.py:86). For a target conformer of a molecule with n GT conformers and
2n draws, P(no draw is its own L) = P0(n) = (1 - 1/n)^(2n)  (n = 1: P0 = 0, gtL == gtLcycle).

Leak-free 'other-conformer' anchor per n_true bin b (INFERENCE, first-order):
    gtL_b - cyc_b  ~=  P0_b * (A_other_b - A_own_b),   A_own_b ~= cyc_b     =>   A_other_b = cyc_b + (gtL_b - cyc_b) / P0_b
  (assumes targets with >= 1 own-L draw score like the cycled run; ignores the 1-vs-2-copies difference).
  A_other_b = AMR of a target whose seeds carry reference-level L of OTHER conformers of the same molecule only.
  Not estimable for n_true = 1 (P0 = 0).

Data: breakdown.log 'by n_true' tables (bins 1, 2-3, 4-10, 11-30, >30); r1 in-job gtL / gtLcycle for B1 and CTRL_base
(3 seeds each, 996 mols); r2 S1 A5acyc / ETKDG / MMFF for CTRL_rematch and B1 (3 seeds); tails per n_true bin from
l_error.csv (ring>=4 basis, as lerror_profile.py). Bin weights: ETKDG population n_true counts 223/227/274/127/84.
Usage: python round3/impact_data/ntrue_leak.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(OUT, '..', '..'))
NB = ['1', '2-3', '4-10', '11-30', '>30']


def ntrue_table(path):
    txt = open(path).read()
    sec = txt.split('### by n_true')[1].split('###')[0]
    out = {}
    for line in sec.splitlines():
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) == 6 and c[0] in NB:
            out[c[0]] = (int(c[1]), float(c[3]))
    return out


def runs(rnd_dir, arm_glob, tag):
    vals = []
    for p in glob.glob(os.path.join(ROOT, rnd_dir, arm_glob, tag, 'breakdown.log')):
        vals.append(ntrue_table(p))
    n = np.array([[v[b][0] for b in NB] for v in vals], float).mean(0)
    m = np.array([[v[b][1] for b in NB] for v in vals], float).mean(0)
    return n, m, len(vals)


def p0(n):
    return 0.0 if n <= 1 else (1 - 1 / n) ** (2 * n)


def nbin(n):
    return '1' if n == 1 else '2-3' if n <= 3 else '4-10' if n <= 10 else '11-30' if n <= 30 else '>30'


# --- per-molecule n_true from l_error.csv (noise0.02 rows: 2L per molecule, 996 molecules)
df = pd.read_csv(os.path.join(ROOT, 'cluster_sync', 'round2', 'data', 'QM9', 'round2_seeds', 'l_error.csv'))
nz = df[df.cond == 'L_noise0.02pa_cyc_ORACLE'].groupby('smiles').size() // 2
mol = pd.DataFrame({'smiles': nz.index, 'n': nz.values})
mol['bin'] = mol.n.map(nbin)
mol['p0'] = mol.n.map(p0)
P0 = mol.groupby('bin').p0.mean().reindex(NB).values
L = ['## Molecules per n_true bin (l_error.csv, 996) and mean P0 = P(no own-L draw) per bin']
L.append('  ' + '  '.join(f'{b}: n={int((mol.bin == b).sum())} P0={p:.3f}' for b, p in zip(NB, P0)))
L.append(f'  single-conformer share: {(mol.n == 1).mean():.3f}; molecule-mean share of conformers WITH own L in gtL: '
         f'{1 - mol.p0.mean():.3f}')

# --- anchors per n_true bin
R1, R2 = 'cluster_sync/results', 'cluster_sync/round2/results'
D = {}
for k, (rd, arm, tag) in {
    'CTRL_gtL': (R1, 'qm9_CTRL_base_100ep_e100_s*', 'steps20_seed0_gtL'),
    'CTRL_cyc': (R1, 'qm9_CTRL_base_100ep_e100_s*', 'steps20_seed0_gtLcycle'),
    'B1_gtL': (R1, 'qm9_B1_train_gtL_e100_s*', 'steps20_seed0_gtL'),
    'B1_cyc': (R1, 'qm9_B1_train_gtL_e100_s*', 'steps20_seed0_gtLcycle'),
    'CTRL_H': (R2, 'qm9_CTRL_rematch_100ep_e100_s*', 'S1_A5acyc_cyc_ORACLE'),
    'B1_H': (R2, 'qm9_B1_train_gtL_e100_s*', 'S1_A5acyc_cyc_ORACLE'),
    'CTRL_E': (R2, 'qm9_CTRL_rematch_100ep_e100_s*', 'S1_etkdg2L'),
    'CTRL_M': (R2, 'qm9_CTRL_rematch_100ep_e100_s*', 'S1_etkdg2L_mmff'),
    'B1_E': (R2, 'qm9_B1_train_gtL_e100_s*', 'S1_etkdg2L'),
    'B1_M': (R2, 'qm9_B1_train_gtL_e100_s*', 'S1_etkdg2L_mmff')}.items():
    D[k] = runs(rd, arm, tag)
W = D['CTRL_E'][0]  # ETKDG-population n_true bin counts
L.append('\n## AMR-R (MAT-R) per n_true bin ' + ' / '.join(NB) + '  [n seeds]')
for k, (n, m, s) in D.items():
    L.append(f'  {k:9s} ' + ' '.join(f'{v:.3f}' for v in m) + f'   n={"/".join(str(int(x)) for x in n)} [{s}]')

A_other = {}
for mdl in ('CTRL', 'B1'):
    g, c = D[f'{mdl}_gtL'][1], D[f'{mdl}_cyc'][1]
    ao = np.where(P0 > 0, c + (g - c) / np.where(P0 > 0, P0, 1), np.nan)
    A_other[mdl] = ao
    L.append(f'\n{mdl}: gtL - gtLcycle per bin: ' + ' '.join(f'{v:+.4f}' for v in g - c) +
             '  (bin 1 must be ~0: identical seeds)')
    L.append(f'{mdl}: implied other-conformer-L AMR per bin (A_other): ' +
             ' '.join('  n/a ' if np.isnan(v) else f'{v:.3f}' for v in ao))

# --- tails per n_true bin (ring>=4 basis) per condition
mol_ring = {}
for s in df.smiles.unique():
    m = Chem.MolFromSmiles(s)
    r = m.GetRingInfo().AtomRings() if m else ()
    mol_ring[s] = bool(r) and max(len(x) for x in r) >= 4
df['ge4'] = df.smiles.map(mol_ring)
df = df.merge(mol[['smiles', 'bin']], on='smiles', how='left')


def tails(cond):
    g = df[(df.cond == cond) & df.ge4]
    return g.groupby('bin').apply(lambda x: (x.ring_dihedral_rmsd > 10).mean()).reindex(NB).values


XH, XE, XM = tails('L_A5acyc_cyc_ORACLE'), tails('L_etkdg2L'), tails('L_etkdg2L_mmff')
L.append('\n## tail10 (ring>=4 rows) per n_true bin: A5acyc ' + ' '.join(f'{v:.3f}' for v in XH) +
         ' | ETKDG ' + ' '.join(f'{v:.3f}' for v in XE) + ' | MMFF ' + ' '.join(f'{v:.3f}' for v in XM))


def wavg(v, mask=None):
    w = W.copy()
    if mask is not None:
        w = w * mask
    ok = ~np.isnan(v)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


multi = np.array([0, 1, 1, 1, 1], float)


def predict(G, H, X):
    return G + (H - G) * X / XH


L.append('\n## Validation (ETKDG / MMFF observed vs predicted), leaky G, by subset')
for mdl in ('CTRL', 'B1'):
    G, H = D[f'{mdl}_gtL'][1], D[f'{mdl}_H'][1]
    for nm, mask in (('all', None), ('n_true>=2', multi)):
        L.append(f'  {mdl} [{nm:9s}] ETKDG pred {wavg(predict(G, H, XE), mask):.4f} obs {wavg(D[f"{mdl}_E"][1], mask):.4f}'
                 f' | MMFF pred {wavg(predict(G, H, XM), mask):.4f} obs {wavg(D[f"{mdl}_M"][1], mask):.4f}')

# --- predictions for reference-like acyclic L (xTB type), tail X on the 49.6 % basis (bin tails scaled from MMFF)
L.append('\n## xTB-type predictions, X = overall ring>=4 >10 deg share (bin tails = MMFF bin tails * X / 0.351)')
L.append('   G variants: leaky = gtL (own L almost always present); noleak = A_other (only other conformers\' L);')
L.append('   bin 1 (single conformer) under noleak: [own] = gtL (right for rigid molecules: right pucker IS own L) or')
L.append('   [pess] = A_other of bin 2-3 (flexible single-conformer molecules whose torsional basin is not sampled).')
rows = []
for X in (0.351, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0):
    Xb = XM * X / 0.3506
    r = {'X': X}
    for mdl in ('CTRL', 'B1'):
        G, H, AO = D[f'{mdl}_gtL'][1], D[f'{mdl}_H'][1], A_other[mdl]
        r[f'{mdl}_leaky_all'] = wavg(predict(G, H, Xb))
        r[f'{mdl}_leaky_multi'] = wavg(predict(G, H, Xb), multi)
        r[f'{mdl}_noleak_multi'] = wavg(predict(AO, H, Xb), multi)
        Gown = AO.copy(); Gown[0] = G[0]
        Gpes = AO.copy(); Gpes[0] = AO[1]
        r[f'{mdl}_noleak_all_own'] = wavg(predict(Gown, H, Xb))
        r[f'{mdl}_noleak_all_pess'] = wavg(predict(Gpes, H, Xb))
    rows.append(r)
T = pd.DataFrame(rows)
pd.set_option('display.width', 250)
L.append(T.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
T.to_csv(os.path.join(OUT, 'pred_ntrue_leak.csv'), index=False)
txt = '\n'.join(L)
open(os.path.join(OUT, 'ntrue_leak.out'), 'w').write(txt)
print(txt)

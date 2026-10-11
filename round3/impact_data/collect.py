"""Impact analyst (round 3, P3): collect every synced summary.txt and breakdown.log into flat tables.

Inputs (read only): cluster_sync/results/**/summary.txt, breakdown.log; cluster_sync/round2/results/**/...
Outputs (this directory): runs.csv (one row per run x tag x training seed), bins.csv (n_rot_heavy bins per run),
                          cond_means.txt (means over training seeds per arm x tag).
Usage: python round3/impact_data/collect.py   (from the repo root)
"""
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.dirname(os.path.abspath(__file__))


def parse_summary(path):
    txt = open(path).read()
    d = {}
    m = re.search(r'SUMMARY (.*)', txt)
    for kv in m.group(1).split():
        k, v = kv.split('=')
        d[k] = float(v)
    for line in re.findall(r'SWEEP (.*)', txt):
        parts = dict(kv.split('=') for kv in line.split())
        thr = float(parts['thr'])
        d[f'covR@{thr:g}'] = float(parts['COV-R_mean'])
        d[f'covP@{thr:g}'] = float(parts['COV-P_mean'])
    return d


def parse_bins(path):
    """n_rot_heavy table of breakdown.log -> list of (bin, n, cov_r, mat_r)"""
    txt = open(path).read()
    sec = txt.split('### by n_rot_heavy')[1].split('###')[0]
    rows = []
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 6 and cells[0] in ('0', '1', '2', '3', '4+'):
            rows.append(dict(bin=cells[0], n=int(cells[1]), cov_r=float(cells[2]), mat_r=float(cells[3])))
    return rows


def arm_seed(run):
    m = re.match(r'(.*)_e100_s(\d)$', run)
    if m:
        return m.group(1), int(m.group(2))
    return run, -1


recs, bins = [], []
for base, rnd in ((os.path.join(ROOT, 'cluster_sync', 'results'), 'r1'),
                  (os.path.join(ROOT, 'cluster_sync', 'round2', 'results'), 'r2')):
    for s in glob.glob(os.path.join(base, '*', '*', 'summary.txt')):
        tagdir = os.path.dirname(s)
        run = os.path.basename(os.path.dirname(tagdir))
        tag = os.path.basename(tagdir)
        if run == 'analysis':
            continue
        arm, seed = arm_seed(run)
        d = parse_summary(s)
        d.update(round=rnd, run=run, arm=arm, seed=seed, tag=tag)
        recs.append(d)
        b = os.path.join(tagdir, 'breakdown.log')
        if os.path.exists(b):
            for r in parse_bins(b):
                r.update(round=rnd, arm=arm, seed=seed, tag=tag)
                bins.append(r)

runs = pd.DataFrame(recs)
runs.to_csv(os.path.join(OUT, 'runs.csv'), index=False)
B = pd.DataFrame(bins)
B.to_csv(os.path.join(OUT, 'bins.csv'), index=False)

g = runs.groupby(['round', 'arm', 'tag'])
tab = g.agg(n_seeds=('MAT-R_mean', 'size'), amr=('MAT-R_mean', 'mean'), amr_sd=('MAT-R_mean', 'std'),
            amr_min=('MAT-R_mean', 'min'), amr_max=('MAT-R_mean', 'max'), n=('n_evaluated', 'mean'),
            covR01=('covR@0.1', 'mean'), covR005=('covR@0.05', 'mean')).reset_index()
with open(os.path.join(OUT, 'cond_means.txt'), 'w') as f:
    f.write('# AMR-R (MAT-R_mean, delta 0.5 A) per arm x tag, over training seeds (unpaired). r2 = PRELIMINARY\n')
    f.write(tab.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    f.write('\n\n# rigid bin (n_rot_heavy = 0) MAT-R per arm x tag (mean over seeds; n = molecules in bin)\n')
    rb = B[B.bin == '0'].groupby(['round', 'arm', 'tag']).agg(n=('n', 'mean'), mat_r=('mat_r', 'mean'),
                                                                 sd=('mat_r', 'std')).reset_index()
    f.write(rb.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    f.write('\n\n# all bins, mean over seeds\n')
    ab = B.groupby(['round', 'arm', 'tag', 'bin']).agg(n=('n', 'mean'), mat_r=('mat_r', 'mean')).reset_index()
    piv = ab.pivot_table(index=['round', 'arm', 'tag'], columns='bin', values='mat_r')
    pn = ab.pivot_table(index=['round', 'arm', 'tag'], columns='bin', values='n')
    piv.columns = [f'mat_{c}' for c in piv.columns]
    pn.columns = [f'n_{c}' for c in pn.columns]
    f.write(pd.concat([piv, pn], axis=1).to_string(float_format=lambda x: f'{x:.3f}'))
print(open(os.path.join(OUT, 'cond_means.txt')).read())

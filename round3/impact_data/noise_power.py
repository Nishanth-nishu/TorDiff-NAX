"""Impact analyst (round 3, P3): noise levels and seeds needed to detect an AMR-R effect.

Sources: runs.csv (collect.py): per-seed AMR-R of every arm x tag (training-seed SD);
         cluster_sync/results/analysis/paired_*.md: paired-bootstrap 95% CI half-widths of MAT-R differences
         (per-molecule means over replicate files, then molecules bootstrapped; tools/paired_compare.py:6-10, 69-88).
Seeds needed (two-sided alpha 0.05, power 0.8, two arms with k training seeds each, seed effect only):
    k >= 2 * ((1.96 + 0.84) * sd / delta)^2
Usage: python round3/impact_data/noise_power.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(OUT, '..', '..'))
R = pd.read_csv(os.path.join(OUT, 'runs.csv'))
lines = ['## Training-seed SD of AMR-R (arms with >= 2 seeds)']
g = R[R.seed >= 0].groupby(['arm', 'tag'])['MAT-R_mean'].agg(['size', 'mean', 'std']).reset_index()
g = g[g['size'] >= 2].sort_values('std')
lines.append(g.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
sd_ctrl = g[g.arm.str.contains('CTRL')]['std']
sd_b1 = g[g.arm.str.contains('B1')]['std']
lines.append(f'\nCTRL-family seed SD: median {sd_ctrl.median():.4f}, max {sd_ctrl.max():.4f}; '
             f'B1 seed SD: median {sd_b1.median():.4f}, max {sd_b1.max():.4f}')
samp = R[R.tag.isin(['R0_base_seed0', 'R1_base_seed1', 'R2_base_seed2'])]['MAT-R_mean']
lines.append(f'sampling-seed SD (released model R0/R1/R2): {samp.std():.4f}; A1 gtL sampling seeds: '
             f'{R[R.tag.str.startswith("A1_gtL_model")]["MAT-R_mean"].std():.4f}')

lines.append('\n## Paired-bootstrap CI half-widths (MAT-R) in round-1 paired tables')
for f in sorted(glob.glob(os.path.join(ROOT, 'cluster_sync', 'results', 'analysis', 'paired_*.md'))):
    for line in open(f):
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) >= 9 and cells[1] == 'MAT-R':
            arm, n, diff, lo, hi = cells[0], cells[2], float(cells[5]), float(cells[6]), float(cells[7])
            lines.append(f'  {os.path.basename(f):38s} {arm:8s} n={n:>4s} diff {diff:+.4f} half-width {(hi-lo)/2:.4f}')

lines.append('\n## Seeds per arm for 80% power (seed noise only; add the molecule CI in quadrature)')
for sd in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
    row = []
    for d in (0.003, 0.005, 0.010, 0.020):
        k = 2 * ((1.96 + 0.84) * sd / d) ** 2
        row.append(f'delta {d:.3f}: k>={max(1, int(np.ceil(k)))}')
    lines.append(f'  sd {sd:.4f}: ' + '; '.join(row))
txt = '\n'.join(lines)
open(os.path.join(OUT, 'noise_power.out'), 'w').write(txt)
print(txt)

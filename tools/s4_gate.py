"""
S4 go/no-go gate (round2/DECISION.md D7; code_plan_2 §8). PASS requires all of:
  (1) pairing >= 98 % clean   : PAIRING_CLEAN line of tools/build_paired_pickles.py --summarize
  (2) pair_ok passes          : included in (1) (pairs failing pair_ok are dropped and counted there)
  (3) V18: lambda = 1 reproduces gtLcycle within the seed SD, for every model given with --v18. MAT-R is compared on
      the molecules evaluated in both runs (the lambda family drops molecules with any unsafe pair); the tolerance is
      the training-seed SD of gtLcycle MAT-R of that model family on the same molecules (--sd_files).
  (4) the smoke test passed    : no SMOKE_*_FAILED line and an S4 SUMMARY line in the smoke log
Usage:
  python s4_gate.py --paired_summary $QM9_PAIRED/SUMMARY.txt --smoke_log $LOGS/tordiff_r2_smoke_<id>.log \
     --v18 CR=<lam1.00 eval.pkl>,<gtLcycle eval.pkl> --sd_files CR=<gtLcycle s0>,<s1>,<s2> [--v18 B1=... --sd_files B1=...]
Exit code 0 = PASS, 1 = FAIL.
"""
import pickle
import re
import sys
from argparse import ArgumentParser

import numpy as np

p = ArgumentParser()
p.add_argument('--paired_summary', required=True)
p.add_argument('--smoke_log', required=True)
p.add_argument('--v18', action='append', default=[])
p.add_argument('--sd_files', action='append', default=[])
p.add_argument('--min_clean', type=float, default=98.0)
a = p.parse_args()


def matr(path):
    with open(path, 'rb') as f:
        R = pickle.load(f)['results']
    return {corr: float(np.mean(np.min(r['rmsd'], axis=1))) for (_, corr), r in R.items()}


ok = True
txt = open(a.paired_summary).read()
m = re.search(r'PAIRING_CLEAN confs (\d+)/(\d+) = ([0-9.]+) %', txt)
pct = float(m.group(3)) if m else -1.0
print(f'(1,2) pairing clean {pct:.3f} % (need >= {a.min_clean})', 'PASS' if pct >= a.min_clean else 'FAIL')
ok &= pct >= a.min_clean
m2 = re.search(r'PAIR_OK confs (\d+)/(\d+) = ([0-9.]+) %', txt)
# (2) the pair_ok guard is applied by construction (unsafe pairs are flagged offline and dropped by the S4 loader);
# its pass rate is reported, not thresholded: DECISION D7 does not fix a number (see IMPLEMENTATION.md §3)
print(f'(2) pair_ok rate {m2.group(3) if m2 else "n/a"} % of verified pairs (S4 trains on these only) INFO')

sd = {}
for spec in a.sd_files:
    name, paths = spec.split('=', 1)
    sd[name] = [matr(q) for q in paths.split(',')]
for spec in a.v18:
    name, paths = spec.split('=', 1)
    lam1, cyc = [matr(q) for q in paths.split(',')]
    common = sorted(set(lam1) & set(cyc) & set.intersection(*[set(d) for d in sd.get(name, [cyc])]))
    common = [c for c in common if np.isfinite(lam1[c]) and np.isfinite(cyc[c])]
    diff = np.mean([lam1[c] for c in common]) - np.mean([cyc[c] for c in common])
    seeds = [np.nanmean([d[c] for c in common]) for d in sd.get(name, [])]
    tol = float(np.std(seeds, ddof=1)) if len(seeds) > 1 else 0.003
    good = abs(diff) <= tol
    print(f'(3) V18 {name}: n={len(common)} MAT-R lam1.00 - gtLcycle = {diff:+.5f} A; seed SD = {tol:.5f}',
          'PASS' if good else 'FAIL')
    ok &= good

log = open(a.smoke_log, errors='replace').read()
fails = re.findall(r'SMOKE_\w+_FAILED[^\n]*', log)
s4 = re.search(r'qm9_SMK_S4_lamcond_e1_s0/steps20_seed0/summary.txt: SUMMARY', log)
good = not fails and bool(s4)
print(f'(4) smoke: failures={fails} S4 SUMMARY present={bool(s4)}', 'PASS' if good else 'FAIL')
ok &= good
print('S4_GATE', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)

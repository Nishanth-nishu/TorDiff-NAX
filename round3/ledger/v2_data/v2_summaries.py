"""V2: independent read of every round-2 summary.txt; per-seed values, mean/min/max, n_evaluated, failures."""
import os, sys, re, statistics
root = sys.argv[1]
tab = {}
for run in sorted(os.listdir(root)):
    p = os.path.join(root, run)
    if not os.path.isdir(p):
        continue
    m = re.match(r'(.*)_e100_s(\d)$', run)
    arm, seed = (m.group(1), m.group(2)) if m else (run, '-')
    for tag in sorted(os.listdir(p)):
        f = os.path.join(p, tag, 'summary.txt')
        if not os.path.isfile(f):
            continue
        first = open(f).readline()
        kv = dict(x.split('=') for x in first.split()[1:])
        sweep = {}
        for line in open(f):
            if line.startswith('SWEEP'):
                d = dict(x.split('=') for x in line.split()[1:])
                sweep[d['thr']] = float(d['COV-R_mean'])
        tab.setdefault((arm, tag), []).append((seed, float(kv['MAT-R_mean']), int(kv['n_evaluated']),
                                              int(kv['n_model_failures']), float(kv['COV-R_mean']), sweep.get('0.100')))
for (arm, tag), v in sorted(tab.items()):
    amrs = [x[1] for x in v]
    print(f'{arm:28s} {tag:28s} seeds={"".join(x[0] for x in v):4s} mean={statistics.mean(amrs):.4f} '
          f'min={min(amrs):.4f} max={max(amrs):.4f} per_seed={[round(a, 4) for a in amrs]} n_eval={[x[2] for x in v]} '
          f'fail={[x[3] for x in v]} covR@0.1={[x[5] for x in v]}')

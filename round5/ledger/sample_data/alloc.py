"""Matched-budget allocation analysis on the exact expected curves (curves.py).
Budget B = sum_i n_model_i (TD's 2K rule) over the analysed molecules; n_i in [nmin, N_i] (pool cap 6K).
Allocators: TD 2K rule; uniform (ignores K); ORACLE greedy on true curves (objective = the metric reported);
static feature allocator (2-fold cross-fitted stratum-mean curves by n_rot bin x K bin; no GT of the molecule)."""
import pickle, sys, heapq, numpy as np, pandas as pd
C = pickle.load(open(sys.argv[1], 'rb')); cur = C['curves']; keys = list(cur)
bd = pd.read_csv(sys.argv[2]).set_index(['smiles', 'corrected_smiles'])
M = len(keys)
nm = np.array([cur[k]['n_model'] for k in keys]); Nn = np.array([cur[k]['N'] for k in keys]); Kt = np.array([cur[k]['K'] for k in keys])
nrot = np.array([bd.loc[k, 'n_rot_heavy'] if k in bd.index else -1 for k in keys])
B = int(nm.sum())
print(f'M={M} budget B={B} mean n/mol={B/M:.1f}')

def metric(k, name):
    c = cur[k]
    if name == 'amr': return -c['amr']          # maximise -AMR
    return c['cov'][float(name)]

def evaluate(alloc, name):
    vals = []
    for i, k in enumerate(keys):
        c = cur[k]; n = int(alloc[i])
        vals.append(c['amr'][n] if name == 'amr' else c['cov'][float(name)][n])
    return float(np.mean(vals))

def greedy(curves, budget, nmin=1, cap=None):
    """curves: list of arrays f_i(n), n=0..N_i (higher better); concave-ish -> greedy with lookahead fix.
    Uses marginal gains; for non-concave steps uses best average gain over up to 8-step lookahead."""
    n = np.full(len(curves), nmin); used = n.sum(); h = []
    def push(i):
        f = curves[i]; top = len(f) - 1 if cap is None else min(cap[i], len(f) - 1)
        if n[i] >= top: return
        best = -1e9; bs = 1
        for s in range(1, min(8, top - n[i]) + 1):
            g = (f[n[i] + s] - f[n[i]]) / s
            if g > best + 1e-15: best, bs = g, s
        heapq.heappush(h, (-best, i, bs))
    for i in range(len(curves)): push(i)
    while used < budget and h:
        g, i, s = heapq.heappop(h)
        s = min(s, budget - used)
        n[i] += s; used += s; push(i)
    return n

res = []
names = ['0.5', '0.1', '0.25', 'amr']
for obj in names:
    f = [np.nan_to_num(np.array(cur[k]['amr'] if obj == 'amr' else cur[k]['cov'][float(obj)]) * (-1 if obj == 'amr' else 1), nan=-9) for k in keys]
    if obj == 'amr':
        for a in f: a[0] = a[1] - 1.0  # n=0 forbidden by nmin anyway
    oracle = greedy(f, B, nmin=1)
    res.append((obj, oracle))

alloc = {'TD 2K rule': nm, 'uniform (no K)': None}
u = np.full(M, B // M); u[:B - u.sum()] += 1
u = np.minimum(u, Nn)  # pool cap; redistribute leftover
left = B - u.sum()
while left > 0:
    room = np.where(u < Nn)[0]; add = np.zeros(M, int); add[room[:left]] = 1; u += add; left = B - u.sum()
alloc['uniform (no K)'] = u
for obj, a in res: alloc[f'ORACLE greedy obj={obj}'] = a

# static feature allocator: 2-fold cross-fitting; predicted curve for a molecule = mean *relative* curve of
# other-fold molecules in its stratum, evaluated at n/n_model (K is known to the protocol)
rng = np.random.default_rng(0); fold = rng.integers(0, 2, M)
kb = np.digitize(Kt, [2, 4, 11, 31]); rb = np.clip(nrot, 0, 4)
grid = np.linspace(0, 3, 61)  # n / n_model
def rel(k, obj):
    c = cur[k]; f = -c['amr'] if obj == 'amr' else c['cov'][float(obj)]
    x = np.arange(len(f)) / c['n_model']; f = np.nan_to_num(f, nan=f[1] if obj != 'amr' else -c['amr'][1])
    return np.interp(grid, x, f)
for obj in ['0.5', '0.1', 'amr']:
    R = np.array([rel(k, obj) for k in keys])
    pred = []
    for i, k in enumerate(keys):
        m = (fold != fold[i]) & (kb == kb[i]) & (rb == rb[i])
        if m.sum() < 5: m = (fold != fold[i]) & (kb == kb[i])
        mc = R[m].mean(0)
        x = np.arange(Nn[i] + 1) / nm[i]
        pred.append(np.interp(x, grid, mc))
    alloc[f'feature (n_rot x K strata, cross-fit) obj={obj}'] = greedy(pred, B, nmin=1)

rows = []
for name, a in alloc.items():
    assert a.sum() <= B, name
    r = dict(allocator=name, budget=int(a.sum()))
    for m in names: r['AMR-R' if m == 'amr' else f'COV-R@{m}'] = evaluate(a, m)
    r['min n'] = int(a.min()); r['max n/n_model'] = float((a / nm).max()); r['share at cap'] = float(np.mean(a == Nn))
    rows.append(r)
df = pd.DataFrame(rows)
# TD convention: 1000 molecules, failures (1000 - M analysed... reported separately) -> COV 0
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
df.to_csv(sys.argv[3], index=False)
pickle.dump(dict(alloc=alloc, keys=keys, nrot=nrot, K=Kt, nm=nm), open(sys.argv[3].replace('.csv', '.pkl'), 'wb'))

# per-sample-count curve (global multiplier) to compare: what budget would the 2K rule need to match the oracle?
print('\nTD rule scaled: n_i = round(s * n_model_i)')
for s in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    a = np.clip(np.round(s * nm).astype(int), 1, Nn)
    print(f's={s}: budget/B={a.sum()/B:.3f} COV-R@0.5={evaluate(a,"0.5"):.4f} COV-R@0.1={evaluate(a,"0.1"):.4f} AMR-R={evaluate(a,"amr"):.4f}')

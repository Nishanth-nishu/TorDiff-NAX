"""Deployment framing (K unknown to the allocator except the TD rule): matched total samples B over M molecules,
n_i in [n1, N_i]; metrics on REALISED prefixes of a random permutation of each molecule's pool (5 permutations).
Allocators: TD 2K rule (uses K); uniform; static n_rot strata (2-fold cross-fit); Good-Turing sequential
(GT-free: next batch to the molecule with the largest singleton fraction of rotamer strings); signal two-stage
(cross-fit); ORACLE greedy on the molecule's own expected curve (upper bound)."""
import pickle, sys, heapq, numpy as np, pandas as pd
from collections import Counter
D, CUR, SIG, BD = sys.argv[1:5]
runs = ['R0_base_seed0', 'R1_base_seed1', 'R2_base_seed2']
ev = [pickle.load(open(f'{D}/{r}/eval.pkl', 'rb'))['results'] for r in runs]
cur = pickle.load(open(CUR, 'rb'))['curves']; sig = pickle.load(open(SIG, 'rb'))
bd = pd.read_csv(BD).set_index(['smiles', 'corrected_smiles'])
keys = [k for k in cur if k in sig]; M = len(keys)
P = [np.concatenate([e[k]['rmsd'] for e in ev], 1) for k in keys]
Nn = np.array([p.shape[1] for p in P]); nm = np.array([cur[k]['n_model'] for k in keys])
nrot = np.array([int(bd.loc[k, 'n_rot_heavy']) for k in keys]); rb = np.clip(nrot, 0, 4)
N1 = 6  # stage-1 samples (= minimum pool size, K = 1)
rng0 = np.random.default_rng(0); fold = rng0.integers(0, 2, M)

def realised(perm_P, alloc):
    a = []; c5 = []; c1 = []
    for p, n in zip(perm_P, alloc):
        mn = p[:, :n].min(1); a.append(mn.mean()); c5.append((mn < 0.5).mean()); c1.append((mn < 0.1).mean())
    return np.mean(a), np.mean(c5), np.mean(c1)

def greedy(curves, budget, nmin, cap):
    n = np.minimum(np.full(len(curves), nmin), cap); used = n.sum(); h = []
    def push(i):
        f = curves[i]; top = min(cap[i], len(f) - 1)
        if n[i] >= top: return
        best, bs = -1e18, 1
        for s in range(1, min(8, top - n[i]) + 1):
            g = (f[n[i] + s] - f[n[i]]) / s
            if g > best + 1e-15: best, bs = g, s
        heapq.heappush(h, (-best, i, bs))
    for i in range(len(curves)): push(i)
    while used < budget and h:
        _, i, s = heapq.heappop(h); s = min(s, budget - used); n[i] += s; used += s; push(i)
    return n

def gt_missing(rot):
    c = Counter(rot); f1 = sum(1 for v in c.values() if v == 1); return f1 / len(rot), len(c)

def pad(f, L):  # absolute-n curve padded flat beyond the pool
    return np.concatenate([f, np.full(L - len(f), f[-1])]) if len(f) < L else f[:L]

rows = []
for Bmul in [float(x) for x in (sys.argv[6].split(",") if len(sys.argv) > 6 else ["0.5", "1.0"])]:
    B = int(round(Bmul * nm.sum()))
    res = {k: [] for k in ['TD 2K rule (uses K)', 'uniform', 'static n_rot strata (cross-fit)', 'Good-Turing sequential (GT-free)',
                           'two-stage signal (cross-fit)', 'ORACLE greedy (AMR obj)', 'ORACLE greedy (COV0.1 obj)']}
    for rep in range(5):
        rng = np.random.default_rng(100 + rep)
        perms = [rng.permutation(n) for n in Nn]
        PP = [p[:, o] for p, o in zip(P, perms)]
        ROT = [[sig[k]['rot'][j] for j in o] for k, o in zip(keys, perms)]
        allocs = {}
        a = np.clip(np.round(Bmul * nm).astype(int), 1, Nn); allocs['TD 2K rule (uses K)'] = a
        u = np.full(M, B // M); u[:B - u.sum()] += 1; u = np.minimum(u, Nn)
        while u.sum() < B:
            room = np.where(u < Nn)[0]; take = min(len(room), B - u.sum()); u[room[:take]] += 1
        allocs['uniform'] = u
        L = Nn.max() + 1
        amr_abs = [-pad(np.nan_to_num(cur[k]['amr'], nan=cur[k]['amr'][1] + 1), L) for k in keys]
        # static strata
        pred = []
        for i in range(M):
            m = (fold != fold[i]) & (rb == rb[i]); mc = np.mean([amr_abs[j] for j in np.where(m)[0]], 0); pred.append(mc)
        allocs['static n_rot strata (cross-fit)'] = greedy(pred, B, N1, Nn)
        # Good-Turing sequential, batches of 2, GT-free
        n = np.minimum(np.full(M, N1), Nn); used = n.sum(); h = []
        def pri(i):
            gm, _ = gt_missing(ROT[i][:n[i]]); return gm + 1e-3 / n[i]  # tiny floor: rigid molecules still get L draws
        for i in range(M):
            if n[i] < Nn[i]: heapq.heappush(h, (-pri(i), i))
        while used < B and h:
            _, i = heapq.heappop(h); s = min(2, B - used, Nn[i] - n[i]); n[i] += s; used += s
            if n[i] < Nn[i]: heapq.heappush(h, (-pri(i), i))
        allocs['Good-Turing sequential (GT-free)'] = n.copy()
        # two-stage: signal on first N1 samples -> stratum (n_rot bin x GT-missing tercile), cross-fit curves
        s1 = np.array([gt_missing(ROT[i][:N1])[0] for i in range(M)])
        q = np.digitize(s1, np.quantile(s1, [1 / 3, 2 / 3]))
        pred = []
        for i in range(M):
            m = (fold != fold[i]) & (rb == rb[i]) & (q == q[i])
            if m.sum() < 5: m = (fold != fold[i]) & (rb == rb[i])
            pred.append(np.mean([amr_abs[j] for j in np.where(m)[0]], 0))
        allocs['two-stage signal (cross-fit)'] = greedy(pred, B, N1, Nn)
        allocs['ORACLE greedy (AMR obj)'] = greedy(amr_abs, B, 1, Nn)
        cov1 = [pad(cur[k]['cov'][0.1], L) for k in keys]
        allocs['ORACLE greedy (COV0.1 obj)'] = greedy(cov1, B, 1, Nn)
        for name, al in allocs.items():
            assert al.sum() <= 1.03 * B + 1, (name, al.sum(), B)
            res[name].append(realised(PP, al) + (float(np.mean(al == Nn)), al.sum() / B))
    for name, v in res.items():
        v = np.array(v); rows.append(dict(budget=f'{Bmul}x sum(2K)', allocator=name, **{'AMR-R': v[:, 0].mean(), 'COV-R@0.5': v[:, 1].mean(),
                    'COV-R@0.1': v[:, 2].mean(), 'AMR sd(5 perms)': v[:, 0].std(), 'share at pool cap': v[:, 3].mean(), 'budget used/B': v[:, 4].mean()}))
df = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(f'M={M}; pool cap N_i = 3 x 2K_i (3 sampling seeds of the released QM9 model); stage-1 n1={N1}')
print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
df.to_csv(sys.argv[5], index=False)
# signal validity: Spearman of stage-1 GT-missing with oracle gain (1x->3x)
gain = np.array([cur[k]['amr'][nm[i]] - cur[k]['amr'][3 * nm[i]] for i, k in enumerate(keys)])
s1 = np.array([gt_missing(sig[k]['rot'][:N1])[0] for k in keys])
nmodes = np.array([len(set(sig[k]['rot'][:nm[i]])) for i, k in enumerate(keys)])
E = [sig[k]['E'] for k in keys]; Esd = np.array([np.nanstd(e[:nm[i]]) for i, e in enumerate(E)])
from scipy.stats import spearmanr
for nmk, s in [('GT-missing(first 6)', s1), ('distinct rotamers in 1x', nmodes), ('MMFF E sd in 1x', Esd), ('n_rot', nrot), ('K (oracle)', np.array([cur[k]['K'] for k in keys]))]:
    r = spearmanr(s, gain, nan_policy='omit'); print(f'spearman(signal={nmk}, oracle AMR gain 1x->3x) = {r.correlation:.3f}')

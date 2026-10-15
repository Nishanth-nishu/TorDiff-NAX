"""Protocol-preserving framing: output exactly n_model_i = 2K_i conformers per molecule, chosen from a pool of
c x 2K_i proposals (c = 1, 2, 3; first c x 2K columns of a random pool permutation). Selectors: random (= TD at
1x compute), torsion farthest-point (GT-free), lowest MMFF single-point energy (GT-free), ORACLE greedy (min AMR,
uses GT; bound). Precision is reported because selection, unlike random reallocation, changes it."""
import pickle, sys, numpy as np, pandas as pd
D, CUR, SIG = sys.argv[1:4]
runs = ['R0_base_seed0', 'R1_base_seed1', 'R2_base_seed2']
ev = [pickle.load(open(f'{D}/{r}/eval.pkl', 'rb'))['results'] for r in runs]
cur = pickle.load(open(CUR, 'rb'))['curves']; sig = pickle.load(open(SIG, 'rb'))
keys = [k for k in cur if k in sig]
def cd(a, b):  # mean circular distance (1 - cos) over torsions
    return (1 - np.cos(np.deg2rad(a[:, None, :] - b[None, :, :]))).mean(-1) if a.shape[1] else np.zeros((len(a), len(b)))
def fps(X, k, rng):
    if X.shape[1] == 0: return rng.choice(len(X), k, replace=False)
    sel = [int(rng.integers(len(X)))]; d = cd(X, X[sel])[:, 0]
    while len(sel) < k:
        j = int(np.argmax(d)); sel.append(j); d = np.minimum(d, cd(X, X[[j]])[:, 0])
    return np.array(sel)
def oracle(Pm, k):
    sel = []; cur_min = np.full(Pm.shape[0], np.inf)
    for _ in range(k):
        cand = np.minimum(cur_min[:, None], Pm).mean(0); cand[sel] = np.inf; j = int(np.argmin(cand)); sel.append(j); cur_min = np.minimum(cur_min, Pm[:, j])
    return np.array(sel)
rows = []
for rep in range(3):
    rng = np.random.default_rng(7 + rep)
    acc = {}
    for k in keys:
        Pm = np.concatenate([e[k]['rmsd'] for e in ev], 1); n = cur[k]['n_model']; o = rng.permutation(Pm.shape[1])
        X = sig[k]['tors'][o]; E = sig[k]['E'][o]; Pm = Pm[:, o]
        for c in (1, 2, 3):
            pool = np.arange(min(c * n, Pm.shape[1]))
            choices = {'random': pool[:n]}
            if c > 1:
                choices['torsion FPS'] = pool[fps(X[pool], n, rng)]
                choices['lowest MMFF E'] = pool[np.argsort(np.nan_to_num(E[pool], nan=np.inf))[:n]]
                choices['ORACLE greedy'] = pool[oracle(Pm[:, pool], n)]
                keep = pool[np.argsort(np.nan_to_num(E[pool], nan=np.inf))[:max(n, int(np.ceil(0.75 * len(pool))))]]
                choices['MMFF-filter 25% + torsion FPS'] = keep[fps(X[keep], n, rng)]
                keep2 = pool[np.argsort(np.nan_to_num(E[pool], nan=np.inf))[:max(n, int(np.ceil(0.5 * len(pool))))]]
                choices['MMFF-filter 50% + torsion FPS'] = keep2[fps(X[keep2], n, rng)]
            for nmn, s in choices.items():
                S = Pm[:, s]
                acc.setdefault((c, nmn), []).append((S.min(1).mean(), (S.min(1) < 0.5).mean(), (S.min(1) < 0.1).mean(), S.min(0).mean(), (S.min(0) < 0.5).mean()))
    for (c, nmn), v in acc.items():
        v = np.array(v).mean(0); rows.append(dict(rep=rep, compute=f'{c}x', selector=nmn, AMR_R=v[0], COV_R05=v[1], COV_R01=v[2], AMR_P=v[3], COV_P05=v[4]))
df = pd.DataFrame(rows).groupby(['compute', 'selector'], sort=False).mean(numeric_only=True).drop(columns='rep')
pd.set_option('display.width', 200); print(f'M={len(keys)} (3 random pool orders)'); print(df.to_string(float_format=lambda v: f'{v:.4f}'))
df.to_csv(sys.argv[4])

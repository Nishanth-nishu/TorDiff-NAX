"""Exact expected recall curves vs number of samples n, per molecule, from the pooled RMSD matrices of
the released TD QM9 model (3 sampling seeds R0/R1/R2, 2K samples each -> pool N = 6K).
For a uniformly random n-subset of the pool (without replacement):
  P(row j covered at d) = 1 - C(N-m_j, n)/C(N, n),  m_j = #pool samples with rmsd < d
  E[min_j] = sum_k w_k(n) r_j,(k),  w_k(n) = [C(N-k+1,n) - C(N-k,n)]/C(N,n)  (k = 1..N, sorted ascending)
Writes curves.pkl: {key: dict(K, n_model, N, cov{d: array[n=0..N]}, amr: array[n=0..N])}."""
import pickle, sys, numpy as np
from math import lgamma
D = sys.argv[1]
runs = ['R0_base_seed0', 'R1_base_seed1', 'R2_base_seed2']
ev = [pickle.load(open(f'{D}/{r}/eval.pkl', 'rb')) for r in runs]
keys = [k for k in ev[0]['results'] if all(k in e['results'] for e in ev)]
print('molecules in all 3 runs', len(keys), [len(e['results']) for e in ev], [e['num_failures'] for e in ev])
THR = [0.05, 0.1, 0.25, 0.5, 0.75]

def logC(a, b):
    return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1) if 0 <= b <= a else -np.inf

def lc_vec(a, n):  # log C(a, n) for vector a
    a = np.asarray(a, float)
    out = np.full(a.shape, -np.inf)
    ok = a >= n
    from scipy.special import gammaln
    out[ok] = gammaln(a[ok] + 1) - gammaln(n + 1) - gammaln(a[ok] - n + 1)
    return out

out = {}
nan_mols = 0
for key in keys:
    mats = [e['results'][key]['rmsd'] for e in ev]
    if any(np.isnan(m).any() for m in mats):
        nan_mols += 1
        continue
    P = np.concatenate(mats, axis=1)  # rows = GT, cols = pool
    K, N = P.shape
    n_model = mats[0].shape[1]
    ns = np.arange(0, N + 1)
    Rs = np.sort(P, axis=1)  # K x N
    cov = {}
    for d in THR:
        m = (P < d).sum(1)  # K
        c = np.zeros(N + 1)
        for n in range(1, N + 1):
            lcN = logC(N, n)
            p_unc = np.exp(lc_vec(N - m, n) - lcN)
            c[n] = np.mean(1 - p_unc)
        cov[d] = c
    amr = np.full(N + 1, np.nan)
    kk = np.arange(1, N + 1)
    for n in range(1, N + 1):
        lcN = logC(N, n)
        S = np.exp(lc_vec(N - kk + 1, n) - lcN)   # P(min >= r_(k))
        S1 = np.exp(lc_vec(N - kk, n) - lcN)      # P(min >= r_(k+1))
        w = S - S1
        amr[n] = float((Rs @ w).mean())
    # precision of the pool (expectation is n-independent for random subsets)
    covp = {d: float(np.mean(P.min(0) < d)) for d in THR}
    out[key] = dict(K=K, n_model=n_model, N=N, cov=cov, amr=amr, covp=covp, amrp=float(P.min(0).mean()),
                    per_seed_amr=[float(m.min(1).mean()) for m in mats],
                    per_seed_cov05=[float((m.min(1) < 0.5).mean()) for m in mats])
print('molecules with NaN rows skipped', nan_mols, 'kept', len(out))
pickle.dump(dict(curves=out, n_fail=ev[0]['num_failures'], keys=keys), open(sys.argv[2], 'wb'))

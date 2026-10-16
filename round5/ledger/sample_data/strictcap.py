"""D-101 check: strictly K-free uniform n per molecule, capped at the pool N_i = 6K_i WITHOUT redistribution."""
import pickle, sys, numpy as np
C = pickle.load(open(sys.argv[1], 'rb'))['curves']; ks = list(C)
nm = np.array([C[k]['n_model'] for k in ks]); N = np.array([C[k]['N'] for k in ks]); B = nm.sum()
for n in (14, 27, 55):
    a = np.minimum(n, N)
    amr = np.mean([C[k]['amr'][x] for k, x in zip(ks, a)]); cov = np.mean([C[k]['cov'][0.5][x] for k, x in zip(ks, a)])
    print(f'uniform n={n}: budget used/B={a.sum()/B:.3f} capped share={np.mean(n > N):.3f} AMR-R={amr:.4f} COV-R@0.5={cov:.4f}')

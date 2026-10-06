"""
Likelihood/energy-based importance resampling of generated conformers (ablation plan H).
Upstream torsional diffusion has NO low-temperature sampling for conformer generation; exact likelihoods are only
used for the torsional Boltzmann generator. This script post-processes a generate_confs.py pickle produced WITHOUT
--no_energy (so every mol carries .mmff_energy and .euclidean_dlogp; with --ode --likelihood full, dlogp is the model
log-density change, otherwise dlogp=0 and the weights reduce to pure MMFF Boltzmann weights).

  log w_i = -E_i / kT - euclidean_dlogp_i      (same form as utils/boltzmann.py:46, kT in kcal/mol)

For each molecule it keeps n_keep = round(len(confs) / oversample) conformers, drawn by weight (without
replacement by default) or by lowest energy (--mode topk_energy) or uniformly (--mode uniform, the control).

Usage:
  python ../tools/resample_confs.py --in_confs pool_seed1.pkl pool_seed2.pkl pool_seed3.pkl pool_seed4.pkl \
      --fill_from baseline.pkl --out_confs resampled_T300.pkl --oversample 4 --temp 300
"""
import pickle
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument('--in_confs', required=True, nargs='+', help='one or more generate_confs.py pickles (pools are concatenated per molecule)')
parser.add_argument('--fill_from', default=None, help='pickle used as-is for molecules missing from all pools '
                    '(e.g. 0-rotatable-bond molecules dropped by --likelihood runs)')
parser.add_argument('--out_confs', required=True)
parser.add_argument('--oversample', type=float, default=4.0, help='generated/kept ratio (generate with --confs_per_mol or 2x csv)')
parser.add_argument('--temp', type=float, default=300.0)
parser.add_argument('--mode', choices=['weights', 'topk_energy', 'uniform'], default='weights')
parser.add_argument('--no_dlogp', action='store_true', help='ignore the model likelihood term (pure MMFF weights)')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
kT = 1.38e-23 * 6.022e23 * args.temp / 4184  # kcal/mol (note: upstream boltzmann.py divides by 4148, a typo)
confs = {}
for path in args.in_confs:
    with open(path, 'rb') as f:
        for smi, mols in pickle.load(f).items():
            confs.setdefault(smi, []).extend(mols)
print(f'pooled {len(args.in_confs)} files -> {len(confs)} molecules')

out, ess_all = {}, []
for smi, mols in confs.items():
    n_keep = max(1, int(round(len(mols) / args.oversample)))
    if args.mode == 'uniform':
        idx = rng.choice(len(mols), n_keep, replace=False)
    else:
        E = np.array([getattr(m, 'mmff_energy', np.nan) for m in mols], dtype=float)
        dlogp = np.zeros_like(E) if args.no_dlogp else np.array([getattr(m, 'euclidean_dlogp', 0.0) for m in mols], dtype=float)
        ok = np.isfinite(E) & np.isfinite(dlogp)
        if ok.sum() < n_keep:
            idx = rng.choice(len(mols), n_keep, replace=False)
        elif args.mode == 'topk_energy':
            idx = np.where(ok)[0][np.argsort(E[ok])[:n_keep]]
        else:
            logw = np.full(len(mols), -np.inf)
            logw[ok] = -E[ok] / kT - dlogp[ok]
            w = np.exp(logw - logw[ok].max()); w /= w.sum()
            ess_all.append(1.0 / np.sum(w ** 2))
            nz = int((w > 0).sum())
            idx = rng.choice(len(mols), min(n_keep, nz), replace=False, p=w)
    out[smi] = [mols[i] for i in idx]

if args.fill_from:
    with open(args.fill_from, 'rb') as f:
        fill = pickle.load(f)
    missing = [k for k in fill if k not in out]
    for k in missing:
        out[k] = fill[k]
    print(f'filled {len(missing)} molecules from {args.fill_from}')
if ess_all:
    print(f'ESS per molecule: mean={np.mean(ess_all):.2f} median={np.median(ess_all):.2f}')
with open(args.out_confs, 'wb') as f:
    pickle.dump(out, f)
print('wrote', args.out_confs, 'molecules:', len(out))

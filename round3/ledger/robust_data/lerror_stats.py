"""Scout ROBUST (round 3): error structure of each round-2 test-time L condition vs the AMR-R of the three models.
Reads only existing files; run from the repo root:  python round3/ledger/robust_data/lerror_stats.py
Inputs:  cluster_sync/round2/data/QM9/round2_seeds/l_error.csv  (one row per seed, RMSD to its source GT; tools/lgeom.py:317-337)
         cluster_sync/round2/results/<run>/<tag>/summary.txt     (SUMMARY line, MAT-R_mean = AMR-R)
Output:  printed tables (saved as round3/ledger/robust_data/lerror_stats.txt)."""
import glob, re
import numpy as np, pandas as pd
from rdkit import Chem

E = pd.read_csv('cluster_sync/round2/data/QM9/round2_seeds/l_error.csv')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)

def q(s, p): return float(np.nanpercentile(s.dropna(), p)) if s.notna().any() else np.nan

print('## T1. L error per condition (rows = seeds; heavy-atom bonds A, heavy angles deg, endocyclic dihedrals deg)')
rows = []
for c, d in E.groupby('cond'):
    rd = d['ring_dihedral_rmsd'].dropna()
    rows.append(dict(cond=c, n=len(d), bond_heavy_mean=d.bond_rmsd_heavy_any.mean(), angle_heavy_mean=d.angle_rmsd_heavy_any.mean(),
                     angle_all_acyc_mean=d.angle_rmsd_all_acyc.mean(), bond_all_acyc_mean=d.bond_rmsd_all_acyc.mean(),
                     ringdih_n=len(rd), ringdih_mean=rd.mean(), ringdih_median=rd.median(), ringdih_p90=q(rd, 90),
                     ringdih_gt5=(rd > 5).mean(), ringdih_gt10=(rd > 10).mean(), ringdih_gt20=(rd > 20).mean()))
T1 = pd.DataFrame(rows).set_index('cond')
print(T1.round(4).to_string())

print('\n## T2. Ring-dihedral error tail by smallest ring size (ETKDG, ETKDG+MMFF, noise 0.04 A/axis)')
ring_size = {}
for s in E.smiles.unique():
    m = Chem.MolFromSmiles(s)
    ri = m.GetRingInfo().AtomRings() if m is not None else ()
    ring_size[s] = min((len(r) for r in ri), default=0)
E['min_ring'] = E.smiles.map(ring_size).clip(upper=7)
for c in ['L_etkdg2L', 'L_etkdg2L_mmff', 'L_noise0.04pa_cyc_ORACLE']:
    d = E[(E.cond == c) & E.ring_dihedral_rmsd.notna()]
    g = d.groupby('min_ring').ring_dihedral_rmsd
    print(c); print(pd.DataFrame(dict(n=g.size(), mean=g.mean(), median=g.median(), frac_gt10=g.apply(lambda s: (s > 10).mean()))).round(3).to_string())

print('\n## T3. AMR-R (MAT-R_mean, delta 0.5 A) per test-time L condition, mean over available training seeds (unpaired, PRELIMINARY)')
res = {}
for f in glob.glob('cluster_sync/round2/results/qm9_*/*/summary.txt'):
    f = f.replace(chr(92), '/')
    run, tag = f.split('/')[-3], f.split('/')[-2]
    line = open(f).read()
    amr = float(re.search(r'MAT-R_mean=([0-9.]+)', line).group(1)); n = int(re.search(r'n_evaluated=(\d+)', line).group(1))
    arm = re.sub(r'_e100_s\d$', '', run)
    res.setdefault((arm, tag), []).append((amr, n))
T3 = pd.DataFrame([dict(arm=a, tag=t, n_seeds=len(v), amr_mean=np.mean([x[0] for x in v]), amr_min=min(x[0] for x in v),
                        amr_max=max(x[0] for x in v), n_mols=v[0][1]) for (a, t), v in sorted(res.items())])
print(T3.round(4).to_string(index=False))

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

# ---------------------------------------------------------------- added 2026-10-08 after V2 disputes D-101, D-102, D-108, D-109
print('\n## T4a. Common molecules of L_etkdg2L, L_lam0.00 (matched RDKit L = training-type) and noise 0.04 (D-108)')
def rows_common(conds):
    sets = [set(E[E.cond == c].smiles) for c in conds]
    com = set.intersection(*sets)
    return com
com = rows_common(['L_etkdg2L', 'L_lam0.00_cyc_ORACLE', 'L_noise0.04pa_cyc_ORACLE'])
print('n_common_mols', len(com))
for c in ['L_etkdg2L', 'L_lam0.00_cyc_ORACLE', 'L_noise0.04pa_cyc_ORACLE']:
    d = E[(E.cond == c) & E.smiles.isin(com)]; rd = d.ring_dihedral_rmsd.dropna()
    print(f'{c:28s} bond_heavy={d.bond_rmsd_heavy_any.mean():.4f} angle_heavy={d.angle_rmsd_heavy_any.mean():.3f} '
          f'angle_heavy_acyc={d.angle_rmsd_heavy_acyc.mean():.3f} ringdih_gt10={(rd > 10).mean():.4f}')

print('\n## T4b. Endocyclic-dihedral error > 10 deg, split by whether the molecule has a ring of >= 4 atoms (D-101, D-108)')
ring_max = {}
for s in E.smiles.unique():
    m = Chem.MolFromSmiles(s)
    ri = m.GetRingInfo().AtomRings() if m is not None else ()
    ring_max[s] = max((len(r) for r in ri), default=0)
for c in ['L_etkdg2L', 'L_noise0.04pa_cyc_ORACLE']:
    d = E[(E.cond == c) & E.ring_dihedral_rmsd.notna()]
    only3 = d[d.smiles.map(ring_max) == 3].ring_dihedral_rmsd; ge4 = d[d.smiles.map(ring_max) >= 4].ring_dihedral_rmsd
    print(f'{c:28s} only_3_rings: n={len(only3)} max_err={only3.max():.6f} | ring_ge4: n={len(ge4)} frac_gt10={(ge4 > 10).mean():.4f}')

print('\n## T4c. A5ring vs GT+noise on the A5 molecule set (D-102); L error columns are means over seeds')
com5 = rows_common(['L_A5ring_cyc_ORACLE', 'L_noise0.04pa_cyc_ORACLE', 'L_noise0.02pa_cyc_ORACLE'])
print('n_common_mols', len(com5))
for c in ['L_A5ring_cyc_ORACLE', 'L_noise0.04pa_cyc_ORACLE', 'L_noise0.02pa_cyc_ORACLE']:
    d = E[(E.cond == c) & E.smiles.isin(com5)]
    print(f'{c:28s} bond_heavy={d.bond_rmsd_heavy_any.mean():.4f} angle_heavy={d.angle_rmsd_heavy_any.mean():.2f} '
          f'angle_heavy_acyc={d.angle_rmsd_heavy_acyc.mean():.2f} angle_all_acyc={d.angle_rmsd_all_acyc.mean():.2f} '
          f'ringdih_mean={d.ring_dihedral_rmsd.mean():.2f}')

print('\n## T5. Round-1 in-job evaluations (same in-job test sets as S2/S3: RDKit L n=935, gtLcycle n=996) (D-109)')
res1 = {}
for f in glob.glob('cluster_sync/results/qm9_*/*/summary.txt'):
    f = f.replace(chr(92), '/')
    run, tag = f.split('/')[-3], f.split('/')[-2]
    if tag not in ('steps20_seed0', 'steps20_seed0_gtLcycle'):
        continue
    line = open(f).read()
    amr = float(re.search(r'MAT-R_mean=([0-9.]+)', line).group(1)); n = int(re.search(r'n_evaluated=(\d+)', line).group(1))
    res1.setdefault((re.sub(r'_e100_s\d$', '', run), tag), []).append((amr, n))
for (a, t), v in sorted(res1.items()):
    print(f'{a:32s} {t:24s} n_seeds={len(v)} amr_mean={np.mean([x[0] for x in v]):.4f} '
          f'amr_min={min(x[0] for x in v):.4f} amr_max={max(x[0] for x in v):.4f} n_mols={v[0][1]}')

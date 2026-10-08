"""V2 independent recompute of l_error.csv statistics (does not import the scout's script)."""
import csv, math, sys, collections
from rdkit import Chem

path = sys.argv[1]
rows = collections.defaultdict(list)
with open(path, newline='') as f:
    for r in csv.DictReader(f):
        rows[r['cond']].append(r)

def fl(x):
    return float(x) if x not in ('', None) else float('nan')

def mean(v):
    v = [x for x in v if not math.isnan(x)]
    return sum(v) / len(v) if v else float('nan')

def median(v):
    v = sorted(x for x in v if not math.isnan(x))
    n = len(v)
    return (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])) if n else float('nan')

print('cond n_rows n_mols bond_heavy_any angle_heavy_any angle_all_acyc bond_all_acyc ringdih_n ringdih_mean ringdih_median frac_gt10')
mols = {}
for c in sorted(rows):
    R = rows[c]
    mols[c] = set(r['smiles'] for r in R)
    rd = [fl(r['ring_dihedral_rmsd']) for r in R]
    rd = [x for x in rd if not math.isnan(x)]
    print(c, len(R), len(mols[c]),
          round(mean([fl(r['bond_rmsd_heavy_any']) for r in R]), 4),
          round(mean([fl(r['angle_rmsd_heavy_any']) for r in R]), 4),
          round(mean([fl(r['angle_rmsd_all_acyc']) for r in R]), 4),
          round(mean([fl(r['bond_rmsd_all_acyc']) for r in R]), 4),
          len(rd), round(mean(rd), 4), round(median(rd), 4), round(sum(x > 10 for x in rd) / len(rd), 4) if rd else None)

# ring size: smallest SSSR ring, and largest ring, per SMILES
def ring_sizes(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return [len(r) for r in Chem.GetSymmSSSR(m)]

rs = {}
for s in set().union(*mols.values()):
    rs[s] = ring_sizes(s)

def by_size(c, key):
    g = collections.defaultdict(list)
    for r in rows[c]:
        x = fl(r['ring_dihedral_rmsd'])
        if math.isnan(x):
            continue
        sz = rs[r['smiles']]
        k = key(sz) if sz else 0
        g[min(k, 7)].append(x)
    return {k: (len(v), round(mean(v), 3), round(median(v), 3), round(sum(x > 10 for x in v) / len(v), 4)) for k, v in sorted(g.items())}

for c in ('L_etkdg2L', 'L_etkdg2L_mmff', 'L_noise0.04pa_cyc_ORACLE', 'L_lam0.00_cyc_ORACLE'):
    print('\n', c, 'by MIN ring size (n, mean, median, frac>10):', by_size(c, min))
    print('  ', c, 'by MAX ring size:', by_size(c, max))

# common-molecule comparisons
def stats_on(c, S):
    R = [r for r in rows[c] if r['smiles'] in S]
    rd = [fl(r['ring_dihedral_rmsd']) for r in R]
    rd = [x for x in rd if not math.isnan(x)]
    return dict(n=len(R), bond=mean([fl(r['bond_rmsd_heavy_any']) for r in R]),
                ang=mean([fl(r['angle_rmsd_heavy_any']) for r in R]),
                ang_all_acyc=mean([fl(r['angle_rmsd_all_acyc']) for r in R]),
                bond_all_acyc=mean([fl(r['bond_rmsd_all_acyc']) for r in R]),
                ang_heavy_acyc=mean([fl(r['angle_rmsd_heavy_acyc']) for r in R]),
                bond_heavy_acyc=mean([fl(r['bond_rmsd_heavy_acyc']) for r in R]),
                rd_mean=mean(rd), rd_gt10=(sum(x > 10 for x in rd) / len(rd)) if rd else float('nan'))

def show(tag, S, conds):
    print('\n', tag, 'n_common_mols =', len(S))
    for c in conds:
        d = stats_on(c, S)
        print('  ', c, {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()})

C1 = mols['L_etkdg2L'] & mols['L_noise0.04pa_cyc_ORACLE']
show('ETKDG vs noise0.04 vs lam0 (common to etkdg & noise)', C1 & mols['L_lam0.00_cyc_ORACLE'],
     ['L_etkdg2L', 'L_lam0.00_cyc_ORACLE', 'L_noise0.04pa_cyc_ORACLE', 'L_noise0.02pa_cyc_ORACLE'])
C2 = mols['L_A5ring_cyc_ORACLE'] & mols['L_noise0.04pa_cyc_ORACLE']
show('A5ring vs noise (common)', C2, ['L_A5ring_cyc_ORACLE', 'L_noise0.04pa_cyc_ORACLE', 'L_noise0.02pa_cyc_ORACLE', 'L_lam0.00_cyc_ORACLE', 'L_A5acyc_cyc_ORACLE'])
show('A5 family all', mols['L_A5ring_cyc_ORACLE'], ['L_A5ring_cyc_ORACLE', 'L_A5acyc_cyc_ORACLE', 'L_lam0.00_cyc_ORACLE'])
show('noise all', mols['L_noise0.04pa_cyc_ORACLE'], ['L_noise0.04pa_cyc_ORACLE', 'L_noise0.02pa_cyc_ORACLE'])
print('\nmol counts:', {c: len(v) for c, v in mols.items()})
print('A5ring mols not in noise:', len(mols['L_A5ring_cyc_ORACLE'] - mols['L_noise0.04pa_cyc_ORACLE']),
      ' noise mols not in A5ring:', len(mols['L_noise0.04pa_cyc_ORACLE'] - mols['L_A5ring_cyc_ORACLE']))

"""Scaffold-novel OOD stratum inside the TD QM9 test set (no retraining needed): a test molecule is scaffold-novel
if its Bemis-Murcko scaffold (generic=False, on the heavy-atom SMILES) occurs in no training molecule (TD split.npy
index 0 into the sorted raw-pickle list, as torsional-diffusion/utils/dataset.py:168-179). Acyclic molecules have
the empty scaffold and are reported separately. Reports R0 per-molecule metrics by stratum."""
import sys, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
D = sys.argv[1]
names = sorted(l.strip() for l in open(f'{D}/qm9_files.txt') if l.strip().endswith('.pickle'))
split = np.load(f'{D}/split.npy', allow_pickle=True)
train = [names[i][:-7] for i in sorted(split[0]) if i < len(names)]
def scaf(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
tr = set(filter(None, (scaf(s) for s in train)))
print('train molecules', len(train), 'distinct train scaffolds', len(tr))
bd = pd.read_csv(f'{D}/R0_base_seed0/breakdown.csv')
test = pd.read_csv(f'{D}/test_smiles.csv')
test['scaffold'] = [scaf(s) for s in test.smiles]
test['stratum'] = ['acyclic' if s == '' else ('unparsable' if s is None else ('scaffold-seen' if s in tr else 'scaffold-novel')) for s in test.scaffold]
print(test.stratum.value_counts().to_string())
m = test.merge(bd, on=['smiles', 'corrected_smiles'], how='left')
m['failed'] = m.mat_r.isna()
g = m.groupby('stratum').agg(n=('smiles', 'size'), failures=('failed', 'sum'), AMR_R=('mat_r', 'mean'), COV_R_success=('cov_r', 'mean'),
                             n_rot=('n_rot_heavy', 'mean'), K=('n_true', 'mean'))
g['COV_R_TDconv'] = m.assign(c=m.cov_r.fillna(0)).groupby('stratum').c.mean()
print(g.to_string(float_format=lambda v: f'{v:.4f}'))
m[['smiles', 'corrected_smiles', 'scaffold', 'stratum']].to_csv(sys.argv[2], index=False)

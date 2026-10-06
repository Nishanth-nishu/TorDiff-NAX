"""
Build seed pickles for the local-structure ablations (ablation plan A1 / G2).

Outputs
  --out_gt_confs : {raw_smi (csv col 0): [GT conformer mols]}   -> generate_confs.py --seed_confs
                   The diffusion then starts from GROUND-TRUTH local structures (bond lengths/angles/ring
                   puckers of GEOM conformers) with torsions re-randomised by perturb_seeds (oracle-L upper bound).
  --out_gt_mols  : {corrected_smi (csv col 2): [GT mol]}         -> generate_confs.py --seed_mols
                   RDKit ETKDG still generates the local structure, but from the GT molecular graph
                   (atom order + stereo tags of the GEOM mol) instead of from the SMILES string.

Key conventions (checked against upstream code):
  * generate_confs.py iterates csv rows as (raw_smi, n_confs, smi) positionally;
    --seed_confs looks up seed_confs[raw_smi], --seed_mols looks up seed_mols[smi]  (diffusion/sampling.py:get_seed)
  * evaluate_confs.py indexes test_mols.pkl with the `smiles` column (== col 0) and compares against the
    `corrected_smiles` column (== col 2).
  * GT conformers are filtered exactly as evaluate_confs.clean_confs does (non-isomeric canonical SMILES must match).

Usage (from the repo root):
  python ../tools/make_seed_pickles.py --test_csv data/QM9/test_smiles.csv --true_mols data/QM9/test_mols.pkl \
      --out_gt_confs data/QM9/test_gt_seed_confs.pkl --out_gt_mols data/QM9/test_gt_seed_mols.pkl
"""
import pickle, copy
from argparse import ArgumentParser
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

parser = ArgumentParser()
parser.add_argument('--test_csv', required=True)
parser.add_argument('--true_mols', required=True)
parser.add_argument('--out_gt_confs', required=True)
parser.add_argument('--out_gt_mols', required=True)
args = parser.parse_args()


def clean_confs(smi, confs):
    # identical to evaluate_confs.py:clean_confs
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    return [c for c in confs
            if Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False) == smi]


df = pd.read_csv(args.test_csv)
cols = list(df.columns)
print('csv columns:', cols)
if 'smiles' in cols and cols[0] != 'smiles':
    print('WARNING: column 0 is not `smiles`; generate_confs.py uses column 0 as raw_smi')
with open(args.true_mols, 'rb') as f:
    true_mols = pickle.load(f)

gt_confs, gt_mols = {}, {}
n_missing = n_empty = 0
for row in df.values:
    raw_smi, smi = row[0], row[2]
    key = row[cols.index('smiles')] if 'smiles' in cols else raw_smi
    if key not in true_mols:
        n_missing += 1
        continue
    confs = clean_confs(smi, true_mols[key])
    if not confs:
        n_empty += 1
        continue
    gt_confs[raw_smi] = confs
    # stereo tags (chiral centres + E/Z) perceived from GT conformer 0's 3D coordinates, so that ETKDG re-embedding
    # from the GT graph (--seed_mols, arm G1) reproduces the GT stereoisomer instead of relying on whatever tags the
    # GEOM mol happens to carry (if none, ETKDG picks stereo at random and G1 would not isolate stereo error)
    m0 = copy.deepcopy(confs[0])
    Chem.AssignStereochemistryFrom3D(m0, confId=m0.GetConformer().GetId(), replaceExistingTags=True)
    gt_mols[smi] = [m0]

print(f'{len(gt_confs)} molecules written; {n_missing} not in true_mols; {n_empty} with no clean GT conformer')
with open(args.out_gt_confs, 'wb') as f:
    pickle.dump(gt_confs, f)
with open(args.out_gt_mols, 'wb') as f:
    pickle.dump(gt_mols, f)

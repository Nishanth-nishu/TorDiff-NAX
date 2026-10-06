"""
Chirality / E-Z consistency of generated conformers vs. ground truth (ablation plan G).

Why: torsion updates are rigid rotations (utils/torsion.py:modify_conformer), so tetrahedral chirality is frozen at
whatever the RDKit seed had; seeds are embedded from the csv SMILES (diffusion/sampling.py:get_seed ->
featurize_mol_from_smiles). Chiral tags are computed but NOT used as node features (utils/featurization.py:59,63).
Conversely, non-ring double bonds count as "rotatable" (get_transformation_mask has no bond-order check), so the
model CAN flip E/Z. Evaluation uses GetBestRMS (proper rotations only), so a wrong enantiomer/diastereomer is never
matched. This script quantifies both effects.

Usage:
  python ../tools/stereo_check.py --confs ../results/qm9_baseline.pkl --test_csv data/QM9/test_smiles.csv \
      --true_mols data/QM9/test_mols.pkl --out ../results/qm9_baseline_stereo.csv
"""
import pickle
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
parser = ArgumentParser()
parser.add_argument('--confs', required=True)
parser.add_argument('--test_csv', required=True)
parser.add_argument('--true_mols', required=True)
parser.add_argument('--out', default=None)
args = parser.parse_args()


def canon_stereo(mol):
    """canonical isomeric SMILES (heavy atoms) with stereo perceived from the 3D conformer"""
    m = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(m)
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True)
    except Exception:
        return None


def n_stereo_elements(smi):
    m = Chem.MolFromSmiles(smi)
    centers = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    si = Chem.FindPotentialStereo(m)
    n_db = sum(1 for s in si if s.type == Chem.StereoType.Bond_Double)
    return len(centers), n_db


df = pd.read_csv(args.test_csv)
with open(args.confs, 'rb') as f:
    gen = pickle.load(f)
with open(args.true_mols, 'rb') as f:
    true = pickle.load(f)

rows = []
for r in df.itertuples(index=False):
    key, corrected = r[0], r[2]
    if corrected not in gen or key not in true:
        continue
    gt_set = {canon_stereo(m) for m in true[key]}
    gen_labels = [canon_stereo(m) for m in gen[corrected]]
    n_c, n_db = n_stereo_elements(corrected)
    rows.append(dict(smiles=key, corrected_smiles=corrected, n_chiral_centers=n_c, n_stereo_double=n_db,
                     n_gt_stereoisomers=len(gt_set),
                     frac_gen_stereo_in_gt=float(np.mean([g in gt_set for g in gen_labels])),
                     n_gen_stereoisomers=len(set(gen_labels))))
out = pd.DataFrame(rows)
print(f'{len(out)} molecules')
for sub, name in [(out, 'all'), (out[out.n_chiral_centers > 0], 'with chiral centers'),
                  (out[out.n_stereo_double > 0], 'with stereo double bonds')]:
    if len(sub):
        print(f'{name:>26s}: n={len(sub):5d}  mean frac of generated confs whose 3D stereo matches a GT conformer = '
              f'{sub.frac_gen_stereo_in_gt.mean():.3f}; mols with ANY mismatch = {(sub.frac_gen_stereo_in_gt < 1).mean():.3f}')
if args.out:
    out.to_csv(args.out, index=False)

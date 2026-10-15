"""E/Z and chirality consistency of generated QM9 conformers, split by stereo element (R-CHEM, round 5, Exp 3).

For each run directory (confs.pkl = generated conformers keyed by corrected SMILES; eval.pkl = TD evaluator output with
per-molecule RMSD matrices, rows = cleaned GT conformers in test_mols order, columns = generated conformers in
confs.pkl order; evaluate_confs.py:94-126), and each test molecule:
  * label each conformer (GT and generated) by two stereo-only canonical SMILES perceived from 3D
    (Chem.AssignStereochemistryFrom3D): DB = double-bond stereo only (atom chiral tags cleared), CH = tetrahedral
    stereo only (bond stereo cleared);
  * a generated conformer is "E/Z-wrong" if its DB label is not among the GT conformers' DB labels (same for CH);
  * planarity of every TD-rotatable double bond: deviation of its dihedral from the nearest of 0 / 180 degrees;
  * with eval.pkl: AMR-R / AMR-P of the molecule (a) as evaluated, (b) with E/Z-wrong conformers dropped (post-hoc
    filter WITHOUT replacement; recall can only get worse, precision is what can improve), and how often an E/Z-wrong
    conformer is the recall argmin of some GT conformer.
Reads only its inputs; writes only --out. Run with the TD venv (pickles hold RDKit mols from that RDKit version).

Usage: python ez_flip_analysis.py --test_csv data/QM9/test_smiles.csv --true_mols data/QM9/test_mols.pkl \
         --runs name=dir [name=dir ...] --out out.csv
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolTransforms

RDLogger.DisableLog('rdApp.*')


def clean_confs(smi, confs):
    """Identical to evaluate_confs.py clean_confs: keeps GT conformers whose non-isomeric SMILES matches."""
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    return [c for c in confs
            if Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False) == smi]


def stereo_labels(mol):
    """(DB-only, CH-only) isomeric canonical SMILES of heavy atoms, stereo perceived from the 3D conformer."""
    try:
        m = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(m)
        m = Chem.RemoveHs(m)
        db = Chem.Mol(m)
        for a in db.GetAtoms():
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        ch = Chem.Mol(m)
        for b in ch.GetBonds():
            b.SetStereo(Chem.BondStereo.STEREONONE)
            b.SetBondDir(Chem.BondDir.NONE)
        return Chem.MolToSmiles(db, isomericSmiles=True), Chem.MolToSmiles(ch, isomericSmiles=True)
    except Exception:
        return None, None


def rot_double_bonds(mol):
    """TD-rotatable (acyclic, both sides > 1 atom in the H graph) double bonds as dihedral quadruples (c, a, b, d)."""
    out = []
    n = mol.GetNumAtoms()
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a0, a1 = b.GetBeginAtom(), b.GetEndAtom()
        n0 = [x.GetIdx() for x in a0.GetNeighbors() if x.GetIdx() != a1.GetIdx()]
        n1 = [x.GetIdx() for x in a1.GetNeighbors() if x.GetIdx() != a0.GetIdx()]
        if not n0 or not n1:
            continue  # one side is a single atom (e.g. C=O): not TD-rotatable
        out.append((n0[0], a0.GetIdx(), a1.GetIdx(), n1[0]))
    return out


def planarity_dev(mol, quads):
    """Max over the molecule's rotatable double bonds of the deviation (deg) from the nearest planar value."""
    conf = mol.GetConformer()
    devs = []
    for q in quads:
        phi = abs(rdMolTransforms.GetDihedralDeg(conf, *q))
        devs.append(min(phi, 180.0 - phi))
    return max(devs) if devs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--true_mols', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.test_csv)
    with open(a.true_mols, 'rb') as f:
        true = pickle.load(f)
    gt_cache = {}
    rows = []
    for spec in a.runs:
        name, d = spec.split('=', 1)
        with open(f'{d}/confs.pkl', 'rb') as f:
            gen = pickle.load(f)
        try:
            with open(f'{d}/eval.pkl', 'rb') as f:
                ev = pickle.load(f)['results']
        except FileNotFoundError:
            ev = {}
        for smi, corr in zip(df.smiles.values, df.corrected_smiles.values):
            if corr not in gen or smi not in true:
                continue
            m0 = Chem.MolFromSmiles(corr)
            si = Chem.FindPotentialStereo(m0)
            n_db = sum(1 for s in si if s.type == Chem.StereoType.Bond_Double)
            n_tet = sum(1 for s in si if s.type == Chem.StereoType.Atom_Tetrahedral)
            if smi not in gt_cache:
                gts = clean_confs(corr, true[smi])
                labs = [stereo_labels(c) for c in gts]
                quads_gt = rot_double_bonds(gts[0]) if gts else []
                gt_cache[smi] = (gts, {l[0] for l in labs}, {l[1] for l in labs},
                                 [planarity_dev(c, quads_gt) for c in gts])
            gts, gt_db, gt_ch, gt_pl = gt_cache[smi]
            if not gts:
                continue
            gconfs = gen[corr]
            glabs = [stereo_labels(c) for c in gconfs]
            db_ok = np.array([l[0] in gt_db for l in glabs])
            ch_ok = np.array([l[1] in gt_ch for l in glabs])
            quads = rot_double_bonds(gconfs[0])
            pl = np.array([planarity_dev(c, quads) for c in gconfs]) if quads else np.array([np.nan])
            row = dict(run=name, smiles=smi, corrected_smiles=corr, n_db_stereo=n_db, n_tet=n_tet,
                       n_rot_double=len(quads), n_gt=len(gts), n_gen=len(gconfs),
                       n_gt_db_labels=len(gt_db), n_gt_ch_labels=len(gt_ch),
                       frac_gen_db_ok=db_ok.mean(), frac_gen_ch_ok=ch_ok.mean(),
                       gen_planar_dev_mean=np.nanmean(pl) if quads else np.nan,
                       gen_frac_db_twist_gt20=np.nanmean(pl > 20) if quads else np.nan,
                       gt_planar_dev_mean=np.nanmean(gt_pl) if quads else np.nan)
            r = ev.get((smi, corr))
            if r is not None and r['rmsd'].shape[1] == len(gconfs) and not np.isnan(r['rmsd']).all():
                R = r['rmsd']
                row['amr_r'] = np.nanmin(R, axis=1).mean()
                row['amr_p'] = np.nanmin(R, axis=0).mean()
                row['cov_r05'] = np.mean(np.nanmin(R, axis=1) < 0.5)
                arg = np.nanargmin(R, axis=1)
                row['frac_gt_whose_argmin_is_ez_wrong'] = float(np.mean(~db_ok[arg]))
                if db_ok.any() and (~db_ok).any():
                    Rf = R[:, db_ok]
                    row['amr_r_filtered'] = np.nanmin(Rf, axis=1).mean()
                    row['amr_p_filtered'] = np.nanmin(Rf, axis=0).mean()
                    row['amr_p_wrong_only'] = np.nanmin(R[:, ~db_ok], axis=0).mean()
                else:
                    row['amr_r_filtered'] = row['amr_r']
                    row['amr_p_filtered'] = row['amr_p']
            rows.append(row)
        print('done', name, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    for name, g in out.groupby('run', sort=False):
        s = g[g.n_db_stereo > 0]
        rd = g[g.n_rot_double > 0]
        msg = (f'{name}: mols={len(g)} | stereoDB mols={len(s)}: mean frac gen E/Z-ok={s.frac_gen_db_ok.mean():.3f}, '
               f'mols with any E/Z-wrong={np.mean(s.frac_gen_db_ok < 1):.3f}, gen confs E/Z-wrong='
               f'{np.sum((1 - s.frac_gen_db_ok) * s.n_gen):.0f}/{s.n_gen.sum()}'
               f' | tet mols={int((g.n_tet > 0).sum())}: frac gen CH-ok={g[g.n_tet > 0].frac_gen_ch_ok.mean():.3f}'
               f' | rot-double mols={len(rd)}: gen planarity dev mean={rd.gen_planar_dev_mean.mean():.1f} deg '
               f'(GT {rd.gt_planar_dev_mean.mean():.1f}), frac gen with a double bond twisted >20 deg='
               f'{rd.gen_frac_db_twist_gt20.mean():.3f}')
        if 'amr_r' in s and s.amr_r.notna().any():
            msg += (f' | stereoDB mols AMR-R={s.amr_r.mean():.4f} filtered(no replacement)={s.amr_r_filtered.mean():.4f}'
                    f' AMR-P={s.amr_p.mean():.4f} filtered={s.amr_p_filtered.mean():.4f}; frac GT argmin E/Z-wrong='
                    f'{s.frac_gt_whose_argmin_is_ez_wrong.mean():.3f}; all-mol AMR-R={g.amr_r.mean():.4f}')
        print(msg)


if __name__ == '__main__':
    main()

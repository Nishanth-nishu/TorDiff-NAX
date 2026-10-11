"""
Round-4 leak-free oracle test: seed-provenance check of one split-half generation (ORACLE inputs).

TD changes only torsions of a seed (rigid rotations about rotatable bonds, diffusion/sampling.py perturb_seeds/sample),
so every bond length of generated conformer i equals that of its seed. Under --seed_confs_cycle the seed of conformer i
is seeds[raw_smi][i mod |seeds|] (diffusion/sampling.py embed_seeds). This script asserts both facts on a confs.pkl
(the precedent is round-2 FIXES X12 in tools/local_structure_analysis.py leak_job), checks the conformer count 2 x
n_conformers of the csv, and reports how many generated conformers are bond-length-identical (max |dev| < tol) to a
conformer of the OTHER half too (it should be none: such a conformer could carry other-half -- i.e. own -- L).

Usage:  python tools/check_split_seeds.py --confs RES/<run>/r4lf_split_AtoB_ORACLE/confs.pkl \
            --seeds SPLIT/seeds_A.pkl --other SPLIT/seeds_B.pkl --csv SPLIT/csv_B.csv
"""
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from rdkit.Chem import rdMolTransforms


def bond_lengths(mol, bonds):
    c = mol.GetConformer()
    return np.array([rdMolTransforms.GetBondLength(c, i, j) for i, j in bonds])


def check(confs_path, seeds_path, other_path, csv_path, tol=1e-3):
    with open(confs_path, 'rb') as f:
        confs = pickle.load(f)
    with open(seeds_path, 'rb') as f:
        seeds = pickle.load(f)
    with open(other_path, 'rb') as f:
        other = pickle.load(f)
    df = pd.read_csv(csv_path)
    st = dict(n_mols=0, n_missing=0, n_confs=0, max_dev_own=0.0, n_other_identical=0, min_dev_other=np.inf)
    for raw, n, corr in df[['smiles', 'n_conformers', 'corrected_smiles']].values:
        if corr not in confs:
            st['n_missing'] += 1
            continue
        gens, S, O = confs[corr], seeds[raw], other[raw]
        assert len(gens) == 2 * int(n), f'{corr}: {len(gens)} generated != 2 x {n}'
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in S[0].GetBonds()]
        bS = [bond_lengths(s, bonds) for s in S]
        bO = [bond_lengths(o, bonds) for o in O]
        for i, g in enumerate(gens):
            bg = bond_lengths(g, bonds)
            dev = float(np.abs(bg - bS[i % len(S)]).max())
            assert dev < tol, f'{corr}: generated conformer {i} does not keep the bond lengths of seed {i % len(S)} ' \
                              f'(max dev {dev:.4f} A): seed order lost, split-half test invalid'
            do = min(float(np.abs(bg - b).max()) for b in bO)
            st['max_dev_own'] = max(st['max_dev_own'], dev)
            st['min_dev_other'] = min(st['min_dev_other'], do)
            st['n_other_identical'] += int(do < tol)
        st['n_mols'] += 1
        st['n_confs'] += len(gens)
    return st


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--confs', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--other', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--tol', type=float, default=1e-3)
    a = ap.parse_args()
    st = check(a.confs, a.seeds, a.other, a.csv, a.tol)
    print('SEEDCHECK', ' '.join(f'{k}={v:.5f}' if isinstance(v, float) else f'{k}={v}' for k, v in st.items()))

"""Graph-level stereo / torsion-type census of a TD test csv (R-CHEM, round 5, Exp 3).

Reads a TD-style csv (columns smiles, n_conformers, corrected_smiles) and, per molecule (corrected_smiles, the SMILES
TD generates from), counts
  * TD-rotatable bonds, by chemical type. TD's rule (utils/torsion.py:9-39, get_transformation_mask) is: a bond of
    the H-explicit graph whose removal disconnects the graph and whose smaller side has > 1 atom. No bond-order check.
    Replicated here with RDKit (an acyclic bond of a connected graph is a bridge); --check_td cross-checks the count
    against TD's own get_transformation_mask (needs the TD repo on PYTHONPATH, i.e. run on the cluster venv).
  * stereo elements by RDKit's FindPotentialStereo (new stereo perception): tetrahedral centres and double bonds,
    potential and specified-in-SMILES.
Output: per-molecule csv + a summary printed to stdout. Reads only its inputs; writes only --out.

Usage: python -I stereo_census.py --csv <test csv> --out <per-molecule csv> [--check_td]
"""
import argparse
from collections import Counter

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

CLASSES = ['double_stereo', 'double_nonstereo', 'triple', 'single_amide_like', 'single_conjugated', 'single_other']


def td_rotatable_bonds(mh):
    """Bond indices of mh (explicit Hs) that TD treats as rotatable: acyclic and smaller side > 1 atom."""
    out = []
    n = mh.GetNumAtoms()
    for b in mh.GetBonds():
        if b.IsInRing():
            continue
        a0 = b.GetBeginAtomIdx()
        # BFS from a0 without crossing b
        seen, stack = {a0}, [a0]
        while stack:
            u = stack.pop()
            for nb in mh.GetAtomWithIdx(u).GetNeighbors():
                v = nb.GetIdx()
                if v in seen:
                    continue
                if {u, v} == {b.GetBeginAtomIdx(), b.GetEndAtomIdx()}:
                    continue
                seen.add(v)
                stack.append(v)
        if min(len(seen), n - len(seen)) > 1:
            out.append(b.GetIdx())
    return out


def is_amide_like(b):
    """Single bond C(=O|=S|=N)-N or C(=O)-O (amide, thioamide, amidine, ester/acid): partial double-bond character."""
    for x, y in [(b.GetBeginAtom(), b.GetEndAtom()), (b.GetEndAtom(), b.GetBeginAtom())]:
        if x.GetSymbol() == 'C' and y.GetSymbol() in ('N', 'O'):
            for nb in x.GetBonds():
                if nb.GetIdx() != b.GetIdx() and nb.GetBondType() == Chem.BondType.DOUBLE and \
                        nb.GetOtherAtom(x).GetSymbol() in ('O', 'S', 'N'):
                    return True
    return False


def classify(b, stereo_db):
    bt = b.GetBondType()
    if bt == Chem.BondType.DOUBLE:
        return 'double_stereo' if b.GetIdx() in stereo_db else 'double_nonstereo'
    if bt == Chem.BondType.TRIPLE:
        return 'triple'
    if is_amide_like(b):
        return 'single_amide_like'
    if b.GetIsConjugated():
        return 'single_conjugated'
    return 'single_other'


def census(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None or '.' in smi:
        return None
    si = Chem.FindPotentialStereo(m)
    stereo_db = {s.centeredOn for s in si if s.type == Chem.StereoType.Bond_Double}
    spec_db = {s.centeredOn for s in si if s.type == Chem.StereoType.Bond_Double
               and s.specified == Chem.StereoSpecified.Specified}
    tet = [s for s in si if s.type == Chem.StereoType.Atom_Tetrahedral]
    mh = Chem.AddHs(m)  # AddHs keeps heavy-atom and heavy-bond indices
    rot = td_rotatable_bonds(mh)
    cls = Counter(classify(mh.GetBondWithIdx(i), stereo_db) for i in rot)
    ring_db_stereo = sum(1 for i in stereo_db if m.GetBondWithIdx(i).IsInRing())
    row = dict(corrected_smiles=smi, n_heavy=m.GetNumAtoms(), n_td_rot=len(rot),
               n_db_stereo_potential=len(stereo_db), n_db_stereo_specified=len(spec_db),
               n_db_stereo_in_ring=ring_db_stereo,
               n_tet_potential=len(tet),
               n_tet_specified=sum(1 for s in tet if s.specified == Chem.StereoSpecified.Specified))
    row.update({f'rot_{c}': cls.get(c, 0) for c in CLASSES})
    return row, mh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check_td', action='store_true')
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    rows, n_bad, n_td_mismatch = [], 0, 0
    if a.check_td:
        from utils.featurization import featurize_mol_from_smiles
        from utils.torsion import get_transformation_mask
    for smi in df.corrected_smiles.values:
        r = census(smi)
        if r is None:
            n_bad += 1
            continue
        row, mh = r
        if a.check_td:
            _, data = featurize_mol_from_smiles(smi, dataset='drugs')
            if data is not None:
                em, _ = get_transformation_mask(data)
                row['n_td_rot_td_code'] = int(em.sum())
                n_td_mismatch += int(em.sum()) != row['n_td_rot']
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    n = len(out)
    print(f'csv={a.csv} molecules={n} unparsable_or_multifrag={n_bad}')
    if a.check_td:
        print(f'TD get_transformation_mask count != this replication: {n_td_mismatch} molecules')
    s = lambda c: int((out[c] > 0).sum())
    print(f'mols with >=1 potential stereo double bond: {s("n_db_stereo_potential")} ({s("n_db_stereo_potential")/n:.1%}); '
          f'with >=1 SPECIFIED in SMILES: {s("n_db_stereo_specified")}; stereo DB in rings: '
          f'{int(out.n_db_stereo_in_ring.sum())} bonds')
    print(f'total potential stereo double bonds: {int(out.n_db_stereo_potential.sum())}; specified: '
          f'{int(out.n_db_stereo_specified.sum())}')
    print(f'mols with >=1 potential tetrahedral centre: {s("n_tet_potential")} ({s("n_tet_potential")/n:.1%}); '
          f'with >=1 specified: {s("n_tet_specified")}; total centres {int(out.n_tet_potential.sum())}, '
          f'specified {int(out.n_tet_specified.sum())}')
    tot = int(out.n_td_rot.sum())
    print(f'TD-rotatable bonds total {tot}; mols with >=1: {s("n_td_rot")}')
    for c in CLASSES:
        print(f'  {c:>18s}: {int(out["rot_"+c].sum()):6d} bonds ({out["rot_"+c].sum()/max(tot,1):.1%}), '
              f'in {s("rot_"+c)} mols')
    print(f'mols whose TD-rotatable set contains a stereo double bond: {s("rot_double_stereo")}')


if __name__ == '__main__':
    main()

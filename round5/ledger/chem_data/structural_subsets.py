"""Version-independent Exp 3 subsets (R-CHEM, consults Q-P1-006/007, Q-P2-007, Q-P3-004).

Structural stereo double bond (EXP-3 §3.1): bond order 2, acyclic, both atoms with >= 2 heavy-atom neighbours
(RDKit Book rule, E-CHEM-070, without calling FindPotentialStereo). "Specified" = the SMILES string carries a
directional bond ('/' or '\\') on that double bond's neighbours, read from the parsed bond directions/stereo.
Counts per test csv: molecules with a TD-rotatable double bond (any), with a structural stereo double bond, with a
specified one; molecules with a TD-rotatable triple bond (K1-canonical scope).
Usage: python -I structural_subsets.py <csv>
"""
import sys
import pandas as pd
from rdkit import Chem, RDLogger
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from stereo_census import td_rotatable_bonds  # noqa: E402

RDLogger.DisableLog('rdApp.*')


def heavy_deg(a):
    return sum(1 for n in a.GetNeighbors() if n.GetAtomicNum() > 1)


d = pd.read_csv(sys.argv[1])
n = n_any = n_struct = n_spec = n_trip = n_union = 0
for s in d.corrected_smiles:
    m = Chem.MolFromSmiles(s)
    if m is None or '.' in s:
        continue
    n += 1
    mh = Chem.AddHs(m)
    rot = [mh.GetBondWithIdx(i) for i in td_rotatable_bonds(mh)]
    dbs = [b for b in rot if b.GetBondType() == Chem.BondType.DOUBLE]
    struct = [b for b in dbs if heavy_deg(b.GetBeginAtom()) >= 2 and heavy_deg(b.GetEndAtom()) >= 2]
    spec = [b for b in struct if b.GetStereo() != Chem.BondStereo.STEREONONE]
    trip = [b for b in rot if b.GetBondType() == Chem.BondType.TRIPLE]
    n_any += bool(dbs)
    n_struct += bool(struct)
    n_spec += bool(spec)
    n_trip += bool(trip)
    n_union += bool(dbs or trip)
print(f'{sys.argv[1]}: molecules {n}; TD-rotatable double bond {n_any}; structural stereo DB {n_struct}; '
      f'specified in SMILES {n_spec}; TD-rotatable triple bond {n_trip}; double or triple (K1-canonical scope) {n_union}; '
      f'rdkit {Chem.rdBase.rdkitVersion}')

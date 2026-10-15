"""Break down potential stereo double bonds of a TD test csv by element pair, ring membership and SMILES specification.
Usage: python -I db_breakdown.py <csv>   (reads corrected_smiles; prints counts)"""
import sys
from collections import Counter
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
d = pd.read_csv(sys.argv[1])
c, mols_spec_any, mols_unspec_any = Counter(), 0, 0
for s in d.corrected_smiles:
    m = Chem.MolFromSmiles(s)
    if m is None:
        continue
    sp = un = False
    for si in Chem.FindPotentialStereo(m):
        if si.type != Chem.StereoType.Bond_Double:
            continue
        b = m.GetBondWithIdx(si.centeredOn)
        k = ''.join(sorted([b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()])) + ('_ring' if b.IsInRing() else '')
        spec = si.specified == Chem.StereoSpecified.Specified
        c[(k, 'specified' if spec else 'unspecified')] += 1
        if not b.IsInRing():
            sp |= spec
            un |= not spec
    mols_spec_any += sp
    mols_unspec_any += un
for k, v in sorted(c.items()):
    print(k, v)
print('molecules with >=1 acyclic stereo DB specified:', mols_spec_any, '; with >=1 acyclic unspecified:', mols_unspec_any)
print('rdkit', Chem.rdBase.rdkitVersion)

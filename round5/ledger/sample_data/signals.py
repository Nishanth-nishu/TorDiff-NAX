"""GT-free signals per molecule from the generated pool (R0, R1, R2 order = eval column order):
rotamer strings (each rotatable heavy-atom torsion binned into 3 x 120 deg), Good-Turing singleton fraction,
new-mode rate of batch 2 vs batch 1, torsion diversity, MMFF94 single-point energy statistics.
Writes signals.pkl {corrected_smi: dict(tors=array[Npool, m], rot=strings list, E=array[Npool])}."""
import pickle, sys, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolTransforms as T, AllChem
RDLogger.DisableLog('rdApp.*')
D = sys.argv[1]
runs = ['R0_base_seed0', 'R1_base_seed1', 'R2_base_seed2']
confs = [pickle.load(open(f'{D}/{r}/confs.pkl', 'rb')) for r in runs]
C = pickle.load(open(sys.argv[2], 'rb'))['curves']

def torsion_quads(mol):
    heavy = lambda a: a.GetAtomicNum() > 1
    quads = []
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType() == Chem.BondType.TRIPLE: continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if not (heavy(a1) and heavy(a2)): continue
        n1 = [n.GetIdx() for n in a1.GetNeighbors() if heavy(n) and n.GetIdx() != a2.GetIdx()]
        n2 = [n.GetIdx() for n in a2.GetNeighbors() if heavy(n) and n.GetIdx() != a1.GetIdx()]
        if not n1 or not n2: continue
        # skip linear (sp) centres
        if any(x.GetHybridization() == Chem.HybridizationType.SP for x in (a1, a2)): continue
        quads.append((min(n1), a1.GetIdx(), a2.GetIdx(), min(n2)))
    return quads

out = {}
for key, c in C.items():
    smi = key[1]
    mols = sum([list(cf[smi]) for cf in confs], [])
    if len(mols) != c['N']:
        print('pool size mismatch', smi, len(mols), c['N']); continue
    q = torsion_quads(mols[0])
    tors = np.array([[T.GetDihedralDeg(m.GetConformer(), *t) for t in q] for m in mols]) if q else np.zeros((len(mols), 0))
    rot = [''.join(str(int(((x + 180 + 60) % 360) // 120)) for x in row) for row in tors]
    E = np.full(len(mols), np.nan)
    try:
        props = AllChem.MMFFGetMoleculeProperties(mols[0])
        for i, m in enumerate(mols):
            ff = AllChem.MMFFGetMoleculeForceField(m, props)
            if ff is not None: E[i] = ff.CalcEnergy()
    except Exception as e:
        pass
    out[key] = dict(tors=tors, rot=rot, E=E, n_tors=len(q))
print('signals for', len(out))
pickle.dump(out, open(sys.argv[3], 'wb'))

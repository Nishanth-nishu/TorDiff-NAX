"""V2: time tools/lgeom.set_internal_subset on QM9-sized molecules (smiles from l_error.csv, two ETKDG embeddings)."""
import sys, time, csv, random
sys.path.insert(0, sys.argv[1])  # tools/
import lgeom
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

smis = []
with open(sys.argv[2]) as f:
    r = csv.DictReader(f)
    seen = set()
    for row in r:
        s = row['smiles']
        if s not in seen and row['cond'] == 'L_A5ring_cyc_ORACLE':
            seen.add(s); smis.append(s)
random.seed(0)
pick = random.sample(smis, 20)
times = []
for s in pick:
    m = Chem.AddHs(Chem.MolFromSmiles(s))
    cids = list(AllChem.EmbedMultipleConfs(m, 2, randomSeed=1))
    if len(cids) < 2:
        continue
    X = m.GetConformer(cids[0]).GetPositions(); Y = m.GetConformer(cids[1]).GetPositions()
    base = Chem.Mol(m); base.RemoveAllConformers(); c = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(X): c.SetAtomPosition(i, p.tolist())
    base.AddConformer(c, assignId=True)
    t0 = time.perf_counter()
    for _ in range(10):
        lgeom.set_internal_subset(base, Y, X)
        lgeom.set_internal_subset(base, X, Y)
    times.append((time.perf_counter() - t0) / 20)
print('n_mols', len(times), 'ms/call mean %.2f median %.2f max %.2f' % (1e3 * np.mean(times), 1e3 * np.median(times), 1e3 * max(times)))

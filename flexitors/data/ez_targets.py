"""The committed E/Z target table (EXP-3 §3.1 revision, §9.3; Q-P2-007 answer; round5/DECISION.md, validity layer).

Built once with the pinned RDKit 2026.03.6 and committed. One row per TD-rotatable structural stereo double bond
(``flexitors.geometry.torsion_types.is_structural_stereo``) of every molecule of the given csv files, keyed by the
csv ``corrected_smiles`` (the SMILES TD generates from) and atom indices of ``Chem.AddHs(Chem.MolFromSmiles(...))``:
TD's reference quadruple (c, a, b, d), whether the SMILES specifies the bond (RDKit 2026.03.6 defines "specified",
EXP-3 §9.3), and the planar target φ_b on that quadruple (0 = cis, π = trans; empty if unspecified).

Every specified target is cross-checked against an ETKDG embedding of the same SMILES (ETKDG enforces SMILES double-
bond stereo): ``embed_check`` is ``agree``, ``disagree`` or ``embed_failed``; the builder refuses to write a table
with any disagreement.
"""
from __future__ import annotations

import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem

from flexitors.geometry.internal import dihedral
from flexitors.geometry.torsion_types import structural_stereo_bonds

RDLogger.DisableLog('rdApp.*')
PINNED_RDKIT = '2026.03.6'  # EXP-3 §9.3
EMBED_SEED = 20261016


def embed_check(mh: Chem.Mol, quad: tuple[int, int, int, int], phi: float) -> str:
    """Compare a SMILES target with the dihedral of an ETKDG embedding on the same quadruple."""
    m = Chem.Mol(mh)
    if AllChem.EmbedMolecule(m, randomSeed=EMBED_SEED) != 0:
        return 'embed_failed'
    tau = dihedral(m.GetConformer().GetPositions(), np.array([quad]))[0]
    return 'agree' if math.cos(tau - phi) > 0 else 'disagree'


def table_rows(dataset: str, smiles: str) -> list[dict]:
    """Target-table rows of one molecule (none if it has no structural stereo bond or does not parse)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None or '.' in smiles:
        return []
    mh = Chem.AddHs(m)
    rows = []
    for b in structural_stereo_bonds(mh):
        c, a, bb, d = b.quad
        rows.append(dict(dataset=dataset, corrected_smiles=smiles, bond_idx=b.bond_idx, c=c, a=a, b=bb, d=d,
                         specified=b.phi is not None, phi=b.phi,
                         embed_check=embed_check(mh, b.quad, b.phi) if b.phi is not None else ''))
    return rows


def main() -> None:
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv', nargs='+', required=True, help='dataset=path to a TD-format csv (corrected_smiles)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if rdBase.rdkitVersion != PINNED_RDKIT:
        raise SystemExit(f'the target table is pinned to RDKit {PINNED_RDKIT}, found {rdBase.rdkitVersion}')
    rows, summary = [], []
    for spec in a.csv:
        name, path = spec.split('=', 1)
        smiles = list(dict.fromkeys(pd.read_csv(path).corrected_smiles))
        new = [r for s in smiles for r in table_rows(name, s)]
        T = pd.DataFrame(new, columns=['dataset', 'corrected_smiles', 'bond_idx', 'c', 'a', 'b', 'd', 'specified',
                                       'phi', 'embed_check'])
        summary.append(f'{name}: {len(smiles)} molecules; structural stereo bonds {len(T)} in '
                       f'{T.corrected_smiles.nunique()} molecules; specified {int(T.specified.sum())} in '
                       f'{T[T.specified].corrected_smiles.nunique()} molecules; embed check '
                       f'{T[T.specified].embed_check.value_counts().to_dict()}')
        rows += new
    T = pd.DataFrame(rows)
    if (T.embed_check == 'disagree').any():
        raise SystemExit('SMILES targets disagree with the ETKDG embedding:\n' +
                         T[T.embed_check == 'disagree'].to_string())
    T.to_csv(a.out, index=False)
    Path(a.out).with_suffix('.txt').write_text(f'# E/Z target table, RDKit {rdBase.rdkitVersion}\n' +
                                               '\n'.join(summary) + '\n')
    print('\n'.join(summary))


if __name__ == '__main__':
    main()

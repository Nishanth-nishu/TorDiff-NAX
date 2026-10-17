"""Per-molecule torsion strata of a test-format csv: m = number of TD-rotatable bonds (TD skips the model when
``edge_mask.sum() == 0``, ``generate_confs.py:150-153``) and n_rot_heavy (the rigid stratum, DECISION fix 4).

One implementation of both counts (``flexitors.geometry.torsion_types``), checked against ``tools/breakdown.py``'s
``n_rot_heavy`` column on gnode118 (review P1 S3). Used by the re-score's m = 0 / m > 0 split (review P2 S8a).
"""
from __future__ import annotations

from argparse import ArgumentParser

import pandas as pd
from rdkit import Chem, RDLogger

from flexitors.geometry.torsion_types import heavy_rotatable_bonds, td_rotatable_bonds

RDLogger.DisableLog('rdApp.*')


def strata(csv: pd.DataFrame) -> pd.DataFrame:
    """smiles, corrected_smiles, m (TD-rotatable bonds) and n_rot_heavy per row; unparsable rows get NaN."""
    rows = []
    for r in csv.itertuples(index=False):
        m = Chem.MolFromSmiles(r.corrected_smiles)
        mh = Chem.AddHs(m) if m is not None else None
        rows.append(dict(smiles=r.smiles, corrected_smiles=r.corrected_smiles,
                         m=len(td_rotatable_bonds(mh)) if mh is not None else float('nan'),
                         n_rot_heavy=len(heavy_rotatable_bonds(mh)) if mh is not None else float('nan')))
    return pd.DataFrame(rows)


def main() -> None:
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    S = strata(pd.read_csv(a.csv))
    S.to_csv(a.out, index=False)
    print(f'{len(S)} rows; m = 0: {int((S.m == 0).sum())}; n_rot_heavy = 0: {int((S.n_rot_heavy == 0).sum())}')


if __name__ == '__main__':
    main()

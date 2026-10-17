"""flexitors.data.ez_targets and the committed table round5/data/ez_targets.csv (EXP-3 §9.3; E-CHEM-066)."""
import math

import pandas as pd
import pytest
from rdkit import rdBase

from flexitors.data.ez_targets import PINNED_RDKIT, table_rows
from flexitors.td import REPO_ROOT

TABLE = REPO_ROOT / 'round5' / 'data' / 'ez_targets.csv'


def test_rows_for_hand_cases():
    rows = table_rows('x', 'C/C=C/C(=O)O')
    assert len(rows) == 1 and rows[0]['specified'] and rows[0]['phi'] == pytest.approx(math.pi)
    assert rows[0]['embed_check'] == 'agree'
    (u,) = table_rows('x', 'CC=CC(=O)O')
    assert not u['specified'] and u['phi'] is None and u['embed_check'] == ''
    assert table_rows('x', 'CC=N') == [] and table_rows('x', 'C.C') == []


@pytest.mark.skipif(not TABLE.exists(), reason='table not built yet')
def test_committed_table():
    T = pd.read_csv(TABLE)
    assert not (T.embed_check == 'disagree').any()
    spec = T[T.specified]
    assert set(spec.embed_check) <= {'agree', 'embed_failed'}
    assert set(spec.phi.round(6)) <= {0.0, round(math.pi, 6)} and T[~T.specified].phi.isna().all()
    for ds, n_struct, n_spec in (('qm9_test', 37, 3), ('drugs_test', 140, 126)):  # E-CHEM-066
        D = T[T.dataset == ds]
        assert D.corrected_smiles.nunique() == n_struct and D[D.specified].corrected_smiles.nunique() == n_spec


@pytest.mark.skipif(not TABLE.exists() or rdBase.rdkitVersion != PINNED_RDKIT, reason='pinned RDKit only')
def test_committed_table_is_reproducible():
    T = pd.read_csv(TABLE)
    for ds, g in T.groupby('dataset'):
        for smi, rows in list(g.groupby('corrected_smiles'))[:40]:
            again = pd.DataFrame(table_rows(ds, smi))
            assert list(again.bond_idx) == list(rows.bond_idx)
            assert list(again[['c', 'a', 'b', 'd']].itertuples(index=False)) == \
                list(rows[['c', 'a', 'b', 'd']].itertuples(index=False))


@pytest.mark.skipif(not TABLE.exists(), reason='table not built yet')
@pytest.mark.cluster
def test_table_indices_hold_under_generation_rdkit():
    """review P1 S2 / P2 S5: under the cluster's RDKit (2022.09.5) and TD's featurisation, every row's bond_idx is the
    same structural double bond with the same TD quadruple, and ETKDG honours every specified target"""
    from rdkit import Chem

    from flexitors.data.ez_targets import embed_check
    from flexitors.geometry.torsion_types import structural_stereo_bonds
    from flexitors.td import import_baseline
    feat = import_baseline('utils.featurization')
    T = pd.read_csv(TABLE)
    disagree = []
    for smi, rows in T.groupby('corrected_smiles', sort=False):
        mol, data = feat.featurize_mol_from_smiles(smi, dataset='drugs')
        mh = Chem.AddHs(Chem.MolFromSmiles(smi))
        assert mol is None or mol.GetNumAtoms() == mh.GetNumAtoms()
        now = {b.bond_idx: b.quad for b in structural_stereo_bonds(mh)}
        for r in rows.itertuples():
            assert now.get(r.bond_idx) == (r.c, r.a, r.b, r.d), (smi, r.bond_idx)
            assert mh.GetBondWithIdx(r.bond_idx).GetBondType() == Chem.BondType.DOUBLE
            if r.specified and embed_check(mh, (r.c, r.a, r.b, r.d), r.phi) == 'disagree':
                disagree.append((smi, r.bond_idx))
    assert disagree == [], disagree

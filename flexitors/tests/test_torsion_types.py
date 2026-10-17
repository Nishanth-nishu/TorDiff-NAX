"""flexitors.geometry.torsion_types (EXP-3 §3.1, §6 tests 1-2; Q-P2-007 answer; plan_P2 V-E3-1)."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from flexitors.geometry.internal import dihedral
from flexitors.geometry.torsion_types import (is_structural_stereo, smiles_phi_target, structural_stereo_bonds,
                                              td_reference_quadruple, td_rotatable_bonds)

REPO = Path(__file__).resolve().parents[2]
CHEM = REPO / 'round5' / 'ledger' / 'chem_data'


def mh(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def qm9_sample(n=60):
    df = pd.read_csv(CHEM / 'qm9_test_smiles_from_local_structure_test.csv')
    smi = [s for s in df.corrected_smiles if Chem.MolFromSmiles(s) is not None and '.' not in s]
    dbl = [s for s in smi if any(b.GetBondType() == Chem.BondType.DOUBLE and not b.IsInRing()
                                 for b in Chem.MolFromSmiles(s).GetBonds())]
    return smi[:n] + dbl[:40]


def census_prototype():
    spec = importlib.util.spec_from_file_location('stereo_census', CHEM / 'stereo_census.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rotatable_set_equals_validated_prototype():
    # stereo_census.td_rotatable_bonds was validated against TD's get_transformation_mask with 0/996 mismatches
    proto = census_prototype()
    for s in qm9_sample():
        m = mh(s)
        assert td_rotatable_bonds(m) == proto.td_rotatable_bonds(m), s


def test_reference_quadruple_is_a_path_through_the_bond():
    for s in qm9_sample():
        m = mh(s)
        for k in td_rotatable_bonds(m):
            c, a, b, d = td_reference_quadruple(m, k)
            bond = m.GetBondWithIdx(k)
            assert {a, b} == {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            assert m.GetBondBetweenAtoms(c, a) and m.GetBondBetweenAtoms(b, d) and c != b and d != a


def test_structural_test():
    cases = {'C/C=C/C': True, 'CC=N': False, 'C=CC': False, 'C/C=N/O': True, 'C1=CCCC1': False,
             '[H]/N=C(\\C)N': False, 'CC=C(C)C': True}
    for s, want in cases.items():
        m = mh(s)
        got = any(is_structural_stereo(m.GetBondWithIdx(k)) for k in td_rotatable_bonds(m))
        assert got is want, s


def test_phi_target_hand_cases():
    # but-2-ene: TD's reference atoms are the two methyl carbons (first neighbours in bond order)
    for s, want in (('C/C=C/C', math.pi), ('C/C=C\\C', 0.0), ('CC=CC', None)):
        bonds = structural_stereo_bonds(mh(s))
        assert len(bonds) == 1 and bonds[0].quad[0] in (0, 3) and bonds[0].quad[3] in (0, 3)
        assert bonds[0].phi == want, s
    # reference-atom swap: in C(C)(/F)=C/C the SMILES stereo atom on C0 is F2, but TD's reference atom is C1 (first
    # neighbour in bond order), so the target must be toggled relative to the written F/C relation
    m = mh('C(C)(/F)=C/C')
    (b,) = structural_stereo_bonds(m)
    assert b.quad[0] == 1 or b.quad[3] == 1
    assert set(m.GetBondWithIdx(b.bond_idx).GetStereoAtoms()) & {2}
    assert b.phi == math.pi  # F2 cis to C4 (Z) => the reference methyl C1 is trans to C4: one toggle


@pytest.mark.parametrize('smiles', ['C/C=C/C', 'C/C=C\\C', 'C/C(F)=C/C', 'C/C(F)=C\\C', 'F/C(C)=C/CC', 'C/C=N/O',
                                    'C/C=N\\O', 'CC/C(C)=C(\\C)O', 'OC/C=C/C(=O)O', 'N#C/C=C\\C=O',
                                    'C/N=C(\\C)CC', 'O/N=C(\\C)CC', 'C(C)(/F)=C/C'])
def test_phi_target_agrees_with_embedding(smiles):
    m = mh(smiles)
    assert AllChem.EmbedMolecule(m, randomSeed=11) == 0
    pos = m.GetConformer().GetPositions()
    for b in structural_stereo_bonds(m):
        assert b.phi is not None
        tau = dihedral(pos, np.array([b.quad]))[0]
        assert math.cos(tau - b.phi) > 0.9, (smiles, b, tau)


def test_unspecified_has_no_target():
    m = mh('CC=CC')
    (k,) = [b for b in td_rotatable_bonds(m) if m.GetBondWithIdx(b).GetBondType() == Chem.BondType.DOUBLE]
    assert smiles_phi_target(m, k, td_reference_quadruple(m, k)) is None


@pytest.mark.skipif(rdBase.rdkitVersion != '2026.03.6', reason='counts of E-CHEM-066 are pinned to RDKit 2026.03.6')
def test_structural_counts_reproduce_e_chem_066():
    for csv, n_struct, n_spec in (('qm9_test_smiles_from_local_structure_test.csv', 37, 3),
                                  ('drugs_test_smiles_corrected_geomol.csv', 140, 126)):
        df = pd.read_csv(CHEM / csv)
        s_cnt = p_cnt = 0
        for s in df.corrected_smiles:
            m = Chem.MolFromSmiles(s)
            if m is None or '.' in s:
                continue
            bonds = structural_stereo_bonds(Chem.AddHs(m))
            s_cnt += bool(bonds)
            p_cnt += any(b.phi is not None for b in bonds)
        assert (s_cnt, p_cnt) == (n_struct, n_spec), csv


@pytest.mark.cluster
def test_matches_td_mask_and_sampling_quadruples():
    """TD's own code: get_transformation_mask (utils/torsion.py:9-39) and the dihedral construction of
    diffusion/sampling.py:129-142, re-run here verbatim on TD's featurisation."""
    from flexitors.td import import_baseline
    feat = import_baseline('utils.featurization')
    tors = import_baseline('utils.torsion')
    for s in qm9_sample():
        mol, data = feat.featurize_mol_from_smiles(s, dataset='qm9')
        if data is None:
            continue
        edge_mask, _ = tors.get_transformation_mask(data)
        edge_index = data.edge_index
        edge_list = [[] for _ in range(int(edge_index.max()) + 1)]
        for p in edge_index.T:
            edge_list[p[0]].append(p[1])
        td_quads = []
        for i, p in enumerate(edge_index.T):
            if edge_mask[i]:
                a, b = p[0], p[1]
                c = edge_list[a][0] if edge_list[a][0] != b else edge_list[a][1]
                d = edge_list[b][0] if edge_list[b][0] != a else edge_list[b][1]
                td_quads.append((int(c), int(a), int(b), int(d)))
        ours = [td_reference_quadruple(mol, k) for k in td_rotatable_bonds(mol)]
        assert ours == td_quads, s


def test_heavy_rotatable_count_hand_cases():
    # tools/breakdown.py:32-50 n_rot_heavy: TD-rotatable and >= 2 heavy atoms on both sides
    from flexitors.geometry.torsion_types import heavy_rotatable_bonds
    for s, want in (('CCCC', 1), ('CCO', 0), ('CC(C)C', 0), ('CCCO', 1), ('OCC1CC1', 1), ('C1CC1', 0),
                    ('CC(=O)NC', 1), ('CCC(=O)NCC', 3), ('C#CC#C', 1)):
        assert len(heavy_rotatable_bonds(mh(s))) == want, s

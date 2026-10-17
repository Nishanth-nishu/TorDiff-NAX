"""flexitors.data.gate0_baselines (EXP-1 §10.3; E-JOINT-004 rev., E-JOINT-007)."""
import json

import pandas as pd
import pytest

from flexitors.data.gate0_baselines import largest_ring, ring_seeds, seeds_to_confs, summarise, thresholds
from flexitors.td import REPO_ROOT

TEST_L_ERROR = REPO_ROOT / 'cluster_sync' / 'round2' / 'data' / 'QM9' / 'round2_seeds' / 'l_error.csv'


def test_seeds_rekeyed_by_corrected_smiles():
    csv = pd.DataFrame(dict(smiles=['a', 'b', 'c'], n_conformers=[1, 1, 1], corrected_smiles=['A', 'B', 'C']))
    out = seeds_to_confs({'a': [1], 'c': [3]}, csv)
    assert out == {'A': [1], 'C': [3]}  # 'b' has no seeds: absent, i.e. a model failure for the evaluator


def test_largest_ring():
    assert [largest_ring(s) for s in ('CCO', 'C1CC1', 'C1CC2CC1C2', 'c1ccccc1C1CC1', '[CH]1C[N]1')] == [0, 3, 5, 6, 3]


def test_thresholds_scale_only_beyond_ten_percent():
    test = {'rigid_amr_r': {'ETKDG': 0.152}, 'ring_share_gt10': {'ETKDG': 0.496}}
    t = thresholds({'rigid_amr_r': {'ETKDG': 0.160}, 'ring_share_gt10': {'ETKDG': 0.60}}, test)
    assert not t['rigid_amr_r']['scaled'] and t['rigid_amr_r']['threshold'] == 0.10
    assert t['ring_share_gt10']['scaled'] and t['ring_share_gt10']['threshold'] == pytest.approx(0.20 * 0.60 / 0.496)


def test_summary_on_synthetic_inputs(tmp_path):
    rows = [dict(cond=c, smiles=s, k=k, src_gt_idx=k, ring_dihedral_rmsd=v)
            for c in ('L_etkdg2L', 'L_etkdg2L_mmff') for s, k, v in
            (('C1CCC1', 0, 20.0), ('C1CCC1', 1, 5.0), ('C1CC1', 0, 0.0), ('CCO', 0, float('nan')))]
    pd.DataFrame(rows).to_csv(tmp_path / 'le.csv', index=False)
    for name in ('val', 'test'):
        for cond in ('L_etkdg2L', 'L_etkdg2L_mmff'):
            d = tmp_path / f'{name}_{cond}'
            d.mkdir()
            pd.DataFrame(dict(n_rot_heavy=[0, 0, 2], mat_r=[0.1, 0.2, 0.9])).to_csv(d / 'breakdown.csv', index=False)
    res = summarise(tmp_path, tmp_path / 'le.csv', tmp_path / 'le.csv')
    assert res['val']['rigid_amr_r']['ETKDG'] == pytest.approx(0.15) and res['val']['n_rigid']['MMFF'] == 2
    assert res['val']['ring_share_gt10']['ETKDG'] == pytest.approx(0.5) and res['val']['n_ring_seeds']['ETKDG'] == 2
    assert len(pd.read_csv(tmp_path / 'val_ring_seeds.csv')) == 2
    J = json.loads((tmp_path / 'gate0_baselines.json').read_text())
    # like-for-like scaling: val and test come from the same recipe here, so the ratio is exactly 1 (review P2 B2)
    assert J['gate0_thresholds']['rigid_amr_r']['val_over_test'] == pytest.approx(1.0)
    assert not J['gate0_thresholds']['ring_share_gt10']['scaled']
    rc = J['recipe_check']['rigid_amr_r']['ETKDG']
    assert rc['rel_diff'] == pytest.approx(0.15 / 0.152 - 1) and not rc['flagged']
    assert J['recipe_check']['rigid_amr_r']['MMFF']['flagged']  # 0.15 vs published 0.128: > 10 %


@pytest.mark.skipif(not TEST_L_ERROR.exists(), reason='l_error.csv is not committed (size)')
def test_ring_baseline_reproduces_published_test_values():
    E = pd.read_csv(TEST_L_ERROR, usecols=['cond', 'smiles', 'k', 'src_gt_idx', 'ring_dihedral_rmsd'])
    et, mm = ring_seeds(E, 'L_etkdg2L'), ring_seeds(E, 'L_etkdg2L_mmff')
    assert len(et) == 4181
    assert round(100 * (et.ring_dihedral_rmsd > 10).mean(), 1) == 49.6
    assert round(100 * (mm.ring_dihedral_rmsd > 10).mean(), 1) == 35.1

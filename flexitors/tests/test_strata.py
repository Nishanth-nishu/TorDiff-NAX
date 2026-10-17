"""flexitors.data.strata and the single n_rot_heavy implementation (review P1 S3)."""
import os
from pathlib import Path

import pandas as pd
import pytest

from flexitors.data.strata import strata

RES = Path(os.environ.get('FLEXITORS_RES', '/scratch/nishanth.r/tordiff/results'))
BREAKDOWN = RES / 'qm9_CTRL_rematch_100ep_e100_s0' / 'steps20_seed0' / 'breakdown.csv'


def test_counts():
    csv = pd.DataFrame(dict(smiles=['a', 'b', 'c'], n_conformers=1, corrected_smiles=['CCCC', 'C1CC1', 'CCO']))
    S = strata(csv)
    assert S.m.tolist() == [3, 0, 2] and S.n_rot_heavy.tolist() == [1, 0, 0]


@pytest.mark.cluster
def test_n_rot_heavy_equals_breakdown_py():
    """tools/breakdown.py:32-50 parses argv at import, so compare against its per-molecule output of a real run"""
    if not BREAKDOWN.exists():
        pytest.skip(f'{BREAKDOWN} not found')
    B = pd.read_csv(BREAKDOWN)
    S = strata(B[['smiles', 'corrected_smiles']].assign(n_conformers=1))
    assert len(B) > 900
    assert (S.n_rot_heavy.values == B.n_rot_heavy.values).all()
    assert (S.m.values == B.n_rot_repo.values).all()  # breakdown's n_rot_repo is TD's rotatable count

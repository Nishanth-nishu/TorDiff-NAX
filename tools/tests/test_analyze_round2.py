"""Local test of tools/analyze_round2.py on synthetic eval.pkl files (FIXES X10): runs end to end, Holm columns only on
the pre-declared primary / secondary rows, S4 subset re-scoring restricts the molecules, missing arms are reported."""
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_eval(path, mols, shift, rng, drop=()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    res = {(m, m): {'rmsd': np.abs(rng.normal(0.2 + shift, 0.1, size=(3, 6)))} for m in mols if m not in drop}
    with open(path, 'wb') as f:
        pickle.dump({'results': res, 'num_failures': len(drop)}, f)


def test_driver(tmp_path):
    rng = np.random.default_rng(0)
    mols = [f'M{i}' for i in range(40)]
    res = tmp_path / 'res'
    runs = {'qm9_CTRL_base_100ep_e100_s{}': 0.0, 'qm9_CTRL_rematch_100ep_e100_s{}': 0.0,
            'qm9_B1_train_gtL_e100_s{}': 0.05, 'qm9_B6_jit0.04pa_e100_s{}': 0.0, 'qm9_S4_lamcond_e100_s{}': -0.02}
    for r, sh in runs.items():
        for s in (0, 1, 2):
            for tag in ('steps20_seed0', 'steps20_seed0_gtLcycle', 'S1_lam0.50_cyc_ORACLE'):
                write_eval(str(res / r.format(s) / tag / 'eval.pkl'), mols, sh, rng, drop=('M0',) if s == 1 else ())
    seeds = tmp_path / 'seeds'
    seeds.mkdir()
    (seeds / 'S4_test_subset.txt').write_text('\n'.join(mols[:20]) + '\n')
    pd.DataFrame({'cond': ['L_lam0.50_cyc_ORACLE'] * 3, 'smiles': mols[:3], 'bond_rmsd_all_any': [0.01] * 3,
                  'angle_rmsd_all_any': [1.0] * 3, 'bond_rmsd_heavy_ring': [0.01] * 3, 'angle_rmsd_heavy_ring': [1.0] * 3,
                  'angle_rmsd_heavy_acyc': [1.0] * 3, 'ring_dihedral_rmsd': [2.0] * 3}).to_csv(seeds / 'l_error.csv',
                                                                                              index=False)
    pd.DataFrame({'smiles': mols, 'n_conformers': 3, 'corrected_smiles': mols}).to_csv(tmp_path / 't.csv', index=False)
    out = tmp_path / 'out'
    r = subprocess.run([sys.executable, os.path.join(TOOLS, 'analyze_round2.py'), '--res', str(res), '--seeds',
                        str(seeds), '--test_csv', str(tmp_path / 't.csv'), '--out', str(out), '--n_boot', '300'],
                       capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    assert r.returncode == 0
    C = pd.read_csv(out / 'contrasts.csv')
    ok = C[C.status == 'ok']
    assert set(ok[ok.role == 'primary'].metric) == {'MAT-R', 'COV-R@0.5'}
    assert set(ok[ok.role == 'secondary'].metric) == {'COV-R@0.1'}
    assert ok[ok.role == 'primary']['p_holm_primary'].notna().all()
    assert ok[ok.role == 'descriptive'].get('p_holm_primary', pd.Series(dtype=float)).isna().all()
    sub = ok[(ok.contrast == 'S4.rdkit_l0_vs_CR@S4subset') & (ok.metric == 'MAT-R') & (ok.universe == 'union')]
    full = ok[(ok.contrast == 'S4.rdkit_l0_vs_CR') & (ok.metric == 'MAT-R') & (ok.universe == 'union')]
    # MAT-R uses molecules finite in every seed (M0 failed in one seed), as tools/paired_compare.py
    assert sub.n.iloc[0] == 19 and full.n.iloc[0] == 39                 # re-scoring on the S4 subset
    inter = ok[(ok.contrast == 'S4.rdkit_l0_vs_CR') & (ok.metric == 'COV-R@0.1') & (ok.universe == 'intersection')]
    assert inter.n.iloc[0] == 39                                         # success intersection drops M0
    s2 = ok[(ok.contrast == 'S2.rdkit_vs_CB') & (ok.metric == 'MAT-R') & (ok.universe == 'union')]
    assert 'noninf_upper95_hier' in s2 and np.isfinite(s2.noninf_upper95_hier.iloc[0])
    assert (C[C.contrast == 'S3.rdkit_vs_CR'].status == 'missing').all()  # no S3 files -> reported missing
    assert ok.arm_seed_sd.notna().any()
    assert (out / 'dose_response_vs_measured_L_error.csv').exists()

"""flexitors.data.qm9_val (EVAL §1.1; Q-P2-010 acceptance test) on a synthetic raw-GEOM directory."""
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from flexitors.data import qm9_val
from flexitors.td import REPO_ROOT

SMILES = ['CCO', 'CCCO', 'CC(C)O', 'CCN', 'C1CC1O', 'CC=O', 'OCC=O', 'CCOC', 'NCCO', 'CC(N)=O', 'C1CCC1', 'CCC#N',
          'CN(C)C', 'OC1CCO1', 'CC(O)CO', 'C#CCO']
BROKEN = 'C1CC1N'  # its only conformer has another graph: excluded_no_clean_gt


def record(smiles, n, seed, geom_smiles=None):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    cids = list(AllChem.EmbedMultipleConfs(m, n, randomSeed=seed))
    return {'smiles': geom_smiles or smiles, 'conformers': [{'rd_mol': Chem.Mol(m, confId=c)} for c in cids]}


@pytest.fixture(scope='module')
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp('qm9raw')
    rd = d / 'qm9'
    rd.mkdir()
    keys = sorted(SMILES + [BROKEN])
    for i, s in enumerate(keys):
        rec = record('CCCN' if s == BROKEN else s, 1 + i % 3, 10 + i, geom_smiles=s)
        with open(rd / f'{s}.pickle', 'wb') as f:
            pickle.dump(rec, f)
    stems = qm9_val.raw_stems(rd)
    assert stems == sorted(keys)
    b = stems.index(BROKEN)
    rest = [i for i in range(len(stems)) if i != b]
    split = np.empty(3, dtype=object)  # split 1 = the broken molecule, 8 good ones and 999 (out of range, dropped)
    split[0], split[1], split[2] = np.array(rest[:4]), np.array(rest[4:12] + [b, 999]), np.array(rest[12:])
    np.save(d / 'split.npy', split, allow_pickle=True)
    # test files built the way TD's are: stem, stored count, GEOM smiles; GT = every raw conformer
    rows, mols = [], {}
    for i in split[2]:
        rec = qm9_val.load_raw(rd, stems[i])
        rows.append((stems[i], len(rec['conformers']), rec['smiles']))
        mols[stems[i]] = [c['rd_mol'] for c in rec['conformers']]
    pd.DataFrame(rows, columns=['smiles', 'n_conformers', 'corrected_smiles']).to_csv(d / 'test.csv', index=False)
    with open(d / 'test_mols.pkl', 'wb') as f:
        pickle.dump(mols, f)
    return d, rd, stems, split


def test_self_check_reproduces_test_files(raw):
    d, rd, _, _ = raw
    rule, S = qm9_val.self_check(d / 'test.csv', d / 'test_mols.pkl', rd)
    assert S.reproduced.all() and S.attrs['scores'][rule] == 1.0
    assert S.gt_positions_equal.all()


def test_build_order_filters_and_format(raw, tmp_path):
    d, rd, stems, split = raw
    val, W, mols = qm9_val.build(rd, d / 'split.npy', tmp_path, 'geom_if_any_graph', n=5, seed=0)
    order = np.random.default_rng(0).permutation(sorted(split[1]))
    walked = [int(x) for x in W.split_value]
    assert walked == [int(x) for x in order[:len(walked)]]  # walk = prefix of the seed-0 permutation
    assert len(val) == 5 and list(val.columns) == ['smiles', 'n_conformers', 'corrected_smiles']
    kept = W[W.status == 'kept']
    assert list(kept.smiles) == list(val.smiles)
    assert set(W.status) <= {'kept', 'excluded_no_clean_gt', 'index_out_of_range', 'excluded_unparsable'}
    for r in val.itertuples():
        assert len(mols[r.smiles]) == r.n_conformers  # uncapped raw conformers
    assert BROKEN not in set(val.smiles)
    nested, _, _ = qm9_val.build(rd, d / 'split.npy', tmp_path / 'more', 'geom_if_any_graph', n=7, seed=0)
    assert list(nested.smiles[:5]) == list(val.smiles)  # EVAL §1.1 item 6: extensions are nested
    with open(tmp_path / 'val_mols.pkl', 'rb') as f:
        assert set(pickle.load(f)) == set(val.smiles)


def test_broken_molecule_is_excluded_with_reason(raw, tmp_path):
    d, rd, stems, split = raw
    # rule 'geom' keeps the GEOM smiles, whose graph no conformer has; 'geom_if_any_graph' falls back to conformer 0
    _, W, _ = qm9_val.build(rd, d / 'split.npy', tmp_path, 'geom', n=8, seed=0)
    st = W.set_index('split_value').status
    assert st[stems.index(BROKEN)] == 'excluded_no_clean_gt'
    assert len(W) == 10 or st.get(999) is None  # walked past both exclusions or stopped before 999
    with pytest.raises(RuntimeError):
        qm9_val.build(rd, d / 'split.npy', tmp_path, 'geom', n=9, seed=0)
    val, _, _ = qm9_val.build(rd, d / 'split.npy', tmp_path, 'geom_if_any_graph', n=9, seed=0)
    assert val.set_index('smiles').loc[BROKEN, 'corrected_smiles'] == 'CCCN'


def test_cli_end_to_end(raw, tmp_path):
    d, rd, _, _ = raw
    out = tmp_path / 'cli'
    subprocess.run([sys.executable, '-m', 'flexitors.data.qm9_val', '--raw_dir', str(rd), '--split',
                    str(d / 'split.npy'), '--test_csv', str(d / 'test.csv'), '--test_mols', str(d / 'test_mols.pkl'),
                    '--out_dir', str(out), '--n', '5', '--train_scaffolds'], check=True, cwd=REPO_ROOT,
                   capture_output=True)
    txt = (out / 'self_check.txt').read_text()
    assert 'PASS' in txt and (out / 'population.txt').exists() and (out / 'provenance.json').exists()
    assert 'scaffold-novel' in (out / 'population.txt').read_text()

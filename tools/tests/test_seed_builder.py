"""
Local test of tools/make_l_seed_pickles.py and tools/build_paired_pickles.py on a synthetic test set
(code_plan_2 §9.1 T5/T9; DECISION D1, D2, defects 1, 3, 4). Needs rdkit + numpy + scipy + pandas + the `rmsd` package
(utils/standardization.py imports it). Run from the repo root:  python -m pytest -q tools/tests/test_seed_builder.py
"""
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(TOOLS)
TD = os.path.join(ROOT, 'torsional-diffusion')
sys.path.insert(0, TOOLS)
import lgeom  # noqa: E402

SMILES = ['CCOC(=O)CCN', 'OCC1CC1C=O', 'CC(C)C(N)=O', 'CC1(C)CC1O', 'C1CC2CC1O2']


def fake_gt(smi, n, seed):
    # one fixed stereoisomer for all GT conformers (as GEOM: every conformer of a molecule shares its stereo)
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
    for iso in EnumerateStereoisomers(Chem.MolFromSmiles(smi)):  # first isomer that embeds (bridged rings)
        m = Chem.AddHs(iso)
        cids = list(AllChem.EmbedMultipleConfs(m, n, randomSeed=seed))
        if len(cids) == n:
            break
    AllChem.MMFFOptimizeMoleculeConfs(m, mmffVariant='MMFF94s')
    return [Chem.Mol(m, confId=c) for c in cids]


@pytest.fixture(scope='module')
def fake_set(tmp_path_factory):
    d = tmp_path_factory.mktemp('fake')
    rows, true_mols = [], {}
    for i, s in enumerate(SMILES):
        gts = fake_gt(s, 3, 100 + i)
        true_mols[s] = gts
        rows.append((s, 3, s))
    pd.DataFrame(rows, columns=['smiles', 'n_conformers', 'corrected_smiles']).to_csv(d / 'test.csv', index=False)
    with open(d / 'true.pkl', 'wb') as f:
        pickle.dump(true_mols, f)
    out = d / 'seeds'
    r = subprocess.run([sys.executable, os.path.join(TOOLS, 'make_l_seed_pickles.py'), '--test_csv', str(d / 'test.csv'),
                        '--true_mols', str(d / 'true.pkl'), '--out_dir', str(out), '--sources', 'etkdg,noise,interp'],
                       cwd=TD, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0
    return d, out, true_mols


def load(out, tag):
    with open(out / f'{tag}.pkl', 'rb') as f:
        return pickle.load(f)


def test_lengths_graphs_single_conformer(fake_set):
    d, out, true_mols = fake_set
    for tag, factor in (('L_etkdg2L', 2), ('L_etkdg2L_mmff', 2), ('L_noise0.04pa_cyc_ORACLE', 2),
                        ('L_lam0.50_cyc_ORACLE', 1), ('L_A5ring_cyc_ORACLE', 1), ('L_A5acyc_cyc_ORACLE', 1)):
        P = load(out, tag)
        assert len(P) >= (4 if 'lam' not in tag and 'A5' not in tag else 2), tag
        for smi, seeds in P.items():
            assert len(seeds) == factor * len(true_mols[smi]), tag
            assert all(s.GetNumConformers() == 1 for s in seeds)
            assert lgeom.same_graph(seeds + [true_mols[smi][0]])
    assert set(load(out, 'L_etkdg2L')) == set(load(out, 'L_etkdg2L_mmff'))


def test_lambda_endpoints_and_noise_crn(fake_set):
    d, out, true_mols = fake_set
    l1, l0 = load(out, 'L_lam1.00_cyc_ORACLE'), load(out, 'L_lam0.00_cyc_ORACLE')
    n2, n4 = load(out, 'L_noise0.02pa_cyc_ORACLE'), load(out, 'L_noise0.04pa_cyc_ORACLE')
    for smi in l1:
        for j, (a, g) in enumerate(zip(l1[smi], true_mols[smi])):
            # lambda = 1 is the GT conformer up to a rigid motion + terminal relabelling: heavy RMSD ~ 0
            h = lgeom.heavy_idx(g)
            Y = lgeom.positions(a)
            G = lgeom.positions(g)  # (up to a heavy-atom graph automorphism, chosen by align_pair)
            assert min(lgeom.rmsd_on(Y, lgeom.kabsch_fit(Y, G[pm], h), h)
                       for pm in lgeom.heavy_automorphism_perms(g)) < 1e-4
        for j, g in enumerate(true_mols[smi]):
            G = lgeom.positions(g)
            e2, e4 = lgeom.positions(n2[smi][j]) - G, lgeom.positions(n4[smi][j]) - G
            assert np.allclose(e4, 2 * e2, atol=1e-5)  # common random numbers across sigma (RDKit pickles coords as float32)
            assert not np.allclose(lgeom.positions(n4[smi][j + len(true_mols[smi])]) - G, e4)  # 2 replicates differ


def test_ring_partition_and_l_error_table(fake_set):
    d, out, _ = fake_set
    E = pd.read_csv(out / 'l_error.csv')
    ring = E[E.cond == 'L_A5ring_cyc_ORACLE']
    assert ring['bond_rmsd_all_ring'].fillna(0).max() < 1e-6 and ring['ring_dihedral_rmsd'].fillna(0).max() < 1e-6
    lam = E[E.cond.str.startswith('L_lam')].groupby('cond')['angle_rmsd_all_any'].mean()
    assert lam['L_lam1.00_cyc_ORACLE'] < 1e-6
    assert lam['L_lam0.00_cyc_ORACLE'] >= lam['L_lam0.50_cyc_ORACLE'] >= lam['L_lam1.00_cyc_ORACLE']


def test_paired_builder_on_fake_std_pickles(fake_set, tmp_path):
    """defect 1: a duplicated raw conformer (same geom_id/energy) must be disambiguated, not abort the build"""
    d, out, true_mols = fake_set
    sys.path.insert(0, TD)
    from utils.standardization import optimize_rotatable_bonds, get_torsion_angles
    raw_dir, std_dir = tmp_path / 'raw', tmp_path / 'std'
    raw_dir.mkdir(); std_dir.mkdir()
    std = {}
    for i, smi in enumerate(SMILES[:3]):
        gts = true_mols[smi]
        confs = [dict(geom_id=100 * i + k, totalenergy=-1.0 * k, boltzmannweight=1.0 / len(gts), rd_mol=g)
                 for k, g in enumerate(gts)]
        confs.append(dict(confs[0]))  # exact duplicate -> must be disambiguated
        with open(raw_dir / f'{smi}.pickle', 'wb') as f:
            pickle.dump(dict(conformers=confs, smiles=smi), f)
        rd = Chem.Mol(gts[0]); rd.RemoveAllConformers()
        AllChem.EmbedMultipleConfs(rd, len(gts), randomSeed=3)
        rot = get_torsion_angles(rd)
        new = []
        for k, g in enumerate(gts):
            single = Chem.Mol(rd, confId=k)
            if rot:
                optimize_rotatable_bonds(single, g, rot, popsize=15, maxiter=15)
            c = dict(confs[k]); c['rd_mol'] = single
            c['rmsd'] = AllChem.AlignMol(lgeom.REMOVE_HS(single), lgeom.REMOVE_HS(g))
            new.append(c)
        std[smi] = dict(conformers=new, smiles=smi)
    with open(std_dir / '000.pickle', 'wb') as f:
        pickle.dump(std, f)
    po = tmp_path / 'paired'
    r = subprocess.run([sys.executable, os.path.join(TOOLS, 'build_paired_pickles.py'), '--std_dir', str(std_dir),
                        '--raw_dir', str(raw_dir) + os.sep, '--out_dir', str(po)], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    assert r.returncode == 0 and 'PAIRING_CLEAN' in r.stdout
    with open(po / '000.pickle', 'rb') as f:
        P = pickle.load(f)
    n = sum(len(v['conformers']) for v in P.values())
    assert n == 9  # every verified pair kept (pair_ok failures are flagged, not dropped; DECISION D1 drop is in S4)
    for v in P.values():
        for c in v['conformers']:
            X, Y = lgeom.positions(c['rd_mol']), c['gt_pos_aligned']
            h = lgeom.heavy_idx(c['rd_mol'])
            # aligned frame reproduces the stored rmsd, or improves on it via a graph automorphism
            assert lgeom.rmsd_on(X, Y, h) <= c['rmsd'] + 1e-4 and 'pair_ok' in c

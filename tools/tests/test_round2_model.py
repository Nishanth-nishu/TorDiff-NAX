"""
Cluster CPU tests of the round-2 TD changes (need torch_geometric + e3nn; code_plan_2 §9.2 C1-C10 plus the golden /
byte-identity tests grafted from code_plan_1 §9.2 M1, M2, M8, M9; DECISION D9, defect 6).

Run inside the venv on gnode118 (no GPU needed):
  R1_SNAPSHOT=$PROJECT/tmp/r1_snapshot/torsional-diffusion REPO=$REPO \
  PRETRAINED=$WORK/pretrained/qm9_default python -m pytest -q -p no:cacheprovider $TOOLS/tests/test_round2_model.py
R1_SNAPSHOT = an unpatched copy (git archive) of the round-1 checkout, used as the golden reference.
"""
import copy
import os
import pickle
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

REPO = os.environ.get('REPO', os.getcwd())
SNAP = os.environ.get('R1_SNAPSHOT')
PRE = os.environ.get('PRETRAINED')
TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, TOOLS)
os.chdir(REPO)

from utils.dataset import TorsionNoiseTransform, ConformerDataset, interp_x_torch  # noqa: E402
from utils.featurization import featurize_mol_from_smiles  # noqa: E402
from utils.torsion import get_transformation_mask  # noqa: E402
from utils.utils import get_model  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.data import Dataset  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from argparse import Namespace  # noqa: E402
import lgeom  # noqa: E402

ARGS = dict(in_node_features=44, in_edge_features=4, ns=16, nv=4, sigma_embed_dim=32, sigma_min=0.0314,
            sigma_max=3.14, num_conv_layers=4, max_radius=5.0, radius_embed_dim=50, scale_by_sigma=True,
            use_second_order_repr=True, no_residual=False, no_batch_norm=False)
SMILES = ['CCOC(=O)CCN', 'CC(O)CC#N', 'OCC1CC1C=O', 'CC(C)C(N)=O', 'FC(F)CCO', 'OCC(N)C(F)(F)F']


def datas(paired=False, n_conf=3):
    out = []
    for i, smi in enumerate(SMILES):
        mol, data = featurize_mol_from_smiles(smi, dataset='qm9')
        AllChem.EmbedMultipleConfs(mol, numConfs=2 * n_conf, randomSeed=1 + i)
        em, mr = get_transformation_mask(data)
        data.edge_mask = torch.tensor(em)
        data.mask_rotate = mr
        P = [torch.tensor(c.GetPositions(), dtype=torch.float) for c in mol.GetConformers()]
        data.pos = P[:n_conf]
        data.weights = [1 / n_conf] * n_conf
        if paired:
            data.gt_pos = P[n_conf:]
        out.append(data)
    return out


# --------------------------------------------------------------------------------------------- golden (defect 6)
@pytest.mark.skipif(not SNAP, reason='R1_SNAPSHOT not set')
def test_golden_model_transform_featurize_identical(tmp_path):
    outs = []
    for root in (SNAP, REPO):
        o = tmp_path / (os.path.basename(os.path.dirname(root)) + '.pt')
        r = subprocess.run([sys.executable, os.path.join(TOOLS, 'tests', 'golden_dump.py'), str(o),
                            '--data_root', os.path.join(REPO, 'data/QM9')], cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        outs.append(torch.load(o))
    a, b = outs
    assert a['model_state'].keys() == b['model_state'].keys()
    assert all(torch.equal(a['model_state'][k], b['model_state'][k]) for k in a['model_state'])     # M1
    assert torch.equal(a['model_out'], b['model_out'])                                                 # M1
    for (p1, e1, s1), (p2, e2, s2) in zip(a['transform'], b['transform']):                             # M2
        assert torch.equal(p1, p2) and torch.equal(e1, e2) and torch.equal(s1, s2)
    assert a['rng_after'][0] == b['rng_after'][0] and np.array_equal(a['rng_after'][1], b['rng_after'][1])
    assert torch.equal(a['rng_after'][2], b['rng_after'][2])                                           # no extra draw
    for v in ('std', 'raw'):                                                                           # M8
        A, B = a['featurized'][v], b['featurized'][v]
        assert not isinstance(A, str), A
        assert len(A) == len(B) > 0
        for x, y in zip(A, B):
            assert x[0] == y[0] and x[2] == y[2] and torch.equal(x[3], y[3]) and torch.equal(x[4], y[4])
            assert all(torch.equal(p, q) for p, q in zip(x[1], y[1]))


@pytest.mark.skipif(not (SNAP and PRE), reason='R1_SNAPSHOT / PRETRAINED not set')
def test_golden_generate_identical(tmp_path):                                                          # M9
    outs = []
    for k, root in enumerate((SNAP, REPO)):
        o = tmp_path / f'confs{k}.pkl'
        r = subprocess.run([sys.executable, 'generate_confs.py', '--model_dir', PRE, '--test_csv',
                            os.path.join(REPO, 'data/QM9/test_smiles.csv'), '--limit_mols', '4', '--seed', '0',
                            '--no_energy', '--inference_steps', '5', '--out', str(o)],
                           cwd=root, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        assert r.returncode == 0, r.stderr[-3000:]
        outs.append(pickle.load(open(o, 'rb')))
    assert outs[0].keys() == outs[1].keys() and len(outs[0]) > 0
    for smi in outs[0]:
        for m1, m2 in zip(outs[0][smi], outs[1][smi]):
            assert np.array_equal(m1.GetConformer().GetPositions(), m2.GetConformer().GetPositions())


# --------------------------------------------------------------------------------------------- S4 model (C1-C4)
def test_C1_lambda_dim0_is_unchanged_architecture():
    torch.manual_seed(0)
    a = get_model(Namespace(**ARGS)).state_dict()
    torch.manual_seed(0)
    b = get_model(Namespace(**ARGS, lambda_embed_dim=0)).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.skipif(not PRE, reason='PRETRAINED not set')
def test_C1b_released_checkpoint_loads_strictly():
    with open(os.path.join(PRE, 'model_parameters.yml')) as f:
        y = yaml.full_load(f)
    m = get_model(Namespace(**y))
    m.load_state_dict(torch.load(os.path.join(PRE, 'best_model.pt'), map_location='cpu'), strict=True)


def _batch(tr, paired=False):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    return next(iter(DataLoader([tr(copy.deepcopy(d)) for d in datas(paired)], batch_size=len(SMILES))))


def test_C2_C3_lambda_plumbing_symmetry():
    torch.manual_seed(0)
    model = get_model(Namespace(**ARGS, lambda_embed_dim=32)).eval()
    b = _batch(TorsionNoiseTransform(sigma_min=0.0314, sigma_max=3.14, l_interp=True), paired=True)
    assert hasattr(b, 'node_lambda') and b.node_lambda.shape[0] == b.num_nodes and not hasattr(b, 'gt_pos')
    with torch.no_grad():
        b0 = copy.deepcopy(b); b0.node_lambda = torch.zeros_like(b.node_lambda)
        b1 = copy.deepcopy(b); b1.node_lambda = torch.ones_like(b.node_lambda)
        s0, s1 = model(b0).edge_pred.clone(), model(b1).edge_pred.clone()
        assert (s0 - s1).abs().max() > 1e-4  # the embedding is live
        from scipy.spatial.transform import Rotation
        R = torch.tensor(Rotation.random(random_state=3).as_matrix(), dtype=torch.float)
        br = copy.deepcopy(b0); br.pos = br.pos @ R.T + 1.0
        assert (model(br).edge_pred - s0).abs().max() / s0.abs().max() < 1e-3  # rotation invariance
        bm = copy.deepcopy(b0); bm.pos = -bm.pos
        assert (model(bm).edge_pred + s0).abs().max() / s0.abs().max() < 1e-3  # parity (0o output)
        bn = copy.deepcopy(b); del bn.node_lambda
        with pytest.raises(AssertionError):
            model(bn)


@pytest.mark.skipif(not PRE, reason='PRETRAINED not set')
def test_C4_generate_guards(tmp_path):
    with open(os.path.join(PRE, 'model_parameters.yml')) as f:
        y = yaml.full_load(f)
    lam_dir = tmp_path / 'lam'
    lam_dir.mkdir()
    y2 = dict(y, lambda_embed_dim=32)
    yaml.dump(y2, open(lam_dir / 'model_parameters.yml', 'w'))
    torch.save(get_model(Namespace(**y2)).state_dict(), lam_dir / 'best_model.pt')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    base = [sys.executable, 'generate_confs.py', '--test_csv', os.path.join(REPO, 'data/QM9/test_smiles.csv'),
            '--limit_mols', '1', '--no_energy', '--inference_steps', '2', '--seed', '0']
    r = subprocess.run(base + ['--model_dir', str(lam_dir)], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and 'l_level' in (r.stderr + r.stdout)
    r = subprocess.run(base + ['--model_dir', PRE, '--l_level', '0'], capture_output=True, text=True, env=env)
    assert r.returncode != 0
    r = subprocess.run(base + ['--model_dir', str(lam_dir), '--l_level', '0.5', '--out', str(tmp_path / 'c.pkl')],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]


# --------------------------------------------------------------------------------------------- transform (C5-C10)
class _DS(Dataset):
    def __init__(self, items, transform):
        super().__init__(None, transform)
        self.items = items

    def len(self):
        return len(self.items)

    def get(self, idx):
        return copy.deepcopy(self.items[idx])  # as ConformerDataset.get (utils/dataset.py)


def test_C5_jitter_fresh_and_dataset_untouched():
    items = datas()
    ref = [p.clone() for p in items[0].pos]
    ds = _DS(items, TorsionNoiseTransform(sigma_min=1e-9, sigma_max=2e-9, l_jitter=0.04))
    a, b = ds[0].pos, ds[0].pos
    assert not torch.equal(a, b)
    for _ in range(50):
        ds[0]
    assert all(torch.equal(p, q) for p, q in zip(items[0].pos, ref))
    # jitter statistics with (almost) no torsion noise: per-axis SD ~ sigma
    tr = TorsionNoiseTransform(sigma_min=1e-9, sigma_max=2e-9, l_jitter=0.04)
    e = []
    for _ in range(400):
        d = copy.deepcopy(items[1]); d.pos = [d.pos[0]]
        e.append((tr(d).pos - items[1].pos[0]).numpy())
    sd = np.concatenate(e).std()
    assert abs(sd - 0.04) < 0.003, sd


def test_C6_worker_seeding_and_reproducibility():
    def run(seed):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        ds = _DS(datas() * 4, TorsionNoiseTransform(sigma_min=0.0314, sigma_max=3.14, l_jitter=0.04))
        L = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
        out = []
        for _ in range(2):
            for bt in L:
                out.append(bt.pos.clone())
        return out
    a, b = run(0), run(0)
    assert all(torch.equal(x, y) for x, y in zip(a, b))  # reproducible with the same seed
    flat = [x[:5].flatten() for x in a]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i].shape == flat[j].shape:
                assert not torch.equal(flat[i], flat[j])  # no duplicated noise across workers / epochs


def test_C7_paired_selection_formulas():
    items = datas(paired=True)
    tr_mix = TorsionNoiseTransform(sigma_min=1e-9, sigma_max=2e-9, l_mix_p_gt=0.5)
    n_gt = 0
    for t in range(2000):
        d0 = items[t % len(items)]
        d = tr_mix(copy.deepcopy(d0))
        k = d.l_conf_idx
        is_gt = torch.allclose(d.pos, d0.gt_pos[k], atol=1e-4)
        assert is_gt or torch.allclose(d.pos, d0.pos[k], atol=1e-4)
        assert not hasattr(d, 'gt_pos')
        n_gt += int(is_gt)
    assert abs(n_gt / 2000 - 0.5) < 0.04, n_gt
    tr_b1cap = TorsionNoiseTransform(sigma_min=1e-9, sigma_max=2e-9, l_mix_p_gt=1.0)
    d = tr_b1cap(copy.deepcopy(items[0]))
    assert torch.allclose(d.pos, items[0].gt_pos[d.l_conf_idx], atol=1e-4)
    tr_int = TorsionNoiseTransform(sigma_min=1e-9, sigma_max=2e-9, l_interp=True)
    lams = []
    for t in range(200):
        d0 = items[t % len(items)]
        np.random.seed(t)
        d = tr_int(copy.deepcopy(d0))
        lam = float(d.node_lambda[0])
        lams.append(lam)
        exp = interp_x_torch(d0.pos[d.l_conf_idx], d0.gt_pos[d.l_conf_idx], lam, d0.edge_index)
        assert torch.allclose(d.pos, exp, atol=1e-4)
    assert 0.0 <= min(lams) and max(lams) <= 1.0 and 0.4 < np.mean(lams) < 0.6
    batch = next(iter(DataLoader([tr_int(copy.deepcopy(x)) for x in items], batch_size=len(items))))
    assert batch.node_lambda.shape[0] == batch.num_nodes


def test_C7b_torch_interp_equals_lgeom():
    from rdkit import Chem
    m = Chem.AddHs(Chem.MolFromSmiles('OCC(N)C(F)(F)F'))
    c = AllChem.EmbedMultipleConfs(m, 2, randomSeed=3)
    X, Y = lgeom.positions(m, c[0]), lgeom.positions(m, c[1])
    ei = []
    for b in m.GetBonds():
        ei += [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), (b.GetEndAtomIdx(), b.GetBeginAtomIdx())]
    ei = torch.tensor(ei).T
    for lam in (0.0, 0.3, 0.5, 1.0):
        a = interp_x_torch(torch.tensor(X), torch.tensor(Y), lam, ei).numpy()
        assert np.abs(a - lgeom.interp_x(X, Y, lam, *lgeom.terminal_atoms(m))).max() < 1e-9


def test_C8_featurize_keeps_pairing_through_reacted_filter():
    from rdkit import Chem
    smi = 'CCOC(=O)CCN'
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    cids = AllChem.EmbedMultipleConfs(m, 3, randomSeed=2)
    reacted = Chem.AddHs(Chem.MolFromSmiles('CCOC(=O)CCO'))
    AllChem.EmbedMolecule(reacted, randomSeed=1)
    confs = []
    for k, c in enumerate(cids):
        mk = Chem.Mol(m, confId=c)
        confs.append(dict(rd_mol=mk, boltzmannweight=1.0, gt_pos_aligned=np.full((m.GetNumAtoms(), 3), float(k))))
        if k == 0:
            confs.append(dict(rd_mol=reacted, boltzmannweight=1.0,
                              gt_pos_aligned=np.full((reacted.GetNumAtoms(), 3), -1.0)))
    ds = ConformerDataset.__new__(ConformerDataset)
    ds.types, ds.boltzmann_resampler = 'qm9', None
    d = ds.featurize_mol(dict(conformers=confs, smiles=smi))
    assert len(d.pos) == len(d.gt_pos) == 3
    assert [float(g[0, 0]) for g in d.gt_pos] == [0.0, 1.0, 2.0]


def test_C10_stale_cache_guard(tmp_path):
    items = datas()  # no gt_pos
    cache = tmp_path / 'c'
    with open(str(cache) + '.train', 'wb') as f:
        pickle.dump(items, f)
    with pytest.raises(AssertionError):
        ConformerDataset('x/', 'unused', 'train', types='qm9', dataset='qm9', cache=str(cache),
                         transform=TorsionNoiseTransform(l_mix_p_gt=0.5))
    ConformerDataset('x/', 'unused', 'train', types='qm9', dataset='qm9', cache=str(cache),
                     transform=TorsionNoiseTransform())  # default transform: no guard


def test_C11_fail_on_nan_and_limit_iters():
    from utils.training import train_epoch
    torch.manual_seed(0)
    model = get_model(Namespace(**ARGS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = _DS(datas() * 3, TorsionNoiseTransform(sigma_min=0.0314, sigma_max=3.14))
    L = DataLoader(ds, batch_size=3, shuffle=False)
    st = {}
    train_epoch(model, L, opt, 'cpu', stats=st, limit_iters=2)
    assert st['n_iter'] == 2 and 0 <= st['data_wait'] <= st['epoch_time']
    with torch.no_grad():
        for p_ in model.parameters():
            p_.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        train_epoch(model, L, opt, 'cpu', fail_on_nan=True)

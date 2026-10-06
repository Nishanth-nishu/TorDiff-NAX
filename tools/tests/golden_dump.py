"""
Golden-output dumper for the round-2 regression tests (DECISION D9 graft from code_plan_1 §9.2 M1/M2/M8; defect 6).
Run with cwd = a torsional-diffusion checkout (the round-1 snapshot OR the patched repo); it writes one torch file whose
contents must be identical between the two checkouts when only default flags are used.

  python golden_dump.py OUT.pt [--data_root data/QM9]

Contents
  model_state  : get_model(QM9 recipe) state_dict under torch.manual_seed(0)                      (M1)
  model_out    : eval-mode edge_pred of that model on a fixed featurized batch                      (M1)
  transform    : TorsionNoiseTransform() outputs (pos, edge_rotate, node_sigma) on 30 datapoints with seeded RNGs,
                 and the python/numpy/torch RNG states after the calls (no extra RNG draw)            (M2)
  featurized   : ConformerDataset(limit 12, cache=None) datapoints for std and raw (pos, weights, x, edge_index)
                                                                                                      (M8)
"""
import copy
import random
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

# single-threaded CPU math: multi-threaded scatter/radius reductions make the forward pass non-deterministic run to run
# (measured: max |diff| ~2e-6 between two runs of the SAME code), which would mask a real regression
torch.set_num_threads(1)

sys.path.insert(0, '.')
from utils.dataset import TorsionNoiseTransform, ConformerDataset  # noqa: E402
from utils.featurization import featurize_mol_from_smiles, qm9_types  # noqa: E402
from utils.torsion import get_transformation_mask  # noqa: E402
from utils.utils import get_model  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

p = ArgumentParser()
p.add_argument('out')
p.add_argument('--data_root', default='data/QM9')
a = p.parse_args()

ARGS = Namespace(in_node_features=44, in_edge_features=4, ns=32, nv=8, sigma_embed_dim=32, sigma_min=0.0314,
                 sigma_max=3.14, num_conv_layers=4, max_radius=5.0, radius_embed_dim=50, scale_by_sigma=True,
                 use_second_order_repr=True, no_residual=False, no_batch_norm=False)
SMILES = ['CCOC(=O)CCN', 'CC(O)CC#N', 'OCC1CC1C=O', 'CC(C)C(N)=O', 'FC(F)CCO']


def datas():
    out = []
    for i, smi in enumerate(SMILES):
        mol, data = featurize_mol_from_smiles(smi, dataset='qm9')
        AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=1 + i)
        em, mr = get_transformation_mask(data)
        data.edge_mask = torch.tensor(em)
        data.mask_rotate = mr
        data.pos = [torch.tensor(c.GetPositions(), dtype=torch.float) for c in mol.GetConformers()]
        data.weights = [1 / 3] * 3
        out.append(data)
    return out


res = {}
torch.manual_seed(0)
model = get_model(ARGS)
res['model_state'] = {k: v.clone() for k, v in model.state_dict().items()}
model.eval()
random.seed(0); np.random.seed(0); torch.manual_seed(0)
tr = TorsionNoiseTransform(sigma_min=0.0314, sigma_max=3.14)
batch = next(iter(DataLoader([tr(copy.deepcopy(d)) for d in datas()], batch_size=5)))
with torch.no_grad():
    res['model_out'] = model(batch).edge_pred.clone()

random.seed(1); np.random.seed(1); torch.manual_seed(1)
T = []
base = datas()
for k in range(30):
    d = tr(copy.deepcopy(base[k % len(base)]))
    T.append((d.pos.clone(), torch.as_tensor(d.edge_rotate).clone(), d.node_sigma.clone()))
res['transform'] = T
res['rng_after'] = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())

F = {}
for variant, std in (('std', f'{a.data_root}/standardized_pickles'), ('raw', None)):
    try:
        ds = ConformerDataset(f'{a.data_root}/qm9/', f'{a.data_root}/split.npy', 'train', types=qm9_types,
                              dataset='qm9', transform=None, num_workers=1, limit_molecules=12, cache=None,
                              pickle_dir=std)
        F[variant] = [(d.canonical_smi, [q.clone() for q in d.pos], list(d.weights), d.x.clone(), d.edge_index.clone())
                      for d in ds.datapoints]
    except Exception as e:  # data not present (local machine)
        F[variant] = repr(e)
res['featurized'] = F
torch.save(res, a.out)
print('wrote', a.out)

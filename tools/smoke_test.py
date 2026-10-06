"""
Environment smoke test for torsional diffusion (run from the repo root, inside the venv, on a GPU node).
Exercises: imports, featurisation (QM9 types), TorsionNoiseTransform + PyG collation of mask_rotate lists,
a train step (e3nn BatchNorm on the 0o-only bond_conv -> the e3nn<=0.5.0 crash), torus score tables,
2-step reverse SDE + ODE sampling, pyg_to_mol, and the reflection (parity) property of the score.
No dataset needed. Exits non-zero on failure.
"""
import copy, sys, time
from argparse import Namespace

import numpy as np
import torch
from rdkit.Chem import AllChem

import diffusion.torus as torus
from diffusion.sampling import embed_seeds, perturb_seeds, sample, pyg_to_mol, get_seed
from utils.dataset import TorsionNoiseTransform
from utils.featurization import featurize_mol_from_smiles
from utils.torsion import get_transformation_mask
from utils.utils import get_model
from torch_geometric.loader import DataLoader

t0 = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)
torch.manual_seed(0); np.random.seed(0)

smiles = ['CCOC(=O)CCN', 'CC(O)CC#N', 'OCC1CC1C=O', 'CC(C)C(N)=O', 'FC(F)CCO']
datas = []
for smi in smiles:
    mol, data = featurize_mol_from_smiles(smi, dataset='qm9')
    assert data.x.shape[1] == 44, data.x.shape
    AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=1)
    edge_mask, mask_rotate = get_transformation_mask(data)
    data.edge_mask = torch.tensor(edge_mask); data.mask_rotate = mask_rotate
    data.pos = [torch.tensor(c.GetPositions(), dtype=torch.float) for c in mol.GetConformers()]
    data.weights = [1 / 3] * 3
    datas.append(data)

transform = TorsionNoiseTransform(sigma_min=0.01 * np.pi, sigma_max=np.pi)
batch_list = [transform(copy.deepcopy(d)) for d in datas]
loader = DataLoader(batch_list, batch_size=len(batch_list))
batch = next(iter(loader)).to(device)

args = Namespace(in_node_features=44, in_edge_features=4, ns=16, nv=4, sigma_embed_dim=32, sigma_min=0.01 * np.pi,
                 sigma_max=np.pi, num_conv_layers=4, max_radius=5.0, radius_embed_dim=50, scale_by_sigma=True,
                 use_second_order_repr=True, no_residual=False, no_batch_norm=False)
model = get_model(args).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
out = model(batch)
score = torch.tensor(torus.score(out.edge_rotate.cpu().numpy(), out.edge_sigma.cpu().numpy()), device=device)
norm = torch.tensor(torus.score_norm(out.edge_sigma.cpu().numpy()), device=device)
loss = ((score - out.edge_pred) ** 2 / norm).mean()
loss.backward(); opt.step()
print(f'train step ok: loss={loss.item():.4f}  n_torsions={int(batch.edge_mask.sum())}')
assert np.isfinite(loss.item())

# parity: reflecting all coordinates must flip the sign of every torsion score (0o pseudoscalar output)
# fresh batches: the train-step batch now holds non-leaf tensors (written by the model) that cannot be deep-copied
def fresh_batch():
    return next(iter(DataLoader([copy.deepcopy(d) for d in batch_list], batch_size=len(batch_list)))).to(device)
model.eval()
with torch.no_grad():
    b1 = fresh_batch(); b1.node_sigma = torch.full_like(b1.node_sigma, 0.3)
    s1 = model(b1).edge_pred.clone()
    b2 = fresh_batch(); b2.node_sigma = torch.full_like(b2.node_sigma, 0.3); b2.pos = -b2.pos
    s2 = model(b2).edge_pred.clone()
err = (s1 + s2).abs().max().item() / (s1.abs().max().item() + 1e-12)
print(f'parity check: max|s(x)+s(-x)|/max|s| = {err:.2e}')
assert err < 1e-3, 'score is not reflection-odd'

# sampling (SDE and ODE, 2 steps) on one molecule
mol, data = get_seed('CCOC(=O)CCN', dataset='qm9')
confs, _ = embed_seeds(mol, data, 4, embed_func=lambda m, n: (AllChem.EmbedMultipleConfs(m, numConfs=n, randomSeed=3), m)[1])
confs = perturb_seeds(confs)
for ode in (False, True):
    res = sample(copy.deepcopy(confs), model, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=2, batch_size=4, ode=ode,
                 likelihood='hutch' if ode else None)
    mols = [pyg_to_mol(mol, c, rmsd=True) for c in res]
    print(f'sampling ok (ode={ode}): {len(mols)} conformers, seed-RMSD {[round(getattr(m, "rmsd", -1), 2) for m in mols]}')
print(f'SMOKE TEST PASSED in {time.time() - t0:.1f}s')
sys.exit(0)

import os.path
from multiprocessing import Pool

from rdkit import Chem
import numpy as np
import glob, pickle, random
import os.path as osp
import torch, tqdm
import copy
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.transforms import BaseTransform
from collections import defaultdict

from utils.featurization import dihedral_pattern, featurize_mol, qm9_types, drugs_types
from utils.torsion import get_transformation_mask, modify_conformer


def interp_x_torch(x, y, lam, edge_index):
    """[round2 S4] x_lam = (1-lam) x + lam y, then every terminal (degree-1) atom is put back on the linearly
    interpolated bond length along its interpolated bond direction. Same formula as tools/lgeom.py interp_x, which
    builds the S1 lambda seeds (equality tested in tools/tests/test_round2_model.py). See lgeom.interp_x for why
    (single-H rotors are L in TD and are never torsion-matched). Endpoints are exact."""
    z = (1.0 - lam) * x + lam * y
    src, dst = edge_index
    deg = torch.bincount(src, minlength=x.shape[0])
    m = (deg[src] == 1) & (deg[dst] > 1)
    t, p = src[m], dst[m]
    if len(t):
        v = z[t] - z[p]
        n = v.norm(dim=1)
        d = (1.0 - lam) * (x[t] - x[p]).norm(dim=1) + lam * (y[t] - y[p]).norm(dim=1)
        ok = n > 1e-6
        z[t[ok]] = z[p[ok]] + v[ok] / n[ok, None] * d[ok, None]
    return z


class TorsionNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi, boltzmann_weight=False,
                 l_jitter=0.0, l_mix_p_gt=0.0, l_interp=False):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.boltzmann_weight = boltzmann_weight
        # [round2] default-off training-L options (DECISION D9 / defect 6: defaults leave the original path untouched)
        assert not (l_mix_p_gt > 0 and l_interp), 'S3 (--l_mix_p_gt) and S4 (--l_interp) are separate arms'
        assert 0.0 <= l_mix_p_gt <= 1.0 and l_jitter >= 0.0
        self.l_jitter, self.l_mix_p_gt, self.l_interp = l_jitter, l_mix_p_gt, l_interp
        self.needs_gt = l_mix_p_gt > 0 or l_interp

    def __call__(self, data):
        # select conformer
        if not self.needs_gt:
            # original lines, unchanged (same statements, same RNG calls)
            if self.boltzmann_weight:
                data.pos = random.choices(data.pos, data.weights, k=1)[0]
            else:
                data.pos = random.choice(data.pos)
        else:
            self._select_paired_L(data)
        if self.l_jitter > 0:
            # [round2 S2] Gaussian jitter of ALL atoms, per axis, fresh at every access (get() deep-copies, :193);
            # applied before the torsion noise so the score target edge_rotate is unchanged.
            # code_plan_2 §5.2 / §6 S2; F6 (per-axis sigma); evidence E5, E8-E10 research_A
            data.pos = data.pos + self.l_jitter * torch.randn_like(data.pos)

        try:
            edge_mask, mask_rotate = data.edge_mask, data.mask_rotate
        except:
            edge_mask, mask_rotate = data.mask_edges, data.mask_rotate
            data.edge_mask = torch.tensor(data.mask_edges)

        sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        data.node_sigma = sigma * torch.ones(data.num_nodes)

        torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=edge_mask.sum())
        data.pos = modify_conformer(data.pos, data.edge_index.T[edge_mask], mask_rotate, torsion_updates)
        data.edge_rotate = torch.tensor(torsion_updates)
        return data

    def _select_paired_L(self, data):
        # [round2 S3/S4/B1cap] pick a conformer INDEX so that the matched RDKit L (data.pos[k]) and its own GT L
        # (data.gt_pos[k], Kabsch-aligned + terminal-relabelled offline by tools/build_paired_pickles.py) stay paired.
        # Equal caps by construction: both halves are the same <=30 conformers. code_plan_2 §5.2; DECISION D1, D6
        if not (hasattr(data, 'gt_pos') and len(data.gt_pos) == len(data.pos)):
            raise AssertionError('stale or unpaired cache: --l_mix_p_gt/--l_interp need a cache built from '
                                 'standardized_pickles_paired')
        n = len(data.pos)
        k = random.choices(range(n), data.weights, k=1)[0] if self.boltzmann_weight else random.randrange(n)
        x, y = data.pos[k], data.gt_pos[k]
        assert x.shape == y.shape
        if self.l_interp:
            # S4: lambda ~ U[0,1] (SHORTLIST S4); unsafe pairs were dropped offline (DECISION D1), so every
            # midpoint is chemically sane. evidence E5/E6 research_A
            lam = float(np.random.uniform())
            data.pos = interp_x_torch(x, y, lam, data.edge_index)
            data.node_lambda = lam * torch.ones(data.num_nodes)
        else:
            # S3 (p=0.5) / B1cap (p=1.0): Bernoulli choice between the two L of the same conformer
            data.pos = y if np.random.uniform() < self.l_mix_p_gt else x
        data.l_conf_idx = k  # for tests only
        del data.gt_pos  # data is a deepcopy (get(), :193); never collate the list

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(sigma_min={self.sigma_min}, '
                f'sigma_max={self.sigma_max})')


class ConformerDataset(Dataset):
    def __init__(self, root, split_path, mode, types, dataset, transform=None, num_workers=1, limit_molecules=None,
                 cache=None, pickle_dir=None, boltzmann_resampler=None):
        # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol

        # [ablation-hooks] names are derived as path[len(root):-7], so root MUST end with '/' (GitHub issue #12)
        if not root.endswith('/'):
            root = root + '/'
        super(ConformerDataset, self).__init__(root, transform)
        self.root = root
        self.types = types
        self.failures = defaultdict(int)
        self.dataset = dataset
        self.boltzmann_resampler = boltzmann_resampler

        if cache: cache += "." + mode
        self.cache = cache
        if cache and os.path.exists(cache):
            print('Reusing preprocessing from cache', cache)
            with open(cache, "rb") as f:
                self.datapoints = pickle.load(f)
        else:
            print("Preprocessing")
            self.datapoints = self.preprocess_datapoints(root, split_path, pickle_dir, mode, num_workers, limit_molecules)
            if cache:
                print("Caching at", cache)
                with open(cache, "wb") as f:
                    pickle.dump(self.datapoints, f)

        if limit_molecules:
            self.datapoints = self.datapoints[:limit_molecules]

        # [round2] stale-cache guard (code_plan_2 V8): paired training options need gt_pos in EVERY datapoint
        if getattr(transform, 'needs_gt', False):
            n_bad = sum(1 for d in self.datapoints if not hasattr(d, 'gt_pos') or len(d.gt_pos) != len(d.pos))
            assert n_bad == 0, f'{n_bad}/{len(self.datapoints)} datapoints lack paired gt_pos: wrong/stale cache {cache}'


    def preprocess_datapoints(self, root, split_path, pickle_dir, mode, num_workers, limit_molecules):
        mols_per_pickle = 1000
        split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
        if limit_molecules:
            split = split[:limit_molecules]
        smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))
        # [ablation-hooks] guard against split indices beyond the number of raw pickles (GitHub issue #20, QM9)
        n_oob = int(np.sum(np.asarray(split) >= len(smiles)))
        if n_oob:
            print(f'WARNING: {n_oob} split indices >= number of raw pickles ({len(smiles)}); dropping them')
            split = [i for i in split if i < len(smiles)]
        if len(smiles) == 0:
            raise FileNotFoundError(f'No *.pickle files found under data_dir={root!r}')
        smiles = smiles[split]

        self.open_pickles = {}
        if pickle_dir:
            smiles = [(i // mols_per_pickle, smi[len(root):-7]) for i, smi in zip(split, smiles)]
            if limit_molecules:
                smiles = smiles[:limit_molecules]
            self.current_pickle = (None, None)
            self.pickle_dir = pickle_dir
        else:
            smiles = [smi[len(root):-7] for smi in smiles]

        print('Preparing to process', len(smiles), 'smiles')
        datapoints = []
        if num_workers > 1:
            p = Pool(num_workers)
            p.__enter__()
        with tqdm.tqdm(total=len(smiles)) as pbar:
            map_fn = p.imap if num_workers > 1 else map
            for t in map_fn(self.filter_smiles, smiles):
                if t:
                    datapoints.append(t)
                pbar.update()
        if num_workers > 1: p.__exit__(None, None, None)
        print('Fetched', len(datapoints), 'mols successfully')
        print(self.failures)
        if pickle_dir: del self.current_pickle
        return datapoints

    def filter_smiles(self, smile):

        if type(smile) is tuple:
            pickle_id, smile = smile
            current_id, current_pickle = self.current_pickle
            if current_id != pickle_id:
                path = osp.join(self.pickle_dir, str(pickle_id).zfill(3) + '.pickle')
                if not osp.exists(path):
                    self.failures[f'std_pickle{pickle_id}_not_found'] += 1
                    return False
                with open(path, 'rb') as f:
                    self.current_pickle = current_id, current_pickle = pickle_id, pickle.load(f)
            if smile not in current_pickle:
                self.failures['smile_not_in_std_pickle'] += 1
                return False
            mol_dic = current_pickle[smile]

        else:
            if not os.path.exists(os.path.join(self.root, smile + '.pickle')):
                self.failures['raw_pickle_not_found'] += 1
                return False
            pickle_file = osp.join(self.root, smile + '.pickle')
            mol_dic = self.open_pickle(pickle_file)

        smile = mol_dic['smiles']

        if '.' in smile:
            self.failures['dot_in_smile'] += 1
            return False

        # filter mols rdkit can't intrinsically handle
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            self.failures['mol_from_smiles_failed'] += 1
            return False

        mol = mol_dic['conformers'][0]['rd_mol']
        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            self.failures['no_substruct_match'] += 1
            return False

        if N < 4:
            self.failures['mol_too_small'] += 1
            return False

        data = self.featurize_mol(mol_dic)
        if not data:
            self.failures['featurize_mol_failed'] += 1
            return False

        edge_mask, mask_rotate = get_transformation_mask(data)
        if np.sum(edge_mask) < 0.5:
            self.failures['no_rotable_bonds'] += 1
            return False

        data.edge_mask = torch.tensor(edge_mask)
        data.mask_rotate = mask_rotate
        return data

    def len(self):
        return len(self.datapoints)

    def get(self, idx):
        data = self.datapoints[idx]
        if self.boltzmann_resampler:
            self.boltzmann_resampler.try_resample(data)
        return copy.deepcopy(data)

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic):
        confs = mol_dic['conformers']
        name = mol_dic["smiles"]

        mol_ = Chem.MolFromSmiles(name)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)

        pos = []
        weights = []
        gt_pos = []  # [round2] paired GT L, only filled from standardized_pickles_paired (code_plan_2 §5.2)
        for conf in confs:
            mol = conf['rd_mol']

            # filter for conformers that may have reacted
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            except Exception as e:
                print(e)
                continue

            if conf_canonical_smi != canonical_smi:
                continue

            pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
            weights.append(conf['boltzmannweight'])
            if 'gt_pos_aligned' in conf:
                # appended in the SAME iteration as pos, so the 'reacted' filter above keeps the pairing intact
                gt_pos.append(torch.tensor(conf['gt_pos_aligned'], dtype=torch.float))
            correct_mol = mol

            if self.boltzmann_resampler is not None:
                # torsional Boltzmann generator uses only the local structure of the first conformer
                break

        # return None if no non-reactive conformers were found
        if len(pos) == 0:
            return None

        data = featurize_mol(correct_mol, self.types)
        normalized_weights = list(np.array(weights) / np.sum(weights))
        if np.isnan(normalized_weights).sum() != 0:
            print(name, len(confs), len(pos), weights)
            normalized_weights = [1 / len(weights)] * len(weights)
        data.canonical_smi, data.mol, data.pos, data.weights = canonical_smi, correct_mol, pos, normalized_weights
        if gt_pos:
            assert len(gt_pos) == len(pos) and all(g.shape == p.shape for g, p in zip(gt_pos, pos)), name
            data.gt_pos = gt_pos

        return data

    def resample_all(self, resampler, temperature=None):
        ess = []
        for data in tqdm.tqdm(self.datapoints):
            ess.append(resampler.resample(data, temperature=temperature))
        return ess


def construct_loader(args, modes=('train', 'val'), boltzmann_resampler=None):
    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    transform = TorsionNoiseTransform(sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                      boltzmann_weight=args.boltzmann_weight,
                                      # [round2] default-off (getattr: old pickled args / yamls lack the keys)
                                      l_jitter=getattr(args, 'l_jitter', 0.0),
                                      l_mix_p_gt=getattr(args, 'l_mix_p_gt', 0.0),
                                      l_interp=getattr(args, 'l_interp', False))
    types = qm9_types if args.dataset == 'qm9' else drugs_types

    for mode in modes:
        dataset = ConformerDataset(args.data_dir, args.split_path, mode, dataset=args.dataset,
                                   types=types, transform=transform,
                                   num_workers=args.num_workers,
                                   limit_molecules=args.limit_train_mols,
                                   cache=args.cache,
                                   pickle_dir=args.std_pickles,
                                   boltzmann_resampler=boltzmann_resampler)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False if mode == 'test' else True,
                            num_workers=getattr(args, 'loader_workers', 0))
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders

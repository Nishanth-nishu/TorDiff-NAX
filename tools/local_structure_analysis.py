"""
Local-structure (L) error analysis for QM9: how far are RDKit/ETKDG local structures from GEOM ground truth,
and what is the best heavy-atom RMSD a *torsion-only* model could reach given RDKit L? (ablation plan J / A0)

Three modes
  --mode test   : for every test molecule, embed K = seeds_per_gt * L ETKDG seeds from the GT molecular graph, then for
                  every GT conformer compute
                    * bond-length RMSD (A) and bond-angle RMSD (deg) vs. its assigned seed and vs. the mean over seeds
                    * floor_rmsd (training analogue, = conformer matching): Hungarian assignment of the first L seeds
                      on the transplant cost, then differential evolution (DE) on the heavy-atom objective
                    * floor_best / floor_best_sym (test-time analogue): min over the top-m of ALL K seeds (ranked by
                      transplant cost) of the DE floor; _sym also re-scores the optimum with GetBestRMS
                      (symmetry-aware, as evaluate_confs.py). Compare THIS with AMR-R (min over K generated confs).
                    * floor_gtother: same with the local structures of the OTHER GT conformers as seeds
                      (= perfect but tau-independent L sampler; TD App. F.1 reports 0.284 A on DRUGS vs 0.324 for RDKit)
                    * angle_oracle_rmsd / local_oracle_rmsd: best seed with its ACYCLIC heavy-atom bond angles
                      (resp. angles + acyclic bond lengths) set to the GT values, then torsions re-optimised. What is
                      left is ring geometry + stereo + branching-centre inconsistency, i.e. what a low-dimensional
                      acyclic angle factor could NOT remove (plan item J-decomp; proxy for gap_synthesis A1(b))
                    * transplant_rmsd_{assigned,best}: GT torsions copied onto the assigned / best seed (no optimisation)
                    * GT-GT local variability (bond/angle RMSD between this GT conformer and the other GT conformers)
                  -> CSV with one row per (molecule, GT conformer)
  --mode run    : RUN-MATCHED floor for an actual generation run (confs.pkl + eval.pkl of evaluate_confs.py): for each
                  GT conformer l, re-optimise the torsions of the top-m generated conformers k (ranked by the observed
                  RMSD) against l, keeping their exact local structures. floor_run_l <= observed min_k RMSD_lk by
                  construction, and it uses the SAME seeds, stereo and K as the run. obs - floor_run is the part of
                  AMR-R a better torsion model could still remove; floor_run is the part only an L change can remove.
  --mode std    : read the standardized (conformer-matched) training pickles and dump conf['rmsd'] (the heavy-atom RMSD
                  left after torsion matching, i.e. the training-time local-structure error), one row per conformer.

Bound directions (review/cross_validation.md, issue M1):
  * Every DE value is an UPPER bound on the true torsion-only optimum for that seed (heuristic optimiser). The DE
    objective is symmetry-aware (min of AlignMol over all heavy-atom automorphisms, see heavy_automorphisms); it
    differs from GetBestRMS only by RDKit's conjugated-terminal-group symmetrisation (e.g. COOH O/O swap).
  * floor_rmsd (one-to-one, L seeds -- like TD's ~0.17 A QM9 figure and conf['rmsd']) is NOT a lower bound on a run's
    AMR-R: the run has K = 2L seeds and no one-to-one constraint. Use floor_best_sym (distributional estimate) or,
    better, floor_run (paired with the run) for "is TD at its floor?" statements.
  * conf['rmsd'] of the shipped pickles is additionally biased UP by the H-inclusive DE objective and maxiter=15.

Usage (repo root, CPU only):
  python ../tools/local_structure_analysis.py --mode test --test_csv data/QM9/test_smiles.csv \
      --true_mols data/QM9/test_mols.pkl --out ../results/local_structure_test.csv --n_workers 16
  python ../tools/local_structure_analysis.py --mode run --confs RES/confs.pkl --results RES/eval.pkl \
      --true_mols data/QM9/test_mols.pkl --out RES/floor_run.csv --n_workers 16
  python ../tools/local_structure_analysis.py --mode std --std_pickles data/QM9/standardized_pickles \
      --out ../results/local_structure_std.csv
"""
import copy, glob, os, pickle, sys
from argparse import ArgumentParser
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms
from scipy.optimize import differential_evolution, linear_sum_assignment

RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.getcwd())  # run from the torsional-diffusion repo root

parser = ArgumentParser()
parser.add_argument('--mode', choices=['test', 'run', 'std', 'leak'], required=True)
parser.add_argument('--test_csv', default='data/QM9/test_smiles.csv')
parser.add_argument('--true_mols', default='data/QM9/test_mols.pkl')
parser.add_argument('--std_pickles', default='data/QM9/standardized_pickles')
parser.add_argument('--confs', default=None, help='--mode run: confs.pkl of generate_confs.py')
parser.add_argument('--results', default=None, help='--mode run: eval.pkl of the patched evaluate_confs.py')
parser.add_argument('--out', required=True)
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--limit_mols', type=int, default=0)
parser.add_argument('--max_gt_confs', type=int, default=30, help='cap GT conformers per molecule (cost)')
parser.add_argument('--seeds_per_gt', type=int, default=2, help='K = seeds_per_gt * L ETKDG seeds (TD test time: K = 2L)')
parser.add_argument('--floor_topm', type=int, default=3, help='DE only on the m best seeds / generated confs per GT conf')
parser.add_argument('--popsize', type=int, default=15)
parser.add_argument('--maxiter', type=int, default=50, help='DE iterations (standardize_confs.py uses 15)')
parser.add_argument('--seed', type=int, default=1, help='RDKit randomSeed; never 0 (RDKit seed 0 => identical conformers)')
parser.add_argument('--mmff_seeds', action='store_true', help='MMFF-relax the ETKDG seeds first (== generate --pre_mmff)')
# [round2 S0] DE convergence check (code_plan_2 §2.1): restarts with seeds de_seed + 7919*r, minimum kept. Defaults
# (1 restart, de_seed = --seed) reproduce round 1 exactly. --de_seed decouples the DE seed from the ETKDG seed.
parser.add_argument('--de_restarts', type=int, default=1)
parser.add_argument('--de_seed', type=int, default=None)
# [round2 S0 leak test] torsion-space source-hit test on a --seed_confs_cycle run (verify_B §4 R2-0(e))
parser.add_argument('--seed_confs', default=None, help='--mode leak: the seed pickle the cycle run used')
parser.add_argument('--raw_dir', default='data/QM9/qm9/', help='--mode leak: raw pickles for Boltzmann weights')
parser.add_argument('--n_perm', type=int, default=200)
args = parser.parse_args()


# ----------------------------------------------------------------------------------------------- geometry helpers
def bonds_angles(mol, heavy_only=True):
    heavy = lambda a: mol.GetAtomWithIdx(a).GetAtomicNum() > 1
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
             if not heavy_only or (heavy(b.GetBeginAtomIdx()) and heavy(b.GetEndAtomIdx()))]
    angles = []
    for j in range(mol.GetNumAtoms()):
        if heavy_only and not heavy(j):
            continue
        nb = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if not heavy_only or heavy(n.GetIdx())]
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                angles.append((nb[a], j, nb[b]))
    return bonds, angles


def local_values(conf, bonds, angles):
    bl = np.array([rdMolTransforms.GetBondLength(conf, i, j) for i, j in bonds])
    ba = np.array([rdMolTransforms.GetAngleDeg(conf, i, j, k) for i, j, k in angles])
    return bl, ba


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan


def relevant_torsions(mol):
    """Rotatable bonds as in upstream get_torsion_angles (utils/standardization.py:63), restricted to those
    that can move heavy atoms (both sides contain >= 2 heavy atoms). Terminal CH3/OH/NH2 rotors are dropped:
    they do not affect the heavy-atom RMSD used by evaluate_confs.py."""
    G = nx.Graph()
    G.add_nodes_from(range(mol.GetNumAtoms()))
    G.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    all_t, rel_t = [], []
    for e in G.edges():
        G2 = G.copy(); G2.remove_edge(*e)
        if nx.is_connected(G2):
            continue
        comps = list(nx.connected_components(G2))
        if min(len(c) for c in comps) < 2:
            continue
        n0 = [n for n in G2.neighbors(e[0])]
        n1 = [n for n in G2.neighbors(e[1])]
        t = (n0[0], e[0], e[1], n1[0])
        all_t.append(t)
        if all(len(c & heavy) >= 2 for c in comps):
            # prefer heavy reference atoms for a well-defined dihedral
            h0 = [n for n in n0 if n in heavy] or n0
            h1 = [n for n in n1 if n in heavy] or n1
            rel_t.append((h0[0], e[0], e[1], h1[0]))
    return all_t, rel_t


def heavy_map(mol):
    return [(a.GetIdx(), a.GetIdx()) for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def heavy_automorphisms(mol, max_maps=1000):
    """[metric-verifier fix, review/metric_verification.md B2] all heavy-atom graph automorphisms of mol as AlignMol
    atom maps. Seeds are embedded from the GT mol (shared indexing), so min over these maps == GetBestRMS on heavy
    atoms (without RDKit's conjugated-terminal-group symmetrisation). With the identity map alone, e.g. an isopropyl
    whose two methyls are embedded in swapped positions can never be matched by torsions (floor 0.46 A instead of
    0.04 A for CC(C)CC#N in review/tests/test_metrics.py)."""
    hv = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    h = Chem.RemoveHs(mol)
    if h.GetNumAtoms() != len(hv) or any(h.GetAtomWithIdx(i).GetAtomicNum() != mol.GetAtomWithIdx(j).GetAtomicNum()
                                         for i, j in enumerate(hv)):
        return [heavy_map(mol)]
    ms = h.GetSubstructMatches(h, uniquify=False, useChirality=False, maxMatches=max_maps)
    return [[(hv[i], hv[q]) for i, q in enumerate(m)] for m in ms] or [heavy_map(mol)]


def heavy_rmsd(prb, ref, prb_cid=-1, ref_cid=-1, amap=None):
    """heavy-atom RMSD after alignment. amap: one atom map (list of (prb, ref) pairs) or a LIST of maps
    (-> min over maps = symmetry-aware)."""
    if amap and isinstance(amap[0], list):
        return min(AllChem.AlignMol(prb, ref, prb_cid, ref_cid, atomMap=m) for m in amap)
    return AllChem.AlignMol(prb, ref, prb_cid, ref_cid, atomMap=amap or heavy_map(ref))


def torsion_floor(seed_mol, seed_cid, gt_mol, torsions, amap, x_extra=None):
    """min over torsions of heavy-atom RMSD (given atom map). seed_mol conformer seed_cid is left AT the optimum.
    x_extra: extra candidate torsion vectors (GT torsions, the conformer's current torsions); the result is never
    worse than any of them."""
    if not torsions:
        return float(heavy_rmsd(seed_mol, gt_mol, seed_cid, -1, amap))
    conf = seed_mol.GetConformer(seed_cid)

    def setx(x):
        for t, v in zip(torsions, x):
            rdMolTransforms.SetDihedralRad(conf, *t, float(v))

    def f(x):
        setx(x)
        return heavy_rmsd(seed_mol, gt_mol, seed_cid, -1, amap)

    base_seed = args.seed if args.de_seed is None else args.de_seed
    cands = []
    for r in range(args.de_restarts):  # [round2 S0] r = 0 with default args == the round-1 call
        res = differential_evolution(f, [(-np.pi, np.pi)] * len(torsions), popsize=args.popsize,
                                     maxiter=args.maxiter, seed=base_seed + 7919 * r, polish=True, tol=1e-6)
        cands.append((res.fun, res.x))
    cands = cands + [(f(x), x) for x in (x_extra or [])]
    best_v, best_x = min(cands, key=lambda c: c[0])
    setx(best_x)
    return float(best_v)


def sym_rmsd(prb, ref, prb_cid=-1, ref_cid=-1):
    """heavy-atom symmetry-aware RMSD exactly as evaluate_confs.py (GetBestRMS on RemoveHs copies)"""
    try:
        return float(AllChem.GetBestRMS(Chem.RemoveHs(Chem.Mol(ref, confId=ref_cid)),
                                        Chem.RemoveHs(Chem.Mol(prb, confId=prb_cid))))
    except Exception:
        return np.nan


def set_acyclic_local(mol, cid, gt_conf, bonds, angles, with_bonds):
    """copy GT values of acyclic heavy-atom bond lengths (optional) and bond angles onto conformer cid (in place).
    Sequential SetAngle at branching centres cannot satisfy all 2n-3 constraints exactly -> achievable upper bound."""
    conf = mol.GetConformer(cid)
    ri = mol.GetRingInfo()
    if with_bonds:
        for i, j in bonds:
            if not ri.NumBondRings(mol.GetBondBetweenAtoms(i, j).GetIdx()):
                rdMolTransforms.SetBondLength(conf, i, j, rdMolTransforms.GetBondLength(gt_conf, i, j))
    for i, j, k in angles:
        ring_jk = ri.NumBondRings(mol.GetBondBetweenAtoms(j, k).GetIdx()) > 0
        ring_ji = ri.NumBondRings(mol.GetBondBetweenAtoms(j, i).GetIdx()) > 0
        if ring_jk and ring_ji:
            continue  # ring angle: belongs to a ring factor, not to an acyclic angle factor
        a, b, c = (i, j, k) if not ring_jk else (k, j, i)  # SetAngle moves the side of c; bond b-c must be acyclic
        try:
            rdMolTransforms.SetAngleDeg(conf, a, b, c, rdMolTransforms.GetAngleDeg(gt_conf, i, j, k))
        except Exception:
            pass


def stereo_labels(mol, cid=-1):
    m = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(m, confId=cid)
    centers = tuple(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
    dbl = tuple((b.GetIdx(), str(b.GetStereo())) for b in m.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE)
    return centers, dbl


# ----------------------------------------------------------------------------------------------- per-molecule job
def clean_confs(smi, confs):
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    return [c for c in confs if Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False) == smi]


def job(item):
    key, corrected, confs = item
    rows = []
    try:
        confs = clean_confs(corrected, confs)[:args.max_gt_confs]
        if not confs:
            return rows
        n = len(confs)
        K = max(n, args.seeds_per_gt * n)
        base = copy.deepcopy(confs[0])
        all_t, rel_t = relevant_torsions(base)
        bonds, angles = bonds_angles(base, heavy_only=True)
        bonds_h, angles_h = bonds_angles(base, heavy_only=False)
        amap = heavy_automorphisms(base)  # [metric-verifier fix] symmetry-aware objective (was heavy_map: identity)

        seeds = copy.deepcopy(base)
        seeds.RemoveAllConformers()
        cids = list(AllChem.EmbedMultipleConfs(seeds, numConfs=K, randomSeed=args.seed))
        if len(cids) != K:
            return [dict(smiles=key, corrected_smiles=corrected, error='embed_failed')]
        if args.mmff_seeds:
            AllChem.MMFFOptimizeMoleculeConfs(seeds, mmffVariant='MMFF94s')

        gt_loc = [local_values(c.GetConformer(), bonds, angles) for c in confs]
        gt_loc_h = [local_values(c.GetConformer(), bonds_h, angles_h) for c in confs]
        sd_loc = [local_values(seeds.GetConformer(c), bonds, angles) for c in cids]
        sd_loc_h = [local_values(seeds.GetConformer(c), bonds_h, angles_h) for c in cids]
        gt_tors = [np.array([rdMolTransforms.GetDihedralRad(c.GetConformer(), *t) for t in rel_t]) for c in confs]

        def transplant(src_mol, src_cid, i):
            tmp = Chem.Mol(src_mol, confId=src_cid)
            for t, v in zip(rel_t, gt_tors[i]):
                rdMolTransforms.SetDihedralRad(tmp.GetConformer(), *t, float(v))
            return tmp

        # transplant cost: GT torsions of conformer i on seed j (all K seeds) and on GT conformer j != i
        cost = np.array([[heavy_rmsd(transplant(seeds, cid, i), gt, -1, -1, amap) for cid in cids]
                         for i, gt in enumerate(confs)])
        cost_gg = np.array([[heavy_rmsd(transplant(confs[j], -1, i), gt, -1, -1, amap) if j != i else np.inf
                             for j in range(n)] for i, gt in enumerate(confs)])
        row_ind, col_ind = linear_sum_assignment(cost[:, :n])  # training analogue: L seeds, one-to-one

        gt_stereo = stereo_labels(confs[0])
        for i, gt in enumerate(confs):
            j = int(col_ind[i])
            tmp = Chem.Mol(seeds, confId=cids[j])
            floor = torsion_floor(tmp, -1, gt, rel_t, amap, x_extra=[gt_tors[i]])
            # test-time analogue: best of all K seeds (DE on the top-m by transplant cost), symmetry-aware re-score
            best, best_sym = np.inf, np.inf
            for jj in np.argsort(cost[i])[:args.floor_topm]:
                t2 = Chem.Mol(seeds, confId=cids[int(jj)])
                v = torsion_floor(t2, -1, gt, rel_t, amap, x_extra=[gt_tors[i]])
                best = min(best, v)
                best_sym = min(best_sym, v, np.nan_to_num(sym_rmsd(t2, gt), nan=np.inf))
            jb = int(np.argmin(cost[i]))
            # perfect-but-independent L sampler: the other GT conformers' local structures as seeds
            gg = np.nan
            if n > 1:
                gg = np.inf
                for jj in np.argsort(cost_gg[i])[:args.floor_topm]:
                    t3 = Chem.Mol(confs[int(jj)])
                    gg = min(gg, torsion_floor(t3, -1, gt, rel_t, amap, x_extra=[gt_tors[i]]))
            # acyclic local-structure oracle on the best seed
            oracles = {}
            for name, wb in (('angle_oracle_rmsd', False), ('local_oracle_rmsd', True)):
                t4 = Chem.Mol(seeds, confId=cids[jb])
                set_acyclic_local(t4, -1, gt.GetConformer(), bonds, angles, with_bonds=wb)
                oracles[name] = torsion_floor(t4, -1, gt, rel_t, amap, x_extra=[gt_tors[i]])
            bl_gt, ba_gt = gt_loc[i]
            bl_sd, ba_sd = sd_loc[j]
            blh_gt, bah_gt = gt_loc_h[i]
            blh_sd, bah_sd = sd_loc_h[j]
            others = [k for k in range(n) if k != i]
            rows.append(dict(
                smiles=key, corrected_smiles=corrected, gt_idx=i, n_gt=n, n_seeds=K,
                n_heavy=base.GetNumHeavyAtoms(), n_atoms=base.GetNumAtoms(),
                n_torsions_repo=len(all_t), n_torsions_heavy=len(rel_t),
                n_ring_atoms=sum(1 for a in base.GetAtoms() if a.GetAtomicNum() > 1 and a.IsInRing()),
                min_ring_size=min((len(r) for r in base.GetRingInfo().AtomRings()), default=0),
                bond_rmsd_assigned=rms(bl_sd - bl_gt), angle_rmsd_assigned=rms(ba_sd - ba_gt),
                bond_rmsd_meanseed=float(np.mean([rms(s[0] - bl_gt) for s in sd_loc])),
                angle_rmsd_meanseed=float(np.mean([rms(s[1] - ba_gt) for s in sd_loc])),
                bond_rmsd_allatom_assigned=rms(blh_sd - blh_gt), angle_rmsd_allatom_assigned=rms(bah_sd - bah_gt),
                bond_rmsd_gtgt=float(np.mean([rms(gt_loc[k][0] - bl_gt) for k in others])) if others else np.nan,
                angle_rmsd_gtgt=float(np.mean([rms(gt_loc[k][1] - ba_gt) for k in others])) if others else np.nan,
                transplant_rmsd_assigned=float(cost[i, j]), transplant_rmsd_best=float(cost[i].min()),
                floor_rmsd=floor, floor_best=float(best), floor_best_sym=float(best_sym),
                floor_gtother=float(gg), **oracles,
                stereo_match=stereo_labels(seeds, cids[j]) == gt_stereo,
                stereo_match_best=stereo_labels(seeds, cids[jb]) == gt_stereo,
            ))
    except Exception as e:  # keep going on odd molecules
        rows.append(dict(smiles=key, corrected_smiles=corrected, error=repr(e)[:200]))
    return rows


def run_test():
    df = pd.read_csv(args.test_csv)
    with open(args.true_mols, 'rb') as f:
        true_mols = pickle.load(f)
    items = []
    for row in df.itertuples(index=False):
        key, corrected = row[0], row[2]
        if 'smiles' in df.columns:
            key = getattr(row, 'smiles')
        if 'corrected_smiles' in df.columns:
            corrected = getattr(row, 'corrected_smiles')
        if key in true_mols:
            items.append((key, corrected, true_mols[key]))
    if args.limit_mols:
        items = items[:args.limit_mols]
    print('molecules:', len(items))
    return pd.DataFrame(pool_map(job, items, chunksize=4))


# ----------------------------------------------------------------------------------------------- run-matched floor
def heavy_idx(m):
    return [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]


def run_job(item):
    key, corrected, gts, gens, M = item
    rows = []
    try:
        gen0 = gens[0]
        _, rel_t = relevant_torsions(gen0)
        gh, th = heavy_idx(gen0), heavy_idx(gts[0])
        # same matching GetBestRMS uses internally (ref.GetSubstructMatches(prb, uniquify=False))
        matches = Chem.RemoveHs(gts[0]).GetSubstructMatches(Chem.RemoveHs(gen0), uniquify=False, maxMatches=1000)
        if not matches:
            return [dict(smiles=key, corrected_smiles=corrected, error='no_substruct_match')]
        amaps = [[(gh[p], th[q]) for p, q in enumerate(mt)] for mt in matches]
        for l, gt in enumerate(gts):
            if np.isnan(M[l]).all():
                continue
            obs = float(np.nanmin(M[l]))
            floor = obs
            for k in np.argsort(np.nan_to_num(M[l], nan=np.inf))[:args.floor_topm]:
                g = Chem.Mol(gens[int(k)])
                x_cur = np.array([rdMolTransforms.GetDihedralRad(g.GetConformer(), *t) for t in rel_t])
                # [metric-verifier fix] optimise the symmetry-aware objective (min over all maps), not only the
                # permutation that was best at the start
                v = torsion_floor(g, -1, gt, rel_t, amaps, x_extra=[x_cur])
                floor = min(floor, v, np.nan_to_num(sym_rmsd(g, gt), nan=np.inf))
            rows.append(dict(smiles=key, corrected_smiles=corrected, gt_idx=l, n_gt=len(gts), n_gen=len(gens),
                             n_torsions_heavy=len(rel_t), obs_min_rmsd=obs, floor_run=float(floor),
                             torsion_headroom=obs - float(floor)))
    except Exception as e:
        rows.append(dict(smiles=key, corrected_smiles=corrected, error=repr(e)[:200]))
    return rows


def run_run():
    with open(args.results, 'rb') as f:
        R = pickle.load(f)
    with open(args.confs, 'rb') as f:
        gen = pickle.load(f)
    with open(args.true_mols, 'rb') as f:
        true_mols = pickle.load(f)
    items, n_skip = [], 0
    for (key, corrected), res in R['results'].items():
        if corrected not in gen or key not in true_mols:
            n_skip += 1
            continue
        gts = clean_confs(corrected, true_mols[key])  # same order as the rows of res['rmsd'] (evaluate_confs.py)
        if len(gts) != res['rmsd'].shape[0]:
            n_skip += 1
            continue
        cap = args.max_gt_confs
        items.append((key, corrected, gts[:cap], gen[corrected], res['rmsd'][:cap]))
    if args.limit_mols:
        items = items[:args.limit_mols]
    print('molecules:', len(items), 'skipped:', n_skip)
    return pd.DataFrame(pool_map(run_job, items, chunksize=2))


def pool_map(fn, items, chunksize):
    rows = []
    if args.n_workers > 1:
        with Pool(args.n_workers) as p:
            for r in p.imap_unordered(fn, items, chunksize=chunksize):
                rows.extend(r)
    else:
        for it in items:
            rows.extend(fn(it))
    return rows


def run_std():
    rows = []
    for path in sorted(glob.glob(os.path.join(args.std_pickles, '*.pickle'))):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        for name, mol_dic in d.items():
            for c in mol_dic['conformers']:
                rows.append(dict(pickle=os.path.basename(path), name=name, smiles=mol_dic['smiles'],
                                 n_heavy=c['rd_mol'].GetNumHeavyAtoms(), n_rot=c.get('num_rotable_bonds'),
                                 match_rmsd=c.get('rmsd'), boltzmannweight=c.get('boltzmannweight')))
        if args.limit_mols and len(rows) > args.limit_mols * 30:
            break
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- leak test (round 2)
def leak_job(item):
    """[round2 S0 (e)] code_plan_2 §2.1 item 4 / verify_B §4 R2-0(e): in TORSION space, is generated conformer i
    nearest to its L-source GT conformer (i mod L under --seed_confs_cycle, diffusion/sampling.py:75)?"""
    key, corrected, gts, gens, seed_mols, w = item
    n_seed = len(seed_mols)
    L = len(gts)
    # FIXES X12 (research_check_B I7): the statistic assumes generated conformer i was seeded by seed i mod L
    # (diffusion/sampling.py:75, cycle) and that confs.pkl keeps that order. Torsion updates are rigid rotations, so
    # every bond length of generated conformer i must equal that of its seed; assert it instead of assuming it.
    bl = lambda m: np.array([rdMolTransforms.GetBondLength(m.GetConformer(), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                             for b in m.GetBonds()])
    for i, g in enumerate(gens):
        dev = float(np.abs(bl(g) - bl(seed_mols[i % n_seed])).max())
        assert dev < 1e-3, f'{key}: generated conformer {i} does not keep the bond lengths of seed {i % n_seed} ' \
                           f'(max dev {dev:.4f} A): seed order lost, leak test invalid'
    if L < 2 or n_seed != L:
        return [dict(smiles=key, corrected_smiles=corrected, error='L<2' if L < 2 else f'n_seed {n_seed} != L {L}')]
    _, rel_t = relevant_torsions(gts[0])
    if not rel_t:
        return [dict(smiles=key, corrected_smiles=corrected, error='no heavy torsion')]
    maps = heavy_automorphisms(gts[0])
    # torsion index tuples under every heavy automorphism (symmetric groups give equivalent torsion definitions)
    tmaps = []
    for mp in maps:
        d = dict(mp)
        if all(all(a in d for a in t) for t in rel_t):
            tmaps.append([tuple(d[a] for a in t) for t in rel_t])
    tmaps = tmaps or [rel_t]
    tg = lambda m, ts: np.array([rdMolTransforms.GetDihedralRad(m.GetConformer(), *t) for t in ts])
    T_gt = [tg(g, rel_t) for g in gts]
    D = np.zeros((len(gens), L))
    for i, g in enumerate(gens):
        cand = [tg(g, ts) for ts in tmaps]
        for l in range(L):
            D[i, l] = min(np.sqrt(np.mean(((c - T_gt[l] + np.pi) % (2 * np.pi) - np.pi) ** 2)) for c in cand)
    src = np.arange(len(gens)) % L
    near = D.argmin(1)
    hit = float(np.mean(near == src))
    rng = np.random.default_rng(0)
    null = float(np.mean([np.mean(near == rng.permutation(src)) for _ in range(args.n_perm)]))
    boltz = float(np.mean([w[src[i]] for i in range(len(gens))])) if w is not None else np.nan
    return [dict(smiles=key, corrected_smiles=corrected, L=L, n_gen=len(gens), hit=hit, perm_null=null,
                 boltz_null=boltz, excess=hit - null)]


def run_leak():
    df = pd.read_csv(args.test_csv)
    with open(args.true_mols, 'rb') as f:
        true_mols = pickle.load(f)
    with open(args.confs, 'rb') as f:
        gen = pickle.load(f)
    with open(args.seed_confs, 'rb') as f:
        seeds = pickle.load(f)
    items = []
    for row in df.itertuples(index=False):
        raw_smi, corrected = row[0], row[2]
        key = getattr(row, 'smiles') if 'smiles' in df.columns else raw_smi
        if corrected not in gen or key not in true_mols or raw_smi not in seeds:
            continue
        gts = clean_confs(corrected, true_mols[key])
        w = None
        rp = os.path.join(args.raw_dir, key + '.pickle')
        if os.path.exists(rp):  # Boltzmann weights by coordinate match to the raw GEOM conformers
            try:
                with open(rp, 'rb') as f:
                    raw = pickle.load(f)['conformers']
                ww = []
                for g in gts:
                    P = g.GetConformer().GetPositions()
                    m = [r['boltzmannweight'] for r in raw if r['rd_mol'].GetNumAtoms() == len(P) and
                         np.allclose(r['rd_mol'].GetConformer().GetPositions(), P, atol=1e-3)]
                    ww.append(m[0] if m else np.nan)
                w = np.array(ww) / np.nansum(ww) if not np.isnan(ww).any() else None
            except Exception:
                w = None
        items.append((key, corrected, gts, gen[corrected], seeds[raw_smi], w))
    if args.limit_mols:
        items = items[:args.limit_mols]
    print('molecules:', len(items))
    return pd.DataFrame(pool_map(leak_job, items, chunksize=4))


if __name__ == '__main__':
    np.random.seed(args.seed)
    df = {'test': run_test, 'run': run_run, 'std': run_std, 'leak': run_leak}[args.mode]()
    if args.mode == 'leak' and 'hit' in df:
        d = df.dropna(subset=['hit'])
        print(f'LEAK n={len(d)} hit={d.hit.mean():.4f} perm_null={d.perm_null.mean():.4f} '
              f'excess={d.excess.mean():.4f} boltz_null={d.boltz_null.mean():.4f}')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = {'test': ['floor_rmsd', 'floor_best', 'floor_best_sym', 'floor_gtother', 'angle_oracle_rmsd',
                     'local_oracle_rmsd', 'transplant_rmsd_best'],
            'run': ['obs_min_rmsd', 'floor_run', 'torsion_headroom'], 'std': ['match_rmsd'],
            'leak': ['hit', 'perm_null', 'excess']}[args.mode]
    if 'error' in df:
        print('rows with error:', int(df['error'].notna().sum()))
    for col in cols:
        if col not in df:
            continue
        d = df[['corrected_smiles', col]].replace([np.inf, -np.inf], np.nan).dropna() if 'corrected_smiles' in df \
            else df[[col]].dropna()
        v = d[col].values
        if not len(v):
            continue
        pm = d.groupby('corrected_smiles')[col].mean().mean() if 'corrected_smiles' in d else np.nan
        # per-molecule mean of per-GT values = the AMR-R-comparable number
        print(f'{col}: n={len(v)} mean={v.mean():.4f} median={np.median(v):.4f} per-molecule-mean={pm:.4f}')
        for thr in (0.05, 0.1, 0.125, 0.25, 0.5, 0.75):
            print(f'  fraction > {thr} A: {(v > thr).mean():.4f}')
    if args.mode == 'test':
        for c in ('bond_rmsd_assigned', 'angle_rmsd_assigned', 'bond_rmsd_gtgt', 'angle_rmsd_gtgt'):
            if c in df:
                print(f'{c}: mean={df[c].mean():.4f}')
        for c in ('stereo_match', 'stereo_match_best'):
            if c in df:
                print(f'{c} rate (seeds embedded from the GT graph):', df[c].mean())
    print('wrote', args.out)

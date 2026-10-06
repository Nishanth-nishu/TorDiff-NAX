"""
Build standardized_pickles_paired: the EXISTING conformer-matched rematch pickles (CTRL_rematch training data, F4) with
each matched conformer's own GT conformer re-attached, Kabsch-aligned and terminal-relabelled (DECISION D1).
round2/code_plan_2.md §4; DECISION D1, D6, defect 1.

Why pair post hoc instead of re-standardizing: standardize_confs.py:79 embeds ETKDG unseeded, so a re-run would draw new
RDKit L. Pairing keeps the RDKit half of S3/S4/B1cap identical to CTRL_rematch's data, conformer for conformer.

Per std conformer c of molecule `name` (std key = raw file stem, standardize_confs.py:154):
  1. candidates = raw conformers with the same geom_id (checked on the cluster: QM9 raw and std conf dicts carry
     'geom_id'); fallback key (boltzmannweight, totalenergy). Defect 1 (vote_1 P2-b): ties are DISAMBIGUATED by the
     decisive check of step 2 instead of aborting; only a residual tie between candidates with different coordinates
     counts as 'ambiguous'.
  2. pairing check: AlignMol(RemoveHs(matched), RemoveHs(gt)) must reproduce the stored conf['rmsd'], which
     standardize_confs.py:108 computed with exactly this call (|diff| < 1e-4 A).
  3. same labelled graph (atom order + bonds; code_plan_2 V1).
  4. lgeom.align_pair (heavy-atom Kabsch applied to all atoms + terminal relabelling) and lgeom.pair_check at
     lambda = 0.5 (stereo / bond / clash). Failing pairs are DROPPED and counted (DECISION D1).
Stored: conf['gt_pos_aligned'] (float32, RDKit frame), conf['pair_heavy_rmsd'], conf['n_swaps'], conf['pair_ok'],
conf['pair_reason']. rd_mol is untouched. Only verification failures (steps 1-3) are dropped; pair_ok failures are kept
and flagged (see the comment in pair_molecule).
Molecules left with no conformer are dropped and counted. Per-file stats go to <out_dir>/stats/NNN.json and
--summarize prints the pairing-clean percentage.

Usage (repo root, CPU):
  python ../tools/build_paired_pickles.py --std_dir data/QM9/standardized_pickles_rematch --raw_dir data/QM9/qm9/ \
      --out_dir data/QM9/standardized_pickles_paired --n_workers 16
  python ../tools/build_paired_pickles.py --out_dir data/QM9/standardized_pickles_paired --summarize
"""
import glob
import json
import os
import pickle
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lgeom  # noqa: E402

RDLogger.DisableLog('rdApp.*')
RMSD_TOL = 1e-4

parser = ArgumentParser()
parser.add_argument('--std_dir', default='data/QM9/standardized_pickles_rematch')
parser.add_argument('--raw_dir', default='data/QM9/qm9/')
parser.add_argument('--out_dir', required=True)
parser.add_argument('--files', nargs='*', default=None, help='only these NNN.pickle basenames (smoke test)')
parser.add_argument('--limit_mols', type=int, default=0, help='per file (smoke test)')
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--summarize', action='store_true')
args = parser.parse_args()


def raw_key(c):
    return (round(float(c.get('boltzmannweight', np.nan)), 12), round(float(c.get('totalenergy', np.nan)), 9))


def pair_molecule(name, mol_dic, cnt, dev_hist):
    raw_path = os.path.join(args.raw_dir, name + '.pickle')
    if not os.path.exists(raw_path):
        cnt['mol_raw_missing'] += 1
        return None
    with open(raw_path, 'rb') as f:
        raw = pickle.load(f)['conformers']
    by_id, by_key = defaultdict(list), defaultdict(list)
    for r in raw:
        if 'geom_id' in r:
            by_id[r['geom_id']].append(r)
        by_key[raw_key(r)].append(r)
    new = []
    for c in mol_dic['conformers']:
        cnt['confs_in'] += 1
        cands = by_id.get(c.get('geom_id'), []) if 'geom_id' in c else []
        if not cands:
            cands = by_key.get(raw_key(c), [])
            cnt['fallback_key_used'] += 1
        if not cands:
            cnt['drop_no_candidate'] += 1
            continue
        m_h = lgeom.REMOVE_HS(c['rd_mol'])
        hits = []
        for r in cands:
            rmsd = AllChem.AlignMol(Chem.Mol(m_h), lgeom.REMOVE_HS(r['rd_mol']))  # == standardize_confs.py:108
            if abs(rmsd - c['rmsd']) < RMSD_TOL:
                hits.append(r)
        # defect 1: duplicates with identical coordinates are equivalent; only genuinely different ones are ambiguous
        uniq = []
        for r in hits:
            P = lgeom.positions(r['rd_mol'])
            if not any(P.shape == lgeom.positions(u['rd_mol']).shape and
                       np.allclose(P, lgeom.positions(u['rd_mol']), atol=1e-6) for u in uniq):
                uniq.append(r)
        if len(cands) > 1:
            cnt['multi_candidate'] += 1
        if not uniq:
            cnt['drop_rmsd_mismatch'] += 1
            continue
        if len(uniq) > 1:
            cnt['drop_ambiguous'] += 1
            continue
        gt = uniq[0]['rd_mol']
        if not lgeom.same_graph([c['rd_mol'], gt]):
            cnt['drop_graph_mismatch'] += 1
            continue
        X, Y = lgeom.positions(c['rd_mol']), lgeom.positions(gt)
        al = lgeom.align_pair(c['rd_mol'], X, Y)
        chk = lgeom.pair_check(c['rd_mol'], X, al['Y_al'])
        dev_hist.append((chk['worst_bond_dev'], chk['min_nonbonded'], al['n_swaps']))
        # DECISION D1: pairs failing pair_ok must not be INTERPOLATED. They are kept here with pair_ok=False and
        # counted; the S4 loader drops them (utils/dataset.py, --l_interp), while S3/B1cap (no midpoints) keep every
        # verified pair so that their RDKit half stays identical to CTRL_rematch's data (F4).
        if not chk['pair_ok']:
            cnt[f'unsafe_pair_{chk["reason"]}'] += 1
        cnt['confs_paired'] += 1
        cnt['confs_pair_ok'] += int(chk['pair_ok'])
        cnt['confs_with_swaps'] += int(al['n_swaps'] > 0)
        c = dict(c)
        c['gt_pos_aligned'] = al['Y_al'].astype(np.float32)
        c['pair_heavy_rmsd'] = al['heavy_rmsd']
        c['n_swaps'] = al['n_swaps']
        c['pair_ok'] = bool(chk['pair_ok'])
        c['pair_reason'] = chk['reason']
        new.append(c)
    if not new:
        cnt['mol_dropped_all_confs'] += 1
        return None
    if len(new) < len(mol_dic['conformers']):
        cnt['mol_partially_paired'] += 1
    out = dict(mol_dic)
    out['conformers'] = new
    return out


def run_file(path):
    base = os.path.basename(path)
    out_path = os.path.join(args.out_dir, base)
    stat_path = os.path.join(args.out_dir, 'stats', base.replace('.pickle', '.json'))
    if os.path.exists(out_path) and os.path.exists(stat_path):
        return base, 'exists'
    with open(path, 'rb') as f:
        std = pickle.load(f)
    cnt, dev, out = Counter(), [], {}
    for i, (name, mol_dic) in enumerate(std.items()):
        if args.limit_mols and i >= args.limit_mols:
            break
        cnt['mols_in'] += 1
        try:
            r = pair_molecule(name, mol_dic, cnt, dev)
        except Exception as e:  # keep going; counted
            print('ERROR', name, repr(e)[:200], flush=True)
            cnt['mol_exception'] += 1
            r = None
        if r is not None:
            out[name] = r
            cnt['mols_out'] += 1
    dev = np.array(dev) if dev else np.zeros((0, 3))
    stats = dict(counts=dict(cnt),
                 worst_bond_dev_q=np.quantile(dev[:, 0], [0.5, 0.9, 0.99, 1.0]).tolist() if len(dev) else [],
                 min_nonbonded_q=np.quantile(dev[:, 1][np.isfinite(dev[:, 1])], [0, 0.01, 0.1, 0.5]).tolist()
                 if len(dev) and np.isfinite(dev[:, 1]).any() else [])
    tmp = out_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(out, f)
    os.replace(tmp, out_path)
    with open(stat_path, 'w') as f:
        json.dump(stats, f)
    return base, cnt


def summarize():
    tot = Counter()
    for p in sorted(glob.glob(os.path.join(args.out_dir, 'stats', '*.json'))):
        tot.update(json.load(open(p))['counts'])
    n_in, n_ok = tot['confs_in'], tot['confs_paired']
    print('files:', len(glob.glob(os.path.join(args.out_dir, 'stats', '*.json'))))
    for k in sorted(tot):
        print(f'  {k}: {tot[k]}')
    pct = 100.0 * n_ok / max(n_in, 1)
    print(f'PAIRING_CLEAN confs {n_ok}/{n_in} = {pct:.3f} %  molecules {tot["mols_out"]}/{tot["mols_in"]}')
    pok = 100.0 * tot['confs_pair_ok'] / max(n_ok, 1)
    print(f'PAIR_OK confs {tot["confs_pair_ok"]}/{n_ok} = {pok:.3f} % of verified pairs (S4 interpolates only these)')
    print('drops by reason:', {k: v for k, v in tot.items() if k.startswith('drop_')})
    print('unsafe (kept, not interpolated) by reason:', {k: v for k, v in tot.items() if k.startswith('unsafe_')})
    return pct


if __name__ == '__main__':
    if args.summarize:
        summarize()
        sys.exit(0)
    os.makedirs(os.path.join(args.out_dir, 'stats'), exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.std_dir, '*.pickle')))
    if args.files:
        files = [f for f in files if os.path.basename(f) in set(args.files)]
    print(len(files), 'std files', flush=True)
    if args.n_workers > 1:
        with Pool(args.n_workers) as p:
            for base, c in p.imap_unordered(run_file, files):
                print(base, dict(c) if not isinstance(c, str) else c, flush=True)
    else:
        for f in files:
            base, c = run_file(f)
            print(base, dict(c) if not isinstance(c, str) else c, flush=True)
    summarize()

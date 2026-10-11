"""Impact analyst (round 3, P3): L-error profile per test-time L condition, overall and per n_rot_heavy bin.

Input : cluster_sync/round2/data/QM9/round2_seeds/l_error.csv (one row per seed paired with its source GT conformer;
        ETKDG / MMFF rows are the angle-Hungarian-assigned seed per GT conformer, make_l_seed_pickles.py:106-114).
Output: lerror_profile.csv (cond x stratum), printed table.
Grouping (same as V3, verify_V3.md R1-R2): unit = csv row, mean over rows within (cond, stratum), NaN skipped.
  ring molecule  = RDKit SSSR of the csv SMILES has >= 1 ring; 'ring>=4' = ring molecule with a ring of >= 4 atoms
                   (the 'excl. all-3-ring' basis); tail10 = share of rows with ring_dihedral_rmsd > 10 deg.
  n_rot_heavy    = tools/breakdown.py rot_bond_counts on the csv (raw) SMILES (breakdown.log uses the corrected
                   SMILES, so bins can differ for a few molecules).
Usage: python round3/impact_data/lerror_profile.py
"""
import os

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, 'cluster_sync', 'round2', 'data', 'QM9', 'round2_seeds', 'l_error.csv')


def rot_heavy(smi):  # tools/breakdown.py:33-50, heavy count only
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.nan
    m = Chem.AddHs(m)
    G = nx.Graph(); G.add_nodes_from(range(m.GetNumAtoms()))
    G.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds())
    heavy = {a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1}
    n = 0
    for e in list(G.edges()):
        G2 = G.copy(); G2.remove_edge(*e)
        if nx.is_connected(G2):
            continue
        comps = list(nx.connected_components(G2))
        if min(len(c) for c in comps) < 2:
            continue
        n += all(len(c & heavy) >= 2 for c in comps)
    return n


def ring_info(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return dict(has_ring=False, ring_ge4=False, min_ring=0)
    rings = m.GetRingInfo().AtomRings()
    if not rings:
        return dict(has_ring=False, ring_ge4=False, min_ring=0)
    sizes = [len(r) for r in rings]
    return dict(has_ring=True, ring_ge4=max(sizes) >= 4, min_ring=min(sizes))


def main():
    df = pd.read_csv(CSV)
    smis = df.smiles.unique()
    info = pd.DataFrame([dict(smiles=s, rot=rot_heavy(s), **ring_info(s)) for s in smis])
    info['bin'] = info.rot.clip(upper=4).map(lambda x: '4+' if x >= 4 else (str(int(x)) if x == x else 'nan'))
    info.to_csv(os.path.join(OUT, 'mol_info.csv'), index=False)
    df = df.merge(info, on='smiles')
    df['tail10'] = (df.ring_dihedral_rmsd > 10).astype(float).where(df.ring_dihedral_rmsd.notna())
    df['tail20'] = (df.ring_dihedral_rmsd > 20).astype(float).where(df.ring_dihedral_rmsd.notna())

    def prof(g):
        r = g[g.has_ring]
        r4 = g[g.ring_ge4]
        return pd.Series(dict(
            n_rows=len(g), n_mols=g.smiles.nunique(),
            ring_bond=r.bond_rmsd_heavy_ring.mean(), ring_angle=r.angle_rmsd_heavy_ring.mean(),
            ringdih_mean=r.ring_dihedral_rmsd.mean(), ringdih_mean_ge4=r4.ring_dihedral_rmsd.mean(),
            tail10_all=r.tail10.mean(), tail10_ge4=r4.tail10.mean(), tail20_ge4=r4.tail20.mean(),
            acyc_bond=g.bond_rmsd_heavy_acyc.mean(), acyc_angle=g.angle_rmsd_heavy_acyc.mean(),
            any_bond=g.bond_rmsd_heavy_any.mean(), any_angle=g.angle_rmsd_heavy_any.mean()))

    rows = []
    for cond, g in df.groupby('cond'):
        rows.append(dict(cond=cond, stratum='all', **prof(g)))
        for b, gb in g.groupby('bin'):
            rows.append(dict(cond=cond, stratum=f'rot{b}', **prof(gb)))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, 'lerror_profile.csv'), index=False)
    pd.set_option('display.width', 250)
    print(P[P.stratum.isin(['all', 'rot0'])].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    # molecule counts per bin (rigid share)
    print('\nmolecules per n_rot_heavy bin (all l_error SMILES):', info.bin.value_counts().to_dict())
    print('ring molecules:', int(info.has_ring.sum()), ' ring>=4:', int(info.ring_ge4.sum()),
          ' all-3-ring:', int((info.has_ring & ~info.ring_ge4).sum()))
    r0 = info[info.bin == '0']
    print('rigid: n=%d, ring>=4 %d, all-3-ring %d, acyclic %d' % (len(r0), r0.ring_ge4.sum(),
          (r0.has_ring & ~r0.ring_ge4).sum(), (~r0.has_ring).sum()))


if __name__ == '__main__':
    main()

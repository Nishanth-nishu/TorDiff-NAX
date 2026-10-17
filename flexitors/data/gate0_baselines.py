"""Gate-0 validation seed sets and baselines (EXP-1 §10.3; round5/DECISION.md fixes 5-6).

Gate 0 (M1) is decided on the QM9 validation subset (EVAL §1.1), so its two baselines are computed there first, on
CPU, with exactly the recipe of the published test numbers:
* seeds: ``tools/make_l_seed_pickles.py --sources etkdg`` (K = 2 n_csv ETKDG seeds embedded from the GT graph, and
  the same embeddings MMFF94s-relaxed; each GT conformer paired with one seed by a Hungarian assignment on heavy-atom
  angle RMSD), giving ``L_etkdg2L`` / ``L_etkdg2L_mmff`` and ``l_error.csv``;
* rigid-stratum AMR-R: the seeds scored directly by the unchanged ``evaluate_confs.py`` (heavy-atom RMSD does not
  depend on torsions when n_rot_heavy = 0, E-JOINT-007), stratum from ``tools/breakdown.py`` (n_rot_heavy = 0,
  DECISION fix 4); published test values ETKDG 0.152 Å, MMFF 0.128 Å (E-JOINT-007);
* ring metric: share of paired seeds of molecules whose largest ring has ≥ 4 atoms with endocyclic ring-dihedral RMSD
  > 10° (E-JOINT-004 rev.); published test values ETKDG 49.6 %, MMFF 35.1 % on 4,181 seeds.
Thresholds (EXP-1 §10.3): rigid AMR-R ≤ 0.10 Å and ring share ≤ 20 %; if a validation ETKDG baseline differs from its
test value by more than 10 % relative, that threshold is scaled by validation ÷ test. The test value is the one
computed here with the SAME recipe (review P2 B2): the published 0.152 Å comes from a CTRL_rematch generation run
(E-JOINT-007), a different recipe, so it is only compared with the same-recipe test value (``recipe_check``, flagged
if they differ by more than the same 10 %).

Subcommands: ``confs`` (seed pickle keyed by the csv ``smiles`` column → generated-conformer pickle keyed by
``corrected_smiles``, the evaluator's format) and ``summary``.
"""
from __future__ import annotations

import json
import pickle
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from flexitors.provenance import write_provenance

RDLogger.DisableLog('rdApp.*')
SOURCES = {'L_etkdg2L': 'ETKDG', 'L_etkdg2L_mmff': 'MMFF'}
PUBLISHED_TEST = {'rigid_amr_r': {'ETKDG': 0.152, 'MMFF': 0.128},      # E-JOINT-007
                  'ring_share_gt10': {'ETKDG': 0.496, 'MMFF': 0.351}}   # E-JOINT-004 rev.
THRESHOLDS = {'rigid_amr_r': 0.10, 'ring_share_gt10': 0.20}            # EXP-1 §10.3
SCALE_IF_REL_DIFF = 0.10                                               # EXP-1 §10.3 scaling rule
RING_DEG = 10.0


def seeds_to_confs(seeds: dict, csv: pd.DataFrame) -> dict:
    """Re-key a seed pickle (csv ``smiles`` column) by ``corrected_smiles``, as ``evaluate_confs.py`` reads model
    output; molecules without seeds stay absent (they count as model failures, the TD convention, EVAL §3)."""
    return {r.corrected_smiles: seeds[r.smiles] for r in csv.itertuples() if r.smiles in seeds}


def largest_ring(smiles: str) -> int:
    """Largest SSSR ring size of the molecule (0 if acyclic or unparsable), from the SMILES as in E-JOINT-004 rev."""
    m = Chem.MolFromSmiles(smiles)
    return max((len(r) for r in m.GetRingInfo().AtomRings()), default=0) if m is not None else 0


def ring_seeds(l_error: pd.DataFrame, cond: str) -> pd.DataFrame:
    """Paired seeds (smiles, k, src_gt_idx) of molecules whose largest ring has ≥ 4 atoms, with their ring error."""
    E = l_error[l_error.cond == cond]
    big = {s for s in E.smiles.unique() if largest_ring(s) >= 4}
    return E[E.smiles.isin(big)][['smiles', 'k', 'src_gt_idx', 'ring_dihedral_rmsd']].reset_index(drop=True)


def rigid_amr(breakdown_csv: Path) -> tuple[float, int]:
    """Mean AMR-R over the n_rot_heavy = 0 molecules of a ``tools/breakdown.py`` per-molecule CSV, and their count."""
    B = pd.read_csv(breakdown_csv)
    r = B[B.n_rot_heavy == 0]
    return float(r.mat_r.mean()), int(len(r))


def thresholds(val: dict, test: dict) -> dict:
    """EXP-1 §10.3 thresholds, scaled by validation ÷ test where the ETKDG baselines differ by > 10 % relative."""
    out = {}
    for key, thr in THRESHOLDS.items():
        ratio = val[key]['ETKDG'] / test[key]['ETKDG']
        scaled = abs(ratio - 1) > SCALE_IF_REL_DIFF
        out[key] = dict(threshold=thr * ratio if scaled else thr, unscaled=thr, val_over_test=ratio, scaled=scaled)
    return out


def summarise(out_dir: Path, val_l_error: Path, test_l_error: Path) -> dict:
    """Baselines on validation and (golden check) test, the Gate-0 thresholds, and the validation ring-seed list."""
    res = {}
    for name, le in (('val', val_l_error), ('test', test_l_error)):
        E = pd.read_csv(le)
        res[name] = {'rigid_amr_r': {}, 'n_rigid': {}, 'ring_share_gt10': {}, 'n_ring_seeds': {}}
        for cond, src in SOURCES.items():
            res[name]['rigid_amr_r'][src], res[name]['n_rigid'][src] = rigid_amr(
                out_dir / f'{name}_{cond}' / 'breakdown.csv')
            R = ring_seeds(E, cond)
            res[name]['ring_share_gt10'][src] = float((R.ring_dihedral_rmsd > RING_DEG).mean())
            res[name]['n_ring_seeds'][src] = int(len(R))
            if name == 'val' and cond == 'L_etkdg2L':
                R.to_csv(out_dir / 'val_ring_seeds.csv', index=False)
    res['published_test'] = PUBLISHED_TEST
    res['gate0_thresholds'] = thresholds(res['val'], res['test'])
    res['recipe_check'] = {k: {src: dict(same_recipe_test=res['test'][k][src], published=pub,
                                         rel_diff=res['test'][k][src] / pub - 1,
                                         flagged=abs(res['test'][k][src] / pub - 1) > SCALE_IF_REL_DIFF)
                               for src, pub in v.items()} for k, v in PUBLISHED_TEST.items()}
    (out_dir / 'gate0_baselines.json').write_text(json.dumps(res, indent=1) + '\n')
    write_provenance(out_dir)
    return res


def main() -> None:
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('confs')
    c.add_argument('--seeds', required=True)
    c.add_argument('--csv', required=True)
    c.add_argument('--out', required=True)
    s = sub.add_parser('summary')
    s.add_argument('--dir', required=True)
    s.add_argument('--val_l_error', required=True)
    s.add_argument('--test_l_error', required=True)
    a = ap.parse_args()
    if a.cmd == 'confs':
        with open(a.seeds, 'rb') as f:
            confs = seeds_to_confs(pickle.load(f), pd.read_csv(a.csv))
        with open(a.out, 'wb') as f:
            pickle.dump(confs, f)
        print(f'{len(confs)} molecules -> {a.out}')
        return
    res = summarise(Path(a.dir), Path(a.val_l_error), Path(a.test_l_error))
    for name in ('val', 'test'):
        r = res[name]
        print(f'{name}: rigid AMR-R ETKDG {r["rigid_amr_r"]["ETKDG"]:.4f} / MMFF {r["rigid_amr_r"]["MMFF"]:.4f} A '
              f'(n = {r["n_rigid"]["ETKDG"]}); >= 4-ring seeds > 10 deg: ETKDG {100 * r["ring_share_gt10"]["ETKDG"]:.1f} '
              f'% / MMFF {100 * r["ring_share_gt10"]["MMFF"]:.1f} % (n = {r["n_ring_seeds"]["ETKDG"]})')
    print('published test:', res['published_test'])
    print('same-recipe test vs published:', json.dumps(res['recipe_check']))
    print('Gate-0 thresholds (validation, scaled by validation / same-recipe test):',
          json.dumps(res['gate0_thresholds']))


if __name__ == '__main__':
    main()

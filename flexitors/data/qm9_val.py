"""QM9 validation subset in the test-file format (EVAL §1.1; EXP-5 §12.4; EXP-1 §10.3; Q-P1-010, Q-P2-010, Q-P3-007).

Procedure (EVAL §1.1 items 1-5):
1. universe = index 1 of ``split.npy`` indexed into the sorted raw-pickle list, as
   ``torsional-diffusion/utils/dataset.py:165-179`` does (indices beyond the list are dropped and counted, as the
   round-1 hook there does);
2. order = ``np.random.default_rng(0).permutation(sorted(split[1]))``; walk it and keep the first 1000 molecules that
   pass the filters (RDKit parses ``corrected_smiles``; at least one conformer survives ``clean_confs``). Embedding
   and featurisation are NOT filtered: those failures are part of the protocol;
3. ground truth = the raw GEOM ``rd_mol`` of every conformer, uncapped;
4. ``smiles`` = the raw GEOM key (pickle stem); ``corrected_smiles`` and ``n_conformers`` follow whichever candidate
   rule reproduces the test files. The builder is first applied to the 1000 test rows (all of which are molecules of
   split index 2) and must reproduce ``test_smiles.csv`` and the conformer counts of ``test_mols.pkl`` for ≥ 99 % of
   rows (the acceptance test of Q-P2-010);
5. outputs: ``val_smiles.csv``, ``val_indices.csv``, ``val_mols.pkl`` (+ its sha256), ``population.txt``,
   ``self_check.txt``.
"""
from __future__ import annotations

import hashlib
import pickle
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from flexitors.geometry.torsion_types import heavy_rotatable_bonds, td_rotatable_bonds
from flexitors.provenance import write_provenance
from flexitors.td import import_baseline

RDLogger.DisableLog('rdApp.*')
ACCEPT = 0.99  # Q-P2-010 / EVAL §1.1 item 4


def clean_confs(smiles: str, confs: list) -> list:
    """The evaluator's GT filter (``evaluate_confs.py:47-54``), from its byte-identical copy in
    ``tools/make_split_half.py``."""
    return import_baseline('make_split_half').clean_confs(smiles, confs)


def _graph(smiles_or_mol: str | Chem.Mol) -> str | None:
    """Non-isomeric canonical SMILES of a SMILES string (parsed unsanitised) or a molecule; None if unparsable."""
    m = Chem.MolFromSmiles(smiles_or_mol, sanitize=False) if isinstance(smiles_or_mol, str) else smiles_or_mol
    return None if m is None else Chem.MolToSmiles(m, isomericSmiles=False)


def corrected_candidates(mol_dic: dict) -> dict[str, str]:
    """Candidate rules for ``corrected_smiles`` (EVAL §1.1 item 4, plus two combinations of them)."""
    geom = mol_dic['smiles']
    confs = [c['rd_mol'] for c in mol_dic['conformers']]
    try:
        conf0 = Chem.MolToSmiles(Chem.RemoveHs(confs[0]))
    except Exception:  # RDKit raises several types when a GEOM conformer graph does not sanitise
        conf0 = Chem.MolToSmiles(Chem.RemoveHs(confs[0], sanitize=False))
    same0 = _graph(geom) == _graph(Chem.RemoveHs(confs[0], sanitize=False))
    return {'geom': geom, 'conf0': conf0,
            'geom_if_conf0_graph': geom if same0 else conf0,
            'geom_if_any_graph': geom if clean_confs(geom, confs) else conf0}


def load_raw(raw_dir: Path, stem: str) -> dict:
    """The raw GEOM record of one molecule."""
    with open(Path(raw_dir) / f'{stem}.pickle', 'rb') as f:
        return pickle.load(f)


def raw_stems(raw_dir: Path) -> list[str]:
    """Raw pickle stems in the sorted order ``utils/dataset.py:170`` indexes with ``split.npy``."""
    return [p.name[:-len('.pickle')] for p in sorted(Path(raw_dir).glob('*.pickle'))]


def self_check(test_csv: Path, test_mols: Path, raw_dir: Path) -> tuple[str, pd.DataFrame]:
    """Apply every candidate rule to the test rows; return the best rule and the per-row comparison."""
    df = pd.read_csv(test_csv)
    with open(test_mols, 'rb') as f:
        true_mols = pickle.load(f)
    rows = []
    for r in df.itertuples(index=False):
        p = Path(raw_dir) / f'{r.smiles}.pickle'
        if not p.exists():
            rows.append(dict(smiles=r.smiles, raw_found=False))
            continue
        d = load_raw(raw_dir, r.smiles)
        confs = [c['rd_mol'] for c in d['conformers']]
        tm = true_mols.get(r.smiles, [])
        same_pos = len(tm) == len(confs) and all(
            np.array_equal(a.GetConformer().GetPositions(), b.GetConformer().GetPositions()) for a, b in zip(tm, confs))
        row = dict(smiles=r.smiles, raw_found=True, n_csv=int(r.n_conformers), n_stored=len(confs),
                   n_test_mols=len(tm), gt_positions_equal=same_pos,
                   n_clean=len(clean_confs(r.corrected_smiles, confs)),
                   parses=Chem.MolFromSmiles(r.corrected_smiles) is not None)
        for rule, smi in corrected_candidates(d).items():
            row[f'corr_{rule}'] = smi == r.corrected_smiles
        rows.append(row)
    S = pd.DataFrame(rows)
    base = S.raw_found & (S.n_stored == S.n_csv) & (S.n_test_mols == S.n_stored)
    score = {c[len('corr_'):]: float((base & S[c].fillna(False).astype(bool)).mean()) for c in S if c.startswith('corr_')}
    best = max(score, key=score.get)
    S['reproduced'] = base & S[f'corr_{best}'].astype(bool)
    S.attrs['scores'] = score
    S.attrs['count_scores'] = {'stored': float((S.n_csv == S.n_stored).mean()),
                               'clean': float((S.n_csv == S.n_clean).mean())}
    return best, S


def murcko(smiles: str) -> str | None:
    """Bemis-Murcko scaffold without chirality, as round5/ledger/sample_data/ood.py:13-16."""
    m = Chem.MolFromSmiles(smiles)
    return None if m is None else MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def describe(smiles: str, corrected: str, n_conf: int) -> dict:
    """Population descriptors of one molecule (EVAL §1.1 item 5)."""
    m = Chem.MolFromSmiles(corrected)
    mh = Chem.AddHs(m)
    rings = m.GetRingInfo().AtomRings()
    largest = max((len(r) for r in rings), default=0)
    ring_class = 'acyclic' if not rings else ('3-ring only' if largest == 3 else
                                             ('ring >= 4, monocyclic' if len(rings) == 1 else 'ring >= 4, polycyclic'))
    return dict(K=n_conf, n_heavy=m.GetNumAtoms(), n_rot_td=len(td_rotatable_bonds(mh)),
                n_rot_heavy=len(heavy_rotatable_bonds(mh)), ring_class=ring_class, scaffold=murcko(smiles))


def population_lines(val: pd.DataFrame, test: pd.DataFrame, train_scaffolds: set[str] | None) -> list[str]:
    """Validation vs test: K, n_rot, n_heavy, ring classes and the scaffold-novel share."""
    out = []
    for name, D in (('val', val), ('test', test)):
        k = D.K
        out.append(f'{name}: n={len(D)}  K mean {k.mean():.2f} median {k.median():.0f} max {k.max()}  '
                   f'n_heavy mean {D.n_heavy.mean():.2f}  n_rot_td mean {D.n_rot_td.mean():.2f}  '
                   f'n_rot_heavy=0 (rigid) {int((D.n_rot_heavy == 0).sum())}')
        out.append(f'  n_rot_heavy bins {D.n_rot_heavy.clip(upper=4).value_counts().sort_index().to_dict()} '
                   f'(4 = 4+)')
        out.append(f'  ring classes {D.ring_class.value_counts().to_dict()}')
        if train_scaffolds is not None:
            novel = D.scaffold.map(lambda s: s not in ('', None) and s not in train_scaffolds)
            out.append(f'  scaffold-novel (Murcko scaffold absent from split index 0): {int(novel.sum())} '
                       f'({100 * novel.mean():.1f} %); acyclic {int((D.scaffold == "").sum())}')
    return out


def build(raw_dir: Path, split_path: Path, out_dir: Path, rule: str, n: int = 1000, seed: int = 0,
          split_index: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Walk the seed-``seed`` permutation of split ``split_index`` and keep the first ``n`` molecules that pass the
    filters; write the test-format files. Returns (kept csv rows with descriptors, walked indices, GT dict)."""
    stems = raw_stems(raw_dir)
    split = np.load(split_path, allow_pickle=True)[split_index]
    universe = sorted(int(i) for i in split)
    order = np.random.default_rng(seed).permutation(universe)
    rows, walked, mols = [], [], {}
    for pos, idx in enumerate(order):
        if len(rows) == n:
            break
        if idx >= len(stems):
            walked.append(dict(order_pos=pos, split_value=int(idx), smiles='', status='index_out_of_range'))
            continue
        stem = stems[idx]
        d = load_raw(raw_dir, stem)
        corrected = corrected_candidates(d)[rule]
        confs = [c['rd_mol'] for c in d['conformers']]
        if Chem.MolFromSmiles(corrected) is None:
            status = 'excluded_unparsable'
        elif not clean_confs(corrected, confs):
            status = 'excluded_no_clean_gt'
        else:
            status = 'kept'
            rows.append(dict(smiles=stem, n_conformers=len(confs), corrected_smiles=corrected))
            mols[stem] = confs
        walked.append(dict(order_pos=pos, split_value=int(idx), smiles=stem, status=status))
    if len(rows) < n:
        raise RuntimeError(f'only {len(rows)} molecules pass the filters')
    out_dir.mkdir(parents=True, exist_ok=True)
    val = pd.DataFrame(rows)
    val.to_csv(out_dir / 'val_smiles.csv', index=False)
    W = pd.DataFrame(walked)
    W.to_csv(out_dir / 'val_indices.csv', index=False)
    with open(out_dir / 'val_mols.pkl', 'wb') as f:
        pickle.dump(mols, f)
    sha = hashlib.sha256((out_dir / 'val_mols.pkl').read_bytes()).hexdigest()
    (out_dir / 'val_mols.pkl.sha256').write_text(f'{sha}  val_mols.pkl\n')
    return val, W, mols


def main() -> None:
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--raw_dir', required=True, help='raw GEOM QM9 pickles (data/QM9/qm9/)')
    ap.add_argument('--split', required=True, help='data/QM9/split.npy')
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--test_mols', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_scaffolds', action='store_true', help='also the scaffold-novel share (reads split 0)')
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rule, S = self_check(Path(a.test_csv), Path(a.test_mols), Path(a.raw_dir))
    S.to_csv(out / 'self_check_rows.csv', index=False)
    frac = float(S.reproduced.mean())
    lines = ['# EVAL §1.1 builder self-check on the 1000 test rows (Q-P2-010 acceptance: >= 99 % reproduced)',
             f'corrected_smiles rule scores (smiles, n_conformers, test_mols counts and corrected_smiles all equal): '
             f'{S.attrs["scores"]}',
             f'n_conformers rule scores (n_csv equal to the count): {S.attrs["count_scores"]}',
             f'chosen rule: {rule}; n_conformers rule: stored conformer count (n_csv == n_stored for '
             f'{int((S.n_csv == S.n_stored).sum())} rows; test_mols count == stored for '
             f'{int((S.n_test_mols == S.n_stored).sum())}; GT positions identical to raw for '
             f'{int(S.gt_positions_equal.sum())})',
             f'reproduced rows: {int(S.reproduced.sum())} / {len(S)} = {100 * frac:.1f} %  -> '
             f'{"PASS" if frac >= ACCEPT else "FAIL"}',
             f'test rows that the filters would exclude: unparsable {int((~S.parses.astype(bool)).sum())}, '
             f'no clean GT {int((S.n_clean == 0).sum())} (kept in the test csv)',
             'residual mismatches:']
    lines += [f'  {r.smiles}' for r in S[~S.reproduced].itertuples()]
    (out / 'self_check.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if frac < ACCEPT:
        raise SystemExit('self-check failed: the validation set is not built')
    val, W, _ = build(Path(a.raw_dir), Path(a.split), out, rule, a.n, a.seed)
    train = None
    if a.train_scaffolds:
        stems = raw_stems(Path(a.raw_dir))
        tr = np.load(a.split, allow_pickle=True)[0]
        train = {murcko(stems[i]) for i in tr if i < len(stems)} - {None}
    test = pd.read_csv(a.test_csv)
    dv = pd.DataFrame([describe(r.smiles, r.corrected_smiles, r.n_conformers) for r in val.itertuples()])
    dt = pd.DataFrame([describe(r.smiles, r.corrected_smiles, r.n_conformers) for r in test.itertuples()
                       if Chem.MolFromSmiles(r.corrected_smiles) is not None])
    cnt = Counter(W.status)
    pop = [f'# QM9 validation subset (EVAL §1.1): first {a.n} of the seed-{a.seed} permutation of split index 1 '
           f'passing the filters; corrected_smiles rule {rule}',
           f'walked {len(W)} positions: {dict(cnt)}', 'excluded:']
    pop += [f'  {r.split_value} {r.smiles} {r.status}' for r in W[W.status != 'kept'].itertuples()]
    pop += population_lines(dv, dt, train)
    pop.append('val_mols.pkl sha256 ' + (out / 'val_mols.pkl.sha256').read_text().split()[0])
    (out / 'population.txt').write_text('\n'.join(pop) + '\n')
    write_provenance(out)
    print('\n'.join(pop))


if __name__ == '__main__':
    main()

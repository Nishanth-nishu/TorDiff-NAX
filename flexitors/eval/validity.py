"""Conformer-level validity screen (EVAL §3.1; EXP-1 §3 row 5, §2.6-§2.8; EXP-3 §9.3).

EVAL §3.1: a generated conformer is invalid if it fails sanitisation, its bond perception from 3D differs from the
input graph, it has a clash, or its stereo differs from the input. Invalid conformers are counted, never dropped
(EVAL §3.1 (c)); this module only measures.

Checks and their sources:
* connectivity from 3D (``rdDetermineBonds``), EXP-1 §3 row 5. A graph bond that is not perceived is a break. A
  perceived bond that is not in the graph counts only if the reference conformer's own perception lacks it: correct
  strained cages (bicyclo[1.1.1]pentane, 1,3-bridgehead distance ≈ 1.85 Å) are perceived with an extra bond by
  RDKit's default covalent factor (review P2 B1); the GT calibration (``main``) measures the residual rate;
* clash: a non-bonded heavy-atom pair closer than 0.75 × the sum of covalent radii, EXP-1 §2.6 step 1;
* chirality: sign of the signed volume χ_c (EXP-1 §2.8) at every degree-4 atom and every degree-3 ring-fusion atom
  (bridgehead, EXP-1 §2.8 "signed volume at each bridgehead"), a structural set that does not depend on RDKit's
  stereo perception (review P2 S4, the E-CHEM-060 lesson; the degree-4 set of ``tools/lgeom.py:232``). Centres with
  |χ_ref| < CHI_MIN in the reference are near-planar and not compared (plan_P2 §2.2, INFERENCE);
* E/Z: geometric label on TD's reference quadruple, matched to the target iff cos(τ_b − φ_b) > 0 (EXP-3 §9.3 primary
  label, against ``round5/data/ez_targets.csv``); planarity |δ_b| ≤ 20° (EXP-3 §3.3, §9.3);
* reported beside, NOT part of EVAL §3.1 validity: the ±15 % bond window of EXP-1 §2.6 step 1 (a constraint-layer
  rule) and ring closure (implied by connectivity, EXP-1 §2.7).

The checks are array-level primitives on positions and precomputed index arrays (``clash_index`` /
``clash_mask``, ``bond_window_mask``, ``chirality_centres`` / ``chirality_flip_mask``) with thin ``Mol`` wrappers,
so the milestone-2 constraint layer (EXP-1 §2.6/§2.8) uses the same definitions.

CLI: the GT calibration — every cleaned GT conformer of a test-format set screened against the molecule's first
cleaned GT conformer (same atom order), so that correct geometry is ≈ 100 % valid before any model output is screened.
"""
from __future__ import annotations

import math
import pickle
from argparse import ArgumentParser
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

from flexitors.data.qm9_val import clean_confs
from flexitors.geometry.internal import bond_lengths, dihedral, signed_volume, wrap_angle
from flexitors.geometry.torsion_types import StereoBond, structural_stereo_bonds
from flexitors.provenance import provenance

RDLogger.DisableLog('rdApp.*')
BOND_TOL = 0.15          # EXP-1 §2.6 step 1: ±15 % of the reference length
CLASH_FACTOR = 0.75      # EXP-1 §2.6 step 1: 0.75 × sum of covalent radii
PLANARITY_DEG = 20.0     # EXP-3 §3.3 / §9.3: |δ_b| ≤ 20°
CHI_MIN = 0.05           # Å³; near-planar reference centres are not compared (plan_P2 §2.2, INFERENCE)


# ------------------------------------------------------------------------------- array-level primitives (batched)
def clash_mask(pos: np.ndarray, pairs: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """True where a pair of ``pairs`` (k, 2) is closer than its limit; positions (..., n, 3) -> (..., k)."""
    return bond_lengths(pos, pairs) < limits


def bond_window_mask(pos: np.ndarray, ref_pos: np.ndarray, bonds: np.ndarray, tol: float = BOND_TOL) -> np.ndarray:
    """True where a bond deviates by more than ``tol`` (relative) from its reference length (EXP-1 §2.6 step 1)."""
    return np.abs(bond_lengths(pos, bonds) / bond_lengths(ref_pos, bonds) - 1) > tol


def chirality_flip_mask(pos: np.ndarray, ref_pos: np.ndarray, centres: np.ndarray,
                        chi_min: float = CHI_MIN) -> np.ndarray:
    """True where the signed volume of a centre row (c, a, b, d) changed sign relative to the reference, among the
    centres with |χ_ref| ≥ ``chi_min`` (EXP-1 §2.8)."""
    chi, chi_ref = signed_volume(pos, centres), signed_volume(ref_pos, centres)
    return (np.abs(chi_ref) >= chi_min) & (np.sign(chi) != np.sign(chi_ref))


# ------------------------------------------------------------------------------- index arrays from the graph
def positions(mol: Chem.Mol, conf_id: int = -1) -> np.ndarray:
    """Atom positions (n, 3) of one conformer, float64."""
    return np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=np.float64)


def bond_pairs(mol: Chem.Mol) -> np.ndarray:
    """(m, 2) begin/end atom indices of every bond, in bond order."""
    return np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], dtype=np.int64).reshape(-1, 2)


def clash_index(mol: Chem.Mol, factor: float = CLASH_FACTOR) -> tuple[np.ndarray, np.ndarray]:
    """Non-bonded heavy-atom pairs (k, 2) and their clash limits ``factor`` × (r_cov,i + r_cov,j) (EXP-1 §2.6)."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    bonded = {frozenset(map(int, b)) for b in bond_pairs(mol)}
    pt = Chem.GetPeriodicTable()
    pairs = [(i, j) for x, i in enumerate(heavy) for j in heavy[x + 1:] if frozenset((i, j)) not in bonded]
    limits = [factor * (pt.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum())
                        + pt.GetRcovalent(mol.GetAtomWithIdx(j).GetAtomicNum())) for i, j in pairs]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2), np.array(limits, dtype=np.float64)


def chirality_centres(mol: Chem.Mol, distinct_only: bool = False) -> np.ndarray:
    """(k, 4) rows (centre, a, b, d): every degree-4 atom and every degree-3 atom in ≥ 2 rings (bridgeheads), with its
    first three neighbours in bond order. Structural, so independent of RDKit's stereo perception.

    ``distinct_only`` keeps only centres whose neighbours are pairwise topologically distinct (graph-symmetry classes,
    ``CanonicalRankAtoms(breakTies=False)``). Use it when the reference is an independently generated conformer (e.g.
    another GT conformer): there a sign change at a centre with two equivalent neighbours (a CH2's two H) is a
    relabelling, not a stereo change. It misses relative ring stereo between equivalent arms (cis/trans decalin), so
    the full set is the one to use against the conformer's own seed (EXP-1 §2.8)."""
    ri = mol.GetRingInfo()
    cls = list(Chem.CanonicalRankAtoms(mol, breakTies=False)) if distinct_only else None
    rows = []
    for atom in mol.GetAtoms():
        deg = atom.GetDegree()
        if deg == 4 or (deg == 3 and ri.NumAtomRings(atom.GetIdx()) >= 2):
            nbrs = sorted((b.GetIdx(), b.GetOtherAtomIdx(atom.GetIdx())) for b in atom.GetBonds())
            if distinct_only and len({cls[v] for _, v in nbrs}) < deg:
                continue
            rows.append([atom.GetIdx(), *(v for _, v in nbrs[:3])])
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


# ------------------------------------------------------------------------------- Mol-level checks
def is_sanitizable(mol: Chem.Mol) -> bool:
    """True if RDKit sanitisation of a copy succeeds (EVAL §3.1). Sanitisation ignores coordinates, so for a TD output,
    whose graph is the input graph, this check can only fail if the input itself does not sanitise."""
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
        return True
    except Exception:  # RDKit raises several exception types for sanitisation failures
        return False


def perceived_bonds(mol: Chem.Mol, conf_id: int = -1) -> set[frozenset[int]]:
    """Bonds perceived from the 3D coordinates alone (``rdDetermineBonds.DetermineConnectivity``, default factor)."""
    m = Chem.RWMol()
    for a in mol.GetAtoms():
        atom = Chem.Atom(a.GetAtomicNum())
        atom.SetFormalCharge(a.GetFormalCharge())
        m.AddAtom(atom)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(positions(mol, conf_id)):
        conf.SetAtomPosition(i, p.tolist())
    m.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineConnectivity(m)
    return {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in m.GetBonds()}


def ez_state(pos: np.ndarray, bonds: list[StereoBond]) -> tuple[np.ndarray, np.ndarray]:
    """For each stereo bond: (matches its target, |δ_b| in degrees to the nearest planar value).

    Bonds without a SMILES target (``phi is None``) count as matching; their planarity is measured against the nearest
    of {0, π} (EXP-3 §3.1, §9.3)."""
    if not bonds:
        return np.zeros(0, dtype=bool), np.zeros(0)
    tau = dihedral(pos, np.array([b.quad for b in bonds]))
    match = np.array([b.phi is None or math.cos(t - b.phi) > 0 for b, t in zip(bonds, tau)], dtype=bool)
    nearest = np.where(np.cos(tau) > 0, 0.0, np.pi)
    return match, np.degrees(np.abs(wrap_angle(tau - nearest)))


@dataclass(frozen=True)
class ValidityReport:
    """Per-conformer screen. Counts that need a reference conformer are None when none was given."""

    sanitizable: bool
    n_missing_bonds: int
    n_extra_bonds: int
    n_extra_bonds_unexplained: int | None
    n_clashes: int
    n_chirality_flips: int | None
    n_ez_mismatches: int
    n_nonplanar: int
    rings_closed: bool
    n_bond_window_all: int | None
    n_bond_window_ring: int | None

    @property
    def valid(self) -> bool | None:
        """EVAL §3.1 validity; None (undetermined) without a reference, because stereo and extra perceived bonds can
        only be judged against one. The bond window and ring closure are reported, not part of it."""
        if self.n_chirality_flips is None or self.n_extra_bonds_unexplained is None:
            return None
        return (self.sanitizable and self.n_missing_bonds == 0 and self.n_extra_bonds_unexplained == 0
                and self.n_clashes == 0 and self.n_chirality_flips == 0 and self.n_ez_mismatches == 0)


def validate_conformer(mol: Chem.Mol, conf_id: int = -1, ref: Chem.Mol | None = None, ref_conf_id: int = -1,
                       stereo_bonds: list[StereoBond] = (), ref_is_seed: bool = True) -> ValidityReport:
    """Run every EVAL §3.1 check on one conformer of ``mol``.

    Args:
        mol: the generated molecule (explicit H; the input graph is its bond list).
        conf_id: conformer to screen.
        ref: reference molecule with the same atom order (the seed, or a GT conformer) for extra perceived bonds,
            chirality and the bond window; without it those are None and ``valid`` is undetermined.
        ref_conf_id: conformer of ``ref``.
        stereo_bonds: the molecule's rows of the E/Z target table.
        ref_is_seed: True if ``ref`` is the seed this conformer was generated from (atom labels carried along); False
            for an independent reference, which restricts chirality to ``chirality_centres(distinct_only=True)``.
    """
    graph = {frozenset(map(int, b)) for b in bond_pairs(mol)}
    seen = perceived_bonds(mol, conf_id)
    ring = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds() if b.IsInRing()}
    pos = positions(mol, conf_id)
    pairs, limits = clash_index(mol)
    extra = seen - graph
    n_unexpl = n_chi = n_win = n_win_ring = None
    if ref is not None:
        ref_pos = positions(ref, ref_conf_id)
        n_unexpl = len(extra - perceived_bonds(ref, ref_conf_id))
        n_chi = int(chirality_flip_mask(pos, ref_pos, chirality_centres(mol, not ref_is_seed)).sum())
        win = bond_window_mask(pos, ref_pos, bond_pairs(mol))
        is_ring = np.array([b.IsInRing() for b in mol.GetBonds()], dtype=bool)
        n_win, n_win_ring = int(win.sum()), int((win & is_ring).sum())
    match, dev = ez_state(pos, list(stereo_bonds))
    return ValidityReport(sanitizable=is_sanitizable(mol), n_missing_bonds=len(graph - seen),
                          n_extra_bonds=len(extra), n_extra_bonds_unexplained=n_unexpl,
                          n_clashes=int(clash_mask(pos, pairs, limits).sum()), n_chirality_flips=n_chi,
                          n_ez_mismatches=int((~match).sum()), n_nonplanar=int((dev > PLANARITY_DEG).sum()),
                          rings_closed=ring <= seen, n_bond_window_all=n_win, n_bond_window_ring=n_win_ring)


def calibrate(csv: pd.DataFrame, true_mols: dict, dataset: str) -> pd.DataFrame:
    """Screen every cleaned GT conformer against the molecule's first cleaned GT conformer (same atom order); E/Z is
    planarity only here (GT indices are not the target table's)."""
    rows = []
    for r in csv.itertuples(index=False):
        gts = clean_confs(r.corrected_smiles, true_mols.get(r.smiles, []))
        if not gts:
            continue
        bonds = [StereoBond(b.bond_idx, b.quad, None) for b in structural_stereo_bonds(gts[0])]
        for k, g in enumerate(gts):
            rep = validate_conformer(g, ref=gts[0], stereo_bonds=bonds, ref_is_seed=False)
            rows.append(dict(dataset=dataset, smiles=r.smiles, gt_idx=k, valid=rep.valid, **asdict(rep)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = ArgumentParser(description='GT calibration of the validity screen (EVAL §3.1; review P2 B1)')
    ap.add_argument('--set', nargs=3, action='append', required=True, metavar=('NAME', 'CSV', 'TRUE_MOLS'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    frames = []
    for name, csv, mols in a.set:
        with open(mols, 'rb') as f:
            frames.append(calibrate(pd.read_csv(csv), pickle.load(f), name))
    D = pd.concat(frames, ignore_index=True)
    D.to_csv(a.out, index=False)
    lines = [f'# GT calibration of flexitors/eval/validity.py (RDKit {provenance()["rdkit"]})']
    for name, g in D.groupby('dataset'):
        mol_invalid = g.groupby('smiles').valid.apply(lambda v: not v.all())
        lines.append(f'{name}: {len(g)} GT conformers of {g.smiles.nunique()} molecules; invalid conformers '
                     f'{int((~g.valid.astype(bool)).sum())} ({100 * (1 - g.valid.mean()):.2f} %), molecules with any '
                     f'{int(mol_invalid.sum())}')
        for col in ('sanitizable', 'rings_closed'):
            lines.append(f'  {col} False: {int((~g[col].astype(bool)).sum())}')
        for col in ('n_missing_bonds', 'n_extra_bonds', 'n_extra_bonds_unexplained', 'n_clashes', 'n_chirality_flips',
                    'n_nonplanar', 'n_bond_window_all'):
            lines.append(f'  {col} > 0: {int((g[col] > 0).sum())} conformers')
    txt = '\n'.join(lines)
    with open(a.out.rsplit('.', 1)[0] + '.txt', 'w') as f:
        f.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()

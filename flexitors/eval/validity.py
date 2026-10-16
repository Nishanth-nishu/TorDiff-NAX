"""Conformer-level validity screen (EVAL §3.1; EXP-1 §3 row 5, §2.6-§2.8; EXP-3 §9.3).

A generated conformer is invalid if it fails sanitisation, its bond perception from 3D differs from the input graph,
it has a clash, or its stereo differs from the input (EVAL §3.1). Invalid conformers are counted, never dropped
(EVAL §3.1 (c)); this module only measures.

Checks and their sources:
* sanitisation and connectivity from 3D (``rdDetermineBonds``) equal to the input graph: EXP-1 §3 row 5;
* ring closure: every ring bond of the input graph is still perceived from 3D (EXP-1 §2.7: in V2 only a bond stretch
  can open a ring), and ring bonds are counted separately in the bond window;
* bond window: each bond within ±15 % of its length in a reference conformer with the same atom order (the seed, or
  a GT conformer), EXP-1 §2.6 step 1;
* clash: a non-bonded heavy-atom pair closer than 0.75 × the sum of covalent radii, EXP-1 §2.6 step 1;
* chirality: sign of the signed volume χ_c of every RDKit-perceived tetrahedral stereocentre, neighbours in canonical
  rank order, compared with the reference conformer (EXP-1 §2.8);
* E/Z: geometric label on TD's reference quadruple, matched to the target iff cos(τ_b − φ_b) > 0 (EXP-3 §9.3
  primary label, against the committed target table of ``flexitors.data.ez_targets``); planarity |δ_b| ≤ 20° on the
  nearest of {0, π} (EXP-3 §3.3, §9.3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from flexitors.geometry.internal import bond_lengths, dihedral, signed_volume, wrap_angle
from flexitors.geometry.torsion_types import StereoBond

BOND_TOL = 0.15          # EXP-1 §2.6 step 1: ±15 % of the reference length
CLASH_FACTOR = 0.75      # EXP-1 §2.6 step 1: 0.75 × sum of covalent radii
PLANARITY_DEG = 20.0     # EXP-3 §3.3 / §9.3: |δ_b| ≤ 20°


@dataclass(frozen=True)
class ValidityReport:
    """Per-conformer screen. Counts that need a reference conformer are None when none was given."""

    sanitizable: bool
    connectivity_ok: bool
    rings_closed: bool
    n_clashes: int
    n_bond_violations: int | None
    n_ring_bond_violations: int | None
    n_chirality_flips: int | None
    n_ez_mismatches: int
    n_nonplanar: int

    @property
    def valid(self) -> bool:
        """EVAL §3.1 validity: sanitisable, connectivity preserved, no clash, stereo as the input."""
        return (self.sanitizable and self.connectivity_ok and self.rings_closed and self.n_clashes == 0
                and not self.n_bond_violations and not self.n_chirality_flips and self.n_ez_mismatches == 0)


def positions(mol: Chem.Mol, conf_id: int = -1) -> np.ndarray:
    """Atom positions (n, 3) of one conformer, float64."""
    return np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=np.float64)


def bond_pairs(mol: Chem.Mol) -> np.ndarray:
    """(m, 2) begin/end atom indices of every bond, in bond order."""
    return np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], dtype=np.int64).reshape(-1, 2)


def is_sanitizable(mol: Chem.Mol) -> bool:
    """True if RDKit sanitisation of a copy succeeds."""
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
        return True
    except Exception:  # RDKit raises several exception types for sanitisation failures
        return False


def perceived_bonds(mol: Chem.Mol, conf_id: int = -1) -> set[frozenset[int]]:
    """Bonds perceived from the 3D coordinates alone (``rdDetermineBonds.DetermineConnectivity``)."""
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


def clash_count(mol: Chem.Mol, conf_id: int = -1, factor: float = CLASH_FACTOR) -> int:
    """Non-bonded heavy-atom pairs closer than ``factor`` × the sum of covalent radii (EXP-1 §2.6)."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) < 2:
        return 0
    pt = Chem.GetPeriodicTable()
    rad = np.array([pt.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in heavy])
    x = positions(mol, conf_id)[heavy]
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    limit = factor * (rad[:, None] + rad[None, :])
    bonded = np.zeros_like(dist, dtype=bool)
    pos_of = {a: k for k, a in enumerate(heavy)}
    for i, j in bond_pairs(mol):
        if i in pos_of and j in pos_of:
            bonded[pos_of[i], pos_of[j]] = bonded[pos_of[j], pos_of[i]] = True
    iu = np.triu_indices(len(heavy), k=1)
    return int(np.sum((dist[iu] < limit[iu]) & ~bonded[iu]))


def bond_window_violations(mol: Chem.Mol, ref: Chem.Mol, conf_id: int = -1, ref_conf_id: int = -1,
                           tol: float = BOND_TOL) -> tuple[int, int]:
    """(all bonds, ring bonds) whose length deviates by more than ``tol`` (relative) from the reference conformer
    (EXP-1 §2.6 step 1). ``ref`` must share the atom order of ``mol``."""
    bonds = bond_pairs(mol)
    rel = np.abs(bond_lengths(positions(mol, conf_id), bonds) / bond_lengths(positions(ref, ref_conf_id), bonds) - 1)
    bad = rel > tol
    ring = np.array([b.IsInRing() for b in mol.GetBonds()], dtype=bool)
    return int(bad.sum()), int((bad & ring).sum())


def stereocentres(mol: Chem.Mol) -> np.ndarray:
    """(k, 4) rows (centre, a, b, d) for every RDKit-perceived potential tetrahedral stereocentre, with a, b, d its
    three lowest-ranked neighbours by canonical rank (EXP-1 §2.8)."""
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    rows = []
    for s in Chem.FindPotentialStereo(Chem.Mol(mol)):
        if s.type != Chem.StereoType.Atom_Tetrahedral:
            continue
        nbrs = sorted((n.GetIdx() for n in mol.GetAtomWithIdx(s.centeredOn).GetNeighbors()), key=lambda i: rank[i])
        if len(nbrs) >= 3:
            rows.append([s.centeredOn, *nbrs[:3]])
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def chirality_flips(mol: Chem.Mol, ref: Chem.Mol, conf_id: int = -1, ref_conf_id: int = -1) -> int:
    """Stereocentres whose signed-volume sign differs from the reference conformer (same atom order)."""
    centres = stereocentres(mol)
    if not len(centres):
        return 0
    s = np.sign(signed_volume(positions(mol, conf_id), centres))
    s_ref = np.sign(signed_volume(positions(ref, ref_conf_id), centres))
    return int(np.sum(s != s_ref))


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


def validate_conformer(mol: Chem.Mol, conf_id: int = -1, ref: Chem.Mol | None = None, ref_conf_id: int = -1,
                       stereo_bonds: list[StereoBond] = ()) -> ValidityReport:
    """Run every EVAL §3.1 check on one conformer of ``mol``.

    Args:
        mol: the generated molecule (explicit H; the input graph is its bond list).
        conf_id: conformer to screen.
        ref: reference molecule with the same atom order (seed or GT conformer) for the bond window and chirality;
            without it those counts are None.
        ref_conf_id: conformer of ``ref``.
        stereo_bonds: the molecule's rows of the E/Z target table.
    """
    graph = {frozenset((int(i), int(j))) for i, j in bond_pairs(mol)}
    seen = perceived_bonds(mol, conf_id)
    ring = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds() if b.IsInRing()}
    n_bond = n_ring = n_chi = None
    if ref is not None:
        n_bond, n_ring = bond_window_violations(mol, ref, conf_id, ref_conf_id)
        n_chi = chirality_flips(mol, ref, conf_id, ref_conf_id)
    match, dev = ez_state(positions(mol, conf_id), list(stereo_bonds))
    return ValidityReport(sanitizable=is_sanitizable(mol), connectivity_ok=seen == graph, rings_closed=ring <= seen,
                          n_clashes=clash_count(mol, conf_id), n_bond_violations=n_bond,
                          n_ring_bond_violations=n_ring, n_chirality_flips=n_chi,
                          n_ez_mismatches=int((~match).sum()), n_nonplanar=int((dev > PLANARITY_DEG).sum()))

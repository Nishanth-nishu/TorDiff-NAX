"""TD's rotatable-bond rule, TD's reference quadruple per rotatable bond, and the structural stereo double-bond test.

This is the single classifier shared by Exp 1 and Exp 3 (EXP-3 §9.5 "One classifier"). All functions take an RDKit
molecule with explicit hydrogens built as TD builds it (``Chem.AddHs(Chem.MolFromSmiles(smiles))``,
``torsional-diffusion/utils/featurization.py:116-119``), whose atom and bond order is TD's ``edge_index`` order
(``featurization.py:83-87``: both directions of bond k at columns 2k and 2k+1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rdkit import Chem

CIS_CODES = (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS)
TRANS_CODES = (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS)


@dataclass(frozen=True)
class StereoBond:
    """One structural stereo double bond (EXP-3 §3.1) with its TD reference quadruple (c, a, b, d) and, when the
    SMILES specifies it, the planar target φ_b (0 = c and d cis, π = trans); ``phi`` is None if unspecified."""

    bond_idx: int
    quad: tuple[int, int, int, int]
    phi: float | None


def _component(mh: Chem.Mol, start: int, cut: int) -> set[int]:
    """Atoms reachable from ``start`` without crossing bond ``cut``."""
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for b in mh.GetAtomWithIdx(u).GetBonds():
            if b.GetIdx() == cut:
                continue
            v = b.GetOtherAtomIdx(u)
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def td_rotated_side(mh: Chem.Mol, bond_idx: int) -> set[int] | None:
    """The atom set TD rotates about bond ``bond_idx``, or None if TD does not treat the bond as rotatable.

    Port of ``utils/torsion.py:9-39`` (``get_transformation_mask``): the bond is rotatable iff removing it disconnects
    the H-explicit graph and the smaller side has more than one atom; the rotated side is
    ``sorted(nx.connected_components(G2), key=len)[0]``. networkx yields the component of the lowest-index atom first
    and the sort is stable, so on a tie the side containing the lowest atom index is the one rotated. There is no
    bond-order test (double and triple bonds are rotatable, E-CHEM-020).
    """
    bond = mh.GetBondWithIdx(bond_idx)
    side = _component(mh, bond.GetBeginAtomIdx(), bond_idx)
    if bond.GetEndAtomIdx() in side:
        return None
    other = set(range(mh.GetNumAtoms())) - side
    small = min((side, other), key=lambda s: (len(s), min(s)))
    return small if len(small) > 1 else None


def td_rotatable_bonds(mh: Chem.Mol) -> list[int]:
    """Indices of the bonds TD treats as rotatable, in bond (= TD edge) order (``utils/torsion.py:9-39``)."""
    return [b.GetIdx() for b in mh.GetBonds() if td_rotated_side(mh, b.GetIdx()) is not None]


def heavy_rotatable_bonds(mh: Chem.Mol) -> list[int]:
    """TD-rotatable bonds with ≥ 2 heavy atoms on each side: the ``n_rot_heavy`` count of ``tools/breakdown.py:32-50``
    that defines the rigid stratum (n_rot_heavy = 0; round5/DECISION.md fix 4, E-JOINT-007)."""
    heavy = {a.GetIdx() for a in mh.GetAtoms() if a.GetAtomicNum() > 1}
    out = []
    for k in td_rotatable_bonds(mh):
        b = mh.GetBondWithIdx(k)
        side = _component(mh, b.GetBeginAtomIdx(), k)
        if len(side & heavy) >= 2 and len(heavy - side) >= 2:
            out.append(k)
    return out


def _first_neighbour(mh: Chem.Mol, atom: int, exclude: int) -> int:
    """First neighbour of ``atom`` other than ``exclude`` in bond-index order (TD's ``edge_list`` order)."""
    nbrs = sorted((b.GetIdx(), b.GetOtherAtomIdx(atom)) for b in mh.GetAtomWithIdx(atom).GetBonds())
    return next(v for _, v in nbrs if v != exclude)


def td_reference_quadruple(mh: Chem.Mol, bond_idx: int) -> tuple[int, int, int, int]:
    """TD's dihedral (c, a, b, d) for a rotatable bond: (a, b) is the masked directed edge, whose second atom lies on
    the rotated side (``utils/torsion.py:19-24``), and c, d are the first neighbours of a, b in ``edge_index`` order
    (``diffusion/sampling.py:135-142``)."""
    side = td_rotated_side(mh, bond_idx)
    if side is None:
        raise ValueError(f'bond {bond_idx} is not TD-rotatable')
    bond = mh.GetBondWithIdx(bond_idx)
    s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    a, b = (e, s) if s in side else (s, e)
    return _first_neighbour(mh, a, b), a, b, _first_neighbour(mh, b, a)


def _heavy_degree(atom: Chem.Atom) -> int:
    return sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)


def is_structural_stereo(bond: Chem.Bond) -> bool:
    """EXP-3 §3.1 structural stereo test: bond order 2, acyclic, both atoms with ≥ 2 heavy neighbours (the partner
    included). Version-independent; it decides what is constrained (EXP-3 §9.4) and classes imine N–H bonds as
    non-stereo (Q-P2-007 answer)."""
    return (bond.GetBondType() == Chem.BondType.DOUBLE and not bond.IsInRing()
            and _heavy_degree(bond.GetBeginAtom()) >= 2 and _heavy_degree(bond.GetEndAtom()) >= 2)


def smiles_phi_target(mh: Chem.Mol, bond_idx: int, quad: tuple[int, int, int, int]) -> float | None:
    """Planar target φ_b on TD's quadruple from the SMILES bond stereo, or None if the bond carries none.

    EXP-3 §3.1: with stereo atoms (s1, s2) and STEREOCIS/STEREOTRANS (or Z/E), φ_b = 0 if c and d are cis and π if
    trans; each of c ≠ s1, d ≠ s2 toggles cis ↔ trans. Z/E are read relative to the stereo atoms as the spec states;
    ``flexitors.data.ez_targets`` checks every target against an ETKDG embedding of the same SMILES.
    """
    bond = mh.GetBondWithIdx(bond_idx)
    code = bond.GetStereo()
    if code not in CIS_CODES + TRANS_CODES:
        return None
    s_atoms = list(bond.GetStereoAtoms())
    c, a, b, d = quad
    s_of = {}
    for s in s_atoms:
        for end in (a, b):
            if mh.GetBondBetweenAtoms(s, end) is not None and s not in (a, b):
                s_of[end] = s
    if set(s_of) != {a, b}:
        raise ValueError(f'bond {bond_idx}: stereo atoms {s_atoms} do not map onto both ends of {quad}')
    cis = code in CIS_CODES
    cis ^= c != s_of[a]
    cis ^= d != s_of[b]
    return 0.0 if cis else math.pi


def structural_stereo_bonds(mh: Chem.Mol) -> list[StereoBond]:
    """Every TD-rotatable structural stereo double bond of ``mh`` with its quadruple and SMILES target."""
    out = []
    for k in td_rotatable_bonds(mh):
        if is_structural_stereo(mh.GetBondWithIdx(k)):
            quad = td_reference_quadruple(mh, k)
            out.append(StereoBond(k, quad, smiles_phi_target(mh, k, quad)))
    return out

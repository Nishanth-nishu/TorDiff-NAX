"""Exact internal coordinates in float64 numpy, batched over conformers.

Dihedrals use the atan2 form everywhere drift or planarity is measured (round5/DECISION.md fix 7; EXP-1 §10.1 item 2):
TD's ``get_torsion_angles`` (``torsional-diffusion/utils/torsion.py:110-124``) clamps the cosine to ±(1 − 1e-5)
(``:120``), so it cannot resolve angles within ≈ 4.5e-3 rad of 0 or π — exactly where planarity is judged.
"""
from __future__ import annotations

import numpy as np


def dihedral(pos: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Dihedral angles (rad, in (−π, π]) of atom quadruples (i, j, k, l), same sign convention as RDKit's
    ``rdMolTransforms.GetDihedralRad``.

    Args:
        pos: positions, shape (..., n_atoms, 3).
        quads: integer array (m, 4).
    Returns:
        array (..., m).
    """
    p = np.asarray(pos, dtype=np.float64)
    q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    p0, p1, p2, p3 = (p[..., q[:, t], :] for t in range(4))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def signed_volume(pos: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Signed volume χ_c = (x_a − x_c)·[(x_b − x_c) × (x_d − x_c)] per row (c, a, b, d) of ``centres``
    (EXP-1 §2.8, neighbours a < b < d in canonical rank order chosen by the caller).

    Args:
        pos: positions, shape (..., n_atoms, 3).
        centres: integer array (k, 4) of (centre, a, b, d).
    Returns:
        array (..., k), in Å³.
    """
    p = np.asarray(pos, dtype=np.float64)
    c = np.asarray(centres, dtype=np.int64).reshape(-1, 4)
    xc = p[..., c[:, 0], :]
    va, vb, vd = (p[..., c[:, t], :] - xc for t in (1, 2, 3))
    return np.sum(va * np.cross(vb, vd), axis=-1)


def bond_lengths(pos: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    """Distances (Å) between the atom pairs of ``bonds`` (m, 2); positions (..., n_atoms, 3)."""
    p = np.asarray(pos, dtype=np.float64)
    b = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    return np.linalg.norm(p[..., b[:, 0], :] - p[..., b[:, 1], :], axis=-1)


def wrap_angle(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (−π, π]."""
    y = np.mod(np.asarray(x, dtype=np.float64) + np.pi, 2 * np.pi) - np.pi
    return np.where(y == -np.pi, np.pi, y)

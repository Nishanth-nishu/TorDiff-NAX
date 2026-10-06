"""
Shared local-structure (L) geometry layer for round 2 (round2/code_plan_2.md Â§1, as amended by round2/DECISION.md).

Pure numpy + scipy + RDKit (no torch / PyG), so it is unit-testable on the local machine (tools/tests/test_lgeom.py).

Used by
  tools/build_paired_pickles.py   S3/S4/B1cap training data: matched RDKit L paired with its own GT L
  tools/make_l_seed_pickles.py    S1 test-time seed pickles (MMFF twin, noise, lambda series, ring/acyclic oracles)

DECISION D1 (3/3 votes): heavy-atom Kabsch fit applied to all atoms; relabel graph-equivalent terminal atoms before
interpolating (an identity map interpolates a methyl rotated by 120 deg through C-H ~ 0.62 A at lambda = 0.5, local check
in code_plan_2 Â§0.3); per-pair `pair_ok` guard (bond length / clash / chirality at lambda = 0.5). Failing pairs are dropped
and counted by the callers.
DECISION D2: the acyclic set includes H, so ring-only and acyclic-only GT L are exact complements.
"""
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from scipy.optimize import linear_sum_assignment

REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)  # == torsional-diffusion/standardize_confs.py:33

# pre-declared pair_ok thresholds (code_plan_2 Â§1.1; fixed before any data were seen)
MAX_BOND_DEV = 0.05     # A, |d(x_0.5) - (d_X + d_Y)/2| over all bonds
MIN_NONBONDED = 0.9     # A, min distance between atoms >= 3 bonds apart at lambda = 0.5


# ----------------------------------------------------------------------------------------------------------- graph
def graph_signature(mol):
    """atomic numbers in index order + sorted bond set with bond order: equal signatures <=> same labelled graph"""
    z = tuple(a.GetAtomicNum() for a in mol.GetAtoms())
    b = tuple(sorted((min(x.GetBeginAtomIdx(), x.GetEndAtomIdx()), max(x.GetBeginAtomIdx(), x.GetEndAtomIdx()),
                      float(x.GetBondTypeAsDouble())) for x in mol.GetBonds()))
    return z, b


def same_graph(mols):
    """code_plan_2 V1/V2: every mol shares atom order and bonds with the first one"""
    s0 = graph_signature(mols[0])
    return all(graph_signature(m) == s0 for m in mols[1:])


def heavy_idx(mol):
    return np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1], dtype=int)


def positions(mol, cid=-1):
    return np.array(mol.GetConformer(cid).GetPositions(), dtype=float)


def with_positions(mol, X):
    """copy of mol (graph + props) with exactly one conformer holding X"""
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(np.asarray(X, dtype=float)):
        conf.SetAtomPosition(i, p.tolist())
    m.AddConformer(conf, assignId=True)
    return m


# ----------------------------------------------------------------------------------------------------------- F1
def kabsch_fit(P, Q, idx):
    """Rigidly move Q onto P, fitting on atom indices idx; proper rotation only (det = +1, as AlignMol).
    Returns the moved Q (all atoms)."""
    Pc, Qc = P[idx].mean(0), Q[idx].mean(0)
    H = (Q[idx] - Qc).T @ (P[idx] - Pc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (Q - Qc) @ R.T + Pc


def rmsd_on(P, Q, idx):
    return float(np.sqrt(((P[idx] - Q[idx]) ** 2).sum(1).mean()))


def permute_terminal(mol, X, Y):
    """Relabel graph-equivalent terminal atoms of Y (degree 1, same parent, element, bond order, formal charge) so
    that |X - Y| is minimal (Hungarian per group). A graph automorphism of the shared graph, so the model sees the same
    labelled graph either way. Returns (Y[perm], perm, n_swapped_atoms)."""
    perm = np.arange(len(Y))
    for p in mol.GetAtoms():
        groups = defaultdict(list)
        for b in p.GetBonds():
            a = b.GetOtherAtom(p)
            if a.GetDegree() == 1:
                groups[(a.GetAtomicNum(), float(b.GetBondTypeAsDouble()), a.GetFormalCharge(), a.GetIsotope())].append(
                    a.GetIdx())
        for idx in groups.values():
            if len(idx) < 2:
                continue
            idx = np.array(idx)
            C = np.linalg.norm(X[idx][:, None] - Y[perm[idx]][None], axis=-1)
            r, c = linear_sum_assignment(C)
            new = perm[idx][c]
            perm[idx[r]] = new
    return Y[perm], perm, int((perm != np.arange(len(Y))).sum())


def heavy_automorphism_perms(mol, max_maps=200):
    """Full-atom index permutations induced by the heavy-atom graph automorphisms of mol (H atoms follow their parent;
    their order inside a parent is fixed later by permute_terminal). Identity first."""
    hv = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    n = mol.GetNumAtoms()
    h = Chem.RemoveHs(mol, sanitize=False)
    if h.GetNumAtoms() != len(hv):
        return [np.arange(n)]
    try:
        ms = h.GetSubstructMatches(h, uniquify=False, useChirality=False, maxMatches=max_maps)
    except Exception:
        return [np.arange(n)]
    hs_of = {i: [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetAtomicNum() == 1] for i in hv}
    perms = [np.arange(n)]
    for mt in ms:
        perm = np.arange(n)
        ok = True
        for q, p in enumerate(mt):          # heavy atom hv[q] takes the position of heavy atom hv[p]
            i, j = hv[q], hv[p]
            perm[i] = j
            if len(hs_of[i]) != len(hs_of[j]):
                ok = False
                break
            for a, b in zip(hs_of[i], hs_of[j]):
                perm[a] = b
        if ok and len(set(perm.tolist())) == n and not np.array_equal(perm, perms[0]):
            perms.append(perm)
    return perms


def align_pair(mol, X, Y):
    """DECISION D1: heavy-atom Kabsch of Y onto X (applied to all atoms), terminal relabelling, twice (the second
    pass is idempotent in tests). Returns dict(Y_al, perm, heavy_rmsd, n_swaps).
    Extension (IMPLEMENTATION.md §2, measured on rematch 000.pickle): before the terminal relabelling, Y is relabelled by
    the heavy-atom graph automorphism with the lowest heavy-atom Kabsch RMSD (the correspondence GetBestRMS uses), so a
    prochiral swap of non-terminal equivalent groups (e.g. the two methyls of an isopropyl) is not interpolated through
    a planar centre. Like the terminal relabelling, this is a graph automorphism: the model sees the same graph."""
    h = heavy_idx(mol)
    best = None
    for pm in heavy_automorphism_perms(mol):
        Yp = Y[pm]
        r = rmsd_on(X, kabsch_fit(X, Yp, h), h)
        if best is None or r < best[0] - 1e-9:
            best = (r, pm)
    perm_tot = best[1].copy()
    Yal = kabsch_fit(X, Y[perm_tot], h)
    for _ in range(2):
        Yal, perm, _ = permute_terminal(mol, X, Yal)
        perm_tot = perm_tot[perm]
        Yal = kabsch_fit(X, Yal, h)
    return dict(Y_al=Yal, perm=perm_tot, heavy_rmsd=rmsd_on(X, Yal, h),
                n_swaps=int((perm_tot != np.arange(len(Y))).sum()))


def terminal_atoms(mol):
    """degree-1 atoms and their (non-terminal) parent, for interp_x"""
    t, p = [], []
    for a in mol.GetAtoms():
        if a.GetDegree() == 1:
            q = a.GetNeighbors()[0]
            if q.GetDegree() > 1:
                t.append(a.GetIdx())
                p.append(q.GetIdx())
    return np.array(t, dtype=int), np.array(p, dtype=int)


def interp_x(X, Y, lam, term, parent):
    """x_lam used by S1 seeds AND by the S4 training transform (torsional-diffusion/utils/dataset.py, same formula in
    torch; equality tested). Linear interpolation, then every terminal atom is put back on the linearly interpolated
    bond length along its interpolated bond direction. Raised deviation from code_plan_2 §1 (round2/IMPLEMENTATION.md):
    single-H rotors (O-H, N-H, S-H) are not rotatable bonds in TD (get_torsion_angles needs >= 2 atoms per side), so DE
    never matches them and plain linear interpolation shrinks their bond (local check: 0.10-0.15 A at lam = 0.5); with
    relabelling alone this would drop most alcohols/amines through the pair_ok bond check, i.e. a chemistry-dependent
    population. Endpoints are unchanged (exact at lam = 0 and 1)."""
    Z = (1.0 - lam) * X + lam * Y
    if len(term):
        v = Z[term] - Z[parent]
        n = np.linalg.norm(v, axis=1)
        d = (1.0 - lam) * np.linalg.norm(X[term] - X[parent], axis=1) + lam * np.linalg.norm(Y[term] - Y[parent], axis=1)
        ok = n > 1e-6
        Z[term[ok]] = Z[parent[ok]] + v[ok] / n[ok, None] * d[ok, None]
    return Z


# ----------------------------------------------------------------------------------------------------------- stereo
def stereo_labels(mol, X=None):
    """chiral centres + double-bond stereo perceived from 3D (as tools/local_structure_analysis.py:217-222)"""
    m = with_positions(mol, X) if X is not None else Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(m, confId=-1)
    centers = tuple(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
    dbl = tuple((b.GetIdx(), str(b.GetStereo())) for b in m.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE)
    return centers, dbl


def tetra_signs(mol, X):
    """sign of the signed volume at every atom with >= 3 heavy-or-H neighbours (robust chirality fingerprint that
    does not depend on CIP perception). Used to detect inversions along the interpolation path."""
    out = []
    for a in mol.GetAtoms():
        nb = [n.GetIdx() for n in a.GetNeighbors()]
        if len(nb) < 3:
            continue
        c = X[a.GetIdx()]
        v = np.linalg.det(np.stack([X[nb[0]] - c, X[nb[1]] - c, X[nb[2]] - c]))
        out.append(np.sign(v))
    return np.array(out)


# ----------------------------------------------------------------------------------------------------------- D1 guard
def pair_check(mol, X, Y_al, lam=0.5):
    """DECISION D1 per-pair guard at lambda = 0.5:
         stereo  : CIP stereo labels equal at both endpoints AND no tetrahedral sign flip between X, x_0.5 and Y_al
                   at sp3 centres (4 neighbours)
         bond    : max |d(x_0.5) - ((1-lam) d_X + lam d_Y)| <= MAX_BOND_DEV
         clash   : min distance between atoms >= 3 bonds apart at x_0.5 >= MIN_NONBONDED
    x_0.5 is built with interp_x (terminal bond lengths restored). Returns dict(pair_ok, stereo_ok, worst_bond_dev,
    min_nonbonded, reason) with reason in {ok, stereo (CIP labels differ), inversion (sp3 sign flip, e.g. swapped
    isopropyl methyls), bond_dev, clash}."""
    Z = interp_x(X, Y_al, lam, *terminal_atoms(mol))
    bi = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], dtype=int).reshape(-1, 2)
    d = lambda P: np.linalg.norm(P[bi[:, 0]] - P[bi[:, 1]], axis=1)
    worst = float(np.abs(d(Z) - ((1.0 - lam) * d(X) + lam * d(Y_al))).max()) if len(bi) else 0.0
    D = Chem.GetDistanceMatrix(mol)
    mask = D >= 3
    if mask.any():
        dz = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
        min_nb = float(dz[mask].min())
    else:
        min_nb = np.inf
    sp3 = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 4]

    def sp3_signs(P):
        s = []
        for i in sp3:
            nb = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()]
            s.append(np.sign(np.linalg.det(np.stack([P[nb[0]] - P[i], P[nb[1]] - P[i], P[nb[2]] - P[i]]))))
        return np.array(s)

    sx, sz, sy = sp3_signs(X), sp3_signs(Z), sp3_signs(Y_al)
    cip_ok = stereo_labels(mol, X) == stereo_labels(mol, Y_al)
    inv_ok = bool(np.all(sx == sy) and np.all(sx == sz))  # also catches swapped non-terminal symmetric groups
    stereo_ok = cip_ok and inv_ok
    reason = 'ok'
    if not cip_ok:
        reason = 'stereo'
    elif not inv_ok:
        reason = 'inversion'
    elif worst > MAX_BOND_DEV:
        reason = 'bond_dev'
    elif min_nb < MIN_NONBONDED:
        reason = 'clash'
    return dict(pair_ok=reason == 'ok', stereo_ok=stereo_ok, worst_bond_dev=worst, min_nonbonded=min_nb, reason=reason)


def pair_check_grid(mol, X, Y_al, lams=(0.25, 0.5, 0.75)):
    """FIXES X1 / research_check_A I1: pair_check on a lambda grid instead of lambda = 0.5 only. Returns the first
    failing lambda's result (with 'lam' set), else the lambda = 0.5 result with the worst bond deviation and the
    smallest non-bonded distance over the grid."""
    res = [dict(pair_check(mol, X, Y_al, lam=l), lam=l) for l in lams]
    for r in res:
        if not r['pair_ok']:
            return r
    out = dict(res[len(res) // 2])
    out['worst_bond_dev'] = max(r['worst_bond_dev'] for r in res)
    out['min_nonbonded'] = min(r['min_nonbonded'] for r in res)
    return out


# ----------------------------------------------------------------------------------------------------------- F5
def internal_sets(mol):
    """ALL-atom bonds / angles / endocyclic dihedrals with ring flags (DECISION D2: H belongs to the acyclic set).
    An angle is a ring angle iff both its bonds are ring bonds (tools/local_structure_analysis.py:206-209)."""
    ring_bond = {}
    bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds.append((i, j, b.IsInRing()))
        ring_bond[(i, j)] = ring_bond[(j, i)] = b.IsInRing()
    angles = []
    for a in mol.GetAtoms():
        j = a.GetIdx()
        nb = [n.GetIdx() for n in a.GetNeighbors()]
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                i, k = nb[x], nb[y]
                angles.append((i, j, k, ring_bond[(j, i)] and ring_bond[(j, k)]))
    dihs = []
    for ring in mol.GetRingInfo().AtomRings():
        n = len(ring)
        for t in range(n):
            dihs.append(tuple(ring[(t + q) % n] for q in range(4)))
    return bonds, angles, dihs


def _bond_len(P, i, j):
    return float(np.linalg.norm(P[i] - P[j]))


def _angle_deg(P, i, j, k):
    u, v = P[i] - P[j], P[k] - P[j]
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _dih_deg(P, a, b, c, d):
    b0, b1, b2 = P[a] - P[b], P[c] - P[b], P[d] - P[c]
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def l_subset_errors(mol, X, R):
    """code_plan_2 Â§1.1 / F5: RMSD of bond lengths (A), angles (deg), endocyclic dihedrals (deg, wrapped) of X vs the
    reference R, for {all-atom, heavy} x {ring, acyclic, all}."""
    bonds, angles, dihs = internal_sets(mol)
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    out = {}

    def rms(v):
        return float(np.sqrt(np.mean(np.square(v)))) if len(v) else np.nan

    for tag, keep in (('all', lambda t: True), ('heavy', lambda t: all(x in heavy for x in t))):
        for sub, flag in (('ring', True), ('acyc', False), ('any', None)):
            bd = [_bond_len(X, i, j) - _bond_len(R, i, j) for i, j, r in bonds
                  if keep((i, j)) and (flag is None or r == flag)]
            an = [_angle_deg(X, i, j, k) - _angle_deg(R, i, j, k) for i, j, k, r in angles
                  if keep((i, j, k)) and (flag is None or r == flag)]
            out[f'bond_rmsd_{tag}_{sub}'] = rms(bd)
            out[f'angle_rmsd_{tag}_{sub}'] = rms(an)
    dd = [((_dih_deg(X, *t) - _dih_deg(R, *t) + 180.0) % 360.0) - 180.0 for t in dihs]
    out['ring_dihedral_rmsd'] = rms(dd)
    return out


# ----------------------------------------------------------------------------------------------------------- S1.5
def set_internal_subset(mol, X_base, X_src, n_passes=3):
    """code_plan_2 Â§1.1 (generalises tools/local_structure_analysis.py:196-214 to ALL atoms, DECISION D2): start from
    X_base and copy every ACYCLIC bond length and every non-ring angle (exocyclic angles included) from X_src.
    SetBondLength/SetAngleDeg always move the side across an acyclic bond, so ring atoms never move relative to each
    other: ring bonds, ring angles and endocyclic dihedrals of X_base are preserved exactly. Sequential setting at
    branching centres cannot meet every constraint, so 3 passes are made and the residual is measured (F5)."""
    m = with_positions(mol, X_base)
    conf = m.GetConformer()
    bonds, angles, _ = internal_sets(mol)
    rb = {}
    for i, j, r in bonds:
        rb[(i, j)] = rb[(j, i)] = r
    for _ in range(n_passes):
        for i, j, r in bonds:
            if r:
                continue
            try:
                rdMolTransforms.SetBondLength(conf, i, j, _bond_len(X_src, i, j))
            except Exception:
                pass
        for i, j, k, r in angles:
            if r:
                continue
            a, b, c = (i, j, k) if not rb[(j, k)] else (k, j, i)  # moved side (c) must be across an acyclic bond
            try:
                rdMolTransforms.SetAngleDeg(conf, a, b, c, _angle_deg(X_src, i, j, k))
            except Exception:
                pass
    return np.array(conf.GetPositions(), dtype=float)


def seed_sources(pair_ok, interpolates):
    """FIXES X1: source pair for every seed slot of one molecule. Non-interpolating sets (lambda 0/1, A5) use their own
    pair. Interpolating sets replace an unsafe pair by the molecule's safe pairs, cycling; None if no safe pair."""
    n = len(pair_ok)
    if not interpolates:
        return list(range(n))
    safe = [i for i, o in enumerate(pair_ok) if o]
    if not safe:
        return None
    out, r = [], 0
    for i in range(n):
        if pair_ok[i]:
            out.append(i)
        else:
            out.append(safe[r % len(safe)])
            r += 1
    return out

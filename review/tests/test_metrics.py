"""
Synthetic unit tests for the evaluation metrics of the TD ablation pipeline (metric-verifier, 2026-09-29).

Covers
  * evaluate_confs.py on branch flexitors-ablation-hooks (run end-to-end on synthetic pickles, exported with git show)
      - RMSD matrix vs an independent Kabsch + graph-automorphism reference (heavy atoms, proper rotations only)
      - COV-R / AMR-R / COV-P / AMR-P mean/median vs an independent re-implementation of TD Eq. 35
      - strict '<' threshold, model failures (COV 0, excluded from AMR), "additional failures" (NaN rows),
        clean_confs filtering of GT conformers, SUMMARY and SWEEP lines
  * RDKit behaviour the metrics rely on: signed dihedrals in (-180, 180], GetBestRMS = proper rotations only
    (enantiomer not matched), symmetrizeConjugatedTerminalGroups (RDKit-version dependent)
  * tools/geometry_metrics.py: atom mapping gen->GT, bond/angle MAE = 0 for identical L, torsion delta, periodic
    wrapping, near-linear (sp) torsions excluded
  * tools/local_structure_analysis.py: relevant_torsions counts, torsion floor ~0 for identical L,
    floor <= transplant, no NaN on nitrile/alkyne molecules
  * tools/breakdown.py: sweep reproduces the evaluator's SUMMARY

Run:  python review/tests/test_metrics.py        (from anywhere; ~1 min, single process, no Pool)
"""
import importlib.util
import os
import pickle
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True  # do not leave __pycache__ in tools/
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem import rdMolTransforms as T

RDLogger.DisableLog('rdApp.*')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TOOLS = os.path.join(ROOT, 'tools')
REPO = os.path.join(ROOT, 'torsional-diffusion')
BRANCH = 'flexitors-ablation-hooks'
PY = sys.executable
TMP = tempfile.mkdtemp(prefix='metric_tests_')
FAILED = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILED.append(msg)


# ------------------------------------------------------------------------------------------------ helpers
def confs_of(smi, n, seed, mmff=False):
    """n single-conformer Mol objects (with H) of smi."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n, randomSeed=seed))
    if mmff:
        AllChem.MMFFOptimizeMoleculeConfs(m)
    return [Chem.Mol(m, confId=c) for c in cids]


def renumber(mol, seed):
    """Same geometry, randomly permuted atom order (generated mols come from SMILES, GT from GEOM: orders differ)."""
    perm = np.random.RandomState(seed).permutation(mol.GetNumAtoms()).tolist()
    return Chem.RenumberAtoms(mol, perm)


def kabsch_rmsd(P, Q):
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T  # proper rotation
    return float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean()))


def ref_best_rms(prb, ref):
    """Independent symmetry-aware heavy-atom RMSD: min over graph isomorphisms (bond-order aware) of Kabsch RMSD."""
    p, r = Chem.RemoveHs(prb), Chem.RemoveHs(ref)
    matches = r.GetSubstructMatches(p, uniquify=False, useChirality=False, maxMatches=100000)
    P, Q = p.GetConformer().GetPositions(), r.GetConformer().GetPositions()
    return min(kabsch_rmsd(P, Q[list(m)]) for m in matches)


def ref_stats(M, thr):
    """TD Eq. 35 (App. G.3) literally: recall over GT rows, precision = swap GT and generated."""
    L, K = M.shape
    cov_r = sum(any(M[l, k] < thr for k in range(K)) for l in range(L)) / L
    amr_r = sum(min(M[l, k] for k in range(K)) for l in range(L)) / L
    cov_p = sum(any(M[l, k] < thr for l in range(L)) for k in range(K)) / K
    amr_p = sum(min(M[l, k] for l in range(L)) for k in range(K)) / K
    return cov_r, amr_r, cov_p, amr_p


def load_tool(name, argv):
    sys.argv = [name] + argv
    spec = importlib.util.spec_from_file_location(name, os.path.join(TOOLS, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ 1. RDKit facts
def test_rdkit_facts():
    m = confs_of('CCCC', 1, 7)[0]
    c = m.GetConformer()
    ok = True
    for v in (-179.0, -60.0, 60.0, 179.0):
        T.SetDihedralDeg(c, 0, 1, 2, 3, v)
        ok &= abs(T.GetDihedralDeg(c, 0, 1, 2, 3) - v) < 1e-6
    check(ok, 'RDKit GetDihedralDeg is SIGNED in (-180, 180] (needed for torsion errors / transplant)')

    # symmetry: permuted atom order and symmetric methyls
    gt = confs_of('CC(C)CO', 1, 3, mmff=True)[0]
    gen = renumber(gt, 1)
    r = rdMolAlign.GetBestRMS(Chem.RemoveHs(gen), Chem.RemoveHs(gt))
    check(r < 1e-6, f'GetBestRMS(renumbered copy) = {r:.2e} ~ 0 (atom-order independent)')

    # swap the two symmetric methyl carbons' coordinates -> AlignMol (identity map) > 0, GetBestRMS = 0
    sw = Chem.Mol(gt)
    conf = sw.GetConformer()
    p0, p2 = conf.GetAtomPosition(0), conf.GetAtomPosition(2)
    conf.SetAtomPosition(0, p2)
    conf.SetAtomPosition(2, p0)
    a = rdMolAlign.AlignMol(Chem.RemoveHs(sw), Chem.RemoveHs(gt))
    b = rdMolAlign.GetBestRMS(Chem.RemoveHs(sw), Chem.RemoveHs(gt))
    check(a > 0.3 and b < 1e-6, f'symmetric methyl swap: AlignMol={a:.3f} (not symmetry aware), GetBestRMS={b:.2e}')

    # enantiomer (mirror image) is NOT matched by GetBestRMS (proper rotations only)
    ch = confs_of('C[C@H](O)CC', 1, 5, mmff=True)[0]
    mir = Chem.Mol(ch)
    cm = mir.GetConformer()
    for i in range(mir.GetNumAtoms()):
        p = cm.GetAtomPosition(i)
        cm.SetAtomPosition(i, (-p.x, p.y, p.z))
    r = rdMolAlign.GetBestRMS(Chem.RemoveHs(mir), Chem.RemoveHs(ch))
    P = Chem.RemoveHs(mir).GetConformer().GetPositions()
    Q = Chem.RemoveHs(ch).GetConformer().GetPositions()
    refl = float(np.sqrt((((P - P.mean(0)) * [-1, 1, 1] - (Q - Q.mean(0))) ** 2).sum(1).mean()))
    check(r > 0.1 and refl < 1e-6, f'mirror image: GetBestRMS={r:.3f} > 0 (reflection not allowed; with reflection {refl:.1e})')

    # conjugated terminal group symmetrisation (COOH after RemoveHs): RDKit-version-dependent default
    ac = confs_of('CC(=O)O', 1, 2, mmff=True)[0]
    sw = Chem.Mol(ac)
    conf = sw.GetConformer()
    o1, o2 = [a.GetIdx() for a in ac.GetAtoms() if a.GetSymbol() == 'O']
    p1, p2 = conf.GetAtomPosition(o1), conf.GetAtomPosition(o2)
    conf.SetAtomPosition(o1, p2)
    conf.SetAtomPosition(o2, p1)
    r_def = rdMolAlign.GetBestRMS(Chem.RemoveHs(sw), Chem.RemoveHs(ac))
    try:
        r_off = rdMolAlign.GetBestRMS(Chem.RemoveHs(sw), Chem.RemoveHs(ac), symmetrizeConjugatedTerminalGroups=False)
    except TypeError:
        r_off = float('nan')
    print(f'INFO COOH O-swap: GetBestRMS default={r_def:.4f}  symmetrize=False={r_off:.4f} '
          '(default symmetrises C(=O)O -> RMSD depends on RDKit version / flag)')
    check(r_def < 1e-6 < r_off, 'RDKit here symmetrises conjugated terminal O,O by default (version-dependent RMSD)')


# ------------------------------------------------------------------------------------------------ 2. evaluator
def test_calc_performance_stats_strict():
    src = subprocess.run(['git', '-C', REPO, 'show', f'{BRANCH}:evaluate_confs.py'], capture_output=True, text=True).stdout
    fn = re.search(r'def calc_performance_stats\(rmsd_array\):.*?return [^\n]*\n', src, re.S).group(0)
    ns = {'np': np, 'threshold': np.arange(0, 2.5, .125)}
    exec(fn, ns)
    M = np.array([[0.5, 0.7, 0.9], [0.2, 0.4999999, 1.0]])
    cr, ar, cp, ap = ns['calc_performance_stats'](M)
    i = 4  # threshold 0.5
    check(ns['threshold'][i] == 0.5, '0.5 is exactly on the evaluator grid (np.arange(0, 2.5, .125)[4])')
    check(cr[i] == 0.5, f'COV-R strict <: row min 0.5 NOT covered at delta=0.5 (got {cr[i]})')
    check(abs(cp[i] - 2 / 3) < 1e-12, f'COV-P strict <: col mins (0.2, 0.4999999, 0.9) -> 2/3? got {cp[i]:.4f}')
    ref = ref_stats(M, 0.5)
    check(np.allclose([cr[i], ar, cp[i], ap], ref), f'calc_performance_stats == TD Eq.35 literal {ref}')


def test_evaluator_end_to_end():
    src = subprocess.run(['git', '-C', REPO, 'show', f'{BRANCH}:evaluate_confs.py'], capture_output=True, text=True).stdout
    ev = os.path.join(TMP, 'evaluate_confs.py')
    open(ev, 'w').write(src)

    mols = {}
    # A: symmetric, 3 GT conformers, 6 generated (renumbered, different L); gen0 == GT0 exactly
    gtA = confs_of('CC(C)CCO', 3, 11, mmff=True)
    genA = [renumber(gtA[0], 0)] + [renumber(m, i + 1) for i, m in enumerate(confs_of('CC(C)CCO', 5, 99))]
    # B: model failure (absent from confs)
    gtB = confs_of('CCCCO', 2, 3, mmff=True)
    # C: "additional failure": generated mols of a different molecule -> GetBestRMS raises
    gtC = confs_of('CCOC', 2, 4, mmff=True)
    genC = confs_of('CCCC', 4, 5)
    # D: one GT conformer of a different molecule, must be removed by clean_confs (L=1 after cleaning)
    gtD = confs_of('CCO', 1, 6, mmff=True) + confs_of('CCN', 1, 6, mmff=True)
    genD = confs_of('CCO', 2, 8)
    true_mols = {'CC(C)CCO': gtA, 'CCCCO': gtB, 'CCOC': gtC, 'CCO': gtD}
    preds = {'CC(C)CCO': genA, 'CCOC': genC, 'CCO': genD}
    pd.DataFrame({'smiles': list(true_mols), 'n_conformers': [3, 2, 2, 2], 'corrected_smiles': list(true_mols)}) \
        .to_csv(os.path.join(TMP, 'test.csv'), index=False)
    pickle.dump(true_mols, open(os.path.join(TMP, 'true.pkl'), 'wb'))
    pickle.dump(preds, open(os.path.join(TMP, 'confs.pkl'), 'wb'))
    out = os.path.join(TMP, 'eval.pkl')
    p = subprocess.run([PY, ev, '--confs', os.path.join(TMP, 'confs.pkl'), '--test_csv', os.path.join(TMP, 'test.csv'),
                        '--true_mols', os.path.join(TMP, 'true.pkl'), '--dataset', 'qm9', '--out_results', out],
                       capture_output=True, text=True, timeout=300, cwd=TMP)
    check(p.returncode == 0, 'evaluate_confs.py (branch) runs end-to-end on synthetic data')
    if p.returncode:
        print(p.stderr[-2000:])
        return None
    R = pickle.load(open(out, 'rb'))
    res = R['results']
    MA = res[('CC(C)CCO', 'CC(C)CCO')]['rmsd']
    refA = np.array([[ref_best_rms(g, t) for g in genA] for t in gtA])
    check(MA.shape == (3, 6) and np.allclose(MA, refA, atol=1e-4),
          f'A: RMSD matrix (L x K = 3 x 6) == independent heavy-atom Kabsch+automorphism RMSD (max |d|={np.abs(MA - refA).max():.1e})')
    check(MA[0, 0] < 1e-6, 'A: generated copy of GT0 (other atom order) has RMSD 0')
    MC = res[('CCOC', 'CCOC')]['rmsd']
    check(np.isnan(MC).all(), 'C: GetBestRMS exception -> whole rows NaN ("additional failure")')
    MD = res[('CCO', 'CCO')]['rmsd']
    check(MD.shape == (1, 2), f'D: clean_confs removed the wrong-SMILES GT conformer (L={MD.shape[0]})')
    S = R['summary']
    check(S['n_model_failures'] == 1 and S['n_evaluated'] == 3 and S['n_additional_failures'] == 1,
          f"failure accounting: evaluated={S['n_evaluated']} model_failures={S['n_model_failures']} additional={S['n_additional_failures']}")

    # independent aggregate: failures (B) count COV 0; C (all NaN) counts COV 0 and is dropped from AMR
    per = {k: ref_stats(v['rmsd'], 0.5) for k, v in res.items() if not np.isnan(v['rmsd']).all()}
    covr = [s[0] for s in per.values()] + [0.0, 0.0]  # + C (NaN -> 0) + B (failure)
    covp = [s[2] for s in per.values()] + [0.0, 0.0]
    amrr = [s[1] for s in per.values()]
    amrp = [s[3] for s in per.values()]
    exp = {'COV-R_mean': 100 * np.mean(covr), 'COV-R_median': 100 * np.median(covr), 'MAT-R_mean': np.mean(amrr),
           'MAT-R_median': np.median(amrr), 'COV-P_mean': 100 * np.mean(covp), 'COV-P_median': 100 * np.median(covp),
           'MAT-P_mean': np.mean(amrp), 'MAT-P_median': np.median(amrp)}
    check(S['threshold'] == 0.5, 'SUMMARY threshold defaults to 0.5 for --dataset qm9')
    bad = {k: (S[k], v) for k, v in exp.items() if abs(S[k] - v) > 1e-9}
    check(not bad, f'SUMMARY == independent TD-Eq.35 aggregation (mean/median, failures=0 COV, excluded from AMR) {bad or ""}')
    sw = S['sweep'][0.5]
    check(np.allclose(sw, (exp['COV-R_mean'], exp['COV-R_median'], exp['COV-P_mean'], exp['COV-P_median'])),
          'SWEEP thr=0.5 == SUMMARY (same conventions)')
    for thr in (0.05, 0.1, 0.25):
        cr = [ref_stats(v['rmsd'], thr)[0] for k, v in res.items() if not np.isnan(v['rmsd']).all()] + [0, 0]
        check(abs(S['sweep'][thr][0] - 100 * np.mean(cr)) < 1e-9, f'SWEEP thr={thr} COV-R mean == independent')
    # --report_threshold off the 0.125 grid must be honoured exactly (was snapped to the nearest grid value)
    out2 = os.path.join(TMP, 'eval_01.pkl')
    p = subprocess.run([PY, ev, '--confs', os.path.join(TMP, 'confs.pkl'), '--test_csv', os.path.join(TMP, 'test.csv'),
                        '--true_mols', os.path.join(TMP, 'true.pkl'), '--dataset', 'qm9', '--out_results', out2,
                        '--report_threshold', '0.1'], capture_output=True, text=True, timeout=300, cwd=TMP)
    S2 = pickle.load(open(out2, 'rb'))['summary'] if p.returncode == 0 else {}
    check(S2.get('threshold') == 0.1 and np.isclose(S2['COV-R_mean'], S2['sweep'][0.1][0])
          and np.isclose(S2['COV-P_median'], S2['sweep'][0.1][3]),
          f"--report_threshold 0.1 honoured exactly (SUMMARY threshold={S2.get('threshold')})")
    return out, genA, gtA, MA


# ------------------------------------------------------------------------------------------------ 3. geometry_metrics
def test_geometry_metrics():
    gm = load_tool('geometry_metrics', ['--results', 'x', '--confs', 'x', '--test_csv', 'x', '--true_mols', 'x',
                                        '--out', 'x'])
    smi = 'CC(C)CCO'
    gt = confs_of(smi, 1, 21, mmff=True)[0]
    # generated = GT geometry, +40 deg on the C2-C3 torsion, then renumbered
    g = Chem.Mol(gt)
    c = g.GetConformer()
    tors = [(0, 1, 3, 4)]  # C0-C1-C3-C4 (bond C1-C3)
    T.SetDihedralDeg(c, *tors[0], T.GetDihedralDeg(c, *tors[0]) + 40.0)
    gen = renumber(g, 3)
    M = np.array([[ref_best_rms(gen, gt)]])
    rows = gm.job((smi, smi, M, [gt], [gen]))
    r = rows[0]
    check('error' not in r, f'geometry_metrics.job runs ({r.get("error", "")})')
    _, _, tlist = gm.topology(Chem.RemoveHs(gen))
    n_t = len(tlist)
    check(r['bond_mae'] < 1e-6 and r['angle_mae'] < 1e-6, f"identical L after renumbering: bond MAE {r['bond_mae']:.1e} A, angle MAE {r['angle_mae']:.1e} deg (atom map correct)")
    check(abs(r['torsion_mae'] - 40.0 / n_t) < 1e-4, f"torsion MAE = 40/{n_t} = {40 / n_t:.3f} deg (got {r['torsion_mae']:.4f})")

    # periodic wrap: GT at +179, gen at -179 -> 2 deg, not 358
    a, b = Chem.Mol(gt), Chem.Mol(gt)
    T.SetDihedralDeg(a.GetConformer(), *tors[0], 179.0)
    T.SetDihedralDeg(b.GetConformer(), *tors[0], -179.0)
    genb = renumber(b, 5)
    rows = gm.job((smi, smi, np.array([[ref_best_rms(genb, a)]]), [a], [genb]))
    check(abs(rows[0]['torsion_mae'] - 2.0 / n_t) < 1e-4, f"torsion wrap +179 vs -179 -> 2/{n_t} deg (got {rows[0]['torsion_mae']:.4f})")

    # near-linear (sp) centres: a 0.02 A wiggle of the nitrile N must not produce a large torsion error
    smi2 = 'CC(C)CC#N'
    gt2 = confs_of(smi2, 1, 4, mmff=True)[0]
    g2 = Chem.Mol(gt2)
    n_idx = [a.GetIdx() for a in g2.GetAtoms() if a.GetSymbol() == 'N'][0]
    pos = g2.GetConformer().GetAtomPosition(n_idx)
    g2.GetConformer().SetAtomPosition(n_idx, (pos.x + 0.02, pos.y - 0.02, pos.z + 0.01))
    gen2 = renumber(g2, 7)
    rows = gm.job((smi2, smi2, np.array([[ref_best_rms(gen2, gt2)]]), [gt2], [gen2]))
    rr = rows[0]
    check('error' not in rr and (np.isnan(rr['torsion_mae']) or rr['torsion_mae'] < 1.0),
          f"sp-centre torsions excluded: 0.02 A N wiggle -> torsion MAE {rr.get('torsion_mae')} deg "
          f"(RMSD {rr.get('rmsd', float('nan')):.3f} A)")


# ------------------------------------------------------------------------------------------------ 4. local_structure
def test_local_structure():
    ls = load_tool('local_structure_analysis', ['--mode', 'test', '--out', os.path.join(TMP, 'ls.csv'),
                                                '--maxiter', '20', '--popsize', '10'])
    m = confs_of('CCCCO', 1, 3, mmff=True)[0]
    all_t, rel_t = ls.relevant_torsions(m)
    check(len(all_t) == 4 and len(rel_t) == 2, f'relevant_torsions(CCCCO): repo={len(all_t)} (4 incl. CH3/OH), heavy={len(rel_t)} (2)')
    # identical L, scrambled torsions -> floor ~ 0
    seed = Chem.Mol(m)
    rng = np.random.RandomState(0)
    for t in rel_t:
        T.SetDihedralRad(seed.GetConformer(), *t, float(rng.uniform(-np.pi, np.pi)))
    maps = ls.heavy_automorphisms(m)
    before = ls.heavy_rmsd(seed, m, -1, -1, maps)
    fl = ls.torsion_floor(seed, -1, m, rel_t, maps)
    check(fl < 1e-3 < before, f'torsion floor with identical L: {before:.3f} A scrambled -> {fl:.2e} A')
    # symmetry-aware RMSD (min over automorphism maps) == GetBestRMS on heavy atoms (no conjugated terminal groups)
    diffs = []
    for smi in ('CC(C)(C)CO', 'CC(C)CC#N', 'OCC(O)CO', 'CC1CC1C'):
        a, b = confs_of(smi, 2, 31, mmff=True)
        mp = ls.heavy_automorphisms(a)
        diffs.append(abs(ls.heavy_rmsd(Chem.Mol(a), b, -1, -1, mp) -
                         rdMolAlign.GetBestRMS(Chem.RemoveHs(a), Chem.RemoveHs(b))))
    check(max(diffs) < 1e-5, f'heavy_rmsd(automorphisms) == GetBestRMS(RemoveHs) (max |d| = {max(diffs):.1e})')
    # full job on nitrile / alkyne / isopropyl molecules: finite, floor <= transplant (same seed), and when the
    # seed's stereo matches GT the floor of these 1-2 torsion molecules is small (was 0.46 A with the identity map)
    for smi in ('CC(C)CC#N', 'CC#CC(C)O', 'CC(C)CCO'):
        gts = confs_of(smi, 3, 17, mmff=True)
        rows = ls.job((smi, smi, gts))
        errs = [r for r in rows if 'error' in r]
        df = pd.DataFrame([r for r in rows if 'error' not in r])
        ok = not errs and len(df) == 3 and np.isfinite(df.floor_rmsd).all() and \
            (df.floor_rmsd <= df.transplant_rmsd_assigned + 1e-9).all() and \
            (df.floor_rmsd[df.stereo_match.astype(bool)] < 0.2).all()
        check(ok, f'local_structure job({smi}): finite, floor<=transplant, floor<0.2 A if stereo matches; '
                  f'floor={df.floor_rmsd.round(3).tolist() if len(df) else errs} '
                  f'stereo_match={df.stereo_match.tolist() if len(df) else ""}')


# ------------------------------------------------------------------------------------------------ 5. breakdown
def test_breakdown(out):
    p = subprocess.run([PY, os.path.join(TOOLS, 'breakdown.py'), '--results', out, '--threshold', '0.5'],
                       capture_output=True, text=True, timeout=120)
    check(p.returncode == 0, 'breakdown.py runs on evaluator output')
    R = pickle.load(open(out, 'rb'))
    line = [ln for ln in p.stdout.splitlines() if ln.startswith('| 0.500')]
    vals = [float(x) for x in line[0].strip('|').split('|')[1:]] if line else []
    check(bool(vals) and np.allclose(vals, R['summary']['sweep'][0.5], atol=0.006),
          f'breakdown sweep at 0.5 == evaluator SWEEP {vals} vs {np.round(R["summary"]["sweep"][0.5], 2).tolist()}')


if __name__ == '__main__':
    test_rdkit_facts()
    test_calc_performance_stats_strict()
    r = test_evaluator_end_to_end()
    test_geometry_metrics()
    test_local_structure()
    if r:
        test_breakdown(r[0])
    print(f'\n{len(FAILED)} failed' + (':\n  ' + '\n  '.join(FAILED) if FAILED else ''))
    sys.exit(1 if FAILED else 0)

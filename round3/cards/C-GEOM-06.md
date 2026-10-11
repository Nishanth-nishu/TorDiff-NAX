# C-GEOM-06: srETKDGv3 seeds (RDKit small-ring torsion terms) as a zero-cost ring-L control

## 1. Idea (scout)
Embed the test-time seeds with RDKit's `srETKDGv3` parameters (small-ring torsion preferences switched on) instead of the
keyword-default ETKDG, optionally followed by MMFF or xTB, and use them as L for CTRL/B1. Revised after D-208: the
cluster default is ETKDG **v1**, so `srETKDGv3` changes several settings at once (ETversion 1 → 2, macrocycle torsions,
1-4 bounds, small-ring torsions); the experiment therefore uses two single-factor steps (v1 → v3, then v3 → srv3). It is
the natural "free" baseline any ring-L source (C-GEOM-01/02/03) must beat.

## 2. Where it comes from (scout)
- RDKit's small-ring torsion terms are off by default (`useSmallRingTorsions` default False); a preconfigured
  `srETKDGv3` parameter object exists [E-GEOM-039]. TD and our seed builders call `EmbedMultipleConfs` without a
  parameter object [E-GEOM-008]; ETKDG has been the Python default since 2018.09 [E-GEOM-040]; in RDKit 2026.03 the
  keyword default is ETKDGv3 without small-ring terms [E-GEOM-041], but in the cluster's 2022.9.5 it is ETKDG v1
  (`ETversion=1`, small-ring and macrocycle terms off) [E-GEOM-052]. So all our training-matching and test seeds are v1.
- PuckerFlow's benchmark (substituent-free 5–8 rings): the small-ring terms raise unrelaxed pucker recall coverage from
  53.3% to 60.1% (AMR 0.14 → 0.13 Å) [E-GEOM-022, E-GEOM-026] but leave all-atom recall AMR at 0.17 Å, and after MMFF
  0.14 vs 0.15 Å [E-GEOM-023].
- Practitioner report (piperazine): srETKDGv3 still produced 28% twisted rings [E-GEOM-045].

## 3. Why it could matter here (scout)
- The QM9 L gap is the ring-pucker tail: ETKDG 28.6% of ring seeds > 10° ring-dihedral RMSD (49.6% excluding the
  all-3-ring molecules) vs 0.01% at λ = 0.75 [E-GEOM-002], worst for 4–7-rings [E-GEOM-003]. Small-ring torsion terms
  target exactly these rings.
- Rigid molecules (≈ 30% of the test set) would show any change without GPU, if the seeds are scored directly on one
  common set (D-205 design) [E-GEOM-004].

## 4. Assumptions that may not transfer (scout)
- Evidence predicts a small effect on our heavy-atom RMSD metric: +7 points pucker coverage but no all-atom AMR change on
  PuckerFlow's set [E-GEOM-022, E-GEOM-023], and persistent twisted 6-rings in practice [E-GEOM-045].
- ETKDG's torsion preferences come from crystal structures (CSD), not GFN2-xTB gas-phase minima, so the small-ring
  terms push puckers toward a different target than GEOM's [E-GEOM-050] (INFERENCE for the small-ring terms
  specifically).
- Training-side consistency: CTRL was matched to ETKDG v1 L [E-GEOM-052]; switching test L only is a (small) shift.
- Molecules whose smallest ring is 3-membered (381 of the ring molecules) mostly have little pucker to fix; only 9% of
  their ETKDG seeds exceed 10° [E-GEOM-003].

## 5. Minimal experiment (scout)
- CPU only, inside the C-GEOM-01 seed builder run, same random seed and molecule set for all three (revised after
  D-208): `L_etkdg2L` (v1, existing), `L_etkdgv3_2L` (`ETKDGv3()`, small-ring terms off) and `L_sretkdg2L`
  (`srETKDGv3()`), each also MMFF-relaxed. Single-factor contrasts: v1 → v3 (version/macrocycle/1-4 settings) and
  v3 → srv3 (small-ring terms only). Report the C-GEOM-01 gate metrics (> 10° share by ring size on both bases, pucker
  coverage, seed AMR-R on the common rigid set, D-205 design).
- GPU (2–6 inference runs, ≤ 1.5 GPU-h) **only if** a contrast lowers the > 10° share (excl. all-3-ring basis) by ≥ 5
  points or the common-rigid-set seed AMR-R by ≥ 0.010 Å. Primary: paired AMR-R of CTRL on the winning variant vs
  `L_etkdg2L`. Expected: the v3 → srv3 step alone ≤ 0.005 Å better (from E-GEOM-023); the v1 → v3 step is not covered by
  our sources (no expectation stated). Note CTRL was matched to v1 L, so either change is also a small train/test
  shift.

## 6. Scout's own call (scout)
Worth trying: MAYBE — near-free as a CPU arm and a necessary baseline for the ring cards, but the literature predicts a
small all-atom effect, so it earns GPU time only through its CPU gate.

## 7. Code grounding (grounder)
- **Touchpoints:** `tools/make_l_seed_pickles.py:89` calls `EmbedMultipleConfs(emb, numConfs=K, randomSeed=args.seed)`
  with no params object, i.e. ETKDG v1 [E-GEOM-052]. Add `--etkdg_params {v1,v3,v3sr}` (default unchanged): set
  `ps.randomSeed = args.seed`, call `EmbedMultipleConfs(emb, K, ps)`, and tag the outputs `L_etkdgv3_2L` /
  `L_sretkdg2L` (+ `_mmff`), written into a new `round3_seeds` dir. ~30 lines.
- **Transfer check (measured in the cluster venv, RDKit 2022.09.5):**

  | params object | ETversion | small-ring | macrocycle |
  |---|---|---|---|
  | `ETKDG()` | 1 | off | off |
  | `ETKDGv3()` | 2 | off | on |
  | `srETKDGv3()` | 2 | on | **off** |

  The card's "v3 → srv3 = small-ring terms only" therefore **also switches macrocycle torsions off**. For a true single
  factor, use `ETKDGv3()` with `useSmallRingTorsions = True`.
- **Pairing caveat:** the same `randomSeed` under different params gives different embeddings (not common random
  numbers). Pair at molecule level, on the common set; embedding failures can differ (`:90-92`).
- **Verdict: IMPLEMENTABLE WITH CHANGES.**
- **Cost:** CPU ≈ 1–2 h of seeds + ≈ 1 h gate; GPU ≤ 1.0 only if the gate passes.
Detail, line numbers and cluster checks: `round3/grounding.md` §3 (shared pieces S-1…S-6 in §2.3).
## 8. Predicted effect on our project (analyst)
Data and model: `round3/impact.md` §3.
- **Ceiling.** Tail-driven, as for C-GEOM-01 (CTRL line).
- **Expected effect: reasoning chain.**
  1. The small-ring torsion terms raise unrelaxed pucker coverage from 53.3 % to 60.1 % on PuckerFlow's
     5–8-ring set, ≈ 15 % fewer misses (E-GEOM-022).
  2. Mapping "15 % fewer bad seeds" onto our tail is generous: the rule coverage of 4-rings (60 % bad) is unknown,
     and CSD-derived preferences target the wrong level of theory (E-GEOM-050). The result is X 49.6 % → 42–46 %.
  3. That gives CTRL −0.007 to −0.014 on the tail line, or −0.005 to −0.011 after the 76 % realisation seen for
     MMFF. After MMFF relaxation: 35.1 % → 30–33 %, i.e. −0.004 to −0.010.
  4. The v1 → v3 step has no source and an unknown sign. CTRL was trained on v1-matched L, so there is a small
     train/test shift.
- **Worst case.** +0.005 (shift with no pucker gain). B1 gains nothing (acyclic L stays RDKit-like).
- **Detectability.** The paired L-source MDE is ≈ 0.007 for relaxed twins of the same embeddings (A2 vs R0). The
  srETKDG variants are different embeddings (grounding.md: same `randomSeed` is not common random numbers), so the
  MDE is ≈ 0.008–0.010. The expected effect is therefore probably undetectable even with 3 checkpoints.
- **Decision value.** Modest. It is the free baseline every ring-L source must beat.
- **Worth trying: MAYBE.** The CPU part is YES: it is near-free inside the C-GEOM-01 builder. GPU only if the tail
  (ring ≥ 4 basis) drops ≥ 5 points; the tail is the right gate metric here. ≈ 3–5 mÅ per GPU-h.

**Revised after cross-read (2026-10-11).**
- In RDKit 2022.9.5, `srETKDGv3()` also turns macrocycle torsions off (grounder). The estimate above (CTRL −0.005 to
  −0.011) now refers to the single-factor contrast `ETKDGv3()` + `useSmallRingTorsions = True`. QM9 rings have
  ≤ 9 atoms, so the macrocycle switch is probably inert.
- The variants are different embeddings, with no common random numbers, so the MDE is ≈ 0.008–0.010 and the expected
  effect is probably undetectable (already noted).
- GPU ≤ 1.0 GPU-h.
- Call unchanged: **MAYBE** (the CPU part yes; GPU only if the tail drops ≥ 5 points).

## 9. Panel (judges)

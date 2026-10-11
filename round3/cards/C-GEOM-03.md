# C-GEOM-03: Learned Cartesian L from a pretrained ET-Flow (qm9-o3) as test-time L — diagnostic before building our own refiner

## 1. Idea (scout)
Generate 2L conformers per test molecule with the released ET-Flow QM9 checkpoint (split to be confirmed in step 0), keep
only their local structure as seeds (TD re-randomises torsions), and run the existing CTRL_rematch and B1 checkpoints on
them. This is the cheapest test of brief Q1 for **learned** L: does a state-of-the-art Cartesian model's L reach the
λ = 0.75 spec, especially the ring-pucker part? Only if it does is a home-built L refiner (round-2 R2A-5 style) worth
its 1–2 days of coding and a training run.

## 2. Where it comes from (scout)
- On GEOM-QM9 (TD protocol), ET-Flow reports recall AMR mean 0.073 Å (median 0.047) vs MCF 0.103 and TD 0.178
  [E-GEOM-028].
- The repo is pip-installable, ships a `qm9-o3` checkpoint with automatic download, and points to the TD split files;
  the PyPI package pins `numpy==1.26.4` and leaves torch unpinned, while only the dev `env.yml` pins
  `pytorch-cuda==12.1` [E-GEOM-037, E-GEOM-036]. The `qm9-o3.ckpt` is stored with `QM9.zip` split files in one Zenodo
  record and the scaffold splits in another [E-GEOM-054].
- ET-Flow's base O(3) model applies a post hoc chirality correction (oriented volume vs RDKit tags, flip on mismatch),
  and that corrected model is the reported 0.073 [E-GEOM-053]; the released `qm9-o3` config defaults to this correction
  inside `sample()`/`predict()` [E-GEOM-054]. (Revised after D-204: the earlier claim that only SO(3) is
  chirality-corrected is withdrawn.)
- Refiner literature, for the "build our own" alternative: FM-refiner starts from upstream conformers rather than noise
  [E-GEOM-029] but only corrects errors inside its noise range [E-GEOM-030]; Equivariant Blurring Diffusion corrects an
  RDKit fragment prior instead of freezing it [E-GEOM-051]; GO-Flow finds internal coordinates the most important
  component, i.e. a pure-Cartesian model is the weaker design [E-GEOM-031].

## 3. Why it could matter here (scout)
- B1 beats CTRL only at λ ≈ 0.75-quality L (0.115 vs 0.116) and collapses on RDKit/MMFF L (0.236 / 0.232)
  [E-GEOM-007]; the spec is in C-GEOM-01 §3 [E-GEOM-001, E-GEOM-002].
- ET-Flow's own AMR-R (0.073) is below our λ = 0.75 CTRL (0.116) and A5ring (0.119) results [E-GEOM-028,
  E-GEOM-013], so its L is a plausible candidate for that spec (INFERENCE: whole-conformer AMR does not split L from
  torsions; the CPU gate measures L directly).
- The rigid subset (≈ 30%) measures ET-Flow's L quality with no torsion model in the loop when the seeds are scored
  directly on one common set (logged bin-0 values are only indicative, D-205) [E-GEOM-004].
- If ET-Flow L fails the spec on rings, a home-built low-noise Cartesian refiner is unlikely to pass it either: wrong
  puckers are large discrete moves outside a refiner's reachable range [E-GEOM-030] (INFERENCE).

## 4. Assumptions that may not transfer (scout)
- **Environment (revised after D-206).** ET-Flow's `numpy==1.26.4` pin conflicts with the TD venv's numpy 1.23.5, so it
  needs its own env [E-GEOM-037, E-GEOM-014]. The torch version and CUDA build are open: the pip package leaves torch
  unpinned; try a CUDA-11.8 torch 2.x first, as in our `setup_env.sh` cu118 profile [E-GEOM-014]; CUDA 12.1 is only the
  dev env's choice. Which torch/lightning combination works is untested.
- **Reproducibility.** The README warns that changed preprocessing may not reproduce paper numbers [E-GEOM-037];
  our evaluator, not the paper's 0.073, is the reference.
- **Chirality (revised after D-204).** Handled by ET-Flow's own post hoc flip [E-GEOM-053, E-GEOM-054]; that flip is
  whole-molecule, so samples with only some stereocentres inverted can remain. Keep our per-seed stereo check
  (`stereo_ok`) and drop failures.
- **Atom order / hydrogens.** ET-Flow outputs must be mapped onto the GT-graph atom order used by our seed pickles
  (substructure match); H positions come from ET-Flow.
- **Split leakage.** The README points to TD split files and the checkpoint is stored next to `QM9.zip` rather than
  the scaffold-split record [E-GEOM-037, E-GEOM-054], but that `qm9-o3` was trained on TD's random split is inferred,
  not shown: step 0 compares `QM9.zip`'s test SMILES with our `test_smiles.csv`; any overlap of its training set with
  our test set kills the card.
- **Scope.** A strong Cartesian generator as L source turns TD into a torsion resampler on top of it; this card is a
  diagnostic for Q1, not a proposed final method.

## 5. Minimal experiment (scout)
- **Step 0 (time-box 1 h):** (a) download `QM9.zip` from the ET-Flow Zenodo record and confirm its test split equals
  TD's 1000 test molecules (and that no test SMILES is in its train split); (b) separate env with numpy 1.26.4, a
  torch 2.x (cu118 first) and `pip install etflow`; smoke-test `predict()` on 5 molecules incl. stereocentres. If (a)
  fails, stop; if (b) exceeds the time-box, stop and record why.
- **Step 1 (GPU, ≤ 1 GPU-h):** sample 2L conformers per test molecule (default step count); map atoms; stereo check;
  write `L_etflow` seed pickle + `l_error.csv` rows with the existing builder's F5 code.
- **Step 2, CPU gate (C-GEOM-01 design, revised after D-205):** ring/acyclic RMSD, share of ring seeds > 10° (both
  bases), pucker coverage, and seed AMR-R on the common rigid set (empty TD `edge_mask`), with ETKDG / MMFF / λ
  references recomputed on the same set; go to GPU if `L_etflow` beats MMFF there (non-oracle comparison). Also evaluate the raw ET-Flow conformers in our evaluator as the
  reference point.
- **Step 3, GPU (inference only):** CTRL_rematch s0–2 and B1 s0–2 on `L_etflow` = 6 runs (+ S3/S4 if final).
  Primary: paired B1-on-ET-Flow-L vs CTRL-on-ET-Flow-L (expected B1 ≤ CTRL if the gate meets the λ = 0.75 row), and
  CTRL-on-ET-Flow-L vs CTRL-on-ETKDG (expected lower). Report raw ET-Flow AMR-R alongside.
- **Decision rule for R2A-5-style home refiner:** build it only if ET-Flow L meets the spec on rings (> 10° share ≤ 1%)
  **and** C-GEOM-01/02 do not; otherwise defer.
- **Cost:** ≤ 2.5 GPU-h + env setup (half a day). No training.

## 6. Scout's own call (scout)
Worth trying: YES (revised after D-204/D-206; was MAYBE) — the two reasons for MAYBE are withdrawn (chirality is already
corrected in the released model; CUDA 12 is not required). It is the cheapest direct answer to "can learned L reach
λ ≈ 0.75?" (≤ 2.5 GPU-h, no training), with explicit kill criteria in step 0 (split identity, env time-box). It is a
diagnostic, not a method, so it ranks below C-GEOM-01/02 in priority.

## 7. Code grounding (grounder)
- **D-210 is a hard precondition.**
  - TD's test set is `test_smiles.csv`; its split indexes the **sorted** raw file list (`utils/dataset.py:168, 171,
    179`).
  - `qm9-o3`'s `train_indices.npy` index ET-Flow's own processed order (V3).
  - gnode118 holds **no GEOM `summary_qm9.json` and no ET-Flow file** (checked `data/QM9/`, `downloads/`).

  Step 0(a) therefore needs `QM9.zip` (2.9 MB, Zenodo; a download, so **user permission** is required; gnode118 can
  reach Zenodo). It passes only if `QM9.zip` carries SMILES that can be intersected with our test columns 0 and 2 and
  the intersection is empty. Index arrays alone are **inconclusive, so STOP**. ORACLE relabelling is no substitute,
  because memorised L would pass the very spec being tested.
- **If step 0 passes:**
  - Separate env: `$PROJECT/micromamba` exists; numpy 1.26.4, torch 2.x with cu121 or cu118 (driver 580.178.04 supports
    both); internet OK. The install needs approval.
  - `tools/etflow_seeds.py` in that env writes `.npz` arrays, not cross-version RDKit pickles.
  - The TD-venv side maps atoms onto the GT-graph mol (`make_l_seed_pickles.py:84-85`), applies `stereo_ok` (`:104`,
    `:116`; ET-Flow's whole-molecule flip [E-GEOM-053/054] can leave partial inversions), and writes `L_etflow.pkl` +
    F5 rows. With `--seed_confs_cycle` fewer valid seeds still work (`sampling.py:86`).
  - Raw ET-Flow conformers go into the unchanged evaluator as the reference row.
  - ~200 lines + an atom-mapping round-trip test.
- **Verdict: CONDITIONAL.**
  - Conclusive step 0: IMPLEMENTABLE WITH CHANGES.
  - Inconclusive, or download not permitted: **NOT FEASIBLE THIS ROUND**.

  The environment risk is lower than §4 states.
- **Cost:** ≤ 1 GPU-h of ET-Flow sampling (unmeasured) + 1.0–1.9 GPU-h inference; ~1 CPU-h; ~half a day env.
Detail, line numbers and cluster checks: `round3/grounding.md` §3 (shared pieces S-1…S-6 in §2.3).
## 8. Predicted effect on our project (analyst)
Data and model: `round3/impact.md` §3. Round-2 numbers are PRELIMINARY. D-210 is OPEN.
- **Ceiling.** Same as C-GEOM-01: CTRL 0.086, B1 0.038 (random-GT-conformer L, ORACLE). Raw ET-Flow reports 0.073
  (published) and 0.083 (FM-refiner re-run) (E-GEOM-028, E-EVAL-008). CTRL cannot go below ≈ 0.086 on any L, so
  **TD + CTRL cannot beat raw ET-Flow**. Only a B1/S3-type model on near-reference L can: B1 reaches 0.073 at a tail
  of ≤ ≈ 14 % if the L is reference-like (impact.md §3f).
- **Expected effect: reasoning chain.**
  1. ET-Flow's AMR of 0.073 implies most puckers are right; I assume a tail of X ≈ 5–15 %.
  2. CTRL is tail-driven: 0.10–0.12 (impact.md §3d).
  3. B1 depends on ET-Flow's bond/angle error. A generative Cartesian model does not land on the xTB minimum.
     - At noise-0.01-level error, B1 ≈ 0.06–0.08.
     - At noise-0.02-level error (0.027 Å), B1 ≈ 0.12. B1 on noise 0.02 is 0.092–0.134 across seeds.
     So B1 lies between 0.05 and 0.15.
  4. Transfer. ET-Flow's AMR is a whole-conformer number and does not separate L from torsions. The CPU gate
     measures L directly.
- **Worst case.**
  - If `qm9-o3` saw any TD test molecule in training (D-210), ET-Flow L is partly memorised GT. Every number is
    then ORACLE-contaminated and optimistic by an unknown amount, which is a validity blocker.
  - With clean but noisy ET-Flow L: CTRL ≈ 0.12 and B1 ≥ CTRL.
- **Detectability.** The effects vs ETKDG (≥ 0.06) are clear with 1 seed. B1 vs CTRL CI ≈ ±0.006.
- **Decision value.** Diagnostic only.
  - Positive: learned Cartesian L can meet the spec, which justifies a home-built refiner only if C-GEOM-01/02 fail
    on rings.
  - Negative: a refiner is unlikely to help.
  - Inconclusive split check: no usable decision at all.
  - Branch A of C-GEOM-01 (xTB L) answers most of the same question at the reference level of theory and without the
    leakage risk.
- **Worth trying: MAYBE.** YES only after step 0(a) conclusively shows no TD test molecule in `qm9-o3` train/val;
  otherwise NO. Run it after C-GEOM-01. Gain per GPU-h: n/a (diagnostic; ≈ 2.5 GPU-h + 0.5 day env).

**Revised after cross-read (2026-10-11).**
- Leak correction (impact.md §3g): B1 needs each target's own-basin L.
- ET-Flow's samples are conformer-specific and accurate (FM-refiner re-run COV-R@0.05 75.7 %, E-EVAL-008), so its
  seeds may carry own-basin L for many targets. B1/S3 on ET-Flow L could then reach 0.05–0.08, but only by TD
  re-using ET-Flow's own conformers (TD + B1 ≈ an ET-Flow resampler). CTRL 0.09–0.12.
- D-210 still decides validity: memorised L would inflate exactly this own-basin share. Downloading `QM9.zip` needs
  user permission (grounder).
- Call unchanged: **MAYBE** (YES only after a conclusive step 0(a)).

## 9. Panel (judges)

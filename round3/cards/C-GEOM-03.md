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
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)

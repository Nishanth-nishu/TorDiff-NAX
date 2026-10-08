# C-GEOM-03: Learned Cartesian L from a pretrained ET-Flow (qm9-o3) as test-time L — diagnostic before building our own refiner

## 1. Idea (scout)
Generate 2L conformers per test molecule with the released ET-Flow QM9 checkpoint (trained on the same TD split), keep
only their local structure as seeds (TD re-randomises torsions), and run the existing CTRL_rematch and B1 checkpoints on
them. This is the cheapest test of brief Q1 for **learned** L: does a state-of-the-art Cartesian model's L reach the
λ = 0.75 spec, especially the ring-pucker part? Only if it does is a home-built L refiner (round-2 R2A-5 style) worth
its 1–2 days of coding and a training run.

## 2. Where it comes from (scout)
- On GEOM-QM9 (TD protocol), ET-Flow reports recall AMR mean 0.073 Å (median 0.047) vs MCF 0.103 and TD 0.178
  [E-GEOM-028].
- The repo is pip-installable, ships a `qm9-o3` checkpoint with automatic download, and uses the TD splits; its
  dev environment pins `pytorch-cuda==12.1` and `numpy==1.26.4` [E-GEOM-037].
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
- The rigid subset (≈ 30%) measures ET-Flow's L quality with no torsion model in the loop [E-GEOM-004].
- If ET-Flow L fails the spec on rings, a home-built low-noise Cartesian refiner is unlikely to pass it either: wrong
  puckers are large discrete moves outside a refiner's reachable range [E-GEOM-030] (INFERENCE).

## 4. Assumptions that may not transfer (scout)
- **Environment.** ET-Flow needs torch ≥ 2.1 / CUDA 12.1 and numpy 1.26.4, so it cannot share the TD venv (python
  3.9, torch 1.13.1+cu117, PyG 2.0.4) [E-GEOM-014, E-GEOM-037]. A separate conda env is needed; RTX 3090 is supported
  by CUDA 12.1 only with a recent enough driver (gnode118 driver UNVERIFIED). Fallback: CPU sampling (slow, not
  timed).
- **Reproducibility.** The README warns that changed preprocessing may not reproduce paper numbers [E-GEOM-037];
  our evaluator, not the paper's 0.073, is the reference.
- **Chirality.** `qm9-o3` is the O(3) model; the paper's chirality-corrected variant is the SO(3) one
  [E-GEOM-048, E-GEOM-037]. Mirror-image samples must be detected (stereo check) and reflected or dropped.
- **Atom order / hydrogens.** ET-Flow outputs must be mapped onto the GT-graph atom order used by our seed pickles
  (substructure match); H positions come from ET-Flow.
- **Split leakage.** Must confirm the `qm9-o3` training config used the TD train split only (README says it uses TD
  split files) [E-GEOM-037].
- **Scope.** A strong Cartesian generator as L source turns TD into a torsion resampler on top of it; this card is a
  diagnostic for Q1, not a proposed final method.

## 5. Minimal experiment (scout)
- **Step 0 (time-box 1 h):** separate conda env with `pip install etflow` on top of a CUDA 12.1 torch; smoke-test on 5
  molecules. If the time-box fails, stop and record why.
- **Step 1 (GPU, ≤ 1 GPU-h):** sample 2L conformers per test molecule (default step count); map atoms; stereo check;
  write `L_etflow` seed pickle + `l_error.csv` rows with the existing builder's F5 code.
- **Step 2, CPU gate:** ring/acyclic RMSD, ring-dihedral share > 10°, pucker coverage, rigid-subset AMR-R of the
  seeds; compare with the λ table (C-GEOM-01 §3). Also evaluate the raw ET-Flow conformers in our evaluator as the
  reference point.
- **Step 3, GPU (inference only):** CTRL_rematch s0–2 and B1 s0–2 on `L_etflow` = 6 runs (+ S3/S4 if final).
  Primary: paired B1-on-ET-Flow-L vs CTRL-on-ET-Flow-L (expected B1 ≤ CTRL if the gate meets the λ = 0.75 row), and
  CTRL-on-ET-Flow-L vs CTRL-on-ETKDG (expected lower). Report raw ET-Flow AMR-R alongside.
- **Decision rule for R2A-5-style home refiner:** build it only if ET-Flow L meets the spec on rings (> 10° share ≤ 1%)
  **and** C-GEOM-01/02 do not; otherwise defer.
- **Cost:** ≤ 2.5 GPU-h + env setup (half a day). No training.

## 6. Scout's own call (scout)
Worth trying: MAYBE — scientifically clean and cheap answer to "can learned L reach λ ≈ 0.75?", but it hinges on a
separate CUDA-12 environment and an O(3) chirality fix; run it after C-GEOM-01's CPU gate, and drop it if the env
time-box fails.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)

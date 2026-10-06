# FlexiTors-Diffusion: Progress Report (Phase 1 — Preparation)

**Date:** 30 September 2026
**Project:** Extending Torsional Diffusion (Jing et al., NeurIPS 2022) toward *FlexiTors-Diffusion*: joint diffusion over torsion angles and local bond geometry
**Current phase:** literature study, code audit and ablation design on QM9 (GEOM-DRUGS comes later)

> **Status in one line:** the literature review, code audit, metric verification and experiment design are all done. **No QM9 experiments have been run yet**, because the GPU cluster (ada.iiit.ac.in, node gnode118) is not reachable from my machine yet (SSH times out; VPN and SSH key setup are pending). Every number below comes from the TorDiff paper or from small local tests on synthetic molecules, not from our own QM9 runs.

---

## 1. Goal

Torsional Diffusion (TorDiff) generates molecular conformers by diffusing only over the torsion angles of rotatable bonds. Bond lengths and bond angles are frozen at the values RDKit produces. My professor suggested three gaps in this design and proposed a joint model that diffuses over torsions *and* a small space of bond-angle corrections.

The goals of Phase 1:
1. Check the proposed gaps against what the paper actually says.
2. Survey recent work (2025–2026) that could address them.
3. Understand the TorDiff code and find where the "frozen geometry" assumption lives.
4. Design a rigorous QM9 ablation study that measures how much this assumption costs.

---

## 2. What was done

The work was split across several AI agents, each checking the others' output against a shared, locally stored library of papers.

### 2.1 Literature study

**What:**
- Stored the TorDiff paper with its full appendix, plus an in-depth analysis (`papers/core/tordiff_analysis.md`).
- Collected 26 related papers, mostly from 2025–2026 (`papers/related/`). Each has a summary note saying which gap it addresses.
- Built a search index of the library (`papers/INDEX.md`) and a synthesis of the gaps with a proposed design (`papers/gap_synthesis.md`).

**Why:** to verify the professor's gaps against the source before spending GPU time on them, and to check whether FlexiTors is still novel.

**Findings on the three gaps:**

| Proposed gap | Verdict | Evidence |
|---|---|---|
| 1. Bond lengths and angles are frozen at RDKit's values | **Holds**, needs tighter wording | Fails on rings and macrocycles. On GEOM-XL, TorDiff produces nothing for 25–27 of the 102 molecules because RDKit can't build a starting structure (reported by the MCF and ET-Flow papers). The TorDiff paper itself blames its XL results on molecule size, not local geometry. |
| 2. Train/test mismatch (trained on true geometry, tested on RDKit geometry) | **Not accurate as stated** | TorDiff already avoids this with *conformer matching* (Sec. 4.1, App. E). Training on true geometry is only an ablation, and there coverage collapses from 72.7% to 34.8% (Table 8). The real issue is that RDKit's geometry caps the achievable accuracy. |
| 3. Torsions and bond angles are coupled, and the model can't capture this | **Holds**, with stronger evidence | App. F.1 (p. 21): on DRUGS, true bond geometry taken from a *different* conformer only lowers the error floor from 0.324 Å to 0.284 Å. Most of the remaining error needs geometry that adapts to the torsions, which is exactly what FlexiTors adds. |

**Most relevant recent work:**
- **GO-Flow (2026):** flow matching over bond lengths, angles and torsions together. It is the closest existing work and the main **novelty risk**. FlexiTors must stand apart by:
  - learning a small correction on top of RDKit's geometry rather than generating all internal coordinates;
  - evaluating under TorDiff's exact protocol;
  - keeping exact likelihoods.
- **PuckerFlow (2026):** generates ring shapes and has already been combined with TorDiff. A ready-made component for rings.
- **DiffDock (2022):** the maths for diffusing over a combination of spaces (here, torsions × bond angles).
- **RINGER (2023):** diffuses bond angles and torsions together for macrocycles.
- **Corso's thesis (2023):** TorDiff's co-author proposes nearly this extension as future work.

**Evaluation finding:** on QM9, the standard 0.5 Å coverage threshold barely separates current methods. We therefore also report stricter thresholds (down to 0.05 Å) and bond/angle errors as secondary metrics.

### 2.2 Code audit

**What:** cloned the official repository, wrote a file-by-file walkthrough (`notes/code_walkthrough.md`), and located every place the frozen-geometry assumption is enforced:

| File | Role |
|---|---|
| `standardize_confs.py:72-118` | RDKit builds the structure; only torsions are fitted to the true conformer |
| `utils/torsion.py:57-75` | Torsion updates are rigid rotations, so bond lengths and angles never change |
| `diffusion/score_model.py:141-160` | The model outputs one score per rotatable bond, nothing for angles |
| `utils/training.py:19-25` | The loss covers torsions only |
| `diffusion/likelihood.py:68-119` | Likelihood is computed with the geometry held fixed |

**Why:** these are the exact places FlexiTors must change.

**Problems found in the original code:**
- **Settings overwritten:** the generation script silently replaces command-line settings with the model's saved training settings. For newly trained models, this turns on a likelihood option that counts molecules with no rotatable bonds as failures and makes ODE sampling 5–10× slower.
- **Identical conformers:** an RDKit random seed of 0 makes all generated conformers identical.
- **Double bonds can flip:** non-ring double bonds count as rotatable, so the model can flip cis/trans isomers.
- **No resume:** training cannot continue after an interruption.
- **Floor overestimated:** the paper's "RDKit floor" figures are computed with hydrogens included and a short optimisation, so they overestimate the floor.

All fixes and experiment hooks are on a separate branch (`flexitors-ablation-hooks`, 6 commits), exported as `patches/0001-ablation-hooks.patch`.

### 2.3 Metric verification

**What:**
- Compared every evaluation metric in the code with the paper's formal definitions.
- Wrote 35 automated tests on synthetic molecules (`review/tests/test_metrics.py`).
- Wrote up the findings in `review/metric_verification.md`.

**Why:** if a metric is wrong, every ablation built on it is wrong.

**Result:**
- The coverage and matching metrics (COV-R, AMR-R, COV-P, AMR-P) match the paper's definition (App. G.3, p. 24):
  - heavy atoms only;
  - symmetry-aware RMSD;
  - a strict `<` threshold;
  - 0.5 Å for QM9.
- Three bugs were found and fixed, all in our own added analysis code. The most important: the RDKit-floor tool ignored molecular symmetry. In one test molecule it reported a floor of 0.459 Å when the true value is 0.073 Å.
- **All 35 tests pass.** For example:
  - the symmetry-aware RMSD matches RDKit's reference function to within 9×10⁻¹¹ Å;
  - angles of +179° and −179° are correctly treated as 2° apart.

### 2.4 Cross-validation of the experimental design

**What:** an independent review of the ablation plan (`review/cross_validation.md`). Each debatable point was argued by three independent reviewers and settled by majority vote.

**Most important finding:** an early claim that TorDiff "already reaches its RDKit floor on QM9" (0.178 vs 0.17 Å) was invalid.
- The paper's floor pairs each true conformer with *one* RDKit structure.
- The reported error lets each true conformer pick the best of *twice as many* generated conformers.

The two numbers are not comparable. The claim was removed, and the valid comparison (a floor measured on our own generated conformers) was added.

**Other fixes:**
- Added an experiment that directly tests torsion–angle coupling (A1c).
- Added an experiment that separates how much of the floor comes from bond angles versus rings, stereochemistry or bond lengths.
- Added a matched baseline so that ablations are not confounded by preprocessing differences.
- Added safeguards so that half-trained models (cut off by the time limit) are never evaluated.
- Added more random seeds, paired bootstrap confidence intervals and multiple-comparison correction (`tools/paired_compare.py`).
- Added a table of which results would *disprove* the FlexiTors motivation.

---

## 3. The QM9 ablation plan

**Primary metrics (fixed before any results):** mean AMR-R and COV-R at the **0.5 Å** threshold. Stricter thresholds and bond/angle errors are secondary.

**Budget:** 4 days on 4 GPUs (at most 384 GPU-hours). The plan uses an estimated **160–243 GPU-hours**.

| Experiment | Question it answers | GPU-h (expected / high) |
|---|---|---|
| RDKit floor analyses (CPU only) | How much error does RDKit's frozen geometry force? | 0 |
| 23 inference runs on the released pretrained model | How does accuracy change when the model gets true, random or matched true geometry? | 20.5 / 38.5 |
| Baseline retrain × 3 seeds | Our own reference model | 31.5 / 48 |
| B1 × 3: train on true geometry | Does the paper's Table 8 collapse repeat on QM9? | 31.5 / 48 |
| B2 × 3: random geometry pairing, with a matched baseline × 3 | Reproduces the paper's random-pairing ablation | 51 / 72 |
| B5 × 3: conformer matching without hydrogens | Does the hydrogen-inclusive matching add noise to training? | 25.5 / 36 |

- **Scheduling:** jobs are chained so that at most 4 GPUs are used at once and no job exceeds the 4-day limit. At the high estimates, everything finishes in about 85 hours.
- **Deferred:** lower-priority ablations (sampling steps, ODE vs SDE sampling, noise schedule, model size, a 250-epoch retrain, xTB energies) are listed in the plan (§3.4) to run if time remains.

Full plan: `notes/ablation_plan_qm9.md`. Cluster scripts: `slurm/`. Analysis tools: `tools/`.

---

## 4. What has been achieved so far

- **Sharper research framing:**
  - One of the three proposed gaps (train/test mismatch) was shown to be inaccurate as stated and was reframed as an accuracy floor imposed by RDKit.
  - The other two are now backed by specific citations.
  - The closest competitor (GO-Flow) has been identified, along with how FlexiTors can stand apart from it.
- **Verified tooling:** the evaluation metrics are confirmed to match the paper, and bugs in both the original code and our analysis code were fixed before they could affect results.
- **A defensible experimental design:** experiments isolate one variable each, primary metrics are fixed in advance, statistics are paired with confidence intervals, and the plan fits the available compute.
- **Ready to run:** the environment setup, data download, training, inference and analysis scripts are written and syntax-checked.

## 5. Next steps

1. **Get cluster access:** connect through the IIIT VPN and set up SSH key login.
2. **Set up gnode118:** install the environment on `/scratch` and download QM9 (0.8 GB).
3. **Run the RDKit floor analysis and the pretrained-model experiments:** first real results expected about 15 hours after access.
4. **Train the baseline and ablation models:** about 3–4 days.
5. **Produce a visual report:** what was done, why, and what it showed.
6. **Move on:** start the FlexiTors prototype, then run the same study on GEOM-DRUGS.

---

## Appendix: Project files

| Path | Contents |
|---|---|
| `papers/core/` | TorDiff paper and in-depth analysis |
| `papers/related/` | 26 related papers with summary notes |
| `papers/INDEX.md`, `papers/gap_synthesis.md` | Literature index; gap analysis and FlexiTors design |
| `notes/code_walkthrough.md` | Code walkthrough |
| `notes/code_concerns.md` | Code issues found |
| `notes/ablation_plan_qm9.md` | Full ablation plan and schedule |
| `review/metric_verification.md`, `review/tests/` | Metric verification and 35 tests |
| `review/cross_validation.md` | Experimental design review |
| `torsional-diffusion/` (branch `flexitors-ablation-hooks`) | Code with fixes and experiment hooks |
| `patches/0001-ablation-hooks.patch` | The same fixes and hooks, as a patch |
| `slurm/`, `tools/` | Cluster job scripts and analysis tools |

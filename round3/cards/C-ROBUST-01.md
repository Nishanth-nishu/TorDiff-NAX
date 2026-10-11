# C-ROBUST-01: Error-matched, factorised L augmentation (RDKit error transplanted ring-wise and acyclic-wise)

## 1. Idea (scout)
Replace isotropic jitter (S2) with training L that carries RDKit's *measured* error, one component at a time. For each
training sample, draw L from four paired versions of the same conformer: true L; RDKit-matched L (CTRL data); true
rings + RDKit-matched acyclic geometry; RDKit-matched rings + true acyclic geometry. The two hybrids are built offline
from the existing paired cache with the A5 hybrid builder. No conditioning input. The test is whether one model keeps
S3's two endpoints *and* exploits partly correct L (good rings only, or good bonds/angles only), which is what any
round-3 L source (Q1 refiner, Q3 ring component, MMFF/xTB relaxation) will deliver.

## 2. Where it comes from (scout)
- Cascaded diffusion: conditioning augmentation works because it removes the train-test mismatch between a stage
  trained on ground truth and the samples it receives [E-ROBUST-023]. The augmentation type mattered: Gaussian noise
  helped at low resolution, but similar augmentation gave negative results at 128×128 and 256×256, where blur was used
  instead [E-ROBUST-025, E-ROBUST-056]. CDM states no general matching rule; "the augmentation should match the upstream
  error" is our INFERENCE from this result (revised after D-110a). CDM mixed clean and augmented inputs 50/50 in
  training [E-ROBUST-026].
- Input perturbation (DDPM-IP) simulates inference-time errors during training [E-ROBUST-029] and acts as a
  regulariser that smooths the prediction function [E-ROBUST-030].
- For molecules, SliDe argues that isotropic coordinate noise is an inappropriate assumption [E-ROBUST-034] and
  perturbs bond lengths, angles and torsions separately with physically set variances [E-ROBUST-035].
- Domain randomisation: with enough variability in training, the target looks like one more variation
  [E-ROBUST-036]. Practitioners fit the randomisation to the real data [E-ROBUST-053] and avoid overly wide
  distributions that hinder learning [E-ROBUST-052].
- Score models are inaccurate where training data are sparse [E-ROBUST-050]. Larger noise covers more of that region
  but corrupts the data more [E-ROBUST-051].
- ProteinMPNN: 0.02 Å backbone noise improved recovery on AlphaFold models but lowered it on exact PDB structures
  [E-ROBUST-031]. The authors suggest that crystallographic refinement "may impart" target information to exact
  coordinates that predicted structures lack [E-ROBUST-032]. They state this as a hypothesis (revised after D-110b).

## 3. Why it could matter here (scout)
- Isotropic noise half-fixes B1. On the same in-job test sets, B1 scores 0.234 on RDKit L and 0.021 on true L; S2
  (σ = 0.04 Å/axis) scores 0.206 and 0.054 [E-ROBUST-060, E-ROBUST-003]. Noise buys some robustness and costs most of
  the true-L gain.
- Isotropic noise does not look like RDKit's error (revised after D-108). In bond/angle RMSD, 0.04 Å/axis noise is
  larger than RDKit in bonds only. Against the matched RDKit L that CTRL/S3 train on, it is 1.60× the bond error and
  0.84× the heavy-angle error [E-ROBUST-006]. The decisive gap is the ring tail. Endocyclic-dihedral error > 10° occurs
  in 0.35% of ring seeds under noise, against 28.6% (ETKDG) and 30.8% (matched RDKit); among molecules with a ring of
  ≥ 4 atoms it is 0.58% vs 49.6% [E-ROBUST-006]. For RDKit the share is 48-92% for smallest rings of 4 to ≥ 7 atoms;
  3-rings carry no such error by construction [E-ROBUST-007].
- B1 is hurt more by structured acyclic error than by isotropic noise of a different profile, while CTRL is not
  (revised after D-102). GT rings + RDKit-matched acyclic geometry give B1 0.188, against 0.154 for GT + 0.04 Å noise
  (every seed worse). CTRL scores 0.119 on both [E-ROBUST-008]. The noise has 2× the bond error but less heavy acyclic-
  angle error (3.37 vs 4.11°), so this is not a per-unit-error comparison (unpaired, 955 vs 996 molecules)
  [E-ROBUST-008].
- Both models gain about equally from true rings over their matched-RDKit base (λ = 0, same 955 molecules): CTRL
  0.197 → 0.119, B1 0.261 → 0.188. Only B1 gains much from true acyclic geometry (−0.103 vs −0.014 for CTRL)
  [E-ROBUST-010] (revised after D-110c/d). INFERENCE: a model trained on each component wrong on its own should keep
  the ring gain and add B1's acyclic gain. A ring component (Q3) is worth −0.078 Å to CTRL on this like-for-like base
  [E-ROBUST-010].
- Mixing in real RDKit error already works. B1's loss on RDKit L is in flexible molecules (≥ 4 rotatable bonds: B1
  0.394, CTRL 0.239), and S3's 50/50 mix brings it back to 0.253 [E-ROBUST-011]. The open question is partial L, and the
  round-2 panel never evaluates S2 or S3 on the A5 hybrids [E-ROBUST-013].
- It reuses existing parts. The per-sample switch and the paired cache exist [E-ROBUST-014]. The hybrid builder exists
  and runs at about 2 ms per call in a local timing (two hybrids per training conformer) [E-ROBUST-015].

## 4. Assumptions that may not transfer (scout)
- Every source is an analogy (images, pre-training, proteins). None trains a conditional model on L with a systematic,
  ring-heavy error [E-ROBUST-025, E-ROBUST-034, E-ROBUST-031].
- The transplant reproduces ETKDG's error only. A learned L source (Q1) will have a different error (smaller bias,
  different ring failures). Mitigation: MMFF L serves as a held-out "other sampler" test, and an optional small
  isotropic jitter can be added on top. Robustness to an unseen sampler is not guaranteed.
- The hybrid builder is approximate: A5acyc seeds keep a 0.58° residual in acyclic angles [E-ROBUST-019]. Training
  hybrids would inherit that.
- Hybrids keep the torsions of their base geometry (GT torsions for the GT-ring hybrid, matched torsions for the
  RDKit-ring hybrid). The TD target is relative to the selected conformer [E-ROBUST-016], so this is consistent, but
  angle setting moves fragments and may shift torsions slightly. The grounder should check this.
- Four-way mixing gives true L only 25% of samples (S3: 50%). That may cost true-L accuracy (S3 0.033 vs B1 0.021 on
  the same in-job set) [E-ROBUST-004, E-ROBUST-060].
- Part of S2's residual gap may be a train/test asymmetry rather than the wrong error type: S2 only ever saw jittered L,
  and was tested on un-jittered L. CDM's non-truncated variant applies the same corruption at sampling time
  [E-ROBUST-028]. An optional Arm 0b (S2 on RDKit L + 0.04 Å/axis jitter) separates the two, but it needs a new seed
  pickle.
- Leak caveat: all `_cyc_ORACLE` test conditions use each molecule's own GT conformers, so any information the true
  acyclic geometry carries about its own torsions (the ProteinMPNN-type "memory", [E-ROBUST-032]) inflates A5acyc and λ
  results. RDKit and MMFF L (non-oracle) are therefore the primary test conditions.

## 5. Minimal experiment (scout)
- **Arm 0 (gate, inference only).** S3 s0/s1 and S2 s0-s2 on `S1_A5ring_cyc_ORACLE`, `S1_A5acyc_cyc_ORACLE` and
  `S1_lam0.75_cyc_ORACLE` (existing seed pickles, existing sets in `slurm/r2_evalsets.tsv`): 15 runs, about 2-4 GPU-h.
  Gate rule (pre-declared): if S3 scores ≤ 0.125 on A5ring (CTRL 0.119) **and** ≤ 0.165 on A5acyc (B1 0.158), S3
  already handles partial L; stop here and report. Otherwise run Arm 1. The gate uses ORACLE conditions to choose
  which *experiment* to run, never which model to keep. Model comparison stays on the non-oracle conditions below;
  the validity judge should confirm this is acceptable under README rule 5 (V2 note).
- **Arm 1 (train): S3-factorised.** Per sample, uniform over {GT, RDKit-matched, GT ring + RDKit acyclic, RDKit ring +
  GT acyclic}. Hybrids are precomputed offline per paired conformer (CPU, estimated < 1 h on 32 cores at about 2 ms per
  call; conformer count not checked), stored next to `gt_pos_aligned`, and selected in `_select_paired_L`. 100 epochs,
  2 seeds (3 if budget allows).
- **Controls.** S3 (2 seeds, round 2), CTRL_rematch (3), B1 (3), S2 (3); all on the same panel.
- **Test conditions.** Primary (non-oracle): RDKit ETKDG (`S1_etkdg2L`), MMFF (`S1_etkdg2L_mmff`). Secondary (ORACLE):
  GT cycled, A5ring, A5acyc, λ 0.25/0.5/0.75, noise 0.02.
- **Primary metrics and direction.** AMR-R on RDKit L: non-inferior to CTRL_rematch (margin +0.005 Å). AMR-R on A5ring:
  lower than S3 (expected ≤ 0.12). Secondary: A5acyc ≤ 0.16; GT ≤ 0.04; COV-R@0.1. Paired bootstrap on common molecules
  (`tools/paired_compare.py`).
- **Cost.** Arm 0: about 3 GPU-h. Arm 1: 2 × 12-16 GPU-h training plus 2 × 10 conditions × 0.25 GPU-h, so about 30-37
  GPU-h. Total about 33-40 GPU-h.

## 6. Scout's own call (scout)
Worth trying: YES (gated by Arm 0). This is the only candidate whose augmentation matches RDKit's measured error
structure. It reuses existing code, and it directly tests whether the torsion model can use the partial L improvements
that Q1/Q3 sources will produce. Arm 0 costs about 3 GPU-h and can cancel it.

Revision log (scout, 2026-10-08, after verify_V2.md). The call is unchanged: the core mechanism (no ring tail in
isotropic noise; partial L never tested) rests on VERIFIED entries. Superseded wording:
- §3 "At 0.04 Å/axis it gives 1.75× RDKit's bond error and 1.18× its angle error" became 1.60× / 0.84× against the
  matched RDKit L (D-108).
- §3 "RDKit-derived error hurts B1 more than larger random error" became the B1-specific, unpaired, different-profile
  observation (D-102).
- §3 "The two trained models use different parts of L ... the natural way to get both gains" and "0.178 → 0.119" became
  the λ = 0 base comparison plus a labelled INFERENCE (D-110c/d).
- §2 "had to match the upstream error" is now labelled INFERENCE (D-110a). §2 ProteinMPNN "because" is now hedged
  (D-110b).
- §3/§4 B1 0.236/0.020 became the like-for-like in-job 0.234/0.021 [E-ROBUST-060].

## 7. Code grounding (grounder)
- **Arm 0: config only.** The seed pickles and eval lines exist (A5 lines and `S1_lam0.75_cyc_ORACLE` in
  `r2_evalsets.tsv`). Add a round-3 set label and a models TSV for S3 s0/s1 and B6 s0–s2. Those tags do not exist yet
  for these models (checked on gnode118), so the provenance guard (`common.sh:107-122`) is satisfied. 15 packed runs
  ≈ 2.6 GPU-h.
- **Arm 1 touchpoints:**
  - **Hybrids:** post-pass `tools/add_hybrids.py` over the existing `standardized_pickles_paired`, so the RDKit and GT
    halves stay identical to S3's. `hyb_gtring = lgeom.set_internal_subset(mol, Y_al, X)` and
    `hyb_rdring = lgeom.set_internal_subset(mol, X, Y_al)` (the A5 recipes, `make_l_seed_pickles.py:236-237`), plus
    residual and clash checks. Output to a new std dir and `cache_paired_hyb`.
  - **Dataset:** `utils/dataset.py` `featurize_mol` appends both lists in the same iteration as `pos`/`gt_pos`
    (`:306-311`); extend the stale-cache guard `:141-144`.
  - **Transform:** a new default-off `l_mix4` in `TorsionNoiseTransform` (`:38-47`). `_select_paired_L` (`:79-102`)
    makes one `np.random` draw, so resume determinism is kept. Delete the lists after selection, as `:100-102` does.
  - **Flags and slurm:** `--l_mix4` in `utils/parsing.py:30-37`; `construct_loader` `:347-352`; a `paired_hyb` case in
    `featurize_qm9.sbatch:29-39` and `ablation_train_array.sbatch:53-63`.
  - **Tests:** extend `test_round2_model.py` C7.
- **Transfer check (torsions, as §4 asked):**
  - The TD target is relative to the selected L (`dataset.py:74-76`), so every hybrid is self-consistent.
  - `SetAngleDeg` preserves the dihedral through the angle's own atoms. Dihedrals about the same bond through other
    substituents shift by up to the angle change (≈ 3–4°), so `hyb_gtring` carries GT torsions only approximately; log
    the drift per hybrid.
  - `set_internal_subset` swallows failures (`lgeom.py:358-366`), so pre-declare a residual threshold.
- **Memory:** the cache grows ≈ 1.32 → ≈ 2.1 GB (each extra position list ≈ 0.37 GB); check MaxRSS in a smoke run.
- **Verdict: IMPLEMENTABLE WITH CHANGES** (~200 lines incl. tests).
- **Cost:** ≈ 1 CPU-h hybrids + ~1 h featurize; GPU 2.6 (Arm 0) + 23.4 (2 trainings × 11.7) + 3.4 (evaluation) ≈ 29.4
  (the card said 33–40).
Detail, line numbers and cluster checks: `round3/grounding.md` §3 (shared pieces S-1…S-6 in §2.3).
## 8. Predicted effect on our project (analyst)
Data and model: `round3/impact.md` §2–3. Round-2 numbers are PRELIMINARY; S3 s1 and S4 are from BRIEF (22:20).
- **Ceiling.**
  - Best measured on the hybrids (λ0 base, ORACLE): A5ring CTRL 0.119; A5acyc B1 0.158. The full-L floor is
    0.086 / 0.038 (random-GT L).
  - On non-oracle RDKit/MMFF L the ceiling is ≈ CTRL (0.178 / 0.1525). No training-scheme change has moved CTRL
    there by more than the B2/B5 sizes (+0.002/+0.006, worse).
- **Expected effect.**
  - Primary, RDKit L: ≈ S3 (0.182 ± 0.003; S3 s0/s1 0.182/0.182).
  - A 25 % true-L share (S3: 50 %) likely costs on true L: S3 0.033 → 0.04–0.05, like S4's 0.041–0.044.
  - A5ring: S3 is unmeasured; my guess is S3 ≈ 0.13–0.16 and the factorised model ≈ 0.115–0.125.
  - A5acyc: factorised ≈ 0.15–0.16.
  - Relevance: templates (C-GEOM-02) give A5ring-type L. xTB gives reference-like acyclic L with partly wrong rings,
    which sits on the B1 line (impact.md §3c) rather than at A5acyc.
  - Transfer: CDM, SliDe and ProteinMPNN have no structured, ring-heavy error.
- **Worst case.** Worse than S3 at both endpoints with no partial-L benefit, for 33–40 GPU-h.
- **Detectability.** Arm 0 effects (S3 vs CTRL on A5ring, likely ≥ 0.01) are clear with 2 seeds. Arm 1's
  non-inferiority at +0.005 needs ≥ 3 seeds (S3-family seed SD unknown; CTRL 0.001).
- **Decision value.** Arm 0 gives S3's response to partial L, which is needed anyway: my S3 prediction for xTB or
  template L is a wide bracket for lack of it. Arm 1 matters only if an A5ring-type source (templates) is adopted.
- **Worth trying:** Arm 0 **YES** (3 GPU-h, high information). Arm 1 **MAYBE**, gated on Arm 0 and on C-GEOM-02
  showing that template L is worth deploying (≈ 0 mÅ per GPU-h on the primary endpoint).

**Revised after cross-read (2026-10-11).**
1. The sentence above, "xTB gives reference-like acyclic L ... which sits on the B1 line", is superseded.
   - Per impact.md §3g, models trained on true L (B1, and S3's true-L half) use *own-conformer* L. For most targets,
     xTB or template L is other-conformer L, on which B1 is at RDKit level.
   - The four-way mixture never shows other-conformer GT L either.
   - INFERENCE: a fifth source, "GT L of conformer j with the torsion target of conformer i (same molecule)", would
     address this directly. I flag it in impact.md §5 D for the panel; it is not a card.
2. Grounder cost: ≈ 29.4 GPU-h in total (Arm 0: 2.6).

Calls unchanged: Arm 0 **YES**; Arm 1 **MAYBE**.

## 9. Panel (judges)

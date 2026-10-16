# EXP-3 spec: stereochemistry-constrained torsional diffusion (R-CHEM, round 5)

Written 2026-10-15. Evidence IDs `E-CHEM-NNN` are in `round5/ledger/evidence_CHEM.md`; data and scripts in
`round5/ledger/chem_data/`. Evaluation conventions follow `round5/specs/EVAL.md` where not stated here. Paths are
relative to `C:\Users\HP\Desktop\tor_diff`.

## 0. Summary and novelty verdict

**What TD does wrong, measured.** TD's rotatable-bond rule has no bond-order test, so acyclic double bonds are
diffused like single bonds [E-CHEM-001, E-CHEM-020, E-CHEM-021]; the model gets no E/Z input [E-CHEM-022]; and every
double-bond torsion is re-randomised before sampling [E-CHEM-023]. On QM9 (released model, R0):
- **10–12 % of generated conformers of the stereo-double-bond molecules have an E/Z configuration absent from the GT
  ensemble: 12.4 / 10.3 / 10.5 % for sampling seeds R0 / R1 / R2 (65, 54, 55 of 526), mean 11.1 %**; 42–50 % of those
  molecules get ≥ 1 such conformer; true stereo on the seeds (G1) does not help [E-CHEM-061, E-CHEM-055].
  Coverage: the E/Z analysis covers the 38 molecules that RDKit 2022.09.5 perceives as having a stereo double bond,
  out of 102 QM9 test molecules with a TD-rotatable double bond. (Revised after D-008 item 3; previously "12.4 %",
  the R0 seed only.)
- **Double bonds are not planar**: mean deviation 16.6° from planar (GT 1.8°, raw RDKit 1.5°); 30.5 % of generated
  conformers of the 96 molecules with a TD-rotatable double bond that have generated conformers (of 102 in the test
  set) have one twisted > 20° [E-CHEM-063].
- Tetrahedral chirality is never changed (1.000) [E-CHEM-062], as TD claims [E-CHEM-005]; it is out of scope.

**Stereo counts.** QM9 test (996 molecules): 108 TD-rotatable double bonds in 102 molecules (Revised after D-008: was "96–100"; 96 of them are in the evaluated 935-molecule set); potential stereo
double bonds 103 molecules / 112 bonds with RDKit 2026.03.6 (67 molecules specified; all acyclic ones are C=N) but
38 molecules / 42 bonds with the cluster's RDKit 2022.09.5 — the stereo flag for C=N is version-dependent
[E-CHEM-050, E-CHEM-060]. Tetrahedral centres: 745 molecules (74.8 %). GEOM-DRUGS test (1000): 142 molecules (14.2 %)
with a potential stereo double bond (58 acyclic C=C, 96 C=N, 3 N=N), 141 with one inside TD's rotatable set; 299
(29.9 %) with stereocentres [E-CHEM-051].

**Plan.** Classify every TD torsion by chemical type; give acyclic double bonds a constrained operator (freeze,
reflected half-circle kernel, or E/Z conditioning); verify stereo at every reverse step; compare with original TD and
TD + post-hoc filter/flip. QM9 is a weak test bed (≈ 100 molecules, all C=N, partly unspecified); the primary test is
GEOM-DRUGS with the released TD DRUGS checkpoint, plus an E/Z-enriched set from DRUGS validation molecules.
*Revised after Q-P2-007 / Q-P3-004 (2026-10-16):* by the version-independent structural test, QM9 has **37** molecules
with a stereo double bond, only **3** specified in the SMILES; DRUGS has **140**, **126** specified [E-CHEM-066]. The 67
"specified" QM9 molecules under RDKit 2026.03.6 are explicit-H imines, invisible to heavy-atom RMSD. QM9 can therefore
only test consistency with the GT ensemble plus planarity; the specified-target endpoint is a DRUGS endpoint. All
consult decisions are collected in §9.

**Novelty verdict: INCREMENTAL (engineering fix + measurement), not a headline contribution.**
- E/Z conditioning by special graph edges: PRIOR ART — suggested by TD itself [E-CHEM-003] and implemented in LoQI
  [E-CHEM-040]; bond-stereo features in VonMisesNet [E-CHEM-044].
- Excluding double/triple bonds from the torsion set: PRIOR ART (VonMisesNet's canonical rotatable definition)
  [E-CHEM-043].
- Post-hoc stereo correction: PRIOR ART for chirality (GeoMol, ET-Flow) [E-CHEM-046]; the E/Z analogue is obvious.
- Reflected/constrained diffusion kernels: PRIOR ART as a mechanism [E-CHEM-048, E-CHEM-049]; applying a reflected
  half-circle kernel to E/Z torsions inside torsional diffusion: no prior found [E-CHEM-047] → INCREMENTAL.
- The measurement (TD flips E/Z in ~12 % of samples and leaves double bonds 17° non-planar) has no published
  counterpart we could verify; a LoQI claim that TD "often produces incorrect stereoisomers" is UNVERIFIED (PDF blocked)
  [E-CHEM-047]. Worth one paragraph / a benchmark table, and a fix every FlexiTors model should inherit.

## 1. Prior art (2022–2026)

| Source | What it does about stereo | Evidence |
|---|---|---|
| TD (2022) | chirality via RDKit L (correct by construction); E/Z ignored; double bonds rotatable; suggests special edges | E-CHEM-001…005 |
| GeoMol (2021), ET-Flow (2024) | chirality fixed post hoc by oriented-volume check and reflection; no E/Z | E-CHEM-046 |
| VonMisesNet (2023) | RDKit canonical rotatable bonds (no double/triple); bond-stereo one-hot feature; ETKDG-Clean fixes unspecified DB stereo | E-CHEM-043, -044 |
| S23D (2025) | shows chirality handling changes benchmark numbers; (misstates TD as using chirality features) | E-CHEM-045, -022 |
| LoQI (2025/26) | all-atom diffusion; auxiliary E/Z (classes 5/6) and tetrahedral (7/8) edges; E/Z metric by ordered descriptor | E-CHEM-040…042 |
| Reflected diffusion (2023), constrained domains (2023) | reflected SDE + constrained DSM on bounded domains | E-CHEM-048, -049 |
| RDKit | stereo double bond iff both atoms have ≥ 2 heavy neighbours and not in a ring < 8 | E-CHEM-070, -071 |

## 2. What our data says

| Quantity (QM9 test, released model unless stated) | Value | Evidence |
|---|---|---|
| TD-rotatable bonds / of which double / triple | 2,880 / 108 (3.75 %; was "3.7 %", the 106-bond stereo figure; Revised after D-008 item 4) / 123 (4.3 %) | E-CHEM-050, -060 |
| TD rule replication vs TD code | 0 mismatches / 996 | E-CHEM-060 |
| Stereo-DB molecules (RDKit 2022.09.5 / 2026.03.6) | 38 / 103 | E-CHEM-060, -050 |
| Generated conformers with E/Z absent from GT (R0, R1, R2; 38 molecules perceived by RDKit 2022.09.5) | 65, 54, 55 of 526 (12.4 / 10.3 / 10.5 %, mean 11.1 %) | E-CHEM-061 |
| Same, G1 (true stereo seeds) | 31–36 of 488–494 | E-CHEM-061 |
| Same, RDKit only (A0) / random torsions | 147 / 227 of 526 | E-CHEM-061 |
| Chirality preserved | 1.000 (A2 MMFF seeds 0.970) | E-CHEM-062 |
| Double-bond planarity deviation, generated vs GT | 16.6° vs 1.8°; 30.5 % of conformers > 20° | E-CHEM-063 |
| AMR-R, molecules with vs without a rotatable double bond | 0.257 vs 0.166 (confounded) | E-CHEM-063 |
| Filtering E/Z-wrong conformers without replacement (stereo subset) | AMR-R 0.2417 → 0.2472 (worse), AMR-P 0.330 → 0.287 | E-CHEM-064 |
| GT conformers whose best match is an E/Z-wrong sample | 2.5 % | E-CHEM-064 |
| GT ensembles with > 1 E/Z label | 3 of 38 molecules | E-CHEM-061 |
| G1 overall AMR-R effect | +0.0001 [−0.0035, 0.0038] | E-CHEM-055 |

Predicted effect (INFERENCE): on QM9 all-molecule AMR-R the ceiling is small. The 96 rotatable-double-bond
molecules are ≈ 10 % of the set; if a constraint removed even a third of their excess error (0.09 Å), the all-molecule
gain would be ≈ 0.003 Å. Expect a clear gain in E/Z fidelity and planarity, a small AMR-P gain on the stereo subset,
and AMR-R gain only when wrong samples are replaced by useful ones. DRUGS: 14 % of molecules affected; same logic.
Step 0 below measures the ceiling before any GPU time.

## 3. Method

### 3.1 Torsion classifier (`flexitors/geometry/torsion_types.py`)
For each TD-rotatable bond (TD rule, E-CHEM-020; reuse the replicated rule of `chem_data/stereo_census.py`,
validated 0/996 mismatches), assign one class:
`single_sp3`, `single_conjugated`, `single_amide_like` (C(=O|=S|=N)–N, C(=O)–O), `double_stereo_specified`,
`double_stereo_unspecified`, `double_nonstereo` (e.g. C=CH2, C=N–H), `triple`.
Stereo flags use RDKit `FindPotentialStereo` [E-CHEM-070] with the version pinned (cluster RDKit 2022.09.5 and local
2026.03.6 disagree on C=N, E-CHEM-060); store both the RDKit flag and the structural test (bond order 2, acyclic,
both atoms ≥ 2 heavy neighbours) and constrain by the structural test, so results do not depend on RDKit's version.

Target planar value per constrained double bond b with TD reference atoms (c, a, b, d) (as chosen in
`diffusion/sampling.py:135-142` / `utils/torsion.py:110-124`): φ_b = 0 if c and d are cis, π if trans. Source of the
target: SMILES stereo when specified (bond stereo atoms s1, s2 and STEREOCIS/TRANS or E/Z; each of c ≠ s1, d ≠ s2
toggles cis↔trans); otherwise (unspecified) the seed's own 3D state, flagged so these bonds are reported separately.
`double_nonstereo` and unspecified bonds get the planarity constraint only (target set {0, π}).
*Revised after Q-P1-006 / Q-P2-007:* (a) "specified" = the structural bond carries SMILES bond stereo when parsed with
the pinned RDKit 2026.03.6, frozen once as a committed target table (SMILES, atom indices of `MolFromSmiles`+`AddHs`,
φ_b on TD's reference quadruple); (b) unspecified and non-stereo bonds must NOT be pinned to the seed's isomer: RDKit's
random isomer is wrong far more often than TD's own choice (A0 147/526 vs R0 65/526 E/Z-absent-from-GT,
E-CHEM-061). For them, K1 does not freeze and K2-inf applies a **late planarity fold**: free until the first reverse
step with σ ≤ 0.1π, then folded around the nearest of {0, π} at that step (§9.2).

### 3.2 Arms for the constrained operator
Let δ_b = wrap(τ_b − φ_b) ∈ (−π, π].

- **K1 freeze (inference only, no retraining).** Remove constrained double bonds from `edge_mask` before
  `perturb_seeds` (sampling.py:105-108) and sampling; τ_b stays at the seed value. Before sampling, check the seed:
  if |δ_b| > π/2 rotate by π (RDKit embeds specified stereo correctly; unspecified may be random). Same mechanism as
  VonMisesNet's canonical rotatable set [E-CHEM-043]; the TD model still scores the remaining bonds.
- **K2 reflected half-circle kernel.** δ_b lives on I = [−π/2, π/2]. Forward process: reflected Brownian motion on I
  with TD's σ(t) [E-CHEM-013, E-CHEM-048]. Its kernel, by the method of images (interval length π, reflections at
  ±π/2), is TD's wrapped normal p_w plus one reflected image:
  p_t(δ | δ0) = p_w(δ − δ0; σ) + p_w(δ + δ0 − π; σ),
  score s(δ | δ0) = [p_w(u1) s_w(u1) + p_w(u2) s_w(u2)] / [p_w(u1) + p_w(u2)], u1 = δ − δ0, u2 = δ + δ0 − π,
  with p_w, s_w from `diffusion/torus.py` (`p`, `score`). Prior at σ_max = π: uniform on I. Unlike TD's kernel, the
  score depends on δ0, so the training transform must store δ0 per constrained edge.
  Reverse step: δ ← fold_I(δ + g²·eps·ŝ + g·√eps·z), fold_I: wrap to (−π, π], then x > π/2 ↦ π − x,
  x < −π/2 ↦ −π − x; the conformer is rotated by the change in δ (via `modify_conformer`, utils/torsion.py:57-75).
  - **K2-inf** (no retraining): TD's score for b, reflection only ("thresholding" view of Lou & Ermon).
  - **K2-train** (retrain/fine-tune): noise for constrained edges drawn as normal then folded; target = s above;
    loss weight 1/E[s²] tabulated per σ like `torus.score_norm`.
- **K3 E/Z conditioning.** Add LoQI-style auxiliary stereo edges [E-CHEM-040] for constrained double bonds: pairs
  (s1, s2) cis/trans as two new edge types, kept in a separate `data.stereo_edge_index` / `stereo_edge_attr` and
  concatenated in `build_conv_graph` (diffusion/score_model.py:183-190) with `in_edge_features` 4 → 6 (zero-initialised
  new input weights when fine-tuning). They must NOT enter `data.edge_index` (the mask/rotation code reads bonds from
  it, utils/torsion.py:12-13). Soft: no hard guarantee.
- **K3+K2**: both.

### 3.3 Per-step stereo verification (`flexitors/sampling/stereo_guard.py`)
After every reverse step: compute δ_b for all constrained bonds in batch with `get_torsion_angles`
(utils/torsion.py:110-124); count violations |δ_b| > π/2 (E/Z) and |δ_b| > 20° in the last 3 steps (planarity) per
step and per molecule (logged). K2 enforces by folding; K1 cannot violate; O and K3 only log. At the end: RDKit
labels from 3D (DB-only and chirality-only canonical SMILES, as `chem_data/ez_flip_analysis.py`) against the target
(SMILES) and against the GT set; chirality check via signed volumes for completeness.

### 3.4 Post-hoc arms
- **F-filter**: drop conformers whose DB-only label ≠ target; resample until K = 2L valid or 3 extra rounds (count
  extra NFE). Without replacement recall gets worse [E-CHEM-064], so replacement is mandatory.
- **F-flip**: rotate each offending double bond by π and re-planarise to φ_b (snap δ_b → 0) — the E/Z analogue of the
  chirality flip [E-CHEM-046]. No extra NFE.

## 4. Evaluation protocol
- Test sets: (i) QM9 test (TD protocol, 2L, δ = 0.5 Å); (ii) **GEOM-DRUGS test** (1000, δ = 0.75 Å), released
  `drugs_default` checkpoint and data from `drugs.tar.gz` (25.3 GB; needs user approval for the download, C-BENCH
  card); (iii) **E/Z-enriched set**: DRUGS validation molecules (not used for training) with ≥ 1 specified acyclic
  stereo C=C or C=N, ≈ 500 molecules sampled with a fixed seed, plus all 141 DRUGS-test stereo molecules.
- ~~Primary endpoint for Exp 3 (pre-declared): **E/Z fidelity** = fraction of generated conformers whose DB-only label
  equals the target, on molecules with specified stereo; and planarity (fraction of constrained bonds within 20°).
  Co-primary non-inferiority: AMR-R and COV-R on all molecules, margin +0.004 Å / −0.5 points (round-2 convention).~~
  *Revised after Q-P2-007 / Q-P1-006 / Q-P3-004:* the primary endpoints and contrast family are now defined in §9.3.
  The E/Z label is geometric (sign of cos δ_b on TD's reference quadruple against the committed target table), and RDKit
  3D labels are secondary. Non-inferiority is on the affected subset; the all-molecule figure is derived.
- Secondary: AMR-R/AMR-P/COV-R@0.5/@0.1 on the stereo subset; LoQI-comparable E/Z score (ordered descriptor equality,
  E-CHEM-041); per torsion class: torsion error of nearest-GT-matched pairs; violations per reverse step; NFE.
- Report both RDKit versions' stereo subsets (E-CHEM-060) and the 3 QM9 molecules whose GT ensembles mix E/Z.
- Stats: paired molecule bootstrap, Holm over the primary family, 3 sampling seeds (inference arms) / 3 training
  seeds (retrained arms).

## 5. Ablations
| Arm | Train | What it isolates |
|---|---|---|
| O | released / CTRL_rematch | baseline |
| O + F-filter | none | post-hoc rejection with resampling |
| O + F-flip | none | post-hoc correction |
| K1 freeze (correct isomer) | none | hard constraint, no training change |
| **K1-wrong** (freeze to the opposite isomer) | none | negative control: separates "fewer degrees of freedom" from "correct stereo" |
| K1-canonical (freeze all double + triple bonds) | none | removes redundant triple-bond torsions too (VonMisesNet rule) |
| K2-inf | none | reflection with an unconstrained score |
| K2-train | QM9 retrain / DRUGS fine-tune | training-consistent constrained kernel |
| K3 | retrain / fine-tune | soft conditioning (LoQI-style edges) |
| K3 + K2-train | retrain / fine-tune | both |

Step 0 (CPU, before any GPU): on R0 outputs, snap every TD-rotatable double bond of each generated conformer to its
target planar value (oracle target from GT for unspecified bonds, labelled ORACLE; *Revised after Q-P1-006*: the
state of that generated conformer's nearest GT conformer, the column argmin of `eval.pkl` rmsd[:, k], with the
majority GT label as a sensitivity row) and re-score AMR-R/P. This bounds
what any K arm can gain on QM9; if the all-molecule AMR-R ceiling is < 0.002 Å, run QM9 inference arms only and move the
retrained arms to DRUGS.

## 6. Coding instructions (`flexitors/`)
| New file | Contents | Touchpoints |
|---|---|---|
| `flexitors/geometry/torsion_types.py` | `td_rotatable_bonds(mol)` (TD rule; test vs `get_transformation_mask`), `classify_torsion(bond, stereo_info)`, `double_bond_targets(mol, ref_atoms)` → φ_b and source flag | utils/torsion.py:9-39; utils/standardization.py:69-89; `round5/ledger/chem_data/stereo_census.py` (port, not copy) |
| `flexitors/diffusion/reflected_torus.py` | `p_reflected`, `score_reflected(delta, delta0, sigma)`, `fold_half_circle`, `score_norm_reflected(sigma)` table | diffusion/torus.py:37-80 |
| `flexitors/sampling/stereo_guard.py` | per-step δ_b, violation counters, fold/freeze enforcement | utils/torsion.py:110-124; diffusion/sampling.py:184-234 |
| `flexitors/sampling/constrained_sampler.py` | reverse loop = sampling.py:116-246 with (i) mask edits (K1), (ii) per-edge fold (K2), (iii) guard hooks | sampling.py:105-108, :204, :234; generate_confs.py (sample_confs) |
| `flexitors/training/transforms.py` (shared with EXP-1) | `ConstrainedNoiseTransform`: folded noise + δ0 for constrained edges | utils/dataset.py:49-76 |
| `flexitors/models/stereo_edges.py` | aux stereo edges; `build_conv_graph` override with 6 edge features | diffusion/score_model.py:183-190, :76 |
| `flexitors/eval/stereo_eval.py` (named in EXP-1 §5.1) | DB-only / chirality-only labels, E/Z fidelity, planarity, LoQI-style score | `round5/ledger/chem_data/ez_flip_analysis.py`, tools/stereo_check.py |
| `flexitors/postprocess/stereo_fix.py` | F-filter (with resampling budget) and F-flip | evaluate_confs.py:47-54 (non-isomeric filter, E-CHEM-028) |
| `experiments/exp3/` | config, sbatch (inference array; fine-tune job), README | slurm/ablation_inference_array.sbatch pattern |

Every mechanism cites its source in a comment (e.g. `# Lou & Ermon 2023, PDF p. 1` for folding; `# TD PDF p. 22 F.3`
for the half-circle partition; `# LoQI stereo.py:3-4` for aux edges).

Tests (`flexitors/tests/`): (1) TD-rule replication equals `get_transformation_mask` on 50 QM9 + 50 DRUGS molecules;
(2) φ_b correct on hand-made (E)/(Z)-but-2-ene, oximes, with reference-atom swaps; (3) `score_reflected` equals a
finite-difference of `log p_reflected` and integrates to a normalised density on I; at σ → small it equals TD's score
away from the boundary; (4) folding keeps δ in I and is an involution at the boundary; (5) K1 leaves frozen torsions
bit-identical through sampling; (6) guard counts zero violations for K1/K2 on 20 molecules, non-zero for O on the 38
stereo molecules; (7) stereo_eval reproduces ezflip_qm9.out's R0 line (65/526) from the same pickle.

## 7. Compute (measured QM9 basis: 100-epoch training 11.7 GPU-h on one RTX 3090; packed inference 15–25 min; ≈ 0.17 GPU-h evaluation per condition and seed — round4/BRIEF.md, C-TRAIN-01)
| Item | GPU-h |
|---|---|
| Step 0 + all stereo analyses | 0 (CPU ≈ 1–2 h) |
| QM9 inference arms (O, F×2, K1×3, K2-inf) × 3 seeds | 7 × 3 × ≈ 0.5 ≈ 10 |
| QM9 K2-train, K3, K3+K2 retrains (3 seeds each) | 3 × 35 = 105 (or fine-tune 20 epochs from CTRL: 3 × 3 × 2.3 ≈ 21) |
| DRUGS inference arms (released model), ×3 seeds | ≈ 7 × 3 × 2–5 ≈ 40–100 (estimate; measure one run first) |
| DRUGS fine-tunes K2-train/K3 (20 epochs from released) | 2 × 0.3–0.9 GPU-days ≈ 15–45 per seed (round-4 BENCH estimate of 23–63 min/epoch) |
Recommendation: Step 0 → QM9 inference arms (≈ 10 GPU-h) → DRUGS inference arms → fine-tunes only if K1/K2-inf leave a
gap between E/Z fidelity and 100 %.
*Revised after Q-P1-006 / Q-P1-007 / Q-P3-004:* arms are generated only on the affected molecules and spliced (§9.1).
Stage A is QM9 ≈ 1 GPU-h plus DRUGS ≤ 4.5–5 GPU-h (1 seed, 140 + 20 molecules; §9.4). The 40–100 GPU-h DRUGS line
applied to full 1000-molecule runs and is superseded.

## 8. Risks
- Too few molecules on QM9 (38–103 depending on RDKit) and mostly C=N, where GT ensembles can mix isomers (3/38);
  DRUGS is needed for any claim.
- RDKit version changes stereo perception of C=N (E-CHEM-060): pin the version; constrain by structural test.
- K1 removes degrees of freedom the model was trained with; distribution shift for neighbouring torsions (K1-wrong and
  K2-train control this).
- Unspecified stereo: the "correct" isomer is unknown at test time; constraining to the seed's random isomer can hurt
  recall. Report specified and unspecified separately; never use the GT isomer outside ORACLE arms.
- Gains in AMR-R may be within noise; the primary endpoint is fidelity, AMR is non-inferiority.
- Novelty is incremental; the contribution is the measurement plus a fix FlexiTors inherits.

## 9. Decisions after the coding consults (added 2026-10-16; Q-P1-005/006/007, Q-P2-006/007, Q-P3-004/008)

### 9.1 Subset generation and splicing (Q-P1-006 item 1, Q-P3-004 item 3) — ACCEPTED, with conditions
- Every Stage A operator is the identity on molecules without a constrained bond. Generate O and every arm only on the
  affected molecules, and splice the identical baseline conformers in for every other molecule. On those molecules
  the paired difference is exactly 0, and the all-molecule difference equals (n_affected / N) × the subset
  difference.
- Conditions:
  1. per-molecule seeding (plan_P3 §3.2);
  2. **O is regenerated on the subset with the same seeding as the arms.** The released conformers serve only as the
     common filler outside the subset;
  3. a test asserts bit-identity of every arm with O on unaffected molecules, using the K1-canonical scope for that
     arm;
  4. absolute all-molecule levels are quoted from one source only: the released pickle for DRUGS, R_s for QM9.
- Subsets:
  - QM9: 102 molecules with a TD-rotatable double bond; 211 for K1-canonical;
  - DRUGS: 155 with a TD-rotatable double bond; 160 for K1-canonical (E-CHEM-066).

### 9.2 Operator details (Q-P1-006 items 4–5)
- K2-inf seeds: `perturb_seeds` makes every torsion uniform on the circle. Folding a uniform angle gives a uniform
  angle on I, which is the K2 prior. **Confirmed.**
- Fold centre:
  - specified structural bonds: φ_b from the target table;
  - unspecified and `double_nonstereo` bonds: **not** the seed's state (P1's default is rejected), but the late
    planarity fold of §3.1 (E-CHEM-061: RDKit's random isomer loses to TD's choice).
- K1: freezes specified structural bonds only. Unspecified bonds stay free.
- F-filter: if fewer than 2K valid conformers remain after 3 extra rounds, fill to 2K with the least-violating
  invalid conformers (smallest |δ_b| − π/2). The fills are flagged and counted as failures in E/Z fidelity, and the
  number of molecules needing a fill is reported. The reason is E-CHEM-064: output without replacement lowers recall.

### 9.3 Endpoints and contrast family (Q-P1-006 item 2, Q-P2-007)
- **E/Z label.**
  - Primary: geometric, cis iff cos δ_b > 0 on TD's reference quadruple, against the committed target table (§3.1).
  - Secondary: RDKit `AssignStereochemistryFrom3D` labels under both 2022.09.5 and 2026.03.6 (E-CHEM-060, -066).
- **"Specified"** is defined by RDKit 2026.03.6 on the structural set. 2022.09.5 produced the existing numbers and
  is kept as the reproduction subset of E-CHEM-061.
- **Primary endpoints:**
  - DRUGS: E/Z fidelity to target on the 126 specified molecules, and planarity (|δ_b| ≤ 20°) on all structural
    bonds;
  - QM9 (confirmatory): E/Z consistency with the GT ensemble on the 37 structural molecules, and planarity on the 102.
- **Primary family:** {K1 − O, K2-inf − O, F-flip − O, F-filter − O} × {E/Z fidelity, planarity}. Paired molecule
  bootstrap, Holm over the 8 tests, per dataset.
- **Non-inferiority (co-primary):** the same four contrasts, on the affected subset. AMR-R margin +0.004 Å, COV-R@0.5
  margin −0.5 points, hierarchical bootstrap. The all-molecule value is derived by §9.1.
- **Secondary:** K1-wrong − K1 and K1-canonical − K1, plus F-filter's extra NFE.

### 9.4 DRUGS Stage A scope (Q-P1-007, Q-P3-004) — download approved by the user
- **Molecules:** the 140 structural-stereo DRUGS test molecules, plus the remainder of the 160-molecule K1-canonical
  scope for that arm only.
- **Arms:** O, K1, K1-wrong, K1-canonical, K2-inf, with F-flip and F-filter on O's output.
- **Seeds:** 1 sampling seed first. E/Z fidelity is resolved at the molecule level over ≈ 140 × 2K conformers.
  Seeds 2–3 go to O and the leading K arm if the budget remains, because they matter for AMR non-inferiority.
- **Budget:** a hard 4.5–5 GPU-h cap, with a measured 20-molecule timing run first. Cut order: K1-canonical, then
  K1-wrong. If K1-wrong is cut, no claim that K1's effect is "about stereo" may be made until it runs.
- **Generation and targets:** generation runs under the cluster's RDKit 2022.09.5. Targets and "specified" come from
  the 2026.03.6 table, and the structural test decides what is constrained.
- **E/Z-enriched validation set:** moves to Stage B, unless budget remains. When run:
  - ≈ 500 molecules from the validation indices of `data/DRUGS/splits/split0.npy`, the split of the released model
    (`model_parameters.yml`: `split_path: data/DRUGS/splits/split0.npy`, papers/blogs/td_gdrive_listing.txt:59);
  - structural and specified stereo, random selection with seed 0;
  - 32 conformers per molecule, 1 seed, fidelity and planarity only (no AMR, so 2K is not needed);
  - molecules whose GT ensembles mix E/Z are kept and flagged, with a sensitivity row excluding them.

### 9.5 What FlexiTors (Exp 1) inherits (Q-P1-005, Q-P2-006, Q-P3-008)
- **Pre-declared rule.** After Exp 3 Stage A, Exp 1's V2 sampler inherits the inference arm (K1 or K2-inf) with the
  highest E/Z fidelity among those passing subset non-inferiority (§9.3). DRUGS decides, or QM9 if DRUGS is missing.
  If the fidelity CIs overlap, K1 wins: no NFE cost and no retraining. F-arms are not inherited, because their extra
  NFE would confound Exp 1's compute comparisons.
- **No K2 transform in V2 training** unless Exp 3's K2-train is tested first. Both inherited operators are
  inference-only, so V2 training does not wait for Exp 3.
- **Applying the operator in Exp 1.** It is applied identically to every arm, the A1 baseline included, so it cancels
  in Exp 1's contrasts. Exp 1 also reports every primary arm without it.
- **Stage A M1** (τ frozen) measures E/Z and planarity only, with no E/Z guard.
  - Reason: the P_W rows of TD-rotatable double bonds keep those dihedrals fixed to first order (EXP-1 §2.2 after
    D-004).
  - A second guard would add backtracks that confound the Gate-0 ring metric.
  - Pre-declared trip-wire: if M1 changes the E/Z state relative to the seed in > 1 % of conformers of the structural
    subset at any σ_b, Stage B adds the guard (|δ_b| > π/2 → backtrack). The 1 % threshold is a decision rule, not a
    sourced number.
- **One classifier.** `flexitors/geometry/torsion_types.py` (§3.1) applies the structural test and is the single
  source of truth for Exp 1 and Exp 3. RDKit's stereo flag is kept only as a secondary column.

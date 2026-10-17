# EXP-2 spec: eliminating conformer matching during training (R-CHEM, round 5)

Written 2026-10-15. Evidence IDs `E-CHEM-NNN` are in `round5/ledger/evidence_CHEM.md`. Paths relative to
`C:\Users\HP\Desktop\tor_diff`. Evaluation conventions follow `round5/specs/EVAL.md` where not stated here.

## 0. Summary and novelty verdict

**Question.** TD trains p(τ | G, L) on "matched" conformers: RDKit L assigned to GT conformers by a von-Mises cost and
Hungarian assignment, then DE-fitted torsions [E-CHEM-009]. TD says the matched conformers are not energy minima, which
makes the task "less physically interpretable and potentially more difficult" [E-CHEM-006, TD PDF p. 22, F.4]. Can a
matching-free training recipe (no assignment, no DE fit) with a better objective do as well or better?

**Framing after R-JOINT's finding (Revised after D-001 and D-003; superseded text kept struck).**
~~Another conformer's true L leaves only a 0.032 Å torsion-optimised floor, yet trained models score 0.22–0.30 Å with
it [E-CHEM-065 = E-JOINT-010; E-CHEM-054].~~ Corrected: the torsion-optimised floor with the best of the 3 closest
*other* GT conformers' L (an oracle best-of selection, not split-half) is 0.036 Å on ring molecules (n_GT > 3 rows;
the published 0.032 included the target itself for n_GT ≤ 3, bug at `tools/local_structure_analysis.py:302`), vs
0.114 Å with RDKit L on the same rows [E-CHEM-065 rev.]. Trained models score 0.22–0.30 Å on other-half L
[E-CHEM-054], but the comparison with RDKit L (0.187) is not budget-matched (the split view has 2·|half| samples), so
only the direction "other-conformer geometry is far better than what the models make of it" is used here [D-001].
What fails is the torsion model's use of L it was not trained with.
*Revised after the budget-matched re-score (2026-10-17, E-CHEM-054 rev.):* split − ref_half AMR-R is
CR −0.038 Å (significant), B1 −0.017 (n.s.), S3 +0.017 (significant), S4 −0.007 (n.s.). The RDKit-matched standard
model does gain from other-conformer true L; the true-L or mixed-L recipes gain nothing (B1, S4) or lose (S3). This
sharpens the EXP-2 framing: the training pairing decides whether a model can use better L. The design is unchanged,
and the L-robustness panel (§4) already uses ref_half. Conformer matching is a *training-distribution* device (which L is paired with which
τ target) [E-CHEM-008, E-CHEM-009]. So Exp 2 is really about what the (L, τ) pairing teaches, and every variant must be
read on several L sources (RDKit, MMFF, split-half other-conformer L), not only RDKit L.

**Prior from our data.** Matching details barely matter on QM9: random pairing + DE (B2) +0.0021 Å, heavy-atom DE (B5)
+0.0058 Å vs CTRL_rematch [E-CHEM-052]; TD Table 8 on DRUGS likewise (72.5 vs 72.7 COV-R) [E-CHEM-010]. Dropping the L
replacement altogether (true L, B1) costs +0.054 Å on RDKit L [E-CHEM-053]. Expect variants A/B within ±0.005 Å; any
gain must come from the objective (C, D).

**Novelty verdict: INCREMENTAL at best; likely a useful negative/diagnostic result.**
- B (random RDKit L + denoising): PRIOR ART — TD Table 8 "Only D.E. matching" [E-CHEM-010]; our B2 [E-CHEM-052].
- C (set-level distribution loss): an OT ensemble loss is PRIOR ART in GeoMol [E-CHEM-030]; scoring-rule losses for
  diffusion are PRIOR ART [E-CHEM-031]. Using a Sinkhorn/OT ensemble loss to fine-tune a torsional score model through
  its sampler, replacing matching: no prior found → INCREMENTAL.
- D (energy-aware, matching-free): mechanism is PRIOR ART — TD's own energy training [E-CHEM-011, -012], Torsional-GFN
  [E-CHEM-036], Torsional Adjoint Sampling (data-free) [E-CHEM-032], EnFlow energy model [E-CHEM-035]. "Conditional
  energy-minimum targets" (D1) as a direct answer to TD's F.4 caveat: no prior found → INCREMENTAL (small).

## 1. Prior art (2022–2026)
| Source | Matching-free? | Objective | Evidence |
|---|---|---|---|
| TD 2022 §4.1/App. E | no (matching) | DSM on wrapped normal | E-CHEM-007…010, -013 |
| TD 2022 §3.6/§4.5 | yes (Boltzmann generator) | importance-weighted DSM on model samples, MMFF energy | E-CHEM-011, -012, -014 |
| GeoMol 2021 | yes | OT (EMD) loss between generated and GT ensembles | E-CHEM-030 |
| Distributional diffusion 2025 | — | energy score / kernel scoring rule instead of L2 | E-CHEM-031 |
| Torsional-GFN 2025 | yes (MD L) | GFlowNet, energy reward | E-CHEM-036 |
| Adjoint Sampling 2025 | yes (no data) | reciprocal adjoint matching on eSEN energy; torsional variant | E-CHEM-032, -034 |
| iDEM 2024 | yes (no data) | MC energy-score regression | E-CHEM-033 |
| EnFlow 2025 | Cartesian (no L) | flow matching + Energy-Matching energy model, guidance | E-CHEM-035 |
Cartesian / internal-coordinate generators (ET-Flow, MCF, GO-Flow, the EXP-1 joint model) need no matching because they
generate L themselves; Exp 2 matters for any model that conditions on an externally sampled L (TD, EXP-1 stage 1).

## 2. What our data says
| Arm (QM9, 3 v 3 seeds, paired vs CTRL_rematch / CTRL_base) | AMR-R diff (Å) | Evidence |
|---|---|---|
| B2 random pairing + DE (≈ variant B) | +0.0021 [0.0006, 0.0037] | E-CHEM-052 |
| B5 heavy-atom DE | +0.0058 [0.0038, 0.0080] | E-CHEM-052 |
| B1 true L, no matching | +0.0536 [0.0469, 0.0600] | E-CHEM-053 |
| Leak-free ORACLE, other-conformer L: CTRL 0.2195 vs RDKit-L 0.1873; B1 0.3046 (split vs RDKit-L is not budget-matched, D-001: descriptive only; own-vs-split leak contrast stands) | — | E-CHEM-054 |
| Budget-matched re-score, split − ref_half (Revised 2026-10-17) | CR −0.038*, B1 −0.017 n.s., S3 +0.017*, S4 −0.007 n.s. | E-CHEM-054 rev. |
| Other-conformer L geometric floor (no model; oracle best of 3 closest others, n_GT > 3) | ~~0.032~~ 0.036 Å ring molecules vs 0.114 RDKit L on the same rows (Revised after D-003) | E-CHEM-065 |
| S5 (MMFF-matched training) on MMFF L | −0.0041 [−0.0064, −0.0019] | round4/BRIEF.md |
Reading: what transfers is "train on the L you deploy" (S5); assignment details do not move AMR-R.

## 3. Method (variants; all use TD's architecture and DSM base loss unless stated)

Notation: molecule G with GT conformers {C_l}; RDKit embeddings {L̂_j}; τ(C) TD torsions with TD reference atoms
(utils/torsion.py:110-124); J_DSM = E_σ λ(σ) ‖s_θ(τ̃, σ | L) − ∇ log p_σ(τ̃ | τ*)‖² with the wrapped normal
(TD PDF p. 4 Eq. 3–4) [E-CHEM-013, E-CHEM-025].

- **A — original matching** (= CTRL_rematch; exists, 3 seeds). Targets: τ* = DE-fit of L̂_{π(l)} to C_l, π = Hungarian
  on von-Mises cost [E-CHEM-009]; code standardize_confs.py:89-107.
- **B-DE — random pairing + DE** (= B2; exists, 3 seeds). π = identity (`--no_match`, standardize_confs.py:19, :89-95).
- **B-T — random L + dihedral transfer (truly matching-free, new).** For each GT conformer C_l, draw a fresh RDKit
  embedding L̂ (no assignment) and set every TD-rotatable torsion of L̂ to τ(C_l) (same reference atoms; one
  `modify_conformer` call, utils/torsion.py:57-75). No optimisation. Optionally re-draw L̂ every epoch (B-T-fresh)
  from a pool of M = 10 embeddings per molecule, which breaks any fixed L–τ pairing.
- **C — B-T data + set-level distribution correction.** Pretrain as B-T, then fine-tune with
  L_C = J_DSM + β · S_ε({X_k}_{k≤K}, {C_l}_{l≤L}),
  where X_k = Φ(L̂_k, τ_k) are model samples from a K_s-step probability-flow ODE (TD's ODE branch,
  sampling.py:198-199; K_s = 5 like TD's resampler, E-CHEM-026), Φ applies torsions with a torch (autograd) port of
  `modify_conformer`, K = min(2L, 32), and S_ε is the debiased Sinkhorn divergence with uniform weights (GeoMol's OT
  ensemble loss, E-CHEM-030, with entropic smoothing) and cost
  c(X, C) = mean over all heavy-atom pairs (i, j) of (d_ij(X) − d_ij(C))² (alignment-free,
  permutation-safe because training L̂ is embedded from the GT mol and shares atom order, E-CHEM-029).
  Differentiable: s_θ → ODE updates → Φ → distances → S_ε; GT conformers and RDKit L̂ are constants. ε = 0.05 Å²
  (pre-declared), β tuned on validation over {0.1, 1}. Alternative loss family for a sensitivity arm: energy score
  (De Bortoli et al.) on torsion-space wrapped distances [E-CHEM-031].
- **D1 — matching-free, energy-aware targets (new, cheap).** Start from B-T (τ(C_l) on L̂), then minimise
  E_MMFF(τ | L̂) over the TD-rotatable torsions only (L̂ fixed; L-BFGS on τ with energies from RDKit MMFF and
  `modify_conformer`), trust region |Δτ_i| ≤ 60°, accept only if heavy-atom RMSD to C_l does not rise by > 0.1 Å
  (else keep the B-T target). Targets are conditional minima of the force field given L̂, the property TD says its
  matched conformers lack [E-CHEM-006]. Energy enters only through targets; nothing is differentiated through it.
- **D2 — D1 + TD energy-based term.** L_D2 = J_DSM(D1 data) + γ · Σ_k w̃_k J_DSM(τ_k), with τ_k ∼ q_θ(· | L̂) from a
  5-step ODE and w̃_k ∝ exp(−E_MMFF(τ_k | L̂)/kT) / q_θ(τ_k | L̂) normalised per molecule (TD Algorithm 1, PDF p. 7,
  E-CHEM-011/-012; implemented in utils/boltzmann.py:22-56, E-CHEM-026). Weights are constants (no gradient through
  E or q). T is tempered (pre-declared grid {500, 1000} K, chosen on validation AMR-R) because AMR-R weights every
  CREST conformer equally and a 300 K target would drop high-energy GT conformers [round-4 E-TRAIN-027].

## 4. Evaluation protocol
- QM9 TD protocol (2L, δ = 0.5 Å); primary AMR-R mean and COV-R@0.5 (pre-registered), Holm over the primary family,
  paired molecule bootstrap, 3 training seeds, two-level bootstrap (seeds × molecules).
- **L-robustness panel (required by the framing):** each arm is evaluated on (i) RDKit L, (ii) MMFF-relaxed L,
  (iii) split-half other-conformer true L (ORACLE, `tools/make_split_half.py`, E-CHEM-054); report the robustness gap
  AMR-R(split) − AMR-R(ref_half), where ref_half re-scores the RDKit-L run with a random 2·|B| subset of generated
  conformers against the same B targets, 10 draws (budget-matched; Revised after D-001, same re-score as EXP-1 §3.6;
  the unmatched split − ref_rdkitL gap is reported only as descriptive). A recipe "works" only if it is non-inferior on (i) and better on (ii) or (iii).
- Secondary: COV-R@0.1, AMR-P; MMFF/xTB energy of generated conformers relative to the GT minimum (ΔE distribution,
  `flexitors/eval/energy.py` from EXP-1); training-loss curves (TD reports matched training is harder, E-CHEM-006);
  per stratum by number of rotatable bonds; E/Z fidelity and planarity (shared with EXP-3).
- **Training-distribution diagnostics, computed for every arm's training set before training (CPU):**
  target-to-GT heavy-atom RMSD (the matching floor), per-torsion wrapped distance |τ* − τ(C_l)|, conditional MMFF energy
  of targets, and an L–τ dependence proxy (accuracy of a small classifier predicting the target conformer index from L
  alone, which is the identity-leak channel).

## 5. Ablations, and how "better objective" is separated from "different training distribution"
| Arm | Data (L, τ* construction) | Objective | Exists? |
|---|---|---|---|
| A | matched + DE | DSM | yes (CTRL_rematch ×3) |
| B-DE | random + DE | DSM | yes (B2 ×3) |
| B-T | random + transfer | DSM | new |
| B-T-fresh | random (re-drawn) + transfer | DSM | new |
| C | B-T | DSM + β·Sinkhorn | new (fine-tune) |
| C-A | A | DSM + β·Sinkhorn | new (fine-tune of CTRL_rematch) |
| C-ctrl | B-T | DSM only, same extra fine-tune steps and samples | new |
| D1 | random + transfer + conditional MMFF minimisation | DSM | new |
| D2 | D1 | DSM + γ·IS-energy DSM | new (fine-tune) |
| D2-uniform | D1 | DSM + γ·DSM on own samples with w̃ ≡ 1 | new |
| A-Bw | matched, conformers drawn by CREST Boltzmann weight (`--boltzmann_weight`) | DSM | new |

Rules (pre-declared):
1. **Objective effect** = arm − its same-data, same-compute control: C − C-ctrl, C-A − (A + equal-step fine-tune),
   D2 − D2-uniform. Only these differences may be called "better objective".
2. **Data effect** = same objective, different data: B-T − A, D1 − B-T, B-T-fresh − B-T.
3. D2-uniform isolates "training on the model's own samples" (an exposure-bias / self-distillation effect) from the
   energy information; A-Bw isolates "low-energy reweighting of GT conformers" with the original objective
   [E-CHEM-027].
4. If C helps on both B-T and A data (C − C-ctrl and C-A − control both < 0), the gain is attributed to the objective;
   if only on B-T, it is a repair of B-T's data, not a better objective.
5. Diagnostics of §4 are reported next to each effect; a data effect is claimed only with the matching diagnostic
   moving in the predicted direction.

Predictions (INFERENCE): B-T ≈ B2 (±0.003 Å on RDKit L); B-T-fresh may improve the split-half robustness gap
(removes fixed L–τ pairings, cf. E-CHEM-065) at ±0.003 on RDKit L; D1 within ±0.004; C −0.000 to −0.008 on AMR-R with a
precision cost possible; D2 at 300 K worse on AMR-R, at tempered T ±0.005.

## 6. Coding instructions (`flexitors/`)
| New file | Contents | Touchpoints |
|---|---|---|
| `flexitors/data/targets.py` | `transfer_torsions(gt_mol, rdkit_mol, rot_bonds)` (B-T); `condmin_targets(...)` (D1: τ-only L-BFGS on MMFF, trust region, acceptance rule, failure counters) | utils/standardization.py:29 (`optimize_rotatable_bonds` signature), :69-89; standardize_confs.py:53-130 (pipeline, `--no_match` :19, :89-95, MMFF hook :83-87) |
| `experiments/exp2/build_targets.py` | writes standardized pickles in TD format for B-T, B-T-fresh (pool of M embeddings), D1 | standardize_confs.py output format :117-154; `slurm/standardize_qm9.sbatch` |
| `flexitors/geometry/torch_torsion.py` | differentiable `apply_torsions(pos, edge_index, mask_rotate, dtau)` (Rodrigues, batched) | port of utils/torsion.py:57-100; test against it |
| `flexitors/training/set_loss.py` | debiased Sinkhorn divergence (log-domain, ε fixed), distance-matrix cost, uniform weights | GeoMol §2.5 [E-CHEM-030] |
| `flexitors/training/finetune_set.py` | C / C-A / C-ctrl fine-tune loop: DSM batch + K_s-step ODE rollout with autograd | utils/training.py:8-49 (DSM, E-CHEM-025); diffusion/sampling.py:184-204 (ODE branch) |
| `flexitors/training/energy_is.py` | D2 / D2-uniform: wraps `BoltzmannResampler` with tempered T and w̃ ≡ 1 option | utils/boltzmann.py:22-56; train.py:115-131, :169-184; parsing.py:59-64 [E-CHEM-026] |
| `flexitors/eval/train_diagnostics.py` | §4 diagnostics on any standardized-pickle directory | tools/build_paired_pickles.py (reader) |
| `experiments/exp2/` | configs per arm, sbatch (train array; fine-tune array; eval on RDKit/MMFF/split-half sets) | slurm/ablation_train_array.sbatch, r4_leakfree_array.sbatch |

Tests: `apply_torsions` equals `modify_conformer` to 1e-5 Å and its gradient matches finite differences; B-T on a GT
conformer's own L reproduces the GT conformer (RMSD < 1e-4 Å); D1 never increases MMFF energy and respects the trust
region; Sinkhorn divergence is ≥ 0, zero for identical sets, symmetric; a 2-molecule overfit test shows S_ε decreasing;
D2-uniform with γ = 0 reproduces the DSM loss exactly; each pickle builder logs failures and never drops silently.

## 7. Compute (QM9: 100-epoch training 11.7 GPU-h on one RTX 3090; evaluation ≈ 0.17 GPU-h per condition and seed)
| Item | GPU-h | CPU |
|---|---|---|
| Target building: B-T, B-T-fresh (10 embeddings/molecule), D1 | 0 | ≈ 0.56 M conformers; B-T ≈ 1–2 h, embeddings ≈ 4 h on 32 cores, D1 ≈ 50 ms each ≈ 8 CPU-h (estimate; time 1,000 first) |
| B-T, B-T-fresh, D1 training, 3 seeds each | 3 × 35 = 105 | — |
| C, C-A, C-ctrl fine-tunes (20 epochs, rollout ≈ 3–5× DSM step cost: estimate) | 3 × 3 × ≈ 7–12 ≈ 63–108 | — |
| D2, D2-uniform fine-tunes (TD: 250 epochs = 6–9 days for 10 k DRUGS molecules [E-CHEM-014]; QM9 subset 20 k molecules × 20 epochs) | 2 × 3 × ≈ 10–25 ≈ 60–150 (measure) | MMFF energies in resampler |
| A-Bw training, 3 seeds | 35 | — |
| Evaluation: 11 arms × 3 seeds × 3 L sources × 0.17 | ≈ 17 | — |
Total ≈ 280–420 GPU-h (≈ 3–4.5 days of the 4-GPU node). Staged plan: (1) CPU diagnostics for all data arms; (2) B-T and
D1 at 1 seed (23 GPU-h) — continue only if either moves RDKit-L AMR-R or the robustness gap by > 2 seed-SD;
(3) C vs C-ctrl at 1 seed; (4) full seeds for survivors.

## 8. Risks
- Small effects (B2/B5 within 0.006 Å) need 3 seeds; most arms will be null — pre-declare that a null is a result.
- C's rollout gradient may be unstable or mode-seeking (OT with small ε); the C-ctrl and C-A arms detect a gain that is
  only extra compute.
- D2 importance weights degenerate (low ESS) on flexible molecules; TD needed annealing and days of GPU [E-CHEM-014];
  data-free samplers report heavy energy-call costs [E-CHEM-034].
- MMFF minima ≠ GFN2-xTB minima (GEOM reference); D1 could move targets away from GT; the acceptance rule bounds this.
- Exp 2 conclusions transfer to EXP-1 stage 1 (any model conditioned on sampled L), not to a fully joint generator.

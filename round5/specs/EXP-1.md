# EXP-1: Joint local-geometry + torsional diffusion (FlexiTors)

Author: R-JOINT (round 5), 2026-10-15. Evidence IDs [E-JOINT-NNN] refer to `round5/ledger/evidence_JOINT.md`.
Round-4 IDs (E-ARCH, E-PRIOR) are quoted only where they were copied into the JOINT ledger and re-checked.
ORACLE = uses test-set true geometry; analysis only, never model selection (`orchestration/README.md` rule 5).
INFERENCE = my design reasoning, not something a source shows.

**Revision 3 (2026-10-17), after the budget-matched leak-free re-score** (`round5/IMPLEMENTATION.md` milestone 1
§5–6; `cluster_sync/round5/rescore/`; E-JOINT-011). The round-4 claim "other-conformer true L is worse than RDKit L
for every model" does **not** survive budget matching. What the result does and does not support, and the design
consequences, are in **§11**; the affected paragraphs carry "Revised after re-score".

**Revision 2 (2026-10-16), after the coding consult (Q-P1-001…004, Q-P1-011, Q-P2-001…005, Q-P3-001…003,
Q-P3-009).** The binding decisions are collected in **§10**, which supersedes conflicting text above. Each
affected paragraph carries a pointer "Revised after Q-…". Real flaws the planners found, now fixed:
- the M0 kill criterion was vacuous (Q-P2-004, Q-P3-003);
- the γ′(t) target diverged at the endpoints (Q-P1-001, Q-P3-002);
- H-terminated W rows let the geometry head do heavy-atom torsion work (Q-P2-001);
- the training cache did not need rebuilding (Q-P1-002, Q-P3-001);
- the A5 − A1 contrast was not compute-matched (Q-P2-005).

**Revision 1 (2026-10-15), after verifier V5's disputes D-001 … D-008 (`round5/ledger/disputes.md`).** The
corrections below are applied in place, and superseded text is kept struck through or quoted so the history stays
readable:
- D-001: the leak-free "worse than RDKit L" comparison is not budget-matched, and a re-score is now pre-declared.
- D-002: the ring error is a bulk error on rings of four or more atoms, not a tail.
- D-003: the 0.032 Å floor was wrong (code bug; corrected value 0.036 Å).
- D-004: P_L does not preserve dihedrals; it is replaced by the ker-W projector, and a chirality guard is
  pre-declared.
- D-005: coupling is now a hypothesis, tested by the new arm A6b.
- D-006: there is one M1 cost.
- D-008: PuckerFlow "proposed", not "announced".

## 0. Summary

TD generates only torsions τ on T^m. It takes the local structure L (bond lengths, bond angles, ring puckers) from an
RDKit seed and never changes it [E-JOINT-024, E-JOINT-026, E-JOINT-040]. Our data put most of the remaining error in
ring geometry [E-JOINT-003, E-JOINT-004]. In the leak-free split-half test [E-JOINT-001], trained models given
another conformer's true L scored worse (CR 0.2195 Å) than the descriptive full-set RDKit-L reference (0.1873 Å).
*(Revised after D-001: that comparison is not budget-matched — the split view has half the samples — so the size and
even the sign of the gap are pending the re-score in §3.6. Only the own-vs-split leak contrast is budget-matched.)*
*(Revised after re-score: at matched budget, another conformer's true L **helps** the RDKit-trained CR model
(AMR-R −0.038 Å overall; −0.015 Å [−0.027, −0.003] on distinct targets). It hurts the GT/mixed-trained B1, S3 and S4
on distinct targets (+0.047 to +0.085 Å) [E-JOINT-011]. See §11.)*

*(Revised after D-003; superseded text: "another conformer's L leaves only a 0.032 Å floor".)* With torsions
optimised directly against the target, L from the target's closest other GT conformers (an oracle best-of-3) leaves a
0.036 Å floor for ring molecules, against 0.114 Å for RDKit L on the same rows [E-JOINT-010, revised]. The 0.032 Å
figure was biased low by a code bug [E-JOINT-031]. So the geometry is good enough. What fails is a torsion model
being handed L from a distribution it was not trained on. This is the same brittleness that shows up on MMFF L
[E-JOINT-005], and the only learned fix that has worked is training on the deployed L source [E-JOINT-006].
**This explains the leak-free result without any appeal to coupling. The leak-free test therefore does not show that
coupling is necessary, and coupling is treated as a hypothesis (§4, arm A6b).**

**The design rule is therefore: the torsion factor must always see L drawn from the same process that produces L at
test time (established). Whether that process should also depend on τ is the hypothesis Exp 1 tests (not
established).** Exp 1 builds this in three steps:

- **V1** is TD with fixed RDKit L. This is the baseline, and it is already run.
- **V2** adds a learned correction to L that is conditioned on the current τ and generated in the same network,
  followed by DAgger fine-tuning of the torsion head.
- **V3** is a joint L + τ diffusion with independent clocks on a product space.

The primary geometry representation is a **Cartesian velocity projected onto ker W ∩ (rigid motions)^⊥**. W stacks
one dihedral gradient per rotatable bond; rigid motions are the 6 translations and rotations. *(Revised after D-004:
the first version projected out the torsion tangents ∂x/∂τ, and that does not preserve dihedrals; see E-JOINT-032.)*
With this representation:
- ring closure needs no special handling;
- a geometry step leaves the rotatable dihedrals unchanged **to first order only**;
- chirality **can** flip and is guarded (§2.8).

A Cremer–Pople variant is a secondary ablation.

**The decisive control is arm A6b (added after D-005).** It uses the same trained geometry factor as an *independent*
L sampler, with τ frozen at the seed. A torsion head is then **re-trained on that sampler's own L** with the S5 /
DAgger recipe, so the train/test L mismatch is removed from the decoupled arm as well. If V2/V3 do not beat A6b, the
contribution is "learned L with sampler-matched training", not "joint L–τ". The original A6 (frozen RDKit-trained
torsion model) stays as a secondary arm, because it confounds coupling with L mismatch.

## 1. Prior-art check (adversarial)

The searches were 18 arXiv abstract queries over 2023-01 to 2026-10 plus two web searches [E-JOINT-079], together
with every relevant paper already in `papers/`. Each claim below is paired with what kills it or what survives.

| Candidate claim | Closest prior art (what it does) | Verdict |
|---|---|---|
| "Relax TD's rigid L / diffuse over full intrinsic coordinates" | TD's own conclusion proposes exactly this [E-JOINT-047]; Corso's thesis proposes pucker hyperspheres and a soft-constraint full-dimensional diffusion [E-JOINT-078] | **PRIOR ART (idea)** |
| "Joint bond-angle + torsion diffusion" | RINGER: wrapped-normal joint diffusion of bond angles and torsions in redundant internal coordinates, SLSQP reconstruction, macrocycles only [E-JOINT-060]. FoldingDiff: 3 bond angles + 3 dihedrals per residue, NeRF, proteins [E-JOINT-062] | **PRIOR ART** |
| "Full internal coordinates (bonds × angles × torsions) for small-molecule conformers" | GO-Flow: one flow per manifold, internal → Cartesian Jacobian, GD-P protocol only, no ring-closure treatment [E-JOINT-063, -064, -065] | **PRIOR ART**, but with a hole (rings) |
| "Ring-pucker factor next to torsions" | PuckerFlow: CP flow, monocyclic, unsubstituted, a two-stage demo with TD; a joint CP × torsion flow is **proposed as a feasible extension** (revised after D-008; was "announced"), and their code builds on the TD codebase [E-JOINT-066..069]. TMCgen: S² angle factor + torsions, metal complexes [E-JOINT-070] | **CONTESTED**: no paper yet for drug-like polycyclic molecules on GEOM [E-JOINT-079]; an active group has proposed it |
| "Learned correction of the RDKit conformer" | CoarsenConf: learns the RDKit → GT Cartesian difference [E-JOINT-075]. FM-refiner: refines any upstream sample, but has no TD + refiner row [E-JOINT-077]. EBD: corrects RDKit fragment centroids [E-JOINT-091] | **PRIOR ART**; none restricts the correction to the L-subspace or keeps a torsion factor |
| "Multi-scale / hierarchical conformer generation" | EBD (fragment → atom, NeurIPS 2024) [E-JOINT-090]; MSGEN (scaffold-first, abstract) [E-JOINT-095]; HierDiff and OctMole (de novo, coarse-to-fine) [E-JOINT-093, -094] | **PRIOR ART: do not use this framing** |
| "Torsion model conditioned on / robust to L" | Torsional-GFN follows mode shifts for unseen L (small MD set) [E-JOINT-074] | **PRIOR ART** for conditioning. Our *brittleness* finding is the counter-evidence |
| Dihedral-preserving Cartesian geometry step: velocity projected onto ker W ∩ rigid^⊥ (revised after D-004; the tangent-space version was wrong [E-JOINT-032]) | GO-Flow goes the other way [E-JOINT-063]; DiffDock's factors are disjoint by construction [E-JOINT-072]; nothing found [E-JOINT-079] | **NOVEL but minor (method detail)**; preserves dihedrals to first order only |
| Coupling is *necessary* | TD F.1 uses other-conformer L as a matching floor only [E-JOINT-049]. Our leak-free result is explained by L mismatch [E-JOINT-010 rev.] | ~~NOVEL~~ → **HYPOTHESIS** (revised after D-001/D-003/D-005); it becomes a finding only if A5/A8 beat A6b |
| Ring geometry dominates TD's QM9 error with the model in the loop | Corso: "works fine for small rings" [E-JOINT-078]; HierDiff: rings are what atom models get wrong (de novo) [E-JOINT-093] | **NOVEL as a measurement**, but leak-corrected numbers are needed first (§3.6) |

**What is and is not new, stated exactly.**
- Not new: the idea of learning L, joint angle + torsion diffusion, internal-coordinate flows for small molecules,
  ring-pucker flows, learned RDKit corrections, and the "multi-scale" framing.
- New, if it works (revised after D-004/D-005):
  1. A torsion-diffusion model whose geometry step is projected onto ker W ∩ rigid^⊥. This preserves the rotatable
     dihedrals **to first order** and keeps rings of any topology (fused, bridged, spiro) closed without a closure
     solver. Superseded wording: "exact separation from torsions" — withdrawn [E-JOINT-032].
  2. *Conditional on A5/A8 beating A6b:* evidence that coupling helps beyond sampler-matched training. The leak-free
     test alone does **not** provide this [E-JOINT-001 rev., E-JOINT-010 rev.].
  3. The first TD-protocol GEOM benchmark of a learned-L torsion model, with ring-pucker, stereo and xTB-energy
     metrics.
- Point 1 is a minor method contribution. Point 3 is unconditional. Point 2 is a hypothesis with a pre-declared
  test. Our absolute numbers (best real use
  0.150 Å vs the Cartesian frontier 0.068–0.073 Å [E-JOINT-009]) rule out a "beats SOTA" framing. QM9 at δ 0.5 Å is
  arguably saturated [E-JOINT-155], so GEOM-DRUGS is needed for a main venue.

## 2. Method: three variants

Notation: conformer x ∈ R^{3n}; rotatable torsions τ ∈ T^m (TD's definition [E-JOINT-025]); local structure L = all
other internal degrees of freedom. x_R is the matched RDKit conformer and x_G its own Kabsch-aligned GT conformer,
both from the paired cache [E-JOINT-023].

### 2.1 V1: TD with fixed RDKit L (frozen baseline)

TD unchanged. The forward kernel on T^m is the wrapped normal, sampled as an unwrapped normal mod 2π [E-JOINT-041].
Training draws σ log-uniform in [0.01π, π], uses Δτ ~ N(0, σ²) per bond, and minimises the DSM loss divided by
E‖∇log p‖² (TD Eq. 3–4, PDF p. 4; code [E-JOINT-022]). Sampling uses a geometric σ schedule with
g = σ√(2 ln(σ_max/σ_min)) and the reverse-SDE step `g²·eps·score + g·√eps·z` (TD Alg. 3, PDF p. 17; code
[E-JOINT-024]). L stays fixed throughout [E-JOINT-026].

Existing runs:
- CTRL_rematch (3 seeds, 100 epochs): RDKit L 0.1786 Å; MMFF seeds 0.1540 Å.
- S5 (MMFF-matched training): 0.1500 Å [E-JOINT-006].

### 2.2 V2: TD + τ-conditioned local-geometry correction (coupled)

**Why coupled rather than independent (revised after D-005).** An independent L generator feeding a *frozen,
RDKit-trained* torsion model reproduces the leak-free failure, because the torsion model sees L from a distribution
it was not trained on [E-JOINT-005, -010 rev.]. That failure is fixed by sampler-matched training, which does not
require coupling. V2 puts both factors in one network so that:
- (a) the geometry head sees the current torsions when it moves L. **This is the coupling hypothesis, and A6b tests
  it**;
- (b) the torsion head is trained, and in stage 2 fine-tuned, on L produced by that geometry head
  [E-JOINT-006, -137]. A6b gets (b) as well, so A5 − A6b isolates (a).

**Dihedral-preserving projector (revised after D-004).** *Superseded version:* P_L = I − Bᵀ(BBᵀ)⁺B with
B = [J_τ; J_rig] (torsion tangents ∂x/∂τ_i from TD Eq. 9 plus rigid motions), with the claim that it "changes no
rotatable torsion to first order". **That claim is false.** P_L removes the tangents ∂x/∂τ_i, but the first-order
change of dihedral i is ∇_x τ_i · δx, and the gradient rows ∇τ_i (Wilson-B rows) are dual to the tangents, not
parallel to them. V5 measured a drift of 0.19–0.25 rad/Å with P_L, the same as with no projection [E-JOINT-032].

*Revised version.* At a conformer x, define:
- W(x) ∈ R^{m×3n}: the gradient of **one** dihedral per rotatable bond, using TD's quadruple rule
  (n0[0], b, c, n1[0]) from `torsional-diffusion/utils/standardization.py:69-90`, by analytic dihedral gradients
  (Wilson B-matrix rows);
- J_rig(x) ∈ R^{6×3n}: three translations plus three rotations x_k × ê (TD Eq. 10, PDF p. 7 [E-JOINT-044]);
- C = [W; J_rig] and P_W(x) = I − Cᵀ(CCᵀ)⁺C.

P_W is the orthogonal projector onto ker W ∩ rigid^⊥, of dimension 3n − 6 − m when C has full rank. V5 checked that
this dimension equals that of P_L, with drift 0.000 rad/Å [E-JOINT-032]. INFERENCE: a direct construction; no source
does this.

A step δx = P_W v therefore:
1. changes the m chosen rotatable dihedrals by zero **to first order**; the second-order drift is O(‖δx‖²) and is
   measured (§5.3);
2. adds no rigid motion;
3. keeps every ring closed, because Cartesian coordinates carry no closure constraint.

Other dihedrals about the same bond (other H or heavy substituent quadruples) differ from the fixed one by angle or
improper terms that belong to L, so they may change; this is intended. Ring closure is a problem only in
internal-coordinate parameterisations: RINGER needs SLSQP [E-JOINT-060] and GO-Flow leaves it open [E-JOINT-064].
Below, read "P_L" as P_W everywhere. The symbol is kept so the earlier text stays readable.

**Training (one sample).** *(Revised after Q-P1-001 / Q-P2-003 / Q-P3-001 / Q-P3-002: endpoint (x₁) prediction, the
pure-L target x_G*, the pair_ok mask and the chirality resampling are specified in §10.1, which supersedes steps 4–6
below.)*
1. Pick conformer k. Set x_R = `data.pos[k]` and x_G = `data.gt_pos[k]` (`dataset.py:79-99` [E-JOINT-023]).
2. Draw σ ~ logU[0.01π, π] (torsion clock) and t ~ U[0,1] (geometry clock), independently. Independent clocks per
   factor follow DiffDock [E-JOINT-072] and the independent-time precedents UniDiffuser, Diffusion Forcing and
   Multiflow (`papers/INDEX.md`, round-4 SCOUT TRAIN rows).
3. Draw Δτ ~ N(0, σ²)^m and apply the **same** Δτ to both endpoints: x_R' = modify(x_R, Δτ),
   x_G' = modify(x_G, Δτ) (`torsion.py:57-75`). This isolates the L difference from the torsion noise.
4. Build the geometry state with the stochastic interpolant of ET-Flow Eq. 2–5 [E-JOINT-076], anchored at RDKit
   rather than noise:
   x_t = interp(x_R', x_G', t) + γ(t)·P_L(x_R') z, with γ(t) = σ_b√(t(1−t)) and z ~ N(0, I).
   `interp` is `interp_x_torch` [E-JOINT-023], which keeps terminal bond lengths.
5. Targets:
   - torsion score ∇log p_σ(Δτ) (TD, unchanged);
   - geometry velocity u = P_L(x_t)·[x_G' − x_R' + γ'(t) P_L(x_R') z].
6. Loss: L = L_TD + w·‖P_L(x_t) v_θ(x_t, σ, t) − u‖² / (3n − 6 − m), with w ∈ {0.1, 1} pre-declared.

Why anchor at RDKit: EBD's own ablation found the RDKit-atom prior beats the coarse prior [E-JOINT-092], and ET-Flow
shows the prior shifts DRUGS AMR-R by 0.083 Å [E-JOINT-076].

**Choice of σ_b (revised after D-002).** *Superseded:* "a heavy tail … multimodal for a minority of seeds". That
statement pooled 3,078 three-membered-ring seeds, which always sit at 0°. On seeds with a ring of ≥ 4 atoms
(4,181 seeds), the ETKDG ring-dihedral error has mean 18.5°, median 9.79°, and 49.6 % exceed 10° [E-JOINT-004 rev.,
recomputed]. The error is the **bulk** of those seeds. The RDKit → GT velocity is therefore large and potentially
multimodal (pucker choice) for about half of all ≥ 4-ring seeds, and an interpolant with too little noise averages
puckers into distorted rings.

σ_b ∈ {0.05, 0.15, 0.3} Å is swept in arm A11. Training is on the full QM9 training set; **evaluation only** is on
the rigid stratum (289 test molecules) and the ≥ 4-ring population. Cost is in §6. INFERENCE: 0.3 Å is the order of a
ring-flip displacement.

**Sampling.** *(Revised after Q-P2-002 / Q-P3-002: x₁-prediction Euler update and exact torsion restoration, §10.2.)*
1. ETKDG seed at t = 0; τ ~ U[0, 2π)^m (TD).
2. For each step n = N … 1:
   - one network call gives (score_τ, v);
   - torsion SDE step exactly as `sampling.py:204`;
   - geometry Euler step x ← x + Δt·P_L(x) v;
   - constraint layer (§2.6).
3. Schedule variants (arm A10): joint (both every step), L-first, L-last.

The NFE is the same as TD's: one forward pass per step returns both heads. The projector adds one m+6 least-squares
solve per molecule per step.

**Stage 2, DAgger (mandatory).** *(Revised after Q-P1-004 / Q-P2-005: roll-outs start from fresh ETKDG embeddings,
E_D = 20 epochs, compute-matched controls; §10.4.)*
1. Roll the geometry head out on training molecules from ETKDG seeds.
2. Re-run conformer matching of GT torsions against the rolled-out L with `standardize_confs.py` and
   `tools/build_paired_pickles.py`.
3. Fine-tune the torsion head with the geometry head frozen.

Without this stage the torsion head sits in the S3/B1 regime that lost on MMFF L [E-JOINT-005]. Both S5
[E-JOINT-006] and RINGER's TD retraining [E-JOINT-061] are in-domain evidence that matching the training L to the
deployed L source is what works.

### 2.3 V3: joint L + τ diffusion on a product space

Both factors are genuine diffusions with their own forward processes running at the same time. On a product space
each factor's kernel is sampled and its score regressed independently [E-JOINT-072], and flow matching on such
products is simulation-free [E-JOINT-073].

**V3a (primary): Cartesian-subspace L factor.**
- Factor space: T^m × E_L(x_R), where E_L(x_R) = range P_L(x_R) is a **fixed** linear subspace per training sample,
  anchored at the matched seed. Write x = x_R + d with d ∈ E_L(x_R).
- The L forward process is a variance-exploding Gaussian on d with closed form:
  d_s = d_0 + σ_L(s)·P_L(x_R) ζ, with d_0 = P_L(x_R)(x_G − x_R).
  (Note after D-004: in V3a τ is a **chart coordinate** — torsions are applied by `modify` after the displacement —
  so the factorisation is consistent whichever complement (P_L or P_W) spans E_L. The dihedral drift of D-004 only
  matters for V2's in-place geometry steps, which therefore use P_W.)
  The perturbation kernel is exactly Gaussian on the subspace. Fixing the anchor keeps it closed-form; projecting at
  the moving x would not.
- Torsions are applied after the displacement: x(s) = modify(x_R + d_s, τ_s). Because modify acts by rigid side
  rotations, it preserves the bond lengths and angles produced by d (INFERENCE from [E-JOINT-026]). Ring puckers live
  entirely in d, since ring bonds are never rotatable [E-JOINT-025].
- Score heads:
  - torsion pseudoscalar (`0o`), TD's pseudotorque layer;
  - L-score as a per-atom `1o` vector from the last atom layer [E-JOINT-021], projected by P_L.
- Loss: TD's DSM on τ plus the DSM on d, each normalised by its kernel's expected score norm. TD normalises this way
  (`training.py:29-31` [E-JOINT-022]).
- Noise schedule: σ_L(s) geometric in [σ_L,min, σ_L,max] = [0.005, 0.3] Å, in step with σ_τ (same s). σ_L,max is
  calibrated to the measured RDKit→GT error tail [E-JOINT-004], not to 1 Å, so the factor is a **bounded
  correction**. A Gaussian start on all 3n coordinates is the documented failure mode [E-JOINT-076, -110]. INFERENCE:
  the honest statement is that V3a does not generate L from scratch; it diffuses L within a calibrated neighbourhood
  of the RDKit L.
- The only difference from V2: V2 is a deterministic bridge from x_R to x_G with a learned velocity. V3a is a
  stochastic diffusion whose L-marginal at s = 0 is p(L | graph, x_R), so it can represent several L per seed, for
  example two puckers.

**V3b (secondary ablation): Cremer–Pople × internal coordinates.**
- Factors: T^m (rotatable torsions) × R^{2k} (CP amplitude/phase pairs per ring, Euclidean as in PuckerFlow
  [E-JOINT-066]) × R^{n_b + n_a} (bond lengths, bond angles; Gaussian with per-type variances).
- Reconstruction:
  1. rebuild monocycles from CP coordinates plus the **seed's own** bond lengths and angles, since dictionary values
     cost ≈ 0.05 Å [E-JOINT-068];
  2. place exocyclic atoms by NeRF from (r, θ, τ);
  3. fit polycyclic systems by SLSQP as in RINGER [E-JOINT-060].
- Why it is secondary:
  - 465 of 885 QM9 test ring molecules are fused, bridged or spiro [E-JOINT-071], and their CP coordinates are not
    independent;
  - NeRF has FoldingDiff's lever-arm effect [E-JOINT-062];
  - it is the variant most exposed to PuckerFlow's proposed extension [E-JOINT-069] (wording revised after D-008).
- What it buys: an exact likelihood for the L factor with a closed-form Jacobian (relevant to EXP-4) and
  interpretable puckers.
- Run only if V3a passes Gate 1.

### 2.4 Score parameterisation and parity

TD requires SE(3)-invariant, parity-equivariant scores. The energy is parity-invariant, so ∇log p must transform as
the coordinate does under x → −x (TD Prop. 2, Eq. 6, PDF p. 6 [E-JOINT-042]). Applying the same argument to each
coordinate:

| Coordinate | Under x → −x | Score irrep | Head | Source |
|---|---|---|---|---|
| Rotatable torsion τ_i | τ → −τ | `0o` pseudoscalar | TD pseudotorque layer, unchanged (`score_model.py:119-132`) | [E-JOINT-020, -041, -042] |
| Ring CP coordinate (V3b) | q_m cos φ_m etc. flip sign | `0o` pseudoscalar | ring-centred clone of the pseudotorque layer | [E-JOINT-067] |
| Bond length r, bond angle θ (V3b) | unchanged | **`0e` scalar** | bond-/angle-centred conv with `out_irreps=f'{ns}x0e'` (the existing ablation hook, `score_model.py:122`) | INFERENCE from [E-JOINT-042] |
| Cartesian L velocity/score (V2, V3a) | v(−x) = −v(x) | **`1o` vector** | per-atom linear readout of the `1o` channels of the last atom layer (`score_model.py:83-89,141-143`) | [E-JOINT-021] |

Getting `0e` vs `0o` wrong does not raise an error. It forces the score to zero on the symmetric part of the
distribution. `test_parity.py` (§5.3) therefore checks the sign behaviour of every head on x and −x.

### 2.5 Noise schedules (pre-declared)

| Factor | Training draw | Sampling schedule | Range | Source |
|---|---|---|---|---|
| τ (all variants) | σ ~ logU | geometric, `steps` points, g = σ√(2 ln(σmax/σmin)) | [0.01π, π] | TD; [E-JOINT-022, -024] |
| L, V2 bridge | t ~ U[0,1] | uniform Δt = 1/N | γ(t) = σ_b√(t(1−t)), σ_b ∈ {0.05, 0.15, 0.3} Å | ET-Flow Eq. 2–5 [E-JOINT-076] |
| L, V3a | s shared with τ (σ_L = f(σ_τ), log-linear map) | same steps as τ | [0.005, 0.3] Å | calibrated to [E-JOINT-004] |
| L, V3b | per-coordinate σ by type (SliDe-style BAT noise, `papers/INDEX.md` 2024_ni_sliced_denoising) | same steps | bonds 0.005–0.03 Å, angles 0.2–3°, CP 0.005–0.2 | [E-JOINT-004] |

A **shared clock** for V3a means one network input, and the torsion and L noise levels stay in fixed proportion.
**Independent clocks** are the ablation; they let V3 also do conditional sampling, i.e. L given τ and τ given L.

### 2.6 Constraint / reconstruction layer

The layer is applied after every geometry step in V2/V3. It is pure geometry (no force field), so it cannot add
"optimisation" (§4):
1. **Connectivity**: each bonded pair stays within ±15 % of its seed length, and no non-bonded heavy-atom pair comes
   closer than 0.75 × the sum of covalent radii. If either check fails, the geometry step is halved, up to 3 times;
   after that the step is skipped. Backtracks and skips are counted and reported.
2. **Chirality** (§2.8).
3. **Planarity of aromatic rings and sp² centres**: optional. It is off in the primary arm and on in arm A9b. Its
   effect is measured, not assumed.
4. **V3b only**: CP → ring reconstruction, NeRF, SLSQP closure for polycycles.

### 2.7 How ring closure is guaranteed

- **V2/V3a**: by representation. Atoms carry Cartesian coordinates and the geometry factor moves them, so no ring can
  open. Only a bond stretch can break a bond, and §2.6 step 1 catches that. Rotatable torsions never touch rings
  [E-JOINT-025].
- **V3b**: by construction for monocycles via the CP inverse [E-JOINT-066]; for polycycles by SLSQP with a reported
  closure residual (RINGER [E-JOINT-060]). If the residual exceeds 0.01 Å the sample is rejected and counted as a
  failure.

### 2.8 How stereochemistry is preserved

- **Tetrahedral centres.** TD cannot change chirality because its moves are rigid side rotations [E-JOINT-026]. V2/V3
  can.
  - For each RDKit-perceived stereocentre c with neighbours a<b<d in canonical rank order, the signed volume is
    χ_c = (x_a − x_c)·[(x_b − x_c) × (x_d − x_c)].
  - If a geometry step flips sign(χ_c) relative to the seed, the step is backtracked (§2.6).
  - **Guard is mandatory (revised after D-004).** V5 observed chirality flips in 1 of 20 random 0.3 Å-RMS projected
    steps [E-JOINT-032]: projection does not protect stereo. The guard runs after **every** geometry step in
    training roll-outs (DAgger) and at sampling. The V3a forward kernel can also cross χ_c = 0, so any noised
    d_s that flips a centre is rejected and resampled during training, and rejections are counted. Flip counts (before guard) and guard backtracks are reported per arm (§3 row 9).
  - Same guard for ring-fusion stereo (cis/trans ring junctions), via the signed volume at each bridgehead.
  - The ET-Flow post hoc flip [E-JOINT-132] is used only as a final reported repair, and only in a separate arm.
- **E/Z.** V1 already can flip E/Z, because non-ring double bonds are treated as rotatable [E-JOINT-025, -131]. This
  belongs to **Exp 3 (R-CHEM)**. Exp 1 measures it with `tools/stereo_check.py` and adopts whatever Exp 3 decides,
  to avoid two fixes for one problem. Consult `round5/consult/Q-JOINT-001.md` (to be filed by the coding agents).
- **Evaluation.** RDKit `AssignStereochemistryFrom3D` gives the canonical isomeric SMILES, which is compared with the
  input for every generated conformer [E-JOINT-030]. Reflections are screened as in [E-JOINT-133].

## 3. Evaluation protocol

All on the TD/GeoMol QM9 split (1000 test molecules, 2K samples, δ 0.5 Å), with the unmodified
`torsional-diffusion/evaluate_confs.py`. Then GEOM-DRUGS (δ 0.75 Å) for whatever passes.

| # | Metric | Definition / tool | Role |
|---|---|---|---|
| 1 | AMR-R mean, COV-R@0.5 | `evaluate_confs.py` | **primary** (pre-registered, `round4/BRIEF.md`) |
| 2 | COV-R@0.1 on the success intersection; AMR-P, COV-P | same | key secondary; P is limited by reference incompleteness [E-JOINT-133] |
| 3 | Bond-length MAE (Å), bond-angle MAE (°), heavy-torsion MAE (°), recall- and precision-matched | `tools/geometry_metrics.py` [E-JOINT-030] | **co-primary physical endpoint** |
| 4 | Ring-dihedral RMSD and the share > 10°; puckering RMSD from the ring mean plane | our `l_error` definition [E-JOINT-004]; PuckerFlow's metric [E-JOINT-150] | mechanism read-out |
| 5 | Validity: sanitisable, connectivity preserved (bond perception from 3D equals the input graph), no clashes | new `flexitors/eval/validity.py` | screen; failures counted, never dropped |
| 6 | Stereo preservation: % with input chirality and E/Z | `tools/stereo_check.py` [E-JOINT-131] | screen + reported |
| 7 | Energy: GFN2-xTB single point; xTB-relaxed E_relax; RMSD moved by relaxation; post-relaxation connectivity change; Boltzmann-weighted E, μ, Δε, E_min vs reference on a 100-molecule subset | xtb 6.7.1 on gnode118 (≈ 0.14 s per QM9 opt [E-JOINT-008]); TD §4.4 protocol [E-JOINT-153]; screens [E-JOINT-133] | physical endpoint |
| 8 | Inference time | core-seconds per conformer on CPU (TD Table 2 units [E-JOINT-153]) and packed GPU wall-clock; NFE | cost |
| 9 | Constraint cost | share of per-step time spent in P_W and §2.6; backtracks and skips per conformer; induced dihedral drift per step, measured on TD's quadruple per rotatable bond (`standardization.py:69-90`) (revised after D-004) | cost / diagnostic |
| 10 | Rigid stratum AMR-R (n_rot = 0, 289 molecules) | `breakdown.log` "by n_rot_heavy" bin 0 [E-JOINT-007] | model-free L read-out (Gate 0) |
| 11 | **Leak-free oracle as an analysis tool** | §3.6 | mechanism test (ORACLE) |

Statistics follow the round-2 protocol: 3 training seeds, per-molecule seed mean, paired molecule bootstrap (5000),
Holm over the primary family (AMR-R and COV-R@0.5 for each pre-declared contrast), and a hierarchical
seed-then-molecule bootstrap as a sensitivity check. Failures are reported per arm, and the success intersection is
stated.

**Pre-declared endpoint rule (from [E-JOINT-152]).** Physical-fidelity metrics can move without any change in RMSD,
and the reverse can also happen. Primary RMSD and primary physical (row 3) endpoints are both declared now. One will
never be reinterpreted as the other after results are in.

### 3.6 Leak-free oracle (ORACLE, analysis only)

Re-run `round4/leakfree.md` §2 unchanged on the V2/V3 checkpoints: `tools/make_split_half.py`,
`tools/check_split_seeds.py`, and `tools/analyze_leakfree.py`. There are two modes:
- **seeds = other half's GT L**, with the geometry factor on: can the model move another conformer's L toward a
  conformation consistent with the τ it generates?
- **seeds = other half's GT L**, with the geometry factor off: torsion head only.

**Budget-matched reference first (added after D-001; CPU only, existing files, to be run before any Exp-1 GPU
time).** For CR, B1, S3 and S4 (`steps20_seed0` `eval.pkl`/`confs.pkl` on gnode118): re-score `ref_rdkitL` with,
per molecule, a random subset of 2·|B| of its generated conformers against the B targets (and 2·|A| against A),
averaged over 10 draws. This gives `ref_half`. Report split − ref_half with a paired molecule bootstrap. Until this
is run, the published gaps (CR +0.032, B1 +0.048, S3 +0.088, S4 +0.064 Å vs full-budget `ref_rdkitL`) are
descriptive only and are not cited as evidence. Owner: the orchestrator's analysis driver;
`tools/analyze_leakfree.py` gains a `--ref_subsample` option (code change, not made by R-JOINT).

Pre-declared predictions (revised after D-001/D-005; superseded: "no worse than its RDKit-L AMR-R, unlike CR
(+0.032) …"):
- (i) a coupled model's split − ref_half gap is no larger than A6b's split − ref_half gap, i.e. no larger than
  that of a decoupled model with sampler-matched training. *(Revised after re-score: (i) is read on the
  **distinct-target view** split_d − ref_half_d, which is now the primary view of this oracle analysis. This choice was made after seeing the round-4 models' re-score, but before any Exp-1 checkpoint exists, so it is pre-declared for Exp 1 only. The
  all-target view is dominated by near-duplicates: COV-R@0.1 is +19.7 to +25.2 pt for every model [E-JOINT-011]. An
  added pre-declared reference: (i′) the coupled model's split_d − ref_half_d ≤ 0, i.e. no worse than RDKit L at
  matched budget, which CR meets (−0.015 Å) and B1/S3/S4 fail.)*
- (ii) its own − split leak gap is smaller than S4's (0.17 Å).

If (i) fails, the coupling hypothesis is not supported by the oracle analysis, whatever the primary metrics show. Report the `split_d` distinct-target view
alongside, since ≈ 40 % of targets have a near-duplicate in the other half [E-JOINT-002].

## 4. Ablation table and the controls against "winning by more optimisation"

Every arm uses the **same fixed list of ETKDG seeds** (same RNG seeds), the same K = 2 × n_GT samples, the same
evaluator, and 100 epochs unless noted.

| Arm | Description | New? | Seeds | Purpose |
|---|---|---|---|---|
| A0 | V1 CTRL_rematch, RDKit L | have | 3 | baseline |
| A1 | V1 + MMFF seeds; S5 | have | 3 | strongest cheap real-use baseline today (0.150) |
| **A1+** | S5 (A1) fine-tuned E_D = 20 epochs on its own MMFF-matched data (added after Q-P2-005, §10.4) | new | 3 | **compute-matched real-use baseline; replaces A1 in the primary family** |
| A2 | V1 + GFN2-xTB-relaxed seeds, torsion model retrained on xTB-matched seeds | new | 3 | **level-matched geometry baseline** [E-JOINT-134]; if Exp 1 ≤ A2, there is no method contribution |
| A3 | V1 parameter-matched: FlexiTors network with the geometry head present but w = 0 and no geometry steps | new | 3 | controls for capacity and code changes; must equal A0 within noise (golden test) |
| A4 | V2, no DAgger | new | 3 | coupled correction |
| A5 | V2 + DAgger | new | 3 | **main V2 arm** |
| A6 | Decoupled, unmatched: A5's geometry head run alone (τ frozen at seed) to produce L, then the frozen A0 torsion model (trained on RDKit L) on that L | new (no training) | 3 | **secondary** (revised after D-005): confounds coupling with train/test L mismatch. *(Revised after re-score: the phrase "reproduces the S3/B1 regime by construction" is withdrawn. The RDKit-trained CR **gains** from independent other-conformer GT L at matched budget, −0.015 Å on distinct targets [E-JOINT-011], which is the oracle ceiling for A6.)* |
| **A6b** | **Decoupled, sampler-matched** (added after D-005): A5's geometry head run alone (τ frozen at seed) as an independent L sampler on the **training** set; GT torsions re-matched to that L (`standardize_confs.py`); a torsion head re-trained (S5 recipe) or DAgger-fine-tuned from A0 on it; at test the same sampler gives L, then this torsion head | new | 3 | **primary coupling control**: A5 − A6b isolates "geometry head sees τ" from "torsion head trained on its own L" |
| A7 | V3a, shared clock | new | 3 | joint diffusion |
| A8 | V3a + DAgger | new | 3 | **main V3 arm** |
| A9 | V3b (CP), only if A8 passes Gate 1 | new | 1→3 | coordinate ablation |
| A9b | A5 or A8 with the planarity projection on | new | 1 | constraint-layer effect |
| A10 | Schedules: joint / L-first / L-last | new | 1 | sampler ablation |
| A11 | σ_b sweep {0.05, 0.15, 0.3} Å, geometry head only, 30 epochs | new | 1 | Gate 0 |
| A12 | V1 at equal wall-clock (more steps, 2× NFE) | new (inference only) | 3 | equal-compute control |
| A13 | A0 output + post hoc learned refiner (FM-refiner style, trained on A0 samples) | new | 1 | "why not just refine TD's output?" [E-JOINT-077] |
| A14 | A0 + MMFF after sampling; A0 + xTB after sampling | partly have (A3 round 1) | 3 | relaxation trades recall for precision [E-JOINT-135]; reported separately, never folded in |

**Pre-declared primary contrasts (Holm family; revised after D-005, and again after Q-P2-005 — A1 is replaced by the
compute-matched A1+, §10.4):** A5 − A1+, A8 − A1+, **A5 − A6b, A8 − A6b**,
each on AMR-R and COV-R@0.5. A5 − A6 and A8 − A6 move to the secondary family. Superseded: "A5 − A6, A8 − A6" as
primary.
**Secondary:** A5/A8 − A2, A4 − A5 (DAgger effect), A7 − A8, A9 − A8, A12 − A0, A13 − A5.

**Controls that stop the model from winning by more optimisation:**
1. **No force field or QM at inference in any primary arm.** The §2.6 layer is pure geometry. Relaxation arms (A14)
   are separate and labelled.
2. **Level-matched geometry baseline A2.** GEOM-QM9 references are GFN2-xTB structures [E-JOINT-134], so xTB-relaxed
   seeds are the "free" upper bound of an L fix. Exp 1 must beat A2, not only A0.
3. **Equal compute.** V2/V3 use the same NFE as TD (one forward pass, two heads). A12 gives V1 the extra wall-clock
   V2/V3 spend on P_L. NFE and core-seconds are reported for every arm.
4. **Equal capacity and code path.** A3 runs the FlexiTors code with the geometry head disabled, and must reproduce
   A0. A golden test checks bit-for-bit equality on a fixture (§5.3).
5. **Equal samples, equal seeds.** K is fixed, and the ETKDG seed list is fixed across arms. RDKit + cluster at an
   equal sample budget is reported because recall is gameable by budget [E-JOINT-154].
6. **Refinement contrast A13.** It isolates the "joint" claim from "post hoc refinement".
7. **Decoupled contrast A6b** (revised after D-005; was A6). It isolates "coupling" from "learned L with
   sampler-matched training". A6 alone could not, because its torsion head is mismatched to its L.
8. **No selection on the test set.** Checkpoints are picked by validation loss; σ_b and w are picked on the validation
   rigid stratum; the leak-free oracle is analysis only.

## 5. Coding instructions

### 5.1 Package layout (`flexitors/`, per `round5/PLAN.md` "Code structure")

TD stays frozen in `torsional-diffusion/`. `flexitors/` imports from it, and nothing is copied except where noted.
Every mechanism carries a comment `# <paper>, PDF p. N, eq. M` or `# [E-JOINT-NNN]`.

| Module | Contents | Reuses (path:line) |
|---|---|---|
| `flexitors/geometry/jacobian.py` | `dihedral_gradients(pos, quads)` returning W ∈ R^{m×3n}, analytic Wilson-B rows for TD's quadruple per rotatable bond (`standardization.py:69-90`), **used by the V2 projector** (revised after D-004); `torsion_jacobian(...)` returning J_τ (tangents; used only by the EXP-4 chart Jacobian); `rigid_jacobian(pos)` returning 6×3n; batched torch versions | `torsional-diffusion/diffusion/likelihood.py:68-74` (`dx_dtau`), `:59-65` (`inertia_tensor`), `:84-93` (rows and rigid removal) |
| `flexitors/geometry/projector.py` | `dihedral_preserving_projector(pos, ...)` returning P_W = I − Cᵀ(CCᵀ)⁺C, C = [W; J_rig] (dense for n ≤ 64, otherwise implicit via `torch.linalg.lstsq`), batched over molecules (no per-molecule Python loop, D-006); `project(v, ...)`; rank check | `jacobian.py` |
| `flexitors/geometry/internal.py` | bonds, angles, torsions, impropers from (pos, edge_index); stereocentre neighbour ordering; signed volumes | `torsional-diffusion/utils/torsion.py:110-124` (`get_torsion_angles`) |
| `flexitors/geometry/rings.py` | ring perception, ring-system classes (isolated / fused / bridged / spiro); CP forward/inverse (V3b) | RDKit `RingInfo`; method of [E-JOINT-066] |
| `flexitors/geometry/constraints.py` | §2.6: bond window, clash test, chirality sign check, backtracking step; counters | `internal.py` |
| `flexitors/geometry/reconstruct.py` | V3b only: NeRF, Cremer segment rebuild, SLSQP closure | [E-JOINT-060, -062, -066] |
| `flexitors/models/heads.py` | `VectorL1oHead` (readout of `1o` channels), `ScalarL0eHead` (bond/angle-centred conv, `0e`), `PuckerL0oHead` (ring-centred, `0o`) | `torsional-diffusion/diffusion/score_model.py:83-89`, `:119-132`, `:167-181` |
| `flexitors/models/flexitors_score.py` | subclass of `TensorProductScoreModel` taking a second clock (t or σ_L) through the existing λ embedding; returns `edge_pred` and `l_pred` | `score_model.py:48-79` (constructor, `lambda_embed_dim`), `:134-165` (forward), `:192-199` (clock embedding) |
| `flexitors/training/transforms.py` | `PairedJointNoiseTransform`: §2.2 steps 1–5 / §2.3; the same Δτ on both endpoints | `torsional-diffusion/utils/dataset.py:18-34` (`interp_x_torch`), `:71-76` (σ, Δτ), `:79-99` (paired selection); `utils/torsion.py:57-75` |
| `flexitors/training/losses.py` | `td_loss` (verbatim from `training.py:25-31`) plus `projected_l_loss` | `torsional-diffusion/utils/training.py:25-31`; `diffusion/torus.py` (score, score_norm) |
| `flexitors/training/train.py` | thin loop based on `training.py:8-49`; logs both losses | `torsional-diffusion/train.py`, `utils/training.py:8-49` |
| `flexitors/training/dagger.py` | roll out the geometry head on train molecules, write seed pickles, call re-matching, fine-tune the torsion head with the geometry head frozen | `tools/build_paired_pickles.py`, `torsional-diffusion/standardize_confs.py`, `utils/standardization.py:28-44` |
| `flexitors/sampling/schedules.py` | TD σ schedule; t / σ_L schedules; joint / L-first / L-last | `torsional-diffusion/diffusion/sampling.py:125-126` |
| `flexitors/sampling/joint_sampler.py` | reverse loop: torsion step as `sampling.py:190-204`, geometry step, constraints; per-step timers | `sampling.py:116-246`, `:290-298` |
| `flexitors/eval/validity.py` | sanitise; connectivity from 3D (`rdDetermineBonds`) equals the input graph; clashes | — |
| `flexitors/eval/geometry_eval.py`, `stereo_eval.py`, `rings_eval.py` | wrappers | `tools/geometry_metrics.py`, `tools/stereo_check.py`, `tools/lgeom.py` |
| `flexitors/eval/energy.py` | xtb 6.7.1 single point and opt, E_relax, post-relaxation connectivity and RMSD | `torsional-diffusion/utils/xtb.py` |
| `flexitors/eval/leakfree.py` | wrappers only, with no new analysis code | `tools/make_split_half.py`, `tools/check_split_seeds.py`, `tools/analyze_leakfree.py` |
| `flexitors/eval/timing.py` | NFE, core-seconds per conformer, constraint-cost accounting | — |
| `experiments/exp1/` | `config_*.yml`, `sbatch_*.sh`, `README.md`; results in `results/exp1/` | `slurm/` patterns |

### 5.2 Smallest viable first milestone (M0 → M1)

- *(Revised after Q-P2-004 / Q-P3-003 / Q-P1-003: the M0 below is superseded by §10.3. Its kill criterion was
  vacuous: a linear projection of an oracle L-displacement recovers ≈ 100 % by construction, on the rigid stratum
  and, to first order, everywhere.)*
- **M0 (CPU only, ≈ 1–2 days of coding).** `geometry/jacobian.py`, `projector.py`, `constraints.py` and their tests.
  Then one offline experiment with **no learning**: take the round-2 paired cache and, for each QM9 test-set rigid
  molecule's matched RDKit seed, apply the *oracle* step x_R + P_W(x_R)(x_G − x_R), as ORACLE analysis only.
  Measure:
  - (a) the dihedral drift caused (revised after D-004: measured as the change of TD's quadruple dihedral per
    rotatable bond, `standardization.py:69-90`; expected O(step²), reported in rad, with P_L (old projector) shown
    beside it as the negative control that V5 measured at 0.19–0.25 rad/Å);
  - (b) how much of the RDKit → GT RMSD the L-subspace captures (fraction of ‖x_G − x_R‖² inside range P_L);
  - (c) the chirality flips.

  This shows whether the subspace carries the error before any GPU is spent. **Kill criterion:** if the projected
  oracle step recovers < 50 % of the rigid-stratum RMSD gap (ETKDG 0.152 → own-GT), the representation is wrong.
- *(Revised after Q-P1-003 / Q-P2-004 / Q-P3-003 / Q-P3-009: which parameters train, the sampler used for the
  read-out, and the fact that **validation** decides σ_b and GO with test read once afterwards — all in §10.3.)*
- **M1 = Gate 0: ≈ 10.5 GPU-h at TD speed, budget ceiling 16 GPU-h (revised after D-006; superseded figures "1 GPU-h
  each" and "4–6 GPU-h").** Train only the V2 geometry head (torsion loss weight 0, τ noise still applied) on the
  **full** QM9 training set: 30 epochs × 3 σ_b values × 1 seed (A11) = 0.3 × 11.7 × 3 = 10.5 GPU-h. The ceiling
  allows ≤ +50 % for the extra head and per-sample projector, which is **unmeasured**. The numpy path in
  `likelihood.py` loops per molecule in Python, so the projector must be a batched torch implementation.
  **Before the Gate-0 ask, a 1-epoch smoke run (≈ 0.12–0.18 GPU-h) measures the real overhead,** and the ask is
  restated from it. Evaluation only is restricted: the rigid stratum (n_rot = 0, 289 test molecules [E-JOINT-007])
  and the ≥ 4-ring seed population.
  **GO** if rigid AMR-R ≤ 0.10 Å (ETKDG 0.152, MMFF 0.128) **and** the share of ≥ 4-ring seeds with ring-dihedral
  RMSD > 10° is ≤ 20 % (ETKDG 49.6 %, MMFF 35.1 %, both on the ≥ 4-ring population [E-JOINT-004 rev.], as confirmed
  after D-002).
- **M2 = Gate 1 (≈ 15 GPU-h + A6b ≈ 15 GPU-h; revised after D-005).** Full V2 (A4), 100 epochs, 1 seed, plus A6
  (no training) and A6b, 1 seed. **GO** to the 3-seed table if A4 < A1 on AMR-R by ≥ 0.005 Å. The coupling claim
  additionally needs A4 < A6b; if A4 ≈ A6b, continue as a "learned L + sampler-matched training" paper and drop the
  joint framing.

### 5.3 Test plan (`flexitors/tests/`)

- `test_projector.py` (revised after D-004):
  - P_W is idempotent and symmetric;
  - W·P_W ≈ 0 and J_rig·P_W ≈ 0, both to 1e-6. The old test "P_L·J_τᵀ ≈ 0" passes for the wrong projector, so it
    is kept only as a documented negative example;
  - analytic dihedral gradients W match central finite differences;
  - rank = 3n − 6 − m;
  - m = 0 molecules work;
  - finite-difference check of `torsion_jacobian` against `modify_conformer` (step 1e-4 rad).
- `test_invariance.py`:
  - a P_W-projected L-step of 0.05 Å RMS changes each TD-quadruple rotatable dihedral by < 1e-3 rad, and the drift
    scales as step² (checked at 0.01 / 0.02 / 0.05 Å);
  - the chirality guard catches a deliberate 0.3 Å step that flips χ_c, which happens in ≈ 1/20 random steps
    [E-JOINT-032];
  - a TD torsion step changes no bond length or angle (> 1e-5 fails), as a regression of [E-JOINT-026].
- `test_parity.py`: on x and −x, the torsion and CP heads negate, the `0e` head is unchanged, and the `1o` head
  negates.
- `test_constraints.py`:
  - an enantiomer (−x) of a chiral molecule is flagged;
  - a deliberate bond stretch is backtracked;
  - GT conformers pass every check.
- `test_rings.py` (V3b): the CP round-trip error on QM9 monocycles is < 1e-6 with the seed's own bonds and angles,
  and the fused/bridged/spiro counts reproduce 420/465 on the test set [E-JOINT-071].
- `test_golden_td.py`: with w = 0 and geometry steps off, `flexitors` training-step loss and sampler output equal TD's
  on a fixed-seed fixture. This is the pattern of `tools/tests/test_leakfree.py`.
- `test_transform.py`: the same Δτ is applied to both endpoints, so the torsion difference between x_R' and x_G'
  equals that between x_R and x_G.

### 5.4 Consults the coding agents must file before choosing

- `Q-*-001`: σ_b, w, and the shared vs independent clock. Defaults are given here; a change needs a source.
- `Q-*-002`: E/Z handling, to be aligned with Exp 3 (R-CHEM).
- `Q-*-003`: DAgger re-matching cost. The round-1 conformer-matching timing is **UNVERIFIED** here; the orchestrator
  should supply it from round-1 logs.

## 6. Compute (from 11.7 GPU-h per 100-epoch QM9 run on one RTX 3090 [E-JOINT-008])

| Item | Estimate | Basis |
|---|---|---|
| M0 | 0 GPU-h, ≈ 1 CPU-h | linear algebra on ≈ 300 molecules |
| Smoke run (1 epoch, geometry head + projector) | ≈ 0.12–0.18 GPU-h | measures the real overhead before the Gate-0 ask (added after D-006) |
| M1 (A11, 3 σ_b × 30 epochs, full QM9 train set) | **≈ 10.5 GPU-h at TD speed; ceiling 16 GPU-h** | 0.3 × 11.7 × 3; ceiling = +50 % unmeasured overhead (revised after D-006; superseded "≈ 12 GPU-h" with an assumed +15 %) |
| M2 (A4 1 seed + A6 + A6b 1 seed) | ≈ 12–18 + 0.5 + 12–18 GPU-h, plus A6b re-matching CPU (UNVERIFIED) | 11.7 × 1.0–1.5 per 100-epoch run |
| A2 seeds (xTB) | ≈ 0.5–3 CPU-h test; train-set matched seeds ≈ 1.5 M confs × 0.14 s ≈ 60 CPU-h (parallel) + one 11.7 GPU-h retrain × 3 | [E-JOINT-008] |
| Main table, new trained arms (A2, A3, A4, A5, **A6b**, A7, A8) × 3 seeds | ≈ 21 runs × 12–18 GPU-h ≈ 250–380 GPU-h | A6b added after D-005 |
| DAgger / re-matching rollouts (A5, A6b, A8) | ≈ 35–60 GPU-h + matching CPU (UNVERIFIED) | training set ≈ 100× test-set inference |
| Inference + xTB eval, all arms | ≈ 15 GPU-h + ≈ 30 CPU-h | 15–25 min per run |
| D-001 budget-matched leak-free re-score | 0 GPU-h, a few CPU-h | existing `eval.pkl`/`confs.pkl` |
| **Decision-gate ask (smoke + M1 + M2)** | **≈ 35–53 GPU-h** (revised; superseded "≈ 30") | |
| **Full QM9 table** | **≈ 300–450 GPU-h ≈ 3–5 days on 4 × 3090** | |
| GEOM-DRUGS | ≈ 10× training (estimate, unmeasured) | `round4/BRIEF.md` |

## 7. Risks

1. *(Revised after re-score: this risk is now stronger. An unmodified RDKit-trained TD model already gains from
   independent better L (CR −0.015 Å on distinct targets [E-JOINT-011]), so A6 and A6b may both gain, and the
   marginal value of coupling (A5 − A6b) is expected to be small.)*
   **Coupling may not matter (revised after D-003/D-005).** The best-of-3 other-conformer floor is 0.036 Å vs
   0.114 Å for RDKit L on the same rows [E-JOINT-010 rev.], and the leak-free failure is explained by L mismatch. So
   **A6b (decoupled, sampler-matched) may do as well as A5**. That is the *a priori* most likely outcome given current
   evidence, and it would be a legitimate negative result for "joint": the paper would become "learned L +
   sampler-matched training". A6b is a pre-declared primary contrast, so this cannot be hidden.
2. **Multimodal L velocity.** Ring flips mean that too small a σ_b averages puckers into distorted rings. Mitigated by
   A11, which is read out cheaply on the rigid stratum. *(Revised after Q-P1-002 / Q-P3-001; superseded text: "the
   round-2 λ-blend defect … means the paired cache must be rebuilt".)* That defect is in the test-time λ seed
   pickles only. The training cache `cache_paired` keeps-and-flags pair_ok failures and **is reused** [E-JOINT-033].
3. **Exposure bias.** Without DAgger, A4 behaves like S3/B1 on MMFF L [E-JOINT-005]. DAgger is mandatory, and its
   matching cost is unmeasured.
4. **RMSD may not move even if geometry improves** [E-JOINT-152]. Both endpoints are pre-declared.
5. **Constraint layer as hidden optimisation.** It is pure geometry with counters reported. A9b measures planarity
   separately.
6. **Competition.** PuckerFlow's group has proposed joint CP × torsion as a feasible extension [E-JOINT-069] (revised
   after D-008), and GO-Flow is the GD-P
   precedent. The claim stays scoped to §1's "new" list.
7. **Absolute level.** Best case (CTRL-type model on λ ≈ 0.5–0.75-quality L, round-4 C-ARCH-01 §3, INFERENCE) is
   AMR-R ≈ 0.11–0.14 Å. That is still above the Cartesian frontier of 0.068–0.073 [E-JOINT-009]. The paper cannot be
   "SOTA".
8. **QM9 saturation** [E-JOINT-155]. COV-R@0.1, internal-geometry and energy metrics carry the QM9 result, and
   GEOM-DRUGS is needed.

## 8. Novelty verdict (revised after D-001/D-003/D-004/D-005/D-008; original kept below)

- **"Joint geometry–torsion modelling" as such is NOT unexplored.**
  - RINGER (2023) diffuses bond angles and torsions jointly for macrocycles [E-JOINT-060].
  - FoldingDiff does angles + dihedrals for proteins [E-JOINT-062].
  - GO-Flow flows bonds × angles × torsions for small molecules [E-JOINT-063].
  - TD and Corso proposed it [E-JOINT-047, -078].
  - PuckerFlow proposes joint CP × torsion as a feasible extension [E-JOINT-069].
  - "Multi-scale conformer generation" is taken by EBD [E-JOINT-090].
- **What remains NOVEL (scoped):**
  1. **(unconditional)** the first TD-protocol GEOM benchmark of a learned-L torsion model on drug-like
     **polycyclic** rings, with pucker, stereo and xTB metrics, plus the leak-free *methodology* (own vs split,
     budget-matched) [E-JOINT-071, -079, -049];
  2. **(minor method detail)** the ker-W ∩ rigid^⊥ Cartesian geometry step: dihedral-preserving to first order and
     made exact by rigid side-rotation restoration (revised after Q-P2-002, §10.2),
     closure-free for any ring topology, chirality-guarded [E-JOINT-032, -063, -072, -079]. It is **not** an "exact
     separation";
  3. **(hypothesis, not a finding)** that coupling L to τ helps beyond sampler-matched training. It counts only if
     A5/A8 beat A6b. The leak-free test does not support it [E-JOINT-001 rev., -010 rev.].
- **Overall: INCREMENTAL.** It becomes NOVEL-as-method only if A5/A8 beat A1/A2 (real use) **and** A6b (coupling).
  Without the first win it is a workshop analysis paper. Without the second it is a "learned L + sampler-matched
  training" paper, with the joint framing dropped.

<details><summary>Original §8 (before revision 1), kept for the record</summary>

- What remains NOVEL (scoped): 1. the evidence that L must be coupled to τ, from the leak-free split-half test with
  the decoupled control; 2. the L-subspace projection construction, which gives exact torsion/L separation and
  closure-free rings of any topology; 3. the first TD-protocol GEOM benchmark … Overall: INCREMENTAL-to-NOVEL,
  contingent on A5/A8 beating both A1/A2 and A6. — Items 1 and 2 were withdrawn after D-003/D-005 and D-004.
</details>

## 9. Known issues (added in revision 1)

1. **Code bug, `tools/local_structure_analysis.py:302`** [E-JOINT-031]. `np.argsort(cost_gg[i])[:args.floor_topm]`
   (with `--floor_topm` default 3, `:74`) includes the target conformer i whenever n_GT ≤ 3. The self cost is
   `np.inf`, which only sorts it last. All 769 rows with `n_gt ≤ 3` in
   `cluster_sync/results/analysis/local_structure_test.csv` have floor_gtother ≤ 2e-5 Å.
   - Affected: `round2/research_A.md:24-34` (0.0323 / 0.0137 / 0.0269), E-ARCH-010, E-JOINT-010, E-CHEM-065.
   - Fix (for a coding agent; research agents do not edit code): drop i from the candidate list before slicing.
   - Re-run the n_GT ≤ 3 molecules on gnode118 and add an other-half-only variant matching the split-half design.
     Until then, cite only the n_GT > 3 values (ring 0.0359 Å, RDKit 0.1143 Å on the same rows).
2. **Leak-free reference not budget-matched** (D-001). The CPU re-score of §3.6 must run before any leak-free gap
   versus RDKit L is cited (EXP-1, EXP-2, `round5/PLAN.md` phase 2).
3. **Projector overhead unmeasured** (D-006). The smoke run must happen before the Gate-0 ask.
4. **DAgger / re-matching CPU cost unmeasured** (§5.4 Q-*-003). It now applies to A5, A6b and A8.
5. **Verification debt.** Ledger entries E-JOINT-130, -132, -136, -151 and -154 are UNVERIFIED-IN-R5 (quotes not
   re-extracted) and must not support any claim in a paper draft until re-checked.

## 10. Revision 2: binding decisions after the coding consult (2026-10-16)

This section supersedes any conflicting text in §2–§9. Each decision names the consult file(s) it answers.

### 10.1 V2 training transform (Q-P1-001, Q-P1-002, Q-P2-001, Q-P2-003, Q-P3-001, Q-P3-002)

1. **Data.** Reuse `cache_paired.{train,val}` [E-JOINT-033]. Keep TD's training population: molecules with m ≥ 1.
   No m = 0 molecules are added, because they have no paired data and adding them would change A3/A0
   comparability. Conformers with `pair_ok = False` stay in the batch with τ loss only, and their geometry loss is
   masked to 0. The masked share is logged.
2. **W rule (dihedral set): heavy-first.** For each rotatable bond (b, c), take the lowest-index heavy neighbour of b
   (≠ c) and of c (≠ b); fall back to H only if a side has no heavy neighbour. Drop a row if
   min(sin θ_abc, sin θ_bcd) < sin 10° (near-linear, e.g. next to a triple bond). If the heavy choice is near-linear,
   try the next neighbour first. Count dropped rows. The share of H-terminated rows and the per-step change of
   **all** heavy-atom dihedrals about rotatable bonds are diagnostics.
   - Reason: with TD's first-neighbour rule an H-terminated row lets a geometry step change the heavy-atom dihedral
     through "L" (Q-P2-001). Evaluation is heavy-atom RMSD.
   - TD's torsion updates are whole-side rotations about the bond, so they do not depend on which quadruple defines
     τ [E-JOINT-026].
   - Option (c), all heavy quadruples, over-constrains angles at sp³ centres.
   - Dihedral values are computed with exact atan2, not TD's clamped `acos` form (`utils/torsion.py:117-123` clamps
     cos to ±(1 − 1e-5)).
3. **Pure-L target x_G\*.** Set the m W-quadruple dihedrals of x_G (own GT) to x_R's values by rigid side rotations,
   using `modify_conformer` with TD's sign convention (`torsion.py:68` comment: "positive rotation if pointing
   inwards"). Unit-test the sign against atan2. Then Kabsch-align x_G\* to x_R on heavy atoms.
   - Side rotations change no bond length or angle [E-JOINT-026], so x_G\* − x_R carries **no W-dihedral
     difference by construction** (Q-P2-003). The residual DE torsion mismatch stays with the torsion factor, which is
     trained on x_R's torsions as in TD.
   - Then apply the same Δτ to x_R and x_G\* (§2.2 step 3).
4. **Parameterisation: endpoint (x₁) prediction.**
   - Input: x_t = interp(x_R′, x_G\*′, t) + γ(t)·P_W(x_R′) z, with γ(t) = σ_b√(t(1−t)) and **t ~ U[0, 1]**.
   - The head outputs x̂₁ through a projected per-atom `1o` displacement: x̂₁ = x_t + P_W(x_t)·Δ̂.
   - Loss: ‖P_W(x_t)(x̂₁ − x_G\*′)‖², summed per molecule, divided by rank(P_W) of that molecule, then averaged over
     the molecules in the batch.
   - Reason: the velocity target's γ′(t) term diverges at t = 0, 1, and ET-Flow states no clipping rule
     [E-JOINT-082]. x₁ prediction is used by HarmonicFlow (p. 4: "we find it helpful to parameterize v to predict x1
     instead of (x1 - x0)") and by PuckerFlow (SI Alg. 1, p. 20) [E-JOINT-080, -081]. No ε is invented.
5. **Chirality in training states.** If x_t flips any stereocentre relative to x_R′ (signed volume, §2.8), resample z
   (≤ 5 tries). If it still flips, mask the geometry loss for that sample and count it. This is consistent with V3a's
   reject-and-resample.
6. **Clocks.** The geometry clock t enters through the λ-embedding path with `lambda_embed_dim = 32`, the round-2 S4
   mechanism [E-JOINT-023]. The torsion clock σ is unchanged. Loss weights: w_τ = 1, w_geo = w (§2.2).

### 10.2 Sampler (Q-P1-001.2, Q-P2-002, Q-P3-002, Q-P3-003.2–3)

- **Geometry update**, deterministic Euler, N = 20 uniform steps on t ∈ [0, 1] (TD's step count; same NFE):
  x ← x + (Δt / (1 − t))·P_W(x)(x̂₁ − x), and the last step sets x ← x + P_W(x)(x̂₁ − x). ET-Flow's stochastic
  sampling in t ∈ [0.8, 1] [E-JOINT-082] is a later ablation, not Gate 0.
- **Exact torsion restoration ON in every sampler**: primary V2, DAgger roll-outs, and the A6/A6b "τ frozen" sampler.
  After each geometry step, the m W-dihedrals are measured (atan2) and each moving side is rotated back by its change
  with `modify_conformer`. This changes no bond length or angle and cannot flip a tetrahedral centre [E-JOINT-026].
  It makes "τ frozen" exact, and it means the torsion head only ever sees torsions it produced. The P_W projection is
  kept so that steps do not fight the restoration. `restore_torsions = off` is an ablation that is measured once.
- **τ-frozen read-out (M1, A6, A6b sampler).** ETKDG seed, no torsion randomisation, torsion-clock input
  σ = σ_min throughout. σ_min lies inside the training range [0.01π, π]. Ring dihedrals are L and are moved.
- **m = 0 molecules**: geometry steps run with P_W = I − J_rigᵀ(J_rig J_rigᵀ)⁺J_rig. TD's sampler skips them
  (`generate_confs.py:150-153`); FlexiTors does not.

### 10.3 M0 and Gate 0 (Q-P1-003, Q-P2-004, Q-P3-003, Q-P3-009)

- **M0, revised: oracle-velocity integration test.**
  - Why: the old criterion was vacuous (see the §5.2 pointer).
  - What is tested: whether the *sampler's* projected, torsion-restored, constraint-guarded update can follow an
    oracle field over large, nonlinear ring moves.
  - Population: the **validation** paired cache (`cache_paired.val`), molecules with ≥ 1 heavy rotatable bond **and**
    a ring of ≥ 4 atoms. Rigid and m = 0 rows are reported as sanity rows.
  - Procedure: integrate x ← x + (1/(N−n))·P_W(x)(x_G\* − x), N = 20, with restoration and the §2.6/§2.8 layer on.
  - Metrics: recovery = 1 − median_pairs[RMSD_heavy(x_N, x_G\*)] / median_pairs[RMSD_heavy(x_R, x_G\*)]; ring-dihedral
    RMSD to x_G\*; W-dihedral drift; chirality flips caught; constraint backtracks and skips.
  - **Kill** if recovery < 50 % or if > 5 % of pairs end with a skipped step or a flipped centre.
  - Test-set pairs are reported once and never decide.
- **M1 training (Q-P1-001.3, Q-P3-009): option (a).** The whole network trains from random initialisation on the
  geometry loss only (w_τ = 0, τ noise still applied). This matches the cost basis 0.3 × 11.7 GPU-h. A TD checkpoint
  cannot be loaded strictly once the clock widens the embedding input. The question Gate 0 answers is "can this
  architecture learn L". **A4 (M2) trains from scratch**, not from the M1 checkpoint, so its epoch count matches
  A0/A3.
- **Who decides (Q-P1-003.3, Q-P2-004.3, Q-P3-003.1): validation decides both σ_b\* and GO; test is read once
  afterwards and reported.**
  - Validation set: a fixed 1000-molecule subset of QM9 split 1 (sorted indices, RNG seed 0), with its ETKDG and MMFF
    baselines computed first on CPU (the rigid stratum and ≥ 4-ring seeds of that subset).
  - Thresholds stay as pre-declared: rigid AMR-R ≤ 0.10 Å, and the ≥ 4-ring > 10° share ≤ 20 %.
  - **Scaling rule:** if a validation ETKDG baseline differs from the test value (0.152 Å; 49.6 %) by more than
    10 % relative, the threshold scales by validation ÷ test.
  - The ring criterion is read for σ_b\* only.
  - Reason: README rule 5 forbids model selection on test, and the planners rightly noted that a test-read gate is a
    decision.
- **Ring metric pairing (Q-P1-003.2, Q-P2-004.4).** Primary: each flowed seed keeps its original (k, src_gt_idx)
  pairing from `l_error.csv`, which is comparable with the 49.6 % baseline. Secondary: the best over all GT
  conformers of the molecule.
- **m = 0 test molecules (Q-P1-002.3).** These are not trained on, so they are reported as a separate sub-row. GO is
  read on the n_rot_heavy = 0 stratum as pre-declared.

### 10.4 A6b, DAgger and compute matching (Q-P1-004, Q-P2-005)

- **Roll-out seeds.** For A5/A8 DAgger and for the A6b training data, roll-outs start from **fresh ETKDG embeddings**
  made as `standardize_confs.py` makes them, then conformer matching. They do **not** start from the cache's
  `data.pos[k]`, whose torsions were already matched to the target GT. A geometry head that sees τ would then see
  GT-matched torsions in training but raw ETKDG torsions at test; for A6b that is a leak plus a mismatch. This is the
  S5 recipe with a new L source [E-JOINT-006].
- **E_D = 20 epochs** for every DAgger / fine-tune stage. This is a design choice, not taken from a source.
- **A6b, primary:** A0 fine-tuned for E_D epochs on the re-matched sampler-L data, with the same optimiser and
  schedule as A5's DAgger stage, so it is compute-matched to A5 (100 + E_D). **A6b-full**, an S5-style retrain from
  scratch for 100 epochs, is secondary and is run only if A5 − A6b is significant (to rule out under-training of A6b).
- **New arm A1+** (Q-P2-005): S5 (A1) fine-tuned for E_D epochs on its own MMFF-matched data, same optimiser and
  schedule, 3 seeds, ≈ 3 × 2.3 GPU-h. It **replaces A1 in the primary family**, which is now A5 − A1+, A8 − A1+,
  A5 − A6b, A8 − A6b. A5 − A1 is reported as secondary. Training GPU-h and epochs are recorded per arm. Restarting
  uses `train.py --restart_dir`.
- **Re-matching cost** stays UNVERIFIED. The orchestrator is to supply the round-1 `standardize_confs.py` CPU-h for
  the QM9 training set; it sets the M2 ask.

### 10.5 D-001 re-score and floor re-run (Q-P1-011, Q-P3-003.4)

- **Family.** One Holm family of 8: CR, B1, S3, S4 × {AMR-R, COV-R@0.5} for split − ref_half, with the paired molecule
  bootstrap and the hierarchical interval beside each row. COV-R@0.1 and split_d − ref_half_d are secondary.
  Interpretation stays descriptive with respect to model selection (it corrects a published claim).
- **ref_half.** It is computed as the **exact expectation** over random subsets of size s = 2·|half| drawn from the
  N generated conformers. With per-target RMSDs sorted ascending r₍₁₎ ≤ … ≤ r₍N₎:
  - E[min] = Σᵢ r₍ᵢ₎·C(N−i, s−1)/C(N, s);
  - P(covered) = 1 − C(N−c, s)/C(N, s), with c the number of samples under δ.

  This is the N → ∞ limit of V5's ~10 draws and has no Monte Carlo noise. Ten random draws are reported as a
  consistency check.
- **Floor re-run.** Both `--mode test` and `--mode test --mmff_seeds`, since both CSVs carry the bug.
  - The other-half variant uses the **halves of `split.tsv`** from `tools/make_split_half.py` (the leak-free halves).
    The 30-conformer cap is applied after splitting, and the 85 molecules with > 30 clean conformers are counted.
  - Primary: top-3 by transplant cost among the other half (the same estimator as the existing column, so the
    bug-fix comparison is clean).
  - Reported beside it: the minimum over **all** other-half conformers for molecules with ≤ 10 per half, as a check
    that top-3 approximates what the split-half seeds cycle through.

## 11. Revision 3: reading the budget-matched re-score (2026-10-17)

**Data** [E-JOINT-011]: 711 molecules (606 distinct-target), split − ref_half, exact expectation, BUDGET Holm family
of 8.

| Model (training L) | AMR-R, all targets | COV-R@0.5, all (pt) | AMR-R, distinct targets |
|---|---|---|---|
| CR (matched RDKit) | **−0.038** [−0.048, −0.029], p_Holm < 0.001 | +0.1 (n.s.) | **−0.015** [−0.027, −0.003] |
| B1 (own GT) | −0.017 [−0.031, −0.003], p_Holm 0.06 | −4.4 | **+0.067** |
| S3 (50/50 mix) | **+0.017** [+0.005, +0.030], p_Holm 0.021 | −8.7 | **+0.085** |
| S4 (λ blends) | −0.007 (n.s.) | −5.3 | **+0.047** |

**Analysis check (R-JOINT, before relying on it).** I re-read `tools/analyze_leakfree.py:426-505`:
- `subset_min_expectation` uses P(min = i-th smallest) = C(N−i, s−1)/C(N, s);
- `subset_cover_prob` uses 1 − C(N−c, s)/C(N, s) with strict `<`, as the evaluator does;
- s = 2|H| per target half is the split runs' budget;
- directions are averaged as in `view()`;
- molecules where 2|H| exceeds the generated count are excluded, not truncated;
- the 32 excluded molecules are removed from every view, so the contrasts stay paired;
- the 10-draw MC means agree with the exact values to ≤ 0.0005 Å, and the round-4 part reproduces round 4 byte for
  byte.

I found no error. Remaining caveat: the distinct-target threshold (0.1 Å) is the round-4 pre-declared one, and
targets of the same CREST ensemble are never fully independent.

**What it supports**
1. The round-4 / PLAN phase-2 statement "with another conformer's true geometry every model is worse than with RDKit
   geometry" was a **budget artefact** and is withdrawn (E-JOINT-001 revision 2). The orchestrator should correct
   `round5/PLAN.md` phase 2; R-CHEM should correct E-CHEM-054.
2. **Better-but-not-own L helps an unmodified, RDKit-trained TD model.** CR gains −0.015 Å on distinct targets, and
   −0.038 Å overall including near-duplicates. That is ≈ 5 % of its distinct-target AMR-R, with no COV-R@0.5 change.
   This is an *oracle ceiling* for a decoupled learned-L arm built on A0 (A6).
3. **Torsion models trained on GT or blended L become identity-dependent.** B1, S3 and S4 are worse than RDKit L on
   distinct targets (+0.047 to +0.085 Å) and lose 4–9 pt COV-R@0.5. Their all-target "gains" come from near-duplicate
   targets; COV-R@0.1 is +20–25 pt for every model. This is the strongest evidence yet for the sampler-matched /
   DAgger rule: never train the torsion head on target GT L [E-JOINT-005, -006, -137].

**What it does not support**
- Coupling being necessary or helpful. Nothing in this test couples L to τ, and the one model that gains is
  decoupled. Coupling stays a hypothesis, tested only by A5/A8 − A6b.
- Any effect size for *learned* L. GT-quality L is the ceiling; a learned L of lower quality gives less.
- "Geometry is the lever" in the strong round-2/3 sense. The leak-free gain for CR (−0.015 Å distinct) is far smaller
  than the own-conformer ring oracle (0.178 → 0.119 Å), most of which was therefore conformer identity
  [E-JOINT-003, read as an upper bound].

**Design and gate consequences (Stage A)**
- **Gates unchanged.** M0, Gate 0 and Gate 1 measure L quality and real use, not the oracle.
- **New pre-declared expectation for A4 (V2 without DAgger).** V2's training feeds the torsion head interpolants
  toward the *own* conformer's GT L (x_t → x_G\* as t → 1). That is the same exposure as S4, which fails on distinct
  targets (+0.047 Å). Expectation: A4 ≥ A1+ on AMR-R, with an S4-like split_d deficit. This is why DAgger stays
  mandatory, and A4 − A5 (the DAgger effect) stays in the secondary family, **now with a pre-declared sign**
  (A5 < A4).
- **§3.6 oracle analysis.** The primary view is split_d − ref_half_d, with prediction (i′) added. CR's −0.015 Å is
  the reference "healthy decoupled" value.
- **A6 is re-scoped.** It is not "doomed by construction". It measures how much learned L helps the unmodified TD
  model, with oracle ceiling −0.015 Å (distinct targets). It stays secondary; A6b stays the primary coupling control.
- **Expected effect sizes for the M2 ask.** A6 − A0 ≥ −0.015 Å (ceiling). The real-use target of A5 − A1+ is
  ≤ −0.005 Å (Gate 1). The coupling contrast A5 − A6b is expected to be small; its detectability at 3 seeds × ≈ 900
  molecules is about ±0.002–0.004 Å (round-2 CI half-widths, e.g. S3 [+0.0013, +0.0052], S4 [−0.0009, +0.0078]). A null A5 − A6b is the *a priori* likely outcome and is
  reported as such.
- **Paper framing.** The unconditional contributions (EXP-1 §8 item 1) now include a correction to our own round-4
  conclusion and the identity-dependence result for GT-trained torsion models. Both are budget-matched oracle
  analyses, labelled ORACLE.

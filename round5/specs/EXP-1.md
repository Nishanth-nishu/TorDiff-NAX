# EXP-1: Joint local-geometry + torsional diffusion (FlexiTors)

Author: R-JOINT (round 5), 2026-10-15. Evidence IDs [E-JOINT-NNN] refer to `round5/ledger/evidence_JOINT.md`.
Round-4 IDs (E-ARCH, E-PRIOR) are quoted only where they were copied into the JOINT ledger and re-checked.
ORACLE = uses test-set true geometry; analysis only, never model selection (`orchestration/README.md` rule 5).
INFERENCE = my design reasoning, not something a source shows.

## 0. Summary

TD generates only torsions τ on T^m. It takes the local structure L (bond lengths, bond angles, ring puckers) from an
RDKit seed and never changes it [E-JOINT-024, E-JOINT-026, E-JOINT-040]. Our data put most of the remaining error in
ring geometry [E-JOINT-003, E-JOINT-004]. The leak-free split-half test [E-JOINT-001] showed that **another
conformer's true L makes every trained model worse than RDKit L** (CR 0.2195 vs 0.1873 Å).

That result needs one qualification before it can drive the design. With torsions optimised directly against the
target, another conformer's L leaves only a 0.032 Å floor for ring molecules [E-JOINT-010]. So the geometry is good
enough. What fails is a torsion model being handed L from a distribution it was not trained on. This is the same
brittleness that shows up on MMFF L [E-JOINT-005], and the only learned fix that has worked is training on the
deployed L source [E-JOINT-006].

**The design rule is therefore: the torsion factor must always see L drawn from the same process that produces L at
test time, and that process should be able to depend on τ.** Exp 1 builds this in three steps:

- **V1** is TD with fixed RDKit L. This is the baseline, and it is already run.
- **V2** adds a learned correction to L that is conditioned on the current τ and generated in the same network,
  followed by DAgger fine-tuning of the torsion head.
- **V3** is a joint L + τ diffusion with independent clocks on a product space.

The primary geometry representation is a **Cartesian velocity projected onto the local-structure subspace**, the
orthogonal complement of the m torsion directions and the 6 rigid-body directions. TD's own likelihood code already
builds that Jacobian [E-JOINT-028]. With this representation, ring closure needs no special handling and the geometry
factor cannot move torsions. A Cremer–Pople variant is a secondary ablation.

The decisive control is arm A6. It takes the same trained geometry factor and uses it as an independent L sampler
feeding the frozen V1 torsion model. If V2/V3 do not beat A6, the contribution is "learned L", not "joint L–τ".

## 1. Prior-art check (adversarial)

The searches were 18 arXiv abstract queries over 2023-01 to 2026-10 plus two web searches [E-JOINT-079], together
with every relevant paper already in `papers/`. Each claim below is paired with what kills it or what survives.

| Candidate claim | Closest prior art (what it does) | Verdict |
|---|---|---|
| "Relax TD's rigid L / diffuse over full intrinsic coordinates" | TD's own conclusion proposes exactly this [E-JOINT-047]; Corso's thesis proposes pucker hyperspheres and a soft-constraint full-dimensional diffusion [E-JOINT-078] | **PRIOR ART (idea)** |
| "Joint bond-angle + torsion diffusion" | RINGER: wrapped-normal joint diffusion of bond angles and torsions in redundant internal coordinates, SLSQP reconstruction, macrocycles only [E-JOINT-060]. FoldingDiff: 3 bond angles + 3 dihedrals per residue, NeRF, proteins [E-JOINT-062] | **PRIOR ART** |
| "Full internal coordinates (bonds × angles × torsions) for small-molecule conformers" | GO-Flow: one flow per manifold, internal → Cartesian Jacobian, GD-P protocol only, no ring-closure treatment [E-JOINT-063, -064, -065] | **PRIOR ART**, but with a hole (rings) |
| "Ring-pucker factor next to torsions" | PuckerFlow: CP flow, monocyclic, unsubstituted, a two-stage demo with TD, and joint CP × torsion flow announced as future work on the TD codebase [E-JOINT-066..069]. TMCgen: S² angle factor + torsions, metal complexes [E-JOINT-070] | **CONTESTED**: no paper yet for drug-like polycyclic molecules on GEOM [E-JOINT-079]; an active group has announced it |
| "Learned correction of the RDKit conformer" | CoarsenConf: learns the RDKit → GT Cartesian difference [E-JOINT-075]. FM-refiner: refines any upstream sample, but has no TD + refiner row [E-JOINT-077]. EBD: corrects RDKit fragment centroids [E-JOINT-091] | **PRIOR ART**; none restricts the correction to the L-subspace or keeps a torsion factor |
| "Multi-scale / hierarchical conformer generation" | EBD (fragment → atom, NeurIPS 2024) [E-JOINT-090]; MSGEN (scaffold-first, abstract) [E-JOINT-095]; HierDiff and OctMole (de novo, coarse-to-fine) [E-JOINT-093, -094] | **PRIOR ART: do not use this framing** |
| "Torsion model conditioned on / robust to L" | Torsional-GFN follows mode shifts for unseen L (small MD set) [E-JOINT-074] | **PRIOR ART** for conditioning. Our *brittleness* finding is the counter-evidence |
| L-subspace projection: Cartesian geometry velocity projected orthogonal to torsion + rigid directions using TD's Jacobian | GO-Flow goes the other way [E-JOINT-063]; DiffDock's factors are disjoint by construction [E-JOINT-072]; nothing found [E-JOINT-079] | **NOVEL (method detail)** |
| Coupling is *necessary*: leak-free split-half test + decoupled control | TD F.1 uses other-conformer L as a matching floor only [E-JOINT-049] | **NOVEL** |
| Ring geometry dominates TD's QM9 error with the model in the loop | Corso: "works fine for small rings" [E-JOINT-078]; HierDiff: rings are what atom models get wrong (de novo) [E-JOINT-093] | **NOVEL as a measurement**, but leak-corrected numbers are needed first (§3.6) |

**What is and is not new, stated exactly.**
- Not new: the idea of learning L, joint angle + torsion diffusion, internal-coordinate flows for small molecules,
  ring-pucker flows, learned RDKit corrections, and the "multi-scale" framing.
- New, if it works:
  1. A torsion-diffusion model whose geometry factor lives in the L-subspace. This construction gives exact separation
     from torsions and closed rings of any topology (fused, bridged, spiro) without a closure solver.
  2. Evidence that coupling is necessary, from the leak-free test plus the decoupled control A6.
  3. The first TD-protocol GEOM benchmark of a learned-L torsion model, with ring-pucker, stereo and xTB-energy
     metrics.
- Point 1 is a modest method contribution. Points 2 and 3 carry the paper. Our absolute numbers (best real use
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

**Why coupled rather than independent.** An independent L generator feeding a frozen torsion model reproduces the
leak-free failure: the torsion model sees L from a distribution it was not trained on [E-JOINT-001, -005, -010]. V2
puts both factors in one network so that:
- (a) the geometry head sees the current torsions when it moves L;
- (b) the torsion head is trained, and in stage 2 fine-tuned, on L produced by that geometry head
  [E-JOINT-006, -137].

**L-subspace projector.** At a conformer x, define:
- J_τ(x) ∈ R^{m×3n}: rows are TD's torsion displacement fields ∂x/∂τ_i (TD Eq. 9, PDF p. 7, [E-JOINT-044]), already
  computed in `likelihood.py:68-93` [E-JOINT-028];
- J_rig(x) ∈ R^{6×3n}: three translations plus three rotations x_k × ê (TD Eq. 10).

Let B = [J_τ; J_rig] and P_L(x) = I − Bᵀ(BBᵀ)⁺B. P_L is the orthogonal projector onto the local-structure subspace,
of dimension 3n − 6 − m. (INFERENCE: a direct construction from TD's Prop. 3 machinery; no source does this.)

A step δx = P_L v therefore:
1. changes no rotatable torsion to first order;
2. adds no rigid motion;
3. keeps every ring closed, because Cartesian coordinates carry no closure constraint.

Ring closure is a problem only in internal-coordinate parameterisations: RINGER needs SLSQP [E-JOINT-060] and GO-Flow
leaves it open [E-JOINT-064].

**Training (one sample).**
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

**Choice of σ_b.** The ring error is a heavy tail: ETKDG ring-dihedral mean is 10.65° but the median is 0.44°, and
49.6 % of seeds exceed 10° [E-JOINT-004]. The RDKit → GT velocity is therefore multimodal for a minority of seeds,
and an interpolant with too little noise averages them into distorted rings. σ_b ∈ {0.05, 0.15, 0.3} Å is swept on
the rigid stratum only, at about 1 GPU-h each (arm A11). INFERENCE: 0.3 Å is the order of a ring-flip displacement.

**Sampling.**
1. ETKDG seed at t = 0; τ ~ U[0, 2π)^m (TD).
2. For each step n = N … 1:
   - one network call gives (score_τ, v);
   - torsion SDE step exactly as `sampling.py:204`;
   - geometry Euler step x ← x + Δt·P_L(x) v;
   - constraint layer (§2.6).
3. Schedule variants (arm A10): joint (both every step), L-first, L-last.

The NFE is the same as TD's: one forward pass per step returns both heads. The projector adds one m+6 least-squares
solve per molecule per step.

**Stage 2, DAgger (mandatory).**
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
  - it is the variant most exposed to PuckerFlow's announced work [E-JOINT-069].
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
| 9 | Constraint cost | share of per-step time spent in P_L and §2.6; backtracks and skips per conformer; induced torsion drift per step | cost / diagnostic |
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

Pre-declared predictions:
- (i) a coupled model's split-half AMR-R is **no worse than its RDKit-L AMR-R**, unlike CR (+0.032), B1 (+0.048) and
  S3 (+0.088) [E-JOINT-001];
- (ii) its own − split leak gap is smaller than S4's (0.17 Å).

If (i) fails, the coupling claim fails, whatever the primary metrics show. Report the `split_d` distinct-target view
alongside, since ≈ 40 % of targets have a near-duplicate in the other half [E-JOINT-002].

## 4. Ablation table and the controls against "winning by more optimisation"

Every arm uses the **same fixed list of ETKDG seeds** (same RNG seeds), the same K = 2 × n_GT samples, the same
evaluator, and 100 epochs unless noted.

| Arm | Description | New? | Seeds | Purpose |
|---|---|---|---|---|
| A0 | V1 CTRL_rematch, RDKit L | have | 3 | baseline |
| A1 | V1 + MMFF seeds; S5 | have | 3 | strongest cheap real-use baseline today (0.150) |
| A2 | V1 + GFN2-xTB-relaxed seeds, torsion model retrained on xTB-matched seeds | new | 3 | **level-matched geometry baseline** [E-JOINT-134]; if Exp 1 ≤ A2, there is no method contribution |
| A3 | V1 parameter-matched: FlexiTors network with the geometry head present but w = 0 and no geometry steps | new | 3 | controls for capacity and code changes; must equal A0 within noise (golden test) |
| A4 | V2, no DAgger | new | 3 | coupled correction |
| A5 | V2 + DAgger | new | 3 | **main V2 arm** |
| A6 | **Decoupled**: A5's geometry head run alone (τ frozen at seed) to produce L, then the frozen A0 torsion model on that L | new (no training) | 3 | **mechanism control: coupling vs "learned L"** |
| A7 | V3a, shared clock | new | 3 | joint diffusion |
| A8 | V3a + DAgger | new | 3 | **main V3 arm** |
| A9 | V3b (CP), only if A8 passes Gate 1 | new | 1→3 | coordinate ablation |
| A9b | A5 or A8 with the planarity projection on | new | 1 | constraint-layer effect |
| A10 | Schedules: joint / L-first / L-last | new | 1 | sampler ablation |
| A11 | σ_b sweep {0.05, 0.15, 0.3} Å, geometry head only, 30 epochs | new | 1 | Gate 0 |
| A12 | V1 at equal wall-clock (more steps, 2× NFE) | new (inference only) | 3 | equal-compute control |
| A13 | A0 output + post hoc learned refiner (FM-refiner style, trained on A0 samples) | new | 1 | "why not just refine TD's output?" [E-JOINT-077] |
| A14 | A0 + MMFF after sampling; A0 + xTB after sampling | partly have (A3 round 1) | 3 | relaxation trades recall for precision [E-JOINT-135]; reported separately, never folded in |

**Pre-declared primary contrasts (Holm family):** A5 − A1, A8 − A1, A5 − A6, A8 − A6, each on AMR-R and COV-R@0.5.
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
7. **Decoupled contrast A6.** It isolates "coupling" from "learned L".
8. **No selection on the test set.** Checkpoints are picked by validation loss; σ_b and w are picked on the validation
   rigid stratum; the leak-free oracle is analysis only.

## 5. Coding instructions

### 5.1 Package layout (`flexitors/`, per `round5/PLAN.md` "Code structure")

TD stays frozen in `torsional-diffusion/`. `flexitors/` imports from it, and nothing is copied except where noted.
Every mechanism carries a comment `# <paper>, PDF p. N, eq. M` or `# [E-JOINT-NNN]`.

| Module | Contents | Reuses (path:line) |
|---|---|---|
| `flexitors/geometry/jacobian.py` | `torsion_jacobian(pos, edge_index, edge_mask, mask_rotate)` returning J_τ ∈ R^{m×3n}; `rigid_jacobian(pos)` returning 6×3n; batched torch versions | `torsional-diffusion/diffusion/likelihood.py:68-74` (`dx_dtau`), `:59-65` (`inertia_tensor`), `:84-93` (rows and rigid removal) |
| `flexitors/geometry/projector.py` | `l_subspace_projector(pos, ...)` returning P_L (dense for n ≤ 64, otherwise implicit via `torch.linalg.lstsq`); `project(v, ...)`; rank check | `jacobian.py` |
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

- **M0 (CPU only, ≈ 1–2 days of coding).** `geometry/jacobian.py`, `projector.py`, `constraints.py` and their tests.
  Then one offline experiment with **no learning**: take the round-2 paired cache and, for each QM9 test-set rigid
  molecule's matched RDKit seed, apply the *oracle* step x_R + P_L(x_R)(x_G − x_R), as ORACLE analysis only.
  Measure:
  - (a) the torsion drift caused (should be about 0);
  - (b) how much of the RDKit → GT RMSD the L-subspace captures (fraction of ‖x_G − x_R‖² inside range P_L);
  - (c) the chirality flips.

  This shows whether the subspace carries the error before any GPU is spent. **Kill criterion:** if the projected
  oracle step recovers < 50 % of the rigid-stratum RMSD gap (ETKDG 0.152 → own-GT), the representation is wrong.
- **M1 = Gate 0 (≈ 4–6 GPU-h).** Train only the V2 geometry head (torsion loss weight 0, τ noise still applied),
  30 epochs, 1 seed, σ_b sweep (A11). Evaluate on the rigid stratum and the ring tail.
  **GO** if rigid AMR-R ≤ 0.10 Å (ETKDG 0.152, MMFF 0.128) **and** the > 10° ring tail ≤ 20 % (ETKDG 49.6 %, MMFF
  35.1 %) [E-JOINT-004, -007].
- **M2 = Gate 1 (≈ 15 GPU-h).** Full V2 (A4), 100 epochs, 1 seed, plus A6 (no training). **GO** to the 3-seed table if
  A4 < A1 on AMR-R by ≥ 0.005 Å and A4 < A6.

### 5.3 Test plan (`flexitors/tests/`)

- `test_projector.py`:
  - P_L is idempotent and symmetric;
  - P_L·J_τᵀ ≈ 0 and P_L·J_rigᵀ ≈ 0, both to 1e-6;
  - rank = 3n − 6 − m;
  - m = 0 molecules work;
  - finite-difference check of `torsion_jacobian` against `modify_conformer` (step 1e-4 rad).
- `test_invariance.py`:
  - a projected L-step of 0.05 Å RMS changes every rotatable torsion by < 1e-3 rad (via `get_torsion_angles`);
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
| M1 (A11, 3 × 30 epochs) | ≈ 12 GPU-h | 0.3 × 11.7 × 1.15 (≈ 15 % head + projector overhead, INFERENCE) × 3 |
| M2 (A4 1 seed + A6) | ≈ 14 + 0.5 GPU-h | 11.7 × 1.15–1.25 |
| A2 seeds (xTB) | ≈ 0.5–3 CPU-h test; train-set matched seeds ≈ 1.5 M confs × 0.14 s ≈ 60 CPU-h (parallel) + one 11.7 GPU-h retrain × 3 | [E-JOINT-008] |
| Main table, new trained arms (A2, A3, A4, A5, A7, A8) × 3 seeds | ≈ 18 runs × 13–15 GPU-h ≈ 240–270 GPU-h | |
| DAgger rollouts (A5, A8) | ≈ 25–40 GPU-h + matching CPU (UNVERIFIED) | training set ≈ 100× test-set inference |
| Inference + xTB eval, all arms | ≈ 15 GPU-h + ≈ 30 CPU-h | 15–25 min per run |
| **Decision-gate ask (M0–M2)** | **≈ 30 GPU-h** | |
| **Full QM9 table** | **≈ 300–350 GPU-h ≈ 4 days on 4 × 3090** | |
| GEOM-DRUGS | ≈ 10× training (estimate, unmeasured) | `round4/BRIEF.md` |

## 7. Risks

1. **The leak-free premise is narrower than "geometry needs coupling".** The geometric floor with other-conformer L
   is 0.032 Å [E-JOINT-010], so A6 (decoupled, learned L) may do as well as A5. That would be a legitimate negative
   result for "joint", and the paper would become "learned L + sampler-matched training". The arm is pre-declared so
   this cannot be hidden.
2. **Multimodal L velocity.** Ring flips mean that too small a σ_b averages puckers into distorted rings. Mitigated by
   A11 on the cheap rigid stratum. The round-2 λ-blend defect (≈ 10 % of slots reused another conformer's blend)
   means the paired cache must be rebuilt, not reused.
3. **Exposure bias.** Without DAgger, A4 behaves like S3/B1 on MMFF L [E-JOINT-005]. DAgger is mandatory, and its
   matching cost is unmeasured.
4. **RMSD may not move even if geometry improves** [E-JOINT-152]. Both endpoints are pre-declared.
5. **Constraint layer as hidden optimisation.** It is pure geometry with counters reported. A9b measures planarity
   separately.
6. **Competition.** PuckerFlow's group has announced joint CP × torsion [E-JOINT-069], and GO-Flow is the GD-P
   precedent. The claim stays scoped to §1's "new" list.
7. **Absolute level.** Best case (CTRL-type model on λ ≈ 0.5–0.75-quality L, round-4 C-ARCH-01 §3, INFERENCE) is
   AMR-R ≈ 0.11–0.14 Å. That is still above the Cartesian frontier of 0.068–0.073 [E-JOINT-009]. The paper cannot be
   "SOTA".
8. **QM9 saturation** [E-JOINT-155]. COV-R@0.1, internal-geometry and energy metrics carry the QM9 result, and
   GEOM-DRUGS is needed.

## 8. Novelty verdict

- **"Joint geometry–torsion modelling" as such is NOT unexplored.** RINGER (2023) diffuses bond angles and torsions
  jointly for macrocycles [E-JOINT-060]. FoldingDiff does angles + dihedrals for proteins [E-JOINT-062]. GO-Flow flows
  bonds × angles × torsions for small molecules [E-JOINT-063]. TD and Corso proposed it [E-JOINT-047, -078]. PuckerFlow
  announced joint CP × torsion [E-JOINT-069]. "Multi-scale conformer generation" is taken by EBD [E-JOINT-090].
- **What remains NOVEL (scoped):**
  1. the evidence that L must be coupled to τ, from the leak-free split-half test with the decoupled control
     [E-JOINT-001, -049];
  2. the L-subspace projection construction, which gives exact torsion/L separation and closure-free rings of any
     topology [E-JOINT-063, -072, -079];
  3. the first TD-protocol GEOM benchmark of a learned-L torsion model on drug-like **polycyclic** rings, with pucker,
     stereo and xTB metrics [E-JOINT-071, -079].
- **Overall: INCREMENTAL-to-NOVEL, contingent on A5/A8 beating both A1/A2 (real use) and A6 (coupling).** Without
  that win it is a workshop analysis paper (leak-free + ring decomposition), not a method paper.

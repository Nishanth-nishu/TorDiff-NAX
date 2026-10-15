# EXP-4: Full-molecule Boltzmann generator p(L, τ) = p_φ(L) p_θ(τ | L)

Author: R-JOINT (round 5), 2026-10-15. Evidence IDs [E-JOINT-NNN] refer to `round5/ledger/evidence_JOINT.md`.

**Revision 1 (2026-10-15), after verifier V5 (ruling: SUPPORTED with corrections; `round5/ledger/disputes.md`):**
- D-007: the variant-D cost was understated by up to about 2×, and §5 is corrected.
- D-004: τ in this chart is a *chart coordinate* (torsions are applied by `modify` after the displacement), not a
  measured dihedral, so the dihedral-drift issue of EXP-1's V2 projector does not affect the density. Any
  complement subspace of the anchor works as E_L; ½ log det g is computed for whichever basis is used.
- D-005: H1 vs H2 now matches the torsion models' training L, and is framed as a test of a hypothesis.
INFERENCE = my derivation or design, not something a source shows.

## 0. Honest scope

TD's Boltzmann generator samples p(τ | L₀). L₀ is the local structure of **one** ground-truth conformer per molecule:
- the paper says so in §4.5 [E-JOINT-046];
- the code says so at `utils/dataset.py:314-316` [E-JOINT-029];
- Transferable Boltzmann Generators says this is exactly why TD cannot sample the full Euclidean Boltzmann
  distribution [E-JOINT-111].

Corso's thesis names the missing piece as an L model with exact likelihoods [E-JOINT-112].

Exp 4 would supply that piece. Three things make the full version very expensive:
- exact likelihoods need the divergence over every dimension, and TD's method was feasible only because the torsion
  space is small [E-JOINT-043];
- the cost of exact Jacobians is the recognised bottleneck of this line of work [E-JOINT-113];
- ESS falls as dimension grows [E-JOINT-114].

**Recommendation: write Exp 4 as a feasibility section, not a headline contribution.** Run it only after Exp 1 Gate 1
passes, and lead with the cheap *harmonic-L* variant (§2.3). That variant reuses TD's torsion likelihood unchanged and
adds an analytic L density. The full diffusion density (§2.4) is evaluated on ≤ 100 molecules only. Energy-based
training of the full model is out of budget (§5).

## 1. Prior-art check

| Claim | Prior art | Verdict |
|---|---|---|
| BG over internal coordinates incl. bond lengths/angles | Noé et al. 2019: Cartesian training distorted bonds/angles, and normalised internal coordinates fixed it [E-JOINT-110] | **PRIOR ART** |
| Transferable BG (zero-shot to unseen molecules) | TBG, dipeptides, Cartesian CNF [E-JOINT-111] | **PRIOR ART** |
| System-agnostic BG over torsions given fixed L | TD §3.6/§4.5 [E-JOINT-045, -046] | **PRIOR ART** (our baseline) |
| Cheap approximate likelihoods for BGs | BoltzNCE (learned EBM likelihood, ≈ 100× faster) [E-JOINT-113] | **PRIOR ART** (a fallback, but it breaks "exact") |
| Coarse-to-fine / multi-scale BG | dual-scale FM BG for molecular clusters (abstract only) [E-JOINT-096]; CG Boltzmann generators [E-JOINT-114] | **PRIOR ART** (other systems) |
| **Transferable BG over p(L, τ) for drug-like molecules with exact likelihoods, evaluated on GEOM** | gap stated by TBG [E-JOINT-111] and Corso [E-JOINT-112]; no paper found (`abs:"Boltzmann generator" AND abs:torsion` returns nothing for 2023–2026) [E-JOINT-079] | **NOVEL; feasibility unproven** |
| Coupled vs decoupled L in a BG (H2 vs H1 below) | none found | **NOVEL** (small, and it pairs with the Exp-1 A6 control) |

## 2. The density in the chosen coordinates, with Jacobian terms

### 2.1 Chart

The variables are:
- n atoms, m rotatable torsions;
- k = 3n − 6 − m local-structure coordinates;
- 3 rotations, with translation removed by centring.

The **anchor** x_A must be a **deterministic function of the graph**. Use ETKDG with a fixed random seed, then a fixed
MMFF optimisation. If the anchor were a random seed, q(x) would be a mixture over seeds, and its density could not be
computed (INFERENCE). Let B_A ∈ R^{3n×k} be an orthonormal basis of range P_L(x_A), the L-subspace of EXP-1 §2.2.
The chart is

  x(τ, d, ω) = C( R(ω) · modify(x_A + B_A d, τ) ),   d ∈ R^k, τ ∈ T^m, ω ∈ so(3),

where:
- `modify` is TD's rigid side rotation (`utils/torsion.py:57-75` [E-JOINT-026]);
- C removes the centre of mass.

The centred configuration space has dimension 3n − 3 = m + k + 3, so **the chart is locally a bijection**. This is
unlike TD's (m+3)-dimensional manifold, which assumes "fixed local structures" (TD App. B.3, PDF p. 16 [E-JOINT-044]).
Validity has to be checked: the chart breaks down if ‖d‖ is large enough to change which atoms the torsions rotate,
or to collapse a bond. The constraint layer of EXP-1 §2.6 bounds d.

### 2.2 Change of variables (extends TD Prop. 3)

With J = [∂x/∂τ, ∂x/∂d, ∂x/∂ω] ∈ R^{3n×(m+k+3)}, evaluated at ω = 0 on the centred conformer, and g = JᵀJ:

  q(x) = q(τ, d) / ( 8π² · √det g ).

TD's Prop. 3 [E-JOINT-044] is the special case k = 0: its g covers m torsions plus 3 rotations, and 8π² is the volume
of SO(3), used to integrate out the uniform rotation. The derivation in TD App. B.3 (PDF p. 16, Eq. 26–30) never uses
the fixed-L assumption except to choose the coordinates, so adding the d columns leaves the argument unchanged
(INFERENCE).

The importance log-weight against the Boltzmann density e^{−E(x)/kT} is

  log w(x) = −E(x)/kT − log q(τ, d) + ½ log det g + const.

Notes on computing J:
- **∂x/∂τ_i**: TD Eq. 9, a centred cross product on the rotating side [E-JOINT-044]. It is already in
  `likelihood.py:68-93` [E-JOINT-028].
- **∂x/∂d_j**: the basis vector b_j carried through the torsion rotations. If an atom on a torsion axis moves, the
  axis also changes. **Compute this column by autograd through a torch `modify_conformer`**, or by central finite
  differences with step 1e-4. Do not attempt a closed form.
- **∂x/∂ω**: x_k × ê, from TD Eq. 10.
- **det g**: for QM9, n ≤ 29, so J is at most 87 × 84. Take an SVD of J or a Cholesky of g; the cost is negligible.

### 2.3 Variant H, a harmonic L factor (recommended first; feasible)

The L directions are stiff. At 300 K, bond fluctuations are a few hundredths of an Å. GEOM conformers are
GFN2-xTB-optimised structures [E-JOINT-134], not thermal samples. So an L density learned from GEOM would have the
wrong width for a Boltzmann target (INFERENCE). Noé et al. found that putting stiff and soft coordinates on a common
scale was essential [E-JOINT-110].

Variant H gives the L factor its thermal width analytically, from the force-field Hessian projected onto the
L-subspace:

  q(d | ·) = N( d*, kT · (B_Aᵀ H B_A)⁻¹ ),   H = ∇²E at the conditioning structure, regularised to be positive definite.

Two orderings are pre-declared:
- **H1, decoupled, in the user's order p_φ(L) p_θ(τ | L).**
  - d* is the projection onto B_A of the MMFF-relaxed anchor, and is independent of τ.
  - p_θ(τ | L) is **TD's torsional BG unchanged**: `dlogp` comes from the ODE with `divergence_full`
    [E-JOINT-028, -043].
- **H2, coupled, p_θ(τ) p_φ(L | τ).**
  - τ comes from the TD / Exp-1 torsion ODE (exact log q(τ) as in TD).
  - d*(τ) is the Exp-1 geometry head's deterministic endpoint (V2 bridge or V3a ODE) for that τ.
  - H is evaluated at modify(x_A + B_A d*, τ).
  - log q(d | τ) is a closed-form Gaussian, and **no divergence over the k L-dimensions is needed**.
  - INFERENCE: q(d | τ) is a valid conditional density only if d*(τ) does not depend on d. That holds because the
    head's endpoint is a function of τ and the anchor alone. A τ-dependent covariance contributes its log-determinant,
    which is computed exactly.
- Pre-declared hypothesis: **ESS(H2) > ESS(H1)**. This is the Boltzmann-side counterpart of Exp-1's coupled-vs-
  decoupled control. *Revised after D-005 (superseded: "A5 vs A6"):* it pairs with **A5 vs A6b**. For the
  comparison to test coupling and not L mismatch, H1's torsion model p_θ(τ | L) must be trained (or BG-fine-tuned)
  on L drawn from H1's own decoupled L distribution, just as H2's is trained on its geometry head's L. A TD model
  trained on RDKit or GT L would fail H1 for the mismatch reason (EXP-1 E-JOINT-010 rev.). It is a hypothesis; the
  current evidence does not predict its sign.

### 2.4 Variant D, a diffusion L factor (evaluation-only feasibility)

The joint model is the V3a model of EXP-1 §2.3, run with the anchor x_A. The prior is Unif(T^m) × N(0, σ_L,max² I_k),
which is exact. The joint probability-flow ODE gives

  log q₀(τ, d) = log p_T(τ_T, d_T) − ∫₀^T [ ½ g_τ²(t) ∇_τ·s_τ + ½ g_L²(t) ∇_d·s_d ] dt

(TD Eq. 7 extended to the product space [E-JOINT-043, -072]).

The divergence must be computed **exactly**, not with Hutchinson's estimator. Hutchinson gives a biased estimate of
q, which is unusable for importance weights. TD made exactly this choice [E-JOINT-043]. The cost is m + k + 1 network
evaluations per ODE step instead of m + 1.

## 3. How TD's likelihood code extends

| Location | Today | Change (in `flexitors/`, TD file untouched) |
|---|---|---|
| `torsional-diffusion/diffusion/likelihood.py:24-39` `divergence_full` | perturbs each torsion by eps and differences `edge_pred` | `flexitors/likelihood/divergence.py`: loop over m torsions **and** k L basis directions. Perturb x by eps·b_j (through the chart), and difference `l_pred · b_j`. Batch the m + k perturbations as one batch of copies instead of m sequential calls. |
| `likelihood.py:68-98` `dx_dtau`, `log_det_jac` | builds m rows, removes rigid motion, sums log singular values | `flexitors/likelihood/jacobian.py`: add the k columns ∂x/∂d (autograd through a torch `modify`) and compute ½ log det g in one SVD of the full J. Golden test: with k = 0 it must equal `log_det_jac − (−½ log det I)` up to TD's constant. |
| `likelihood.py:114-122` `populate_likelihood` | `euclidean_dlogp = dlogp − ½ log|det I| − log_det_jac` | add `+ log q(d | ·)` (H) or the d-part of the ODE integral (D) to `dlogp`, and replace the Jacobian term by ½ log det g of §2.2 |
| `diffusion/sampling.py:198-202` ODE branch | accumulates `−½ g² eps div` for torsions | `flexitors/sampling/joint_sampler.py` ODE mode: accumulate both factors' terms with their own g |
| `utils/boltzmann.py:22-53` `BoltzmannResampler.resample` | MMFF94s energy, `logweights = −E/kT − euclidean_dlogp`, `ess = 1/Σw²` | reuse unchanged after `populate_likelihood` is swapped; add a GFN2-xTB energy option through `utils/xtb.py` |
| `utils/boltzmann.py:56-125` `BaselineResampler` | AIS with Metropolis–Hastings torsion moves | add a **Cartesian** AIS / MH baseline: small isotropic moves on all atoms. This is the only baseline whose target is the full Euclidean distribution. |
| `test_boltzmann.py:30-37` | DRUGS, 3–7 rotatable bonds, first-conformer L | `flexitors/eval/boltzmann_eval.py`: QM9 test molecules with m ≥ 1, deterministic anchor, strata by m |

## 4. ESS evaluation

- **Metric.** As in TD Table 4: ESS = (Σ w)² / Σ w² out of K = 32 samples per molecule [E-JOINT-029, -046]. Also
  report:
  - ESS/K;
  - the variance of log w;
  - the share of total weight in the top sample;
  - a molecule-bootstrap 95 % CI.

  Temperatures are 300, 500 and 1000 K (TD's set). kT uses `boltzmann.py:24`.
- **Energy.** MMFF94s is primary, for comparability with TD (`likelihood.py:19-21`). GFN2-xTB single point (xtb 6.7.1)
  is secondary because it is GEOM's reference level [E-JOINT-134].
- **Targets, kept separate.** This matters because the models are otherwise not comparable.
  - **Full Euclidean target** e^{−E(x)/kT} on centred conformers. H1, H2, D and Cartesian AIS can all be scored on
    it.
  - TD's BG is supported on an (m+3)-dimensional manifold, so its q(x) is singular against this target. It **cannot**
    be scored here; reporting an ESS for it would be wrong.
  - **L₀-conditional target** e^{−E(modify(x_A, τ))/kT} over τ. TD's BG, uniform τ and torsional AIS are scored here.
    This reproduces TD Table 4's setting on QM9 and is a sanity reference only.
- **Baselines on the full target.**
  - B-unif: uniform τ, d ~ the harmonic Gaussian around the MMFF-relaxed anchor (H1 without a learned torsion model).
  - B-AIS: Cartesian AIS with 5 / 20 / 100 MH steps, the variance tuned as TD did for torsions [E-JOINT-046].
- **Strata.** Report by m (1, 2, 3, ≥ 4) and by ring system (none / isolated / fused-bridged-spiro).
  Pre-declared expectation, from [E-JOINT-114]: ESS falls with dimension and with ring complexity.

## 5. Feasibility on QM9 with our compute

Throughput basis: a TD QM9 test run (1000 molecules, ≈ 2 × n_GT samples, 20 steps) takes 15–25 min on one RTX 3090
[E-JOINT-008]. That is O(5 × 10⁵) network evaluations, so ≈ 2–3 × 10⁴ NFE per minute (INFERENCE, order of magnitude).
QM9 has n ≤ 29 atoms, so k = 3n − 6 − m ≈ 40–80.

| Variant | NFE per sample (20 ODE steps) | 100 molecules × 32 samples × 3 T | 1000 molecules | Feasible? |
|---|---|---|---|---|
| TD BG (L₀-conditional) | 20 × (m + 1) ≈ 60–80 | < 0.5 GPU-h | ≈ 3 GPU-h | yes |
| **H1 / H2** | as TD + the Exp-1 geometry pass (≈ 20) + an MMFF Hessian on CPU (≈ 2·3n gradient calls, ms) | **≈ 0.5–1 GPU-h + < 1 CPU-h** | ≈ 5–10 GPU-h | **yes** |
| D (exact divergence), separate sampling per temperature | 20 × (m + k + 1) ≈ 900–1700 | **≈ 4.8–13.6 GPU-h** (8.6 × 10⁶ – 1.6 × 10⁷ NFE) | **≈ 48–136 GPU-h** | eval on ≤ 100 molecules only (revised after D-007; superseded "3–6 / 30–60") |
| D, one sample set reweighted at all 3 temperatures (q(x) is not temperature-conditioned, so only E/kT changes) | same | ≈ 1.6–4.5 GPU-h (2.9–5.4 × 10⁶ NFE) | ≈ 16–45 GPU-h | **planned setting**; per-temperature sampling only for temperature-trained models (TD BG) |
| Energy-based training, TD Alg. 1 [E-JOINT-045], full train set, H | ≈ 10⁸ NFE per epoch (≈ 35–45 GPU-h per epoch) | — | — | **no** |
| Energy-based training, D | ≈ 10⁹ NFE per epoch | — | — | **no** |
| Energy-based training, H, on a 1000-molecule subset, 5-step resampler | ≈ 6 × 10⁵ NFE per epoch (≈ 0.4 GPU-h) | 50 epochs ≈ 20 GPU-h | — | optional, last |

**Total ask (revised after D-007):** ≈ 10–15 GPU-h for H1, H2, the TD BG reference and the baselines on 100–1000
molecules, plus ≈ 1.6–4.5 GPU-h for a D feasibility point on 100 molecules with shared samples (ceiling 13.6 GPU-h if
per-temperature sampling is needed). Superseded: "≈ 5 GPU-h for a D point". H1 now needs a torsion model trained on
H1's own L (D-005), which is one extra 100-epoch run (≈ 12–18 GPU-h) unless an Exp-1 A6b checkpoint is reused. The
throughput basis (2–3 × 10⁴ NFE/min) is inferred from TD test-run wall-clock, not measured for exact-divergence
batches; a 10-molecule timing run precedes E4-M3. Everything depends on Exp-1 checkpoints (A5 or A8 for H2; A6b for H1), so no new training is needed beyond them
except the optional energy-trained H.

**Milestones.**
- **E4-M0 (CPU).** Implement the §2.2 Jacobian and the golden test against TD's `log_det_jac` at k = 0. Then a
  closed-form check: for a toy molecule whose E is exactly harmonic in d, H1 with the correct Hessian must give
  ESS/K ≈ 1 when τ is held fixed.
- **E4-M1.** Run H1 vs TD BG vs B-unif on 100 QM9 molecules at 300/500/1000 K.
- **E4-M2.** Run H2 (needs Exp-1 A5) vs H1. This is the coupling hypothesis.
- **E4-M3.** One D point on 100 molecules, used to report the exact-likelihood cost against ESS.

**Tests.** The k = 0 equivalence with TD; finite-difference check of the J columns; the harmonic-toy ESS; parity
(q(x) = q(−x) within numerical error for an achiral molecule); and a check that ½ log det g is invariant to the choice
of orthonormal basis B_A.

## 6. Risks

1. **ESS collapse with dimension** [E-JOINT-114]. This is the expected outcome for D. H mitigates it only to the
   extent that the true L distribution is harmonic around one minimum per torsional state. Ring flips and other
   anharmonic L basins are **not** covered by H. For ring molecules, D or a pucker-mixture is needed.
2. **The anchored chart breaks down** for large d (bond collapse, or the rotating side changes). It must be bounded and
   checked, and samples outside it are counted.
3. **Hessian conditioning.** MMFF Hessians can have soft or negative modes away from minima. H needs eigenvalue
   flooring, and the floor biases the weights. Report sensitivity to the floor.
4. **Energy level.** MMFF and GFN2-xTB Boltzmann targets differ. Choosing MMFF for TD comparability means the BG is
   not at GEOM's reference level.
5. **Comparability trap.** TD's BG and the full-space models have different targets (§4). A single "ESS improved"
   headline would be invalid.
6. **Novelty is narrow.** BGs over internal coordinates and transferable BGs already exist [E-JOINT-110, -111]. The
   contribution is the p(L, τ) factorisation on drug-like molecules plus the coupled-vs-decoupled comparison.

## 7. Novelty verdict

- **PRIOR ART:** BGs over bonds, angles and torsions (Noé 2019); transferable BGs (TBG 2024); torsional BGs (TD 2022);
  approximate-likelihood BGs (BoltzNCE 2025).
- **NOVEL, narrowly:**
  1. a transferable BG over p(L, τ) for drug-like molecules with exact likelihoods and the extended Prop.-3 Jacobian
     on a deterministic-anchor chart;
  2. the coupled (H2) vs decoupled (H1) ESS comparison.
- **Expected outcome:** a feasibility section with a cost-vs-ESS table, not a headline result. Promote it only if
  H2 ≫ H1 on QM9 at 300 K with non-overlapping CIs.

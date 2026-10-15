# C-ARCH-01: FlexiTors-Bridge — torsional diffusion jointly with a learned RDKit-L → reference-L bridge in one TD network

## 1. Idea (scout)
Add to TD a second generative factor: a Cartesian "L bridge" that moves the RDKit seed's local structure (bonds,
angles, ring puckers) towards reference-level L, trained by flow matching on the round-2 paired cache (matched RDKit L
↔ own GT L), and run jointly with torsional diffusion in the same score network (product space, independent clocks σ
for torsions and λ for L). Then fine-tune the torsion head on the bridge's own outputs (DAgger), because every model
trained on true/mixed L so far gets worse on improved-but-imperfect L.

## 2. Where it comes from (scout)
**Mechanism, with the exact math of each source.**
- Torsion factor = TD unchanged: wrapped-normal kernel on T^m, sampled as an unwrapped normal mod 2π, DSM loss with
  precomputed scores; one parity-odd pseudoscalar per rotatable bond [E-ARCH-031, E-ARCH-022, E-ARCH-020].
- Product-space rule: forward processes run independently per factor and each factor's score/velocity is regressed
  independently [E-ARCH-032]; flow matching on such products is simulation-free [E-ARCH-061].
- L factor = stochastic-interpolant flow matching as in ET-Flow (Eq. 2–5): x_t = t x1 + (1 − t) x0 + σ√(t(1−t)) z,
  target v = x1 − x0 + (1 − 2t)/(2√(t(1−t))) σ z [E-ARCH-050]; FM-refiner shows such a flow can start from a
  data-centred state instead of noise [E-ARCH-054, E-GEOM-029]; EBD starts from an RDKit-derived coarse structure
  instead of noise [E-ARCH-056]; the source distribution matters (Gaussian vs harmonic prior: DRUGS AMR-R 0.640 vs
  0.557) [E-ARCH-051, E-ARCH-060].
- Proposed instantiation (INFERENCE, assembled from the above and our code):
  - Training sample: conformer k, x_R = `data.pos[k]` (matched RDKit L), x_G = `data.gt_pos[k]` (own GT L, Kabsch-aligned,
    terminal-relabelled) [E-ARCH-023]; σ ~ log-U (TD), λ ~ U[0,1] independently [E-ARCH-032].
  - Apply the same torsion update Δτ ~ wrapped N(0, σ²) to both endpoints: x_R' = modify(x_R, Δτ), x_G' = modify(x_G, Δτ).
  - Input x_λ = interp(x_R', x_G', λ) + γ(λ) z with `interp_x_torch` (keeps terminal bond lengths) and
    γ(λ) = σ_b √(λ(1−λ)); λ fed through the existing S4 λ embedding [E-ARCH-023].
  - Targets: torsion score of Δτ (TD) and per-atom velocity u = x_G' − x_R' + γ'(λ) z, read from a new `1o` per-atom head
    on the existing atom trunk [E-ARCH-021]. Loss L = L_τ + w‖v_θ − u‖².
  - Sampling: ETKDG seed at λ = 0, torsions from TD's uniform prior; per step a TD reverse-SDE torsion step
    [E-ARCH-024] and an Euler L step x ← x + Δλ v_θ (schedules: joint, L-first, L-last as a pre-declared ablation).
  - DAgger stage: roll the L head out on training molecules, conformer-match the rolled-out L to GT torsions, and
    fine-tune the torsion head at λ = 1 on it [E-ARCH-062, E-ARCH-006].
- Design motivation from prior proposals: Corso's thesis says TD's error is bounded by the cheminformatics L and
  proposes using the torsional manifold as a soft constraint on a full-dimensional diffusion [E-ARCH-033]; TD itself
  takes p(L) from RDKit [E-ARCH-030]; p(τ | L) depends on L [E-ARCH-048].

**What is new (Q1, to be checked by the novelty red-teamer).**
- vs TD: L is generated, not frozen [E-ARCH-030].
- vs FM-refiner: source is the RDKit-shaped L error (paired bridge), not Gaussian-perturbed data; runs inside
  torsional diffusion with a torsion factor rather than as a post-hoc all-coordinate refiner; FM-refiner never refined
  TD [E-ARCH-054, E-ARCH-055].
- vs ET-Flow / DiTMC: keeps the torus factor for large torsion moves; the Cartesian factor only moves L-scale
  displacements [E-ARCH-050, E-ARCH-052].
- vs GO-Flow: no internal-coordinate Jacobians and ring closure is automatic in Cartesians (GO-Flow does not address
  closure) [E-ARCH-035, E-ARCH-037]; TD protocol, not GD-P [E-ARCH-036].
- vs EBD: atom-level RDKit L bridged to reference L with torsion diffusion, not fragment-level blurring [E-ARCH-056].
- vs our S4: S4 conditions on λ but never moves L [E-ARCH-007].
- Search found no joint torsion + learned-L model for organic molecules (2024–2026) [E-ARCH-072]. UNVERIFIED: generic
  paired "bridge matching" precedents outside chemistry (e.g. image-to-image Schrödinger bridges) are not in our store;
  the claim must be scoped to conformer generation.

## 3. Why it could matter here (scout)
- Targets the error source: ring geometry is the lever with the model in the loop (CTRL λ0 0.1966 → A5ring 0.1186)
  [E-ARCH-001]; the ring-pucker tail separates RDKit/MMFF (49.6 % / 35.1 % > 10°) from λ ≥ 0.75 quality (0.02 %)
  [E-ARCH-002]; bond/angle size matters less than puckers [E-ARCH-012]. A Cartesian bridge fixes rings of every class,
  including fused/bridged systems that a Cremer–Pople factor cannot represent easily [E-ARCH-003].
- Addresses the round-2 failure mode directly: S3/B1 lose on MMFF L (+0.039 / +0.080) [E-ARCH-005]; the only learned
  real-use gain is training on the deployed L source (S5 −0.0041) [E-ARCH-006]; a λ-aware model told the right level
  beats CTRL on λ = 0.5 blends by −0.0090 [E-ARCH-007]. In the bridge the torsion head always knows λ (its own clock)
  and, after DAgger, has seen the bridge's own output.
- Avoids the conformer-identity leak by construction: L and τ are co-generated, nothing comes from test GT
  [E-ARCH-009]; an independent reference-level L would leave only a small floor (ring molecules 0.117 → 0.032 Å)
  [E-ARCH-010].
- Predicted effect (INFERENCE from the λ curve [E-ARCH-008] and the rigid stratum [E-ARCH-004]): if the bridge reaches
  λ ≈ 0.5-quality L, CTRL-type AMR-R ≈ 0.13–0.14, with λ-awareness ≈ −0.009 better; at λ ≈ 0.75-quality ≈ 0.11–0.12;
  if it only reaches MMFF quality, ≈ 0.150 (no gain over today's best real use). Range 0.10–0.15 vs TD 0.178 and best
  real use 0.150. It cannot be expected to reach ET-Flow / DiTMC (0.068–0.083) [E-ARCH-052] unless the bridge is close
  to own-conformer L (CTRL 0.082, B1 0.021 there) [E-ARCH-008].
- DRUGS: TD 0.582 vs Cartesian frontier 0.375–0.44 [E-ARCH-053]; the same code runs on DRUGS (the paired cache needs
  rebuilding there).

## 4. Assumptions that may not transfer (scout)
- Multimodality: the velocity from an RDKit seed to its own GT L is ambiguous when GT puckers differ between conformers;
  the interpolant noise γ must be large enough (≈ ring-flip scale, 0.3–0.5 Å) to split paths, which ET-Flow and
  FM-refiner use for other reasons [E-ARCH-050, E-ARCH-054]. Too small → averaged, distorted rings.
- The λ blends are not clean physical states (≈ 10 % of round-2 λ slots reused another conformer's blend; pair_ok drops
  ≈ 12 % of pairs) [E-ARCH-007, E-ARCH-008].
- Exposure bias: the bridge's endpoint is not true L; without DAgger the torsion head is in the S3/B1 regime that lost
  on MMFF [E-ARCH-005]. DAgger needs conformer matching on rolled-out L (CPU-heavy).
- Overlap between factors: the bridge velocity also changes torsions a little; disentanglement as in DiffDock is not
  enforced [E-ARCH-032].
- No likelihood for L (unlike a CP flow) [E-ARCH-034].

## 5. Minimal experiment (scout)
- **Gate 0 (≈ 5 GPU-h + CPU):** train the joint model for 20–30 epochs on QM9 (1 seed); evaluate the L bridge alone on
  the rigid stratum (no torsions involved, model-free read-out) [E-ARCH-004] and the ring-dihedral > 10° share
  [E-ARCH-002]. GO if rigid AMR-R ≤ 0.10 (MMFF 0.128, ETKDG 0.152) and tail ≤ 20 %.
- **Main (if GO):** arms (3 seeds, 100 epochs, ETKDG seeds at test, no oracle): CTRL_rematch; CTRL + MMFF; S5;
  Bridge-joint; Bridge-joint + DAgger; schedule ablation (L-first / L-last, 1 seed); a Gaussian-source refiner contrast
  is C-ARCH-03. Primary AMR-R and COV-R@0.5; key secondary COV-R@0.1 on the success intersection; paired bootstrap,
  Holm. Expected: Bridge + DAgger < CTRL + MMFF by ≥ 0.01 Å.
- Cost: ≈ 250–400 code lines + tests; main ≈ 5 × 3 × 13 GPU-h ≈ 200 GPU-h worst case (≈ 90–110 GPU-h if only the
  two bridge arms are new) + DAgger rollouts (≈ 25–40 GPU-h packed inference on the training set) + conformer-matching
  CPU. DRUGS ≈ 10× training.

## 6. Scout's own call (scout)
Worth trying: YES (gated by Gate 0) — the only option that learns L for every ring class, co-generates L with τ (no
leak), reuses round-2 code, and has a defensible novelty claim; DAgger is mandatory, not optional.

## 7. Code grounding (grounder)
Scout's touchpoints for the grounder to check: `torsional-diffusion/utils/dataset.py:18-34` (interp), `:71-76` (torsion
noise → apply to both endpoints), `:79-99` (paired selection → return both endpoints and λ);
`torsional-diffusion/diffusion/score_model.py:65-79`, `:194-199` (λ embedding exists), `:83-89`, `:141-143` (add a `1o`
per-atom head after the atom layers), `:156-165` (outputs); `torsional-diffusion/utils/training.py:25-31` (add the L
loss); `torsional-diffusion/diffusion/sampling.py:180-183` (constant `l_level` → per-step λ schedule), `:184-235`
(add the L Euler step), `:290-298` (position update); `standardize_confs.py` and `tools/build_paired_pickles.py` for
DAgger re-matching.
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)

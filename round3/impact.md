# Round 3 impact analysis (P3, impact analyst)

Written 2026-10-08 to 2026-10-11 by the impact analyst; cross-read with `grounding.md` done (§5). Scripts and outputs are in `round3/impact_data/` (listed in §1).
All round-2 numbers are PRELIMINARY: they are unpaired means over each run's molecules. No per-molecule round-2 output
is synced, so no paired round-2 statistic exists. S3 seed 1 and S4 seeds 0–2 come from BRIEF.md (22:20 update; in-job,
not synced). "Best non-oracle today" means CTRL_rematch on MMFF-relaxed ETKDG L, 0.1525 Å (3 seeds, 936 molecules,
`cond_means.txt`). Gains are in mÅ of AMR-R (δ = 0.5 Å), where a negative change is better.

## 0. Summary table

| Card | P2 status | Ceiling on our data (ORACLE-bounded) | Expected AMR-R effect (worst) | Detectable? Seeds | GPU-h | Gain per GPU-h | Worth trying |
|---|---|---|---|---|---|---|---|
| C-GEOM-01 xTB seeds | SUPPORTED | Leak-corrected (§3g): CTRL ≈ 0.11–0.145, B1 ≈ 0.19–0.27 with other-conformer L. The leaky `gtL` floors 0.086 / 0.038 need each target's own-basin L | CTRL 0.146 (0.13–0.155; worst ≈ MMFF 0.153). B1 0.10–0.22, central ≈ CTRL: gains only on targets where an xTB seed lands in the target's own basin. S3 0.10–0.16, the most promising arm. *Revised after cross-read* | CTRL vs MMFF: borderline (paired ±0.005). B1/S3 − CTRL: CI ≈ ±0.006 | 1.0–2.4 (+≈ 6 CPU-h) | ≈ 5 (best arm, probability-weighted; was 9) | **YES** (highest decision value) |
| C-GEOM-02 ring templates | SUPPORTED | CTRL ≈ 0.10–0.12 (A5ring-type, ETKDG acyclic); with xTB polish, as C-GEOM-01 (leak-corrected) | CTRL 0.141 vs ETKDG 0.178 (0.108–0.16; worst ≈ ETKDG) | Yes at central value (−0.037 vs ETKDG) | 1.0–1.9 (+2–3 days coding; grounder) | ≈ 6 vs MMFF (≈ 19 vs ETKDG) | **YES** (after C-GEOM-01 CPU step) |
| C-GEOM-03 ET-Flow L | SUPPORTED, D-210 OPEN | TD + CTRL cannot beat raw ET-Flow. B1/S3 can come close only through own-basin L, i.e. by re-using ET-Flow's own conformers | CTRL 0.09–0.12; B1/S3 0.05–0.15 (depends on ET-Flow's basin coverage and noise) | Yes (large effects) | ≤ 1 + 1.0–1.9 (+0.5 day env) | n/a (diagnostic) | **MAYBE**: YES only after a conclusive split check (D-210) |
| C-GEOM-04 learned pucker model | SUPPORTED | If every in-domain bad seed (48 %) were fixed: CTRL ≈ 0.135–0.14 | Small (PuckerFlow's own all-atom gain is 0.01–0.02 Å, on its own benchmark) | – | weeks + training | ≪ 1 | **NO** (this round) |
| C-GEOM-05 ML-potential relaxation | SUPPORTED | ≤ xTB on puckers (local optimiser) | CTRL ≈ C-GEOM-01; B1 likely MMFF-like (DFT minima are not the reference) | – | 1–2 + env | < C-GEOM-01 | **NO** (dominated by xTB) |
| C-GEOM-06 srETKDGv3 seeds | SUPPORTED | Tail-driven, as for C-GEOM-01 | CTRL −0.005 to −0.010 vs v1 (worst +0.005: v1→v3 shift) | Probably not: variants are different embeddings (grounder), MDE ≈ 0.008–0.010 | ≤ 1.0 (CPU-gated) | ≈ 3–5 | **MAYBE** (run the CPU part inside the C-GEOM-01 builder) |
| C-EVAL-01 literature block + "how good must L be" | SUPPORTED | – (analysis) | Already computable (§3f–g): with own-conformer L (leaky), CTRL's floor is 0.086 and B1 reaches 0.073 at tail ≤ ≈ 14 %. With other-conformer L only, CTRL's floor is 0.11–0.145 and B1's 0.19–0.27, so no literature level is reachable. *Revised after cross-read* | n/a | 0 (~2 CPU-h; cluster) | n/a | **YES** |
| C-EVAL-02 local / stratified diagnostics | SUPPORTED | – | Bins already separate L effects from model effects (§3). Random-GT runs of S3/CTRL_rematch + the n_true inversion (§3g) measure each model's own-vs-other-L sensitivity, which decides C-GEOM-01 | n/a | 0.9 (item (d); YES) | n/a | **YES** |
| C-EVAL-03 xTB E_relax metrics | SUPPORTED | – | No AMR effect (a metric); identical for arms that share ETKDG L | n/a | 0 (≈ 8 CPU-h; +0.3–0.5 GPU-h for the validation half, grounder) | n/a | **YES** (xtb 6.7.1 already on gnode118) |
| C-EVAL-04 non-learned baselines | SUPPORTED | – | B-mmff ≈ 0.20–0.21; B-xtb ≈ 0.17–0.21 (rigid stratum ≈ xTB seeds) | Yes vs TD (≥ 0.02) | 0 (B-clust TD arm 2.6) | n/a | **YES** for B-mmff/B-xtb; **MAYBE** for B-clust |
| C-EVAL-05 xTB-after yardstick | SUPPORTED | Rigid stratum = xTB seeds | CTRL + xTB-after 0.13–0.17 (worst A3-like 0.186); COV-R@0.05 ≫ 3.9 % | Yes | 0 (CPU) | n/a | **YES** (yardstick only) |
| C-EVAL-06 DFT re-reference / ensemble properties | SUPPORTED | – | – | – | – | – | **NO** (agree) |
| C-ROBUST-01 factorised error-matched aug. | SUPPORTED | A5ring best today: CTRL 0.119; A5acyc best: B1 0.158 | RDKit/MMFF L ≈ S3 (±0.003): no primary gain. Gain only on partial-L sources | Arm 0: yes. Arm 1 primary: equivalence needs ≥ 3 seeds | Arm 0: 2.6; Arm 1: 26.8 (grounder) | Arm 0 n/a; Arm 1 ≈ 0 on primary | Arm 0 **YES**; Arm 1 **MAYBE** (gated) |
| C-ROBUST-02 blind λ (S4 w/o input) | SUPPORTED | S3 is already "blind" at the endpoints | Blind ≈ S4 ± 0.005; prior P(gate) ≈ 20 % (S4 worse than S3 on true L: 0.043 vs 0.034). The gate is computable now on the cluster (grounder) | Equivalence at ±0.005 needs 3–6 seeds | 26.5 (grounder) | ≈ 0 | **NO** unless the measured gate passes |
| C-ROBUST-03 sampler-matched mixture (S3-MMFF) | SUPPORTED | MMFF L: rigid stratum fixed by L (0.128); only flexible bins can move | 0 to −0.006 vs CTRL + MMFF (worst +0.003); prior P(S5 ≤ 0.150) ≈ 30 %, computable now (grounder) | Needs 3 seeds for −0.005 | 26 (grounder) | ≤ 0.2 | **MAYBE → NO** unless the measured S5 gate passes; keep the recipe for xTB-matched training |
| C-ROBUST-04 L guidance / score interpolation | SUPPORTED | Lower envelope of CTRL and B1 = CTRL on every non-oracle L | 0 (likely worse recall) | – | ~5 | 0 | **NO** (agree) |
| C-ROBUST-05 p_gt sweep / curriculum | SUPPORTED | Headroom 0.003 (RDKit L), 0.012 (true L, ORACLE) | Trades one endpoint for the other | – | 72–96 | ≈ 0.03 | **NO** (agree) |

Priority if the user funds GPU: C-GEOM-01 (with B1, S3 and S4 as well as CTRL; S4 at pre-declared λ = 0 and λ = 1) and
the random-GT runs for S3 and CTRL_rematch (C-EVAL-02 (d), 0.9 GPU-h; they measure the own-vs-other-L sensitivity that decides C-GEOM-01) → C-GEOM-02 → C-ROBUST-01 Arm 0. Before the panel, compute the C-ROBUST-02/03 gates on the cluster (CPU; grounder). The
EVAL cards are CPU-only and should run alongside (C-EVAL-01/02 now, C-EVAL-03/04/05 once the shared xTB runner
and graph filter (grounding.md S-1/S-2) exist; no install is needed).

## 1. Data used and how
- `collect.py` parses every synced `summary.txt` and `breakdown.log` (round 1: `cluster_sync/results/*/*`; round 2:
  `cluster_sync/round2/results/*/*`) into `runs.csv`, `bins.csv` (n_rot_heavy bins 0/1/2/3/4+) and `cond_means.txt`.
- `lerror_profile.py` profiles `l_error.csv` per condition and per n_rot_heavy bin. It gives `lerror_profile.csv`,
  `mol_info.csv` and `.out`. The grouping is V3's (unit = csv row; ring ≥ 4 basis for the tail); my overall numbers
  equal V3's (ETKDG > 10° tail 49.61 %, MMFF 35.06 %).
- `dose_response.py`: a per-bin linear fit on the ORACLE conditions (`dose_response.out`). It is shown for
  completeness; it over-predicts the real ETKDG → MMFF gain (§3), so it is not used for predictions.
- `mixture_model.py`: the two-anchor model used for every L-source prediction (`mixture_model.out`,
  `pred_mixture_grid.csv`).
- `noise_power.py`: seed SDs, paired-CI half-widths and seeds-needed table (`noise_power.out`).
- `ntrue_leak.py` (added after the cross-read): conformer-identity leak of anchor G per n_true bin, and the
  leak-corrected predictions (`ntrue_leak.out`, `pred_ntrue_leak.csv`; §3g).
- `fill_section8.py` / `append_section8.py`: write my card §8 sections (no other section touched).
- **Data defect found (affects the λ curve).** In `L_lam0.25/0.50/0.75`, 1268 of 12,707 slots (10 %, in 258
  molecules) reuse the blended seed of another GT conformer (`*.population.txt`: `replaced_seeds 1268`). For those rows
  `l_error.csv` measures the error against the *source* GT (`k != src_gt_idx`), not the slot's GT. The λ ∈ (0,1) points
  therefore understate the effective L error. They sit above every curve fitted on the clean conditions (rigid bin,
  CTRL: λ 0.75 observed 0.077, clean-fit prediction 0.025). λ is therefore not a clean dose axis. I use the clean
  conditions (ETKDG, MMFF, λ0, λ1, A5ring, A5acyc, noise 0.02/0.04) and the round-1 random-GT-conformer oracle.
- Not possible locally: per-molecule regression of AMR on L error (C-EVAL-01 §5). It needs the cluster's
  `breakdown.csv` / eval pickles, and no GT conformer pickle is synced.

## 2. Noise and detectability (`noise_power.out`)
- Training-seed SD of AMR-R: CTRL family median 0.0009 Å (max 0.0020); B1 median 0.0031 Å. B1 on noisy L is much
  larger: 0.014 Å at noise 0.04 and 0.022 Å at noise 0.02. Sampling-seed SD: 0.0008 Å (released model, R0–R2) and
  0.0026 Å (A1, true L).
- Paired-bootstrap 95 % CI half-widths (molecules resampled, replicates averaged per molecule):
  - same L, different model (B2, B5 vs CTRL_rematch, 3 v 3 seeds): 0.0016–0.0021 Å;
  - same model, different L (A2 vs R0): 0.0049 Å; G1: 0.0037 Å;
  - B1 vs CTRL: 0.006 Å.
  A change of L source moves molecules heterogeneously, so L-source contrasts are about 3× noisier than model contrasts.
- Rules used below:
  - L-source contrast with the same checkpoints: the minimum detectable effect (80 % power) is ≈ 0.007 Å.
  - Model contrast on the same L: the MDE is ≈ 0.003 Å with 3 v 3 CTRL-type seeds, and ≈ 0.006–0.010 Å for B1-type
    models (SD 0.003–0.005 needs k ≥ 4–16 seeds for Δ = 0.005).
  - From the table: SD 0.0015 needs k ≥ 4 seeds for Δ 0.003 and k ≥ 2 for Δ 0.005; SD 0.003 needs k ≥ 6 for Δ 0.005.

## 3. Dose-response: measured L error → AMR-R
**3a. Rigid molecules are a model-free L readout.** Bin n_rot_heavy = 0 holds 284 of 935 molecules, and 316 of the 318
rigid molecules in `l_error.csv` contain a ring of ≥ 4 atoms. Its MAT-R is the same for CTRL and B1 within 0.006 Å on
every L (`cond_means.txt`):

| test-time L | rigid tail > 10° | rigid heavy-angle err (°) | rigid AMR-R CTRL / B1 |
|---|---|---|---|
| ETKDG | 57.3 % | 3.68 | 0.152 / 0.155 |
| MMFF | 34.8 % | 2.74 | 0.128 / 0.127 |
| noise 0.02 (ORACLE) | 0 % | 1.62 | 0.032 / 0.029 |
| noise 0.04 (ORACLE) | 0.9 % | 3.23 | 0.058 / 0.057 |
| A5ring (ORACLE) | 0 % | 1.83 | 0.043 / 0.048 |
| random-GT-conformer L (r1 `gtL`, ORACLE) | 0 % | GT-GT | 0.018 / 0.011 |

With correct puckers, even 3° / 0.05 Å random error costs rigid molecules only 0.05 Å. ETKDG's 0.152 is mostly the
pucker tail.

**3b. Linear fit on ORACLE conditions fails on real samplers.** The per-bin fit AMR = a + bA·angle + bT·tail
(`dose_response.out`) fits the clean ORACLE conditions (CTRL in-sample RMSE 0.003–0.008 per bin). Anchored on ETKDG,
it predicts MMFF at 0.146 (CTRL; observed 0.1525) and 0.184 (B1; observed 0.232). MMFF's bond/angle gains do not pay
out as the oracles suggest, so this fit is not used for predictions.

**3c. Ring-tail mixture model (used for predictions; `mixture_model.out`).** Per bin, AMR is interpolated linearly in
the ring-tail share between two anchors:
- **G**: L that is physically consistent and at reference level with correct puckers, i.e. true L of a *random* GT
  conformer (round-1 in-job `gtL`; no conformer-identity leak, unlike `gtLcycle`). *Revised after cross-read: the no-leak claim is wrong. `gtL` draws with replacement from the molecule's own conformers, so ≈ 92 % of targets still see their own L; see §3g.*
- **H**: A5acyc, i.e. RDKit rings with GT acyclic L.

Bins are reweighted to the ETKDG population.
- CTRL (out-of-sample check on the two non-oracle points): ETKDG predicted 0.1782 vs observed 0.1777; MMFF predicted
  0.1446 vs observed 0.1525. CTRL's AMR on real samplers is therefore almost entirely a function of the ring tail:
  **CTRL ≈ 0.086 + 0.20 · X** (X = > 10° share on the 49.6 % basis). It realises about 76 % of the predicted MMFF gain,
  so I add the MMFF residual (+0.008) to CTRL predictions as a calibration.
- B1: the same line, B1 ≈ 0.038 + 0.25 · X, holds only if the acyclic L is reference-like (B1 tolerates another GT
  conformer's L: 0.037). *Revised after cross-read: it does not. 0.037 is mostly own-conformer L; with other conformers' L, B1 ≈ 0.16–0.35 (§3g).* ETKDG/MMFF acyclic L is not reference-like, and the line misses them badly (predicted 0.153 /
  0.112 vs observed 0.236 / 0.232). Whether an L source's acyclic part is "reference-like" is the single most
  important unknown in round 3.

**3d. Predicted AMR-R on a new L source with reference-like acyclic L (xTB-type), by ring tail X.** *Revised after cross-read:* this table uses the leaky anchor G, i.e. it assumes every target gets its own conformer's L. That is optimistic, strongly for B1 and mildly for CTRL; §3g has the leak-corrected values. Table basis: 49.6 % tail basis; ETKDG
population; CTRL includes the +0.008 calibration; B1 is the reference-like branch only.

| X (> 10° share) | 49.6 % (ETKDG) | 35.1 % (MMFF) | 30 % | 25 % | 20 % | 15 % | 10 % | 5 % | 0 % |
|---|---|---|---|---|---|---|---|---|---|
| CTRL | 0.177 | 0.153 | 0.144 | 0.136 | 0.128 | 0.119 | 0.111 | 0.103 | 0.094 |
| B1 (reference-like) | 0.143 | 0.113 | 0.102 | 0.091 | 0.080 | 0.070 | 0.059 | 0.049 | 0.038 |
| S3 (bracket) | 0.15–0.18 | 0.12–0.156 | 0.11–0.147 | 0.10–0.139 | 0.09–0.131 | 0.08–0.122 | 0.07–0.114 | 0.06–0.106 | 0.045–0.097 |
| rigid stratum (model-free) | 0.15 | 0.11–0.13 | 0.10–0.11 | 0.09–0.10 | 0.07–0.09 | 0.06–0.07 | 0.05–0.06 | 0.03–0.05 | 0.02–0.03 |

- How the S3 bracket is built. On true L, S3 sits 20 % of the way from B1 to CTRL (0.034 vs 0.022 / 0.085). On
  RDKit L it is CTRL + 0.003. The lower edge applies if S3 behaves B1-like on reference-like L; the upper edge is
  CTRL + 0.003. S4 (λ = 1 on true L 0.041–0.044) should sit near the same bracket.
- Reading of the table. If xTB fixes no more puckers than MMFF did, CTRL gains nothing over MMFF (0.153 vs 0.1525).
  If xTB's acyclic L is reference-like, B1 beats CTRL at every tail level, by 0.026–0.048 Å.

**3e. Bonds and angles alone (CTRL).** With correct puckers, random bond/angle error of 0.027 Å / 1.6° costs CTRL
+0.016 (noise 0.02 vs λ1, cycled). Fixing MMFF-level bonds/angles is therefore worth at most ≈ 0.008 Å to CTRL, and
I give CTRL ranges as "table value − 0.008 to table value".

**3f. Literature levels (C-EVAL-01).** *Revised after cross-read:* the thresholds below hold only if the L source gives each target its own conformer's L (leaky anchor). With other-conformer L only, CTRL's floor is 0.11–0.145 and B1's 0.19–0.27, so no checkpoint reaches any literature level (§3g). With own-conformer reference-like acyclic L:
- CTRL never reaches the published 0.073 or the 0.083 re-run level (floor 0.086 on the ETKDG population).
- B1 reaches 0.073 at X ≤ ≈ 14 %, 0.083 at ≤ 18 %, DMT-B 0.090 at ≤ 21 %, and MCF-B 0.103 at ≤ 26 %. It beats
  today's best non-oracle 0.1525 at X ≤ 46 %.

These thresholds are converted from the bin-weighted basis by the MMFF ratio 1.166.

**3g. Conformer-identity leak in anchor G (added after the cross-read; `impact_data/ntrue_leak.py`, `.out`).**
The grounder showed that round-1 `gtL` draws each seed with replacement from the molecule's *own* GT conformers
(`diffusion/sampling.py:86`). So 25 % of molecules (single conformer) get exactly the cycled seeds, and about 92 % of
target conformers have their own L among the draws. Anchor G is therefore mostly leaky.
- **Method.** In each n_true bin, gtL − gtLcycle ≈ P0 × (A_other − A_own). P0 = (1 − 1/n)^{2n} is the chance that no
  draw is the target's own L. Inverting this gives A_other, the AMR of a target whose seeds carry only *other*
  conformers' reference L.
- **Sanity check.** Bin 1 has Δ = 0.0000 for both models, as it must (identical seeds).

| n_true bin | 2–3 | 4–10 | 11–30 | > 30 |
|---|---|---|---|---|
| P0 | 0.072 | 0.111 | 0.127 | 0.133 |
| B1: cycled (own L) → implied other-conformer L | 0.016 → **0.35** | 0.025 → **0.23** | 0.038 → **0.16** | 0.046 → **0.17** |
| CTRL: cycled → implied other-conformer L | 0.069 → 0.17 | 0.105 → 0.10 | 0.130 → 0.16 | 0.136 → 0.15 |
| for comparison, on ETKDG L: B1 / CTRL | 0.236 / 0.189 | 0.267 / 0.186 | 0.257 / 0.185 | 0.291 / 0.182 |

- **Noise.** Dividing by P0 amplifies seed noise about 8–14×: ±0.03–0.04 for B1 and ±0.015–0.03 for CTRL per bin.
  The B1 conclusion is robust to this; the CTRL values are only indicative.
- **Reading.** B1's true-L advantage is almost entirely conformer identity. Given reference-level L from a *different*
  conformer of the same molecule, B1 does about as badly as on RDKit L. CTRL loses about 0.03 on average, within
  noise.
- **Leak-corrected floors** (X = 0, ETKDG population):
  - CTRL 0.111 (single-conformer molecules keep own L, which is right for rigid ones) to 0.145 (pessimistic), against
    0.086 leaky.
  - B1 0.19 to 0.27, against 0.038 leaky.
  - On the multi-conformer subset (n_true ≥ 2): CTRL 0.105 leaky vs 0.138 leak-free; B1 0.048 vs 0.249.
- **What this means for a non-oracle L source.**
  - An xTB-relaxed seed that lands in the target conformer's own basin (pucker *and* torsions) reproduces that
    conformer's GT L legitimately: GT conformers do not move under xtb 6.7.1 (grounder probe, 10/10). This is
    non-oracle own L.
  - Seeds in other basins give other-conformer L.
  - For B1, S3 and S4 (λ = 1) on xTB L, the deciding quantity is therefore s, the share of GT conformers with an
    own-basin xTB seed. The `--no_model --no_random` seed run of the grounder's S-3 tool measures it as the seeds'
    COV-R at 0.05–0.1 Å over all molecules, a cheap CPU number to add to the gate.
  - Rough model: B1(xTB) ≈ 0.3 · rigid part (tail-driven, ≈ 0.11) + 0.7 · [s · 0.04 + (1 − s) · 0.25]. That gives
    0.10 (s = 0.75), 0.14 (s = 0.55) and 0.17 (s = 0.35), against CTRL ≈ 0.146.
  - S3 is CTRL-like on non-own L (it matches CTRL on RDKit L, 0.182 vs 0.179) and near-B1 on own L (0.034). It should
    take the better of both: 0.10–0.16, central ≈ 0.13. It becomes the most promising arm, but its own-vs-other-L
    response is unmeasured (C-EVAL-02 (d) random-GT runs).
- **CTRL model check by n_true bins.** Stratifying the §3c model by n_true instead of n_rot_heavy predicts ETKDG well
  (0.176 vs 0.178) but MMFF worse (0.130 vs 0.152). Real samplers realise only 57–76 % of the predicted tail gain, and
  I use that range for CTRL below.

## 4. Per-card notes
Each card's §8 holds the full reasoning chain. The one-line version is in §0. Cross-cutting points:
1. **The decisive unknown is not the size of the L error but whether the acyclic L is reference-like.** B1 tolerates
   another conformer's GT L (0.037). It fails on MMFF L (0.232) even though MMFF's bond/angle errors are smaller than
   noise 0.02 (0.117). C-GEOM-01 tests this at ≈ 2 GPU-h and should include B1 and S3/S4, not only CTRL.
   *Revised after cross-read:* the first sentence is superseded. B1 does not tolerate another conformer's GT L (§3g: ≈ 0.16–0.35).
   The decisive unknown is s, the share of target conformers for which the L source puts a seed in the target's own
   basin. It is measurable on CPU as the seeds' COV-R@0.05–0.1 (§5 A).
2. **For CTRL, only the ring tail matters** (§3c). Any L-source card that does not lower the > 10° share below
   MMFF's 35 % will not beat CTRL + MMFF by a detectable margin.
3. **The CPU gate in C-GEOM-01/02/06 needs one change.** Its rigid-set "≥ 0.010 Å vs MMFF" rule can pass on
   bond/angle improvement alone (oracle slope 0.016 Å per degree: MMFF 2.74° → xTB ≈ 1.2° gives ≈ 0.025 Å on rigid
   molecules) without any pucker change. The tail share and pucker coverage are the informative gate metrics. Because
   the GPU step costs ≤ 3 GPU-h, gating it is barely worth the decision cost for C-GEOM-01: run CTRL/B1/S3 on xTB seeds
   whenever the seeds exist.
4. **Round-2 caveats.** All S1 numbers are unpaired and on different populations (936 / 955 / 906 / 996). The
   mixture model mixes the 996-molecule round-1 in-job runs (anchor G) with the 955-molecule round-2 runs (anchor H),
   and bin reweighting only partly corrects for this. The MMFF residual (0.008) is my estimate of the model error for
   CTRL; for B1 there is no non-oracle validation point.

## 5. Cross-read with code grounder
Round 1 (2026-10-11, before the grounder's ROBUST sections existed) is superseded by this round 2. Below I answer
`grounding.md` §5 (5.1 and 5.2) and its per-card §7 points point by point, now including the C-ROBUST sections. Card §8
sections that changed carry "Revised after cross-read" with the old text kept.

**A. The grounder's main disagreement: anchor G leaks conformer identity (5.2.1). AGREE; it changes my B1/S3 numbers.**
- Their facts reproduce exactly:
  - 25.4 % single-conformer molecules on the 996-molecule set (23.9 % on the 955 λ set);
  - 92.2 % molecule-mean share of conformers with their own L among the draws;
  - gtL − gtLcycle = 0.0000 in the n_true = 1 bin for both models (`ntrue_leak.out`).
- Excluding single-conformer molecules does not remove the leak, because 92 % of the remaining targets still see their
  own L. I therefore inverted the leak per bin (§3g) instead of only dropping n_true = 1.
- Corrected values:

  | floor (X = 0) | CTRL | B1 |
  |---|---|---|
  | leaky anchor, all molecules | 0.086 | 0.038 |
  | leak-free, all molecules | 0.111–0.145 | 0.19–0.27 |
  | multi-conformer subset, leaky → leak-free | 0.105 → 0.138 | 0.048 → 0.249 |

- B1 with other conformers' exact GT L ≈ B1 on RDKit L. So the "reference-like acyclic L" question in my §3c is the
  wrong question. The right one is whether the L source puts some seed in each target's own basin (s, §3g).
- Corrected predictions:
  - C-GEOM-01: CTRL 0.146 (0.13–0.155), unchanged in substance. B1 changes from bimodal 0.10–0.12 / 0.20–0.23 to
    0.10–0.22, central ≈ CTRL. S3 is 0.10–0.16, central ≈ 0.13, and is now the most promising arm. The best-arm gain
    drops from ≈ 9 to ≈ 5 mÅ per GPU-h. The call stays YES: decision value is high, and the GPU cost is ≤ 2.4 GPU-h.
  - C-EVAL-01: the literature thresholds hold only with own-conformer L.
  - C-GEOM-03: ET-Flow's samples are conformer-specific, so s may be high. B1/S3 could then do well, but only by
    re-using ET-Flow's own conformers.
- New, cheap gate metric (no new code): s = COV-R@0.05–0.1 of the xTB seeds themselves, from the grounder's S-3
  `--no_model --no_random` run over all molecules, not just rigid ones.

**B. Measured facts that change my estimates (grounder 2.1–2.3).**
1. **xtb 6.7.1 runs on gnode118 via the module binary, no install.** AGREE. C-EVAL-03 is now YES unconditionally.
   C-EVAL-03/04/05 and C-GEOM-01 CPU work can start once S-1/S-2 exist.
2. **≈ 0.14 s per QM9 seed optimisation.** AGREE. C-GEOM-01 needs ≈ 6 CPU-h, C-EVAL-03 ≈ 8, C-EVAL-05 ≈ 2 per arm,
   B-xtb ≈ 6. No call changes, but every xTB card is now cheap enough that its CPU cost is not a selection criterion.
3. **GT conformers do not move under xTB (0.000 Å, 10/10).** AGREE. This is what makes an own-basin xTB seed
   reproduce the target's own L legitimately (§3g). The grounder rightly notes that the probe does not show ETKDG seeds
   reaching the reference basin. That is the unknown s, measured by the gate.
4. **TD's `utils/xtb.py` temp-dir race (S-1.1).** AGREE. It is a correctness precondition: with a Pool, a seed can
   silently receive another molecule's coordinates when atom counts match. All my xTB predictions assume S-1 is fixed
   and fallbacks are counted (gate only if ≥ 95 % of seeds relaxed). No numeric change.
5. **No validation path (S-5); λ must be pre-declared.** AGREE. See C below for the S4 protocol.
6. **`srETKDGv3()` also disables macrocycle torsions in RDKit 2022.9.5.** AGREE. My C-GEOM-06 estimate now refers to
   the corrected single-factor contrast `ETKDGv3()` + `useSmallRingTorsions = True`. QM9 rings have ≤ 9 atoms, so the
   macrocycle switch is probably inert, but the contrast should not rely on that. Variants are different embeddings, so
   the MDE is ≈ 0.008–0.010. The call stays MAYBE (CPU only unless the tail drops ≥ 5 points).

**C. Remaining disagreements from round 1. Do I now agree?**
1. **C-EVAL-01 fit population: now AGREE.** The grounder adopted the seed-reuse defect (5.1(e), C-EVAL-01 transfer
   check: fit on the 648 molecules without replaced slots, or as a sensitivity row). I accept its within-molecule fit,
   since the confounding argument is right, with the 648-molecule set as primary and the 906 set as sensitivity.
   - One leftover inconsistency in `grounding.md`: its C-EVAL-01 verdict line still says "within-molecule fit on the
     906-molecule population". It should say 648 (primary).
   - New caveat from A: the λ axis is itself cycled, so every λ point carries own-conformer identity. For B1, the
     regression measures distance to the target's own L, not reachable L quality. Add the §3g leak inversion (or
     random-GT runs) as the correction.
2. **S4 λ protocol: PARTLY AGREE.** We agree that λ must be pre-declared and never picked on test. I still ask for
   **two** pre-declared levels on new L sources, λ = 0 (the round-2 rule) and λ = 1, both reported and neither selected
   (+3 packed runs ≈ 0.5 GPU-h).
   - The grounder's 5.2.3 accepts B1/S3 on xTB seeds, and λ = 1 is S4's analogue of that B1 contrast.
   - After A, I expect S4 at λ = 1 to behave like B1, i.e. to need own-basin seeds, so it may lose to λ = 0. That
     makes the pair more informative, not less.
   - If the panel wants one level, keep λ = 0 and drop S4 at λ = 1, because B1 covers that contrast.

**D. Other grounder points.**
- **C-EVAL-01 feasibility (5.2.5): AGREE.** The per-molecule regression is possible on the cluster (CPU). I meant
  "not locally".
- **C-EVAL-02 GPU (5.2.4): AGREE**, and I upgrade the item. The random-GT runs for S3/CTRL_rematch (≈ 0.9 GPU-h) are
  not a clean ceiling. With the §3g inversion, they measure each model's own-vs-other-L sensitivity, which decides
  C-GEOM-01's B1/S3 outcome.
- **Round-2 gates computable now (5.2.6): AGREE.** My C-ROBUST-02 (≈ 20 %) and C-ROBUST-03 (≈ 30 %) gate
  probabilities are priors and should be replaced by the measured gates before the panel. Neither is synced locally, so
  I cannot compute them.
- **C-ROBUST-01 (§7): AGREE** with the post-pass hybrid builder and ≈ 29.4 GPU-h. A new consequence of A: the four-way
  mixture never shows the model *another conformer's* GT L. S3, S4 and B1 likewise always pair true L with its own
  torsion target.
  - INFERENCE: a fifth training source, "GT L of conformer j with the torsion target of conformer i (same molecule)",
    would teach a model to use reference-level non-own L. That is exactly what an xTB- or template-based L source
    delivers for most targets.
  - I flag it for the panel and scouts as a candidate, not as a card. The paired cache holds each molecule's GT
    conformers, which the grounder should confirm.
  - Arm 0 YES and Arm 1 MAYBE stay.
- **C-ROBUST-02/03/04/05 (§7): AGREE** with the costs (26.5 / 26.1 / 3–5 / 73 GPU-h) and with the NO calls on 04/05.
  02 and 03 stay gated on the measured gates.
- **Cost differences (5.2.8): AGREE**, adopted in §0.
- **Gate design (5.2.3): AGREE**, plus the s metric in A.

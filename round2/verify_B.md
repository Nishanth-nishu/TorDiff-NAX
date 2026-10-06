# Verification of round2/research_B.md (citation and claim verifier)

Written 2026-10-06. I did not edit research_B.md, and I read research_A.md and verify_A.md only after finishing my own checks (§6).

**How I checked**
- **Quotes.** Each of the 30 quoted strings in research_B §7 was searched in the cited fulltext. Before matching I lower-cased the text and removed whitespace, hyphenation and every non-alphanumeric character. The page is the number of form feeds before the match plus one.
- **PDF spot checks.** I checked these directly in the PDF with `pdftotext -f N -l N`: TD p.7, 26 and 27; S23D p.16; FM-refiner p.7 and 14; ET-Flow p.19; Corso p.63.
- **Code.** Every `path:line` was opened in `torsional-diffusion/` and `tools/`. The local repo is on branch `main` (18e0bd5), which contains the ablation-hook patches.
- **Numbers.** All numbers were recomputed from `cluster_sync/results/` with my own scripts (in the scratchpad: `q.py`, `ls.py`, `ls2.py`, `cov.py`, `emb.py`).

## 1. Paper citations

All PDFs and fulltext files exist. 29 of the 30 quotes match verbatim on the stated page. The FM-refiner caption (#17) matches only after the missing δ glyph is restored, and research_B flags that itself.

| # | claim | location | verdict | note |
|---|---|---|---|---|
| 1 | TD trains on matched L because of the train/test L shift | TD p.7 | VERIFIED | |
| 2 | "this shift significantly hurts performance" | TD p.7 | VERIFIED | |
| 3 | GT-L training: lower loss, bad inference | TD p.26 | VERIFIED | Table 8 item 5: model "tested (as always) on RDKit local structures", i.e. our B1-on-RDKit cell |
| 4 | Table 8: train on GT L 34.8 / 0.920 vs Random 30.9 / 0.922 | TD p.27 | VERIFIED | Same rows in pdftotext: AMR-R mean **0.920** (GT-L) vs **0.922** (Random), COV-R 34.8 vs 30.9. DRUGS, δ = 0.75 Å |
| 5 | Only D.E. matching 0.588 (vs baseline 0.582) | TD p.27 | VERIFIED | TD calls it "only marginally worse" (p.26) |
| 6 | L accuracy matters most on QM9 | TD p.26 | VERIFIED | |
| 7 | OMEGA has better L on QM9 | TD p.26 | VERIFIED | |
| 8 | Ring torsions are part of L | TD p.4 | VERIFIED | |
| 9 | Weak on puckered and fused rings | TD p.23 | VERIFIED | |
| 10 | GT-other-L floor 0.284 vs 0.324 Å (DRUGS) | TD p.21 | VERIFIED | |
| 11 | EBD: RDKit fragment shift "can potentially harm" | EBD p.5 | VERIFIED | The "fix" gloss is also supported on p.5: blurring converges to the RDKit fragment structure, and the prior is centred on it |
| 12 | GeoMol predicts LS first | GeoMol p.4 | VERIFIED | |
| 13 | GeoMol builds cycles jointly | GeoMol p.16 | VERIFIED | |
| 14 | ET-Flow harmonic prior | ET-Flow p.1 | VERIFIED | |
| 15 | ET-Flow QM9 96.47 / 0.073 | ET-Flow p.19 | VERIFIED | Appendix Table 9 ("Additional OOD results", QM9 RS row). The same numbers are in main-text Table 2 (p.7), which is the better citation |
| 16 | QM9 "likely to be saturated" | S23D p.16 | VERIFIED (quote) / PARTIAL SUPPORT | S23D argues saturation from **mean AMR < 0.1 Å**, not from δ = 0.5 coverage. It supports "QM9 is saturated" but not specifically "δ = 0.5 is saturated" |
| 17 | FM-refiner adopts δ = 0.05 because median COV is 100 % at 0.5 | FM-refiner p.7 | VERIFIED | Table 2 caption. The match needs the δ glyph restored, which research_B declares |
| 18 | ET-Flow 75.72 / 0.083 at δ = 0.05 | FM-refiner p.7 | VERIFIED | Same values as the 50-step row of Table 6. ET-Flow itself reports AMR-R 0.073; the difference is a different run or protocol |
| 19 | MCF-B 20 → 100 steps: 62.13 → 68.90 | FM-refiner p.14 | VERIFIED | Table 6, QM9, δ = 0.05. For ET-Flow, 20 → 50 steps is flat (75.65 → 75.72), so "more steps help" is model-dependent |
| 20 | Nikitin's bugs concern valency and bond order | Nikitin p.1 | VERIFIED | T9 ("none directly") is correct |
| 21 | Fragmented GEOM molecules come from xTB failures | Nikitin p.8 | DOES NOT SUPPORT (for QM9) | The quote is about **GEOM-Drugs** ("on the raw GEOM-Drugs dataset"). Applying it to `C=C1C(=O)C=NN1C.N` in GEOM-QM9 is a plausible analogy, not evidence |
| 22 | Recommended bond/angle/torsion metrics | Nikitin p.11 | VERIFIED (quote) / PARTIAL SUPPORT | The metric compares generated structures with their **GFN2-xTB-optimised** counterparts, not with GT. `geometry_metrics.py` vs GT is a different metric |
| 23 | Conditioning augmentation fixes cascade train-test mismatch | Ho p.3 | VERIFIED | "crucial" is also on p.3 |
| 24 | Gaussian noise is most effective | Ho p.6 | VERIFIED | Only "at low resolutions" (blur is preferred at high resolution) |
| 25 | Amortised augmentation strength | Ho p.6 | VERIFIED | |
| 26 | Ning: modelling the prediction error during training | Ning p.2 | VERIFIED | Analogy: DDPM-IP perturbs x_t, not a conditioning input |
| 27 | Corso thesis: ring-puckering coordinates | Corso p.63 | VERIFIED | Printed page = PDF page = 63. It is a future-work suggestion |
| 28 | PuckerFlow rings as the TD front end | PuckerFlow p.9 | VERIFIED | |

## 2. Code citations

| citation | claim | verdict | note |
|---|---|---|---|
| tools/paired_compare.py:12-15, 64-72, 99 | failures = COV 0 over the union; universe at :99 | VERIFIED | |
| evaluate_confs.py:81-84, 152 | model failure counted; `[0]*num_failures` appended | VERIFIED | |
| diffusion/sampling.py:54-63 | `[]` when ETKDG yields too few conformers; `--pre_mmff` does not act on `--seed_confs` | VERIFIED | mmff is applied only at :64, inside `if not seed_confs` |
| diffusion/sampling.py:75 | cycle: GT conformer i mod L | VERIFIED | The non-cycle arm is `random.choice`, i.e. **with replacement** (relevant in §5) |
| utils/dataset.py:24-43 (:29, :40-41) | hook location for L noise | VERIFIED | |
| utils/dataset.py:200-239 + T7 "raw path keeps no-rotor molecules" | composition confound | **PARTLY WRONG** | `utils/dataset.py:177-180` (`filter_smiles`, shared by the raw and standardized paths) drops molecules with no rotatable bonds in **both** paths. Only the 30-conformer cap (`standardize_confs.py:59-67`) and the `rdkit_no_embed` drop (`:79-82`) differ |
| standardize_confs.py:18, 58-67, 74-76, 79-82 | --mmff; cap; no-rotor drop; no-embed drop | VERIFIED | |
| tools/local_structure_analysis.py:1-37, 196-214, 293-298 | docstring; `set_acyclic_local`; oracle call | VERIFIED | Important detail: the oracle runs on a **single** seed `jb` (`:285`, `:296`), while `floor_best_sym` is a minimum over the top-m seeds plus a GetBestRMS re-score (`:280-284`). See §3(b) |
| tools/make_seed_pickles.py, geometry_metrics.py; slurm VARIANT=mmff; deferred TSV lines | exist as described | VERIFIED | `--maxiter` and `--floor_topm` exist (defaults 50 and 3) |

## 3. Recomputed numbers

### (a) A1 COV-R@0.5 gain on the success intersection

No per-molecule eval.pkl is available locally. I bounded the gain from the summary lines and the paired tables.
- **Common set.** In `paired_vs_R0`, A1 MAT-R has n = 927. That is the set of molecules finite in all 3 R0 seeds, so 69 of the 996-molecule universe are excluded. research_B used 935.
- **Check of my universe model.** It reproduces the table's R0 = 88.978 exactly from the per-seed SUMMARY lines: (89.0293, 935), (88.1758, 931), (88.6612, 931).
- **R0 on the 927 common molecules: [95.13, 95.60].** research_B's 95.2 multiplies the 996-universe value 88.98 by the 1000/935 denominator, which mixes two denominators.
- **A1 MAT-R on the 69 excluded molecules** is (0.0810·996 − 0.0837·927)/69 = **0.045 Å** (0.043–0.046 after rounding). Two things follow:
  - These cage molecules are *easy* for A1. This also contradicts `RESULTS_QM9.md` l.60, which calls them "among the hardest".
  - By Markov's inequality, A1's COV@0.5 on them is ≥ 90.8 %. So A1 on the common set lies in [96.22, 96.91].
- **Gain on the intersection: +0.6 to +1.8 points**, not "+1.0 to +4.8".
  - research_B's upper end is loose because it ignores the A1 AMR on the excluded molecules.
  - Its lower end is slightly too high because of the wrong n and the mixed denominator.
  - The qualitative claim stands, and more strongly: almost all of the +7.5 is failure accounting.
- **Status.** Exact value: UNVERIFIABLE locally; it needs the eval.pkl files and R2-0(a).

### (b) "Acyclic angles + lengths remove ≤ 24 % of the L-attributable floor"

These parts reproduce (939 molecules, multi-fragment molecule excluded):

| quantity | research_B | my value |
|---|---|---|
| row means | 0.0796 / 0.0694 / 0.0544 / 0.0250 | same |
| per-molecule floor_best_sym | 0.1095 | 0.1095 |
| per-molecule floor_gtother | 0.0258 | 0.0258 |
| local-oracle removal | 0.0168 | 0.0168 |
| conservative removal | 0.0198 | 0.0196 |
| share of 0.0837 | ≤ 24 % | 20.1 % / 23.4 % |
| per-class table | as shipped | reproduces to 4 decimals |
| floor mass in ring molecules | 94 % | 93.8 % |

The bound itself is wrong for three reasons:
- **Population mismatch.**
  - `floor_gtother` is NaN for the 225 molecules that have a single GT conformer. So the 0.0258 is a mean over 714 molecules, while 0.1095 is a mean over 939.
  - On the matched 714 molecules, the floor is 0.101 and the L-attributable part is 0.0752. Acyclic GT then removes 0.0197 (**26 %**), or 0.0225 conservatively (**30 %**). Both are above 24 %.
  - If single-conformer molecules are given gtother = 0 instead, the share is 19–22 %.
- **Wrong bound direction.**
  - The oracle runs on one seed (the best by transplant cost); the floor is a minimum over 3 seeds with a symmetric re-score. The oracle is also a DE upper bound, and sequential `SetAngle` cannot hit every target.
  - Each of these biases makes the oracle's RMSD too high. The recorded removal is therefore an under-estimate of what GT acyclic L buys, so "at most 24 %" points the wrong way.
  - My figures: 27.1 % of rows have angle oracle > floor (research_B: 26 %). For molecules with no torsion the figure is 40 %.
  - (verify_A notes an opposite bias: the oracle copies the conformer's own GT values.)
- **Verdict: WRONG as an upper bound.** A fair statement is "about 20–30 % in this oracle, with a downward-biased estimator". The qualitative conclusion, that rings carry most of the floor, survives.

### (c) Molecules with no heavy-atom torsion: 30 %

- **285/939 = 30.4 %** have no heavy-atom torsion. research_B's 284 leaves out one acyclic molecule that also has none, which it put in its "acyclic" class.
- R0's `breakdown.log` gives 281 molecules with MAT-R 0.149. VERIFIED, with an off-by-one in the count.

### (d) R0's per-molecule error that is L-limited

- `floor_run.log` (R0): floor 0.1195 / observed 0.1667 = **71.7 %** per molecule. Per GT conformer it is 0.0993 / 0.1661 = 59.8 %.
- The arithmetic is VERIFIED.
- The floor is a DE upper bound with top-m = 3, so 72 % is an **upper** bound on the L share. "≥ 72 %" would be wrong.
- research_B writes "~72 %" and acknowledges the direction under T3, but its §0 and deferred-table wording should read "≤ ~72 % (60 % per conformer)".

### (e) Transfer of the RDKit-trained model to MMFF L

| per GT conformer | R0 | A2 | Δ |
|---|---|---|---|
| floor_run | 0.0993 | 0.0832 | −0.0161 |
| observed | 0.1661 | 0.1490 | −0.0171 |
| headroom | 0.0668 | 0.0658 | |

- **Numbers: VERIFIED.** Per molecule: floor −0.023, observed −0.024, headroom 0.0471 → 0.0463. The MMFF angle RMSD of 2.44° is also verified.
- **Interpretation: OVERREACH.**
  - The A2 shift is a smaller shift *toward* GT (angles 3.9° → 2.4°). B1's shift goes *away* from its training L and is larger.
  - "CM-trained models are robust to L shift; GT-trained ones are not" therefore compares different directions and magnitudes.
  - R2-1, as designed, is the right test of this.

### Other checks
- **VERIFIED:** all paired-table values research_B quotes (B1, CTRL, gtL, gtLc, B2, B5, A1c, A2, A3, G1), the all_summaries values (B1 per-seed COV 82.8 / 82.0 / 84.4, A0 0.2298 / 0.2422, 1order 0.1826, released 0.1765), local_structure_std (0.1628 / 0.1086 / 95.8 %), COV-R@0.5 median = 100 in all 52 arms, MMFF floor 0.0915, and rigid rings 0.148 → 0.120.
- **Minor mismatches:**
  - "7.7 % of GT conformers > 0.05 Å for gtother": I get 7.9 % (excluding the multi-fragment molecule) and 8.3 % (including it). The "capped near ~92 %" also ignores the 225 single-conformer molecules.
  - "3.89° / 1.68°" are the excluding-multi-fragment values (3.885 / 1.680), but they are attributed to the log, which says 3.882 / 1.674.
  - "0.016 Å for B1 against 0.0037 for **CTRL**" (§1.6 (i)): 0.0037 is the released model's figure. CTRL_base's own gtL − gtLc difference is 0.0842 − 0.0820 = 0.0022.
  - "0.1 Å has the smallest relative replicate SD": **not supported.** R0's SD is 0.39 at a mean of 37.5 (1.0 %), against 0.43 at 89.0 (0.5 %) at δ = 0.5. Across arms the SD at 0.1 is 0.24–0.55, not 0.24–0.39.
  - The CTRL vs released gap is ≈ 0.004, not 0.005.
- **ETKDG re-embed check** (RDKit 2026.03.6, seed 0, 4 conformers): I get 15/57 molecules with default settings and 16/57 with `useRandomCoords`. research_B reports 10 and 14. The result depends on the seed; the conclusion ("mostly intrinsic") holds.
- **UNVERIFIABLE from the brief:** the "2 days × 4 GPUs = 192 GPU-h" budget. The brief says max 4 days and "4x RTX 3090 (verify)".

## 4. Per-ablation verdicts

### R2-0 (CPU hygiene and diagnostics): APPROVE WITH CHANGES
- (a)–(d) are sound, and (a) is essential: §3(a) shows the COV claim shrinks to ≈ +1.
- (e) The leak diagnostic is confounded as specified, for two reasons:
  - A generated conformer seeded with GT conformer j's L can reach GT j *exactly*, so any accurate model, even with no L→τ decoding, beats the 1/L chance rate in RMSD.
  - `random.choice` vs cycle differ in L coverage. With K = 2L draws with replacement, ~e⁻² ≈ 13.5 % of GT L are never drawn, so gtLc − gtL partly measures stratified sampling, not coupling (§5).
  - The fix:
    - Test in **torsion space**: is the generated τ nearest to the source GT τ, compared with the other GT τ, weighted by Boltzmann weight?
    - Use a no-model null (gtLcycle + random torsions; `A1c_gtL_cycle_sanity` already exists) next to CTRL.
    - Add a cycle-vs-random contrast for CTRL to calibrate the coverage effect.
- Also add: re-run the acyclic oracle on all top-m seeds, and report matched-population (n_gt > 1) shares.

### R2-1 (L-quality dose-response, inference only): APPROVE WITH CHANGES
- Label all GT-L conditions (noise, GT + MMFF) as **test-set oracles**.
- Isotropic Cartesian noise raises the irreducible heavy-atom floor directly (≈ σ√3 per atom). Every model then gets worse, whatever its robustness.
  - Compute a CPU floor (or a `--mode run` floor) for each condition.
  - Compare B1 − CTRL per condition, not B1's absolute curve.
- White noise is not the structured, biased error of RDKit L or of a learned L. An ε* stated as angle RMSD alone is an incomplete spec, so report bond, angle and ring-subset errors separately.
- State whether σ is per-axis or 3D.
- Use `--seed_confs_cycle` or random consistently with the anchors.
- Cost (7.5 GPU-h) is correct.

### R2-2 (B6, noise-augmented GT-L training): APPROVE WITH CHANGES
- The control is correct: B1 is the single-factor control, and CTRL_base is the target. B6 vs CTRL still inherits the T7 composition difference (30-conformer cap and no-embed drop only; see §2).
- Fix the arithmetic: σ√2 at σ = 0.04 per axis is 0.057 Å, not 0.04 Å.
- Pre-register success thresholds:
  - non-inferiority to CTRL_base on RDKit L, with a margin such as 0.004 Å (≈ 3× the training-seed SD);
  - superiority over CTRL on GT L, labelled as oracle.
- B6b (1 seed) is exploratory only.
- Consider a "mixed-L" augmentation arm (GT L or CM-RDKit L per sample), because white noise may not cover the systematic RDKit bias.
- Cost (≈ 47 GPU-h) is correct and fits.

### R2-5 (ring vs acyclic GT L with the model in the loop): APPROVE WITH CHANGES
- The two arms are built by different mechanisms (constrained `coordMap` embedding vs sequential `SetAngle` on one seed). Their fidelity differs, which biases the comparison toward the ring arm.
  - Report the achieved ring-subset and acyclic-subset bond/angle RMSD to GT per arm.
  - Build A5-acyc on every seed.
- About 30 % of molecules are rigid rings with no torsion. For them A5-ring ≈ A1 by construction, so stratify the "fraction of gap closed" by torsion count and ring size.
- Embedding failures change the population, so use the intersection, as research_B says.
- State `random.choice` vs cycle (it must match A1).
- Cost (3 GPU-h) is correct.

### R2-4 (B3, MMFF-matched TD): APPROVE
- CTRL_rematch ± `--pre_mmff` is the right control: same RDKit build and the same pipeline.
- Report the number of molecules lost to `mmff_error` (a composition change).
- Start the standardization first, as proposed. The ~12 CPU-h estimate is unverified.
- The "expected ~0.14" is a guess.

### R2-6 (steps / σ_min / ODE): APPROVE (low priority)
- Cheap (3 GPU-h). The released-model arms have sampling-seed replicates.
- The B1_gtL arm is a single-training-seed oracle condition, and the expected effects (≤ 0.005) are close to the detection limit. Treat it as descriptive.

### Endpoint change (§3, "COV-R@0.1 co-primary on the intersection"): REJECT AS A UNILATERAL CHANGE (needs user approval)
- The user pre-registered δ = 0.5 Å as the single primary threshold (`review/cross_validation.md` §6b; `paired_compare.py:20-21`). Making COV-R@0.1 co-primary is a protocol change. research_B does say it needs a user decision.
- **Is a pre-declared SECONDARY endpoint enough? Yes, for the round-2 decisions.**
  - AMR-R mean is already primary and has the power (it detected A2 at −0.026 while COV@0.5 did not).
  - COV-R@0.1 on the success intersection, declared *before* round-2 runs as a key secondary with its own Holm family (or gated after AMR-R), gives the dynamic range without changing the protocol.
  - Also pre-declare:
    - the failure rate as a separate endpoint;
    - COV-R@0.5 on the intersection as a sensitivity analysis next to the protocol (union, failures = 0) primary.
- research_B's §3 dynamic-range table itself uses the failure-inflated R0 values it criticises.

## 5. Additional finding: what the gtLc − gtL contrast measures

- `sampling.py:75`: the non-cycle arm draws GT L **with replacement**. With K = 2L draws, ~13.5 % of GT conformers never get their own L, whereas cycle gives every GT conformer its L twice.
- Recall rewards that coverage. So A1c − A1 (−0.0037) and B1_gtLc − B1_gtL (−0.016) are not pure "τ↔L coupling".
  - Their size grows with how much a model gains from *exact* own L, through decoding **or** simple accuracy.
  - This weakens research_B's evidence (i) for the "B1 reads τ from L" mechanism, and the "G3 is invisible to CM models" conclusion (§1.3).
- The same caveat applies to the BRIEF's A1c line.

## 6. Agreement with verify_A / research_A (read after my checks)

- **Agree.** verify_A independently found the same two issues in research_A's parallel claims:
  - the oracle's single-seed bias, so "at most" is the wrong direction;
  - the gtLcycle coverage confound.
- **Agree.** verify_A also notes that `--pre_mmff` is ignored on the `--seed_confs` path, which research_B states correctly.
- **Agree.** verify_A says ET-Flow is better cited from main-text Table 2.
- **Overlap.** research_A's D1 ring-stratum table uses per-row means over a different grouping. Both agents reach the same conclusion, "rings dominate; the acyclic factor is a minority share".
- **Overlap.** research_A's R2A-4 is the same idea as research_B's R2-2.

## 7. Summary

- **Citations.** All 28 citations exist and are on the stated pages.
  - The TD Table 8 DRUGS values are correct (AMR-R 0.920 for GT-L vs 0.922 for Random).
  - S23D supports QM9 saturation only through AMR < 0.1 Å, not through δ = 0.5 coverage.
  - Nikitin p.8 is about GEOM-Drugs, not QM9.
  - One code claim is partly wrong: T7's no-rotor point (dataset.py:177-180 filters both paths).
- **Numbers.**
  - Most numbers reproduce.
  - The A1 intersection gain is +0.6 to +1.8, not +1.0 to +4.8. The direction is the same and the effect is smaller.
  - "≤ 24 % acyclic" is wrong as a bound: the populations are mismatched (26–30 % on matched molecules) and the estimator is biased downward.
  - The 72 % L-limited figure is an upper bound.
  - The MMFF numbers are verified, but the robustness interpretation overreaches.
- **Ablations.**
  - Approved with changes: R2-0 (leak diagnostic redesign), R2-1, R2-2 and R2-5.
  - Approved: R2-4 and R2-6.
  - The endpoint change needs user approval. A pre-declared secondary COV-R@0.1 on the intersection is sufficient.

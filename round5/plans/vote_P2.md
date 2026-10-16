# Vote of planner P2 (round 5, Stage A)

Written 2026-10-16 after reading `plan_P1.md`, `plan_P3.md`, every answer in `round5/consult/*.md`, and the revised
spec blocks: EXP-1 §10 (rev. 2), EXP-3 §9, EXP-5 §12, EVAL §1.1/§3.1. Orchestrator data used: full QM9 re-matching
took 2.9–4.3 h on 16 CPUs (MMFF variant 11.0 h on 6 CPUs); the GEOM-DRUGS download is approved and queued. No code
was changed and nothing was committed.

Severity: **B** = would produce wrong or test-selected science, or cannot run as written; **S** = should fix;
**N** = nit. "Superseded" means the research answers have since overruled it (it is listed under (2), not counted
against the planner).

## 0. Defects in my own plan (stated first so the ranking is honest)
| # | Where | Defect | Status |
|---|---|---|---|
| P2-a | `plan_P2.md` §3.4 "Test pool", Q-P2-008 | I took "max K = 30" from `qm9_test_smiles_from_local_structure_test.csv`, which is a **capped, filtered** csv. The real test csv has K up to 727 (86 molecules > 30; Σ2K = 27,462, E-SAMPLE-054). So max(128, 12K) is wrong, my claim "128 ≥ 4K for every molecule" is false (80 molecules have 4K > 128), and my Exp 5 GPU-h is too low by ≈ 1.8×. | **B**, overruled by EXP-5 §12.1 |
| P2-b | §3.4 "PG batch rule", V-E5-6 | "One batch per molecule" is infeasible for K = 727 (2,908 conformers) and departs from PG's published 128-batch protocol (E-SAMPLE-033). | **B**, overruled by EXP-5 §12.5 |
| P2-c | §3.3, Q-P2-007 | The QM9 primary E/Z endpoint was fidelity to "specified" stereo; only 3 QM9 molecules are specified by the structural test (E-CHEM-066). | superseded (EXP-3 §9.3) |
| P2-d | §2.4, §3.2 | V2 velocity target with γ′(t), TD quadruple rule, restoration off, 3n−6−m normalisation, count-only chirality in x_t: all PENDING defaults that EXP-1 §10 replaced. | superseded |

## 1. Defects in P1 and P3

### 1.1 P1 (`round5/plans/plan_P1.md`)
| # | Sev. | Location | Defect and failure scenario |
|---|---|---|---|
| P1-1 | **B** | `plan_P1.md:394-395` | **GO is decided on the test set** ("GO iff test rigid AMR-R(σ_b*) ≤ 0.10 Å"). The first GPU gate becomes a test-set decision, against README rule 5 and EXP-1 §4 control 8. EXP-1 §10.3 now has validation decide GO. |
| P1-2 | **B** | `plan_P1.md:337-339` | Default "apply the `pair_ok` filter by setting the attribute the guard reads" turns on `dataset.py:145-162`, which **drops** conformers and whole molecules. That changes the training population against A0/A3 (the round-2 S4 limitation, FIXES "User ruling"), so A3 can no longer equal A0. EXP-1 §10.1.1 requires masking the geometry loss, with every conformer kept. |
| P1-3 | **B** | `plan_P1.md:441-466` (+ `:612-613` goldens) | There is **no per-molecule seeding**: all goldens and generation use TD's single global stream (`generate_confs.py:94-95`). A subset csv then draws a different random stream from the full run. "Each arm reproduces TD bit-for-bit for the same RNG state" (`:465`) cannot hold between a subset run and the released R_s, and K-arm − O on the subset is not paired by common random numbers. EXP-3 §9.1 conditions 1–2 require per-molecule seeding and O regenerated on the subset. |
| P1-4 | S | `plan_P1.md:181-182, 444` | K1 removes bonds from `edge_mask`/`mask_rotate` before `perturb_seeds`. That shrinks the uniform draw (`sampling.py:107`) and the `z` vector (`:191`), so inside an affected molecule every other bond gets different noise in K1 than in O. The contrast stays unbiased but loses within-molecule pairing. An update-mask K1 (z drawn on the full edge set; `modify_conformer` skips zero updates, `torsion.py:60`) is equivalent for the model and keeps the noise common (P2 V-E3-3). |
| P1-5 | S | `plan_P1.md:296-298` | Projector uses `torch.linalg.pinv(CCᵀ, hermitian=True)` with no row normalisation, no near-linear row drop and unspecified precision. Near-linear quadruples (TD treats triple bonds as rotatable) have row norms ∝ 1/sin θ. The Gram matrix squares the condition number, so in float32 the default relative cut-off can truncate genuine W directions, and **W·P_W ≠ 0 silently**. EXP-1 §10.1.2 now drops rows below sin 10°; rows should also be normalised and solved in float64. |
| P1-6 | S | `plan_P1.md:361-371` | M0 on **test** seeds with an AMR-R "recovery" on the rigid stratum. P1 itself notes this is ≈ 100 % by construction (`:369-371`), yet keeps it as the kill rule. Superseded by EXP-1 §10.3 (oracle-velocity integration on validation pairs). |
| P1-7 | S | `plan_P1.md:398-402, 674` | "A6 preview" runs an M2 arm in Stage A (0.35 GPU-h). DECISION.md approves Exp 1 only up to the first gate M1. A6 is confounded by design (EXP-1 §4, D-005), so an early, unreplicated A6 number invites exactly the over-reading D-005 warned about. |
| P1-8 | S | `plan_P1.md:241-245` | Other-half floor by parity over the 30-capped list. For the 85 molecules with > 30 clean conformers these are not the leak-free halves. EXP-1 §10.5 requires `split.tsv` halves with the cap applied after splitting. |
| P1-9 | N | `plan_P1.md:535-536` | MMFF94 chosen explicitly instead of TD's MMFF94s; EXP-5 §12.4 / Q-P1-009.4 now require MMFF94s. |
| P1-10 | N | `plan_P1.md:449` | The stereo guard uses TD's `get_torsion_angles`, whose cos clamp (`torsion.py:120`) blurs angles within ≈ 0.26° of 0/π. This is harmless for the |δ| > π/2 test but not for planarity statistics; EXP-1 §10.1.2 mandates atan2. |

### 1.2 P3 (`round5/plans/plan_P3.md`)
| # | Sev. | Location | Defect and failure scenario |
|---|---|---|---|
| P3-1 | **B** | `plan_P3.md:153` vs `:395` | `rigid_stratum` is defined as TD `edge_mask.sum() == 0`, but the Gate-0 stratum is **n_rot_heavy = 0** (289 molecules, E-JOINT-007, `breakdown.log` bin 0), which P3 itself quotes at `:395`. Molecules with only methyl/OH rotors have m > 0 and would silently leave the stratum. Gate 0 would then be read on a different, smaller population than its 0.152 / 0.128 Å baselines, and the 289-molecule assertion would fail or be skipped. |
| P3-2 | **B** | `plan_P3.md:429-430` | σ_b is chosen on validation, but **GO is read on test**; same defect as P1-1. Overruled by EXP-1 §10.3. |
| P3-3 | **B** | `plan_P3.md:518-519`, `:717` | PG with "one batch holding all 4K_i conformers": infeasible for K = 727 (2,908 conformers) and not PG's published protocol (EXP-5 §12.5). I made the same error (P2-b). |
| P3-4 | **B** | `plan_P3.md:511-514` | D-101 is treated as moot ("ready as one extra config line if R-SAMPLE wants it"), so no K-free block is generated. The record then keeps the capped 0.0042 Å lower bound on which the allocation drop was decided. EXP-5 §12.1 requires the max(128, 4K) seed-0 pool and the uncapped ORACLE recompute. |
| P3-5 | S | `plan_P3.md:235-236, 455-458` | K1 by `edge_mask`/`mask_rotate` surgery: same loss of within-molecule common random numbers as P1-4. |
| P3-6 | S | `plan_P3.md:568-572` | Patches TD's frozen `utils/xtb.py` although P3 states that no Stage A path uses it (`:571-572`). The fix is real, but it changes a pinned baseline file for no Stage A benefit; P3's own `XTBRunner` (`:573-577`) already covers the need. Defer to EXP-4. |
| P3-7 | S | `plan_P3.md:218`, `:700` | `TDSampler` defaults to `batch_size=128` while golden G1 compares with `generate_confs.py` (default 32, `generate_confs.py:34`). With a different batch size the `z` draws (`sampling.py:191`) map to different conformers, so G1 cannot be bit-exact unless the golden pins 32. State it. |
| P3-8 | S | `plan_P3.md:376-379` | M0 on test λ pickles: superseded by validation `cache_paired.val` with x_G* (EXP-1 §10.3). |
| P3-9 | S | `plan_P3.md:181-184` | Projector without row normalisation (rcond 1e-8 on the eigenvalues of an unnormalised Gram matrix): the same conditioning risk as P1-5. |
| P3-10 | N | `plan_P3.md:240` | Stereo guard uses the clamped `get_torsion_angles` (as P1-10). |
| P3-11 | N | `plan_P3.md:806` | Selector margin AMR-P ≤ +0.002 Å is unsourced; overruled (0.004, E-SAMPLE-057). |

Strengths I would keep from each:
- **P1:** the single gateway with a SHA-pin test of TD files (`plan_P1.md:602`); the AST dead-code test
  (`:630-632`); column-subset scoring of one `eval.pkl` per pool (`:538-542`), which is exact and cheap; and the
  D-101/D-102 designs, which R-SAMPLE adopted almost verbatim.
- **P3:** per-molecule seeding keyed by SMILES (`plan_P3.md:290-303`); splicing by **copying** O's conformers
  (`:300-303`); provenance check, OOM retry, resumable generation, GPU-h ledger, `STATUS.md` and pre-flight
  (`:281-354`); failure-count equality for spliced arms (`:307-308`); the floor re-run as its own regression
  (`:553-557`); DRUGS evaluation sharding (`:334-337`).

## 2. Conflicts between each plan and the research answers (binding)
| Topic | Binding answer | P1 | P2 (mine) | P3 |
|---|---|---|---|---|
| W quadruple | heavy-first, drop sin < sin 10°, atan2 (EXP-1 §10.1.2) | TD rule (`:127-129`) ✗ | TD default, heavy-first flag ✗ (pending) | TD rule (`:163-167`) ✗ |
| L target | pure-L x_G* (§10.1.3) | x_G ✗ | x_G ✗ (pending) | x_G ✗ |
| Parameterisation | x₁ prediction, t ~ U[0,1], no ε (§10.1.4) | t_ε = 1e-3 ✗ | velocity ✗ | ε = 1e-3 ✗ |
| Loss normalisation | per molecule ÷ rank(P_W) | 3n−6−m ✗ | 3n−6−m′ ✗ | 3n−6−m ✗ |
| x_t chirality | resample ≤ 5, then mask (§10.1.5) | count only ✗ | count only ✗ | not stated ✗ |
| pair_ok | keep, mask geometry loss (§10.1.1) | **drop** ✗ (P1-2) | not addressed ✗ | mask ✓ |
| Torsion restoration | ON everywhere (§10.2) | absent ✗ | ablation flag, off ✗ | absent ✗ |
| τ-frozen σ input | σ_min | σ_min ✓ | not stated | not stated |
| M1 trunk | whole network from scratch (§10.3) | ✓ | not stated | ✓ |
| M0 | oracle-velocity integration on validation | test, AMR recovery ✗ | validation, m ≥ 1 rings, linear step ✗ | test λ pickles ✗ |
| Who decides GO | validation (σ_b* and GO), scaling rule | test ✗ | validation ✓ | test ✗ |
| Rigid stratum | n_rot_heavy = 0, 289, m = 0 sub-row | ✓ | ✓ | m = 0 ✗ (P3-1) |
| ref_half | exact expectation primary, 10 draws as check (§10.5) | 10 draws primary ✗ | 10 draws primary, exact as check ✗ | 10 draws ✗ |
| Floor other half | `split.tsv` halves, cap after split; also re-run `--mmff_seeds` | parity on capped list ✗; mmff optional ✗ | parity ✗ | `split.tsv` ✓; mmff not mentioned ✗ |
| Exp 3 seeding / O | per-molecule; O regenerated on subset (§9.1) | ✗ (P1-3) | ✓ | ✓ |
| K1 scope | specified structural bonds only (§9.2) | all constrained ✗ | constrained set ✗ | constrained set ✗ |
| K2-inf centre for unspecified | late planarity fold at σ ≤ 0.1π (§9.2) | seed's state ✗ | not stated ✗ | seed3d ✗ |
| Exp 3 QM9 endpoint | GT-consistency on 37; DRUGS fidelity on 126 (§9.3) | specified ✗ | specified ✗ | specified ✗ |
| NI test | on the affected subset (§9.3) | all-molecule ✗ | all-molecule ✗ | all-molecule (derived) ✗ |
| DRUGS scope | 140 structural (+160 for K1-canonical), 1 seed first, cap 4.5–5 | 141, 3 seeds ✗ | 141 ✗ | 141, 1 seed, 4.5 ✓≈ |
| D-101 pool | seed 0 max(128, 4K); seeds 1–2 4K (§12.1) | ✓ | 12K ✗ | no K-free block ✗ |
| PG batch / tuning | 128; pick among 4 README settings at c = 1 on validation | 128 ✓; untuned ✗ | one batch ✗; 5 configs ≈ | one batch ✗; untuned ✗ |
| Exp 5 family | {P3−P1, P3−P2b} × {AMR-R, COV-R@0.5}; P6−P2b descriptive | ✓ | P6−P6r in primary ✗ | ✓ |
| Selector | 5 candidates incl. both FPS starts; margins 0.004 / 0.5; futility | margins missing ✗ | ✓ (one start) | 0.002 ✗ |
| Energy | MMFF94s (`likelihood.py:19-20`) | MMFF94 ✗ | MMFF94 ✗ | not stated |
| Invalid conformers | keep (primary), validity-adjusted secondary, never drop (EVAL §3.1) | validity not built ✗ | ✓ | `validity.py` ✓ |
| Validation set | builder must reproduce the test files for ≥ 99 % of rows; uncapped K (EVAL §1.1) | pending | ✓ (rule adopted) | pending |

## 3. Votes (one line each)
1. **Base plan: P2** for the validity layer (decision files + `selection_guard`, update-mask K1, correct rigid
   stratum, V-check table, goldens G1–G11), with **P3's §3/§9 job machinery** and **P1's gateway, SHA pin and AST
   dead-code test** merged in. Reason: the base fixes the decisions that can produce test-selected or mispopulated
   science (P1-1/2/3, P3-1/2/4); the other two contribute what P2 lacks (cluster reliability, smallest import surface).
2. **Package layout: PLAN.md's six subpackages; one TD gateway module (P1's `td.py` with its chdir + table check);
   `eval/budget.py` as the single budget/timing module (P3); no `diffusion/`, `postprocess/` or `utils` names.**
   Reason: it is the approved layout, has one importer of TD, and has no duplicate accounting.
3. **Sampler hook design: one `reverse_loop` (SDE branch only) with a per-edge update hook and a position hook;
   noise drawn on the full TD edge set; K1 as an update mask; PG and plain TD call TD's own `sample()` unchanged
   (P1).** Reason: one loop with a G1 golden; common random numbers inside affected molecules; no PG port to verify.
4. **D-101 pool: EXP-5 §12.1 exactly** (seed 0 max(128, 4K), seeds 1–2 4K, a prefix-only test, uncapped ORACLE
   recompute on columns < 128, CPU). Reason: binding; K-free; my 12K was based on a capped csv.
5. **D-102 arms: P0, P1, P2, P2b, P3, P6 + ORACLE; primary {P3−P1, P3−P2b}; PG batch 128, setting picked among the
   4 README settings on validation at c = 1; selector by the §12.4 rule with the futility stop.** Reason: binding,
   NFE-matched, and equal tuning budgets for PG and the selector.
6. **Exp 1 M1 per EXP-1 rev. 2:** heavy-first W with the near-linear drop, row normalisation and float64 solve;
   x_G* target; x₁ prediction with t ~ U[0,1]; ÷ rank(P_W); `pair_ok` masked, not dropped; x_t chirality resample
   then mask; restoration ON; τ-frozen read-out at σ_min; whole network from scratch; revised M0 on validation;
   validation decides σ_b* and GO (scaling rule), test read once; rigid = n_rot_heavy = 0 with the m = 0 sub-row;
   the 1 % E/Z trip-wire; smoke before the M1 ask. Reason: binding; it removes every test-set decision and every
   path by which torsion work could be booked as L.
7. **Exp 3 scope: EXP-3 §9** — QM9 102 (211 for K1-canonical) and DRUGS 140 (160) subsets with per-molecule seeding,
   O regenerated on the subset, spliced by copying; K1 on specified bonds only; K2-inf with the late planarity fold;
   geometric primary label against the committed 2026.03.6 table; NI on the affected subset; DRUGS 1 seed first
   under a 4.5–5 GPU-h cap; enriched set to Stage B. Reason: binding, and version-independent.
8. **Cluster/job machinery: P3's** (provenance check, per-molecule SMILES-keyed seeding, OOM retry, resumable
   generation, GPU-h ledger with `FX_BUDGET_STOP`, `STATUS.md`, pre-flight, DRY_RUN, failure-count equality for
   spliced arms, DRUGS eval sharding), **without the TD `utils/xtb.py` patch** in Stage A, and with M1 packing
   allowed only by P3's measured rule. Reason: it encodes the round-2 failure lessons; the patch touches a pinned
   file for no Stage A use.
9. **Schedule and cut order:** CPU first (ref_half exact, floor ×2 files, M0, validation set, E/Z table, Step 0,
   goldens) → smokes → 1-epoch smoke → M1 on 3 GPUs while GPU 4 runs Exp 5 validation pools and PG settings → Exp 3
   QM9 → Exp 5 test pools (blind until `selector.json` is hashed) → Gate 0 on validation, then test once → DRUGS.
   Estimate at TD speed: smokes 0.45 + M1 10.5 + Gate 0 ≈ 0.7 (1000-molecule validation) + Exp 3 QM9 ≈ 1.0 + Exp 5
   ≈ 6.5 (EXP-5 §12.6) + DRUGS ≤ 5 ≈ **24 GPU-h**; ≈ 29.5 at the M1 ceiling.
   Pre-declared cut order:
   (1) DRUGS K1-canonical;
   (2) Exp 5 P6 (descriptive);
   (3) DRUGS seeds 2–3;
   (4) DRUGS K1-wrong, with EXP-3 §9.4's claim restriction;
   (5) the rest of DRUGS moves to a follow-up ask.
   Never cut: goldens/smokes, the three σ_b runs, validation-decided Gate 0, QM9 O/K1/K1-wrong/K2-inf, Exp 5
   P0/P1/P2b/P3 and the validation selection. Reason: it keeps every primary contrast and every negative or NFE
   control; the orchestrator's 2.9–4.3 h re-matching figure moves A6b/DAgger CPU cost out of the critical path for
   Stage B, and A1+ (≈ 7 GPU-h) must enter the Stage B ask.

## 4. Final ranking
1. **P2** — no blocker left after the consult answers except P2-a/P2-b (pool size and PG batching), which are
   overruled identically for P3, while P2 is the only plan whose decision rules were already validation-only, whose
   rigid stratum is right, and which keeps the training population and common random numbers intact.
2. **P1** — smallest and cleanest import surface; D-101/D-102 adopted nearly as written. But it decides GO on test
   (P1-1), drops `pair_ok` conformers (P1-2) and has no per-molecule seeding for spliced Exp 3 arms (P1-3).
3. **P3** — best cluster engineering (to be merged), but it has four blockers: the wrong Gate-0 stratum definition
   (P3-1), GO on test (P3-2), infeasible one-batch PG (P3-3) and no K-free pool (P3-4).

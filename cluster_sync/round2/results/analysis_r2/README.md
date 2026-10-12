
Round-2 analysis driver (FIXES X10). Every comparison, Holm family and decision rule is declared IN THIS FILE before any
round-2 result exists (commit time = pre-registration). Run after the round-2 arrays:

  python tools/analyze_round2.py --res $RES --seeds $QM9_SEEDS2 --test_csv $QM9_TEST_CSV --out $RES/analysis_r2

Endpoints (SHORTLIST "Endpoints", unchanged):
  primary   : MAT-R (= AMR-R) mean and COV-R@0.5, TD protocol (union of molecules, model failure = COV 0), paired
              molecule bootstrap, Holm over the PRIMARY family (all primary contrasts x 2 metrics)
  secondary : COV-R@0.1 on the success intersection (molecules evaluated in every file of arm and ref), own Holm family
  also      : failure counts per file; COV-R@0.5 on the intersection (sensitivity); COV-R@0.05 (exploratory, no
              Holm); AMR-P / COV-P (descriptive)
  every row : training-seed SD of the arm and ref aggregates next to the bootstrap CI (research_check_B I7)
  S2 margin : non-inferiority of B6 to CTRL_base on RDKit L, margin 0.004 A AMR-R, judged with a HIERARCHICAL bootstrap
              (training seeds resampled, then molecules), i.e. with seed variance included (research_check_B I7)
  S4        : every S4 row is repeated on S4's own test-molecule subset (molecules with >= 1 pair_ok-safe matched pair,
              $QM9_SEEDS2/S4_test_subset.txt), with the controls re-scored on that subset (user ruling 2026-10-06).
              LIMITATION, printed next to every S4 result: re-scoring removes the test-set difference but not the
              training-set one (S4 trains on ~4 % fewer molecules than its controls; DECISION.md 2026-10-07).
  ORACLE    : S1 / S1.5 / lambda-series rows use test-set GT L and are analysis, never model selection.
  S1 / S4 dose-response is reported against the MEASURED L error (l_error.csv: bond / angle / ring-subset RMSD to GT),
  not against nominal lambda or sigma (research checks A and B).


Files: contrasts.csv (all rows; primary/secondary Holm columns), failures.csv, dose_response_vs_measured_L_error.{csv,png}, mmff_pre_failures.csv

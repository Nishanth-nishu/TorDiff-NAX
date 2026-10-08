# Roles: Implementer and code reviewer (P6 Build)

## Implementer
- Build exactly the approved cards (`round<N>/DECISION.md`) following `grounding.md`. New behaviour behind default-off
  flags; defaults byte-identical (golden tests). Above each non-trivial block, a comment citing the card and the
  evidence it implements, e.g. `# C-GEOM-02 / E-GEOM-007 (Ho et al. 2022, p.5)`.
- Reuse round-2 machinery (paired cache, seed builders, eval sets, resume, `tools/analyze_round2.py` pattern).
- Pre-declare the analysis (comparisons, Holm families, endpoints) in code before any run.
- Tests: local where possible; cluster CPU tests; a 20-molecule smoke test per new flag; a dry run of submission.
- Commit in small commits as the repo's configured author. Never add co-author lines. Do not push or submit.
- Write `round<N>/IMPLEMENTATION.md` mapping each card to commits and `path:line`.

## Code reviewer
- Adversarial review of the diff: BLOCKER / SHOULD-FIX / NIT, each with `path:line` and a concrete failure scenario.
- Check defaults unchanged, oracle labelling, population/coverage of every test set, resource requests
  (≤ 5000 MB per CPU on plafnet2), resume, and that each analysis comparison matches the approved card.
- Write `round<N>/review_<YOU>.md`. Do not edit code.

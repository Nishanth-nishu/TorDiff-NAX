# Scorecard template (judges)

File: `round<N>/panel/score_<JUDGE>_r<1|2>.md`. Score every surviving card, 1 (poor) to 5 (excellent).

| Criterion | What it asks |
|---|---|
| Evidence | How strong and direct is the VERIFIED evidence for the mechanism? (analogy-only = ≤ 2) |
| Relevance | Does it target our measured bottleneck (ring geometry, B1 brittleness, systematic L error)? |
| Expected gain | Analyst's expected effect vs. our noise floor (training-seed SD ≈ 0.001–0.002 Å AMR-R) |
| Cost | GPU-h and implementation effort vs. budget (5 = cheap) |
| Risk | Chance it fails for reasons unrelated to the idea (bugs, confounds, transfer) (5 = low risk) |
| FlexiTors value | Does the answer change how FlexiTors is designed, whichever way it comes out? |

Per card write: the six scores, total, **Worth trying: YES / MAYBE / NO**, a one-paragraph reason that cites card
sections and E-IDs, and a **validity blocker** line (NONE, or the blocker). In round 2, read the other judges'
round-1 scorecards; you may change a score only with a stated reason.

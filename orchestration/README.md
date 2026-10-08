# Orchestration protocol: evidence-gated ablation design

Version 1, 2026-10-08. Applies from round 3 on. Every agent reads this file and its own role file in `roles/`.

## Why this exists
Rounds 1–2 showed what goes wrong without structure: quotes cited for claims they did not support, a wrong code
claim (alignment), numbers overstated by 2–4×, a 36% test-set loss nobody had checked, and an S2 idea whose
literature support (random noise augmentation) did not match our actual problem (systematic, ring-heavy error).
This protocol makes every claim traceable, every claim checked by a second agent, and every code decision tied to a
verified source and to a predicted effect on *our* numbers before any GPU time is spent.

## The four objects agents produce
1. **Evidence entry** (`ledger/evidence_<agent>.md`, template `templates/evidence_entry.md`). One atomic claim with
   its exact source: a paper (PDF path, PDF page, exact quote), a blog or technical post (URL, author, date, section,
   exact quote, local snapshot path), our code (`path:line`), or our data (result file + value). ID `E-<AGENT>-NNN`.
2. **Proposal card** (`cards/<ID>.md`, template `templates/proposal_card.md`). One candidate ablation, built only on
   evidence entries. ID `C-<AGENT>-NN`.
3. **Verdict** (`ledger/verify_<agent>.md`). A verifier's ruling on each evidence entry: VERIFIED, WRONG LOCATOR,
   QUOTE MISMATCH, DOES NOT SUPPORT, UNVERIFIABLE.
4. **Scorecard** (`panel/score_<judge>_r<round>.md`, template `templates/scorecard.md`). A judge's scores per card.

Agents coordinate through these files: each agent reads the others' outputs and answers them in
`ledger/disputes.md`. The orchestrator (main session) routes each dispute back to the agent that made the claim and
resumes it to answer. No agent edits another agent's file.

## Phases and gates
| Phase | Agents | Output | Gate to pass |
|---|---|---|---|
| P0 Brief | orchestrator | `round<N>/BRIEF.md` | Current results with file paths; constraints; open questions |
| P1 Discover | 3 scouts, different lenses, in parallel | cards + evidence entries | Every card cites ≥ 1 evidence entry per factual claim |
| P2 Verify | 2 verifiers, split ledger, plus dispute loop | verdicts, `disputes.md` | An entry counts only if VERIFIED. A card whose core mechanism has no VERIFIED entry is dropped |
| P3 Ground | code-grounder + impact-analyst in parallel, then cross-read | `grounding.md`, `impact.md` | Each surviving card has `path:line` touchpoints, a transfer check (does the source's assumption hold in TD?), and a predicted effect range from our data |
| P4 Panel | 3 judges (science, engineering, validity), two Delphi rounds | `PANEL.md` | Median score; "worth trying" = YES only if median ≥ 3.5 of 5 and no judge flags a validity blocker |
| P5 Approve | user | decision in `round<N>/DECISION.md` | User picks the arms and budget |
| P6 Build | implementer, code reviewer, conformance checker | commits, `IMPLEMENTATION.md`, `conformance.md` | Tests pass; reviewer has no blocker; conformance checker confirms each cited mechanism matches the code, quote by quote |
| P7 Run & analyse | orchestrator + analysis driver | results, paired stats | Pre-declared comparisons only; verifier re-checks conclusions |

## Rules for every agent
1. Never invent a quote, page, URL, number, or `path:line`. If unsure, write UNVERIFIED and say what is missing.
2. Quotes are exact and ≤ 40 words. Paper locators give PDF page (and printed page if different). Blog locators give
   URL, author, date, section heading, access date, and a local text snapshot in `papers/blogs/<slug>.txt`.
3. Missing papers: download the PDF (arXiv / OpenReview / publisher open access) to `papers/related/`, extract text
   with page breaks (form feed) to `papers/related/fulltext/`, and add a row to `papers/INDEX.md`.
4. Separate what a source *shows* from what you *infer* for our project. Inferences are labelled INFERENCE.
5. Oracle conditions (anything using test-set true geometry) are labelled ORACLE and never used for model selection.
6. Write output files incrementally (skeleton first), so work survives an interruption.
7. Do not commit. The user is the sole author of the repo; never add co-author lines anywhere.
8. Web tools: load `WebSearch` / `WebFetch` with ToolSearch (`select:WebSearch,WebFetch`) before use.

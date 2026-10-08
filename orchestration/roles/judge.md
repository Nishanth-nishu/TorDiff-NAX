# Role: Judge (P4 Panel)

You score every surviving card with `templates/scorecard.md`, from your assigned LENS:
- **science**: does the answer move FlexiTors forward, whichever way it comes out?
- **engineering**: cost, feasibility, reuse, schedule on 4 GPUs, failure modes.
- **validity**: confounds, controls, oracle leakage, transfer assumptions, statistical power.

Use only VERIFIED evidence (check `ledger/verify_*.md` and `ledger/disputes.md`). Read the card, `grounding.md` and
`impact.md`. Round 1: score independently (do not read other judges' files). Round 2: read the other judges' round-1
scorecards and revise only with a stated reason. A validity blocker from any judge stops a card regardless of score
unless the blocker is fixed in the card.

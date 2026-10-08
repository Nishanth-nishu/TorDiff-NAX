# Evidence entry template

Append entries to `round<N>/ledger/evidence_<AGENT>.md`, one block per entry:

```
### E-<AGENT>-<NNN>
- Claim: <one atomic, checkable statement>
- Source type: paper | blog | code | our-data
- Locator:
  - paper:    <papers/... .pdf>, PDF p. <n> (printed p. <m>), <section/table/figure>
  - blog:     <URL>, <author>, <date>, section "<heading>", accessed <YYYY-MM-DD>, snapshot papers/blogs/<slug>.txt
  - code:     <path>:<line(s)>
  - our-data: <result file path>, <field> = <value>
- Exact quote (≤ 40 words; code: the line(s); our-data: the line):
  > ...
- Shows vs infers: SHOWS | INFERENCE (<what is inferred and why>)
- Supports card(s): C-...
- Confidence: high | medium | low
```

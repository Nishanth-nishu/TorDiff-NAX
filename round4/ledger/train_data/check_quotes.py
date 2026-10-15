"""Scout TRAIN: re-check every '> ' quote line in round4/ledger/evidence_TRAIN.md against its source.
Matching is on alphanumerics only (case-folded), so whitespace, hyphenation breaks and lost glyphs do not matter.
Paper quotes are checked on the stated PDF page of the local fulltext (form-feed pages); if a page list is given,
on any of them. Blog / code / our-data quotes are checked anywhere in the named file. Run from the repo root."""
import re, sys, os
L = open('round4/ledger/evidence_TRAIN.md', encoding='utf-8').read()
blocks = re.split(r'\n(?=### E-TRAIN-)', L)
def norm(x):
    return re.sub(r'[^0-9a-z]', '', x.lower())
def fulltext_for(pdf):
    base = os.path.basename(pdf)[:-4]
    for c in (f'papers/related/fulltext/{base}.readingorder.txt', f'papers/related/fulltext/{base}.txt',
              f'papers/core/{base}.fulltext.txt'):
        if os.path.exists(c):
            return c
    return None
ok = bad = 0
for b in blocks:
    m = re.match(r'### (E-TRAIN-\d+)', b)
    if not m:
        continue
    eid = m.group(1)
    quotes = [q[4:] for q in b.split('\n') if q.startswith('  > ')]
    loc = re.search(r'- Locator:(.*?)\n- Exact quote', b, re.S).group(1)
    srcs = []
    pm = re.search(r'(papers/(?:core|related)/[\w.]+\.pdf)[^\n]*?PDF p+\. ([\d, and]+)', loc)
    if 'Source type: paper' in b and pm:
        ft = fulltext_for(pm.group(1))
        pages = [int(x) for x in re.findall(r'\d+', pm.group(2))]
        txt = open(ft, encoding='utf-8', errors='replace').read().split('\f')
        hay = ''.join(txt[p - 1] for p in pages if p - 1 < len(txt))
        srcs.append((ft + f' p.{pages}', norm(hay)))
        import subprocess  # two-column pages: also try the layout extraction of the same pages
        for pg in pages:
            lay = subprocess.run(['pdftotext', '-layout', '-f', str(pg), '-l', str(pg), pm.group(1), '-'],
                                 capture_output=True, text=True, errors='replace').stdout
            lines = lay.splitlines()
            width = max((len(x) for x in lines), default=0)
            for mid in range(int(0.3 * width), int(0.75 * width), 2):  # try column boundaries
                left = ' '.join(x[:mid] for x in lines); right = ' '.join(x[mid:] for x in lines)
                srcs.append((f'layout p.{pg} split {mid}', norm(left) + '|' + norm(right)))
    else:
        for f in re.findall(r'((?:papers|round3|round4|cluster_sync|torsional-diffusion|tools)/[\w./-]+\.(?:txt|out|md|py|csv|log))', loc):
            if os.path.exists(f):
                srcs.append((f, norm(open(f, encoding='utf-8', errors='replace').read())))
        if 'contrasts.csv' in loc and os.path.exists('round4/ledger/train_data/contrasts_extract.txt'):
            srcs.append(('contrasts_extract', norm(open('round4/ledger/train_data/contrasts_extract.txt').read())))
    for q in quotes:
        nq = norm(q)
        hit = any(nq in h for _, h in srcs)
        ok += hit; bad += (not hit)
        if not hit:
            print(f'MISS {eid}: {q[:90]} | sources={[s for s, _ in srcs]}')
print(f'{ok} quotes found, {bad} missing')

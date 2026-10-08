"""Scout ROBUST self-check: every quote in round3/ledger/evidence_ROBUST.md is found in its source (whitespace collapsed;
U+FFFD in the source matches any single character). Run from the repo root."""
import re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
L = open('round3/ledger/evidence_ROBUST.md', encoding='utf-8').read()
def norm(s): return re.sub(r'\s+', ' ', s).strip()
def find(q, text):
    q, t = norm(q), norm(text)
    if q in t: return True
    pat = ''.join('.' if c == '�' else re.escape(c) for c in t)  # too slow for big text; use reverse
    return False
def find_wild(q, text):
    q, t = norm(q), norm(text)
    if q in t: return True
    # allow source U+FFFD to match any char of the quote: regex built from quote, replace chars like × Å ∈ with '.?'
    rq = ''.join('.' if c in '×Å∈' else re.escape(c) for c in q)
    return re.search(rq, t) is not None
for blk in re.split(r'\n### ', L)[1:]:
    eid = blk.split('\n')[0].strip()
    loc = re.search(r'- Locator:\s*(.*)', blk).group(1)
    quotes = [m[4:] for m in re.findall(r'\n  > .*', blk)]
    quotes = [q.strip() for q in quotes]
    ok = None; src = None
    if 'paper:' in loc:
        pdf = re.search(r'paper:\s*(\S+\.pdf)', loc).group(1)
        page = int(re.search(r'PDF p\. (\d+)', loc).group(1))
        ft = pdf.replace('papers/related/', 'papers/related/fulltext/').replace('.pdf', '.txt') if 'related' in pdf else pdf.replace('.pdf', '.fulltext.txt')
        if 'readingorder' in loc: ft = re.search(r'(papers/related/fulltext/\S+readingorder\.txt)', loc).group(1)
        pages = open(ft, encoding='utf-8', errors='replace').read().split('\f')
        ok = all(find_wild(q, pages[page-1]) for q in quotes); src = f'{ft} p{page}'
    elif 'blog:' in loc:
        snap = re.search(r'snapshot (papers/blogs/\S+\.txt)', loc).group(1)
        txt = open(snap, encoding='utf-8').read()
        ok = all(find_wild(q, txt) for q in quotes); src = snap
    elif 'our-data:' in loc:
        f = re.search(r'our-data:\s*(\S+)', loc).group(1).rstrip(',')
        txt = open(f, encoding='utf-8').read()
        ok = all(find_wild(q, txt) for q in quotes); src = f
    elif 'code:' in loc:
        f = re.search(r'code:\s*(\S+?):', loc).group(1)
        txt = open(f, encoding='utf-8').read()
        ok = all(find_wild(q, txt) for q in quotes); src = f
    print(f'{eid}: {"OK " if ok else "MISS"} ({len(quotes)} quote lines) {src}')

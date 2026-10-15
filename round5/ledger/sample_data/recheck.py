"""Re-check BENCH quotes: each '…'-separated fragment must occur (whitespace/quote-normalised) in the cited PDF
page(s) (pdftotext -layout -f N -l N) or in the cited snapshot / code / data file."""
import re, sys, subprocess, os, unicodedata
root = sys.argv[1]; ids = sys.argv[2].split(',')
t = open(os.path.join(root, 'round4/ledger/evidence_BENCH.md'), encoding='utf8').read()
blocks = re.split(r'\n(?=### E-BENCH-)', t)
def norm(s):
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'").replace('[δ]', '')
    s = re.sub(r'[^A-Za-z0-9.%=]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().lower()
for b in blocks:
    m = re.match(r'### (E-BENCH-\d+)', b)
    if not m or m.group(1) not in ids: continue
    q = re.search(r'> (.*)', b)
    quote = q.group(1) if q else ''
    frags = [f.strip().strip('"').strip() for f in re.split(r'…|/ "', quote) if len(f.strip().strip('"')) > 8]
    srcs = re.findall(r'((?:papers|torsional-diffusion|cluster_sync|round\d|tools|RESULTS_QM9)[^\s,;:)]*\.(?:pdf|txt|py|md|csv|log))', b)
    pages = re.findall(r'PDF p\. (\d+)', b)
    texts = []
    for s in dict.fromkeys(srcs):
        fp = os.path.join(root, s)
        if not os.path.exists(fp): texts.append(('MISSING ' + s, '')); continue
        if s.endswith('.pdf'):
            for pg in dict.fromkeys(pages):
                out = subprocess.run(['pdftotext', '-layout', '-f', pg, '-l', pg, fp, '-'], capture_output=True).stdout.decode('utf8', 'replace')
                texts.append((f'{s} p{pg}', out))
        else:
            texts.append((s, open(fp, encoding='utf8', errors='replace').read()))
    alltext = norm(' '.join(x[1] for x in texts))
    res = [(f[:40], norm(f) in alltext) for f in frags]
    ok = all(r[1] for r in res) and res
    print(m.group(1), 'OK' if ok else 'CHECK', [x[0] for x in texts], [] if ok else res)

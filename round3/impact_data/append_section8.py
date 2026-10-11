"""Impact analyst helper: append a 'Revised after cross-read' block at the END of each card's
'## 8. Predicted effect on our project (analyst)' section (just before '## 9.'); earlier §8 text is kept as history.
Input: text file with blocks '@@@ <CARD-ID>' + body.  Usage: python round3/impact_data/append_section8.py <file>
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HEAD8, HEAD9 = '## 8. Predicted effect on our project (analyst)\n', '\n## 9. Panel (judges)'
for b in re.split(r'^@@@ ', open(sys.argv[1], encoding='utf-8').read(), flags=re.M)[1:]:
    cid, body = b.split('\n', 1)
    path = os.path.join(ROOT, 'round3', 'cards', cid.strip() + '.md')
    txt = open(path, encoding='utf-8').read()
    i8, i9 = txt.index(HEAD8), txt.index(HEAD9)
    assert i8 < i9 and txt.count(HEAD9) == 1
    if 'Revised after cross-read (2026-10-11' in txt[i8:i9]:
        print(f'{cid}: already has a revision block, skipped')
        continue
    pre = txt[:i9].rstrip('\n')
    txt = pre + '\n\n' + body.rstrip('\n') + '\n' + txt[i9:]
    open(path, 'w', encoding='utf-8', newline='\n').write(txt)
    print(f'{cid}: appended {body.count(chr(10))} lines')

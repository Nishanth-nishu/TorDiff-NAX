"""Impact analyst helper: insert the analyst's text under each card's existing empty
'## 8. Predicted effect on our project (analyst)' heading (only that section is touched).

Input: a text file with blocks '@@@ <CARD-ID>' followed by the section body.
Refuses to write if the section already has content (heading not directly followed by '## 9.').
Usage: python round3/impact_data/fill_section8.py <sections.txt>
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HEAD = '## 8. Predicted effect on our project (analyst)\n'
blocks = re.split(r'^@@@ ', open(sys.argv[1], encoding='utf-8').read(), flags=re.M)[1:]
for b in blocks:
    cid, body = b.split('\n', 1)
    path = os.path.join(ROOT, 'round3', 'cards', cid.strip() + '.md')
    txt = open(path, encoding='utf-8').read()
    i = txt.index(HEAD) + len(HEAD)
    if not txt[i:].startswith('## 9.'):
        print(f'{cid}: section 8 not empty, skipped')
        continue
    txt = txt[:i] + body.rstrip('\n') + '\n\n' + txt[i:]
    open(path, 'w', encoding='utf-8', newline='\n').write(txt)
    print(f'{cid}: inserted {body.count(chr(10))} lines')

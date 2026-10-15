import sys, urllib.request, urllib.parse, re, time
q = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': n, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
import time
for _a in range(4):
    try:
        x = urllib.request.urlopen(url, timeout=60).read().decode('utf-8'); break
    except Exception as _e:
        print('# retry', _a, _e); time.sleep(15)
for e in x.split('<entry>')[1:]:
    t = re.sub(r'\s+', ' ', re.search(r'<title>(.*?)</title>', e, re.S).group(1))
    i = re.search(r'<id>(.*?)</id>', e).group(1)
    d = re.search(r'<published>(.*?)</published>', e).group(1)[:10]
    print(d, i.replace('http://arxiv.org/abs/', ''), '|', t)

import sys, urllib.request, urllib.parse, re
ids = sys.argv[1]
url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({'id_list': ids, 'max_results': 50})
x = urllib.request.urlopen(url, timeout=60).read().decode('utf-8')
for e in x.split('<entry>')[1:]:
    t = re.sub(r'\s+', ' ', re.search(r'<title>(.*?)</title>', e, re.S).group(1))
    s = re.sub(r'\s+', ' ', re.search(r'<summary>(.*?)</summary>', e, re.S).group(1))
    i = re.search(r'<id>(.*?)</id>', e).group(1)
    au = re.findall(r'<name>(.*?)</name>', e)
    cm = re.search(r'<arxiv:comment[^>]*>(.*?)</arxiv:comment>', e, re.S)
    print('##', i, '|', t, '|', ', '.join(au[:6]), '|', cm.group(1) if cm else '')
    print(s); print()

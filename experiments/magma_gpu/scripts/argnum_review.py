# Per-function review table for argnum_scan.py hits: prints each flagged function's signature with positions and every
# `info = -k` check with the parameter that position k actually names.  Usage: argnum_review.py <magma root> <scan output>
import re, os, sys
src = open(__file__.replace("argnum_review.py", "argnum_scan.py")).read(); ns = {}
exec(src.split('out = []')[0].replace("root = sys.argv[1]", "root=''"), ns)
root = sys.argv[1]; flagged = {}
for l in open(sys.argv[2]):
    m = re.match(r'(\S+):\d+ (\S+)', l)
    if m: flagged.setdefault(m.group(1), set()).add(m.group(2))
for f, names in flagged.items():
    s = ns['zonly'](ns['strip'](open(os.path.join(root, f), errors='ignore').read()))
    for m in re.finditer(r'\b(magma_\w+)\s*\(([^;{}()]*(?:\([^()]*\)[^;{}()]*)*)\)\s*\{', s):
        if m.group(1) not in names: continue
        pn = []
        for p in m.group(2).split(','):
            if not p.strip(): continue
            mm = re.search(r'(\w+)\s*(\[[^\]]*\])*\s*$', p.strip()); pn.append(mm.group(1) if mm else '?')
        i = m.end(); d = 1
        while d and i < len(s): d += (s[i] == '{') - (s[i] == '}'); i += 1
        body = s[m.end():i]
        print('##', f, m.group(1), ' '.join('%d:%s' % (j+1, n) for j, n in enumerate(pn)))
        for c in re.finditer(r'if\s*\(([^{};]*?)\)\s*\{?\s*(?:\*info|info|arginfo|\*arginfo)\s*=\s*-\s*(\d+)\s*;', body):
            k = int(c.group(2))
            if k == 0 or k > 60: continue
            print('   -%-2d -> %-12s  if (%s)' % (k, pn[k-1] if k <= len(pn) else '?', ' '.join(c.group(1).split())[:75]))
        names.discard(m.group(1))

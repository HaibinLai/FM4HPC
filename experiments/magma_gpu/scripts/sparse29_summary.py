# Summarise round29_sparse CSV files: verdict counts per solver, and every LIE / GAP / FALSEFAIL case.
# Usage: sparse29_summary.py dir_with_csv
import sys, glob, os, collections
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.csv'))):
    for l in open(f, errors='ignore'):
        p = l.strip().split(',')
        if len(p) < 12 or p[0] in ('solver',) or not p[4].isdigit(): continue
        rows.append(dict(solver=p[0], prec=p[1], fmt=p[2], mat=p[3], ret=p[5], info=p[6], it=p[7], fr=p[8], ta=p[9], tr=p[10], v=p[11], extra=','.join(p[12:])))
c = collections.defaultdict(collections.Counter)
for r in rows: c[r['solver']][r['v']] += 1
print(f"{len(rows)} runs")
for s in sorted(c): print(f"  {s:12s} " + ' '.join(f"{k}={v}" for k, v in sorted(c[s].items())))
print("\nnon-OK, non-honest cases grouped by (solver, verdict, matrix): formats / preconds")
g = collections.defaultdict(list)
for r in rows:
    if r['v'] in ('LIE', 'GAP', 'FALSEFAIL'): g[(r['solver'], r['v'], r['mat'])].append(f"{r['fmt']}/{r['prec']}:info={r['info']},it={r['it']},rel={r['tr']}")
for k in sorted(g): print(f"  {k}: {g[k][:4]}{' ...' if len(g[k]) > 4 else ''} ({len(g[k])})")
print("\nhonest FAIL grouped by (solver, matrix):")
h = collections.defaultdict(list)
for r in rows:
    if r['v'] == 'FAIL': h[(r['solver'], r['mat'])].append(f"{r['fmt']}/{r['prec']}:{r['info']}")
for k in sorted(h): print(f"  {k}: {sorted(set(x.split(':')[1] for x in h[k]))} ({len(h[k])})")

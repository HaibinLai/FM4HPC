# Summarise round27_sweep CSV output: routine/kind/size of every non-OK case, grouped.
# Usage: sweep_summary.py file.csv [...] [--detail ROUTINE]
import sys, collections
V = {"OK", "INFO", "WRONG", "WARN", "NFDIFF"}
def parse(line):
    f = line.rstrip('\n').split(',')
    vi = max(i for i, x in enumerate(f) if x in V)
    note = ','.join(f[vi + 1:]); verdict = f[vi]; ratio, il, im, n, m, kind = f[vi - 1], f[vi - 2], f[vi - 3], f[vi - 4], f[vi - 5], f[vi - 6]
    return dict(group=f[0], routine=','.join(f[1:vi - 6]), kind=kind, m=int(m), n=int(n), im=im, il=il, ratio=ratio, verdict=verdict, note=note)
rows = []
args = [a for a in sys.argv[1:] if not a.startswith('--')]
det = sys.argv[sys.argv.index('--detail') + 1] if '--detail' in sys.argv else None
for fn in args:
    if fn == det: continue
    for l in open(fn, errors='ignore'):
        if l.startswith('group,') or l.count(',') < 8: continue
        try: rows.append(parse(l))
        except ValueError: pass
if det:
    for r in rows:
        if r['routine'] == det and r['verdict'] != 'OK': print(r)
    sys.exit()
c = collections.Counter((r['routine'], r['verdict']) for r in rows)
tot = collections.Counter(r['routine'] for r in rows)
for rt in sorted(tot):
    bad = {v: c[(rt, v)] for v in ['INFO', 'WRONG', 'WARN', 'NFDIFF'] if c[(rt, v)]}
    if not bad: continue
    kinds = collections.Counter(r['kind'].split('#')[0] for r in rows if r['routine'] == rt and r['verdict'] != 'OK')
    sizes = sorted({(r['m'], r['n']) for r in rows if r['routine'] == rt and r['verdict'] != 'OK'})
    print(f"{rt:42s} cases={tot[rt]:5d} {bad}  kinds={dict(kinds)}  sizes={sizes[:12]}{'...' if len(sizes) > 12 else ''}")

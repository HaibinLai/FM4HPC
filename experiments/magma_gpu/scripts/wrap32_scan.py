# Workspace-size expressions (lwmin, lrwmin, liwmin, lwkopt, lwkmin, minwrk, maxwrk, ...) that contain a product of two
# size variables, evaluated in 32-bit magma_int_t (LP64 build): smallest n (square case m = n) at which the expression
# exceeds INT_MAX, with nb = 64 and the other block sizes as given.  Usage: wrap32_scan.py <magma root> [glob ...]
import re, sys, glob, os
root = sys.argv[1]
pats = sys.argv[2:] or ['src/z*.cpp']
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'argnum_scan.py')).read(); ns = {}
exec(src.split('out = []')[0].replace("root = sys.argv[1]", "root=''"), ns)
LHS = r'\b(lwmin|lrwmin|liwmin|lwkopt|lwkmin|minwrk|maxwrk|lwork_min|lrwork_min|lwmin_\w*|wrkbl|ldwork|lddwork)\s*=\s*([^;]+);'
INT_MAX = 2**31 - 1
def ev(expr, n):
    e = expr.replace('magma_int_t', '').replace('(double)', '')
    e = re.sub(r'\bmax\b', 'max', e); e = re.sub(r'\bmin\b', 'min', e)
    env = dict(n=n, m=n, N=n, M=n, nb=64, NB=64, nrhs=1, k=n, nn=n, ngpu=1, minmn=n, mn=n, lq2=4*n, lwstg1=128*n,
               lwstg2=4*n, ldda=n, lda=n, max=max, min=min)
    return eval(e, {'__builtins__': {}}, env)
for pat in pats:
    for f in sorted(glob.glob(os.path.join(root, pat))):
        s = ns['zonly'](ns['strip'](open(f, errors='ignore').read()))
        for m in re.finditer(LHS, s):
            lhs, expr = m.group(1), ' '.join(m.group(2).split())
            if not re.search(r'\b(n|m|N|M|minmn|k)\s*\*\s*(n|m|N|M|minmn|k)\b', expr) and 'n*n' not in expr.replace(' ', ''):
                continue
            try: ev(expr, 10)
            except Exception: continue
            lo, hi = 1, 200000
            if ev(expr, hi) <= INT_MAX: continue
            while lo < hi:
                mid = (lo + hi) // 2
                if ev(expr, mid) > INT_MAX: hi = mid
                else: lo = mid + 1
            line = s[:m.start()].count('\n') + 1
            print('%-32s %-8s n >= %6d  (%s)' % ('%s:%d' % (os.path.relpath(f, root), line), lhs, lo, expr[:70]))

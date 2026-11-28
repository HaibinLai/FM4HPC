# Cross-queue hazard scan (round 31): async work on the caller's queue followed, with no intervening sync, by a call to a
# routine that creates its own queue and launches work on it.  magma_queue_destroy syncs the device (cublasDestroy), so
# callee -> caller hand-offs are ordered; caller -> callee ones are not.
import re, glob, os, sys, collections
root = sys.argv[1]
srcs = sorted(glob.glob(os.path.join(root, 'src', '*.cpp')))
fdef = re.compile(r'^(?:extern "C"\s*)?magma_int_t\s*\n?\s*(magma_\w+)\s*\(', re.M)
bodies = {}
for f in srcs:
    t = open(f, errors='ignore').read()
    for m in re.finditer(r'\n(magma_\w+)\s*\(([^;{]*?)\)\s*\{', t):
        name = m.group(1); start = m.end(); depth = 1; i = start
        while depth and i < len(t):
            depth += {'{': 1, '}': -1}.get(t[i], 0); i += 1
        bodies.setdefault(name, (f, t[start:i]))
own = {n for n, (f, b) in bodies.items() if 'magma_queue_create' in b}
# a callee "launches before syncing the caller" unless the first thing it does is a sync of everything
SYNC = re.compile(r'magma_queue_sync\s*\(|magma_[sdczi]get(matrix|vector)\s*\(|magma_[sdczi]set(matrix|vector)\s*\(|magma_i[sdcz]amax\s*\(|magmablas_[sdcz]lan\w+\s*\(|magmablas_[sdcz]lag2[sdcz]\s*\(|magma_[sdcz]nrm2\s*\(|magma_[sdcz]dot\w*\s*\(|magma_[sdcz]asum\s*\(|cudaDeviceSynchronize|magma_queue_destroy|magma_free\s*\(|magma_event_sync')
ASYNC = re.compile(r'\b(magmablas_\w+|magma_[sdcz](?:gemm|gemv|trsm|trmm|syrk|herk|axpy|scal|copy|swap|copymatrix|copyvector|larfb_gpu)\w*|magma_[sdczi]\w*_async)\s*\([^;]*\bqueue\w*\b[^;]*\)\s*;', re.S)
hits = []
for name, (f, b) in bodies.items():
    if not re.match(r'.*/(z|zc|dx|ds|dsh|dh|xsh|xhs)', f): pass
    stmts = [s.strip() for s in re.split(r';', b)]
    pending = None
    for s in stmts:
        s1 = ' '.join(s.split())
        if SYNC.search(s1): pending = None; continue
        m = re.search(r'\b(magma_\w+)\s*\(', s1)
        callee = None
        for mm in re.finditer(r'\b(magma_\w+)\s*\(', s1):
            if mm.group(1) in own and mm.group(1) != name: callee = mm.group(1); break
        if callee and pending:
            hits.append((os.path.basename(f), name, pending, callee, s1[:160]))
            pending = None; continue
        if ASYNC.search(s1 + ';') and 'queue' in s1 and not callee:
            pending = s1[:120]
print(len(own), 'routines create their own queue')
for h in hits:
    if os.path.basename(h[0])[0] in 'zd' or os.path.basename(h[0]).startswith(('x', 'magma')):
        print(' | '.join(h))

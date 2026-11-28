# For every function with an `info` / `arginfo` result, compare each argument check `if (<cond>) info = -k;` with the
# position of the parameter the condition tests (its first parameter name).  Each function's usual offset k − position
# is taken as its numbering convention (e.g. MAGMA's −1 for a leading ngpu); checks that deviate from it are reported.  Usage: python3 argnum_scan.py <magma-2.10.0 root> [glob ...]
import re, sys, glob, os
root = sys.argv[1]
pats = sys.argv[2:] or ['src/z*.cpp', 'magmablas/z*.cpp', 'magmablas/z*.cu', 'sparse/src/z*.cpp', 'sparse/control/z*.cpp']
def zonly(s):
    """keep the complex (z) side of `#ifdef COMPLEX` / `#if defined(PRECISION_z)`-style blocks; blank the other side"""
    out, stack = [], []
    for ln in s.split('\n'):
        t = ln.strip()
        if re.match(r'#\s*if', t):
            cplx = bool(re.search(r'COMPLEX|PRECISION_z', t)) and 'REAL' not in t and '!' not in t
            real = bool(re.search(r'REAL|PRECISION_d|PRECISION_s', t)) and not cplx
            stack.append('c' if cplx else ('r' if real else 'n')); out.append(''); continue
        if re.match(r'#\s*else', t) and stack:
            stack[-1] = {'c': 'r', 'r': 'c', 'n': 'n'}[stack[-1]]; out.append(''); continue
        if re.match(r'#\s*endif', t) and stack:
            stack.pop(); out.append(''); continue
        out.append('' if 'r' in stack else ln)
    return '\n'.join(out)
def strip(s):
    s = re.sub(r'/\*.*?\*/', lambda m: re.sub(r'[^\n]', ' ', m.group(0)), s, flags=re.S)
    return re.sub(r'//[^\n]*', lambda m: ' ' * len(m.group(0)), s)
out = []
for pat in pats:
    for f in sorted(glob.glob(os.path.join(root, pat))):
        s = zonly(strip(open(f, errors='ignore').read()))
        for m in re.finditer(r'\b(magma_\w+)\s*\(([^;{}()]*(?:\([^()]*\)[^;{}()]*)*)\)\s*\{', s):
            name, params = m.group(1), m.group(2)
            plist = [p.strip() for p in params.split(',') if p.strip()]
            pnames = []
            for p in plist:
                mm = re.search(r'(\w+)\s*(\[[^\]]*\])?\s*$', p)
                pnames.append(mm.group(1) if mm else '')
            if not any(n in ('info', 'arginfo') or 'info' in n for n in pnames) and 'info' not in s[m.end():m.end()+3000]:
                continue
            i = m.end(); d = 1
            while d and i < len(s):
                d += (s[i] == '{') - (s[i] == '}'); i += 1
            body = s[m.end():i]
            checks = []
            for c in re.finditer(r'if\s*\(([^{};]*?)\)\s*\{?\s*(?:\*info|info|arginfo|\*arginfo)\s*=\s*-\s*(\d+)\s*;', body):
                cond, k = c.group(1), int(c.group(2))
                if k == 0 or k > 60: continue
                subj = next((t for t in re.findall(r'[A-Za-z_]\w*', cond) if t in pnames), None)
                if subj is None: continue      # condition on locals only (e.g. wantz flags) - skip
                pos = pnames.index(subj) + 1
                checks.append((k - pos, k, subj, pos, s[:m.end() + c.start()].count('\n') + 1, ' '.join(cond.split())[:60]))
            if not checks: continue
            deltas = [c[0] for c in checks]
            dom = max(set(deltas), key=deltas.count)
            for d, k, subj, pos, line, cond in checks:
                if d != dom:
                    out.append('%s:%d %s  reports -%d for %s (param %d; usual offset %+d)  (%s)' % (
                        os.path.relpath(f, root), line, name, k, subj, pos, dom, cond))
for o in out: print(o)
print(len(out), 'suspicious checks')

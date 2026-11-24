"""Cross-check of the Lean recombination bound (OzRecombine.lean): for N terms accumulated in double (first assigned, rest +=):
    2^53 * |computed - exact sum| <= 2*(N-1) * sum|T_k|          (N <= 2^52)
(1) on the REAL term lists of the library model (level order, D_H, limit 12) for several input families (exact sums via Fractions);
(2) on adversarial lists: random permutations, ascending/descending magnitudes, heavy cancellation, huge dynamic range.
Reports the maximum of (2^53 * error) / (2*(N-1)*sum|T|); the Lean bound requires <= 1.
"""
import sys, math
from fractions import Fraction as F
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fp16_fp64'))
import fastmodel as FM

def terms_by_element(a, b, limit=12):
    da, ea, _ = FM.split_fast(a, limit); db, eb, _ = FM.split_fast(np.asarray(b).T, limit)
    M, N = a.shape[0], b.shape[1]; out = [[[] for _ in range(N)] for _ in range(M)]
    for level in range(len(da) + len(db) - 1):
        for i in range(len(da)):
            j = level - i
            if not 0 <= j < len(db): continue
            p = da[i] @ db[j].T
            t = np.ldexp(p, ea[i][:, None] + eb[j][None, :])
            for x in range(M):
                for y in range(N): out[x][y].append(float(t[x, y]))
    return out

def accumulate(ts):
    acc = ts[0]
    for t in ts[1:]: acc = acc + t                                # binary64 addition
    return acc

def ratio(ts):
    N = len(ts)
    if N < 2: return 0.0
    exact = sum((F(t) for t in ts), F(0)); err = abs(F(accumulate(ts)) - exact)
    denom = 2 * (N - 1) * sum(abs(F(t)) for t in ts)
    return float(F(2 ** 53) * err / denom) if denom != 0 else 0.0

rng = np.random.default_rng(3)
fam = {'uniform': lambda s: rng.uniform(-1, 1, s), 'gaussian': lambda s: rng.standard_normal(s),
       'spread100': lambda s: np.ldexp(rng.uniform(-1, 1, s), rng.integers(-100, 101, s)),
       'rowscale': lambda s: rng.uniform(-1, 1, s) * np.ldexp(1.0, rng.integers(-50, 51, (s[0], 1)))}
print('(1) real term lists of the library model (level order):')
for name, g in fam.items():
    worst = 0.0; cnt = 0; nmax = 0
    for _ in range(6):
        M, N, K = 6, 6, int(rng.choice([16, 64, 200]))
        A = g((M, K)); B = g((K, N)); T = terms_by_element(A, B)
        for x in range(M):
            for y in range(N):
                worst = max(worst, ratio(T[x][y])); cnt += 1; nmax = max(nmax, len(T[x][y]))
    print(f'    {name:10} elements {cnt:4d}  terms per element up to {nmax:3d}  max ratio = {worst:.3e}')

print('(2) adversarial term lists:')
worst = {}
for trial in range(4000):
    N = int(rng.integers(2, 400)); kind = rng.choice(['perm', 'asc', 'desc', 'cancel', 'range'])
    if kind == 'range': ts = list(np.ldexp(rng.uniform(-1, 1, N), rng.integers(-300, 300, N)))
    elif kind == 'cancel':
        h = rng.uniform(-1, 1, N // 2 + 1); ts = list(np.concatenate([h, -h]) * np.ldexp(1.0, rng.integers(-5, 5)))[:N]; rng.shuffle(ts)
    else:
        ts = list(rng.standard_normal(N) * np.ldexp(1.0, rng.integers(-20, 20, N)))
        if kind == 'asc': ts.sort(key=abs)
        elif kind == 'desc': ts.sort(key=abs, reverse=True)
    ts = [float(t) for t in ts]
    worst[kind] = max(worst.get(kind, 0.0), ratio(ts))
print('    ' + '   '.join(f'{k}: {v:.3e}' for k, v in worst.items()) + '   (Lean bound requires <= 1)')

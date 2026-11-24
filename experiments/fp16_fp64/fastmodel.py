"""Vectorized version of the OzBLAS D_H arithmetic model in compare.py (forward summation only).

Same IEEE operations as compare.emulate(...)['forward']: row splitting with sigma/tau, FP16-exact slices,
FP32 partial dot (here: float64 matmul, asserted FP32-representable => equals the exact dot the original
model asserts), ldexp scaling by the row/column exponents, level-ordered double accumulation.
Validated bit-for-bit against compare.emulate on existing inputs (see validate_fastmodel.py).
`log2fn` replaces math.log2 in tau=ceil(log2(rowmax)) so libm / GPU-style variants can be compared.
Rows whose residual is exactly zero are frozen (slice 0, exponent 0): the same explicit extension as compare.py.
"""
import math
import numpy as np

def make_gpu_log2(bad):
    """GPU-style log2: exact powers 2^e with e in `bad` come out slightly above e (tau = e+1)."""
    def f(x):
        m, e = math.frexp(x)
        return (e - 1) + 2.0**-45 if m == 0.5 and (e - 1) in bad else math.log2(x)
    return f

def split_fast(mat, limit, log2fn=math.log2, rho_mode=0):
    r = np.array(mat, dtype=np.float64, copy=True)
    k = r.shape[1]
    # splitEpsMode: 0 and 2 give the same rho for half (getEpse2<half> == getEpse<half> == 24); 9 = 'overflow-ver'
    # Source: getRho<double,half> has a special case 'for half when m=n=k=1': dim_ = (dim == 1) ? 2 : dim, used by ALL splitEpsMode formulas
    # (the generic template would give rho=41 at K=1; verified: GPU slices at K=1 match rho=42 via dump_split).
    k2 = max(2, k)
    dimterm = math.log2(2.0 * math.sqrt(k2)) if rho_mode == 9 else math.log2(k2)
    rho = math.ceil(53 - (24 - dimterm) / 2)
    ds, es, frozen = [], [], 0
    for _ in range(limit):
        if not r.any(): break
        mx = np.abs(r).max(axis=1); nz = mx > 0
        tau = np.zeros(r.shape[0], dtype=np.int64)
        for i in np.nonzero(nz)[0]: tau[i] = math.ceil(log2fn(float(mx[i])))
        frozen += int((~nz).sum())
        # Source: CONST * scalbn(1., rho+tau) * splitShift -- scalbn(1,1024) overflows BEFORE the 0.75 factor, unlike ldexp(.75, 1024).
        with np.errstate(over='ignore'):
            sigma = (0.75 * np.ldexp(1.0, rho + tau))[:, None]
        if not np.all(np.isfinite(sigma[nz])): raise ValueError('sigma overflow')
        piece = np.where(nz[:, None], (r + sigma) - sigma, 0.0)
        r = r - piece
        d = np.ldexp(piece, -tau[:, None])
        if not np.array_equal(d.astype(np.float16).astype(np.float64), d):
            raise ValueError('slice not exactly representable in FP16')
        ds.append(d); es.append(tau)
    return ds, es, {'rho': rho, 'slices': len(ds), 'zero_padded_row_steps': frozen}

def emulate_fast(a, b, limit=12, log2fn=math.log2, fast=False, rho_mode=0):
    """a: m x k, b: k x n (float64). Returns (forward output m x n, meta)."""
    # cuozblasRgemm: for m==1 && n==1 (DOT) the library forces fastMode=0 and the batched flag off.
    if a.shape[0] == 1 and np.asarray(b).shape[1] == 1: fast = False
    da, ea, ma = split_fast(a, limit, log2fn, rho_mode)
    db, eb, mb = split_fast(np.asarray(b).T, limit, log2fn, rho_mode)
    maxlevel = min(len(da) - 1, len(db) - 1) if fast else len(da) + len(db) - 2   # fastMode=1: maxlevel=min(nSplitA-1,nSplitB-1)
    out = None; terms = 0
    for level in range(maxlevel + 1):
        for i in range(len(da)):
            j = level - i
            if not 0 <= j < len(db): continue
            p = da[i] @ db[j].T
            if not np.array_equal(p.astype(np.float32).astype(np.float64), p):
                raise AssertionError('FP32 partial sum not exact')
            t = np.ldexp(p, ea[i][:, None] + eb[j][None, :])
            out = t if out is None else out + t
            terms += 1
    if out is None: out = np.zeros((a.shape[0], np.asarray(b).shape[1]))
    return out, {'A': ma, 'B': mb, 'partial_gemms': terms}


def load_gpu_tau_table(path):
    """Parse log2table output ('T e k tau'): GPU tau for x = 2^e shifted by k ulps (k in -8..8)."""
    tab = {}
    for line in open(path):
        p = line.split()
        if len(p) == 4 and p[0] == 'T': tab[(int(p[1]), int(p[2]))] = int(p[3])
    return tab

def make_gpu_log2_table(tab):
    """log2 stand-in returning tau-0.5 (so ceil gives the measured GPU tau) for x within 8 ulps of a power of two
    (normal range, measured table); math.log2 elsewhere. Measured on CUDA 12.0.140 / V100 only."""
    def f(x):
        m, ex = math.frexp(x)
        lo = math.ldexp(0.5, ex); hi = math.ldexp(1.0, ex)
        u = math.ulp(lo)
        if x - lo <= 8 * u:
            key = (ex - 1, int(round((x - lo) / u)))
        elif hi - x <= 8 * math.ulp(x):
            key = (ex, -int(round((hi - x) / math.ulp(x))))
        else:
            return math.log2(x)
        t = tab.get(key)
        return math.log2(x) if t is None else t - 0.5
    return f

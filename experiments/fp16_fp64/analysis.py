"""Shared analysis helpers for GPU-vs-model comparisons (vectorized; no per-element Fractions except sampling)."""
import math
from fractions import Fraction as Fr
import numpy as np

def _key(x):
    """Monotonic unsigned key of binary64 values (no int64 overflow for opposite signs; -0 and +0 differ by 1 here)."""
    u = np.ascontiguousarray(x, dtype=np.float64).view(np.uint64)
    return np.where(u >> np.uint64(63), ~u, u | np.uint64(1 << 63))

def compare(gpu, model):
    """Bitwise comparison treating -0 == +0. Returns dict(identical, ndiff, total, max_ulp, nonfinite_gpu, nonfinite_model)."""
    g = np.asarray(gpu, dtype=np.float64) + 0.0; m = np.asarray(model, dtype=np.float64) + 0.0
    fin = np.isfinite(g) & np.isfinite(m)
    same_nonfin = (~fin) & ((np.isnan(g) & np.isnan(m)) | (g == m))
    eq = np.where(fin, g.view(np.uint64) == m.view(np.uint64), same_nonfin)
    kg, km = _key(g), _key(m)
    ud = np.where(kg > km, kg - km, km - kg)
    ulp = np.where(fin, ud, np.where(eq, np.uint64(0), np.uint64(2**64 - 1)))
    return {'identical': bool(eq.all()), 'ndiff': int((~eq).sum()), 'total': int(eq.size), 'max_ulp': int(ulp.max()) if ulp.size else 0}

def exact_sample(opA, opB, alpha, beta, c0, idxs):
    """Exact (Fraction) value of alpha*(opA@opB)+beta*c0 at the sampled (i,j); assumes finite inputs."""
    out = []
    for i, j in idxs:
        s = sum((Fr(float(x)) * Fr(float(y)) for x, y in zip(opA[i], opB[:, j])), Fr(0))
        v = Fr(float(alpha)) * s
        if beta != 0.0 and c0 is not None: v += Fr(float(beta)) * Fr(float(c0[i, j]))
        out.append(v)
    return out

def _two_product_parts(x, y):
    """Exact product parts (p, e) with p+e == x*y, vectorized (Veltkamp split; no overflow/underflow assumed)."""
    c = 134217729.0
    def split(a): t = c * a; hi = t - (t - a); return hi, a - hi
    p = x * y; xh, xl = split(x); yh, yl = split(y)
    return p, ((xh * yh - p) + xh * yl + xl * yh) + xl * yl

def _fast_err(row, col, outvals):
    """For alpha=1,beta=0: correctly rounded exact dot and errors |out-dot| via exact parts + math.fsum (fast for large K)."""
    p, e = _two_product_parts(np.asarray(row, dtype=np.float64), np.asarray(col, dtype=np.float64))
    parts = np.concatenate([p, e]).tolist(); dot = math.fsum(parts)
    return dot, [abs(math.fsum(parts + [-float(v)])) for v in outvals]

def sampled_errors(opA, opB, alpha, beta, c0, outs, nsamp=48, seed=0):
    if alpha == 1.0 and beta == 0.0 and opA.shape[1] > 2048:
        M, N = outs[next(iter(outs))].shape; rng = np.random.default_rng(abs(int(seed))); names = list(outs)
        idxs = [(int(rng.integers(M)), int(rng.integers(N))) for _ in range(min(nsamp, M * N))]
        dots = []; errs = {n: [] for n in names}
        for (i, j) in idxs:
            d, es = _fast_err(opA[i], opB[:, j], [outs[n][i, j] for n in names]); dots.append(abs(d))
            for n, e in zip(names, es): errs[n].append(e)
        scale = max(dots)
        return {n: (float(max(errs[n]) / scale) if scale and all(np.isfinite(outs[n][i, j]) for (i, j) in idxs) else None) for n in names}
    return _sampled_errors_exact(opA, opB, alpha, beta, c0, outs, nsamp, seed)

def _sampled_errors_exact(opA, opB, alpha, beta, c0, outs, nsamp=48, seed=0):
    """Max |out-exact| / max|exact| over a random sample of elements, per named output array."""
    M, N = outs[next(iter(outs))].shape; rng = np.random.default_rng(abs(int(seed)))
    idxs = [(int(rng.integers(M)), int(rng.integers(N))) for _ in range(min(nsamp, M * N))]
    ex = exact_sample(opA, opB, alpha, beta, c0, idxs); scale = max((abs(v) for v in ex), default=Fr(0))
    res = {}
    for name, arr in outs.items():
        if not np.all(np.isfinite([arr[i, j] for i, j in idxs])): res[name] = None; continue
        e = max(abs(Fr(float(arr[i, j])) - v) for (i, j), v in zip(idxs, ex))
        res[name] = float(e / scale) if scale else (0.0 if e == 0 else None)
    return res

def apply_alpha_beta(t, alpha, beta, c0):
    """OzBLAS Axpby: C = fma(alpha, t, beta*C0) -- fma exact via Fractions only when needed."""
    if alpha == 1.0 and beta == 0.0 and (c0 is None or np.all(np.isfinite(c0))):
        return t + 0.0
    c0 = np.zeros_like(t) if c0 is None else c0
    out = np.empty_like(t)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            bc = beta * c0[i, j]                      # rounded product, as in the kernel
            if not (math.isfinite(bc) and math.isfinite(t[i, j])): out[i, j] = alpha * t[i, j] + bc; continue
            out[i, j] = float(Fr(float(alpha)) * Fr(float(t[i, j])) + Fr(float(bc)))   # single rounding = fma
    return out

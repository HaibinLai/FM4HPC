"""The post-check of IrsContract.lean (`wrapper_sound`) applied to cuSOLVER outputs.

Check, for a returned x (no trust in the solver's iter):
  Rh  = max_i |r_i| with r = b - A x computed by Dot2 (Ogita-Rump-Oishi 2005: TwoProduct via Veltkamp split + TwoSum),
        error |r_i - r_i,exact| <= u |r_i,exact| + g_N^2 sum_j |terms|, N = n + 1 terms per row (b_i and -a_ij x_j)
        => R <= (1+eps) Rh + gR (B + A X), eps = u/(1-u), gR = g_N^2/(1-u)
  Ah  = FP64 max row sum of |A|, |Ah - A| <= g_n A
  X, B exact (max of |.|), c = sqrt(n) * 2^-53
  test: (1+eps) Rh + gR (B + X Ah/(1-gA)) < c X Ah/(1+gA)       -- evaluated exactly with Fractions
Wrapper: x_cusolver if the test passes, else x = cusolverDnDDgesv(A, b) and test again; returns (x, certified).
Also shows `fp64_cannot_certify`: the same test with an FP64 residual (gR = g_{n+1}) is never satisfiable.
Usage (roxy): python wrapper_demo.py
"""
import ctypes as C
import math
from fractions import Fraction as F

import numpy as np

import irs_probe as P
import irs_scale as S

for nm in ("cusolverDnDXgesv", "cusolverDnDSgesv", "cusolverDnDDgesv"):
    getattr(P.cusolver, nm + "_bufferSize").argtypes = P.cusolver.cusolverDnDHgesv_bufferSize.argtypes
    getattr(P.cusolver, nm).argtypes = P.cusolver.cusolverDnDHgesv.argtypes

U = F(1, 2 ** 53)
SPLIT = 134217729.0  # 2^27 + 1


def two_prod(a, b):
    p = a * b
    ca = SPLIT * a; ah = ca - (ca - a); al = a - ah
    cb = SPLIT * b; bh = cb - (cb - b); bl = b - bh
    return p, al * bl - (((p - ah * bh) - al * bh) - ah * bl)


def two_sum(a, b):
    s = a + b
    z = s - a
    return s, (a - (s - z)) + (b - z)


def residual_dot2(A, b, x):
    """r = b - A x, Dot2 for every row at once"""
    s = b.copy()
    c = np.zeros_like(b)
    for j in range(A.shape[1]):
        p, pe = two_prod(-A[:, j], np.full(A.shape[0], x[j]))
        s, se = two_sum(s, p)
        c = c + (pe + se)
    return s + c


def gamma(n):
    return n * U / (1 - n * U)


def check(A, b, x, residual="dot2"):
    n = A.shape[0]
    if not np.all(np.isfinite(x)):
        return False
    r = residual_dot2(A, b, x) if residual == "dot2" else b - A @ x
    Rh = F(float(np.abs(r).max()))
    Ah = F(float(np.abs(A).sum(axis=1).max()))
    X = F(float(np.abs(x).max()))
    B = F(float(np.abs(b).max()))
    gA = gamma(n)
    if residual == "dot2":
        eps, gR = U / (1 - U), gamma(n + 1) ** 2 / (1 - U)
    else:  # plain FP64 residual: |r_hat - r| <= g_{n+1} (|b| + |A||x|)
        eps, gR = F(0), gamma(n + 1)
    c = F(math.isqrt(n)) * U if math.isqrt(n) ** 2 == n else None
    assert c is not None, "use n a perfect square so that c = sqrt(n) u is exact"
    lhs = (1 + eps) * Rh + gR * (B + X * (Ah / (1 - gA)))
    rhs = c * X * (Ah / (1 + gA))
    return lhs < rhs


def wrapped(nm, A, b):
    r = S.legacy(nm, A, b)
    if check(A, b, r["x"]):
        return r["x"], True, "solver", r["niters"]
    x = S.legacy("cusolverDnDDgesv", A, b)["x"]
    return x, check(A, b, x), "fallback DDgesv", r["niters"]


if __name__ == "__main__":
    print(f"cuSOLVER {P.ver.value}", flush=True)
    n = 1024
    stats = {}
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        A0 = rng.choice([-1.0, 1.0], (n, n)) * rng.uniform(0.5, 1.0, (n, n))
        xt = rng.uniform(-1, 1, n)
        b0 = A0 @ xt
        for nm in ("cusolverDnDBgesv", "cusolverDnDXgesv", "cusolverDnDSgesv"):
            for k in (0, 20, 40, -60):
                A, b = A0 * 2.0 ** k, b0 * 2.0 ** k
                raw = S.legacy(nm, A, b)
                fe_raw = np.abs(raw["x"] - xt).max() / np.abs(xt).max()
                x, cert, src, it = wrapped(nm, A, b)
                fe = np.abs(x - xt).max() / np.abs(xt).max()
                fp64_ok = check(A, b, raw["x"], residual="fp64")
                key = "raw WRONG" if raw["niters"] >= 0 and fe_raw > 1e-6 else "raw ok-ish"
                stats.setdefault((key, src, cert), 0)
                stats[(key, src, cert)] += 1
                print(f"seed={seed} {nm} 2^{k:<4d} cuSOLVER iter={it:4d} err={fe_raw:9.3g} -> wrapper uses {src:16s} "
                      f"certified={cert} err={fe:9.3g}   (FP64-residual check passes: {fp64_ok})", flush=True)
    # a matrix on which FP64 GEPP itself fails: the wrapper must report certified=False
    m = 1024
    W = np.tril(-np.ones((m, m)), -1) + np.eye(m)
    W[:, -1] = 1.0
    xt = np.random.default_rng(9).uniform(-1, 1, m)
    bw = W @ xt
    x, cert, src, it = wrapped("cusolverDnDSgesv", W, bw)
    print(f"wilkinson n={m}: wrapper uses {src}, certified={cert}, err={np.abs(x - xt).max() / np.abs(xt).max():.3g}")
    print("\nsummary (raw cuSOLVER verdict, what the wrapper returned, certified):")
    for k, v in sorted(stats.items()):
        print(f"   {k}: {v}")

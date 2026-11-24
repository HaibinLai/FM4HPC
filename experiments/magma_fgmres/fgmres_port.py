"""Faithful CPU port of MAGMA v2.10.0 magma_dsgesv_iteref_gpu (GMSTRS path): dxgesv_gmres_gpu.cpp l.406-600 and
dgmres_plu_gpu.cpp l.139-297, to check the predictions of lean/OptimiKahan/MagmaFgmres.lean numerically before the GPU.

Kept exactly: x0 = 0; cte = ||A||_inf * eps * sqrt(n) (eps = 2^-53); gmtol = cte; maxiter = restrt = min(n, 500);
ro = ||b - A x||_2; `if fabs(ro) <= 1e-18: break`; eps1 = tol*ro at the first pass; the Arnoldi/Givens recurrence
with EPSILON = 1e-18 / EPSMAC = 1e-16; exit test `i+1 >= restrt or ro <= eps1 or iters >= maxiter`; *iter = iters;
no residual check after FGMRES, no fallback.  Low-precision part: FP32 LU with partial pivoting (sgetrf) and
dsgetrs = convert rhs to FP32, solve in FP32, convert back.  (MAGMA's GPU kernels round differently; the predictions
below do not depend on that.)
Predictions (Lean theorem in brackets):
  P1 [enforced_rule]        relative residual ||r||_2/||b||_2 <= sqrt(n) eps ||A||_inf: backward error grows like ||A||
  P2 [tol_ge_one_one_step]  ||A||_inf * eps * sqrt(n) >= 1  =>  iter = 1, FP32-level error, info = 0
  P3 [tiny_rhs_returns_zero] ||b||_2 <= 1e-18  =>  x = 0, iter = 0
  P4 [gmstrs_always_success] iter >= 0 in every case
"""
import math
import sys

import numpy as np

EPS = 2.0 ** -53
EPSILON, EPSMAC = 1e-18, 1e-16


def sgetrf(A):
    LU = A.astype(np.float32).copy()
    n = LU.shape[0]
    piv = np.arange(n)
    for k in range(n):
        p = k + int(np.argmax(np.abs(LU[k:, k])))
        if p != k:
            LU[[k, p]] = LU[[p, k]]
            piv[[k, p]] = piv[[p, k]]
        if LU[k, k] != 0:
            LU[k + 1:, k] /= LU[k, k]
            LU[k + 1:, k + 1:] -= np.outer(LU[k + 1:, k], LU[k, k + 1:])
    return LU, piv


def dsgetrs(LU, piv, v):
    y = v[piv].astype(np.float32)
    n = len(y)
    for i in range(n):
        y[i] -= LU[i, :i] @ y[:i]
    for i in range(n - 1, -1, -1):
        y[i] = (y[i] - LU[i, i + 1:] @ y[i + 1:]) / LU[i, i]
    return y.astype(np.float64)


def fgmres(A, b, LU, piv, maxiter, restrt, tol):
    n = len(b)
    x = np.zeros(n)
    iters = 0
    while True:
        r = b - A @ x
        ro = math.sqrt(r @ r)
        if abs(ro - 0.0) <= EPSILON:
            break
        V = np.zeros((restrt + 1, n)); W = np.zeros((restrt, n))
        V[0] = r * (1.0 / ro)
        if iters == 0:
            eps1 = tol * ro
        HH = np.zeros((restrt, restrt + 1)); c = np.zeros(restrt); s = np.zeros(restrt); rs = np.zeros(restrt + 1)
        rs[0] = ro
        i = -1
        while True:
            i += 1
            iters += 1
            W[i] = dsgetrs(LU, piv, V[i])
            V[i + 1] = A @ W[i]
            for j in range(i + 1):
                HH[i, j] = V[j] @ V[i + 1]
                V[i + 1] -= HH[i, j] * V[j]
            t = math.sqrt(V[i + 1] @ V[i + 1])
            HH[i, i + 1] = t
            if abs(t - 0.0) > EPSILON:
                V[i + 1] *= 1.0 / t
            for k in range(1, i + 1):
                k1 = k - 1
                t = HH[i, k1]
                HH[i, k1] = c[k1] * t + s[k1] * HH[i, k]
                HH[i, k] = -s[k1] * t + c[k1] * HH[i, k]
            Hii, Hii1 = HH[i, i], HH[i, i + 1]
            gam = math.sqrt(Hii * Hii + Hii1 * Hii1)
            if abs(gam - 0.0) <= EPSILON:
                gam = EPSMAC
            c[i], s[i] = Hii / gam, Hii1 / gam
            rs[i + 1] = -s[i] * rs[i]
            rs[i] = c[i] * rs[i]
            HH[i, i] = c[i] * Hii + s[i] * Hii1
            ro = abs(rs[i + 1])
            if i + 1 >= restrt or ro <= eps1 or iters >= maxiter:
                break
        rs[i] = rs[i] / HH[i, i]
        for ii in range(2, i + 2):
            k = i - ii + 1
            t = rs[k]
            for j in range(k + 1, i + 1):
                t -= HH[j, k] * rs[j]
            rs[k] = t / HH[k, k]
        for j in range(i + 1):
            x += rs[j] * W[j]
        if ro <= eps1 or iters >= maxiter:
            break
    return x, iters


def dsgesv_iteref(A, b):
    n = len(b)
    if np.abs(A).max() > np.finfo(np.float32).max:
        return None, -2  # dlag2s overflow -> fallback (not simulated)
    LU, piv = sgetrf(A)
    Anrm = np.abs(A).sum(axis=1).max()
    cte = Anrm * EPS * math.sqrt(n)
    itermax = min(n, 500)
    x, iters = fgmres(A, b, LU, piv, itermax, itermax, cte)
    return x, iters


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    rng = np.random.default_rng(1)
    A0 = rng.choice([-1.0, 1.0], (n, n)) * rng.uniform(0.5, 1.0, (n, n))
    xt = rng.uniform(-1, 1, n)
    b0 = A0 @ xt
    thr = math.sqrt(n) * EPS
    print(f"port of magma_dsgesv_iteref_gpu (GMSTRS), n = {n}, ||A0||_inf = {np.abs(A0).sum(1).max():.4g}")
    print(f"P2 predicts iter = 1 once ||A||_inf >= 1/(eps sqrt(n)) = {1 / (EPS * math.sqrt(n)):.3g}, i.e. scale >= "
          f"{1 / (EPS * math.sqrt(n)) / np.abs(A0).sum(1).max():.3g}")
    for k in (-80, -60, -40, -20, 0, 10, 20, 30, 40, 45, 50, 60):
        s = 2.0 ** k
        A, b = A0 * s, b0 * s
        x, it = dsgesv_iteref(A, b)
        an = np.abs(A).sum(1).max()
        tol = an * EPS * math.sqrt(n)
        r = b - A @ x
        bwd = np.abs(r).max() / (an * max(np.abs(x).max(), 1e-300)) / thr
        rel = np.linalg.norm(r) / np.linalg.norm(b)
        fe = np.abs(x - xt).max() / np.abs(xt).max()
        print(f"scale 2^{k:<4d} ||b||_2={np.linalg.norm(b):9.3g} tol={tol:9.3g} iter={it:4d} "
              f"rel.residual={rel:9.3g} (P1 bound {tol:9.3g}) bwd/thr={bwd:9.3g} fwd_err={fe:9.3g}")

"""CPU check of the GMSTRS fix (gmstrs_fix.patch, lean/OptimiKahan/MagmaFixed.lean) before any GPU run.

Original vs fixed, on the faithful port of ../fgmres_port.py, for an FP32 LU (dsgesv_iteref) and an FP16 LU
(stand-in for dhgesv_iteref's half-precision factorization), A = 2^k A0, b = 2^k b0, n = 256.
* original: ../fgmres_port.py unchanged (gmtol = cte, `ro <= 1e-18` break, no check, no fallback).
* fixed:    the same FGMRES with the patch's three changes (gmtol = sqrt(n)*eps, break only on ro == 0 with a scaled
            norm, FP16 range check), then the certified test, which is the C++ function from the patch compiled
            verbatim on the CPU (certify_cpu.cpp -> libcertify.dylib, called through ctypes); fail -> FP64 LU, iter < 0.
For every returned x the documented condition ||b - A x||_inf < sqrt(n) eps ||A||_inf ||x||_inf is decided EXACTLY
(Fractions).  The claim checked is fixed_post: iter >= 0 => exact condition holds.
Usage: python3 port_fixed.py /path/to/libcertify.dylib [n]
"""
import ctypes
import math
import sys
import time
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, "..")
import fgmres_port as P  # noqa: E402

EPS = 2.0 ** -53


def getrf(A, dtype):
    LU = A.astype(dtype).copy()
    n = LU.shape[0]
    piv = np.arange(n)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(n):
            p = k + int(np.nanargmax(np.abs(LU[k:, k]))) if not np.all(np.isnan(LU[k:, k])) else k
            if p != k:
                LU[[k, p]] = LU[[p, k]]
                piv[[k, p]] = piv[[p, k]]
            if LU[k, k] != 0:
                LU[k + 1:, k] /= LU[k, k]
                LU[k + 1:, k + 1:] -= np.outer(LU[k + 1:, k], LU[k, k + 1:])
    return LU, piv


def getrs(LU, piv, v):
    dt = LU.dtype
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = v[piv].astype(dt)
        n = len(y)
        for i in range(n):
            y[i] -= LU[i, :i] @ y[:i]
        for i in range(n - 1, -1, -1):
            y[i] = (y[i] - LU[i, i + 1:] @ y[i + 1:]) / LU[i, i]
    return y.astype(np.float64)


def fgmres(A, b, solve, maxiter, tol, fixed):
    """../fgmres_port.fgmres with the low-precision solve as a parameter; `fixed` = the patch's two FGMRES changes"""
    n = len(b)
    x = np.zeros(n)
    iters = 0
    restrt = maxiter
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        while True:
            r = b - A @ x
            ro = float(np.linalg.norm(r)) if fixed else math.sqrt(r @ r)
            if (ro == 0.0) if fixed else (abs(ro) <= P.EPSILON):
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
                W[i] = solve(V[i])
                V[i + 1] = A @ W[i]
                for j in range(i + 1):
                    HH[i, j] = V[j] @ V[i + 1]
                    V[i + 1] -= HH[i, j] * V[j]
                t = math.sqrt(V[i + 1] @ V[i + 1])
                HH[i, i + 1] = t
                if abs(t) > P.EPSILON:
                    V[i + 1] *= 1.0 / t
                for k in range(1, i + 1):
                    k1 = k - 1
                    t = HH[i, k1]
                    HH[i, k1] = c[k1] * t + s[k1] * HH[i, k]
                    HH[i, k] = -s[k1] * t + c[k1] * HH[i, k]
                Hii, Hii1 = HH[i, i], HH[i, i + 1]
                gam = math.sqrt(Hii * Hii + Hii1 * Hii1)
                if abs(gam) <= P.EPSILON:
                    gam = P.EPSMAC
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


def solve_iteref(A, b, low, fixed, lib):
    """returns (x, iter, path)"""
    n = len(b)
    itermax = min(n, 500)
    an = float(np.abs(A).sum(axis=1).max())
    cte = an * EPS * math.sqrt(n)
    if fixed and low == np.float16 and not (np.abs(A).max() <= 65504.0):
        return np.linalg.solve(A, b), -2, "fallback(range)"
    LU, piv = getrf(A, low)
    x, it = fgmres(A, b, lambda v: getrs(LU, piv, v), itermax, (EPS * math.sqrt(n)) if fixed else cte, fixed)
    if not fixed:
        return x, it, "returned"
    Af = np.asfortranarray(A)
    ok = lib.certify(0, n, 1, Af.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                     np.ascontiguousarray(b).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                     np.ascontiguousarray(x).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_double(1.0))
    if ok:
        return x, it, "certified"
    return np.linalg.solve(A, b), -(it + 1), "fallback(check)"


def exact_accept(A, b, x):
    if not np.all(np.isfinite(x)):
        return False, math.inf
    n = len(b)
    Aq = [[F(float(v)) for v in row] for row in A]
    xq = [F(float(v)) for v in x]
    rinf = max(abs(F(float(b[i])) - sum(Aq[i][j] * xq[j] for j in range(n))) for i in range(n))
    anrm = max(sum(abs(v) for v in row) for row in Aq)
    xinf = max(abs(v) for v in xq)
    rhs = F(math.sqrt(n)) * F(EPS) * xinf * anrm  # sqrt(n) exact for n = 256
    return rinf < rhs, (float(rinf / rhs) if rhs > 0 else math.inf)


def main():
    lib = ctypes.CDLL(sys.argv[1])
    lib.certify.restype = ctypes.c_int
    lib.certify.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    rng = np.random.default_rng(1)
    A0 = rng.choice([-1.0, 1.0], (n, n)) * rng.uniform(0.5, 1.0, (n, n))
    xt = rng.uniform(-1, 1, n)
    b0 = A0 @ xt
    print(f"n = {n}; documented condition decided exactly; ratio = ||r||_inf / (sqrt(n) eps ||A|| ||x||) (< 1 = holds)")
    viol = {"orig": 0, "fixed": 0}
    for low, name in ((np.float32, "FP32 LU (dsgesv_iteref)"), (np.float16, "FP16 LU (stand-in for dhgesv_iteref)")):
        print(f"\n{name}")
        for k in (-80, -70, -60, -20, 0, 10, 20, 30, 40, 60):
            A, b = A0 * 2.0 ** k, b0 * 2.0 ** k
            row = f"  k={k:4d} |"
            for fixed in (False, True):
                t0 = time.time()
                x, it, path = solve_iteref(A, b, low, fixed, lib)
                ok, ratio = exact_accept(A, b, x)
                if it >= 0 and not ok:
                    viol["fixed" if fixed else "orig"] += 1
                fe = float(np.max(np.abs(x - xt)) / np.max(np.abs(xt))) if np.all(np.isfinite(x)) else math.nan
                row += (f" {'fixed' if fixed else 'orig '}: iter={it:4d} {path:16s} ratio={ratio:9.3g} fwd={fe:8.2g}"
                        f" {'VIOLATES' if it >= 0 and not ok else 'ok      '} |")
            print(row, flush=True)
    print(f"\nreturned with iter >= 0 but violating the documented condition: original {viol['orig']}, "
          f"fixed {viol['fixed']}")


if __name__ == "__main__":
    main()

"""Forward-looking test of cuSOLVER mixed-precision iterative refinement (contract: CONTRACTS.md section 6).

Contract prediction: cuSOLVER documents convergence as RNRM < sqrt(N)*XNRM*ANRM*EPS*BWDMAX (a '<' test), so a
NaN/Inf residual is NOT accepted; with fallback (default) the solution is recomputed in FP64.
Inputs: scaled Wilkinson growth matrices s*W(n) (W_ii = 1, W_ij = -1 for i > j, W_in = 1), x_true = ones,
b = A x_true (exact).  Partial pivoting grows the last column of U to s*2^(n-1): it overflows FP16 (65504) or
FP32 (~2^128) while FP64 stays exact (n <= 40).  Also: NaN in b; random well-conditioned control.
Reports iter (negative = documented failure code), info, max |x - 1| (or NaN count).
"""
import ctypes as C
import math
import os
import sys

import numpy as np

BASE = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64")
os.add_dll_directory(BASE)
cudart = C.WinDLL(os.path.join(BASE, "cudart64_13.dll"))
cusolver = C.WinDLL(os.path.join(BASE, "cusolver64_12.dll"))

R_64F, R_32F, R_16F = 1203, 1204, 1205
REFINE_CLASSICAL = 1102


def ck(st, what):
    if st != 0:
        raise RuntimeError(f"{what}: cusolver status {st}")


def cuda_ck(err, what):
    if err != 0:
        f = cudart.cudaGetErrorString
        f.restype = C.c_char_p
        raise RuntimeError(f"{what}: {f(err).decode()}")


def dmalloc(nbytes):
    p = C.c_void_p()
    cuda_ck(cudart.cudaMalloc(C.byref(p), C.c_size_t(max(nbytes, 8))), "cudaMalloc")
    return p


def h2d(p, a):
    a = np.ascontiguousarray(a)
    cuda_ck(cudart.cudaMemcpy(p, a.ctypes.data_as(C.c_void_p), C.c_size_t(a.nbytes), 1), "H2D")


def d2h(p, dtype, count):
    out = np.empty(count, dtype=dtype)
    cuda_ck(cudart.cudaMemcpy(out.ctypes.data_as(C.c_void_p), p, C.c_size_t(out.nbytes), 2), "D2H")
    return out


h = C.c_void_p()
ck(cusolver.cusolverDnCreate(C.byref(h)), "cusolverDnCreate")
ver = C.c_int()
cusolver.cusolverGetVersion(C.byref(ver))


def wilkinson(n, s):
    A = np.eye(n)
    A[np.tril_indices(n, -1)] = -1.0
    A[:, -1] = 1.0
    return A * s


def solve(kind, A, b, fallback=True):
    """kind in {'DH','DS','D'}: DHgesv / DSgesv / IRSXgesv(64F,64F) as FP64 reference; returns x, iter, info."""
    n = A.shape[0]
    Af = np.asfortranarray(A).ravel(order="F")
    dA, dB, dX = dmalloc(Af.nbytes), dmalloc(b.nbytes), dmalloc(b.nbytes)
    dinfo = dmalloc(4)
    h2d(dA, Af)
    h2d(dB, b.astype(np.float64))
    h2d(dX, np.full(n, 777.0))
    it = C.c_int(12345)
    if kind in ("DH", "DS") and fallback:
        dipiv = dmalloc(4 * n)
        lw = C.c_size_t()
        fb = getattr(cusolver, f"cusolverDn{kind}gesv_bufferSize")
        ck(fb(h, n, 1, dA, n, dipiv, dB, n, dX, n, None, C.byref(lw)), "bufferSize")
        dW = dmalloc(lw.value)
        f = getattr(cusolver, f"cusolverDn{kind}gesv")
        ck(f(h, n, 1, dA, n, dipiv, dB, n, dX, n, dW, C.c_size_t(lw.value), C.byref(it), dinfo), kind + "gesv")
    else:
        params, infos = C.c_void_p(), C.c_void_p()
        ck(cusolver.cusolverDnIRSParamsCreate(C.byref(params)), "ParamsCreate")
        low = {"DH": R_16F, "DS": R_32F, "D": R_64F}[kind]
        ck(cusolver.cusolverDnIRSParamsSetSolverPrecisions(params, R_64F, low), "SetPrecisions")
        ck(cusolver.cusolverDnIRSParamsSetRefinementSolver(params, REFINE_CLASSICAL), "SetRefinement")
        if not fallback:
            ck(cusolver.cusolverDnIRSParamsDisableFallback(params), "DisableFallback")
        ck(cusolver.cusolverDnIRSInfosCreate(C.byref(infos)), "InfosCreate")
        lw = C.c_size_t()
        ck(cusolver.cusolverDnIRSXgesv_bufferSize(h, params, n, 1, C.byref(lw)), "IRS bufferSize")
        dW = dmalloc(lw.value)
        ck(cusolver.cusolverDnIRSXgesv(h, params, infos, n, 1, dA, n, dB, n, dX, n, dW, C.c_size_t(lw.value),
                                       C.byref(it), dinfo), "IRSXgesv")
    cuda_ck(cudart.cudaDeviceSynchronize(), "sync")
    x = d2h(dX, np.float64, n)
    info = int(d2h(dinfo, np.int32, 1)[0])
    return x, it.value, info


def describe(x, xtrue):
    bad = int(np.sum(~np.isfinite(x)))
    err = float(np.max(np.abs(x[np.isfinite(x)] - xtrue[np.isfinite(x)]))) if bad < len(x) else math.nan
    return f"nonfinite={bad:3d}/{len(x)} max|x-x_true|={err:.3g}"


if __name__ == "__main__":
    print(f"cuSOLVER version {ver.value}", flush=True)
    rng = np.random.default_rng(4)
    cases = []
    for n, s in [(10, 2.0 ** 10), (20, 1.0), (40, 1.0), (16, 1.0)]:
        cases.append((f"W({n})*2^{int(math.log2(s))}  growth 2^{n - 1 + int(math.log2(s))}", wilkinson(n, s)))
    for n, s in [(40, 2.0 ** 80), (40, 2.0 ** 100), (30, 2.0 ** 120)]:
        cases.append((f"W({n})*2^{int(math.log2(s))} growth 2^{n - 1 + int(math.log2(s))}", wilkinson(n, s)))
    R = rng.uniform(-1, 1, (256, 256)) + 256 * np.eye(256)
    cases.append(("random diag-dominant 256", R))
    for name, A in cases:
        n = A.shape[0]
        xt = np.ones(n)
        b = A @ xt  # exact for the Wilkinson cases (small integer multiples of a power of two)
        print(f"\n{name}", flush=True)
        for kind, fb in [("D", True), ("DH", True), ("DS", True), ("DH", False), ("DS", False)]:
            x, it, info = solve(kind, A, b, fb)
            label = {"D": "FP64 (IRS 64/64)", "DH": "FP16 factor", "DS": "FP32 factor"}[kind] + ("" if fb else ", no fallback")
            print(f"  {label:26s} iter={it:5d} info={info:3d}  {describe(x, xt)}", flush=True)
    # NaN in b
    A = R
    b = A @ np.ones(256)
    b[0] = math.nan
    print("\nrandom 256 with b[0] = NaN", flush=True)
    for kind, fb in [("D", True), ("DH", True), ("DS", True), ("DS", False)]:
        x, it, info = solve(kind, A, b, fb)
        print(f"  {kind:3s} fallback={fb!s:5s} iter={it:5d} info={info:3d}  nonfinite={int(np.sum(~np.isfinite(x)))}/256", flush=True)

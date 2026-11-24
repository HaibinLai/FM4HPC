"""Black-box contract test of cuSOLVER mixed-precision iterative refinement (cusolverDnIRSXgesv), roxy / RTX 5070 Ti.

Contract (cuSOLVER 13.5 docs, sec. 2.4.2.6/2.4.2.7/2.4.1.45-50):
  R  : stop when RNRM < sqrt(N) * XNRM * ANRM * EPS * BWDMAX (BWDMAX = 1, inf-norms) or after MaxIters (50);
       niters < 0 on failure (-1 not worth it, -2 overflow converting to lower precision, -3 factorization failure,
       -5 overflow during computation, -maxiter), then fallback to main-precision getrf/getrs (default ON).
  G  : "a solution with same normwise backward error as the main precision"; with fallback "always a good solution";
       with fallback disabled "does not guarantee that the solution is the good one" (but niters < 0 flags it).
Families (each derived from a gap in R):
  S  scale sweep      : the -2 check looks at entries, not at LU growth -> overflow inside the low-precision LU
  T  tiny scale       : underflow to FP16 subnormals/zero is not an overflow -> is it detected?
  K  conditioning     : convergence region of each refinement solver; forward vs backward error
  N  non-finite / edge: NaN/Inf in b or A, b = 0, Wilkinson growth matrix (finite FP64 LU, FP16 LU overflows)
Per run: niters, outer iters, last residual-history norm, time, and our own FP64 normwise backward error
  eta = ||b - A x||_inf / (||A||_inf ||x||_inf) (and its ratio to the documented threshold sqrt(n) eps),
  forward error vs an FP64 cusolverDnDDgesv-equivalent (IRS with lowest = 64F) solution, and NaN flags.
Usage: python irs_probe.py [family letters, default SNTK]
"""
import ctypes as C
import math
import os
import sys
import time

import numpy as np

BASE = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64")
os.add_dll_directory(BASE)
cudart = C.WinDLL(os.path.join(BASE, "cudart64_13.dll"))
cusolver = C.WinDLL(os.path.join(BASE, "cusolver64_12.dll"))

R_64F, R_32F, R_16F, R_16BF, R_TF32 = 1203, 1204, 1205, 1206, 1207
REF_CLASSICAL, REF_CLASSICAL_GMRES, REF_GMRES, REF_GMRES_GMRES = 1102, 1103, 1104, 1105
EPS = 2.0 ** -53  # LAPACK dlamch('Epsilon') = 2^-53 (rounding)
PREC = {"64F": R_64F, "32F": R_32F, "16F": R_16F, "16BF": R_16BF, "TF32": R_TF32}
REF = {"CL": REF_CLASSICAL, "GM": REF_GMRES, "CLGM": REF_CLASSICAL_GMRES, "GMGM": REF_GMRES_GMRES}


def cuda_ck(err, what):
    if err != 0:
        f = cudart.cudaGetErrorString
        f.restype = C.c_char_p
        raise RuntimeError(f"{what}: {f(err).decode()}")


def ck(st, what):
    if st != 0:
        raise RuntimeError(f"{what}: cusolver status {st}")


class Dev:
    def __init__(self, nbytes):
        self.p = C.c_void_p()
        cuda_ck(cudart.cudaMalloc(C.byref(self.p), C.c_size_t(max(nbytes, 8))), "cudaMalloc")

    def put(self, a):
        a = np.ascontiguousarray(a)
        cuda_ck(cudart.cudaMemcpy(self.p, a.ctypes.data_as(C.c_void_p), C.c_size_t(a.nbytes), 1), "H2D")

    def get(self, dtype, count):
        out = np.empty(count, dtype=dtype)
        cuda_ck(cudart.cudaMemcpy(out.ctypes.data_as(C.c_void_p), self.p, C.c_size_t(out.nbytes), 2), "D2H")
        return out

    def free(self):
        cudart.cudaFree(self.p)


h = C.c_void_p()
ck(cusolver.cusolverDnCreate(C.byref(h)), "cusolverDnCreate")
ver = C.c_int()
cusolver.cusolverGetVersion(C.byref(ver))
cusolver.cusolverDnIRSParamsSetTol.argtypes = [C.c_void_p, C.c_double]
cusolver.cusolverDnIRSXgesv_bufferSize.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_size_t)]
cusolver.cusolverDnIRSXgesv.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int,
                                        C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_void_p, C.c_size_t,
                                        C.POINTER(C.c_int), C.c_void_p]


def irs(A, b, lowest="16F", ref="CL", fallback=True, maxiters=None):
    """x = A \\ b with IRSXgesv (main precision FP64).  Returns dict."""
    n = A.shape[0]
    b2 = b.reshape(n, -1)
    nrhs = b2.shape[1]
    params, infos = C.c_void_p(), C.c_void_p()
    ck(cusolver.cusolverDnIRSParamsCreate(C.byref(params)), "ParamsCreate")
    ck(cusolver.cusolverDnIRSParamsSetSolverPrecisions(params, R_64F, PREC[lowest]), "SetPrec")
    ck(cusolver.cusolverDnIRSParamsSetRefinementSolver(params, REF[ref]), "SetRef")
    if maxiters:
        ck(cusolver.cusolverDnIRSParamsSetMaxIters(params, maxiters), "SetMaxIters")
    ck((cusolver.cusolverDnIRSParamsEnableFallback if fallback else cusolver.cusolverDnIRSParamsDisableFallback)(params),
       "Fallback")
    ck(cusolver.cusolverDnIRSInfosCreate(C.byref(infos)), "InfosCreate")
    ck(cusolver.cusolverDnIRSInfosRequestResidual(infos), "RequestResidual")
    lw = C.c_size_t()
    ck(cusolver.cusolverDnIRSXgesv_bufferSize(h, params, n, nrhs, C.byref(lw)), "bufferSize")
    dA, dB, dX, dW, dI = Dev(A.nbytes), Dev(b2.nbytes), Dev(b2.nbytes), Dev(lw.value), Dev(4)
    dA.put(np.asfortranarray(A).ravel(order="F"))
    dB.put(np.asfortranarray(b2).ravel(order="F"))
    dX.put(np.full(n * nrhs, 777.0))
    dI.put(np.array([-12345], dtype=np.int32))
    niters = C.c_int(-12345)
    cuda_ck(cudart.cudaDeviceSynchronize(), "sync")
    t0 = time.perf_counter()
    st = cusolver.cusolverDnIRSXgesv(h, params, infos, n, nrhs, dA.p, n, dB.p, n, dX.p, n, dW.p, lw, C.byref(niters), dI.p)
    cuda_ck(cudart.cudaDeviceSynchronize(), "sync")
    dt = time.perf_counter() - t0
    x = dX.get(np.float64, n * nrhs).reshape((n, nrhs), order="F")
    info = int(dI.get(np.int32, 1)[0])
    outer = C.c_int(-12345)
    cusolver.cusolverDnIRSInfosGetOuterNiters(infos, C.byref(outer))
    mi = C.c_int()
    cusolver.cusolverDnIRSInfosGetMaxIters(infos, C.byref(mi))
    hist = C.c_void_p()
    last_r = float("nan")
    hst = cusolver.cusolverDnIRSInfosGetResidualHistory(infos, C.byref(hist))
    hrows = []
    if hst == 0 and hist.value:
        ldh = mi.value + 1
        arr = np.ctypeslib.as_array(C.cast(hist, C.POINTER(C.c_double)), shape=(2 * ldh,))
        nvalid = max(0, min(ldh, abs(outer.value) + 1))
        hrows = [(arr[i], arr[i + ldh]) for i in range(nvalid)]
        if hrows:
            last_r = float(hrows[-1][1])
    for d in (dA, dB, dX, dW, dI):
        d.free()
    cusolver.cusolverDnIRSInfosDestroy(infos)
    cusolver.cusolverDnIRSParamsDestroy(params)
    return dict(x=x.reshape(b.shape), status=st, niters=niters.value, outer=outer.value, info=info, t=dt,
                last_r=last_r, hist=hrows)


def eta(A, x, b):
    """normwise backward error with FP64 residual, and the documented threshold ratio."""
    if not np.all(np.isfinite(x)):
        return math.nan, math.nan
    r = b - A @ x
    rn = np.abs(r).max()
    an = np.abs(A).sum(axis=1).max()
    xn = np.abs(x).max()
    if an * xn == 0:
        return (0.0, 0.0) if rn == 0 else (math.inf, math.inf)
    e = rn / (an * xn)
    return e, e / (math.sqrt(A.shape[0]) * EPS)


CONFIGS = [("16F", "CL"), ("16F", "GM"), ("16BF", "GM"), ("TF32", "GM"), ("32F", "CL")]


def report(name, A, b, configs=CONFIGS, fallback_off=True):
    ref = irs(A, b, "64F", "CL")
    xr = ref["x"]
    e_ref, q_ref = eta(A, xr, b)
    print(f"\n{name}: n={A.shape[0]}  FP64 (lowest=64F): niters={ref['niters']} {ref['t'] * 1e3:.1f} ms "
          f"eta/(sqrt(n)eps)={q_ref:.3g} nan={int(np.isnan(xr).sum())}", flush=True)
    fbs = (True, False) if fallback_off else (True,)
    for low, rf in configs:
        for fb in fbs:
            r = irs(A, b, low, rf, fallback=fb)
            x = r["x"]
            e, q = eta(A, x, b)
            nn = int((~np.isfinite(x)).sum())
            fe = (np.abs(x - xr).max() / np.abs(xr).max()) if np.all(np.isfinite(xr)) and np.abs(xr).max() > 0 else math.nan
            flag = ""
            if r["niters"] >= 0 and (nn or not (q < 1.0)):
                flag = "   <-- accepted but eta above threshold / non-finite"
            print(f"   {low:4s}/{rf:4s} fb={'on ' if fb else 'off'} niters={r['niters']:4d} outer={r['outer']:4d} "
                  f"info={r['info']} st={r['status']} {r['t'] * 1e3:7.1f} ms  eta/thr={q:9.3g} fwd_vs_fp64={fe:9.3g} "
                  f"nonfinite={nn} last_hist_rnrm={r['last_r']:.3g}{flag}", flush=True)


rng = np.random.default_rng(7)


def gen(n, scale=1.0):
    A = rng.uniform(-1, 1, (n, n)) * scale
    return A, A @ rng.uniform(-1, 1, n)


def with_cond(n, kappa):
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.logspace(0, -math.log10(kappa), n)
    A = (Q1 * s) @ Q2.T
    return A, A @ rng.uniform(-1, 1, n)


def wilkinson(n):
    A = np.tril(-np.ones((n, n)), -1) + np.eye(n)
    A[:, -1] = 1.0
    return A, A @ rng.uniform(-1, 1, n)


if __name__ == "__main__":
    fam = sys.argv[1] if len(sys.argv) > 1 else "SNTK"
    print(f"cuSOLVER version {ver.value}; EPS = 2^-53; threshold sqrt(n)*eps", flush=True)
    n = 2048
    if "S" in fam:
        for s in (1.0, 1e2, 1e3, 3e3, 1e4, 3e4, 6e4, 7e4, 1e6):
            A, b = gen(n, s)
            report(f"S scale={s:g} (max|a|={np.abs(A).max():.4g})", A, b, configs=[("16F", "CL"), ("16F", "GM"), ("16BF", "GM")])
    if "T" in fam:
        for s in (1e-4, 1e-6, 1e-8, 1e-10, 1e-30):
            A, b = gen(n, s)
            report(f"T scale={s:g}", A, b, configs=[("16F", "CL"), ("16F", "GM"), ("16BF", "GM")])
    if "N" in fam:
        A, b = gen(n)
        b1 = b.copy(); b1[5] = math.nan
        report("N b[5]=NaN", A, b1)
        A1 = A.copy(); A1[3, 7] = math.inf
        report("N A[3,7]=Inf", A1, b)
        A1 = A.copy(); A1[3, 7] = math.nan
        report("N A[3,7]=NaN", A1, b)
        report("N b=0", A, np.zeros(n))
        for m in (512, 1000):
            A, b = wilkinson(m)
            report(f"N wilkinson growth 2^{m - 1}", A, b)
    if "K" in fam:
        for kappa in (1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e15):
            A, b = with_cond(n, kappa)
            report(f"K kappa={kappa:g}", A, b, fallback_off=False)

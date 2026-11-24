"""Scale-invariance test of cuSOLVER IRS (derived from the contract: the documented stopping rule
RNRM < sqrt(N)*XNRM*ANRM*EPS is invariant under (A, b) -> (2^k A, 2^k b), and IEEE arithmetic commutes exactly with
2^k scaling while no value overflows or becomes subnormal).  A has entries +-[0.5, 1), so 2^k A is in the FP16 normal
range for k in [-13, 15] and in the BF16/FP32 normal range for |k| <= ~120.  Any change of x (bitwise) or of niters
across k inside that range proves a non-scale-invariant constant in the implementation.

Also: exact (rational) residual of one accepted solution, to rule out residual rounding; legacy cusolverDnDHgesv/DBgesv.
Usage (roxy): python irs_scale.py
"""
import ctypes as C
import math
from fractions import Fraction

import numpy as np

import irs_probe as P

rng = np.random.default_rng(31)
n = 2048
A0 = rng.choice([-1.0, 1.0], (n, n)) * rng.uniform(0.5, 1.0, (n, n))
xt = rng.uniform(-1, 1, n)
b0 = A0 @ xt

cusolver = P.cusolver
for nm in ("cusolverDnDHgesv", "cusolverDnDBgesv"):
    getattr(cusolver, nm + "_bufferSize").argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_void_p,
                                                      C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_void_p,
                                                      C.POINTER(C.c_size_t)]
    getattr(cusolver, nm).argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int,
                                      C.c_void_p, C.c_int, C.c_void_p, C.c_size_t, C.POINTER(C.c_int), C.c_void_p]


def legacy(nm, A, b):
    n = A.shape[0]
    dA, dB, dX, dP, dI = P.Dev(A.nbytes), P.Dev(b.nbytes), P.Dev(b.nbytes), P.Dev(4 * n), P.Dev(4)
    dA.put(np.asfortranarray(A).ravel(order="F"))
    dB.put(b)
    lw = C.c_size_t()
    P.ck(getattr(cusolver, nm + "_bufferSize")(P.h, n, 1, dA.p, n, dP.p, dB.p, n, dX.p, n, None, C.byref(lw)), "bs")
    dW = P.Dev(lw.value)
    it = C.c_int(-12345)
    st = getattr(cusolver, nm)(P.h, n, 1, dA.p, n, dP.p, dB.p, n, dX.p, n, dW.p, lw, C.byref(it), dI.p)
    P.cuda_ck(P.cudart.cudaDeviceSynchronize(), "sync")
    x = dX.get(np.float64, n)
    info = int(dI.get(np.int32, 1)[0])
    for d in (dA, dB, dX, dP, dI, dW):
        d.free()
    return dict(x=x, niters=it.value, info=info, status=st)


def exact_eta(A, x, b):
    """exact ||b - A x||_inf / (||A||_inf ||x||_inf), rational arithmetic over integer mantissas"""
    def dec(v):
        m, e = math.frexp(float(v))
        return int(m * (1 << 53)), e - 53
    xd = [dec(v) for v in x]
    worst = Fraction(0)
    for i in range(A.shape[0]):
        terms = [dec(b[i])]
        terms[0] = (-terms[0][0], terms[0][1])
        for j in range(A.shape[1]):
            ma, ea = dec(A[i, j])
            terms.append((ma * xd[j][0], ea + xd[j][1]))
        e0 = min(t[1] for t in terms)
        s = abs(sum(t[0] << (t[1] - e0) for t in terms))
        r = Fraction(s) * Fraction(2) ** e0
        worst = max(worst, r)
    an = Fraction(float(np.abs(A).sum(axis=1).max()))  # FP64 norm of A: fine to ~1e-13 relative
    xn = Fraction(float(np.abs(x).max()))
    return float(worst / (an * xn)) / (math.sqrt(A.shape[0]) * P.EPS)


if __name__ == "__main__":
    print(f"cuSOLVER {P.ver.value}; n={n}; A entries +-[0.5,1); x_true uniform; threshold sqrt(n)*eps", flush=True)
    for low, rf, ks in (("16F", "GM", (-12, -8, -4, -2, 0, 2, 4, 8, 12, 15)),
                        ("16F", "CL", (-12, -4, 0, 4, 12)),
                        ("16BF", "GM", (-100, -40, -12, -4, 0, 4, 12, 40, 100)),
                        ("32F", "GM", (-100, -12, 0, 12, 100))):
        base = None
        print(f"\n{low}/{rf}:", flush=True)
        for k in ks:
            s = 2.0 ** k
            A, b = A0 * s, b0 * s
            r = P.irs(A, b, low, rf)
            x = r["x"]
            e, q = P.eta(A, x, b)
            fe = np.abs(x - xt).max() / np.abs(xt).max()
            if k == 0:
                base = x
            same = "" if base is None or k == 0 else ("  x bitwise = k=0" if np.array_equal(x, base) else "  x DIFFERS from k=0")
            print(f"   k={k:5d} niters={r['niters']:4d} eta/thr={q:9.3g} fwd_err_vs_xtrue={fe:9.3g} {r['t'] * 1e3:6.1f} ms{same}",
                  flush=True)
        # rerun the k < 0 ones now that base exists
        for k in ks:
            if k >= 0:
                break
            s = 2.0 ** k
            x = P.irs(A0 * s, b0 * s, low, rf)["x"]
            print(f"   (recheck k={k}) x {'bitwise = k=0' if np.array_equal(x, base) else 'DIFFERS from k=0'}", flush=True)
    print("\nlegacy API (default settings), k = 0 and k = 12:", flush=True)
    for nm in ("cusolverDnDHgesv", "cusolverDnDBgesv"):
        for k in (0, 12):
            s = 2.0 ** k
            r = legacy(nm, A0 * s, b0 * s)
            e, q = P.eta(A0 * s, r["x"], b0 * s)
            fe = np.abs(r["x"] - xt).max() / np.abs(xt).max()
            print(f"   {nm} k={k:3d} niters={r['niters']:4d} info={r['info']} status={r['status']} eta/thr={q:9.3g} "
                  f"fwd_err={fe:9.3g}", flush=True)
    # exact residual of one accepted GMRES solution at k = 12 (FP16), n reduced for the rational check
    m = 512
    As, bs = A0[:m, :m] * 2.0 ** 12, (A0[:m, :m] @ xt[:m]) * 2.0 ** 12
    r = P.irs(As, bs, "16F", "GM")
    print(f"\nexact check n={m} 16F/GM k=12: niters={r['niters']} eta/thr(FP64 residual)={P.eta(As, r['x'], bs)[1]:.4g} "
          f"eta/thr(exact residual)={exact_eta(As, r['x'], bs):.4g}", flush=True)

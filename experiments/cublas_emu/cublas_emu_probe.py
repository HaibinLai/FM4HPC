"""Black-box probe of cuBLAS FP64 emulation (Fixed-Point / Ozaki, ADP) on sm_120 via ctypes (no compiler).

The docs promise (cuBLAS 1.5.2.1 / 1.5.4): ADP "computes the proper number of fixed-point mantissa bits required to
maintain accuracy equal or better than FP64" and the emulation supports "the full range of finite input values
(including denormals)"; Fixed-Point implicitly supports no Inf/NaN (needs the special-values mask).

For every case and mode we report, per output matrix:
  comp  = max_ij |C - ref| / (u * (|A||B|)_ij)      DGEMM's a-priori bound is about k (gamma_k)
  >k    = number of entries whose comp exceeds k (worse than what FP64 GEMM guarantees)
  bits  = mantissa bits retained, read from the device pointer set by
          cublasSetFixedPointEmulationMantissaBitCountPointer (-1 = native FP64 was used)
plus non-finite handling and -0.  ref is exact (Python integers).
Usage: python cublas_emu_probe.py [case-substring]
"""
import ctypes as C
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np

BASE = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64")
os.add_dll_directory(BASE)
cudart = C.WinDLL(os.path.join(BASE, "cudart64_13.dll"))
cublas = C.WinDLL(os.path.join(BASE, "cublas64_13.dll"))

# enums (cublas_api.h / library_types.h of cuBLAS 13.10, CUDA runtime 13.5)
MATH_DEFAULT, MATH_FP64_EMU = 0, 8
STRAT_PERFORMANT, STRAT_EAGER = 1, 2
MANT_DYNAMIC, MANT_FIXED = 0, 1
SV_DEFAULT, SV_NONE = 0xFFFF, 0
U = 2.0 ** -53


def ck(status, what):
    if status != 0:
        name = cublas.cublasGetStatusName
        name.restype = C.c_char_p
        raise RuntimeError(f"{what} failed: status {status} ({name(status).decode()})")


def cuda_ck(err, what):
    if err != 0:
        f = cudart.cudaGetErrorString
        f.restype = C.c_char_p
        raise RuntimeError(f"{what} failed: {f(err).decode()}")


class Dev:
    def __init__(self, nbytes):
        self.p = C.c_void_p()
        cuda_ck(cudart.cudaMalloc(C.byref(self.p), C.c_size_t(max(nbytes, 8))), "cudaMalloc")
        self.n = nbytes

    def put(self, arr):
        a = np.ascontiguousarray(arr)
        cuda_ck(cudart.cudaMemcpy(self.p, a.ctypes.data_as(C.c_void_p), C.c_size_t(a.nbytes), 1), "H2D")

    def get(self, dtype, count):
        out = np.empty(count, dtype=dtype)
        cuda_ck(cudart.cudaMemcpy(out.ctypes.data_as(C.c_void_p), self.p, C.c_size_t(out.nbytes), 2), "D2H")
        return out

    def free(self):
        cudart.cudaFree(self.p)


h = C.c_void_p()
ck(cublas.cublasCreate_v2(C.byref(h)), "cublasCreate")
ver = C.c_int()
ck(cublas.cublasGetVersion_v2(h, C.byref(ver)), "cublasGetVersion")
bits_dev = Dev(4)

cublas.cublasDgemm_v2.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_double),
                                  C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.POINTER(C.c_double), C.c_void_p, C.c_int]


def configure(mode):
    ck(cublas.cublasSetMathMode(h, MATH_DEFAULT if mode == "native" else MATH_FP64_EMU), "SetMathMode")
    if mode == "native":
        return
    ck(cublas.cublasSetEmulationStrategy(h, STRAT_PERFORMANT if mode == "emu_perf" else STRAT_EAGER), "SetStrategy")
    ck(cublas.cublasSetEmulationSpecialValuesSupport(h, SV_NONE if mode == "emu_eager_sv0" else SV_DEFAULT), "SetSV")
    if mode.startswith("emu_fixed"):
        ck(cublas.cublasSetFixedPointEmulationMantissaControl(h, MANT_FIXED), "SetMantControl")
        ck(cublas.cublasSetFixedPointEmulationMaxMantissaBitCount(h, int(mode[len("emu_fixed"):])), "SetMaxBits")
    else:
        ck(cublas.cublasSetFixedPointEmulationMantissaControl(h, MANT_DYNAMIC), "SetMantControl")
    ck(cublas.cublasSetFixedPointEmulationMantissaBitCountPointer(h, C.cast(bits_dev.p, C.POINTER(C.c_int))),
       "SetBitCountPointer")


def dgemm(mode, A, B):
    """C = A @ B with the given cuBLAS mode; returns (C, bits, seconds)."""
    m, k = A.shape
    _, n = B.shape
    configure(mode)
    dA, dB, dC = Dev(A.nbytes), Dev(B.nbytes), Dev(m * n * 8)
    dA.put(np.asfortranarray(A).ravel(order="F"))
    dB.put(np.asfortranarray(B).ravel(order="F"))
    dC.put(np.full(m * n, 12345.0))
    bits_dev.put(np.array([-2], dtype=np.int32))
    one, zero = C.c_double(1.0), C.c_double(0.0)
    cudart.cudaDeviceSynchronize()
    t0 = time.perf_counter()
    ck(cublas.cublasDgemm_v2(h, 0, 0, m, n, k, C.byref(one), dA.p, m, dB.p, k, C.byref(zero), dC.p, m), "Dgemm")
    cuda_ck(cudart.cudaDeviceSynchronize(), "sync")
    dt = time.perf_counter() - t0
    Cm = dC.get(np.float64, m * n).reshape((m, n), order="F")
    bits = int(bits_dev.get(np.int32, 1)[0])
    for d in (dA, dB, dC):
        d.free()
    return Cm, bits, dt


# ---------------- exact reference ----------------
def decompose(x):
    """finite double -> (M, E) with x = M * 2^E exactly (M int)."""
    if x == 0.0:
        return 0, 0
    m, e = math.frexp(x)
    return int(m * (1 << 53)), e - 53


def exact_ref(A, B):
    m, k = A.shape
    n = B.shape[1]
    Ad = [[decompose(float(A[i, l])) if math.isfinite(A[i, l]) else None for l in range(k)] for i in range(m)]
    Bd = [[decompose(float(B[l, j])) if math.isfinite(B[l, j]) else None for l in range(k)] for j in range(n)]
    ref = [[None] * n for _ in range(m)]
    aab = [[None] * n for _ in range(m)]
    for i in range(m):
        ai = Ad[i]
        for j in range(n):
            bj = Bd[j]
            terms = []
            bad = False
            for l in range(k):
                a, b = ai[l], bj[l]
                if a is None or b is None:
                    bad = True
                    break
                if a[0] and b[0]:
                    terms.append((a[0] * b[0], a[1] + b[1]))
            if bad:
                continue
            if not terms:
                ref[i][j] = Fraction(0)
                aab[i][j] = Fraction(0)
                continue
            e0 = min(t[1] for t in terms)
            s = sum(t[0] << (t[1] - e0) for t in terms)
            sa = sum(abs(t[0]) << (t[1] - e0) for t in terms)
            sc = Fraction(2) ** e0
            ref[i][j] = s * sc
            aab[i][j] = sa * sc
    return ref, aab


def evaluate(Cm, ref, aab, A, B):
    m, n = Cm.shape
    k = A.shape[1]
    comp, over_k, wrong, nonfin_bad, negzero = 0.0, 0, 0, 0, 0
    ref_nonfinite = 0
    for i in range(m):
        for j in range(n):
            c = float(Cm[i, j])
            r = ref[i][j]
            if r is None:  # an Inf/NaN input feeds this entry
                ref_nonfinite += 1
                if math.isfinite(c):
                    nonfin_bad += 1
                continue
            if not math.isfinite(c):
                nonfin_bad += 1
                continue
            e = abs(Fraction(c) - r)
            if aab[i][j] > 0:
                q = float(e / (aab[i][j] * Fraction(U)))
                comp = max(comp, q)
                if q > k:
                    over_k += 1
            elif e > 0:
                comp = math.inf
            if r != 0 and e >= abs(r):
                wrong += 1
            if r == 0 and c == 0 and math.copysign(1, c) < 0:
                negzero += 1
    return dict(comp=comp, over_k=over_k, wrong=wrong, nonfin_bad=nonfin_bad, ref_nonfinite=ref_nonfinite, negzero=negzero)


# ---------------- cases ----------------
rng = np.random.default_rng(3)
Ur = lambda *s: rng.uniform(-1.0, 1.0, size=s)
spread = lambda s, X: np.ldexp(Ur(*s), rng.integers(0, X + 1, size=s))
cases = []


def case(name, m, n, k, fill):
    cases.append((name, m, n, k, fill))


case("rand_s0", 128, 128, 512, lambda m, n, k: (Ur(m, k), Ur(k, n)))
case("rand_s30", 128, 128, 512, lambda m, n, k: (spread((m, k), 30), spread((k, n), 30)))
case("rand_s60", 128, 128, 512, lambda m, n, k: (spread((m, k), 60), spread((k, n), 60)))
case("rand_s200", 64, 64, 256, lambda m, n, k: (spread((m, k), 200), spread((k, n), 200)))


def gram_pos(m, n, k):
    A = rng.uniform(0.9, 1.0, size=(m, k))
    return A, A.T.copy()


case("gram_pos", 64, 64, 2048, gram_pos)


def avg_hides(m, n, k):
    A, B = Ur(m, k), Ur(k, n)
    A[0, :] = 2.0 ** -100
    A[0, 0] = 1.0
    B[:, 0] = 1.0
    B[0, 0] = 2.0 ** -100
    return A, B


case("avg_hides", 64, 64, 64, avg_hides)


def zero_lane0(m, n, k):
    A, B = spread((m, k), 30), spread((k, n), 30)
    A[:, ::32] = 0.0
    B[::32, :] = 0.0
    return A, B


case("zero_lane0", 64, 64, 256, zero_lane0)


def tri(m, n, k, X=60):
    A, B = spread((m, k), X), spread((k, n), X)
    return np.triu(A), np.tril(B)


case("tri_s60", 128, 128, 128, tri)


def wide_row(m, n, k):
    A, B = Ur(m, k), Ur(k, n)
    A[0, :] = np.ldexp(Ur(k), rng.integers(-500, 501, size=k))
    return A, B


case("wide_row0", 64, 64, 256, wide_row)


def cancel(m, n, k):
    A, B = Ur(m, k), Ur(k, n)
    A[0, :] = 0.0
    A[0, 0], A[0, 1], A[0, 2] = 2.0 ** 60, 1.0, -(2.0 ** 60)
    B[0:3, :] = 1.0
    return A, B


case("cancel_row0", 64, 64, 256, cancel)
case("big511", 64, 64, 256, lambda m, n, k: (np.ldexp(Ur(m, k), 511), np.ldexp(Ur(k, n), -511)))
case("bigB515", 64, 64, 256, lambda m, n, k: (Ur(m, k), np.ldexp(Ur(k, n), 515)))
case("big1000", 64, 64, 256, lambda m, n, k: (np.ldexp(Ur(m, k), 1000), np.ldexp(Ur(k, n), -1000)))
case("tinyA600", 64, 64, 256, lambda m, n, k: (np.ldexp(Ur(m, k), -600), Ur(k, n)))
case("tinyA1000", 64, 64, 256, lambda m, n, k: (np.ldexp(Ur(m, k), -1000), Ur(k, n)))
case("denormA", 64, 64, 256, lambda m, n, k: (np.ldexp(Ur(m, k), -1050), np.ldexp(Ur(k, n), 1000)))


def with_special(v):
    def f(m, n, k):
        A, B = Ur(m, k), Ur(k, n)
        A[0, 0] = v
        return A, B
    return f


case("inf_a00", 64, 64, 256, with_special(math.inf))
case("nan_a00", 64, 64, 256, with_special(math.nan))
case("negzero", 32, 32, 64, lambda m, n, k: (np.full((m, k), -0.0), rng.uniform(0.5, 1.0, size=(k, n))))

MODES = ["native", "emu_perf", "emu_eager", "emu_eager_sv0", "emu_fixed55"]

if __name__ == "__main__":
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"cuBLAS version {ver.value}; modes {MODES}; u = 2^-53", flush=True)
    # timing / whether emulation is engaged at all on this GPU
    if not filt or filt == "timing":
        A, B = Ur(4096, 4096), Ur(4096, 4096)
        for mode in MODES[:3]:
            dgemm(mode, A[:256, :256], B[:256, :256])  # warm-up
            Cm, bits, dt = dgemm(mode, A, B)
            print(f"timing 4096^3 {mode:14s} bits={bits:3d} time={dt * 1e3:8.1f} ms  ({2 * 4096 ** 3 / dt / 1e12:.2f} TFLOP/s)",
                  flush=True)
        if filt == "timing":
            sys.exit(0)
    for name, m, n, k, fill in cases:
        if filt and filt not in name:
            continue
        A, B = fill(m, n, k)
        A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
        t0 = time.perf_counter()
        ref, aab = exact_ref(A, B)
        print(f"\n{name:12s} m={m} n={n} k={k}  (exact ref {time.perf_counter() - t0:.1f}s)", flush=True)
        for mode in MODES:
            try:
                Cm, bits, dt = dgemm(mode, A, B)
            except RuntimeError as e:
                print(f"  {mode:14s} ERROR {e}", flush=True)
                continue
            r = evaluate(Cm, ref, aab, A, B)
            print(f"  {mode:14s} bits={bits:3d} comp={r['comp']:<11.4g} >k={r['over_k']:<5d} wrong>=100%={r['wrong']:<5d} "
                  f"nonfinite_mishandled={r['nonfin_bad']}/{r['ref_nonfinite']} negzero={r['negzero']}  C00={Cm[0, 0]!r}",
                  flush=True)

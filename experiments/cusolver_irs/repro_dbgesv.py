"""Minimal reproducer: cusolverDnDBgesv / cusolverDnDXgesv / cusolverDnDSgesv (default settings, fallback on) return
iter >= 0 ("converged") with a wrong solution when A and b are scaled by 2^40 (entries ~1e12, far inside the range of
BF16/TF32/FP32), while the same call on the unscaled system is accurate.  Scaling by a power of two is exact, so the
true solution is identical, and the documented stopping rule RNRM < sqrt(N)*XNRM*ANRM*EPS is scale invariant.
Run: python repro_dbgesv.py   (Windows; any CUDA 11/12/13 NVIDIA pip wheels of cusolver + cuda-runtime in the venv)
"""
import ctypes as C
import math
import os
import sys

import numpy as np

# find the CUDA DLLs of whichever NVIDIA wheels are installed (cu13 layout or the per-library cu11/cu12 layout)
NV = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
dlls = {}
for root, _, files in os.walk(NV):
    if any(f.lower().endswith(".dll") for f in files):
        os.add_dll_directory(root)
        for f in files:
            dlls[f.lower()] = os.path.join(root, f)
pick = lambda pre: dlls[sorted(k for k in dlls if k.startswith(pre) and k.endswith(".dll") and "mg" not in k)[0]]
rt = C.WinDLL(pick("cudart64_"))
cs = C.WinDLL(pick("cusolver64_"))
rtv = C.c_int()
rt.cudaRuntimeGetVersion(C.byref(rtv))
h = C.c_void_p()
assert cs.cusolverDnCreate(C.byref(h)) == 0
ver = C.c_int()
cs.cusolverGetVersion(C.byref(ver))


def dev(a):
    p = C.c_void_p()
    rt.cudaMalloc(C.byref(p), C.c_size_t(max(a.nbytes, 8)))
    rt.cudaMemcpy(p, a.ctypes.data_as(C.c_void_p), C.c_size_t(a.nbytes), 1)
    return p


def gesv(name, A, b):
    n = A.shape[0]
    dA, dB = dev(np.asfortranarray(A).ravel(order="F").copy()), dev(b.copy())
    dX, dP, dI = dev(np.zeros(n)), dev(np.zeros(n, np.int32)), dev(np.zeros(1, np.int32))
    f, fb = getattr(cs, name), getattr(cs, name + "_bufferSize")
    fb.argtypes = [C.c_void_p, C.c_int, C.c_int] + [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int,
                                                    C.c_void_p, C.POINTER(C.c_size_t)]
    f.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int,
                  C.c_void_p, C.c_size_t, C.POINTER(C.c_int), C.c_void_p]
    lw = C.c_size_t()
    assert fb(h, n, 1, dA, n, dP, dB, n, dX, n, None, C.byref(lw)) == 0
    dW = C.c_void_p()
    rt.cudaMalloc(C.byref(dW), lw)
    it = C.c_int()
    st = f(h, n, 1, dA, n, dP, dB, n, dX, n, dW, lw, C.byref(it), dI)
    rt.cudaDeviceSynchronize()
    x = np.empty(n)
    rt.cudaMemcpy(x.ctypes.data_as(C.c_void_p), dX, C.c_size_t(x.nbytes), 2)
    info = np.empty(1, np.int32)
    rt.cudaMemcpy(info.ctypes.data_as(C.c_void_p), dI, C.c_size_t(4), 2)
    for p in (dA, dB, dX, dP, dI, dW):
        rt.cudaFree(p)
    return x, it.value, int(info[0]), st


rng = np.random.default_rng(0)
n = 1024
A0 = rng.choice([-1.0, 1.0], (n, n)) * rng.uniform(0.5, 1.0, (n, n))
x_true = rng.uniform(-1, 1, n)
b0 = A0 @ x_true
print(f"cuSOLVER version {ver.value}, CUDA runtime {rtv.value}, {os.path.basename(pick('cusolver64_'))}, n = {n}")
for name in ("cusolverDnDDgesv", "cusolverDnDSgesv", "cusolverDnDXgesv", "cusolverDnDBgesv"):
    for k in (0, 20, 40, -60):
        A, b = A0 * 2.0 ** k, b0 * 2.0 ** k
        x, it, info, st = gesv(name, A, b)
        bwd = np.abs(b - A @ x).max() / (np.abs(A).sum(axis=1).max() * np.abs(x).max())
        thr = math.sqrt(n) * 2.0 ** -53
        print(f"{name:18s} scale 2^{k:<3d} status={st} info={info} iter={it:4d}  "
              f"backward error / documented threshold = {bwd / thr:9.3g}   max rel. error of x = "
              f"{np.abs(x - x_true).max() / np.abs(x_true).max():9.3g}")

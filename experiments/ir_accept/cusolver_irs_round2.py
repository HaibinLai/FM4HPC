"""Round 2: force the low-precision path with n=1024 block-diagonal matrices diag(s*W(m), R) (R random,
diagonally dominant).  Relative pivot growth 2^(m-1) inside the W block overflows FP16 even after any global
scaling; with s = 2^100 and m = 40 the absolute growth (2^139) also overflows FP32 if no scaling is done."""
import math, time
import numpy as np
import cusolver_irs_probe as P

rng = np.random.default_rng(8)
n = 1024
cases = []
for m, s in [(0, 1.0), (20, 1.0), (40, 1.0), (40, 2.0 ** 100)]:
    A = rng.uniform(-1, 1, (n, n)) * 0.01 + n * np.eye(n)
    if m:
        A[:m, :] = 0.0; A[:, :m] = 0.0
        A[:m, :m] = P.wilkinson(m, s)
    cases.append((f"diag(W({m})*2^{int(math.log2(s))}, R) n={n}" if m else f"R only n={n}", A))
for name, A in cases:
    xt = np.ones(n)
    b = A @ xt
    print(f"\n{name}", flush=True)
    for kind, fb in [("D", True), ("DH", True), ("DS", True), ("DH", False), ("DS", False)]:
        t0 = time.perf_counter()
        x, it, info = P.solve(kind, A, b, fb)
        dt = (time.perf_counter() - t0) * 1e3
        lab = {"D": "FP64 (IRS 64/64)", "DH": "FP16 factor", "DS": "FP32 factor"}[kind] + ("" if fb else ", no fallback")
        print(f"  {lab:26s} iter={it:5d} info={info:3d} {dt:7.1f}ms  {P.describe(x, xt)}", flush=True)

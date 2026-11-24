"""Test predictions derived from the ESC contract (CublasEsc.lean) on cuBLAS 13.10 / sm_120.
Model: if every block of k has a zero in both x and y, coarsened ESC = min(x_p, y_q) - Z + const, so the
reported bit count changes by exactly the shift of min(max exponent of A, max exponent of B).
P1: scalings of a zero-every-32 uniform pair; predictions relative to the unscaled case.
P2: boundary block x=[2^300, 1, 0...], y=[0, 2^-1035, 0...]; true C = 2^-1035 (subnormal) everywhere."""
import math
import numpy as np
import cublas_emu_probe as P

rng = np.random.default_rng(21)
n = 1024
A0, B0 = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
Az, Bz = A0.copy(), B0.copy()
Az[:, ::32] = 0.0
Bz[::32, :] = 0.0
print("P1: zero every 32 along k, n=1024, mode emu_eager", flush=True)
base = None
for sa, sb, pred in [(0, 0, 0), (100, 0, 0), (-100, 0, -100), (100, 100, 100), (100, -100, -100), (-300, -300, -300), (0, 200, 0)]:
    _, bits, _ = P.dgemm("emu_eager", np.ldexp(Az, sa), np.ldexp(Bz, sb))
    _, bits_dense, _ = P.dgemm("emu_eager", np.ldexp(A0, sa), np.ldexp(B0, sb))
    if base is None:
        base = bits
    print(f"  A*2^{sa:<5d} B*2^{sb:<5d} bits={bits:5d}  change={bits - base:+5d}  predicted change={pred:+5d}  "
          f"{'OK' if bits - base == pred else 'MISMATCH'}   (dense, no zeros: bits={bits_dense})", flush=True)

print("P2: boundary block, m=n=k=64", flush=True)
m = k = 64
A = np.zeros((m, k)); B = np.zeros((k, m))
A[:, 0] = 2.0 ** 300; A[:, 1] = 1.0
B[1, :] = 2.0 ** -1035
truth = 2.0 ** -1035
pred_bits = base - 1 + (-1035)  # min(xp, yq) goes from about -1 (uniform data) to -1035
print(f"  predicted reported bits about {pred_bits} (fallback), result exact", flush=True)
for mode in ("native", "emu_eager", "emu_fixed55"):
    C, bits, _ = P.dgemm(mode, A, B)
    ok = int(np.sum(C == truth))
    print(f"  {mode:12s} bits={bits:5d}  entries equal to 2^-1035: {ok}/{m * m}  C00={C[0, 0]!r}", flush=True)

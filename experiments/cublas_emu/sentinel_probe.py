"""Out-of-sample test of the fitted ADP model (esc_model.py) and of the sentinel condition proved in AdpEsc.lean.

S1 zero_denorm: in block 0, row 0 of A is (0, 2^-1074 x 63), the rest of row 0 is zero; column 0 of B is (2^1000, 1, 1, ...).
   The only nonzero products in C[0,0] are 2^-1074 * 1, so C[0,0] = 63 * 2^-1074 (exact in FP64).
   Block 0 estimate = max(MaxA + MinB, MinA + MaxB) = max(-1074 + 0, L + 1000).
   - safe sentinel (L + 1023 <= 2 * (-1074), e.g. cuBLAS's fitted L = -4195): est = -1074, ESC = 1000 -> ~1054 bits -> native
   - unsafe sentinel (e.g. L = -1075):  est = -75, ESC ~ 1 -> emulation with ~55 bits -> C[0,0] = 0 (100% wrong)
S2 zero_denorm_mild: same with 2^-1074 -> 2^-1000 and 2^1000 -> 2^60 so that the correct choice (~114 bits) is emulated.
Usage (roxy): python sentinel_probe.py      Usage (model only, any machine): python3 sentinel_probe.py model
"""
import sys
from fractions import Fraction

import numpy as np

n = 2048


def build(tiny_e, big_e, seed=5):
    rng = np.random.default_rng(seed)
    A, B = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
    A[0, :] = 0.0
    A[0, 1:64] = 2.0 ** tiny_e
    B[:, 0] = 1.0
    B[0, 0] = 2.0 ** big_e
    return A, B


CASES = [("S1 zero_denorm", -1074, 1000), ("S2 zero_denorm_mild", -1000, 60)]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "model":
        import esc_model as M
        for name, te, be in CASES:
            A, B = build(te, be)
            print(f"{name:22s} predicted bits: fitted L=-4195 -> {M.esc_max(A, B, -4195) + 54}, "
                  f"unsafe L=-1075 -> {M.esc_max(A, B, -1075) + 54}")
        sys.exit(0)
    import cublas_emu_probe as P
    import cublas_emu_large as G
    for name, te, be in CASES:
        A, B = build(te, be)
        r00, _ = G.exact_entry(A[0, :], B[:, 0])
        samples = [(0, 0), (0, 1), (1, 0), (5, 7)]
        line = f"{name:22s} exact C00={float(r00)!r}"
        for mode in ("native", "emu_perf", "emu_eager"):
            C, bits, dt = P.dgemm(mode, A, B)
            w, over, wrong = G.sampled_eval(C, A, B, samples)
            line += f" | {mode}: bits={bits} C00={float(C[0, 0])!r} relerr00={float(abs(Fraction(float(C[0, 0])) - r00) / abs(r00)):.3g} comp={w:.3g} wrong={wrong}"
        print(line, flush=True)

"""What the zero-aware estimate (AdpEsc.lean `fixedEst`, esc_model.py variant='fixed') would buy on the GPU.
cuBLAS cannot be patched, so the fixed ADP decision is emulated by running cuBLAS in FIXED mantissa mode with the bit count
the fixed estimate predicts (bits_fix = model + 54), compared with the default dynamic decision and native FP64.
Accuracy: sampled exact check (comp = |C-ref| / (u (|A||B|)_ij)).   Usage (roxy): python fix_benefit.py
"""
import numpy as np

import cublas_emu_probe as P
import cublas_emu_large as G
import esc_model as M

rng = np.random.default_rng(23)


def best_time(mode, A, B, reps=3):
    best, C, bits = None, None, None
    for _ in range(reps):
        C, bits, dt = P.dgemm(mode, A, B)
        best = dt if best is None else min(best, dt)
    return C, bits, best


def cases():
    n = 4096
    A, B = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
    A[:, ::32] = 0.0
    B[::32, :] = 0.0
    yield "zero_gap32 n=4096", A, B
    yield "tri_s0 n=4096", np.triu(rng.uniform(-1, 1, (n, n))), np.tril(rng.uniform(-1, 1, (n, n)))
    n = 2048
    sp = lambda: np.ldexp(rng.uniform(-1, 1, (n, n)), rng.integers(0, 61, size=(n, n)))
    yield "tri_s60 n=2048", np.triu(sp()), np.tril(sp())


if __name__ == "__main__":
    for name, A, B in cases():
        n = A.shape[0]
        bits_paper = M.esc_max(A, B, -4195) + 54
        bits_fix = M.esc_max(A, B, -4195, variant="fixed") + 54
        samples = [(0, 0), (n - 1, n - 1), (n - 1, 0), (0, n - 1)] + [(int(i), int(j)) for i, j in rng.integers(0, n, (60, 2))]
        print(f"{name}: model bits paper={bits_paper} fixed={bits_fix}", flush=True)
        for mode in ("native", "emu_eager", f"emu_fixed{bits_fix}"):
            C, bits, dt = best_time(mode, A, B)
            w, over, wrong = G.sampled_eval(C, A, B, samples)
            print(f"   {mode:14s} bits={bits:5d} {dt * 1e3:8.1f} ms  sampled comp={w:.3g} >k={over} wrong={wrong}", flush=True)

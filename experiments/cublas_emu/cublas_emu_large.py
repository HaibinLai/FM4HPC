"""Second round on cuBLAS FP64 emulation (sm_120): large problems where PERFORMANT actually emulates.
E1  zero-block pathology: dense uniform A, B (n^3) with a zero every `gap` positions along k (A columns, B rows).
    Reports retained bits (device pointer; -1 = native, -2 = pointer untouched) and time per mode.
E2  adversarial inputs at n = 2048 under PERFORMANT / EAGER, checked against exact values of sampled entries
    (row 0 x all sampled columns, column 0 x sampled rows, plus random entries), comp = |C-ref|/(u (|A||B|)_ij).
Usage: python cublas_emu_large.py [E1|E2]
"""
import math
import sys
import time
from fractions import Fraction

import numpy as np

import cublas_emu_probe as P

U = 2.0 ** -53
rng = np.random.default_rng(11)


def timed(mode, A, B, reps=2):
    best, bits, C = None, None, None
    for _ in range(reps):
        C, bits, dt = P.dgemm(mode, A, B)
        best = dt if best is None else min(best, dt)
    return C, bits, best


def exact_entry(a_row, b_col):
    terms = []
    for x, y in zip(a_row, b_col):
        if x == 0.0 or y == 0.0:
            continue
        mx, ex = P.decompose(float(x))
        my, ey = P.decompose(float(y))
        terms.append((mx * my, ex + ey))
    if not terms:
        return Fraction(0), Fraction(0)
    e0 = min(t[1] for t in terms)
    s = sum(t[0] << (t[1] - e0) for t in terms)
    sa = sum(abs(t[0]) << (t[1] - e0) for t in terms)
    sc = Fraction(2) ** e0
    return s * sc, sa * sc


def sampled_eval(C, A, B, samples):
    worst, over, wrong = 0.0, 0, 0
    k = A.shape[1]
    for i, j in samples:
        r, aab = exact_entry(A[i, :], B[:, j])
        c = float(C[i, j])
        if not math.isfinite(c):
            wrong += 1
            continue
        e = abs(Fraction(c) - r)
        if aab > 0:
            q = float(e / (aab * Fraction(U)))
            worst = max(worst, q)
            over += q > k
        if r != 0 and e >= abs(r):
            wrong += 1
    return worst, over, wrong


def E1():
    print("E1: zero every `gap` positions along k; dense uniform otherwise", flush=True)
    for n in (1024, 2048):
        A0, B0 = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
        for gap in (0, 16, 32, 64, 128, 256, 512):
            A, B = A0.copy(), B0.copy()
            if gap:
                A[:, ::gap] = 0.0
                B[::gap, :] = 0.0
            line = f"n={n:5d} gap={gap:4d}"
            for mode in ("native", "emu_perf", "emu_eager"):
                _, bits, dt = timed(mode, A, B)
                line += f"  {mode}: bits={bits:5d} {dt * 1e3:8.1f}ms"
            print(line, flush=True)
    # one zero only (single entry) in a dense matrix
    n = 2048
    A, B = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
    for desc, mut in (("A[0,0]=0", lambda A, B: A.__setitem__((0, 0), 0.0)),
                      ("A[:,0]=0 (one zero column)", lambda A, B: A.__setitem__((slice(None), 0), 0.0)),
                      ("A row0 all 0", lambda A, B: A.__setitem__((0, slice(None)), 0.0)),
                      ("A tiny 2^-900 at [:,::32]", lambda A, B: A.__setitem__((slice(None), slice(None, None, 32)), 2.0 ** -900))):
        A2, B2 = A.copy(), B.copy()
        mut(A2, B2)
        line = f"n={n} {desc:28s}"
        for mode in ("native", "emu_perf", "emu_eager"):
            _, bits, dt = timed(mode, A2, B2)
            line += f"  {mode}: bits={bits:5d} {dt * 1e3:8.1f}ms"
        print(line, flush=True)


def E2():
    n = 2048
    print(f"E2: adversarial inputs at n={n}, sampled exact check", flush=True)
    U_ = lambda *s: rng.uniform(-1, 1, s)
    spread = lambda s, X: np.ldexp(U_(*s), rng.integers(0, X + 1, size=s))

    def avg_hides():
        A, B = U_(n, n), U_(n, n)
        A[0, :] = 2.0 ** -100; A[0, 0] = 1.0
        B[:, 0] = 1.0; B[0, 0] = 2.0 ** -100
        return A, B

    def wide_row():
        A, B = U_(n, n), U_(n, n)
        A[0, :] = np.ldexp(U_(n), rng.integers(-500, 501, size=n))
        return A, B

    def tri():
        return np.triu(spread((n, n), 60)), np.tril(spread((n, n), 60))

    def big_row():
        A, B = U_(n, n), U_(n, n)
        A[0, :] *= 2.0 ** 900
        B[:, 0] *= 2.0 ** -900
        return A, B

    def cancel_heavy():
        # each dot product sums +-2^60 pairs that cancel exactly plus O(1) terms
        A, B = U_(n, n), U_(n, n)
        A[:, 0] = 2.0 ** 60; A[:, 1] = -(2.0 ** 60)
        B[0, :] = 1.0; B[1, :] = 1.0
        return A, B

    cases = [("rand_s0", lambda: (U_(n, n), U_(n, n))), ("rand_s60", lambda: (spread((n, n), 60), spread((n, n), 60))),
             ("avg_hides", avg_hides), ("wide_row0", wide_row), ("tri_s60", tri), ("big_row0_2^900", big_row),
             ("cancel_heavy", cancel_heavy)]
    samples = [(0, 0), (0, 1), (1, 0)] + [(0, int(j)) for j in rng.integers(0, n, 40)] + \
              [(int(i), 0) for i in rng.integers(0, n, 40)] + [(int(i), int(j)) for i, j in rng.integers(0, n, (80, 2))]
    for name, f in cases:
        A, B = f()
        line = f"{name:15s}"
        for mode in ("native", "emu_perf", "emu_eager"):
            C, bits, dt = timed(mode, A, B, reps=1)
            w, over, wrong = sampled_eval(C, A, B, samples)
            line += f" | {mode}: bits={bits:5d} {dt * 1e3:7.1f}ms comp={w:9.3g} >k={over} wrong={wrong}"
        print(line, flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "E1E2"
    if "E1" in which:
        E1()
    if "E2" in which:
        E2()

"""Host model of the ADP bit-count decision (arXiv 2511.13778 Sec. 4, coarsened ESC), fitted to the bit counts that
cuBLAS 13.10 reported on the RTX 5070 Ti (cublas_emu_results.txt, cublas_emu_E1.txt, cublas_emu_E2.txt).

Model (one integer for the whole GEMM):
    e(v)      = exponent of v (floor(log2|v|)); e(0) = L (sentinel)
    x_i       = max_k e(a_ik)              y_j = max_k e(b_kj)
    est_ij    = max over k-blocks of length BLK of max(MaxA_ib + MinB_bj, MinA_ib + MaxB_bj)
    bits      = max_ij (x_i + y_j - est_ij) + K
The inputs are regenerated from the same seeds as the probes, so the prediction can be compared with the measurement.
Usage: python3 esc_model.py [fit]      (no GPU needed)
"""
import re
import sys

import numpy as np

BLK = 64


def expo(M, L, denorm="true"):
    """floor(log2|v|); zero -> L.  denorm='field' uses the biased exponent field (subnormals -> -1023)."""
    M = np.asarray(M, dtype=np.float64)
    _, e = np.frexp(M)
    e = e.astype(np.int64) - 1
    if denorm == "field":
        f = ((M.view(np.uint64) >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64) - 1023
        e = np.where(f == -1023, -1023, e)
    return np.where(M == 0.0, L, e)


def esc_max(A, B, L, denorm="true", blk=BLK, variant="paper"):
    """max_ij (x_i + y_j - est_ij).  variant='fixed' adds the two candidates proved safe in AdpEsc.lean:
    the first and last index of each block (an actual product) and the pigeonhole candidate."""
    m, k = A.shape
    ea, eb = expo(A, L, denorm), expo(B, L, denorm)
    x = ea.max(axis=1)
    y = eb.max(axis=0)
    nb = (k + blk - 1) // blk
    pad = nb * blk - k
    if pad:
        ea = np.pad(ea, ((0, 0), (0, pad)), constant_values=L)
        eb = np.pad(eb, ((0, pad), (0, 0)), constant_values=L)
    ea3 = ea.reshape(m, nb, blk)
    eb3 = eb.reshape(nb, blk, -1)
    maxA, minA = ea3.max(axis=2), ea3.min(axis=2)  # (m, nb)
    maxB, minB = eb3.max(axis=1), eb3.min(axis=1)  # (nb, n)
    if variant == "fixed":
        nzA = (A != 0).astype(np.int64)
        nzB = (B != 0).astype(np.int64)
        if pad:
            nzA = np.pad(nzA, ((0, 0), (0, pad)))
            nzB = np.pad(nzB, ((0, pad), (0, 0)))
        cntA, cntB = nzA.reshape(m, nb, blk).sum(axis=2), nzB.reshape(nb, blk, -1).sum(axis=1)
        big = np.int64(1) << 40
        mnzA = np.where(ea3 == L, big, ea3).min(axis=2)
        mnzB = np.where(eb3 == L, big, eb3).min(axis=1)
        firstA, lastA = ea3[:, :, 0], ea3[:, :, blk - 1]
        firstB, lastB = eb3[:, 0, :], eb3[:, blk - 1, :]
        okfA, oklA = firstA != L, lastA != L
        okfB, oklB = firstB != L, lastB != L
    worst = -(1 << 62)
    n = B.shape[1]
    for i0 in range(0, m, 64):
        sl = slice(i0, min(m, i0 + 64))
        c1 = maxA[sl, :, None] + minB[None, :, :]
        c2 = minA[sl, :, None] + maxB[None, :, :]
        est = np.maximum(c1, c2)
        if variant == "fixed":
            neg = -(np.int64(1) << 50)
            pig = np.where(cntA[sl, :, None] + cntB[None, :, :] > blk, mnzA[sl, :, None] + mnzB[None, :, :], neg)
            rf = np.where(okfA[sl, :, None] & okfB[None, :, :], firstA[sl, :, None] + firstB[None, :, :], neg)
            rl = np.where(oklA[sl, :, None] & oklB[None, :, :], lastA[sl, :, None] + lastB[None, :, :], neg)
            est = np.maximum(np.maximum(est, pig), np.maximum(rf, rl))
        est = est.max(axis=1)  # (rows, n)
        v = x[sl, None] + y[None, :] - est
        worst = max(worst, int(v.max()))
    return worst


# ---------- regenerate the probes' inputs ----------
def probe_cases():
    src = open("cublas_emu_probe.py").read()
    body = src[src.index("# ---------------- cases ----------------"):src.index("MODES = ")]
    g = {"np": np, "math": __import__("math")}
    exec(body, g)
    rng = g["rng"]
    rng.uniform(-1.0, 1.0, size=(4096, 4096))  # the timing run draws A, B first
    rng.uniform(-1.0, 1.0, size=(4096, 4096))
    for name, m, n, k, fill in g["cases"]:
        A, B = fill(m, n, k)
        yield name, np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)


def measured_small():
    out, cur = {}, None
    for line in open("cublas_emu_results.txt"):
        h = re.match(r"^(\S+)\s+m=\d+", line)
        if h:
            cur = h.group(1)
        mm = re.match(r"^\s+emu_eager\s+bits=\s*(-?\d+)", line)
        if mm and cur:
            out[cur] = int(mm.group(1))
    return out


def large_cases():
    """E1 (n = 1024, 2048, gaps) then E2, each from a fresh default_rng(11) (they were run as separate invocations)."""
    rng = np.random.default_rng(11)
    for n in (1024, 2048):
        A0, B0 = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
        for gap in (0, 16, 32, 64, 128, 256, 512):
            A, B = A0.copy(), B0.copy()
            if gap:
                A[:, ::gap] = 0.0
                B[::gap, :] = 0.0
            yield f"E1 n={n} gap={gap}", A, B
    rng = np.random.default_rng(11)
    n = 2048
    U_ = lambda *s: rng.uniform(-1, 1, s)
    spread = lambda s, X: np.ldexp(U_(*s), rng.integers(0, X + 1, size=s))
    yield "E2 rand_s0", U_(n, n), U_(n, n)
    yield "E2 rand_s60", spread((n, n), 60), spread((n, n), 60)
    A, B = U_(n, n), U_(n, n)
    A[0, :] = 2.0 ** -100; A[0, 0] = 1.0; B[:, 0] = 1.0; B[0, 0] = 2.0 ** -100
    yield "E2 avg_hides", A, B
    A, B = U_(n, n), U_(n, n)
    A[0, :] = np.ldexp(U_(n), rng.integers(-500, 501, size=n))
    yield "E2 wide_row0", A, B
    yield "E2 tri_s60", np.triu(spread((n, n), 60)), np.tril(spread((n, n), 60))
    A, B = U_(n, n), U_(n, n)
    A[0, :] *= 2.0 ** 900; B[:, 0] *= 2.0 ** -900
    yield "E2 big_row0", A, B
    A, B = U_(n, n), U_(n, n)
    A[:, 0] = 2.0 ** 60; A[:, 1] = -(2.0 ** 60); B[0, :] = 1.0; B[1, :] = 1.0
    yield "E2 cancel_heavy", A, B


MEASURED_LARGE = {
    "E1 n=1024 gap=0": 59, "E1 n=1024 gap=16": 4248, "E1 n=1024 gap=32": 4248, "E1 n=1024 gap=64": 4248,
    "E1 n=1024 gap=128": 60, "E1 n=1024 gap=256": 60, "E1 n=1024 gap=512": 59,
    "E1 n=2048 gap=0": 59, "E1 n=2048 gap=16": 4248, "E1 n=2048 gap=32": 4248, "E1 n=2048 gap=64": 4248,
    "E1 n=2048 gap=128": 60, "E1 n=2048 gap=256": 59, "E1 n=2048 gap=512": 59,
    "E2 rand_s0": 59, "E2 rand_s60": 115, "E2 avg_hides": 154, "E2 wide_row0": 62, "E2 tri_s60": 4308,
    "E2 big_row0": 59, "E2 cancel_heavy": 73,
}

if __name__ == "__main__":
    L = int(sys.argv[2]) if len(sys.argv) > 2 else -4195
    K = int(sys.argv[3]) if len(sys.argv) > 3 else None
    denorm = sys.argv[4] if len(sys.argv) > 4 else "true"
    rows = []
    small = measured_small()
    for name, A, B in probe_cases():
        if name in ("inf_a00", "nan_a00") or small.get(name, -1) < 0:
            continue
        rows.append((name, esc_max(A, B, L, denorm), small[name], A, B))
    for name, A, B in large_cases():
        rows.append((name, esc_max(A, B, L, denorm), MEASURED_LARGE[name], A, B))
    if K is None:  # fit K on the zero-free cases, then report
        ks = [meas - e for _, e, meas, A, B in rows if (A != 0).all() and (B != 0).all()]
        K = max(set(ks), key=ks.count)
    print(f"model: BLK={BLK} L={L} K={K} denormals={denorm}")
    bad = 0
    for name, e, meas, A, B in rows:
        pred = e + K
        fixed = esc_max(A, B, L, denorm, variant="fixed") + K
        flag = "" if pred == meas else "   <-- mismatch"
        bad += pred != meas
        print(f"{name:20s} measured={meas:5d} predicted={pred:5d} with-fix={fixed:5d}{flag}")
    print(f"{len(rows) - bad}/{len(rows)} match")

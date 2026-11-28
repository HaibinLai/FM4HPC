"""Faithful CPU port of the control flow of MAGMA v2.10.0 magma_zchesv_gpu (real symmetric case, zchesv_gpu.cpp).
Prediction (MagmaDense.zchesv_stub): when the FP32 refinement does not converge within ITERMAX-1 = 29 corrections,
the routine returns iter = -31, info = 0 and X = B (the FP64 hetrf/hetrs fallback is commented out).
FP32 factorization: LDL^T WITHOUT pivoting (magma_chetrf_nopiv_gpu), here a plain float32 implementation.
Matrices: block diagonal with 2x2 blocks [[d, 1], [1, d]] (well conditioned, kappa ~ 1, indefinite);
small d gives large growth in the non-pivoted factorization."""
import numpy as np, sys
u = 2.0**-53

def ldlt_nopiv32(A):
    n = A.shape[0]; L = np.eye(n, dtype=np.float32); D = np.zeros(n, dtype=np.float32)
    A = A.astype(np.float32).copy()
    for j in range(n):
        D[j] = A[j, j] - np.float32(np.sum(L[j, :j]**2 * D[:j], dtype=np.float32))
        if D[j] == 0: return None, None, j + 1
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.float32(np.sum(L[i, :j] * L[j, :j] * D[:j], dtype=np.float32))) / D[j]
    return L, D, 0

def solve32(L, D, r):
    y = np.array(r, dtype=np.float32); n = len(y)
    for i in range(n): y[i] -= np.float32(np.dot(L[i, :i], y[:i]))
    y /= D
    for i in reversed(range(n)): y[i] -= np.float32(np.dot(L[i+1:, i], y[i+1:]))
    return y

def zchesv(A, b, ITERMAX=30):
    n = len(b); Anrm = np.max(np.sum(np.abs(A), axis=1)); cte = Anrm * u * np.sqrt(n)
    L, D, info = ldlt_nopiv32(A)
    if info: return dict(iter=-3, info=info, X='untouched')          # fallback: info != 0, X not written
    x = solve32(L, D, b).astype(np.float64); r = b - A @ x
    if not (np.max(np.abs(r)) > np.max(np.abs(x)) * cte): return dict(iter=0, info=0, X=x)
    for it in range(1, ITERMAX):                                         # for (iiter=1; iiter < ITERMAX; )
        x = x + solve32(L, D, r).astype(np.float64); r = b - A @ x
        if not (np.max(np.abs(r)) > np.max(np.abs(x)) * cte): return dict(iter=it, info=0, X=x)
    return dict(iter=-ITERMAX-1, info=0, X=b.copy())                     # stub: zlacpy(B -> X), info stays 0

rng = np.random.default_rng(0)
for d in [2.0**-4, 2.0**-12, 2.0**-20, 2.0**-24, 2.0**-27]:
    nb = 4; n = 2 * nb
    A = np.kron(np.eye(nb), np.array([[d, 1.0], [1.0, d]]))
    xt = rng.uniform(-1, 1, n); b = A @ xt
    res = zchesv(A, b)
    X = res['X']
    if isinstance(X, str):
        print(f"d=2^{int(np.log2(d)):>4}  iter={res['iter']:>4} info={res['info']}  X untouched"); continue
    err = np.max(np.abs(X - xt)) / np.max(np.abs(xt)); isB = np.array_equal(X, b)
    print(f"d=2^{int(np.log2(d)):>4}  iter={res['iter']:>4} info={res['info']}  X==B:{isB}  rel.err={err:.2e}  "
          f"cond(A)={np.linalg.cond(A):.2f}")

print("--- coupled saddle-type matrices A = [[d, c^T], [c, M]] (M random symmetric, well conditioned) ---")
for d in [2.0**-4, 2.0**-10, 2.0**-16, 2.0**-20, 2.0**-24]:
    for seed in range(3):
        g = np.random.default_rng(seed); n = 16
        c = g.uniform(-1, 1, n - 1); Q, _ = np.linalg.qr(g.standard_normal((n - 1, n - 1)))
        M = Q @ np.diag(g.uniform(1, 2, n - 1)) @ Q.T
        A = np.zeros((n, n)); A[0, 0] = d; A[0, 1:] = c; A[1:, 0] = c; A[1:, 1:] = M
        xt = g.uniform(-1, 1, n); b = A @ xt
        res = zchesv(A, b); X = res['X']
        if isinstance(X, str):
            print(f"d=2^{int(np.log2(d)):>4} seed={seed} iter={res['iter']:>4} info={res['info']}  X untouched"); continue
        err = np.max(np.abs(X - xt)) / np.max(np.abs(xt))
        print(f"d=2^{int(np.log2(d)):>4} seed={seed} iter={res['iter']:>4} info={res['info']}  X==B:{np.array_equal(X, b)}"
              f"  rel.err={err:.2e}  cond(A)={np.linalg.cond(A):.1f}")

"""Faithful CPU port of MAGMA v2.10.0 magma_dgesv_rbt (real double), from:
  src/zgerbt_gpu.cpp   init_butterfly: u[i], v[i] = exp((rand()/RAND_MAX - 0.5)/10), i < 2n, rand() unseeded (glibc)
  magmablas/zgerbt.cu  zprbt: level 1 on the four quadrants with u[n:], v[n:]; level 2 on the whole matrix with u[:n], v[:n]
                       zprbt_mtv: level-1 halves then level 2;  zprbt_mv: level 2 then level-1 halves
  magmablas/zgerbt_kernels.cu
     elementary (matrix, l.47-85): split Ar1=ceil(Ar/2); out = D_u R A R D_v with R = [[I,I],[I,-I]] (zero-padded)
     apply_transpose_vector (U^T b, l.171-183): **if (n < 2) return;**  out = D_u R b
     apply_vector (V y, l.116-137): if (n < 1) return;  out = R D_v y
  src/zgesv_nopiv_gpu.cpp, zgetrf_nopiv*, zgetrs_nopiv_gpu.cpp: LU without pivoting
  src/zgerfs_nopiv_gpu.cpp: classical refinement on the TRANSFORMED system (dAo, dBo), no fallback
Prediction: for n = 1, 2, 3 the transformed right-hand side is not U^T b (a size-1 block skips the u scaling),
so the transformed system is inconsistent with the original one and x = V y is wrong, info = 0."""
import numpy as np

def glibc_rand(count, seed=1):
    r = [0] * (344 + count)
    r[0] = seed
    for i in range(1, 31):
        r[i] = (16807 * r[i - 1]) % 2147483647
    for i in range(31, 34): r[i] = r[i - 31]
    for i in range(34, 344 + count): r[i] = (r[i - 31] + r[i - 3]) & 0xFFFFFFFF
    return [x >> 1 for x in r[344:344 + count]]

assert glibc_rand(3) == [1804289383, 846930886, 1681692777], glibc_rand(3)
RAND_MAX = 2147483647

def init_butterfly(n2, state):          # n2 = 2n; u, v interleaved draws per index
    vals = glibc_rand(state + 2 * n2)[state:]
    u = np.array([np.exp((vals[2*i]   / RAND_MAX - 0.5) / 10) for i in range(n2)])
    v = np.array([np.exp((vals[2*i+1] / RAND_MAX - 0.5) / 10) for i in range(n2)])
    return u, v, state + 2 * n2

def elementary(A, i0, j0, m, k, u, v):  # in place on block A[i0:i0+m, j0:j0+k]; u, v are the segments
    m1, m2, k1, k2 = (m+1)//2, m-(m+1)//2, (k+1)//2, k-(k+1)//2
    B = A[i0:i0+m, j0:j0+k].copy(); out = B.copy()
    for x in range(m1):
        for y in range(k1):
            a00 = B[x, y]; a01 = B[x, k1+y] if y < k2 else 0.0
            a10 = B[m1+x, y] if x < m2 else 0.0; a11 = B[m1+x, k1+y] if (x < m2 and y < k2) else 0.0
            b1, b2, b3, b4 = a00 + a01, a10 + a11, a00 - a01, a10 - a11
            out[x, y] = u[x] * v[y] * (b1 + b2)
            if y < k2: out[x, k1+y] = u[x] * v[k1+y] * (b3 + b4)
            if x < m2: out[m1+x, y] = u[m1+x] * v[y] * (b1 - b2)
            if x < m2 and y < k2: out[m1+x, k1+y] = u[m1+x] * v[k1+y] * (b3 - b4)
    A[i0:i0+m, j0:j0+k] = out

def prbt(A, u, v):
    n = A.shape[0]; n1, n2 = (n+1)//2, n-(n+1)//2; U, V = u[n:], v[n:]
    elementary(A, 0, 0, n1, n1, U[0:], V[0:]);   elementary(A, 0, n1, n1, n2, U[0:], V[n1:])
    elementary(A, n1, 0, n2, n1, U[n1:], V[0:]); elementary(A, n1, n1, n2, n2, U[n1:], V[n1:])
    elementary(A, 0, 0, n, n, u[0:], v[0:])

def apply_transpose_vector(n, du, b, off):
    if n < 2: return                                   # l.174: the size-1 case is skipped
    n1, n2 = (n+1)//2, n-(n+1)//2; seg = b[off:off+n].copy()
    for i in range(n1):
        a1 = seg[i] + (seg[n1+i] if i < n2 else 0.0); a2 = seg[i] - (seg[n1+i] if i < n2 else 0.0)
        b[off+i] = du[i] * a1
        if i < n2: b[off+n1+i] = du[n1+i] * a2

def apply_vector(n, du, b, off):
    if n < 1: return
    n1, n2 = (n+1)//2, n-(n+1)//2; seg = b[off:off+n].copy()
    for i in range(n1):
        a1 = du[i] * seg[i]; a2 = du[n1+i] * seg[n1+i] if i < n2 else 0.0
        b[off+i] = a1 + a2
        if i < n2: b[off+n1+i] = a1 - a2

def prbt_mtv(b, u):
    n = len(b); n1, n2 = (n+1)//2, n-(n+1)//2
    apply_transpose_vector(n1, u[n:], b, 0); apply_transpose_vector(n2, u[n+n1:], b, n1); apply_transpose_vector(n, u, b, 0)

def prbt_mv(y, v):
    n = len(y); n1, n2 = (n+1)//2, n-(n+1)//2
    apply_vector(n, v, y, 0); apply_vector(n1, v[n:], y, 0); apply_vector(n2, v[n+n1:], y, n1)

def lu_nopiv_solve(A, b):
    n = len(b); L = np.eye(n); U = A.copy(); info = 0
    for j in range(n):
        if U[j, j] == 0 and info == 0: info = j + 1
        for i in range(j+1, n):
            if U[j, j] != 0: L[i, j] = U[i, j] / U[j, j]
            U[i, j:] -= L[i, j] * U[j, j:]
    return L, U, info

def solve_lu(L, U, r):
    y = np.linalg.solve(L, r)
    with np.errstate(all='ignore'): return np.linalg.solve(U, y) if np.all(np.diag(U) != 0) else np.full_like(r, np.nan)

def gesv_rbt(A0, b0, refine=True, state=0):
    n = len(b0); u, v, state = init_butterfly(2*n, state)
    A = A0.copy(); b = b0.copy(); prbt(A, u, v); prbt_mtv(b, u)
    Ao, bo = A.copy(), b.copy()
    L, U, info = lu_nopiv_solve(A, b)
    y = solve_lu(L, U, b) if info == 0 else b.copy()          # gesv_nopiv: getrs only if info == 0
    if refine:                                                  # gerfs_nopiv on (Ao, bo), starts with info = 0
        info = 0; eps = 2.0**-53; cte = np.max(np.sum(np.abs(Ao), 1)) * eps * np.sqrt(n)
        r = bo - Ao @ y
        if np.max(np.abs(r)) > np.max(np.abs(y)) * cte:
            for it in range(1, 30):
                y = y + solve_lu(L, U, r); r = bo - Ao @ y
                if not (np.max(np.abs(r)) > np.max(np.abs(y)) * cte): break
    x = y.copy(); prbt_mv(x, v)
    return x, info, (u, v, Ao, bo)

# --- self-check of the port: the matrix transform equals explicit U^T A V for sizes where the vector kernels are complete
def dense_UV(n, u, v):
    UT = np.eye(n); prbt_cols = np.eye(n)
    for j in range(n): e = np.eye(n)[:, j].copy(); prbt_mtv(e, u); UT[:, j] = e
    for j in range(n): e = np.eye(n)[:, j].copy(); prbt_mv(e, v); prbt_cols[:, j] = e
    return UT, prbt_cols
rng = np.random.default_rng(0)
print("n  | ||prbt(A) - (U^T)A V|| / ||A||   (U^T from prbt_mtv, V from prbt_mv)  | dgesv_rbt rel.err (refine)  info")
for n in [1, 2, 3, 4, 5, 8, 16, 33, 64]:
    u, v, _ = init_butterfly(2*n, 0)
    A0 = rng.standard_normal((n, n)) + n * np.eye(n); xt = rng.standard_normal(n); b0 = A0 @ xt
    A = A0.copy(); prbt(A, u, v); UT, V = dense_UV(n, u, v)
    cons = np.linalg.norm(A - UT @ A0 @ V) / np.linalg.norm(A0)
    x, info, _ = gesv_rbt(A0, b0, refine=True, state=0)
    print(f"{n:<3}|  {cons:.2e}                                                   |  {np.linalg.norm(x-xt)/np.linalg.norm(xt):.2e}                   {info}")

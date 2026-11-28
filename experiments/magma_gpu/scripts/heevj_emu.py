# Line-by-line emulation of zheevj_batched_parallel_small_sm_kernel (MAGMA 2.10.0, magmablas/zheevj_batched_small_sm.cu)
# for n = 2 (one rotation pair j = 0, k = 1), sequential semantics.  Compares with numpy eigh.
import math, cmath
import numpy as np
EPS = 2.0**-52
def heevj(A, tol=30, max_sweeps=100):
    n = 2; sA = [[complex(A[i][j]) for j in range(n)] for i in range(n)]
    for i in range(n): sA[i][i] = complex(sA[i][i].real, 0)
    sA[0][1] = sA[1][0].conjugate()                     # lower -> Hermitian
    sV = [[1+0j, 0j], [0j, 1+0j]]; sorder = [0, 1]
    sweep = 0; converge = 0
    while sweep < max_sweeps and converge == 0:
        j, k = 0, 1; converge = 1; sweep += 1
        ajj = sA[j][j].real; akk = sA[k][k].real; ajk = sA[j][k]
        t_ol = EPS * tol * math.sqrt(abs(ajj) * abs(akk)); a = abs(ajk); d = abs(ajj - akk)
        noconv = 0
        if a > t_ol:
            noconv = 1
            t = (2 * math.copysign(a, ajj - akk)) / (d + math.sqrt(d * d + 4 * a * a))
            cs = 1.0 / math.sqrt(1 + t * t); sn = cs * t; ca = ajk.real / a; sa = ajk.imag / a
            j00 = cs; j11 = cs; j01 = -sn * complex(ca, sa); j10 = sn * complex(ca, -sa)
            for i in range(n):  # A = A * J
                aj = j00 * sA[i][j] + j10 * sA[i][k]; ak = j01 * sA[i][j] + j11 * sA[i][k]; sA[i][j] = aj; sA[i][k] = ak
            for i in range(n):  # A = J' * A
                aj = j00 * sA[j][i] + j10.conjugate() * sA[k][i]; ak = j01.conjugate() * sA[j][i] + j11 * sA[k][i]; sA[j][i] = aj; sA[k][i] = ak
            for i in range(n):  # V = V * J
                vj = j00 * sV[i][j] + j10 * sV[i][k]; vk = j01 * sV[i][j] + j11 * sV[i][k]; sV[i][j] = vj; sV[i][k] = vk
        converge &= (not noconv)
    sW = [sA[i][i].real for i in range(n)]
    if sW[0] > sW[1]: sW[0], sW[1] = sW[1], sW[0]; sorder[0], sorder[1] = sorder[1], sorder[0]
    Z = [[sV[i][sorder[c]] for c in range(n)] for i in range(n)]
    return sW, Z, sweep, sA
for A in ([[2,1],[1,3]], [[0,1],[1,0]], [[1,2],[2,-1]], [[3,1],[1,2]], [[5,1],[1,1]], [[2,1],[1,2]], [[-3,1],[1,-2]]):
    W, Z, sw, sA = heevj(A); M = np.array(A, dtype=complex); Zm = np.array(Z)
    res = max(np.abs(M @ Zm[:, c] - W[c] * Zm[:, c]).max() for c in range(2))
    off = abs(sA[0][1])
    print(f"A={A}: W={[round(x,6) for x in W]} sweeps={sw} final|offdiag|={off:.2e} residual={res:.2e}  numpy={np.round(np.linalg.eigvalsh(M),6)}")

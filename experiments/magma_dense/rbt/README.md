# MAGMA v2.10.0 `magma_dgesv_rbt`: CPU port and counterexamples (2026-11-25, CPU only, not yet run on GPU)

`rbt_port.py` ports, line by line from the source:
- `init_butterfly` (src/zgerbt_gpu.cpp): u[i], v[i] = exp((rand()/RAND_MAX − 0.5)/10), i < 2n, interleaved draws. glibc `rand()` (TYPE_3, seed 1) is reimplemented and checked against its first outputs (1804289383, 846930886, 1681692777).
- `zprbt` (magmablas/zgerbt.cu): level 1 on the four quadrants with u[n:], v[n:], then level 2 on the whole matrix with u[:n], v[:n]. Elementary block (zgerbt_kernels.cu l.47–85): out = D_u·R·A·R·D_v, R = [[I, I], [I, −I]], halves ceil/floor, zero padding.
- `zprbt_mtv` (Uᵀb) and `zprbt_mv` (V y); apply_transpose_vector has `if (n < 2) return;` (l.174), apply_vector has `if (n < 1) return;`.
- no-pivot LU, `gesv_nopiv` (getrs only if info == 0), `zgerfs_nopiv` control flow (info reset to 0, ITERMAX = 30, reject-form test, no fallback), and `iter` dropped by zgesv_rbt.

Self-check: for n ≥ 4 the transformed matrix equals (Uᵀ from mtv)·A·(V from mv) to 1e-15, and dgesv_rbt solves to ~3e-16 (`rbt_port_results.txt`).

## Finding 1: wrong answer for n = 1, 2, 3 (deterministic, independent of the rand() values)
The size-1 early return in apply_transpose_vector skips the u scaling of b, while the matrix kernel scales 1×1 blocks. The transformed system is inconsistent with the original one; refinement works on that same inconsistent pair, so it cannot fix it. Diagonally dominant random A, refine = true:

| n | ‖prbt(A) − UᵀAV‖/‖A‖ | rel. error | info |
|---|---|---|---|
| 1 | 6.6e-2 | 6.0e-2 | 0 |
| 2 | 6.7e-2 | 3.4e-2 | 0 |
| 3 | 3.7e-3 | 3.4e-3 | 0 |
| ≥ 4 | ~1e-15 | ~3e-16 | 0 |

## Finding 2: a well-conditioned A for which dgesv_rbt returns garbage with info = 0 (`rbt_zero_pivot.py`)
A = U⁻ᵀ M V⁻¹ with M well conditioned (singular values 1…3) and M[0,0] = 0, n = 64, using the u, v of the first init_butterfly call in a glibc process. After FP64 rounding the transformed (0,0) entry is 7.1e-18 (not exactly 0):

| case | cond(A) | transformed A[0,0] | dgesv_rbt | pivoted FP64 solve |
|---|---|---|---|---|
| M[0,0] = 0 | 3.09 | 7.1e-18 | **info = 0, x finite, rel. error 3.6e35** | 1.4e-15 |
| M[0,0] = 1e-14 | 3.12 | 9.9e-15 | info = 0, 3.8e-16 (refinement repairs it) | 1.3e-15 |
| M[0,0] = 1e-8 | 3.05 | 1.0e-8 | info = 0, 3.7e-16 | 9.6e-16 |
| control | 3.17 | 0.125 | info = 0, 2.9e-16 | 1.7e-15 |

Matrix and RHS saved as `A_zero_pivot_n64.npy`, `b_zero_pivot_n64.npy`.

## Caveats
- The n = 64 instance depends on the process's rand() state (first call, glibc). Any earlier rand() call, another libc, or a second zgesv_rbt call in the same process gives other u, v.
- On the GPU the transform's rounding (FMA contraction) may give a different tiny value or an exact 0 for A[0,0]; with an exact 0, getf2_nopiv sets info and skips the column scaling, getrs is skipped, gerfs starts from the unsolved Uᵀb, and the singular U yields Inf/NaN, which the reject-form test accepts — again info = 0. Both need GPU confirmation.
- The hybrid GPU path factors panels on the CPU (getf2_nopiv for panels narrower than 128); the port uses one unblocked no-pivot LU.

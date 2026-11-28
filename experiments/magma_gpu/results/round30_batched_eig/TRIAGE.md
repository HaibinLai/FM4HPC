# Round 30: batched Jacobi eigensolver and SVD (roxy, Windows), triage

`probes/round30_batched_eig.cpp` tests two routines:
- `magma_zheevj_batched_small_sm` with n = 1–32;
- `magma_zgesvj_batched` on square and tall shapes up to 131 × 64.

Each batch holds 56 matrices: 8 spectrum kinds × 7 seeds. The kinds are spread (alternating signs), clustered 1e−12, cond 1e12, rank n/2, rank 1, zero, identity and NaN. Results are compared with LAPACK zheev / zgesvd.

## Finding 1 (Windows / LLP64): the sort swaps only the low 32 bits of every 64-bit value

Symptoms:
- **heevj**: 313 of 784 cases are WRONG with info 0. The sorted eigenvalues are slightly perturbed: relative errors of about 2⁻²⁰, which gives the ratio ≈ 2³² seen in the CSV. They are also paired with the wrong eigenvectors (residual ratio about 1e15).
- **gesvj**: 1016 of 1568 cases are WRONG. Only n = 1, which needs no sort, is clean.

Isolation (`heevj_2x2.txt`):
- Every 2 × 2 matrix whose Jacobi diagonal ends in descending order (a_jj ≥ a_kk, positive definite or not) gets a W that is not sorted, while its eigenvector columns are sorted. Matrices that end ascending, and diagonal matrices, are correct.
- The failure does not depend on batchCount (`heevj_batchcount.txt`).
- A line-by-line Python emulation of the kernel is correct on the same matrices (`scripts/heevj_emu.py`, `heevj_2x2_emulation.txt`).

Root cause: `magmablas/swap_scalar.cuh` swaps a double (and each half of a complex double, and a whole single-complex value) as one `long`:

    magmablas_dswap_scalar_device(double &a, double &b) { swap_internal<long>((long*)&a, (long*)&b); }

On Windows `long` is 32 bits; the probe prints `sizeof(long) = 4`. So only the low 32 bits of the two doubles are exchanged. Meanwhile the int index array is swapped correctly, so the values and the vectors end up permuted differently.

Confirmation without touching the library (`heevj_sort_flag.txt`): the expert interface with sort_flag = 0 gives 0 wrong of 64; sort_flag = 1, the default, gives 4/16, 16/16, 16/16 and 16/16 wrong at n = 2, 5, 16, 32.

Affected code: the users of swap_scalar.cuh, which are [zcds]heevj/syevj_batched_small_sm, [zcds]gesvj_batched_small_sm and sort.cu/sort.cuh, the sort used by the blocked gesvj. The s precision (float via int) is unaffected. Linux (LP64) builds are unaffected; earlier V100 runs of heevj were correct.

Lean: `swap_long_llp64_wrong`.

## Finding 2: blocked zgesvj scales U by 1/σ with no zero guard

Zero matrices with n ≥ 15 (the blocked path) give non-finite U with info 0, in 7 of 7 cases at every size. n ≤ 8, where the fused small-size kernel is used, is fine. The code is `rsigma[j] = 1. / ssigma[j]` in gesvj_kernels.cuh (finalize vectors), so σ = 0 gives ∞, and 0·∞ gives NaN. This is platform-independent by reading.

Lean: `gesvj_zero_sigma_nan`.

## Minor

- The heevj documentation says the eigenvalues are sorted "so that S(i) >= S(i+1)", but the code (comment "ascending order") sorts ascending.
- For larger sizes, gesvj prints "zheevj ... requires too many threads or too much shared memory" to stdout (200 lines); its internal fallback continues.

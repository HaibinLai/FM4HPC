# Round 27 sweep: triage of every non-OK cluster

Sweep: `probes/round27_sweep.cpp` on roxy (RTX 5070 Ti, sm_120), MAGMA 2.10.0 vs OpenBLAS 0.3.34 LAPACK. One CSV per
operator group in this directory (17,327 cases in total) (`svd.csv` from the rerun without Inf inputs; see the end). Summaries were made with
`scripts/sweep_summary.py`. Follow-ups are in `probes/round27_focus.cpp` and `results/round27_focus_results.txt`.

Verdicts used below:
- **NEW**: a new MAGMA defect, reproduced in isolation.
- **KNOWN**: an already recorded finding, seen again.
- **HARNESS**: a false positive caused by the sweep itself.
- **NOT A BUG**: behaviour that is allowed, documented, or caused by rounding.
- **UNREPRODUCED**: a single occurrence that did not recur.

| Group | Routine | Cluster | Verdict | Reason / evidence |
|---|---|---|---|---|
| blas | magmablas_zgemm | k = 0, all 12 sizes (beta = 0.5; and beta = 0 with C = NaN) | **NEW** | zgemm.cu l.249 returns when k ≤ 0 before C := beta·C. In the focus run, C was left unchanged with beta = 0.5, and NaN stayed in C with beta = 0 (`gemm_k0_beta_ignored`). |
| blas | magmablas_zgemv, zhemv | beta = 0 with y = NaN, all n ≥ 1 | **NEW** | `y = alpha·Ax + beta·y` is unconditional (gemv_template_device.cuh l.81/107/209, zhemv.cu l.468). Their own docs say y need not be set when beta = 0. The focus run gave y = NaN where 2 was expected (`gemv_beta0_reads_y`). |
| blas | lange, lanhe, lacpy, laset, lascl, geadd, transpose | — | OK | All agree with LAPACK, including NaN propagation of the norms. |
| lu | zgetrf_native | NaN/Inf input: info > 0, LAPACK and zgetrf_gpu give 0 | **NEW (low)** | The two modes of the same API disagree on status. |
| lu | zgetrf_nopiv_gpu | rand/big/tiny: occasional large residual | NOT A BUG | No pivoting is unstable by design. |
| lu | zgetrf, zgetrf_gpu | — | OK | |
| chol | zpotrf_native | indefinite or zero: "WRONG" with info equal | HARNESS | After info > 0 the trailing contents are unspecified; the sweep compared their finiteness. |
| qr | zgeqp3 | dupcol at n = 63, 65, 128, 255, 256, 513 and three rectangles: A·P ≠ Q·R (residual ratio 1e12–1e14), info 0 | **NEW (major)** | Isolated in the focus run: residual 1–6, and the rank is not revealed (\|R(n−1,n−1)\| = 0.1–0.27 against 1e−16 from LAPACK). On random rank-n/2 matrices, 18 of 78 cases are wrong. Two root causes in magma_zlaqps were confirmed by patching a copy: 18 → 0 wrong (`geqp3_swap_restore_short_all`, `geqp3_norm_rows_off_by_one`). |
| qr | zgeqrf_gpu | all kinds: "WRONG" | HARNESS | The documented interface keeps the diagonal blocks of R in dT, not in dA (zgeqrf_gpu.cpp l.97). |
| qr | zgeqlf | −7 at most sizes | HARNESS | The sweep's lwork was below the documented 2·nb² (it was sized with the geqrf nb). |
| qr | all | NaN/Inf: NFDIFF | HARNESS | The sweep hard-coded the LAPACK output as finite for this group. |
| solve | zgels, zgels_gpu | zero column: info 0 where LAPACK gives i; zero matrix: NaN solution where LAPACK gives 0 | **NEW (deviation)** | MAGMA documents only info ≤ 0, but the result is a silent NaN solution (`gels_singular_nan_success`). |
| solve | zhesv | NaN input: both give info > 0 but at different positions | NOT A BUG | Bunch–Kaufman comparisons with NaN are arbitrary. |
| solve | zgesv(_gpu), zposv(_gpu) | Inf input: NFDIFF | NOT A BUG | Both report the same info; non-finite contents are unspecified. |
| inv | ztrtri_gpu | n = 257, 513 with a zero diagonal: info 0 / 1 where LAPACK gives 1 / n | **KNOWN** | Block-local info, already recorded as `trtri_gpu_instances`, now reproduced on sm_120. Its singularity check is commented out (ztrtri_gpu.cpp l.164–174). |
| inv | ztrtri_gpu | n = 0: −5 | **NEW** | The small path calls LAPACK with lda = n (`small_path_lda_n_rejects_n0`). |
| inv | zgetri_gpu, zpotri(_gpu), ztrtri | — | OK | |
| batched | zpotrf_batched, zgetri_outofplace_batched | singular/indefinite: "WRONG" with info equal | HARNESS | The same unspecified-contents comparison as above. |
| batched | zgetrf_batched | dupcol n = 3, 8: info 0 where LAPACK gives n | NOT A BUG | The pivot is exactly zero in LAPACK's operation order but 1e−17 in MAGMA's. |
| batched | zgeqrf_batched | NaN: NFDIFF | HARNESS | LAPACK finiteness was hard-coded, as in the qr group. |
| reduce | zgebrd | (146, 129) zcolmid/dupcol: \|d\|, \|e\| differ | NOT A BUG (likely) | After an exact zero, the remaining Householder vectors are arbitrary, so the bidiagonal is not unique. Not checked further. |
| reduce | zhetrd | — | OK | |
| heig | zheevdx, zheevdx_2stage | n = 1: mout not set | **NEW** | The quick return sets only w. In the focus run, mout stayed −77 (`heevdx_n1_mout_unset`). zheevr and zheevx set it. |
| heig | zheevd_gpu (+ zheevdx_gpu, zheevr_gpu, zheevx_gpu, zpotrf_mgpu) | n = 0: −5 / −6 / −4 | **NEW** | The small path calls LAPACK with lda = n. The focus `nzero` run covered all six; zpotrf_gpu returns early and gives 0. |
| heig | zheevdx_2stage(V) | −14 for small n | HARNESS | The sweep's lwork was below the stage-2 requirement. |
| heig | zheevd_gpu(N) | herm_big, n = 129: ratio 3.8e16 once | UNREPRODUCED | 30 repeats with the identical input and workspace, plus a scaling sweep, were all correct. |
| heig | all | NaN/Inf: info and finiteness differ from LAPACK zheev | NOT A BUG | Behaviour on non-finite input is unspecified. |
| geev | zgeev | NaN: both −4, except MAGMA 0 at one size | NOT A BUG (minor) | LAPACK 3.12 rejects NaN with −4, and MAGMA follows except in one small case. |
| svd | zgesvd(S), zgesdd(S) | NaN input: info > 0 where the LAPACK reference (computed with job N) gives 0; all finite inputs OK (rerun without Inf: 338 OK of 378 cases) | NOT A BUG | Behaviour on non-finite input is unspecified. |
| svd | zgesvd, zgesdd | Inf input with vectors: hang | NOT MAGMA | OpenBLAS 0.3.34's own LAPACK zgesvd/zgesdd hang on this 3×3 matrix with job = S (`svd_inf_hang.txt`). MAGMA's small path calls LAPACK, so it inherits the hang. The SVD group was rerun without Inf. |

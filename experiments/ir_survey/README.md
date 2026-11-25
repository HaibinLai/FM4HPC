# Running the predictions from the IR acceptance-logic survey (node CPU, 2026-11-24/25)

- Predictions: `operators/IR_SURVEY_zh.md`.
- Machine: eightsocket, CPU only, no GPU. Toolchain: a conda environment under `/ssd/haibin/lean4hpc-ir/`, approved by the user and deleted after the runs. Build scripts in this directory; `env_node.sh` sets the paths.
- Each library is built from source at the commit we read, except MUMPS, which uses conda-forge `mumps-seq` 5.8.2 (we read 5.8.1).

## Summary

| Library / routine | Prediction | Result | File |
|---|---|---|---|
| SuperLU (serial) a8d98be, `dgssvx`+`dgsrfs` | D11: NaN dropped by `SUPERLU_MAX` | **Confirmed.** 2 decoupled blocks, b[0]=NaN or Inf: 500 of 1000 x entries NaN, info=0, berr = 6.94e-17 (same as the finite baseline). ferr = NaN (not masked). | `superlu_nan_results.txt` |
| SuperLU_DIST 239280c, `psgssvx_d2` (FP32 LU, FP64 refinement) | D11 | **Confirmed, stronger than serial.** b[0]=NaN: 500 NaN in x, but info, normwise bound 1.03e-7, componentwise bound 1.03e-7 and berr 2.96e-16 are **all identical to the clean baseline**. NaN in the second block: berr = NaN, componentwise bound = FLT_MAX, normwise still 1.03e-7. | `sludist_d2_results.txt` |
| MUMPS 5.8.2 (seq), ICNTL(10)=3, ICNTL(11)=2 | D1 via `idamax` | **Confirmed for b[0]=NaN.** INFOG(1)=0, omega1 = 1.94e-16 (same as baseline), 500 NaN in x. NaN at b[500] or b[999] gives INFOG(1)=2 (a warning). | `mumps_nan_results.txt` |
| PaStiX e2ecf99, GMRES refinement | `normx` computed once; x0=0 ⇒ every restart uses r=b | **Confirmed.** x0 = 0 (documented as allowed: "any other initial guess"): runs to itermax=250, x ≈ 249·x_true, ‖b−Ax‖/‖b‖ = 249, return code `PASTIX_SUCCESS`. On 1D (n=1000) and 3D (n=8000) Laplacians. Normal use and nonzero x0 are fine. | `pastix_gmres_x0_results.txt` |
| PLASMA 548e0bd, `plasma_dsgesv` | P1 D1/D11: NaN accepted; P2 D9: failure skips the FP64 fallback | **Both confirmed.** P1: b with NaN (row 0 or 150), or FP32 pivot growth (2^90·W(40), 2^120·W(40); FP64 solves exactly): retval 0, iter 0, all of X NaN. P2: FP32-singular A or b overflowing FP32: retval ≠ 0, iter = −31 (claims fallback), X never written. | `plasma_dsgesv_results.txt` |
| STRUMPACK edc9a72 | D3: absolute atol = 1e-10; D10: mixed solver always SUCCESS | **Confirmed, and broader than predicted.** (A,b) × 1e-12 or 1e-20: x ≡ 0 with SUCCESS, both in `SparseSolverMixedPrecision<float,double>` and in the **default double `SparseSolver<double>`**. b[0]=NaN: x ≡ 0 (not NaN), SUCCESS. PREC_GMRES at 1e-12 is fine. | `strumpack_mixed_results.txt` |
| SLATE 878439b, LAPACK API `slate_dsgesv` | D7: x ignored, X aliases b; D10: info = 0 hard-coded | **Confirmed, worse than predicted.** Well-conditioned 100×100 diagonally dominant matrix: info=0, iter=23, x untouched, **b overwritten with zeros**, so the solution exists in neither array. Exactly singular A: info=0. | `slate_dsgesv_results.txt` |

## Not run

- MAGMA `zchesv` and `dshposv_gmres`: they need the GPU.
- hpl-ai, Ginkgo, PETSc: lower priority, not run.

## Notes and corrections made while running

- PaStiX: the first version of the probe generated the RHS after `numfact` and used 192 threads. Its control case looked wrong, but that was the harness: PaStiX's own `example/simple` passed. The spm Laplacian generator gives a near-singular matrix, so the verdict uses the true residual, not the error against x_true. Separately, BiCGSTAB with x0 = 0 on the 1D Laplacian once returned residual 4e10 with SUCCESS. Another run gave 7e-16, so it is not reproducible; it is not counted.
- STRUMPACK: the first verdict used a 1e-8 error threshold. The library's default `rel_tol` is 1e-6, so cases within 1e-5 are counted as fine.
- SuperLU_DIST: setting `NANROW=` to an empty string still injected a NaN at row 0 (`atoi("")`), so the baseline was rerun with the variable unset.
- PLASMA: OpenBLAS (pthreads) warned about OpenMP loops. The warnings are filtered out of the saved results.

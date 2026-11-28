# Running the predictions from the IR acceptance-logic survey (node CPU, 2026-11-24/25)

- Predictions: `operators/IR_SURVEY_zh.md`.
- Machine: eightsocket, CPU only, no GPU. Toolchain: a conda environment under `/ssd/haibin/lean4hpc-ir/`, approved by the user and deleted after the runs. Build scripts in this directory; `env_node.sh` sets the paths.
- Each library is built from source at the commit we read, except MUMPS, which uses conda-forge `mumps-seq` 5.8.2 (we read 5.8.1).

## Summary

| Library / routine | Prediction | Result | File |
|---|---|---|---|
| SuperLU (serial) a8d98be, `dgssvx`+`dgsrfs` | D11: NaN dropped by `SUPERLU_MAX` | **Confirmed.** 2 decoupled blocks, b[0]=NaN or Inf: 500 of 1000 x entries NaN, info=0, berr = 6.94e-17, a clean value (equal to the finite single-tridiagonal run; no finite two-block baseline was run). ferr = NaN (not masked). Smallest reproducer in the same file: A = I, n = 2, b = (NaN, 1) gives info=0, berr = 0. | `superlu_nan_results.txt` |
| SuperLU_DIST 239280c, `psgssvx_d2` (FP32 LU, FP64 refinement) | D11 | **Confirmed, stronger than serial.** b[0]=NaN: 500 NaN in x, but info, normwise bound 1.03e-7, componentwise bound 1.03e-7 and berr 2.96e-16 are **all identical to the clean baseline**. (The normwise bound 1.03e-7 equals the formula's floor `max(1,sqrt(nz))·eps` = sqrt(3)·2^-24, `psgsrfs_d2.c:566`, which is why it cannot change; berr equal to the baseline is the masking.) NaN in the second block: berr = NaN, componentwise bound = FLT_MAX, normwise still 1.03e-7. | `sludist_d2_results.txt` |
| MUMPS 5.8.2 (seq), ICNTL(10)=3, ICNTL(11)=2 | NaN-masking backward error | **Confirmed for b[0]=NaN.** INFOG(1)=0, omega1 = 1.94e-16 (same as baseline), 500 NaN in x. NaN at b[500] or b[999] gives INFOG(1)=2 (a warning, probably the near-zero solution-norm warning, `dsol_aux.F:1125`, inferred). Mechanism (corrected 2026-11-25): not `idamax` — DXMAX stayed finite; the NaN rows fall through the per-row comparisons in `dsol_aux.F:889–899` (inferred from code; the accept-form test at l.904 is itself NaN-safe). | `mumps_nan_results.txt` |
| PaStiX e2ecf99, GMRES refinement | `normx` computed once; x0=0 ⇒ every restart uses r=b | **Confirmed.** x0 = 0 (documented as allowed: "any other initial guess"): runs to itermax=250, x ≈ 249·x_true, ‖b−Ax‖/‖b‖ = 249, return code `PASTIX_SUCCESS`. On 1D (n=1000) and 3D (n=8000) Laplacians. Normal use and nonzero x0 are fine. | `pastix_gmres_x0_results.txt` |
| PLASMA 548e0bd, `plasma_dsgesv` | P1 D1/D11: NaN accepted; P2 D9: failure skips the FP64 fallback | **Both confirmed.** P1: b with NaN (row 0 or 150), or FP32 pivot growth (2^90·W(40), 2^120·W(40); FP64 `dgesv` returns info = 0, the probe did not compare it with x_true): retval 0, iter 0, all of X NaN. P2: FP32-singular A or b overflowing FP32: retval ≠ 0, iter = −31, X never written. The −31 is most likely read from `Rnorm`/`Xnorm` buffers that are `malloc`ed (`zcgesv.c:216–217`) and never written once the tasks are skipped (inferred), not a real iteration count. Probe artifacts: `relerr_vs_dgesv = 0` on NaN rows comes from `fmax` dropping NaN; `dgesv_info = -7` on NaN-b rows is LAPACKE rejecting NaN input. | `plasma_dsgesv_results.txt` |
| STRUMPACK edc9a72 | D3: absolute atol = 1e-10; D10: mixed solver always SUCCESS | **Confirmed, and broader than predicted.** (A,b) × 1e-12 or 1e-20: x ≡ 0 with SUCCESS, both in `SparseSolverMixedPrecision<float,double>` and in the **default double `SparseSolver<double>`**. b[0]=NaN: x ≡ 0 (not NaN), SUCCESS. PREC_GMRES at 1e-12 is fine. | `strumpack_mixed_results.txt` |
| SLATE 878439b, LAPACK API `slate_dsgesv` | D7: x ignored, X aliases b; D10: info = 0 hard-coded | **Confirmed, worse than predicted.** Well-conditioned 100×100 diagonally dominant matrix: info=0, iter=23, x untouched, b overwritten (printed b[0..2] = 0 and max|b − x_true| = 1, consistent with all zeros), so the solution is in neither array. Inferred mechanism: with X aliasing B, X and B are both driven to 0 and the reject-form test then passes at iter 23. Exactly singular A: info=0. | `slate_dsgesv_results.txt` |

## Not run

- MAGMA `zchesv` and `dshposv_gmres`: they need the GPU.
- hpl-ai, Ginkgo, PETSc: lower priority, not run.

## Notes and corrections made while running

- PaStiX: the first version of the probe generated the RHS after `numfact` and used 192 threads. Its control case looked wrong, but that was the harness: PaStiX's own `example/simple` passed. The spm Laplacian generator gives a near-singular matrix, so the verdict uses the true residual, not the error against x_true. Separately, BiCGSTAB with x0 = 0 on the 1D Laplacian once returned residual 4e10 with SUCCESS. Another run gave 7e-16, so it is not reproducible; it is not counted.
- STRUMPACK: the first verdict used a 1e-8 error threshold. The library's default `rel_tol` is 1e-6, so cases within 1e-5 are counted as fine.
- SuperLU_DIST: setting `NANROW=` to an empty string still injected a NaN at row 0 (`atoi("")`), so the baseline was rerun with the variable unset.
- PLASMA: OpenBLAS (pthreads) warned about OpenMP loops. The warnings are filtered out of the saved results.

## Corrections after the report-drafting audit (2026-11-25)

A second read of the raw result files while drafting the upstream reports (`operators/upstream_reports/`) found:
- SuperLU: the "same as the finite baseline" comparison was with a different matrix (single tridiagonal); see the table. SuperLU was built with the node's gcc 13.3, not the conda toolchain (`build_superlu.sh` does not source `env_node.sh`).
- PaStiX: `IPARM_NBITER` is a running maximum across calls (`pastix_task_refine.c:163`). The GMRES x0 = 0 row's 250 is real (the previous call had 0); the 250 on later rows is carried over and is not their iteration count. The saved probe prints a `rel_err=` column that the saved results lack, so the results came from a slightly earlier probe revision.
- MUMPS mechanism corrected (table). PLASMA and SLATE wording narrowed to what the result files show (table).
- The survey predicted A = I, b = (NaN, 1) for SuperLU_DIST and MUMPS; only the two-block n = 1000 input was run for them.

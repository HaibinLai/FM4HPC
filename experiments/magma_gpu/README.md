# MAGMA 2.10.0 on GPU 0: confirmation of the code-level predictions (2026-11-25)

**Environment**
- Node and GPU: eightsocket, GPU 0 only (Tesla V100-PCIE-32GB, driver 580.178.04). GPU 0 was idle before the runs.
- MAGMA: official tarball (`inputs/tarball.sha256`), built with `scripts/build_magma.sh` and `inputs/make.inc.used`: CUDA 12.0, `GPU_TARGET = Volta`, system reference LAPACK/BLAS (`liblapack.so.3`, `libblas.so.3`), dense + sparse libraries.
- Work directory: `/ssd/haibin/magma_gpu`, deleted after the runs.

**Probes** (sources in this directory; raw outputs in the `*_results.txt` files)
- `probes/dense_probe.cpp`, with modes `dense`, `rbt64` (its own process, so it is the first `rand()` use), `nan1`, `nan2` and `clean2`.
- `probes/sparse_probe.cpp`.
- `probes/audit_probe.cpp`: the dense status-loss audit.
- `shims/guard_malloc.c`: an LD_PRELOAD guard for out-of-bounds writes. `compute-sanitizer` on the node lacks its injection library.
- `probes/fix_probe.cpp` and `../magma_fgmres/magma_fgmres_gpu.cpp`: the patched GMSTRS, see below.
- Inputs `inputs/zchesv_inputs.txt` and `inputs/rbt_zero_pivot_n64.txt` are bit-identical to the CPU-port inputs (hex floats).

## Layout

| Directory | Contents |
|---|---|
| `probes/` | the C++ probes, one per GPU round (`roundN_probe.cpp`) plus the early probes (`dense_probe.cpp`, `sweep_probe.cpp`, …) |
| `scripts/` | build scripts (`build_magma.sh`, `build_openblas_magma.sh`), per-round runners (`run_roundN.sh`, `wait_and_runN.sh`) and the source scanners `info_init_scan.py` (functions that read `*info` before writing it; HANDOFF §29.32) , `argnum_scan.py` + `argnum_review.py` (argument checks whose `−k` is not the tested argument; §29.33) , `dropped_status_scan.py` (sub-call status never read; §29.34) and `wrap32_scan.py` (workspace formulas that overflow 32-bit `magma_int_t`; §29.36) |
| `scripts/roxy/` | Windows scripts for roxy (RTX 5070 Ti, sm_120): import libraries from the CUDA wheels (`mkimplib.bat`), the MAGMA CMake/Ninja build (`build_magma_roxy.bat`), probe compile/run (`build_probe.bat`, `run_probe.bat`) and the round-27 sweep runner (`run_sweep.bat`); see HANDOFF §29.41 |
| `shims/` | LD_PRELOAD libraries: `guard_malloc.c` (device overflow guard), `fail_malloc.c` / `fail_malloc_cpu.c` / `fail_dsteqr.c` (fault injection), `segv_bt.c` |
| `inputs/` | matrix inputs, `make.inc.used`, `tarball.sha256` |
| `results/` | everything the node printed, per round |

On the node everything runs from one flat directory, `/ssd/haibin/magma_gpu`. Upload with, for example,
`tar -czf - -C scripts build_magma.sh run_roundN.sh -C ../probes roundN_probe.cpp -C ../shims fail_malloc.c | ssh … 'tar xzf -'`,
and copy the printed output back into `results/`. GPU use: GPU 0, and GPU 1 for the multi-GPU rounds (authorised 2026-11-26); check
that the GPUs are idle first, and delete the node directory afterwards.

## Dense mixed-precision and refinement routines (`results/dense_results.txt`)

| Prediction (Lean / CPU port) | GPU 0 |
|---|---|
| `x_overflow_accepted`: a = 2^-100, b = 2^28, so the FP64 solution 2^128 overflows FP32 | dsgesv_gpu and dsposv_gpu return **iter = 0, info = 0, x = inf** (n = 1) or **nan** (n = 2, 1024). Control a = 2^-90 is exact. |
| `zcgesv_b_overflow_accepted`: zcgesv does not check B | b = 2^200: **dsgesv iter = 0, info = 0, x all NaN** (n = 2, 1024) |
| `zcposv_b_overflow_fallback` | dsposv with the same b: iter = −2, exact (correct fallback) |
| `zchesv_stub`: X = B, info = 0 | d = 2^-24, seeds 1 and 2: **iter = −31, info = 0, X == B**, relative errors 0.45 and 1.84 |
| `diverge_accepted` | d = 2^-24, seed 0: **iter = 27 ("converged"), info = 0, x contains Inf** |
| (control) d = 2^-20 | all three seeds converge (iter 12–15), errors 4e-16 to 8e-16 |
| `spd_tiny_rhs_returns_zero` (‖b‖₂ ≤ 2^-53) | dshposv with ‖b‖₂ = 1e-17: **x = 0, iter = 0**. With ‖b‖₂ = 1e-15: normal. |
| s² tolerance (`enforced_rule`) | dshposv with (A, b)·2^40: **iter = 1, backward error 5.5e7 × threshold**, relative error 1.8e-7. With 2^-40: iter = 256 = maxiter, reported as success. |
| `dshposv_always_success` | return value 0 on every call. Even at k = 0, bwd/thr = 2.96 (above the documented threshold). |
| RBT n ≤ 3 (vector butterfly skips size-1 blocks) | **dgesv_rbt n = 1, 2, 3: relative errors 6.0e-2, 2.6e-2, 2.7e-2, info = 0**. n = 4–6: about 1e-16. |
| RBT n = 64 constructed zero pivot (first rand() use) | **info = 0, relative error vs LAPACK dgesv 1.58e13** (cond(A) = 3.09). The CPU port predicted 3.6e35; the magnitude differs through rounding. |

## dshposv with nrhs = 2: device out-of-bounds write (`results/guard_results.txt`, LD_PRELOAD guard)

| Run | Result |
|---|---|
| nrhs = 2, b[0] = NaN (NaN passes `dlag2s`, so FGMRES runs to restrt = 500) | **OUT-OF-BOUNDS WRITE: allocation of 2,048,000 bytes (`d_w` = 500 columns × n2 = 512 × 8 B), guard bytes [0, 4095] modified**, i.e. exactly one extra column, as `nonconverged_oob` predicts |
| nrhs = 1, b[0] = NaN | no OOB; **iter = 500, info = 0, ret = 0** (`spd_nan_runs_to_maxiter`) |
| nrhs = 2, finite b | no OOB (converges in 2 steps); **column 2 of X is all zero** |

## Sparse solvers (`results/sparse_results.txt`)

| Prediction | GPU 0 |
|---|---|
| `cg_merge_false_convergence`: A = diag(2,3), b = (2, i) | **info = 0, numiter = 1, x = (2, i)**, true residual / ‖b‖ = 1.26 (√8/√5), relative error 1.14. A = diag(1, 1.5), b = (1, 0.5i): same result. |
| complex HPD (n = 64, cond 10) does not converge | info = −205 after 1000 iterations, errors 2.7e4 and 2.8e3. Real SPD in complex storage (control): 32 iterations, about 1e-10. |
| `bicg_breakdown_success` | b = (NaN, 1): **info = 0, numiter = 1**, final_res = NaN, x left at 0 |
| `cgres_exact_x0_nonspd` | x0 = exact solution: **info = −203 (NONSPD)**. Control x0 = 0: info = 0. |
| `wrapper_unhandled_success` | solver = ITERREF: prints "solver class not supported", **returns 0, x = x0 = (7, 7)** |
| `pcg_merge_upgrade` | x0 ~ 1e6, maxiter = 40: **info = 0 with final_res/‖b‖ = 2.9e-6** (2.9e4 × rtol). maxiter = 50: 3.9e-9, info = 0. maxiter ≤ 30: −201. |

## Dense status-loss audit (`results/audit_results.txt`)

| Finding (source line) | GPU 0 |
|---|---|
| `zgbtrs_batched_strided`: dB pointer array taken from `get_dAarray()` (zgbtrs_batched.cpp:343) | after a correct gbtrf_batched_strided (info 0), **all 4 batch solutions are NaN, return value 0** |
| `zheevdx_2stage` Upper: he2hb returns NOT_IMPLEMENTED and the value is discarded (zheevdx_2stage.cpp:445) | prints "not yet implemented", **info = 0, m = 512, eigenvalue error 12.9** (spectrum −12.8…12.9). Lower: 1e-13. |
| `zhesv_nopiv_gpu`: last pivot unchecked | [[1,1],[1,1]] and [[0]]: **info = 0, x = NaN**. Control: correct. |
| `zpotf2`: 0-based `step` written as info (zpotf2.cu:316) | A(0,0) = −1: LAPACK info = 1. **zpotrf_native(Upper) info = 0, zpotf2_gpu info = 0 (both uplo)**. zpotrf_native(Lower) = 1. |
| `ztrtri_gpu`: singularity check commented out; the block info is overwritten and not offset | A(0,0) = 0: **info = 0** (LAPACK 1), finite garbage inverse. A(300,300) = 0: **info = 45** (LAPACK 301). A(511,511) = 0: **info = 256** (LAPACK 512). |

## Notes
- In `results/audit_results.txt`, the relerr column of T1 shows 0 for batches 1–3 because `fmax` drops NaN. x[0] is NaN in all four batches.
- The divergence-versus-X = B outcome of zchesv moved between seeds relative to the CPU port (FP32 rounding differs), but both failure kinds occur.

## The GMSTRS fix compiled and run (`../magma_fgmres/fix/gmstrs_fix_v2.patch`)

The first patch (`gmstrs_fix.patch`) did not compile. It added `#include <cmath>` and `<vector>` after `magma_internal.h`, whose `min`/`max` macros break `<algorithm>`. `gmstrs_fix_v2.patch` moves the two includes before it; nothing else changes.

Setup: the same binary was run against the original and the patched `libmagma.so`, switching `LD_LIBRARY_PATH`.

**24-call probe** (`results/fgmres24_original.txt`, `results/fgmres24_fixed.txt`; n = 1024, A = 2^k·A0, k = −80…60, dsgesv_iteref and dhgesv_iteref):

| | calls | report success (iter ≥ 0) | of these, violate the documented condition (bwd/thr ≥ 1 or non-finite) |
|---|---|---|---|
| original | 24 | 24 | **20** |
| fixed | 24 | 15 | **0** |

In the fixed run, the 9 non-success calls fall back to FP64 and also satisfy the condition (bwd/thr 0.689):
- dhgesv with k ≤ −40: certification fails after FP16 underflow, iter = −501.
- dhgesv with k ≥ 20: the FP16 range check gives iter = −2.

dsgesv takes iter = 4 at every scale (scale-invariant), with bwd/thr ≤ 0.011.

**trans and nrhs** (`fixprobe_*.txt`, n = 1024):

| case | original | fixed |
|---|---|---|
| trans = MagmaTrans | iter = 2, info = 0, **error 6.6e-2 against the Aᵀx = b solution** (it solves Ax = b) | iter = −1 (FP64 path), 3e-15 |
| nrhs = 2 | **column 2 all zero**, iter = 2, info = 0 | both columns about 3e-15 |

**Cost** (`fixprobe_*.txt`, `results/cost_results.txt`). Wall time on well-conditioned k = 0 systems, best of 3:

| n | dsgesv_iteref orig → fixed | dhgesv_iteref orig → fixed | FP64 dgesv_gpu | host certification (D2H + Dot2) |
|---|---|---|---|---|
| 1024 | 0.53 → 0.49 s | 0.31 → 0.46 s | 0.08 s | 0.03 s |
| 2048 | 0.49 → 0.79 s | 0.42 → 0.64 s | 0.19 s | 0.12 s |
| 4096 | 1.36 → 2.20 s | 0.57 → 1.32 s | 0.74 s | 0.55 s |
| 8192 | 5.34 → 8.32 s | 1.59 → 4.78 s | 5.43 s | 2.35 s (copy 0.54 + Dot2 1.80) |

How to read these timings:
- The fixed runs do one more iteration (3 vs 2) and are far more accurate (1e-14 vs 1e-11), because the original tolerance carries an extra ‖A‖.
- Most of the added time is the host-side certification: a D2H copy of A and a serial row-wise Dot2 over column-major A. This is O(n²) work, which a GPU Dot2 kernel would make negligible.
- **This build links the reference CPU LAPACK/BLAS**, so the hybrid routines' CPU panels dominate: FP64 dgesv_gpu takes 5.4 s at n = 8192. These timings are therefore not representative of MAGMA's performance, and they cannot answer whether the fixed mixed-precision solver beats FP64.

## The fix with an optimized BLAS, and patch v3 (`results/perf_openblas_results.txt`, `results/perf_v3_results.txt`, `results/perf_large_results.txt`, `results/perf_small_results.txt`)

The reference-LAPACK timings above cannot judge the fix, so MAGMA was rebuilt against OpenBLAS 0.3.28 (built from source, `NO_FORTRAN=1 C_LAPACK=1`, 32 threads; `scripts/build_openblas_magma.sh`). Three libraries, same probe binary:
- `orig`: unpatched MAGMA 2.10.0;
- `fix` (v2): certification on the host (D2H copy of A plus a serial Dot2);
- `fix3` (v3, `../magma_fgmres/fix/gmstrs_fix_v3.patch`): the same Dot2 and row sums computed on the GPU by a one-thread-per-row kernel (`magmablas/dgmstrs_certify.cu`); only n-vectors are copied back, and the certified comparison is unchanged.

**Correctness is unchanged.** With OpenBLAS, the original reports success on 24/24 calls and violates the documented condition on **23** (20 with the reference build: the FP32 rounding differs). v2 and v3 both report success on 15 and violate it on **0**. With v3, trans = T gives iter = −1 and 3e-15; nrhs = 2 gives 3e-15 on both columns.

**Cost, n = 8192** (best of 3 after a warm-up; well-conditioned, k = 0):

| build | dsgesv_iteref | dhgesv_iteref |
|---|---|---|
| orig | 0.21–0.26 s | 0.17–0.18 s |
| v2 (host certification) | 3.34 s | 3.27 s |
| v3 (GPU certification) | 0.23–0.30 s | 0.20–0.35 s |
| FP64 dgesv_gpu | 0.29–0.43 s | |

At n = 16384 (`results/perf_large_results.txt`): orig 0.85 / 0.73 s, v3 0.90 / 0.91 s, FP64 1.09 s.

How to read these timings:
- The node was shared (load average 10–16 from other users' CPU jobs), so repeated runs of the same build differ by up to ±30%. Ranges above are the spread over runs.
- v3 removes the 11–16× slowdown of v2. At n ≥ 8192, v3 is within 25% of the original and still faster than FP64.
- At n ≤ 4096 (`results/perf_small_results.txt`, interleaved runs), v3 adds 0.03–0.17 s, i.e. 1.3–3×. This is one extra iteration plus the certification launch and copies; the absolute times are small.
- v3 always does one more iteration than the original (3 vs 2). It is also 1000× more accurate (1e-14 vs 1e-11), because the original's tolerance carries an extra ‖A‖∞.
- On a V100 FP64 is fast (7.8 TFLOP/s), so mixed precision gains little here even in the original. The question this round answers is only whether the certified gate costs more than the factorization; for n ≥ 8192 it does not.

## Sparse preconditioner setup (`results/precond_results.txt`)

| finding | GPU 0 |
|---|---|
| Jacobi setup misses a stored zero diagonal (`jacobi_stored_zero_passes`) | A = [[0,1],[1,2]] with the zero stored: **setup info = 0, D⁻¹ = (NaN, 0.5)**. In complex precision 1/(0+0i) gives NaN; real precision would give Inf. |
| ILU numerical zero pivot (`ilu_numerical_zero_unchecked`, predicted to end in PBiCG info = 0) | **not as predicted: `magma_z_precondsetup` crashes the process (segfault, exit 139)** for A = [[0,1],[1,0]] with the zero diagonal stored. The control [[2,1],[1,2]] gives setup 0 and PBiCG x = (0, 1). The HIP branch of `zilu.cpp` carries the comment "this version has the bug fixed where a zero on the diagonal causes a crash". The crash location was not determined (no gdb on the node). |

## Not run on GPU (all run in the second and third GPU rounds)
- LOBPCG with n > m (dead parameter check). There is a risk of a hang.
- An IC(0) breakdown matrix.
- The FGMRES sticky-success sequence, which needs a crafted mid-cycle NaN.
- The dense-audit items 6–15: potrf_vbatched Upper, hgetrf padding, gesvj_batched, getrf_m, and others.

The node work directory (2.6 GB) was deleted after the runs.

## Second GPU round (rebuilt from the same tarball; `probes/audit2_probe.cpp`, `results/audit2_results.txt`, `inputs/icc_breakdown_A.txt`)

| Audit item | GPU 0 |
|---|---|
| 6. `potrf_vbatched` Upper is ignored | **The lower triangle is factored instead.** With "not referenced" garbage below the diagonal, it returns info = 2 for a valid Upper input; the upper residual is 13. |
| 12. `potrf_vbatched` does not zero `info_array` | A stale info of 7 means **the matrix is not touched and info stays 7** |
| 13. `getri_outofplace_batched` never writes `info_array` | The sentinel 12345 **is unchanged** after the call |
| 14. `gegqr_gpu` reports rank as info | ikind 1 and 3 with A·1e-20: **info = 0 while ‖QᵀQ − I‖ = 1**. ikind 4 is fine. |
| 7. `hgetrf_gpu` trailing GEMM over padded width | n = 1000 with ldda = 1000: **20,903 factor entries differ from the ldda = 1024 run (max diff 14.8), info = 0 in both**. n = 1024: identical. The guard saw no write past the allocation. |
| LOBPCG dead parameter check (`lobpcg_check_dead`) | n_ev = 8 > m = 6 is not rejected. The illegal size reaches `gegqr` ("parameter 2 had an illegal value"), all eigenvalues are NaN, and it returns info = −201 (SLOW), not a parameter error. It also writes a file `residualNorms` into the working directory. |
| ICC with an IC(0)-breakdown SPD matrix (6×6, λ_min 0.059, IC(0) pivot 6 = −0.042 on the CPU) | **Not reproduced.** Setup info = 0 and PCG converges in 4 iterations to 1e-15, with no visible effect. The unchecked numerical-factorization status remains a code fact. |

Not run: 10 (gesvj_batched non-convergence), 11 (getrf_m needs a matrix larger than GPU memory), 15 (hgetrf singular panel), FGMRES sticky success (all run in the third round). The node directory (1.3 GB) was deleted.

## Third GPU round (`probes/audit3_probe.cpp`, `results/audit3_results.txt`)

| Item | GPU 0 |
|---|---|
| 10. `gesvj_batched` does not report non-convergence | **Worse than predicted: with one NaN entry, info = 0 at n = 16, 64 and 256 (both paths) and all singular values NaN.** The clean companion matrix is fine. |
| 11. `getrf_m` info offset | **Worse than predicted: an exactly zero column at 100/300/600/900 gives info = 0** (LAPACK: 101/301/601/901), with and without `MAGMA_NGR_NB=256`. Root cause (main session, zgetrf_m.cpp l.28, l.400–401): the routine ends with `if (*info >= 0) magma_zgetrf_piv(…, info)`, and `magma_zgetrf_piv` begins with `*info = 0`, so any singularity info is wiped. The big-panel loop never records `iinfo` either. `magma_zgetrf` / `zgesv` (CPU interface) dispatch here when the matrix does not fit in GPU memory. |
| 15. `hgetrf_gpu` singular panel | info = 1 is reported, but **all 589,824 trailing entries of dA equal Aᵀ**: the in-place transpose is not undone. The docs say "factorization has been completed". It also prints "error sgetrf inside xhsgetrf voici info 1". |
| FGMRES sticky success (`fgmres_sticky_success`) | Scan of 1152 solves (n ∈ {4, 8, 16, 32}, ‖b‖ ∈ 1e-8…1e-14, rtol ∈ {1e-20, 1e-30}, restart ∈ {2, 3, 5, 10}, 3 matrix kinds × 3 seeds): **252 report info = 0 with x non-finite (226) or a true residual above 1e-6·‖b‖.** |

Node directory (1.3 GB) deleted; GPU 0 idle afterwards.

## Sparse sweep: all 32 linear solver classes × 6 inputs (`probes/sweep_probe.cpp`, `results/sweep_results.txt`, `results/sweep_idr_restart.txt`)

Setup:
- Each solver runs in its own process through `magma_z_solver`, with maxiter 1000, restart 30, atol 1e-16, rtol 1e-10, and no preconditioner.
- Inputs (all n = 64): a real SPD tridiagonal system (control); a complex HPD system (cond 10); a complex non-Hermitian system; a tiny right-hand side b ~ 1e-20 with random x0; b[0] = NaN; and an exact x0, where an integer diagonal system gives residual 0.
- **SILENT** = info 0 although x is non-finite or the true relative residual exceeds 1e-6. **false-failure** = an error status although the answer is correct.

| Input | Result |
|---|---|
| b[0] = NaN | **SILENT for exactly BICG, PBICG, PTFQMR, PTFQMRMERGE.** This is the group that `MagmaSparse.overwrite_group_size` derived from the post-loop classification of the sources. All other 28 solvers report failure. |
| tiny b, random x0 | **SILENT for 21 of 32**: info 0 with a wrong answer; 5 of them (BICGSTAB, BICGSTABMERGE, PBICGSTABMERGE, CG, CGMERGE) return x0 with numiter 0, the other 16 iterate. (Corrected from "23" after an independent recount.) Correct: GMRES, PGMRES, JACOBI. Reported failure: the QMR family, TFQMR(MERGE), IDRMERGE, PIDRMERGE. |
| exact x0 | BICGSTAB destroys x0 (0/0 → NaN, −202). **false-failure**: BICGSTABMERGE and PBICGSTABMERGE (−202), CG (−203 NONSPD, `cgres_exact_x0_nonspd`), JACOBI (−202). |
| complex HPD / non-Hermitian | CGMERGE and PCGMERGE fail (−205; unconjugated dots). The other merge variants are fine. QMR (non-merge) returns a correct answer with −201 on both (false failure). TFQMR family: −201 on non-Hermitian. |
| **real SPD control** | **IDRMERGE and PIDRMERGE fail on every input, including this one**: −202 with NaN (restart 30 or 4), or stagnation at relative residual 0.297 after 128 = 2n iterations with restart 50 (s = 1; this shows the maxiter cap of `idr_params` for the two MERGE variants only — IDR/PIDR converged before reaching it, so for those the cap rests on code reading). IDR/PIDR with the same parameters converge in 4–10 iterations. |

## Fifth GPU round (`probes/round5_probe.cpp`, `results/round5_results.txt`, `results/round5_ilu_bt.txt`, `results/round5_ilu_pcg.txt`)

| Item | GPU 0 |
|---|---|
| bpcg multi-RHS through `magma_z_solver` (b is 64×2, solver CG) | **column 0 relerr 8.5e-11, column 1 relerr 1 (never solved), info 0**. The code computes `num_vecs = b.num_rows / A.num_rows = 1` (zbpcg.cpp l.74), yet the wrapper routes to zbpcg precisely when `b.num_cols > 1`. |
| zgesv_rbt_batched | An exactly singular matrix (zero column) gets **return 0, info_array 0, relative error 0.85**: the butterfly mixes the zero column, so the non-pivoted LU sees no exact zero. **Leak confirmed**: over 200 calls (n = 4096), device free memory dropped 25.17 MB (0.126 MB/call ≈ dv = 0.131 MB) and host RSS grew 52.7 MB (0.26 MB/call = hu + hv). |
| bombard / bombard_merge | **Not reproduced**: iter_res = final_res = true residual. The copy-paste only matters when TFQMR wins, and here BiCGSTAB or CGS won. |
| single precision | c gbtrs_batched_strided → NaN, return 0; c trtri_gpu A(300,300) = 0 → info 45; s potf2_gpu A(0,0) = −1 → info 0; c cg_merge diag(2,3), b = (2,i) → info 0 after 1 iteration, x = (2, i). **All match the d/z results.** |
| ILU crash location | Backtrace via an in-process SIGSEGV handler on an alternate stack, needed because it is a stack overflow: **infinite recursion in `magma_zmtransposeconjugate`** under `magma_zcumilusetup_transpose`. The chain: ILU setup detects the stored-zero pivot and fails (with outer PCG: setup returns −3009, `L.memory_location = 0`, nnz 0). With outer PBICG, `magma_z_precondsetup` overwrites that failure with the transpose setup (`precond_setup_info_lost`). The transpose is called on the un-built L; location 0 matches neither the CPU (571) nor the DEV (572) branch, so it converts to CSR and calls itself forever. Lean: `MagmaSparse.transpose_unbuilt_diverges`. |

Node directory deleted; GPU 0 idle.

## Sixth and seventh GPU rounds: the second audit (`probes/round6_probe.cpp`, `results/round6_results.txt`, `probes/round7_probe.cpp`, `results/round7_results.txt`)

MAGMA 2.10.0 was rebuilt from the same tarball (reference LAPACK, `scripts/build_magma.sh`). Each mode runs in its own process, because several of them fault. The full table is in `operators/magma/COVERAGE_zh.md` §3b.

| Item | GPU 0 |
|---|---|
| `magmablas_slag2h`, one entry 1e5 | **illegal memory access, info = 0**: the kernel receives `&magma_flag` taken in host code |
| `magmablas_slag2h`, NaN / `dlag2s`, NaN | info = 0 (Inf is flagged by dlag2s) |
| `magmablas_slag2h_batched`, 2 of 3 matrices overflow | illegal memory access, info_array = 0 0 0 |
| `shpotrf_gpu`/`_native`, SPD tridiag × 2^40, n = 2048 | info = 1025, 917,760 non-finite entries; `spotrf_gpu` info 0. k ≤ 30 correct. |
| `shpotrf_gpu`, workspace allocation failure | no crash: that path is compiled only for HIP; info = 1 for the zero matrix |
| `dshposv_gpu` n = 1024 × 2^40 | fine (it scales the matrix first) |
| `magma_dstedx`, d[3] = NaN, split at e[9] = 0 | **info = 0, 10 NaN eigenvalues**; LAPACK dsteqr on that block: info 9 |
| `magma_dgels_gpu`, column 0 = 0, n = 256 | **info = 0**, x finite, ‖b−Ax‖/‖b‖ = 95, ‖Aᵀr‖∞ = 848; LAPACK info = 1 |
| `magma_dgeqp3_gpu`, jpvt[0] = 1 | **info = 0, column 0 returned unchanged**; jpvt[1] = 1: **segfault**; documented LWORK: −8 |
| `magma_dsygvdx`, n = 50, RangeI il = iu = 50 (and RangeV) | **m = 50**, w[0] = λ_min |
| `magma_zmscale(UNITDIAG)`, [[−4,1],[1,−4]] and [[2i,1],[1,2i]] | return 0, every entry NaN |
| `magma_z_spmv`: A 4×4, x 3×1 / DENSE / CSC / SELL-P 300×1 | return 0 with: y untouched / y = Aᵀx = (1,5) / y = (1,2) / 44 rows unwritten |
| complex ICC + PCG, Hermitian tridiag(−i,4,i) | −202 after 1000 iterations (CG alone: 17; real tridiag with ICC: 1) |
| ILU(1), arrowhead n = 200 (symbolic storage overflow) | not silent: setup −3009 |
| sync-free ILU (CUDA ≥ 11) | setup 0, PBiCGSTAB −202; `L.drow` holds the 127 row indices (0 1 1 2 2 3 …) in an array of 65, `L.dcol` the column pointers. Under `guard_malloc.so` the corruption lands elsewhere and the solver reports info 0 at relative residual 1, so the guard run is not used as evidence. |

Node directory deleted; GPU 0 idle.

## Eighth GPU round: the remaining code-reading findings (`probes/round8_probe.cpp`, `results/round8_results.txt`)

Run with `MALLOC_CHECK_=3`, one process per mode. The full table is in `operators/magma/COVERAGE_zh.md` §3b.

| Item | GPU 0 |
|---|---|
| `magma_z_spmv` with A, x, y on the CPU | return 0; y does not receive A·x (it reads reused freed memory) |
| CSR → CSRD, row without a diagonal | return 0, then `malloc(): invalid size (unsorted)` abort (heap overflow); ELLD stores column −1 |
| `zmdiff` | I₂ vs {(0,1)} ⇒ 0; 1+5i vs 1−5i ⇒ 0 |
| `zmfrobenius` | *norm = 5 on entry ⇒ √5 for A vs A; 2×2 vs 3×3 ⇒ return 0, norm untouched |
| `zmdiagadd(+100)` | the row without a stored diagonal is not shifted |
| `zmtranspose`, 1×3 CPU CSR | result 1×3 |
| `.mtx` with row index 9 in a 3×3 file | segfault |
| `applyprecond` ILU / ICC; `applyprecond_right(ConjTrans)` | return 0, x untouched |
| `dgeqrs3_gpu`, bad `lddb` | info −8 (argument 9; `dgeqrs_gpu` says −9) |
| `zhegvx`, B = −I, n = 512, 20 calls | info 513 each time; device free memory −423.6 MB |
| CPU `magma_dgels`, m = n = 10, queried LWORK = 8192 (nb = 64) | nrhs = 4096 fine; nrhs = 16384: "DORMQR parameter number 12 had an illegal value" and the process ends inside the call |
| `magma_zgels_batched` | declared in `magma_zbatched.h`, 0 exported symbols |

Node directory deleted; GPU 0 idle.

## Ninth GPU round: multi-GPU drivers with ngpu = 1 and the last unaudited files (`probes/round9_probe.cpp`, `results/round9_results.txt`)

| Item | GPU 0 |
|---|---|
| Range V, A = 1e-150·diag(1..1000), 100 eigenvalues in (0, 1.005e-148] | zheevdx, zheevdx_m, zheevdx_2stage, zheevdx_2stage_m: **m = 10, info 0** (n = 200: 1-stage m = 2; 2-stage falls back to LAPACK and is right) |
| NaN pair in a Hermitian n = 1000 matrix | each of the four drivers: "DLASCL parameter number 4 had an illegal value" and the process ends; LAPACK zheevd returns info 0 |
| `zunmtr_m(ngpu=1, Right, Lower, 10×200)` | info 0, C unchanged (2000/2000); single-GPU zunmtr multiplies |
| `zhetrd_he2hb` n = 256, nb = 32 | lwork 1056: info 0, band differs by 2.5e119; lwork 100: `corrupted size vs. prev_size` abort |
| `zgeqrf_m(ngpu = 0)` | SIGFPE |
| `zcgeqrsv_gpu`, column 0 = 0 / A = [0] | iter 0, info 0, non-finite x; LAPACK zgels info 1 |
| `zgglse`, B = [0 0] | info 0, x = (−inf, nan), (inf, nan); LAPACK info 1 |
| `zggrqf` m = p = n = 200, lwork = 200 | info −7 after all of A and B were modified |
| `zmscale_generate(UNITCOL, Left)` | `free(): double free detected` |
| `zpotrf_mgpu_right` | declared, 0 exported symbols |

Node directory deleted; GPU 0 idle.

## Tenth GPU round: allocation-failure fault injection (`shims/fail_malloc.c`, `probes/round10_probe.cpp`, `scripts/run_round10.sh`, `results/round10_results.txt`, `results/round10_backtraces.txt`)

`fail_malloc.so` (LD_PRELOAD) fails the k-th `cudaMalloc`/`cudaHostAlloc`/`cudaMallocHost` after the probe arms it; `FI_BT=1` prints the backtrace of the failing call. For each driver the sweep counts the allocations (k = 0), then fails each in turn and compares with LAPACK.

| driver (allocations) | ok | reported | **silent wrong** | crash / exit |
|---|---|---|---|---|
| zgesv (4) | 3 | 1 | 0 | 1 segfault (zgetrf_gpu pinned) |
| zposv (3) | 3 | 0 | 0 | 1 segfault (zpotrf_gpu pinned) |
| zheevd (3) | 2 | 1 | 0 | 1 dlascl exit (zhetrd) |
| zheevd_m (6) | 3 | 1 | 0 | 3 dlascl exit |
| zhegvd (7) | 2 | 3 | **1** (zhegst_gpu, err 599) | 1 segfault, 1 dlascl exit |
| zhegvd_m (9) | 4 | 2 | **1** (zhegst_m, err 599) | 3 dlascl exit |
| zheevdx_2stage (7) | 3 | 4 | **1** (zhetrd_he2hb, err 0.498) | 0 |
| zheevdx_2stage_m (8) | 3 | 5 | **1** (dlaex0_m in zstedx_m, err 1) | 0 |
| zgeev (3) | 2 | 1 | **1** (zgehrd, err 0.067) | 0 |
| zgeev_m (2) | 2 | 0 | **1** (zgehrd_m, err 0.067) | 0 |
| zgesvd (3) | 3 | 1 | 0 | 0 |

(The counts per row add up to allocations + 1, because k = 0 is the unfailed run.)

- `zgeqrf_ooc`, m = 20000, n = 4096, GPU memory filled to 298 MB free (NB = 640 < n, so out of core): lwork = n ⇒ segfault. With 19 MB free (NB = 0) it returns −113 and does not hang.
- ngpu = 2 with only GPU 0 visible: `MAGMA_NUM_GPUS` is clamped to 1; `zpotrf_m(2, …)` and `zheevd_m(2, …)` give correct results with info 0, and leave "invalid device ordinal" as the sticky CUDA error.

Node directory deleted; GPU 0 idle.

## Eleventh and twelfth GPU rounds (`round11_*`, `round12_*`)

**Round 11, helpers under fault injection.** `zunmtr`, `zunmbr` and `zunghr` each make one device allocation, inside zunmqr or zungqr. When it fails, all three return info 0. zunmtr and zunmbr return C unchanged, and zunghr returns a wrong Q.

**Round 12, status logic in the magmablas kernels:**

| Item | GPU 0 | LAPACK |
|---|---|---|
| `zgetrf_vbatched` 64×64, A(40,40) = 0 | info 0 (A(10,10) = 0: 10) | 40 |
| `zgbtrf_batched` n = 200, kl = ku = 64, A(5,5) = 0 / only A(200,200) = 0 | 0 / 1 | 5 / 200 |
| `zgetf2_gpu` 8×8, A(3,3) = 0 | info 0, 30 non-finite entries | 3 |
| `zpotf2_gpu`, `zpotrf_native(Upper)`, A(3,3) = 0 or NaN | info 0 | 3 |
| `zpotrf_batched`, A(10,10) = NaN, n = 12 / 16 | 0 / 10 | 10 |
| `zgetrf_native` m = 16384, n = 64, zero column 3 (m = 4096: 3) | info 0 | 3 |
| `zgbtrf_batched` m = 6, n = 4 / m = 4, n = 6 | info 4 / info 0, ipiv 5 > m | 0 / 4 |
| `zheevj_batched_small_sm`, NaN pair | info 0; 0 of 8 eigenvalues non-finite (NaN silently absorbed) | — |
| `zgetrf_nopiv_expert_vbatched`, one pivot below tol | info 0 (count not incremented) | (documented 1) |
| `zgesv_batched` n = 8, singular | info 3, but B has 3 non-finite entries | B untouched |

Node directory deleted; GPU 0 idle.

## Thirteenth GPU round (`probes/round13_probe.cpp`, `results/round13_results.txt`)

| Item | GPU 0 | LAPACK |
|---|---|---|
| `zgetrf_expert_gpu_work(MagmaNative, nb 128, recnb 32 / 64)`, 128×128 I, A(50,50) = 0 | info 50 / **18** | 50 |
| `zgetrf_native` 512×32, zero columns 3 and 20, 300 runs | info 3 every time (race not reproduced) | 3 |
| `zgetf2_fused_batched` 4×8 random (8×8 control: 0) | info **5** | 0 |
| `zgetrf_batched`, column (0, NaN, 0, …, 5), 4×4 / 16×8 / 64×8 / 512×8 | 0 / **1** / **1** / **1** | 0 |
| `magmablas_zlascl` bad n / bad ldda | −3 / −7 | −7 / −9 |
| `zmscale_matrix_rhs(UNITDIAG)`, [[0,1],[1,2]], b = (1,1) | ret −100, b = (inf, 0.5), A unchanged | — |
| `zmtransfer` of a CSRLIST-tagged matrix, CPU → DEV | ret 0, empty result | — |

Node directory deleted; GPU 0 idle.

## Fourteenth GPU round (`probes/round14_probe.cpp`, `scripts/run_round14.sh`, `results/round14_results.txt`)

| Item | GPU 0 |
|---|---|
| `zlascl2`, D placed to end exactly at an unmapped page (CUDA VMM), m = 128 / 100 | no fault (the load is apparently moved inside the bounds check) |
| `zpotf2_gpu` n = 128, 1000 runs | all identical and correct |
| `zlag2c`, two host threads × two queues × 2000 calls | **the overflowing matrix reported 0 in 1889 calls; the clean one reported 1 in 7** |
| `zheevj` expert, 64 × 16×16, max_sweeps 0 / 1 / 100 × 200 | **info 0 with all wrong** / info ≠ 0 for all / 0 failures in 12,800 |
| `zmtranspose`, 1×1000 CPU CSR | **segfault** |
| `magma_z_csr_mtx`, header 3 entries, 2 present | **ret 0, nnz 3, entries (0,1) (1,1) (1,1)** |
| zunmqr, zunmql, zunmlq, zungqr2 under fault injection | report −113 / −112 |
| ztrevc3, ztrevc3_mt | no device allocation |

Node directory deleted; GPU 0 idle.

## Fifteenth GPU round: the sparse control/blas per-file sweep (`probes/round15_probe.cpp`, `results/round15_results.txt`)

| Item | GPU 0 |
|---|---|
| `zgecsrmv(MagmaTrans)`, A = [[0,1],[0,0]], x = (1,0) | y = (0,0); Aᵀx = (0,1) |
| ELLPACKT, A = [[1,1],[0,1]], x = (Inf,1) | y = (NaN, 1): row 1 (padded) is right; the NaN is complex Inf arithmetic — not a defect |
| `zmdotc`, v = r = (i) | output buffer not written (inconclusive); confirmed in round 16 |
| `zgemvmdot_shfl`, n = 64, k = 2, all ones | (1, 1) instead of (64, 64) |
| `zmgecsrmv`, m = 2, n = 3, 2 vectors | 1 2 7 4 instead of 1 2 4 5 |
| SELL-P 4×4 identity, 2 vectors | second column not computed |
| `zvread`, two-column file | values (1, 0), then `free(): corrupted unsorted chunks` |
| `zcsrget` on a device CSR | ret 0; val[0] reads 4.9e-310 (freed) |
| `zcompact`, norm NaN | active 0, cBlock 0 |
| `zmcsrcompressor_gpu` on CPU CSR | ret 0, 0×0 |
| `zfrobenius([i], [0])` | 0 |
| `zmdiagdom`, empty row | avg NaN, ret 0 |
| `zsort({NaN, NaN})` | does not return (120 s timeout) |

Node directory deleted; GPU 0 idle.

## Sixteenth GPU round (`probes/round16_probe.cpp`, `results/round16_results.txt`)

| Item | GPU 0 |
|---|---|
| `zmdotc` (device output), v = r = (i), k = 1 / 2 | −1 / −1, −1 (unconjugated; exact 1) |
| `zcsr_sort`, row {(1,10), (0,20)} | A(0,0) = 10, A(0,1) = 20 (values not moved) |
| `zcsrcoo_transpose`, 1×3 | result 1×3 |
| `zmbdiagdom`, blocksizes [3] on 2×2 | ret 0, no visible effect |
| `zbitonic_sort`, length 6 | error printed twice, ret 0, result unsorted |
| `zvget_dev` of a CPU vector | ret 0; returned pointer invalid (cudaMemcpy: invalid argument) |
| `zcuspaxpy` | correct (buffer size 0) |
| `zcuspmm` device / CPU 1×1 | −3003 / segfault |
| `zcsr_sort_gpu` on CPU CSR | memory_location CPU, val now a device pointer |
| `zge3pt` m = 1, sentinel 1000 before x | y = 998 (reads dx[−1]) |
| ELLRT SpMV, alignment 32 × blocksize 64 | "invalid configuration argument", y unchanged, ret 0 |
| `zjaccard_weights`, 2×2 all ones | off-diagonal 1/3 (exact 1) |
| `zlobpcg_shift`, 2 rows × 3 vectors | row 1 not shifted |

Node directory deleted; GPU 0 idle.

## Seventeenth GPU round (`probes/round17_probe.cpp`, `scripts/run_round17.sh`, `results/round17_results.txt`)

The probe declares the non-public routines itself and calls their exported symbols.

| Item | GPU 0 |
|---|---|
| `magma_zgecscsyncfreetrsm_analysis/_solve`, correct CSC arrays, lower bidiagonal, b = 1, n = 2 / 4096 | x = (1, 2) / 4095 of 4096 entries wrong (x = 1, 2, 2, 2, …); exact 1, 0, 1, 0, …; return 0 |
| `magma_zcsrget_gpu` of a CPU CSR | ret 0; cudaMemcpy from the returned val: invalid argument |
| `magma_zorderstatistics` {1, NaN} / {NaN, 1} | ret 0, element NaN / MAGMA_ERR_NAN |
| `magma_zrowentries`, DENSE row (i, 1) | max_nnz_row 1 |
| `magma_ztrisolve`, L = [[0,0],[1,1]], b = (1,1) | x = (1, 0), ret 0 (no solution exists) |
| zungqr, zungqr_gpu, zunmrq, zunmqr2_gpu, zunmql2_gpu, zgerbt_gpu under fault injection | all report −113 |

Node directory deleted; GPU 0 idle.

## Eighteenth GPU round (`probes/round18_probe.cpp`, `scripts/run_round18.sh`, `results/round18_results.txt`)

| Item | GPU 0 |
|---|---|
| `magma_zunglq` called as `magma_z.h` declares it (dT = device buffer, nb) | segfault |
| `magma_zunglq` called as defined (host work, lwork), fault injection on its one allocation | info −113 (reported) |

Node directory deleted; GPU 0 idle.

## Nineteenth GPU round: items from the independent review (`probes/round19_probe.cpp`, `results/round19_results.txt`)

| Item | GPU 0 |
|---|---|
| `zungbr` (Q / P) under fault injection (allocation in zungqr2 / zunglq) | **info 0, relative error 52 / 8.6** |
| `zlascl_2x2` ldda = 2 < m = 4 / lddw = 1 | info −4 (ldda is argument 6) / info 0 |
| `zresidual`, A = I₄, b 4×2 | res = (2, −1); expected (2, 4) |
| `zresidual`, A 0×0 | ret 0; not a valid test (b also had 0 rows); round 23: SIGFPE |
| `zcsrsplit`, unsorted last row | ret 0; not a valid test (bsize 1 cannot reach the path); round 23: heap corruption |

## Twentieth round: non-z findings (`probes/round20_probe.cpp`, `scripts/run_round20.sh`, `scripts/wait_and_run20.sh`, `results/round20_results.txt`)

The first build failed (a conflicting declaration and a missing `dsytri` prototype; both fixed). GPU 0 was then occupied by another job of the user, so the run was queued by `scripts/wait_and_run20.sh` until GPU 0 had no process and < 500 MiB used for 3 consecutive minutes.

| Item | GPU 0 |
|---|---|
| #1 zheevdx / zheevdx_2stage, n = 300 random, range V (0.5, 30] | m = 145 (= LAPACK) but max\|AV − VΛ\|/max\|A\| = 13.5, max\|VᴴV − I\| = 296, info 0 |
| #7 zheevdx diag(1..300), (10, 20] | m = 11 (LAPACK 10) |
| #3 native LU n = 2400: default nb / expert nb = 600, recnb = 32 | correct / 1200 invalid pivots, max\|PA − LU\|/max\|A\| = 52.7, info 0 |
| #4 gesvj_batched, column 0 = 0, n = 32, 64, 128, 256 (all blocked) / A = 0 | U has n / all non-finite entries, info 0 |
| #9 gesvj_batched entries ~1e160 / ~1e-170 | S = Inf / U non-finite, info 0 |
| #5 dsidi inverse after LAPACK dsytrf, Upper / Lower | 0.318 / 0.576 relative difference: inconclusive (dsidi expects the LINPACK factor format); superseded by round 21 |

Node directory deleted; GPU 0 idle.

## Twenty-first GPU round (`shims/fail_dsteqr.c`, `probes/round21_probe.cpp`, `scripts/run_round21.sh`, `results/round21_results.txt`)

| Item | GPU 0 |
|---|---|
| `magma_zstedx` n = 300 (two blocks of 150), LAPACK `dsteqr_` interposed to fail at call 1 / 3 / 4 of 4 | info 226 / **226** / 11626: failures in different blocks return the same code |
| `magma_dsidi(job=111)` after LAPACK dsytrf, indefinite U / L, SPD U / L | max\|AX − I\| = 0.367 / 1.06 / 0.131 / 1.16, info 0; inertia correct |
| `magmablas_dsort_batched`, n = 4096 with one NaN | index array: 1 duplicated, 1 missing |
| `magma_zgesvj_batched`, 70,000 × 64×64 (first 65,534 = I) | info 0 for all; 47 of 47 sampled matrices beyond 65,534 wrong (relative error up to 0.49) |

Node directory deleted; GPU 0 idle.

## Twenty-second and twenty-third GPU rounds (`round22_*`, `shims/fail_malloc_cpu.c`, `round23_*`)

| Item | GPU 0 |
|---|---|
| `nm -D`: magma_zdiagcheck_cpu / magmablas_zdiinertia / magma_zmlumerge | declared but not defined / defined but not declared / defined but not declared |
| `magmablas_zdiinertia`, diag (1, −1, NaN, 2) | +/−/0 = 2/1/1 (NaN counted as zero) |
| `zmcsrcompressor` on a magma_zcsrset matrix (ownership 0) with one stored zero | ret 0, nnz 3 → 3 |
| `zmlumerge`, k-th of 5 host allocations failing (`fail_malloc_cpu.so`) | k = 4: double free (inferred: the abort falls between the k = 3 and k = 5 lines, the k = 4 label was lost in the aborted output); others −112 |
| `zresidual`, A 0×0 and b 4×1 (round 19 had b 0×1, which never reaches the division) | SIGFPE |
| `zcsrsplit`, bsize 2, unsorted last row (round 19 used bsize 1) | D contains column index 681594464, then `free(): invalid pointer` |
| `magmablas_zheinertia` with a host ipiv / with a device ipiv | output not written (the kernel reads ipiv from device memory) / +/−/0 = 2/1/1 (NaN counted as zero) |

Node directory deleted; GPU 0 idle.

## Twenty-fourth GPU round: finding #6, 32-bit sizes in the D&C eigensolver (`probes/round24_probe.cpp`, `scripts/run_round24.sh`, `results/round24_results.txt`)

`magma_zstedx_m(ngpu = 1)` keeps Z (n × n complex) in host memory; only the divide-and-conquer workspace 3n(n/2+1) doubles is on the GPU, and that size is computed in 32-bit `magma_int_t`. lrwork was passed as INT_MAX with a correctly sized host buffer.

| n | 3n(n/2+1) / as int32 | lrwmin / as int32 | GPU 0 |
|---|---|---|---|
| 4,000 | 2.4e7 / same | 3.2e7 / same | info 0; eigenvalue error 4.5e-14, orthogonality 3.9e-15 |
| 38,000 | 2.17e9 / −2,128,853,296 | 2.89e9 / −1,406,815,295 | info −12 from dstedx_m (it receives lrwork − n² = 703,483,647 against a true minimum of 1,444,152,001; the real requirement exceeds INT_MAX); the GPU workspace allocation is not reached |
| 53,600 | 4.31e9 / 14,633,504 | 5.75e9 / 1,451,167,105 | **segfault**; several 32-bit expressions overflow on the path (workspace size, rwork + n², lrwork − n², ldz·j copy indices), crash site not established |

Node directory deleted; GPU 0 idle.

## Rounds 26–27 on roxy (RTX 5070 Ti, sm_120, Windows; HANDOFF §29.41–§29.42)

The MAGMA build uses MSVC 14.44 with nvcc 13.5 from the PyPI wheels and OpenBLAS 0.3.34 (`scripts/roxy/`).

- **Round 26** (`probes/round26_probe.cpp`, `results/round26_results.txt`): library-level checks of the code-reading findings. All nine items behave as predicted.
- **Round 27** (`probes/round27_sweep.cpp`, `results/round27_sweep/`): a differential sweep of most dense operators against LAPACK, 17,327 cases.
  - Every non-OK cluster is triaged in `results/round27_sweep/TRIAGE.md`.
  - Follow-ups are in `probes/round27_focus.cpp` and `results/round27_focus_results.txt`.
  - New defects: zgeqp3 on rank-deficient matrices (A·P ≠ Q·R; root causes in `magma_zlaqps`, confirmed with the patched copies `probes/zlaqps_patched*.cpp`), magmablas_zgemm with k = 0, gemv/hemv with beta = 0, zgels on singular input, zheevdx with n = 1, and n = 0 rejected by six GPU-interface routines.
- **Where the defects are**: a statistical analysis of all findings is in `operators/magma/BUG_DISTRIBUTION_zh.md`.
- **Round 28** (`probes/round28_sweep.cpp`, `results/round28_sweep/` with `TRIAGE.md`): 5,125 structured ill-conditioned inputs. zgeqp3 fails on every rank-1, cond-1e12 and graded matrix of order ≥ 63; all pass with the patched zlaqps (`round28_fixed.exe`, `scripts/roxy/build_fix28.bat`). zgeqp3_gpu needs a device tau, although the header declares a host pointer.
- **Round 29** (`probes/round29_sparse.cpp`, `results/round29_sparse/` with `TRIAGE.md`): 942 runs of 21 sparse solvers × 8 structured 1024-order matrices × CSR/ELL/SELLP × preconditioners, each checked against the true residual. No new defect; every non-OK cluster maps to a known finding (M083, M084, M089, M103, Jacobi, absolute atol) or is an honest failure.
- **Round 30** (`probes/round30_batched_eig.cpp`, `results/round30_batched_eig/` with `TRIAGE.md`): batched Jacobi eigensolver and SVD. On Windows, the sort in `swap_scalar.cuh` swaps doubles as 32-bit `long`, which corrupts sorted eigenvalues and singular values (heevj 313/784, gesvj 1016/1568 wrong; sort_flag = 0 gives 0/64). The blocked gesvj gives NaN U for σ = 0. The kernel emulator is `scripts/heevj_emu.py`.

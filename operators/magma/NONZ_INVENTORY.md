# MAGMA 2.10.0 non-z status-writing files: inventory (read-only audit)

Root: magma_full/magma-2.10.0. "Non-z" = no `@precisions normal z` origin (hand-written d-source or precision-agnostic).
Known/excluded: dstedx block-dsteqr info erased by dlascl; dlaex0_m alloc failure lost in zheevdx_2stage_m;
gesvj/heevj NaN-as-converged; heevj max_sweeps=0.

| file | purpose | reached from z? | verdict |
|---|---|---|---|
| src/dlaex0.cpp | D&C driver (LAPACK dlaed0, ICOMPQ=2) + range at last merge | zstedx -> dstedx (zheevd/zheevdx/_gpu/_2stage, zhegvd*) | clean (info encoding = LAPACK; matches 0/1-based); minor: queue leaked on error return |
| src/dlaex0_m.cpp | multi-GPU dlaex0, allocates dwork | zstedx_m -> dstedx_m (zheevd_m, zheevdx_m, *_2stage_m) | known alloc-loss; finding #6 (32-bit size overflow; GPU round 24: n = 38,000 ⇒ −12, n = 53,600 ⇒ segfault, crash site not established); minor: dw/queues leaked and device not restored on dsteqr/dlaex1 error return |
| src/dlaex1.cpp | rank-1 merge (LAPACK dlaed1), dlamrg moved into dlaex3 | via dlaex0 | clean |
| src/dlaex1_m.cpp | multi-GPU dlaex1 | via dlaex0_m | clean |
| src/dlaex3.cpp | secular eq. + vectors (LAPACK dlaed3) + magma_dvrange/dirange subset selection | via dlaex1 | finding #1 (effect site: dvrange on scaled d); finding #7 (minor, [vl,vu] closed) ; dlaed4 failure handling = LAPACK; workspace bounds OK (n12<=n1, n23<=n2) |
| src/dlaex3_m.cpp | multi-GPU dlaex3; ngpu==1 delegates to dlaex3 | via dlaex1_m | same as dlaex3 (#1, #7); no new status logic |
| src/dstedx.cpp, src/dstedx_m.cpp (extra) | tridiagonal D&C driver called by zstedx/zstedx_m | yes | finding #1 (root cause: vl/vu not scaled by orgnrm); finding #2 (dlaex0 info not remapped to global indices); plus the known dlascl erase |
| src/dmove_eig.cpp (extra) | selects eigenvalues for range V/I after stedx | zheevdx*, zheevdx_2stage* | no status logic; #7 (closed interval) |
| src/dlaln2.cpp | 1x1/2x2 scaled solve (LAPACK dlaln2) | no (only dlaqtrsd) | clean: tables, pivots, scaling, INFO=1 perturbation identical to LAPACK |
| src/dlaqtrsd.cpp | quasi-triangular singular solve for dtrevc3 | no (dtrevc3/_mt only; ztrevc3 uses zlatrsd) | clean: matches dtrevc inner loops incl. offsets; ierr from dlaln2 ignored as in LAPACK; info only arg checks |
| src/dsidi.cpp | det/inertia/inverse from dsytrf (LINPACK dsidi) | no library caller (testing_dsytrf/zhetrf REAL only) | finding #5 (uplo=Lower inverse silently uses upper triangle; lda check is `lda<0`). GPU round 21: the inverse is wrong for Upper too (SPD, no 2×2 pivots: max|AX − I| = 0.131), so the Lower explanation does not account for all of it; the cause of the Upper failure is undetermined |
| magmablas/getrf_setup_pivinfo.cu | setup_pivinfo(_batched/_vbatched), adjust_ipiv(_batched/_vbatched), gbtrf_adjust_ju | zgetrf_gpu(Native), zgetrf_recpanel_native, zgetf2_native, zgetrf(_panel)_batched, zgetf2_batched, zgetrf(_panel)_vbatched, zgbtrf_batched | finding #3 (adjust_ipiv silently no-op for m>1024); setup_pivinfo/gbtrf ju logic clean |
| magmablas/gesvj_kernels.cuh | blocked batched one-sided Jacobi: vector update (with heevj-converged skip), ptr setup, finalize values/vectors | zgesvj_batched / zgesvj_expert_batched (blocked path, non-fused sizes) | finding #4 (1/sigma with sigma==0 -> NaN U/V, fused path guards it); finding #8 (heevj_info/nsweeps not offset per batch chunk); finding #9 (unscaled sum of squares overflow/underflow); launch error -100 returned but ignored by caller (SUSPECTED, low); the heevj-converged skip in the vector update turns F9's NaN-as-converged into a silent skip of the update |
| magmablas/sort.cu (extra) | magmablas_dsort_batched used by zgesvj_batched for S + permutation | zgesvj_batched | finding #10 (minor, NaN -> rank-sort produces non-permutation for n>2048) |
| magmablas/vbatched_check.cu (extra) | arg checkers for vbatched getrf/gemm/trsm/herk/... | z*_vbatched | clean (arg-only); minor: batchCount<0 writes m[batchCount] before the batchCount check |
| magmablas/vbatched_aux.cu (extra) | max/min helpers for vbatched | z*_vbatched | no status logic |
| magmablas/magma_templates.h (extra) | magma_getidmax reductions (pivot search) | z getf2 kernels | no status logic (NaN never wins a comparison; indices stay in range) |
| magmablas/dsiinertia.cu, ssiinertia.cu (added after the second review) | inertia from a symmetric indefinite factor (d/s only) | no library caller in src | same `diag > 0 / < 0 / else zero` classification as zdiinertia, so a NaN pivot counts as a zero eigenvalue (zdiinertia GPU-confirmed in round 22); not run separately |
| magmablas/zgesvj_heevj_vector_updates.cu (z file, listed here because it consumes heevj status) | U/V update in blocked gesvj | zgesvj_batched | skips the update when heevj reports convergence; with heevj's NaN-as-converged (F9) the skip is silent; covered by the round-20 gesvj runs |
| sparse/blas/magma_trisolve.cpp (added after the second review) | frees cuSPARSE triangular-solve descriptors | ztrisolve | no status logic |
| sparse/control/mmio.cpp (added after the second review) | NIST Matrix Market banner/size reader | magma_z_csr_mtx | returns MM_* codes correctly; the problems are in its caller (unchecked indices and fscanf results, rounds 8 and 14) |
| src/dbulge_*.cpp, dgeev*, dgesvd, dgesdd, dsyevd*, dsygvd*, dtrevc3*, dgmres/dxgesv/xshgetrf/xhsgetrf/shpotrf, magmablas/h*.cu | d/s/mixed-only drivers and kernels | no (z has its own generated copies) | out of scope (not called from z) |

## Findings index
Note (second review): `src/dlaex3.cpp` uses an OpenMP loop whose last writer decides a shared value; it was not run, so no effect has been observed (the last writer of `*info = iinfo` wins).

1. dstedx(_m) range V: d/e scaled by 1/orgnrm, vl/vu not scaled -> dlaex3 dvrange selects wrong eigenvector columns; dmove_eig then returns uncomputed columns. CONFIRMED-BY-READING; GPU ✔ round 20.
2. dstedx(_m) alleig split path: dlaex0 info returned unmapped (sub-block-relative, base m+1) vs LAPACK dstedc remap. CONFIRMED-BY-READING; GPU-confirmed in round 21 (info 226 for failures in two different blocks).
3. adjust_ipiv silent no-op when m>1024 (batched/vbatched: >MAX_NTHREADS). CONFIRMED-BY-READING. — GPU ✔ round 20
4. gesvj finalize: 1/sigma without zero guard -> NaN vectors, info 0. CONFIRMED-BY-READING. — GPU ✔ round 20
5. dsidi Lower inverse uses upper triangle. CONFIRMED-BY-READING (not reachable from z library code). — GPU round 21: the inverse is also wrong for Upper (0.131), cause undetermined
6. dlaex0_m / zheevdx dwork size 3*n*(n/2+1) in 32-bit magma_int_t overflows (n>=37838). GPU round 24: the overflow is real (−12 at n = 38,000 from dstedx_m, which receives lrwork − n² = 703,483,647 against its minimum 1,444,152,001; segfault at n = 53,600, crash site not established).
7. dvrange/dmove_eig use [vl,vu] instead of documented (vl,vu]. CONFIRMED-BY-READING, minor. — GPU ✔ round 20
8. gesvj update driver: heevj_info/heevj_nsweeps not advanced by chunk offset i (dU/dG are). CONFIRMED-BY-READING. — GPU ✔ round 21
9. gesvj finalize_svd: plain sum of squares, no scaling -> sigma inf/0 for |a|>~1e154 or <~1e-162. CONFIRMED-BY-READING. — GPU ✔ round 20
10. sort rank kernel with NaN -> duplicate positions, non-permutation index. CONFIRMED-BY-READING, minor (downstream of known NaN issue). — GPU ✔ round 21

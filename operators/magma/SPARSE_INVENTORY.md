# MAGMA 2.10.0 sparse: z-precision status/acceptance-logic inventory (66 files)

Read-only audit of sparse/control/magma_z*.cpp (30) and sparse/blas z-files (36).
Line numbers refer to the original files. "known finding" = already reported, not re-derived.

## sparse/control (30)

| file | purpose (short) | verdict |
|---|---|---|
| magma_zcsrsplit.cpp | split CSR into block-diagonal D and remainder R | read, clean (missing diagonal is detected and returned as -1)；独立复核：未排序行越界；第十九轮的 bsize = 1 没有走到越界代码，第二十三轮 bsize = 2 ⇒ 垃圾列号与 `free(): invalid pointer`（✔，`csrsplit_unsorted_overflow`） |
| magma_zfree.cpp | zmfree / zprecondfree | read, clean (dval/val etc. are unions, so the mixed col/dcol frees alias correctly) |
| magma_zfrobenius.cpp | Frobenius norm of A-B on a shared pattern | NEW finding #1: real parts only (61-62); no location/format/row-count check; always returns SUCCESS |
| magma_zmatrix_tools.cpp | addrowindex, csrcoo_transpose, createrowptr, tril/triu, csr_sort | NEW finding #2: zcsr_sort sorts col but not val, and `omp parallel` without `for` (532-535); csrcoo_transpose has dims not swapped, row_ptr sized num_rows but indexed by col (104-154) |
| magma_zmatrixchar.cpp | zrowentries (max_nnz_row) | NEW finding #3 (low): DENSE counts only real part != 0 (67); other CPU storages return 0 with max_nnz_row unset; `magma_free` on a host buffer (82) |
| magma_zmconvert.cpp | storage format conversion | known finding: CSR->DENSE row-major; CSRD/ELLD diagonal overflow；第二次复核：csr_compressor 清理时对二级指针 valn 调 magma_free_cpu（只在分配失败时，未运行） |
| magma_zmcsrcompressor.cpp | remove stored zeros (CPU) | read, clean；第二次复核更正：矩阵不拥有数组时什么也不做并返回 0（第二十二轮 ✔，`compressor_not_owned_noop`） |
| magma_zmcsrpass.cpp | zcsrset / zcsrget (host CSR arrays in/out) | NEW finding #4: zcsrget on a non-CPU-CSR matrix returns pointers into A_CSR, which cleanup frees (149-157) |
| magma_zmcsrpass_gpu.cpp | zcsrset_gpu / zcsrget_gpu | NEW finding #5: same dangling-pointer pattern for device arrays (150-157); recursive status ignored (152) |
| magma_zmdiagdom.cpp | diagonal / block-diagonal dominance stats | NEW finding #6: zmbdiagdom start/end overflow or uninitialised when sum(blocksizes) != num_rows (172-181); zero diagonal or empty row gives Inf/NaN, NaN poisons avg; returns 0 |
| magma_zmdiff.cpp | norm of A-B | known finding: common entries only, real parts only |
| magma_zmfrobenius.cpp | Frobenius norm of A-B on pattern S | known finding: uninitialised *norm; nothing computed on row mismatch |
| magma_zmgenerator.cpp | banded / 5pt / 27pt stencil generators | read, clean (real-only nnz count at 120 is overwritten by csr_compressor; zmcsrcompressor status ignored at 340 is malloc-only) |
| magma_zmilustruct.cpp | symbolic ILU(k) structure | known finding: zsymbilu ignores zsymbolic_ilu status |
| magma_zmio.cpp | Matrix Market read/write | known finding: unchecked indices, unchecked fscanf |
| magma_zmlumerge.cpp | merge strict-L and U into one CSR | read, clean；第二次复核更正：第 4 次主机分配失败时 double free（第二十二轮 ✔，由中止位置推断，`lumerge_double_free`） |
| magma_zmscale.cpp | scaling, diagonal add, scale generate/apply | known finding: UNITDIAG/UNITROW non-finite with return 0; rhs partial update; zmdiagadd skips rows; UNITCOL double free |
| magma_zmshrink.cpp | shrink m x n (m<n) to square | read, clean |
| magma_zmslice.cpp | row-slice into local/nonlocal parts | read, clean (slice/num_slices not validated, but that is caller-contract) |
| magma_zmtransfer.cpp | host/device transfer | known finding: unhandled storage returns 0 |
| magma_zmtranspose.cpp | transpose / transpose-conjugate | known finding: transposeconjugate infinite recursion for memory_location 0 |
| magma_zmtranspose_cpu.cpp | CPU CSR transpose | known finding: dims not swapped, row_ptr heap overflow, frees caller rowidx |
| magma_zsolverinfo.cpp | solver info print / init / free | read, clean (printing; error-path frees use mismatched free only on malloc failure) |
| magma_zsort.cpp | quicksorts, order statistics, bitonic merge | NEW finding #7: zsort/zmsort hang on NaN (73-80, 152-166); zorderstatistics NaN check skips the pivot val[length-1] (380, 405), so it returns NaN with info 0; bitonic recursive statuses dropped (534-535) |
| magma_zutil_sparse.cpp | command-line option parsing | read, clean (invalid options are printed and defaulted by design) |
| magma_zvinit.cpp | dense vector init / random init | read, clean |
| magma_zvio.cpp | print / read / write dense vector | NEW finding #8: zvread's second pass recounts columns with strtok on the already tokenised buffer, so count=1 (202-203); a two-column (complex) file is misread, with a heap overflow at 233; fopen unchecked (156) |
| magma_zvpass.cpp | zvset / zvcopy (host) | read, clean；第二次复核：泄漏一个主机副本（轻微，未运行） |
| magma_zvpass_gpu.cpp | zvset_dev / zvget / zvget_dev / zvcopy_dev | NEW finding #9: zvget_dev on a CPU vector returns v_DEV.dval, then frees v_DEV (178-183) |
| magma_zvtranspose.cpp | dense vector block transpose | read, clean |

## sparse/blas (36)

| file | purpose (short) | verdict |
|---|---|---|
| magma_z_blaswrapper.cpp | magma_z_spmv / spmm / spmv_shift dispatch | known finding: spmv dim mismatch returns 0; DENSE computes A^T x; CSC swapped arrays; CPU-side frees caller arrays; SpMM descriptor shape |
| magma_zcuspaxpy.cpp | C = alpha*A + beta*B via cuSPARSE csrgeam | NEW finding #10: CUDA>=11 macro passes an uninitialised `void *buf` with no bufferSize query; geamNnz/geam status unchecked (156, 184); no dim check |
| magma_zcuspmm.cpp | C = A*B via cuSPARSE | NEW finding #11: CUDA 11.x csrgemm2 gets NULL info and an uninitialised buffer, status ignored (49-65, 160, 190); CUDA>=12 cleanup destroys uninitialised descriptors via CHECK_CUSPARSE inside cleanup (221, 315-317) |
| magma_zdiagcheck.cu | GPU zero/missing-diagonal check | read, clean；第二次复核：头文件声明的 magma_zdiagcheck_cpu 在库中不存在（第二十二轮 nm 确认，`declaration_mismatches`） |
| magma_zmatrixtools_gpu.cu | GPU init kernels, zcsr_sort_gpu | NEW finding #12: zcsr_sort_gpu has no location/format check and does not check the sort/alloc calls (194-221); SWAP(dval) is unconditional (225) |
| magma_zmconjugate.cu | in-place conjugate of values | no status logic (pure compute kernel / helper) |
| magma_zmcsrcompressor_gpu.cu | GPU zero removal | NEW finding #13: fallback branch frees the compressed dA and then transfers the freed dA back (172-180); a CPU CSR input comes back as 0x0 with return 0 |
| magma_ztrisolve.cpp | cuSPARSE trisolve analysis/solve | NEW finding #14 (low): no zero-pivot query, so Inf/NaN comes back with return 0 (143, 163) |
| zcompact.cu | LOBPCG drop converged columns | NEW finding #15: `dnorms[j] > tol` treats NaN as converged (37); writes active[n] beyond the documented size N |
| zge3pt.cu | 3-point stencil matvec | NEW finding #16 (low): m=1 reads dx[-1] (33-36) |
| zgecscsyncfreetrsm.cu | sync-free CSC triangular solve | NEW finding #17: the consumer spin-wait is commented out (76-91), so the solve is racy and wrong for any non-diagonal L and returns SUCCESS; launches use the default stream and are unchecked |
| zgecsrmv.cu | CSR SpMV / shifted SpMV | NEW finding #18: transA ignored (MagmaTrans computes A x and returns 0); beta=0 still reads y (suspected) |
| zgeellmv.cu | ELL SpMV | NEW finding #19: transA ignored (same as #18) |
| zgeellrtmv.cu | ELLRT SpMV | NEW finding #20: alignment*blocksize>1024 launch failure unchecked, returns 0 with y unchanged (258-294); transA ignored |
| zgeelltmv.cu | ELLPACKT SpMV | NEW finding #21: zero-padding guard commented out (37), so Inf/NaN in x[0] poisons padded rows; transA ignored |
| zgemvmdot.cu | multi-dot zmdotc / zgemvmdot | NEW finding #22: unconjugated `v[i]*r[i]` (47, 112); k=0 gives 0 shared memory but 256 writes (483/491, suspected) |
| zgesellcmmv.cu | SELLC/SELLP SpMV | known finding: alignment-1 grid sized by columns |
| zgesellcmv.cu | SELL-C SpMV | NEW finding #23: transA ignored (same as #18) |
| zilu.cpp | ILU/IC setup and triangular apply | known finding: CUDA>=11 csr2cscEx2 macro swaps CSC outputs (sync-free); complex ICC uses TRANSPOSE; transpose-setup statuses ignored |
| zjaccard_weights.cu | Jaccard weights | NEW finding #24: indexes J's arrays with A's row-pointer positions and uses rowidx for ju (57-58); wrong weights, return 0 |
| zjacobisetup.cu | Jacobi setup (D^-1, c = D^-1 b) | known finding: stored zero diagonal |
| zlobpcg_maxpy.cu | LOBPCG Y += X | no status logic (pure compute kernel / helper) |
| zlobpcg_residuals.cu | LOBPCG R = AX - X*evals | no status logic (pure compute kernel / helper) |
| zlobpcg_shift.cu | LOBPCG row shift | NEW finding #25: `row` is never used as an offset, so only row 0 is shifted (24-29); oversized launch prints and returns SUCCESS |
| zmdot_shfl.cu | warp-shuffle multi-dot | NEW finding #26: complex warpReduceSum discards the shuffled sum and returns `val` (63-108), so dots are about v[0]*r[0]; unconjugated for k>=2 but conjugated for k=1 |
| zmdotc.cu | merged 1-4 dot products | NEW finding #27: unconjugated `v0[i]*w0[i]` (35-36 and siblings) |
| zmergebicgstab.cu | merged BiCGSTAB kernels | no status logic (pure compute kernel / helper) |
| zmergecg.cu | merged CG kernels | known finding: unconjugated dots |
| zmergecgs.cu | merged CGS kernels | no status logic (pure compute kernel / helper) |
| zmergeidr.cu | IDR smoothing kernels | no status logic (pure compute kernel / helper) |
| zmergeqmr.cu | merged QMR kernels | no status logic (pure compute kernel / helper) |
| zmergetfqmr.cu | merged TFQMR kernels | no status logic (pure compute kernel / helper) |
| zmgecsrmv.cu | CSR SpMV, multiple vectors | NEW finding #28: y stride uses num_cols (46-47), which is OOB or stale for m != n; num_vecs>=7 exceeds 48 KB shared memory and the launch fails silently |
| zmgeellmv.cu | ELL SpMV, multiple vectors | NEW finding #29: same as #28 |
| zmgeelltmv.cu | ELLPACKT SpMV, multiple vectors | NEW finding #30: same as #28 (y stride, shared-memory limit) |
| zmgesellcmmv.cu | SELLP SpMV, multiple vectors | NEW finding #31: non-texture block has num_vecs/2 in y/z (1034-1086), so half of y is never written; alignment>=4 reads x column-major while the wrapper passes row-major; >1024 threads only printf |

## Counts

| verdict | control | blas | total |
|---|---|---|---|
| known finding | 9 | 5 | 14 |
| NEW finding | 9 | 22 | 31 |
| read, clean | 12 | 1 | 13 |
| no status logic | 0 | 8 | 8 |
| total | 30 | 36 | 66 |

Of the 13 "read, clean", 5 were later revised by the reviews: zcsrsplit, zmcsrcompressor, zmlumerge, zvpass, zdiagcheck.

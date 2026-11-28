> Historical review report. File paths and line numbers refer to the layout before the reorganisation of 2026-11-26; see `lean/OptimiKahan/README.md` for where the Lean files went and `experiments/magma_gpu/README.md` for the new `results/`, `probes/`, … directories.

# Independent review of the MAGMA 2.10.0 inventories and coverage doc (2026-11-25)

Source: `scratchpad/magma_full/magma-2.10.0` (read-only). Documents checked: `operators/magma/INVENTORY.md`, `operators/magma/SPARSE_INVENTORY.md`, `operators/magma/COVERAGE_zh.md`. Result files: `experiments/magma_gpu/`.

---

## Task 1: completeness of the listings

Method: `ls` on the filesystem, compared with `comm` against the file columns parsed out of each inventory table.

| set | filesystem | inventory | missing from inventory | listed but absent |
|---|---|---|---|---|
| `src/z*.cpp` | 202 | 202 rows | none | none |
| (c) rows vs the 41-helper table | 41 | 41 | identical sets | none |
| `magmablas/z*` files hit by `info\[ \| \*info \| info_array \| dinfo \| _flag` | 39 | 39 | identical sets | none |
| `sparse/src/z*.cpp` + `magma_z*` wrapper/custom | 38 + 5 = 43 | 43 (grouped rows expanded) | none | none |
| `sparse/control` + `sparse/blas` z files | 30 + 36 = 66 | 66 | none | none |

**Result: the z-file listings are complete and exact.** Every listed file exists.

The listings have scope gaps. These files are not z-prefixed but write status, and they sit in the z call graph or have no z twin, so the statement "s, c, d are generated from z" (coverage §4) does not cover them:
- `src`: `dlaex0.cpp`, `dlaex1.cpp`, `dlaex3.cpp` (and `_m` versions; reached by every zheevd/zstedx path), `dlaln2.cpp`, `dlaqtrsd.cpp` (trevc helpers), `dsidi.cpp`, plus the FP16/mixed routines `dxgesv_gmres_gpu.cpp`, `dshposv_gmres_gpu.cpp`, `dfgmres_spd_gpu.cpp`, `dgmres_plu_gpu.cpp`, `xhsgetrf_gpu.cpp`, `xshgetrf_gpu.cpp`, `shpotrf_gpu.cpp`. The FP16/mixed routines are partly covered in COVERAGE §1/§2/§3b and MAGMA_REPORT_fgmres_ir.md. None of the three docs mentions `dlaex1`, `dlaex3`, `dlaln2`, `dlaqtrsd` or `dsidi`. Spot read of `dlaex3.cpp:336-345`: OpenMP threads write `*info = iinfo` from dlaed4 under a critical section (last writer wins). That is benign, but it was never audited.
- `src/core_zhbtype{1,2,3}cb.cpp` and `core_zlarfy.cpp` are z-precision files that the `z*.cpp` glob misses. They write no status (0 hits), so nothing is lost.
- `magmablas`: `getrf_setup_pivinfo.cu`, `gesvj_kernels.cuh` (consumes heevj info, see Task 2), `slag2h.cu` (covered in §3b), `dsiinertia.cu`/`ssiinertia.cu`. Not inventoried.
- `sparse/blas/magma_trisolve.cpp` (28 lines) and `sparse/control/mmio.cpp` are precision-independent and not inventoried.

---

## Task 2: are the "other 126" magmablas z files really free of status output?

Method: a wider grep over all 165 `magmablas/z*` files (.cu .cpp .cuh .h) for any identifier containing `info`, `flag` or `conv`. I removed argument-check boilerplate (`info = -k`, `xerbla`, `return info`) and inspected every remaining hit (35 files).

**One status-writing file is outside the 39:**
- **`magmablas/zgetf2_nopiv_devicefunc.cuh`**: `zgetf2_nopiv_fused_device(..., int &linfo, int gbstep, ...)`.
  - l.52 `linfo++;` counts replaced small pivots.
  - l.58 `linfo = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbstep+i+1) : linfo;`
  - The caller `zgetf2_nopiv_fused_kernel_vbatched` (`zgetf2_nopiv_kernels_var.cu:285-326`) writes the value to `info_array[batchid]`.
  - The original grep misses it because the parameter is named `linfo` and passed by reference.
  - I read it and found no new defect. The kernel is launched with ntcol = 1, so the un-offset `dsx` shared array does not race. Identity padding keeps the padded diagonal entries at 1, so the F13 pattern (spurious info when m < n) does not occur here. NaN pivots are neither replaced nor reported, which matches LAPACK. Its counting is correct (`linfo++` on an int), unlike F10's `*info++`.
  - The fused-vbatched nopiv path is therefore status logic that the F10 row does not cover. It should be listed.

**Status consumers (they make decisions on another kernel's info) that are outside the 39:**
- `zgesvj_heevj_vector_updates.cu` → `gesvj_kernels.cuh:259-261`: `if(local_heevj_info == 0 && local_heevj_nsweeps == 1) return;` skips the vector update. With F9 (NaN off-diagonal treated as converged, info 0), the update is skipped silently. This adds no new root cause, but it is an acceptance decision that is not inventoried.
- `zdiinertia.cu:29-35` (public `magmablas_zdiinertia`): `if (diag > 0) ... else if (diag < 0) ... else zeig++` counts a NaN diagonal entry as a zero eigenvalue. It writes the `dneig[3]` output. Nothing in `src` calls it. SUSPECTED, low.

The remaining hits only use info as a return or dispatch code: gemm/gemv/hemm/herk/syrk/trsm/trmm vbatched checkers, `zgemv_batched(_smallsq)`, `zunm2r_batched_*` launch checks, `zgbtrs_kernels.cu` arginfo, `ztrtri_*` and `ztrsm*` arg checks. These are not status outputs.

---

## Task 3: spot-checks of "clean / no status / helper" verdicts (18 files)

| # | file | doc verdict | my verdict |
|---|---|---|---|
| 1 | magmablas/zgetrf_batched_smallsq_noshfl.cu | clean | AGREE. The izamax tie and NaN behaviour (l.80-90) matches LAPACK izamax; info is first zero pivot. |
| 2 | magmablas/zgeqrf_batched_smallsq.cu | only info = 0 | AGREE |
| 3 | magmablas/zhetrs.cu | arg checks only | AGREE (numbering -1,-2,-3,-5,-8 matches the signature) |
| 4 | magmablas/zlascl_diag.cu | arg checks only | AGREE (-1,-2,-3,-5,-7 correct) |
| 5 | **magmablas/zlascl_2x2.cu** | "只有参数检查，干净" | **DISAGREE, CONFIRMED-BY-READING** (see A) |
| 6 | magmablas/zgeqr2x_gpu-v4.cu | arg checks only | AGREE |
| 7 | magmablas/zgesvj_kernels.cu | clean | AGREE. The sum-based test (`sweep_sum == sub_batch*iters && info_sum == 0`) is equivalent to "all ones" because heevj always runs ≥ 1 sweep when max_sweeps > 0. |
| 8 | magmablas/zgetf2_nopiv_kernels.cu | status correct | AGREE (l.34 skips the panel after the first zero pivot, as the side note says) |
| 9 | src/zlatrsd.cpp | (c) arg errors only | AGREE |
| 10 | **src/zungbr.cpp** | (c) "propagates…" | **PARTIAL DISAGREE, CONFIRMED-BY-READING** (see B) |
| 11 | src/zbulge_back.cpp | (c) alloc only | AGREE. `n_gpu = ne` is forced (l.160), so only the GPU branch runs, and it writes `*info`. The hybrid branch, which would drop info, is dead code. |
| 12 | sparse/blas/magma_zdiagcheck.cu | read, clean | AGREE (minor: `magma_zdiagcheck_cpu` is declared in `magmasparse_z.h:1224` but defined nowhere; NaN diagonal passes) |
| 13 | **sparse/control/magma_zcsrsplit.cpp** | read, clean | **DISAGREE, SUSPECTED** (see C) |
| 14 | sparse/control/magma_zmcsrcompressor.cpp | read, clean | Mostly AGREE (see D, error path / minor) |
| 15 | sparse/control/magma_zmlumerge.cpp | read, clean | Mostly AGREE (see D) |
| 16 | sparse/control/magma_zvpass.cpp | read, clean | AGREE for status (`*m`, `*n` and `val` are written). Minor: in `magma_zvcopy` the host copy `v_CPU` is never freed for device vectors (leak). |
| 17 | **sparse/src/zresidual.cpp / zresidualvec.cpp** | "自身干净" | **DISAGREE, CONFIRMED-BY-READING** (see E) |
| 18 | magmablas/zgeqr2.cpp, zlaswp_batched.cu | clean / no status | AGREE (grep plus skim) |

### A. `magmablas/zlascl_2x2.cu`: wrong argument number, and lddw is never checked (CONFIRMED-BY-READING)
Signature: `(type[1], m[2], dW[3], lddw[4], dA[5], ldda[6], queue, info)`.
```
120    *info = 0;
...
126    else if ( ldda < max(1,m) )
127        *info = -4;
```
- The ldda error is reported as argument 4, but ldda is argument 6. This is the same class as F15 (zlascl).
- `lddw` is never validated, although the kernels read `W(2+ind, ·)` and `W(m+1, ·)`, so lddw must be ≥ m+2.
- Caller: `zlahef_gpu.cpp:320,608` (iinfo ignored). The arguments are internally generated, so the severity is low. The "clean" verdict is still wrong.

### B. `src/zungbr.cpp`: drops sub-call status, missing from the `helper_iinfo_dropped` list (CONFIRMED-BY-READING; effect SUSPECTED)
```
195            magma_zungqr2( m, n, k, A, lda, tau, /*work, lwork,*/ &iinfo );
215                magma_zungqr2( m-1, m-1, m-1, A(1,1), lda, tau, /*work, lwork,*/ &iinfo );
224            magma_zunglq( m, n, k, A, lda, tau, work, lwork, &iinfo );
244                magma_zunglq( n-1, n-1, n-1, A(1,1), lda, tau, work, lwork, &iinfo );
...
250    return *info;   // never updated from iinfo
```
- Round 14 shows that zungqr2 correctly returns -113 on allocation failure. zungbr then returns info 0 with A not turned into Q; on the m < k path A holds already-shifted reflectors.
- COVERAGE §3d and §4 list only zunmtr, zunmbr, zunghr and zungtr. The helper table only says zungbr "内部调用 zungqr/zunglq，其分配失败的报告见下", without stating that the status is dropped. zungbr should be added to `helper_iinfo_dropped`. It is not fault-injected.

### C. `sparse/control/magma_zcsrsplit.cpp`: diagonal-first placement assumes sorted columns (SUSPECTED)
The fill loop puts the diagonal at `D->row[k]`, entries with col > k at `nnz_diag`, and entries with col < k at `nnz_diag+1`. For a row stored unsorted, e.g. `[diag, lower]`:
- the diagonal goes to slot s;
- the lower entry goes to s+2, which is outside the row;
- slot s+1 stays uninitialised.

On the last row this writes past `D->val`/`D->col` (allocated with `nnz_diag`). With `[upper, lower, diag]` the diagonal overwrites the upper entry. The function returns 0. MAGMA does not guarantee sorted CSR (`magma_zcsr_sort` exists precisely because it isn't guaranteed). Also, the "zero on diagonal" check tests only `col == k`, not the value, and the message prints `(i,i)` instead of `(k,k)`. Not run on GPU.

### D. Error-path issues (CONFIRMED-BY-READING, low; only reachable on allocation failure)
- `magma_zmlumerge.cpp`: `magma_free_cpu(A->col); magma_free_cpu(A->val);` leaves the pointers set. If the following `CHECK(malloc)` fails, `cleanup: magma_zmfree(A)` frees them again (double free). U is not checked to have L's row count.
- `magma_zmcsrcompressor.cpp`:
  - With `A->ownership == false` it does nothing and returns 0.
  - In the CPU branch A's arrays are freed before the compressor runs, so the same dangling or double-free risk applies.
  - The helper it calls, `magma_z_csr_compressor` (in `magma_zmconvert.cpp`), has `cleanup: if (info != 0) { magma_free_cpu( valn ); ... }`. Here `valn` is `magmaDoubleComplex**`, so this frees the address of the caller's pointer field (UB) instead of `*valn`.

### E. `sparse/src/zresidual.cpp` / `zresidualvec.cpp` (CONFIRMED-BY-READING)
```
zresidual.cpp:64    magma_int_t num_vecs = b.num_rows*b.num_cols/A.num_rows;   // before any check
zresidual.cpp:68    if ( A.num_rows == b.num_rows ) {   ... *res = nrm2(dofs, r.dval)   // column 0 only
```
1. With `A.num_rows == 0`, this is an integer division by zero (SIGFPE) instead of a quick return or NOT_SUPPORTED. The same happens at l.151 (`_slice`) and in `zresidualvec.cpp:73` (`% A.num_rows`) when b has rows.
2. For an n×k block b (the natural `num_rows == A.num_rows`, `num_cols = k` layout), the first branch is taken and only `res[0]` is written; `res[1..k-1]` are left untouched. zbpcg allocates `residual[num_vecs]` (`zbpcg.cpp:110` and again at `:118`, a leak), calls `magma_zresidual` at `:232`, and prints `residual[i]` at `:268`. This overlaps the known `bpcg_col0` finding (which concerns the num_vecs computation in zbpcg), but the residual helpers have the same column-0-only defect of their own. They are not clean.

---

## Task 4: "confirmed" coverage rows vs result files (9 checked)

| doc row | file | doc says | file says | match |
|---|---|---|---|---|
| §1 GMSTRS original | fgmres24_original.txt | 24 calls, 20 violations; dhgesv k ≥ 20 all NaN, info 0 | bwd/thr > 1 or non-finite in 9 (dsgesv) + 11 (dhgesv) = 20; dhgesv k = 20..60: NaN, info 0 | ✔ |
| §1 patch v2 | fgmres24_fixed.txt | 0 violations | all 24 bwd/thr ≤ 0.689 | ✔ |
| §2 #11 getrf_m | audit3_results.txt | never reports singular | info 0 vs LAPACK 101/301/601/901, both NB settings | ✔ |
| §3 fgmres sticky | audit3_results.txt | 252 of 1152 | "scanned 1152 solves: 252 report info = 0 …" | ✔ |
| §3e F1–F11 | round12_results.txt | F1 0 vs 40; F2 0/1; F3 30; F5 0/10; F8 4 and ipiv 5; F10 0; F11 3 non-finite | identical | ✔ (F9: info 0 matches, but the file also shows 0 non-finite eigenvalues, i.e. the NaN was silently ignored. The doc does not mention this.) |
| §3b dgels_gpu | round7_results.txt | info 0, residual 95 | info 0, 95.2 | ✔ |
| §3b zhegvx leak | round8_results.txt | 423.6 MB / 20 calls | 423.62 MB, info 513 | ✔ |
| **§3 early exit / tiny b** | sweep_results.txt | **"23/32 静默返回 x0"** (README l.182 also says 23 of 32) | **21 of 32** tiny-b rows are SILENT-FAIL; only 5 of them have numiter = 0 (BICGSTAB, BICGSTABMERGE, PBICGSTABMERGE, CG, CGMERGE); the other 16 iterated (e.g. BICG 30, PTFQMR 257) and are not x0 | **MISMATCH.** The README's own breakdown (3 correct + 8 reported) gives 32 − 11 = 21. "returns x0" is accurate only for 5. |
| **§3d fault injection** | round10_results.txt | 55 injections: **7** info 0 with wrong result, 8 DLASCL, 3 segv, **37** reported or unaffected | 55 injections (57 non-baseline lines minus 2 zgeqrf_ooc lines): **6** SILENT-WRONG (zhegvd k4, zhegvd_m k2, zheevdx_2stage k2, zheevdx_2stage_m k4, zgeev k2, zgeev_m k1), 8 DLASCL, 3 rc = 139, **38** (19 reported + 19 ok) | **MISMATCH:** 7 should be 6 and 37 should be 38. The doc's own location list names only 6. |
| §3 IDR params "128 = 2n" | sweep_idr_restart.txt | ✔ | numiter = 128 appears only for IDRMERGE/PIDRMERGE (n = 64); IDR/PIDR converge in 10 iterations | Partial: the number is right, but the evidence comes from the merge variants, while the inventory attributes `idr_params` to zidr/zpidr. The same `min(2*n, maxiter)` line exists in all six IDR files, so the claim holds by reading. |

Also checked and consistent: round14 zlag2c 1889/7; heevj max_sweeps = 0, 64/64 wrong; n = 64 in sweep_probe.cpp.

---

## Summary of problems found
1. **Listings**: all z files are present, and no listed file is missing. Scope gap: real-only and FP16 status files in the z call graph (dlaex1/3, dlaln2, dlaqtrsd, dsidi, getrf_setup_pivinfo, …) are in no inventory.
2. **Status writer outside the 39**: `magmablas/zgetf2_nopiv_devicefunc.cuh` (writes `linfo`, which becomes `info_array`). I read it and found it correct. Status consumers outside the 39: `zgesvj_heevj_vector_updates.cu`/`gesvj_kernels.cuh`, and `zdiinertia.cu` (NaN counted as a zero eigenvalue).
3. **Verdict disagreements**:
   - zlascl_2x2.cu (ldda reported as −4, lddw unchecked): confirmed.
   - zungbr.cpp (drops zungqr2/zunglq iinfo; missing from `helper_iinfo_dropped`): confirmed by reading.
   - zresidual.cpp / zresidualvec.cpp (division by zero when n = 0; only column 0 for n×k b): confirmed.
   - magma_zcsrsplit.cpp (out-of-bounds write on unsorted CSR rows): suspected.
   - Minor error-path double frees in zmlumerge, zmcsrcompressor and magma_z_csr_compressor.
4. **Number mismatches**:
   - §3 "23/32 静默返回 x0" should be 21/32, and only 5 actually return x0.
   - §3d "7 次 info 0、结果错误 / 37 次" should be 6 / 38.
   - The IDR "128 = 2n" evidence comes from the MERGE variants.

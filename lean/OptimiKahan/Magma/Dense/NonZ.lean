import OptimiKahan.Magma.Dense.StatusAudit

/-!
# MAGMA non-z files on the z call graph, independent-review findings, 32-bit size arithmetic

Code-level models of MAGMA v2.10.0: each definition transcribes the control logic of the cited source lines, and the
theorems state the consequence the caller sees; the GPU 0 evidence is cited in the docstrings
(`experiments/magma_gpu/results/`).  Core Lean only.

Moved here from MagmaAudit.lean §27, MagmaAudit.lean §28, MagmaAudit.lean §29, MagmaAudit.lean §31 when the two original files were split by subject; names and the
namespace `MagmaAudit` are unchanged.
-/

namespace MagmaAudit

/-! ## 27. Independent-review findings (HANDOFF §29.20, `round19_results.txt`) -/

/-- `magmablas_zlascl_2x2` (zlascl_2x2.cu l.120–127): `ldda < max(1,m)` reports −4 (ldda is argument 6); `lddw`
    (argument 4) is not checked although W is read up to row m+1.  `(reported for bad ldda, bad lddw rejected?)` -/
def lascl2x2Check : Int × Bool := (-4, false)
/-- GPU 0: ldda = 2 < m = 4 ⇒ info −4; lddw = 1 ⇒ info 0 -/
theorem lascl2x2_args : lascl2x2Check = (-4, false) ∧ (-4 : Int) ≠ -6 := by decide

/-! ## 28. Non-z files on the z call graph (HANDOFF §29.21, `operators/MAGMA_NONZ_INVENTORY.md`, `round20_results.txt`).
    GPU 0 (round 20): #1 zheevdx / zheevdx_2stage, n = 300, range V (0.5, 30]: m = 145 (right) but
    max|AV − VΛ|/max|A| = 13.5 and max|VᴴV − I| = 296, info 0; #3 native LU nb = 600: 1200 invalid pivots,
    max|PA − LU|/max|A| = 52.7, info 0; #4 blocked gesvj with a zero column: U non-finite, info 0 (n = 32 … 256);
    #7 diag(1..300), (10, 20]: m = 11 (LAPACK 10); #9 entries ~1e160 ⇒ S = Inf, ~1e-170 ⇒ U non-finite, info 0;
    #5 inconclusive (dsidi expects the LINPACK factor format, so the LAPACK-dsytrf input was not valid) -/

/-- #1 `dstedx(_m)` range V (dstedx.cpp l.320–324): d, e are scaled by `1/orgnrm` and `dlaex0` selects with the
    unscaled `vl, vu`.  Integer model: eigenvalues `k` (k = 1..20) scaled by `1/s` are compared with the same bounds -/
def selectedScaledDown (s vl vu : Nat) : List Nat := ((List.range 20).map (· + 1)).filter (fun k => decide (vl * s < k ∧ k ≤ vu * s))
/-- **with s = 2 the selection on scaled values picks a different set than on the true ones** -/
theorem stedx_rangev_unscaled : selectedScaledDown 1 3 6 = [4, 5, 6] ∧ selectedScaledDown 2 3 6 ≠ [4, 5, 6] := by decide

/-- #2 `dstedx` (l.277–280): a sub-block's `dlaex0` info (base `m + 1`, block-relative) is returned as is; LAPACK
    `dstedc` maps it to `(info/(m+1) + start)·(n+1) + info mod (m+1) + start` (0-based start) -/
def stedcRemap (n m start info : Nat) : Nat := (info / (m + 1) + start) * (n + 1) + info % (m + 1) + start
theorem stedx_subblock_info_unmapped : stedcRemap 300 150 150 226 ≠ 226 := by decide
-- GPU 0 (round 21, LD_PRELOAD shim forcing one LAPACK dsteqr_ call to fail inside magma_zstedx, n = 300 split 150 + 150):
-- a failure in the first leaf of block 1 and in the first leaf of block 2 both return info = 226 — the block offset is lost.

/-- the reported info as a function of the block a failure occurs in: independent of `start` -/
def stedxReported (_start blockInfo : Nat) : Nat := blockInfo
theorem stedx_blocks_indistinguishable : stedxReported 0 226 = stedxReported 150 226 := rfl

/-- #3 `adjust_ipiv` (getrf_setup_pivinfo.cu l.264): `if (m > 1024) { fprintf(…); return; }` — the pivots of the second
    half stay panel-relative.  Global pivot for a local pivot `p` at offset `off`: -/
def adjustIpiv (m off p : Nat) : Nat := if m > 1024 then p else p + off
theorem adjust_ipiv_noop_large : adjustIpiv 1200 1200 3 = 3 ∧ adjustIpiv 1000 1000 3 = 1003 := by decide

/-- #4 blocked `gesvj` (gesvj_kernels.cuh l.660, l.764): `rsigma = 1/sigma` without the zero guard of the fused path
    (zgesvj_batched_small_sm.cu l.312: `s == 0 ? 1 : 1/s`).  `none` = Inf; `0 · Inf = NaN` gives NaN in U or V. -/
def recipBlocked : Nat → Option Nat | 0 => none | s => some s      -- reciprocal defined (value irrelevant) iff s ≠ 0
def recipFused : Nat → Option Nat | 0 => some 1 | s => some s
theorem gesvj_blocked_zero_sigma : recipBlocked 0 = none ∧ recipFused 0 = some 1 := ⟨rfl, rfl⟩

/-- #5 `magma_dsidi` (dsidi.cpp l.133, l.224): uplo = Lower is accepted but the inverse branch reads `A(k, k+1)` and
    `A(j, k)` with `j ≤ k`, i.e. the upper triangle.  GPU round 21 found the inverse wrong for Upper too (SPD, no 2×2
    pivots: max|AX − I| = 0.131), which this reading does not explain; that cause is undetermined (`dsidi_inverse_wrong`). -/
def dsidiReadsUpper (_lower : Bool) : Bool := true
theorem dsidi_lower_reads_upper : dsidiReadsUpper true = true := rfl

/-- #7 range V interval (dlaex3.cpp l.38, dmove_eig.cpp l.40): only `w < vl` is excluded, so `[vl, vu]` instead of the
    documented `(vl, vu]` -/
def inRangeMagma (vl vu w : Nat) : Bool := decide (vl ≤ w ∧ w ≤ vu)
def inRangeLapack (vl vu w : Nat) : Bool := decide (vl < w ∧ w ≤ vu)
theorem rangev_closed_interval : inRangeMagma 10 20 10 = true ∧ inRangeLapack 10 20 10 = false := by decide

/-! ## 29. Round 21 (HANDOFF §29.23, `round21_results.txt`) -/

/-- #8 `gesvj_kernels.cuh` l.314: the U/G pointer arrays are advanced by the chunk offset `i`, `heevj_info` /
    `heevj_nsweeps` are not; the convergence flags a chunk reads belong to problem `j` of chunk 0, not `i + j` -/
def flagIndexRead (_chunkOffset j : Nat) : Nat := j
def flagIndexNeeded (chunkOffset j : Nat) : Nat := chunkOffset + j
/-- **every problem past the first chunk reads another problem's flags** (GPU 0: 70,000 matrices of 64×64, matrices
    0..65533 = I, the rest random: 47 of 47 sampled matrices beyond 65,534 have wrong singular values, relative error up
    to 0.49, all info 0) -/
theorem gesvj_chunk_flags_not_offset : flagIndexRead 65534 5 ≠ flagIndexNeeded 65534 5 := by decide

/-- #5 `magma_dsidi` with the documented input (LAPACK dsytrf output): GPU 0 measured max|A·X − I| = 0.367 / 1.06
    (indefinite, Upper / Lower) and 0.131 / 1.16 (SPD, no 2×2 pivots), all with info 0; inertia correct.  Recorded as
    the observation `(info, inverse correct)`: -/
def dsidiInverse : Int × Bool := (0, false)
theorem dsidi_inverse_wrong : dsidiInverse = (0, false) := rfl

/-- #10 `magmablas/sort.cu` rank sort: the rank of x is `#{y : y > x}` (descending).  With IEEE semantics a NaN is
    greater than nothing and nothing is greater than it, so it gets rank 0, the same rank as the maximum.  `none` = NaN. -/
def gtN : Option Nat → Option Nat → Bool | some a, some b => decide (a > b) | _, _ => false
def rank (xs : List (Option Nat)) (x : Option Nat) : Nat := (xs.filter (fun y => gtN y x)).length
/-- **NaN and the maximum share rank 0: the index array is not a permutation** (GPU 0: n = 4096, one NaN: one index
    duplicated, one missing) -/
theorem sort_nan_rank_collision : rank [some 3, none, some 1] (some 3) = 0 ∧ rank [some 3, none, some 1] none = 0 := by decide

/-! ## 31. #6: 32-bit size arithmetic in the multi-GPU divide-and-conquer path (round 24, `round24_results.txt`) -/

/-- two's-complement wrap of a natural number to a 32-bit signed int -/
def wrap32 (v : Nat) : Int := let r := v % 4294967296; if r < 2147483648 then r else (r : Int) - 4294967296

/-- `dlaex0_m.cpp` l.151 `3*n*(n/2 + 1)` and `zstedx_m.cpp` l.181 `1 + 4*n + 2*n*n`, evaluated in 32-bit `magma_int_t` -/
def dcWork (n : Nat) : Int := wrap32 (3 * n * (n / 2 + 1))
def lrwMin (n : Nat) : Int := wrap32 (1 + 4 * n + 2 * n * n)

/-- **the 32-bit size expressions wrap** (values exact).  What GPU 0 showed: n = 38,000 ⇒ info −12 — the negative `lrwMin`
    makes zstedx_m's own check pass, and the rejection comes from `magma_dstedx_m`, which receives `lrwork − n·n` =
    703,483,647 against its true minimum 1,444,152,001; the requirement 2.89e9 exceeds INT_MAX, so no 32-bit lrwork can be
    valid (a correct, reported error; the dlaex0_m workspace allocation is not reached).  n = 53,600 ⇒ segfault; on that
    path `dcWork`, `rwork + n·n`, `lrwork − n·n` and the `ldz·j` copy indices all overflow, so which one crashes is not
    established. -/
theorem dc_sizes_wrap : dcWork 38000 = -2128853296 ∧ lrwMin 38000 = -1406815295 ∧
    dcWork 53600 = 14633504 ∧ lrwMin 53600 = 1451167105 := by decide

/-! ## 31b. The `lrwmin` of every divide-and-conquer eigensolver wraps for n ≥ 32,767 (HANDOFF §29.36) -/

/-- `lrwmin = 1 + 5*n + 2*n*n` in `zheevd(_gpu,_m)`, `zheevdx(_gpu,_m)`, `zheevdx_2stage(_m)`, `zhegvd(_m)`,
    `zhegvdx(_m)`, `zhegvdx_2stage(_m)` (e.g. zheevd.cpp l.212), evaluated in 32-bit `magma_int_t`
    (`experiments/magma_gpu/scripts/wrap32_scan.py` lists the 14 drivers plus zstedx(_m); `lwmin = 2n + n²` wraps from n = 46,340). -/
def heevdLrwmin (n : Nat) : Int := wrap32 (1 + 5 * n + 2 * (n * n))
/-- the check `lrwork < lrwmin` (zheevd.cpp l.226) -/
def heevdLrworkRejected (n : Nat) (lrwork : Int) : Bool := decide (lrwork < heevdLrwmin n)

/-- general form: for every n in [32767, 46339] the minimum is negative, so the routine's own check accepts every
    lrwork ≥ 0 and the workspace query (`rwork[0] = lrwmin`) reports a negative size, although the true requirement
    exceeds INT_MAX so that no 32-bit lrwork is large enough.  The call then proceeds through `zhetrd` (A is
    overwritten) and the rejection comes from `magma_dstedx` as −12, which in zheevd's numbering names `liwork`. -/
theorem heevd_lrwmin_wraps_all (n : Nat) (h1 : 32767 ≤ n) (h2 : n ≤ 46339) :
    heevdLrwmin n < 0 ∧ (∀ lrwork : Int, 0 ≤ lrwork → heevdLrworkRejected n lrwork = false) ∧
    2147483647 < 1 + 5 * n + 2 * (n * n) := by
  have a : 32767 * 32767 ≤ n * n := Nat.mul_le_mul h1 h1
  have b : n * n ≤ 46339 * 46339 := Nat.mul_le_mul h2 h2
  have hlt : 1 + 5 * n + 2 * (n * n) < 4294967296 := by omega
  have hneg : heevdLrwmin n < 0 := by
    unfold heevdLrwmin wrap32
    simp only [Nat.mod_eq_of_lt hlt]
    rw [if_neg (by omega)]; omega
  refine ⟨hneg, fun lrwork h => ?_, by omega⟩
  simp [heevdLrworkRejected]; omega

theorem heevd_lrwmin_instance : heevdLrwmin 32767 = -2147450882 ∧ heevdLrwmin 32766 = 2147385343 := by decide

/-! ## 31c. In-kernel 32-bit offsets: `magmablas_zlacpy` (HANDOFF §29.39) -/

/-- `zlacpy.cu` l.36–62: the kernel takes `int ldda` and computes `dA += ind + iby*ldda` in `int`, with
    `iby = blockIdx.y*64`.  The host wrapper (l.350–402) splits the matrix into super-blocks of
    `max_blocks*64 ≈ 4.19M` columns, so inside one launch `iby` reaches n − 64 for every realistic n; the product
    therefore wraps once (column index)·ldda exceeds INT_MAX, i.e. for matrices with more than 2³¹ elements — in the
    default LP64 build and equally in an ILP64 build, whose purpose is large matrices but whose kernels keep `int`
    parameters.  No documentation states the limit.  Evidence: code reading; the other 202 `int lda/ldda`
    parameters of the z kernels were not checked one by one.  `(32-bit offset, true offset)` -/
def lacpyOffset (ind iby ldda : Nat) : Int × Nat := (wrap32 (ind + iby * ldda), ind + iby * ldda)

/-- instance: m = n = ldda = 50,000 (single precision: 10 GB per matrix, fits a 32 GB V100); the last block column
    starts at iby = 49,984 and its first offset wraps to a negative value -/
theorem lacpy_offset_wraps : (lacpyOffset 0 49984 50000).1 < 0 ∧ (lacpyOffset 0 49984 50000).2 = 2499200000 := by
  decide

/-- general form: whenever the true offset lies in (INT_MAX, 2³²), the kernel addresses memory before the matrix -/
theorem lacpy_offset_negative_all (ind iby ldda : Nat) (h1 : 2147483647 < ind + iby * ldda)
    (h2 : ind + iby * ldda < 4294967296) : (lacpyOffset ind iby ldda).1 < 0 := by
  unfold lacpyOffset wrap32
  simp only [Nat.mod_eq_of_lt h2]
  rw [if_neg (by omega)]; omega

/-! ## 31d. 32-bit offsets across magmablas (audit of all 123 z kernel files, HANDOFF §29.40) -/

/-- classification of `magmablas/z*.cu` by whether a matrix-wide column offset `j*ld` is formed in 32-bit `int` inside
    a kernel (independent read-only audit, spot-checked by hand): WRAPS (89: 48 non-batched, 34 batched-only, 7 with
    `magma_int_t` products that wrap only in LP64), SAFE-64 (4: ztranspose, ztranspose_inplace, zhemv_mgpu,
    zgemv_conj — plus the main `zlaswp_kernel`), BOUNDED (6), NOT-APPLICABLE (24) -/
def offsetAudit : List (String × Nat) := [("WRAPS", 89), ("SAFE-64", 4), ("BOUNDED", 6), ("NOT-APPLICABLE", 24)]
theorem offset_audit_covers_all : (offsetAudit.map Prod.snd).sum = 123 := by decide

/-- `zpotrf_gpu.cpp` l.145: the host macro `dA(i,j) = dA + i + j*ldda` multiplies two `magma_int_t` (32-bit in LP64),
    whereas `zgetrf_gpu.cpp` l.191 casts both to `size_t`.  For ldda = n = 50,000 the pointer handed to cuBLAS for column
    j = 42,950 is computed from a wrapped product. -/
def potrfHostColOffset (j ldda : Nat) : Int := wrap32 (j * ldda)
theorem potrf_gpu_host_offset_wraps :
    potrfHostColOffset 42949 50000 = 2147450000 ∧ potrfHostColOffset 42950 50000 < 0 := by decide

/-- `magma_zgetrf_native` (zgetrf_gpu.cpp l.382, 440; `SWP_CHUNK` undefined) swaps rows of the transposed matrix with
    `magma_zlaswp_columnserial`, whose kernel indexes `dA[i1*lda + tid]` in `int` (zlaswp_batched.cu l.248–264); the
    same wrap for every pivot row i1 ≥ 42,950 when lda = 50,000 (the default hybrid `magma_zgetrf_gpu` uses the
    size_t-safe `zlaswp_kernel` instead). -/
def laswpColumnserialOffset (i1 lda tid : Nat) : Int := wrap32 (i1 * lda + tid)
theorem getrf_native_laswp_wraps : laswpColumnserialOffset 42950 50000 0 < 0 := by decide

/-- `magmablas_ztranspose_batched_stride` (ztranspose.cu l.318–328): batches are processed in chunks of
    `max_batchCount = 65534`, but each chunk is passed as `dA_array + i` (an element offset of i) instead of
    `dA_array + i*stride`; every matrix of every chunk after the first is read from and written to the wrong place.
    `(start of chunk c as passed, correct start)` -/
def transposeChunkStart (c stride : Nat) : Nat × Nat := (c * 65534, c * 65534 * stride)
theorem transpose_stride_chunk_wrong_all (c stride : Nat) (hc : 1 ≤ c) (hs : 2 ≤ stride) :
    (transposeChunkStart c stride).1 ≠ (transposeChunkStart c stride).2 := by
  unfold transposeChunkStart
  have : 0 < c * 65534 := Nat.mul_pos (by omega) (by omega)
  intro h
  have h2 : c * 65534 * 1 < c * 65534 * stride := Nat.mul_lt_mul_of_pos_left (by omega) this
  omega

/-- `magmablas_zswapblk` (zswapblk.cu l.19, l.105): pivot targets are stored in `short ipiv[]` as `im − offset`; any
    target more than 32,767 rows below the block start is truncated (two's complement to 16 bits) -/
def toShort (v : Nat) : Int := let r := v % 65536; if r < 32768 then r else (r : Int) - 65536
theorem swapblk_short_truncates : toShort 32767 = 32767 ∧ toShort 32768 = -32768 ∧ toShort 40000 = -25536 := by decide

/-! ## General forms -/

/-- **adjust_ipiv in general**: for every panel taller than 1024 rows the pivots are left panel-relative -/
theorem adjust_ipiv_noop_all (m off p : Nat) (h : 1024 < m) : adjustIpiv m off p = p := by
  simp [adjustIpiv, h]

/-- the exact wrap: for `v` below 2³² the 32-bit value is `v` or `v − 2³²` -/
theorem wrap32_cases (v : Nat) (h : v < 4294967296) : wrap32 v = v ∨ wrap32 v = (v : Int) - 4294967296 := by
  unfold wrap32; simp only [Nat.mod_eq_of_lt h]; split <;> simp

end MagmaAudit

#print axioms MagmaAudit.lascl2x2_args
#print axioms MagmaAudit.stedx_rangev_unscaled
#print axioms MagmaAudit.stedx_subblock_info_unmapped
#print axioms MagmaAudit.adjust_ipiv_noop_large
#print axioms MagmaAudit.gesvj_blocked_zero_sigma
#print axioms MagmaAudit.dsidi_lower_reads_upper
#print axioms MagmaAudit.rangev_closed_interval
#print axioms MagmaAudit.stedx_blocks_indistinguishable
#print axioms MagmaAudit.gesvj_chunk_flags_not_offset
#print axioms MagmaAudit.dsidi_inverse_wrong
#print axioms MagmaAudit.sort_nan_rank_collision
#print axioms MagmaAudit.dc_sizes_wrap
#print axioms MagmaAudit.adjust_ipiv_noop_all
#print axioms MagmaAudit.wrap32_cases
#print axioms MagmaAudit.heevd_lrwmin_wraps_all
#print axioms MagmaAudit.heevd_lrwmin_instance
#print axioms MagmaAudit.lacpy_offset_wraps
#print axioms MagmaAudit.lacpy_offset_negative_all
#print axioms MagmaAudit.offset_audit_covers_all
#print axioms MagmaAudit.potrf_gpu_host_offset_wraps
#print axioms MagmaAudit.getrf_native_laswp_wraps
#print axioms MagmaAudit.transpose_stride_chunk_wrong_all
#print axioms MagmaAudit.swapblk_short_truncates

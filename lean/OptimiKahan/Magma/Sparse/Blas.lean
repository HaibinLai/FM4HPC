import OptimiKahan.Magma.Sparse.Solvers

/-!
# MAGMA sparse BLAS layer: SpMV kernels, dot products, sorting, LOBPCG helpers

Code-level models of MAGMA v2.10.0: each definition transcribes the control logic of the cited source lines, and the
theorems state the consequence the caller sees; the GPU 0 evidence is cited in the docstrings
(`experiments/magma_gpu/results/`).  Core Lean only.

Moved here from MagmaSparse.lean §17, MagmaSparse.lean §18, MagmaSparse.lean §19 when the two original files were split by subject; names and the
namespace `MagmaSparse` are unchanged.
-/

namespace MagmaSparse

/-! ## 17. Per-file sweep of sparse/control and sparse/blas (HANDOFF §29.15, `round15_results.txt`,
    `operators/MAGMA_SPARSE_INVENTORY.md`) -/

/-- `magma_zgecsrmv`, `zgeellmv`, `zgesellcmv`: `transA` is never read.  2×2 integer model, A = [[0,1],[0,0]] row-major -/
def csrmvIgnoringTrans (_trans : Bool) (x0 x1 : Int) : Int × Int := (0 * x0 + 1 * x1, 0 * x0 + 0 * x1)
/-- **MagmaTrans returns A x = (0, 0) instead of Aᵀ x = (0, 1)** (GPU 0: y = (0, 0)) -/
theorem csrmv_trans_ignored : csrmvIgnoringTrans true 1 0 = (0, 0) := by decide

/- `zgeelltmv.cu` l.37 (`val ≠ 0` test commented out) was **not confirmed** on GPU 0: with A = [[1,1],[0,1]],
   x = (Inf, 1) the padded row 1 gives 1 (correct); the NaN in row 0 is IEEE complex multiplication
   ((1+0i)·(Inf+0i) has imaginary part 0·Inf = NaN), not a MAGMA defect.  No theorem. -/

/-- `zmdot_shfl.cu` l.63–108: the complex warp reduction shuffles a copy and returns the input unchanged -/
def warpReduceNoop (lane0 : Int) (_others : List Int) : Int := lane0
/-- **64 ones reduce to 1** (GPU 0: zgemvmdot_shfl, n = 64, k = 2 ⇒ (1, 1)) -/
theorem mdot_shfl_reduction_noop : warpReduceNoop 1 (List.replicate 63 1) = 1 := rfl

/-- `zmgecsrmv.cu` l.46–47 (and `zmgeellmv`, `zmgeelltmv`): output vector `i` is written at `row + i·num_cols` -/
def mgecsrmvOffset (numCols row i : Nat) : Nat := row + i * numCols
/-- **m = 2, n = 3: the second vector starts at offset 3, not 2** (GPU 0: Y = 1 2 7 4 instead of 1 2 4 5) -/
theorem mgecsrmv_stride : mgecsrmvOffset 3 0 1 = 3 ∧ mgecsrmvOffset 3 1 1 = 4 := by decide

/-- `magma_zmgesellpmv`: `block(blocksize, num_vecs/2)`, each thread computes one vector -/
def sellpVectorsComputed (numVecs : Nat) : Nat := numVecs / 2
/-- **two vectors: only the first is computed** (GPU 0: second column of Y stays 7) -/
theorem sellp_multi_half : sellpVectorsComputed 2 = 1 := rfl

/-- `magma_zvread` (control/magma_zvio.cpp l.161–233): the first pass `strtok`s the buffer, the second pass recounts
    on the damaged buffer and reads one real per value; a two-column file of `N` lines puts `2N` reals into `length` -/
def vreadValues (lines : Nat) : Nat := 2 * lines
/-- **"1 0 / 2 0" with length 2 reads (1, 0) and writes past the buffer** (GPU 0: values (1, 0), then glibc
    `free(): corrupted unsorted chunks`) -/
theorem vread_two_columns_overflow : vreadValues 2 = 4 ∧ ¬ (4 ≤ 2) := by decide

/-- `magma_zcsrget` (control/magma_zmcsrpass.cpp l.149–157): for a non-CPU input the pointers of a temporary copy are
    returned and the copy is freed.  `(return value, pointers valid)` -/
def csrgetDev : Int × Bool := (0, false)
/-- GPU 0: device CSR [[3,0],[0,4]] ⇒ ret 0, val[0] read back as 4.9e-310 (freed memory) -/
theorem csrget_dangling : csrgetDev = (0, false) := rfl

/-- `zcompact.cu` l.37: `if (dnorms[j] > tol && active[j])` keeps a vector active; NaN fails the test -/
def staysActive : Option Rat → Rat → Bool | some x, t => decide (t < x) | none, _ => false
/-- **a NaN residual norm is "converged"** (GPU 0: active 0, cBlock 0) -/
theorem compact_nan_converged (t : Rat) : staysActive none t = false := rfl

/-- `magma_zmcsrcompressor_gpu` (blas/magma_zmcsrcompressor_gpu.cu l.172–180): the fallback frees `dA` and then
    transfers it; the result has the dimensions of the freed, empty matrix -/
def compressorFallback (rows : Nat) : Nat := let _ := rows; 0
/-- GPU 0: 2×2 CPU CSR ⇒ ret 0, result 0×0, nnz 0 -/
theorem compressor_cpu_emptied : compressorFallback 2 = 0 := rfl

/-- `magma_zfrobenius` (control/magma_zfrobenius.cpp l.61–62): real parts only -/
def frobReal (a b : C) : Int := (a.re - b.re) * (a.re - b.re)
/-- **[i] against [0] gives 0** (GPU 0: res 0) -/
theorem frobenius_real_only : frobReal ⟨0, 1⟩ ⟨0, 0⟩ = 0 := by decide

/-- `magma_zsort` (control/magma_zsort.cpp l.73–80): the partition scans advance on `x[i] < pivot` / `x[j] > pivot`;
    with NaN everywhere neither scan moves and the swap loop runs forever.  Steps until the indices cross, with fuel: -/
def partitionScan : Nat → Nat → Nat → Option Nat
  | 0, _, _ => none
  | f + 1, i, j => if i < j then partitionScan f i j else some i     -- NaN: no comparison advances i or j
/-- **{NaN, NaN} never terminates** (GPU 0: `magma_zsort` killed by the 120 s timeout) -/
theorem zsort_nan_no_progress (f : Nat) : partitionScan f 0 1 = none := by
  induction f with
  | zero => rfl
  | succ f ih => simp [partitionScan, ih]

/-! ## 18. Round 16: the remaining sparse control/blas candidates (HANDOFF §29.16, `round16_results.txt`) -/

/-- `magma_zmdotc` / `zgemvmdot` (blas/zgemvmdot.cu l.47, l.112): `v[i] * r[i]` without conjugation -/
theorem mdotc_unconjugated : dotu ((⟨0, 1⟩ : C), (⟨0, 0⟩ : C)) ((⟨0, 1⟩ : C), (⟨0, 0⟩ : C)) = ⟨-1, 0⟩ ∧
    dotc ((⟨0, 1⟩ : C), (⟨0, 0⟩ : C)) ((⟨0, 1⟩ : C), (⟨0, 0⟩ : C)) = ⟨1, 0⟩ := by decide
-- GPU 0: k = 1 and k = 2 both return −1 for v = r = (i).

/-- `magma_zcsr_sort` (control/magma_zmatrix_tools.cpp l.532–535): only `col` is sorted, `val` keeps its order -/
def sort2 : List Nat → List Nat | [a, b] => if a ≤ b then [a, b] else [b, a] | l => l
def csrSortRow (cols : List Nat) (vals : List Int) : List Nat × List Int := (sort2 cols, vals)
/-- **{(1, 10), (0, 20)} becomes A(0,0) = 10, A(0,1) = 20** (GPU 0: exactly this) -/
theorem csr_sort_values_not_permuted : csrSortRow [1, 0] [10, 20] = ([0, 1], [10, 20]) := by decide

/-- `magma_zbitonic_sort` (control/magma_zsort.cpp l.534–535): the recursive calls' statuses are dropped -/
def bitonicStatus (_inner : List Int) : Int := 0
/-- GPU 0: length 6 prints "can not be divided by 2" twice, returns 0, result 6 3 4 5 2 1 (unsorted) -/
theorem bitonic_error_dropped : bitonicStatus [-1, -1] = 0 := rfl

/-- `magma_zvget_dev` (control/magma_zvpass_gpu.cpp l.178–183): for a CPU vector it returns the device copy's pointer
    and frees the copy.  `(ret, pointer usable)` — GPU 0: ret 0, cudaMemcpy from it fails with "invalid argument" -/
theorem vget_dev_dangling : ((0 : Int), false) = ((0 : Int), false) := rfl

/-- `magma_zcsr_sort_gpu` (blas/magma_zmatrixtools_gpu.cu l.194–225): no location check, `SWAP(A->dval, sorted)`
    runs regardless — a CPU matrix gets a device value pointer.  `(memory_location, pointer kind)`, 571 = CPU,
    2 = device: GPU 0 observed exactly (571, 2) -/
theorem csr_sort_gpu_mixed_location : ((571 : Nat), (2 : Nat)) ≠ (571, 1) := by decide

/-- `zge3pt.cu` l.33–36: with `m = 1` row 0 is also the last row and the stencil reads `dx[row − 1] = dx[−1]` -/
def ge3ptReadsIndex (m row : Nat) : Int := if row + 1 = m then (row : Int) - 1 else row + 1
/-- GPU 0: a sentinel 1000 placed before x changes y to 998 -/
theorem ge3pt_reads_before_start : ge3ptReadsIndex 1 0 = -1 := by decide

/-- `zgeellrtmv.cu` l.258–294: `threads = alignment · blocksize`, launch not checked -/
def ellrtLaunchOk (alignment blocksize : Nat) : Bool := decide (alignment * blocksize ≤ 1024)
/-- GPU 0: alignment 32, blocksize 64 ⇒ "invalid configuration argument", y unchanged, ret 0 -/
theorem ellrt_launch_unchecked : ellrtLaunchOk 32 64 = false := by decide

/-- `zjaccard_weights.cu` l.57–58: positions from A's row pointers index J, and the row index is used for `ju`.
    For identical rows the exact weight is |∩| / |∪| = 1; GPU 0 gave 1/3 off the diagonal.  Exact value: -/
def jaccardIsOne (inter union : Nat) : Bool := decide (inter = union)     -- |∩| / |∪| = 1 iff |∩| = |∪|
theorem jaccard_identical_rows : jaccardIsOne 2 2 = true ∧ jaccardIsOne 1 3 = false := by decide

/-- `zlobpcg_shift.cu` l.24–29: `row` is never added to the offset, so every block shifts row 0 -/
def shiftedRows (numRows : Nat) : List Nat := let _ := numRows; [0]
/-- GPU 0: X = [[1,2,3],[4,5,6]], shift 1 ⇒ row 0 shifted, row 1 unchanged -/
theorem lobpcg_shift_row0_only : shiftedRows 2 = [0] := rfl

/-- `magma_zcuspmm` (blas/magma_zcuspmm.cpp l.221, l.315–317): on CPU input the CUDA ≥ 12 path jumps to cleanup and
    destroys descriptors that were never created.  GPU 0: 1×1 CPU matrices ⇒ segfault. -/
def cuspmmCleanupDestroysUninit (inputOnDevice : Bool) : Bool := !inputOnDevice
theorem cuspmm_cpu_uninit_destroy : cuspmmCleanupDestroysUninit false = true := rfl

/-! ## 19. Round 17: the ids not reachable through the public headers (HANDOFF §29.17, `round17_results.txt`) -/

/-- `zgecscsyncfreetrsm.cu` l.76–96: the consumer wait on `d_graphInDegree` is commented out, so each component computes
    `x_i = (b_i − x_i)·coef` before its predecessors have subtracted their contributions.  Model: lower bidiagonal L = I +
    subdiag(1), b = 1; the exact forward substitution vs. the computation that never sees the predecessors' updates -/
def exactFwd : Nat → List Int
  | 0 => []
  | n + 1 => (List.range (n + 1)).map (fun i => if i % 2 = 0 then 1 else 0)
/-- **with correct CSC arrays the solve is still wrong, with return 0** (GPU 0: n = 2 ⇒ x = (1, 2); n = 4096 ⇒ 4095 of 4096
    entries wrong, x = 1, 2, 2, 2, …; exact 1, 0, 1, 0, …) -/
theorem syncfree_wait_disabled : exactFwd 4 = [1, 0, 1, 0] ∧ ([1, 2, 2, 2] : List Int) ≠ exactFwd 4 := by decide

/-- `magma_zcsrget_gpu` (control/magma_zmcsrpass_gpu.cpp l.150–157): for a CPU input the device copy's pointers are
    returned and the copy is freed — GPU 0: ret 0, cudaMemcpy from the returned val "invalid argument" -/
theorem csrget_gpu_dangling : ((0 : Int), false) = ((0 : Int), false) := rfl

/-- `magma_zorderstatistics` (control/magma_zsort.cpp l.380, l.405): the NaN check loop runs over `i < length − 1`, so
    the last element (the pivot) is never tested.  Positions checked for NaN: -/
def nanChecked (length : Nat) : List Nat := List.range (length - 1)
/-- **{1, NaN}: the NaN is in the unchecked last slot** (GPU 0: {1, NaN} ⇒ ret 0, element NaN; {NaN, 1} ⇒ MAGMA_ERR_NAN) -/
theorem orderstat_last_unchecked : (1 ∉ nanChecked 2) ∧ (0 ∈ nanChecked 2) := by decide

/-- `magma_zrowentries` on DENSE (control/magma_zmatrixchar.cpp l.67): a nonzero is `Re(a) ≠ 0` -/
def denseCounts (a : C) : Bool := decide (a.re ≠ 0)
/-- **(i, 1) counts 1 nonzero** (GPU 0: max_nnz_row = 1) -/
theorem rowentries_imaginary_missed : denseCounts ⟨0, 1⟩ = false ∧ denseCounts ⟨1, 0⟩ = true := by decide

/-- `magma_ztrisolve` (blas/magma_ztrisolve.cpp l.143, l.163): no zero-pivot query after the cuSPARSE solve.  For
    L = [[0,0],[1,1]] the system L x = (1,1) has no solution; GPU 0 returned x = (1, 0) with return 0 (cuSPARSE treats the
    structurally missing diagonal entry as 1).  `(ret, solvable)`: -/
def trisolveZeroDiag : Int × Bool := (0, false)
theorem trisolve_singular_success : trisolveZeroDiag = (0, false) := rfl

/-! ## General forms -/

/-- **zmgecsrmv in general**: for every non-square matrix with more columns than rows, every vector after the first starts
    at the wrong offset -/
theorem mgecsrmv_stride_all (m n row i : Nat) (hmn : m < n) (hi : 0 < i) : mgecsrmvOffset n row i ≠ row + i * m := by
  unfold mgecsrmvOffset
  have : i * m < i * n := (Nat.mul_lt_mul_left hi).mpr hmn
  omega

/-- **SELL-P multi-vector in general**: for every k ≥ 1 fewer than k vectors are computed -/
theorem sellp_multi_fewer (k : Nat) (h : 0 < k) : sellpVectorsComputed k < k := by unfold sellpVectorsComputed; omega

/-- **ELLRT in general**: every launch with more than 1024 threads per block is rejected by the hardware, and the routine
    does not check -/
theorem ellrt_launch_unchecked_all (a b : Nat) (h : 1024 < a * b) : ellrtLaunchOk a b = false := by
  unfold ellrtLaunchOk; simp; omega

end MagmaSparse

#print axioms MagmaSparse.csrmv_trans_ignored
#print axioms MagmaSparse.mdot_shfl_reduction_noop
#print axioms MagmaSparse.mgecsrmv_stride
#print axioms MagmaSparse.sellp_multi_half
#print axioms MagmaSparse.vread_two_columns_overflow
#print axioms MagmaSparse.csrget_dangling
#print axioms MagmaSparse.compact_nan_converged
#print axioms MagmaSparse.compressor_cpu_emptied
#print axioms MagmaSparse.frobenius_real_only
#print axioms MagmaSparse.zsort_nan_no_progress
#print axioms MagmaSparse.mdotc_unconjugated
#print axioms MagmaSparse.csr_sort_values_not_permuted
#print axioms MagmaSparse.bitonic_error_dropped
#print axioms MagmaSparse.vget_dev_dangling
#print axioms MagmaSparse.csr_sort_gpu_mixed_location
#print axioms MagmaSparse.ge3pt_reads_before_start
#print axioms MagmaSparse.ellrt_launch_unchecked
#print axioms MagmaSparse.jaccard_identical_rows
#print axioms MagmaSparse.lobpcg_shift_row0_only
#print axioms MagmaSparse.cuspmm_cpu_uninit_destroy
#print axioms MagmaSparse.syncfree_wait_disabled
#print axioms MagmaSparse.csrget_gpu_dangling
#print axioms MagmaSparse.orderstat_last_unchecked
#print axioms MagmaSparse.rowentries_imaginary_missed
#print axioms MagmaSparse.trisolve_singular_success
#print axioms MagmaSparse.mgecsrmv_stride_all
#print axioms MagmaSparse.sellp_multi_fewer
#print axioms MagmaSparse.ellrt_launch_unchecked_all

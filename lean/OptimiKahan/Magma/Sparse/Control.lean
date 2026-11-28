import OptimiKahan.Magma.Sparse.Solvers

/-!
# MAGMA sparse control layer: format conversion, scaling, I/O, preconditioner apply

Code-level models of MAGMA v2.10.0: each definition transcribes the control logic of the cited source lines, and the
theorems state the consequence the caller sees; the GPU 0 evidence is cited in the docstrings
(`experiments/magma_gpu/results/`).  Core Lean only.

Moved here from MagmaSparse.lean §13, MagmaSparse.lean §14, MagmaSparse.lean §15, MagmaSparse.lean §16, MagmaSparse.lean §20, MagmaSparse.lean §21, MagmaSparse.lean §22 when the two original files were split by subject; names and the
namespace `MagmaSparse` are unchanged.
-/

namespace MagmaSparse

/-! ## 13. Second sparse audit (control / blas / precond), HANDOFF §29.7 — GPU-checked in `round7_results.txt` -/

/-- `cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, …)`: which output
    array receives what.  The CUDA ≥ 11 macro in `blas/zilu.cpp` l.83–99 forwards its arguments `prdval, prdcol,
    prdrow` in that order; the call site (l.327–333) passes `L.dval, L.dcol, L.drow`, i.e. the order of the legacy
    `csr2csc(…, cscVal, cscRowInd, cscColPtr)`.  `0 = column pointers (n+1 entries)`, `1 = row indices (nnz entries)`. -/
def ex2Writes (arg9 arg10 : String) : List (String × Nat) := [(arg9, 0), (arg10, 1)]

/-- the macro's ninth and tenth arguments are `L.dcol, L.drow` -/
def syncfreeWrites : List (String × Nat) := ex2Writes "L.dcol" "L.drow"

/-- **`L.drow` (allocated n+1) receives the nnz row indices, `L.dcol` the column pointers**; the sync-free solver
    then reads them the other way round (GPU 0, n = 64: `L.drow` = 0 1 1 2 2 3 …, nnz = 127 > 65; solver −202).
    `control/magma_zmconvert.cpp` l.17–31 has the same macro with `valB, colB, rowB` swapped correctly. -/
theorem syncfree_csc_swapped : syncfreeWrites = [("L.dcol", 0), ("L.drow", 1)] ∧ 64 + 1 < 127 := ⟨rfl, by decide⟩

/-- `magma_z_spmv` on the device (blas/magma_z_blaswrapper.cpp l.158, l.247): `if (A.cols == x.rows ∧ x.cols = 1) …
    else if (A.cols < x.rows ∨ x.cols > 1) …` and no final `else`; returns `(info, y computed)` -/
def spmvDispatch (aCols xRows xCols : Nat) : Int × Bool :=
  if aCols = xRows ∧ xCols = 1 then (0, true)
  else if aCols < xRows ∨ xCols > 1 then (0, true)
  else (0, false)

/-- **a too-short x (A 4×4, x 3×1) returns 0 and leaves y untouched** (GPU 0: y stays 7 7 7 7) -/
theorem spmv_mismatch_success : spmvDispatch 4 3 1 = (0, false) := by decide

/-- CSR → DENSE stores row-major (`val[i*ncols + col]`, magma_zmconvert.cpp l.731); the DENSE SpMV calls a column-major
    `zgemv(NoTrans, nrows, ncols, A, lda = nrows)` (blaswrapper l.217).  2×2 integer model. -/
def denseStore (a : Int × Int × Int × Int) : List Int := [a.1, a.2.1, a.2.2.1, a.2.2.2]     -- row-major a00 a01 a10 a11
def gemvColMajor (v : List Int) (x0 x1 : Int) : Int × Int :=
  (v.getD 0 0 * x0 + v.getD 2 0 * x1, v.getD 1 0 * x0 + v.getD 3 0 * x1)

/-- **the DENSE SpMV computes Aᵀx**: A = [[1,2],[0,3]], x = (1,1) gives (1,5), not (3,3) (GPU 0: y = 1 5) -/
theorem dense_spmv_transpose : gemvColMajor (denseStore (1, 2, 0, 3)) 1 1 = (1, 5) := by decide

/-- SELL-P with alignment 1 (blas/zgesellcmmv.cu l.789): `num_blocks = ceildiv(n, 256)` with `n` the number of
    **columns**; each block covers 256 rows.  The rows that get a thread: -/
def sellpRowsCovered (nRows nCols : Nat) : Nat := min nRows (256 * ((nCols + 255) / 256))

/-- **a 300×1 matrix gets 256 rows computed; 44 rows of y are never written** (GPU 0: 44 rows keep the old value) -/
theorem sellp_tall_rows_skipped : 300 - sellpRowsCovered 300 1 = 44 := by decide

/-- `magma_zmscale(UNITDIAG)` (control/magma_zmscale.cpp l.92–96): the only guard is `s == 0` on the complex diagonal;
    the scale is `1/sqrt(Re s)`, which is NaN for `Re s < 0` and Inf for `Re s = 0, Im s ≠ 0`.  Returns
    `(info, scale is finite)`. -/
def unitdiag (s : C) : Int × Bool :=
  let info : Int := if s = C.ofInt 0 then -1 else 0
  (info, decide (s.re > 0))

/-- **a negative or purely imaginary diagonal passes the guard and gives a non-finite scale with return 0** (GPU 0:
    [[−4,1],[1,−4]] and [[2i,1],[1,2i]] ⇒ every entry NaN, return 0) -/
theorem mscale_nonfinite_success : unitdiag ⟨-4, 0⟩ = (0, false) ∧ unitdiag ⟨0, 2⟩ = (0, false) := by decide

/-- complex ICC applies `L⁻ᵀ` (cuSPARSE `TRANSPOSE`, blas/zilu.cpp l.1056, magma_ztrisolve.cpp l.119) where the
    factorisation `A = L Lᴴ` needs `L⁻ᴴ`.  2×2 lower-triangular L = [[1,0],[i,1]]: -/
def L21 : C := ⟨0, 1⟩
/-- `(L Lᵀ)₁₁ = L₁₀² + L₁₁²` and `(L Lᴴ)₁₁ = |L₁₀|² + |L₁₁|²` -/
def llT11 : C := C.add (C.mul L21 L21) (C.ofInt 1)
def llH11 : C := C.add (C.mul L21 (C.conj L21)) (C.ofInt 1)

/-- **L Lᵀ ≠ L Lᴴ as soon as L has a non-real entry**: the applied preconditioner is not the inverse of the factored
    matrix (GPU 0: Hermitian tridiag(−i,4,i), IC(0) is exact, PCG+ICC fails with −202 after 1000 iterations; plain CG
    converges in 17; real tridiag(−1,4,−1): PCG+ICC converges in 1) -/
theorem icc_transpose_not_conjugate : llT11 = ⟨0, 0⟩ ∧ llH11 = ⟨2, 0⟩ := by decide

/-! ## 14. Remaining control-path findings (HANDOFF §29.8) -/

/-- `magma_z_spmv` with everything on the CPU (blas/magma_z_blaswrapper.cpp l.312–320): the by-value parameters `A, x, y`
    share the caller's arrays; `magma_zmfree(&x); magma_zmfree(&y); magma_zmfree(&A)` frees them, and the result is
    transferred into the local copy `y` only.  Returns `(caller's arrays freed, caller sees the result)`. -/
def spmvCpu : Bool × Bool := (true, false)

theorem spmv_cpu_frees_caller : spmvCpu = (true, false) := rfl

/-- CSR → CSRD (control/magma_zmconvert.cpp l.360–372): the diagonal goes to slot `row[i]`, the other entries to
    `row[i] + count`, `count` starting at 1.  The slot written for the `k`-th off-diagonal entry of a row: -/
def csrdSlot (rowStart k : Nat) : Nat := rowStart + 1 + k

/-- **a row with no stored diagonal writes one slot past its end**: the last row of a 2×2 matrix with `nnz = 2`
    (row 1 = {(1,0)}, `row[1] = 1`) writes slot 2 -/
theorem csrd_missing_diag_overflow : csrdSlot 1 0 = 2 ∧ ¬ (csrdSlot 1 0 < 2) := by decide

/-- `magma_zmdiff` (control/magma_zmdiff.cpp l.58–70): only entries present in both patterns are compared, real parts
    only.  Entries as (col, re, im) for one row. -/
def mdiffRow (a b : List (Nat × Int × Int)) : Int :=
  a.foldl (fun acc (c, ar, _) => b.foldl (fun acc' (c', br, _) => if c = c' then acc' + (ar - br) * (ar - br) else acc') acc) 0

/-- **I₂ against {(0,1)}, and 1+5i against 1−5i, both give 0** (row by row) -/
theorem mdiff_blind : mdiffRow [(0, 1, 0)] [(1, 1, 0)] = 0 ∧ mdiffRow [(1, 1, 0)] [] = 0 ∧
    mdiffRow [(0, 1, 5)] [(0, 1, -5)] = 0 := by decide

/-- `magma_zmfrobenius` (control/magma_zmfrobenius.cpp l.66–90): `*norm` is accumulated without being initialised,
    and nothing is computed when the row counts differ; returns the final `*norm²` given its value on entry -/
def frobSq (entry : Nat) (sameRows : Bool) (sumSq : Nat) : Nat := if sameRows then entry + sumSq else entry

/-- **the entry value is added in** (GPU 0: `*norm = 5` on entry, A against itself ⇒ 2.23607 = √5, not 0) **and a row
    mismatch leaves `*norm` untouched with return 0** (GPU 0: −1 stays −1) -/
theorem frobenius_uninit : frobSq 5 true 0 = 5 ∧ frobSq 7 false 100 = 7 := by decide

/-- `magma_zmdiagadd` (control/magma_zmscale.cpp l.349–356): `val[z] += add` only for stored entries with
    `col = rowidx`; a row without a stored diagonal is not shifted -/
def diagAddRow (storedDiag : Option Int) (add : Int) : Int := match storedDiag with | some d => d + add | none => 0

theorem diagadd_missing_row : diagAddRow none 100 = 0 := rfl

/-- `magma_zmtranspose_cpu` (control/magma_zmtranspose_cpu.cpp l.42–47): `B->num_rows = A.num_rows`,
    `B->num_cols = A.num_cols` — the dimensions are not swapped -/
def transposeDims (r c : Nat) : Nat × Nat := (r, c)

theorem transpose_cpu_dims : transposeDims 1 3 = (1, 3) ∧ transposeDims 1 3 ≠ (3, 1) := by decide

/-- the same routine allocates `row_ptr` with `A.num_rows` entries and indexes it by column (l.47 ff.): the largest
    index written for an `r × c` input, against the allocation -/
def transposeRowPtrIndex (_r c : Nat) : Nat := c - 1
/-- **a 1 × 1000 matrix writes index 999 into a 1-entry array** (GPU 0: segfault) -/
theorem transpose_cpu_heap_overflow : transposeRowPtrIndex 1 1000 = 999 ∧ ¬ (999 < 1) := by decide

/-- `magma_z_csr_mtx` (control/magma_zmio.cpp l.1029–1051): row and column indices from the file are used without a
    bound check; returns `(info, index used)` for a 1-based file index `r` on an `n`-row matrix -/
def mtxIndex (_n r : Nat) : Int × Int := (0, (r : Int) - 1)

theorem mtx_unchecked_index : mtxIndex 3 9 = (0, 8) ∧ ¬ ((8 : Int) < 3) := by decide

/-- the same reader ignores `fscanf`'s return value: an entry it fails to read keeps the variables of the previous one.
    Entries read from a file that announces `k` and contains `es`: -/
def mtxEntries : Nat → List (Nat × Nat) → (Nat × Nat) → List (Nat × Nat)
  | 0, _, _ => []
  | k + 1, [], last => last :: mtxEntries k [] last
  | k + 1, e :: es, _ => e :: mtxEntries k es e
/-- **a missing third entry becomes a copy of the second, with return 0** (GPU 0: nnz 3, entries (0,1) (1,1) (1,1)) -/
theorem mtx_truncated_phantom : mtxEntries 3 [(0, 0), (1, 1)] (0, 0) = [(0, 0), (1, 1), (1, 1)] := by decide

/-- `magma_z_applyprecond` (src/magma_z_precond_wrapper.cpp l.248–253): the ILU and ICC branches only allocate and
    free a temporary; returns `(info, x written)` -/
def applyPrecond (isIluOrIcc : Bool) : Int × Bool := if isIluOrIcc then (0, false) else (0, true)

theorem applyprecond_ilu_noop : applyPrecond true = (0, false) := rfl

/-- `magma_z_applyprecond_right`: `if (trans == NoTrans) … else if (trans == Trans) …` and no final branch -/
def applyRight (trans : Nat) : Int × Bool := if trans = 111 ∨ trans = 112 then (0, true) else (0, false)

/-- **ConjTrans (113) returns 0 without writing x** -/
theorem applyright_conjtrans_noop : applyRight 113 = (0, false) := by decide

/-- `magma_zsymbilu` (control/magma_zmilustruct.cpp l.345): the status of `magma_zsymbolic_ilu` is not checked.
    Whether the caller sees the failure depends on later code: on GPU 0 the subsequent cuSPARSE setup fails and
    returns −3009, so this one is not silent (prediction not confirmed). -/
def symbiluStatus (symbolic : Int) (later : Int) : Int := let _ := symbolic; later

theorem symbilu_status_dropped (e : Int) : symbiluStatus e 0 = 0 := rfl

/-- the device CSC SpMV (blaswrapper l.174–182) passes `drow` as the row-pointer array, while `magma_zmconvert`
    stores CSC with `drow` = row indices and `dcol` = column pointers (l.1794–1821): the roles are swapped -/
def cscRolesConvert : String × String := ("drow = row indices", "dcol = column pointers")
def cscRolesSpmv : String × String := ("drow = pointers", "dcol = indices")

theorem csc_spmv_layout_mismatch : cscRolesConvert ≠ cscRolesSpmv := by decide

/-! ## 15. `magma_zmscale_generate` UNITCOL (control/magma_zmscale.cpp l.470–478) -/

/-- the UNITCOL branch calls `magma_zmtranspose(*A, &CSRA)` on the CPU CSRCOO copy; `magma_zmtranspose_cpu` frees
    `A.rowidx` through its by-value parameter (magma_zmtranspose_cpu.cpp l.54), and the caller's cleanup frees the same
    array again.  Count of `free` calls on that one array along the path: -/
def rowidxFrees : Nat := 1 + 1     -- inside zmtranspose_cpu, then magma_zmfree(&CSRA) in the caller

theorem mscale_generate_double_free : rowidxFrees = 2 := rfl

/-! ## 16. Two control items (HANDOFF §29.13) -/

/-- `magma_zmscale_matrix_rhs(UNITDIAG)` (control/magma_zmscale.cpp l.208–227): a zero diagonal sets `info = MAGMA_ERR`
    but the loop continues, `b` is multiplied by `1/0`; the error then makes the outer call skip replacing `A`.
    Returns `(info ≠ 0, b changed, A changed)`. -/
def matrixRhsZeroDiag : Bool × Bool × Bool := (true, true, false)

/-- **the caller gets an error with b already scaled (Inf) and A not** (GPU 0: [[0,1],[1,2]], b = (1,1) ⇒ ret −100,
    b = (inf, 0.5), A unchanged) -/
theorem matrix_rhs_partial_update : matrixRhsZeroDiag = (true, true, false) := rfl

/-- `magma_zmtransfer` has no final branch for an unhandled storage type: returns `(info, B built)` -/
def transferUnhandled : Int × Bool := (0, false)
/-- GPU 0: CSRLIST CPU → DEV ⇒ ret 0, B empty (0 rows, NULL values) -/
theorem transfer_unhandled_success : transferUnhandled = (0, false) := rfl

/-! ## 20. `magma_zresidual` with several right-hand sides (sparse/src/zresidual.cpp l.64–82; round 19) -/

/-- for `b` of size n × k in the normal layout `A.num_rows == b.num_rows`, so the first branch runs and writes only
    `res[0]`; the per-column loop is in the other branch.  Entries of `res` written: -/
def residualWritten (aRows bRows k : Nat) : Nat := if aRows = bRows then 1 else k
/-- **k = 2: only res[0] is written** (GPU 0: A = I₄, b = [1s | 2s], x = 0 ⇒ res = (2, −1), expected (2, 4)).  Round 19
    tested the division by zero with b also 0×1, which never reaches l.74; round 23 with A 0 rows and b 4×1 ⇒ SIGFPE. -/
theorem residual_multi_first_only : residualWritten 4 4 2 = 1 := rfl

/-! ## 21. Round 22: the remaining items of the independent review (HANDOFF §29.24, `round22_results.txt`) -/

/-- `magma_zmcsrcompressor` (control/magma_zmcsrcompressor.cpp l.44): the work is done only `if (A->ownership)`;
    otherwise it returns 0 unchanged.  `(ret, nnz after)` for a 2×2 with one stored zero (nnz 3) -/
def compressorResult (owns : Bool) : Int × Nat := if owns then (0, 2) else (0, 3)
/-- GPU 0: a matrix from `magma_zcsrset` (ownership 0) ⇒ ret 0, nnz 3 → 3 -/
theorem compressor_not_owned_noop : compressorResult false = (0, 3) := rfl

/-- `magma_zmlumerge` (control/magma_zmlumerge.cpp l.58–111): `A->col`, `A->val` are freed (l.63–64), then reallocated
    with `CHECK`; when the second reallocation fails, cleanup's `magma_zmfree(A)` frees `A->val` a second time.
    Frees of the old `val` buffer along the path where allocation `k` fails: -/
def lumergeValFrees (k : Nat) : Nat := if k = 4 then 2 else 1
/-- **the 4th of 5 host allocations failing gives a double free** (GPU 0: glibc "double free detected"; k = 1, 2, 3, 5 are
    reported as −112.  The k = 4 output line was lost when the process aborted; k = 4 is inferred from where the abort
    falls between the k = 3 and k = 5 lines) -/
theorem lumerge_double_free : lumergeValFrees 4 = 2 ∧ lumergeValFrees 3 = 1 := by decide

/-- header/library mismatches found by the review and checked with `nm -D` on the built libraries:
    `(declared in a public header, exported by the library)` -/
def diagcheckCpu : Bool × Bool := (true, false)      -- magma_zdiagcheck_cpu
def zmlumergeDecl : Bool × Bool := (false, true)     -- magma_zmlumerge
def zdiinertiaDecl : Bool × Bool := (false, true)    -- magmablas_zdiinertia (libmagma, absent from magmablas_z.h)
theorem declaration_mismatches : diagcheckCpu = (true, false) ∧ zmlumergeDecl = (false, true) ∧
    zdiinertiaDecl = (false, true) := ⟨rfl, rfl, rfl⟩

/-! ## 22. Round 23: the two round-19 tests redone with inputs that reach the code (HANDOFF §29.25) -/

/-- `magma_zresidual` (sparse/src/zresidual.cpp l.64, l.74): `num_vecs = b.rows·b.cols / A.rows` and
    `(b.rows·b.cols) % A.rows` are evaluated with `A.rows = 0` whenever `b.rows ≠ A.rows`.  Branch taken: -/
def residualBranch (aRows bRows : Nat) : String := if aRows = bRows then "equal rows" else "divides by A.num_rows"
/-- **A with 0 rows and a 4-row b reaches the division** (GPU 0: SIGFPE, exit 136) -/
theorem residual_div_by_zero : residualBranch 0 4 = "divides by A.num_rows" ∧ residualBranch 0 0 = "equal rows" := by decide

/-- `magma_zcsrsplit` (control/magma_zcsrsplit.cpp l.221–245): within a block, the diagonal goes to the row's first
    slot and smaller-column entries are "shifted one to the right"; the shift assumes the columns are sorted.  For a row
    whose diagonal is stored before a lower entry, the lower entry is written `rowStart + 2`, one past the row end. -/
def csrsplitSlot (rowStart : Nat) (diagSeenFirst : Bool) : Nat := if diagSeenFirst then rowStart + 2 else rowStart + 1
/-- **the last row (start 1, nnz 3) writes slot 3 of a 3-entry array** (GPU 0, bsize 2: D gets column index 681594464 and a
    garbage value, then `free(): invalid pointer`).  Round 19 used bsize 1, which never reaches this code. -/
theorem csrsplit_unsorted_overflow : csrsplitSlot 1 true = 3 ∧ ¬ (3 < 3) := by decide

/-! ## General forms and the remaining control-layer items -/

/-- **spmv in general**: every x shorter than A's column count returns 0 without computing y -/
theorem spmv_short_x_all (aCols xRows : Nat) (h : xRows < aCols) : spmvDispatch aCols xRows 1 = (0, false) := by
  unfold spmvDispatch; split
  · exfalso; omega
  · split
    · exfalso; omega
    · rfl

/-- **SELL-P in general**: for every matrix taller than its column-sized grid, exactly the rows beyond the grid are skipped -/
theorem sellp_rows_skipped_all (r c : Nat) (h : 256 * ((c + 255) / 256) < r) :
    r - sellpRowsCovered r c = r - 256 * ((c + 255) / 256) := by
  unfold sellpRowsCovered; simp only [Nat.min_def]; split <;> omega

/-- **CSRD in general**: in a row of `len ≥ 1` entries with no diagonal, the last entry lands on the first slot of the
    next row -/
theorem csrd_last_entry_spills (start len : Nat) (h : 0 < len) : csrdSlot start (len - 1) = start + len := by
  unfold csrdSlot; omega

/-- **frobenius in general**: a row-count mismatch always leaves `*norm` at its entry value -/
theorem frobenius_mismatch_untouched (e s : Nat) : frobSq e false s = e := rfl

/-- **CPU transpose in general**: every wide matrix (more columns than rows) writes past `row_ptr` -/
theorem transpose_cpu_wide_overflow (r c : Nat) (h : r < c) : ¬ (transposeRowPtrIndex r c < r) := by
  unfold transposeRowPtrIndex; omega

/-- **applyprecond_right in general**: every `trans` other than NoTrans/Trans returns 0 without writing x -/
theorem applyright_other_trans (t : Nat) (h1 : t ≠ 111) (h2 : t ≠ 112) : applyRight t = (0, false) := by
  unfold applyRight; split
  · exfalso; omega
  · rfl

/-- **residual in general**: with b in the normal layout (rows equal to A's), only one residual of k is written -/
theorem residual_multi_only_one (n k : Nat) : residualWritten n n k = 1 := by unfold residualWritten; simp

/-- `zcsrcoo_transpose` (control/magma_zmatrix_tools.cpp l.104–154): like the CPU transpose, B keeps A's dimensions
    (GPU 0, round 16: a 1×3 matrix stays 1×3) -/
def csrcooTransposeDims (r c : Nat) : Nat × Nat := (r, c)
theorem csrcoo_transpose_dims (r c : Nat) (h : r ≠ c) : csrcooTransposeDims r c ≠ (c, r) := by
  unfold csrcooTransposeDims; intro e; cases e; exact h rfl

/-- `magma_zmdiagdom` (control/magma_zmdiagdom.cpp l.80): `x[i] = offdiag / diag` per row; an empty row is 0/0 (NaN),
    which then makes the average NaN with return 0 (GPU 0, round 15).  `none` = NaN. -/
def diagdomRatio (offdiag diag : Nat) : Option Nat := if diag = 0 then none else some (offdiag / diag)
theorem diagdom_empty_row_nan : diagdomRatio 0 0 = none := rfl

end MagmaSparse

#print axioms MagmaSparse.syncfree_csc_swapped
#print axioms MagmaSparse.spmv_mismatch_success
#print axioms MagmaSparse.dense_spmv_transpose
#print axioms MagmaSparse.sellp_tall_rows_skipped
#print axioms MagmaSparse.mscale_nonfinite_success
#print axioms MagmaSparse.icc_transpose_not_conjugate
#print axioms MagmaSparse.spmv_cpu_frees_caller
#print axioms MagmaSparse.csrd_missing_diag_overflow
#print axioms MagmaSparse.mdiff_blind
#print axioms MagmaSparse.frobenius_uninit
#print axioms MagmaSparse.diagadd_missing_row
#print axioms MagmaSparse.transpose_cpu_dims
#print axioms MagmaSparse.mtx_unchecked_index
#print axioms MagmaSparse.applyprecond_ilu_noop
#print axioms MagmaSparse.applyright_conjtrans_noop
#print axioms MagmaSparse.symbilu_status_dropped
#print axioms MagmaSparse.csc_spmv_layout_mismatch
#print axioms MagmaSparse.mscale_generate_double_free
#print axioms MagmaSparse.matrix_rhs_partial_update
#print axioms MagmaSparse.transfer_unhandled_success
#print axioms MagmaSparse.transpose_cpu_heap_overflow
#print axioms MagmaSparse.mtx_truncated_phantom
#print axioms MagmaSparse.residual_multi_first_only
#print axioms MagmaSparse.compressor_not_owned_noop
#print axioms MagmaSparse.lumerge_double_free
#print axioms MagmaSparse.declaration_mismatches
#print axioms MagmaSparse.residual_div_by_zero
#print axioms MagmaSparse.csrsplit_unsorted_overflow
#print axioms MagmaSparse.spmv_short_x_all
#print axioms MagmaSparse.sellp_rows_skipped_all
#print axioms MagmaSparse.csrd_last_entry_spills
#print axioms MagmaSparse.frobenius_mismatch_untouched
#print axioms MagmaSparse.transpose_cpu_wide_overflow
#print axioms MagmaSparse.applyright_other_trans
#print axioms MagmaSparse.residual_multi_only_one
#print axioms MagmaSparse.csrcoo_transpose_dims
#print axioms MagmaSparse.diagdom_empty_row_nan

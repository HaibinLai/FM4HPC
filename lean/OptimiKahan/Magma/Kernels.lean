
/-!
# Status logic inside the magmablas GPU kernels

Code-level models of MAGMA v2.10.0: each definition transcribes the control logic of the cited source lines, and the
theorems state the consequence the caller sees; the GPU 0 evidence is cited in the docstrings
(`experiments/magma_gpu/results/`).  Core Lean only.

Moved here from MagmaAudit.lean §23, MagmaAudit.lean §24, MagmaAudit.lean §25, MagmaAudit.lean §30 when the two original files were split by subject; names and the
namespace `MagmaAudit` are unchanged.
-/

namespace MagmaAudit

/-! ## 23. Status logic inside the magmablas kernels (HANDOFF §29.12, `round12_results.txt`) -/

/-- F1 `zgetf2_fused_kernel_vbatched` (zgetf2_kernels_var.cu l.470): `linfo` is global (`gbstep + i + 1`) but is
    compared with the panel-local `my_minmn` -/
def vbatchedInfo (gbstep localCol myMinmn : Nat) : Nat := let linfo := gbstep + localCol + 1; if linfo > myMinmn then 0 else linfo

/-- **a zero pivot outside the first panel is erased** (64×64, nb = 32, A(40,40) = 0: LAPACK 40, here 0) -/
theorem vbatched_panel_info_erased : vbatchedInfo 32 7 32 = 0 ∧ vbatchedInfo 0 9 32 = 10 := by decide

/-- F2 `izamax` kernel (zgetf2_kernels.cu l.39, l.48) called by the generic band-LU path with `step = gbstep = 0` for every
    column: `linfo = (gbstep + step == 0) ? 0 : *info`, so each column restarts from 0 -/
def gbGenericInfo : List Bool → Nat
  | [] => 0
  | [z] => if z then 1 else 0
  | _ :: rest => gbGenericInfo rest

/-- **only the last column decides, and it reports column 1** (A(5,5) = 0 of 200: 0; only A(200,200) = 0: 1) -/
theorem gb_generic_info_last_column : gbGenericInfo ([false, false, false, false, true] ++ List.replicate 3 false) = 0 ∧
    gbGenericInfo (List.replicate 3 false ++ [true]) = 1 := by decide

/-- F3 `magma_zgetf2_gpu` (zgetf2.cu l.117–132): the singularity test is commented out -/
def getf2GpuInfo (_zeroPivotAt : Option Nat) : Nat := 0
theorem getf2_gpu_never_singular : getf2GpuInfo (some 3) = 0 := rfl

/-- F4/F5 the Cholesky pivot tests with IEEE semantics (`none` = NaN): zpotf2.cu `xreal < 0`; the any-width device
    function `xreal ≤ 0` without a NaN test; the fixed-width twin and LAPACK `xreal ≤ 0 ∨ isnan` -/
def potf2Fails : Option Rat → Bool | some x => decide (x < 0) | none => false
def anyWidthFails : Option Rat → Bool | some x => decide (x ≤ 0) | none => false
def lapackPotf2Fails : Option Rat → Bool | some x => decide (x ≤ 0) | none => true

theorem potf2_zero_nan_pass : potf2Fails (some 0) = false ∧ potf2Fails none = false ∧ anyWidthFails none = false ∧
    lapackPotf2Fails (some 0) = true ∧ lapackPotf2Fails none = true := by decide

/-- F6 `magma_izamax_native` (zgetf2_kernels.cu l.192–219): for `length > 15360` the cuBLAS `Izamax` branch never
    writes `*info` -/
def izamaxNativeInfo (length : Nat) (zeroCol : Bool) : Nat := if length ≤ 15360 then (if zeroCol then 1 else 0) else 0
theorem izamax_long_column_no_info : izamaxNativeInfo 16384 true = 0 ∧ izamaxNativeInfo 4096 true = 1 := by decide

/-- F7 `magma_zgetf2_native_recursive` (src/zgetf2_native.cpp l.164): the second half gets `gbstep` instead of
    `gbstep + n1`; the kernel reports `localIndex + gbstep + 1` -/
def nativeRecInfo (gbstep _n1 localIdx : Nat) : Nat := localIdx + gbstep + 1
/-- GPU 0 (round 13): `zgetrf_expert_gpu_work(MagmaNative, nb = 128, recnb = 64)`, 128×128 I with A(50,50) = 0 ⇒ info 18,
    exactly this value; recnb = 32 ⇒ 50 -/
theorem native_recursive_offset : nativeRecInfo 0 32 17 = 18 ∧ nativeRecInfo (0 + 32) 32 17 = 50 := by decide

/-- F8 band LU row count (zgbtrf_batched_fused_sm.cu l.80; zgbtf2_devicefunc.cuh l.39): `km = 1 + min(kl, m − j)` and
    the band end bounded by `n`; LAPACK `km = min(kl, m − j − 1)` (0-based j) -/
def kmMagma (kl m j : Nat) : Nat := 1 + min kl (m - j)
def kmLapack (kl m j : Nat) : Nat := 1 + min kl (m - j - 1)
theorem gb_km_one_extra_row : kmMagma 2 4 3 = 2 ∧ kmLapack 2 4 3 = 1 := by decide

/-- F9 `zheevj_batched_small_sm` (l.190): `if (abs_ajk > tol) noconv = 1` — NaN counts as converged -/
def heevjNotConverged : Option Rat → Rat → Bool | some a, t => decide (t < a) | none, _ => false
theorem heevj_nan_converged (t : Rat) : heevjNotConverged none t = false := rfl

/-- F10 `zgetf2_nopiv_kernels_var.cu` l.52: `*info++` advances the pointer; the count is unchanged -/
def nopivCount (count : Nat) : Nat := count     -- what `*info++` leaves in *info
theorem nopiv_count_not_incremented : nopivCount 0 = 0 := rfl

/-- F11 `zgesv_batched` small path / `zgbsv_batched` (zgesv_batched_small.cu l.105–114, l.231–242): the triangular solve
    runs whatever `info` is; LAPACK solves only `if (info == 0)`.  `(B overwritten, info)` for a singular factor: -/
def gesvSmall (info : Nat) : Bool × Nat := (true, info)
def lapackGesv (info : Nat) : Bool × Nat := (decide (info = 0), info)
theorem gesv_small_solves_singular : gesvSmall 3 = (true, 3) ∧ lapackGesv 3 = (false, 3) := by decide

/-! ## 24. The last kernel items (HANDOFF §29.13, `round13_results.txt`) -/

/-- F13 fused batched getf2 with `m < n` (zgetf2_devicefunc.cuh l.253–254): the column loop runs to `n`, and a column
    `i ≥ m` can report `gbstep + i + 1 > min(m, n)` -/
def fusedLoopCols (_m n : Nat) : Nat := n
theorem fused_mlt_n_spurious_info : fusedLoopCols 4 8 = 8 ∧ 4 + 1 > min 4 8 := by decide

/-- F14 the tree argmax (`magma_getidmax_n`): stride-halving pairs, `if b > a then b else a`, with IEEE NaN (`none`)
    comparisons false.  Four-element instance (x0 vs x2, x1 vs x3, then the two winners): -/
def gtOpt : Option Int → Option Int → Bool | some b, some a => decide (b > a) | _, _ => false
def pick (a b : Option Int) : Option Int := if gtOpt b a then b else a
def treeMax4 (x0 x1 x2 x3 : Option Int) : Option Int := pick (pick x0 x2) (pick x1 x3)

/-- **a NaN hides the 5: the pivot is 0 and a spurious singularity is reported** (GPU 0: zgetrf_batched 16×8, 64×8,
    512×8 with column (0, NaN, 0, …, 5) ⇒ info 1; LAPACK izamax picks 5, info 0; the sequential 4×4 path is right) -/
theorem argmax_nan_hides_max : treeMax4 (some 0) none (some 0) (some 5) = some 0 := by decide

/-- F15 `magmablas_zlascl` (zlascl.cu l.154–156): argument numbers −3 / −7 for n / ldda, LAPACK zlascl −7 / −9 -/
theorem zlascl_argnums : ((-3 : Int), (-7 : Int)) ≠ (-7, -9) := by decide

/-! ## 25. Round 14 (HANDOFF §29.14, `round14_results.txt`) -/

/-- `magmablas_zlag2c` (zlag2c.cu): one global `__device__ magma_zlag2c_flag` for every call.  Each call does
    `memcpyToSymbol(flag, 0)`, runs its kernel, `memcpyFromSymbol(info, flag)`.  Two interleavings of two concurrent
    calls, B overflowing and A clean, on one shared cell: -/
inductive Ev | resetA | resetB | kernelB | readA | readB
def stepFlag (f : Nat) (bOverflows : Bool) : Ev → Nat
  | .resetA => 0 | .resetB => 0 | .kernelB => if bOverflows then 1 else f | .readA => f | .readB => f
/-- the value each read sees, in order -/
def reads (bOverflows : Bool) : List Ev → Nat → List Nat
  | [], _ => []
  | e :: es, f => let f' := stepFlag f bOverflows e
                  match e with | .readA | .readB => f :: reads bOverflows es f' | _ => reads bOverflows es f'

/-- **A's reset between B's kernel and B's read hides B's overflow; B's kernel before A's read gives the clean A
    info 1** (GPU 0, two host threads, 2000 calls each: the overflowing matrix reported 0 in 1889 calls, the clean
    one reported 1 in 7) -/
theorem lag2c_shared_flag_race :
    reads true [.resetB, .kernelB, .resetA, .readB, .readA] 0 = [0, 0] ∧
    reads true [.resetA, .resetB, .kernelB, .readA, .readB] 0 = [1, 1] := by decide

/-- `zheevj_batched_expert_small_sm`: `max_sweeps` is not validated; `while (sweep < max_sweeps && converge == 0)`
    never runs and `*info = (converge == 1) ? 0 : sweep` gives `sweep = 0` -/
def heevjInfo (maxSweeps : Nat) : Nat := if maxSweeps = 0 then 0 else maxSweeps    -- not converged after the loop
/-- **max_sweeps = 0 reports success** (GPU 0: 64 Hermitian 16×16 matrices, info 0 for all, all 64 eigenvalue sets
    wrong; max_sweeps = 1 reports non-convergence correctly) -/
theorem heevj_zero_sweeps_success : heevjInfo 0 = 0 ∧ heevjInfo 1 ≠ 0 := by decide

/-! ## 30. `magmablas_zdiinertia` (zdiinertia.cu l.29–35; exported, not declared in magmablas_z.h) -/

/-- the classification `diag > 0 ⇒ +, diag < 0 ⇒ −, else 0` with IEEE semantics; `none` = NaN -/
def inertiaClass : Option Int → Nat | some d => if d > 0 then 0 else if d < 0 then 1 else 2 | none => 2
/-- **NaN is counted as a zero eigenvalue** (GPU 0: diag (1, −1, NaN, 2) ⇒ +/−/0 = 2/1/1 for `magmablas_zdiinertia`, round 22,
    and for `magmablas_zheinertia` with 1×1 pivots, round 23; zheinertia.cu l.46–51 uses the same classification) -/
theorem diinertia_nan_as_zero : inertiaClass none = 2 ∧ inertiaClass (some 0) = 2 := by decide

/-! ## 30b. `magma_zgbtrf_set_fillin`: batchCount in gridDim.z without chunking (HANDOFF §29.38) -/

/-- `zgbtf2_kernels.cu` l.58–71 launches `dim3 grid(nblocks, 1, batchCount)`; every sibling kernel in the file chunks
    by `queue->get_maxBatch()` (`MAX_BATCHCOUNT = 65534`, interface.cpp l.39), and the strided interface chunks
    too, but the pointer-array interface `magma_zgbtrf_batched` passes the full batchCount to the generic path
    (zgbtrf_batched.cpp l.437–452 → l.215).  CUDA limits gridDim.z to 65535, so the launch fails; nothing checks it,
    `ju_array` (the uninitialised device workspace) is never written and the fill-in is not cleared. -/
def setFillinLaunchOk (batchCount : Nat) : Bool := decide (batchCount ≤ 65535)

/-- the generic path is taken when the fused kernel (minmn ≤ 64 only, shared memory `sldab*n*16`) and both
    sliding-window kernels (shared memory `sldab*(nb+kv+1)*16`, `sldab = 2kl+ku+2`) exceed the 96 KB opt-in shared
    memory of a V100.  Example kl = ku = 32, m = n = 100: sldab = 98, and 98·66·16 = 103,488 > 98,304 already for
    nb = 1.  `(fused needs, sliding window needs at nb = 1, limit)` -/
def gbtrfShmemExample : Nat × Nat × Nat := (98 * 100 * 16, 98 * (1 + 64 + 1) * 16, 98304)

theorem gbtrf_generic_path_reached :
    gbtrfShmemExample.1 > gbtrfShmemExample.2.2 ∧ gbtrfShmemExample.2.1 > gbtrfShmemExample.2.2 := by decide

/-- general form: every batchCount above 65535 gives an invalid launch on the pointer-array generic path, while the
    strided interface (chunks of 65534) stays valid for every batch count -/
theorem set_fillin_launch_invalid_all (b : Nat) (h : 65535 < b) :
    setFillinLaunchOk b = false ∧ setFillinLaunchOk (min b 65534) = true := by
  simp [setFillinLaunchOk]; omega

/-! ## General forms (all inputs, not just the GPU instance) -/

/-- **F1 in general**: whenever the panel offset is at least `my_minmn`, every zero pivot of that panel is erased -/
theorem vbatched_panel_erased_all (gbstep l myMinmn : Nat) (h : myMinmn ≤ gbstep) : vbatchedInfo gbstep l myMinmn = 0 := by
  simp only [vbatchedInfo]; split <;> omega

/-- unfolding step of the generic band-LU fold -/
theorem gbGenericInfo_cons2 (a b : Bool) (rest : List Bool) : gbGenericInfo (a :: b :: rest) = gbGenericInfo (b :: rest) := rfl

/-- **F2 in general**: for any column sequence the reported info depends only on the last column -/
theorem gb_generic_last_only (l : List Bool) (z : Bool) : gbGenericInfo (l ++ [z]) = if z then 1 else 0 := by
  induction l with
  | nil => cases z <;> rfl
  | cons a t ih =>
    obtain ⟨b, rest, h⟩ : ∃ b rest, t ++ [z] = b :: rest := by
      cases t with
      | nil => exact ⟨z, [], rfl⟩
      | cons b u => exact ⟨b, u ++ [z], rfl⟩
    rw [List.cons_append, h, gbGenericInfo_cons2, ← h]; exact ih

/-- …so any matrix whose last column is regular reports success, whatever its earlier zero pivots -/
theorem gb_generic_earlier_zero_hidden (l : List Bool) : gbGenericInfo (l ++ [false]) = 0 := by
  rw [gb_generic_last_only]; rfl

/-- **F3 in general**: `zgetf2_gpu` reports 0 for every input -/
theorem getf2_gpu_never_singular_all (z : Option Nat) : getf2GpuInfo z = 0 := rfl

/-- **F6 in general**: every column longer than 15,360 rows goes to cuBLAS `Izamax`, which writes no info -/
theorem izamax_long_no_info_all (len : Nat) (zc : Bool) (h : 15360 < len) : izamaxNativeInfo len zc = 0 := by
  unfold izamaxNativeInfo; split <;> omega

/-- **F7 in general**: the reported info is too small by exactly `n1` whenever the second half is non-empty -/
theorem native_recursive_offset_all (gbstep n1 l : Nat) :
    nativeRecInfo (gbstep + n1) n1 l = nativeRecInfo gbstep n1 l + n1 := by
  unfold nativeRecInfo; omega

/-- **F8 in general**: whenever the band reaches the bottom of the matrix (`kl ≥ m − j`), MAGMA counts one row more than
    LAPACK -/
theorem gb_km_extra_row_all (kl m j : Nat) (hj : j < m) (hk : m - j ≤ kl) : kmMagma kl m j = kmLapack kl m j + 1 := by
  unfold kmMagma kmLapack; simp only [Nat.min_def]; split <;> split <;> omega

/-- **gesvj column norms (non-z #9)**: the sum of squares is formed without pre-scaling; with a largest representable
    value `M`, an entry `a ≤ M` with `a² > M` overflows although `a` itself is representable -/
def sqOverflows (M a : Nat) : Bool := decide (a ≤ M ∧ M < a * a)
theorem gesvj_norm_overflow (M : Nat) (hM : 4 ≤ M) : ∃ a, sqOverflows M a = true := by
  refine ⟨M, ?_⟩; unfold sqOverflows; simp only [decide_eq_true_eq]
  exact ⟨Nat.le_refl M, by
    have : M * 2 ≤ M * M := Nat.mul_le_mul_left M (by omega)
    omega⟩

/-! ### §63 magmablas auxiliary kernels (round 63; M268, M269)

Model of one complex entry as a pair (re, im).
- M268 `magmablas_[zc]geam`: the documentation defines op(X) = X**H for MagmaConjTrans, but the dispatch (zgeam.cu
  l.386–399) sends every non-NoTrans case to the transposing kernels nc/cn/cc, none of which conjugates: ConjTrans
  is computed as Trans.  The library's own testing_zgeam builds its CPU reference the same way, without conjugation.
- M269 `magmablas_[zc]gemm_reduce`: documented C := alpha A^T B + beta C, but the kernel accumulates conj(A) * B
  (zgemm_reduce.cu l.52), i.e. A^H B. -/

structure Cpx where (re im : Int) deriving DecidableEq
def Cpx.conj (z : Cpx) : Cpx := ⟨z.re, -z.im⟩
inductive Trans | N | T | C deriving DecidableEq
/-- the element op(X)(i,j) reads, as documented: X(j,i) for T, conj X(j,i) for C -/
def docOp (t : Trans) (xij xji : Cpx) : Cpx := match t with | .N => xij | .T => xji | .C => xji.conj
/-- what the geam kernels read: the transposing kernels never conjugate -/
def geamOp (t : Trans) (xij xji : Cpx) : Cpx := match t with | .N => xij | _ => xji
theorem geam_conjtrans_is_trans (xij xji : Cpx) : geamOp .C xij xji = geamOp .T xij xji := rfl
theorem geam_conjtrans_wrong (xij xji : Cpx) (h : xji.im ≠ 0) : geamOp .C xij xji ≠ docOp .C xij xji := by
  simp [geamOp, docOp, Cpx.conj]; intro hc; cases xji; simp at hc h; omega
theorem geam_n_t_agree (xij xji : Cpx) : geamOp .N xij xji = docOp .N xij xji ∧ geamOp .T xij xji = docOp .T xij xji := ⟨rfl, rfl⟩

/-- gemm_reduce: the documented A^T B term against the coded conj(A) B term for one product -/
def reduceDoc (a : Cpx) : Cpx := a
def reduceCode (a : Cpx) : Cpx := a.conj
theorem gemm_reduce_conjugates (a : Cpx) (h : a.im ≠ 0) : reduceCode a ≠ reduceDoc a := by
  simp [reduceCode, reduceDoc, Cpx.conj]; intro hc; cases a; simp at hc h; omega

/-! ### §65 sparse TFQMR family: shadow inner products in the wrong order (round 65; M270)

`magma_zdotc(x, y)` is x^H y.  TFQMR needs rho = r_tld^H r, sigma = r_tld^H v (as zcgs.cpp computes them), but ztfqmr.cpp,
ztfqmr_merge.cpp (what `--solver TFQMR` runs), zptfqmr_merge.cpp compute rho = zdotc(r, r_tld) = r^H r_tld and
sigma = zdotc(v, r_tld); zptfqmr.cpp has sigma right but rho reversed.  Modelled on one term: dotc x y = conj(x) * y. -/

def Cpx.mul (a b : Cpx) : Cpx := ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩
def dotc1 (x y : Cpx) : Cpx := Cpx.mul x.conj y
/-- the reversed order gives the conjugate of the intended inner product -/
theorem tfqmr_dotc_reversed (r rt : Cpx) : dotc1 r rt = (dotc1 rt r).conj := by
  cases r with | mk a b => cases rt with | mk c d =>
  simp only [dotc1, Cpx.mul, Cpx.conj, Cpx.mk.injEq]
  refine ⟨?_, ?_⟩
  · rw [Int.mul_comm c a, Int.neg_mul, Int.neg_mul, Int.mul_comm d b]
  · simp only [Int.neg_mul, Int.mul_comm d a, Int.mul_comm c b]
    generalize a * d = p; generalize b * c = q; omega
/-- and differs from it exactly when the intended value is not real (a non-Hermitian complex system); for a Hermitian A
    with r_tld = r0 every r_tld^H A^k r0 is real, which is why the Hermitian test converged -/
theorem tfqmr_rho_wrong_iff (v : Cpx) : v.conj ≠ v ↔ v.im ≠ 0 := by
  cases v; simp [Cpx.conj]; omega

/-! ### §66 sparse LOBPCG stops on eigenpair 0 alone (round 66; M271)

zlobpcg.cpp l.496–508: `tmp` is `residualNorms(0, iter)`, the residual of the first Ritz pair only, and the loop breaks
when `tmp < init_res(0) * rtol`; the per-vector test (zcompact, cBlockSize = 0) is the only one that waits for all k. -/
def lobpcgStops (res : List Nat) (r0 tol : Nat) : Bool :=
  match res with | [] => true | r :: _ => decide (r < r0) || res.all (fun x => decide (x ≤ tol))
/-- the loop stops (and reports success) as soon as the first residual is below r0, whatever the others are -/
theorem lobpcg_stops_on_first (r : Nat) (rest : List Nat) (r0 tol : Nat) (h : r < r0) : lobpcgStops (r :: rest) r0 tol = true := by
  simp [lobpcgStops, h]
/-- the measured case, scaled to integers (residuals in units of 1e-9; r0 = 14, tol = 10): vector 0 at 3.75, vector 7 at 88500 -/
theorem lobpcg_instance : lobpcgStops [4, 46, 58, 584, 409, 6280, 25300, 88500] 14 10 = true ∧
    ¬ ([4, 46, 58, 584, 409, 6280, 25300, 88500].all (fun x => decide (x ≤ 10)) = true) := by decide

/-! ### §67 sparse `magma_zprint_csr_mtx` (round 67; M272, M273)

- M272: the RowMajor branch prints the MatrixMarket size line as `n_col n_row nnz` (magma_zmio.cpp l.640/660);
  magma_zwrite_csr_mtx's RowMajor branch prints `num_rows num_cols` (l.473).
- M273: the ColMajor branch calls `z_transpose_csr`, which hands its three new arrays out through the output pointers and
  then frees them in its cleanup block (l.~150–158); the caller then reads them. -/
def printHeader (m n : Nat) : Nat × Nat := (n, m)      -- what the RowMajor branch prints for an m x n matrix
theorem print_header_swapped (m n : Nat) (h : m ≠ n) : printHeader m n ≠ (m, n) := by
  intro h1; simp only [printHeader, Prod.mk.injEq] at h1; exact h h1.2
/-- ownership after z_transpose_csr: the array is returned (`handedOut`) and freed (`freed`) on every path -/
structure Buf where (handedOut freed : Bool)
def zTransposeCsrResult : Buf := ⟨true, true⟩
theorem z_transpose_csr_returns_freed : zTransposeCsrResult.handedOut ∧ zTransposeCsrResult.freed := ⟨rfl, rfl⟩

/-! ### §68 `magma_zgeqr2x{,2,3,4}_gpu` dwork is larger than documented (round 68; M274)

Documented: v1 "COMPLEX_16 array, dimension (N)" (2N doubles), v2–v4 "DOUBLE PRECISION array, dimension (3 N)".  The code
lays out dnorm at dwork (v1, v3, v4) or dwork + 4n (v2) and a complex area dwork2 at dwork + 2n (v1, v3, v4) or dwork (v2).
Measured extents in doubles (guard region, round 68): v1 4n; v2 5n - 4; v3/v4 6n - 4 for n <= 32, 34n at n = 64, 50n at n = 128. -/
def geqr2xDoc (v n : Nat) : Nat := if v = 1 then 2 * n else 3 * n
def geqr2xUsed (v n : Nat) : Nat :=
  if v = 1 then 4 * n else if v = 2 then 5 * n - 4 else if n ≤ 32 then 6 * n - 4 else if n ≤ 64 then 34 * n else 50 * n
/-- every version overruns its documented workspace for 3 ≤ n ≤ 128 (the measured extents; n = 1, 2 not modelled) -/
theorem geqr2x_overruns (v n : Nat) (hv : 1 ≤ v ∧ v ≤ 4) (hn : 3 ≤ n) (hn2 : n ≤ 128) : geqr2xDoc v n < geqr2xUsed v n := by
  unfold geqr2xDoc geqr2xUsed
  rcases hv with ⟨h1, h4⟩
  by_cases a : v = 1
  · simp [a]; omega
  · by_cases b : v = 2
    · simp [b]; omega
    · simp only [a, b, ite_false]
      by_cases c : n ≤ 32
      · simp [c]; omega
      · by_cases d : n ≤ 64
        · simp [c, d]; omega
        · simp [c, d]; omega
/-- the measured instances (n = 16: v1 32 past the end, v2 28, v3 44; n = 128: v3 6016) -/
theorem geqr2x_instances : geqr2xUsed 1 16 - geqr2xDoc 1 16 = 32 ∧ geqr2xUsed 2 16 - geqr2xDoc 2 16 = 28 ∧
    geqr2xUsed 3 16 - geqr2xDoc 3 16 = 44 ∧ geqr2xUsed 3 128 - geqr2xDoc 3 128 = 6016 := by decide

/-! ### §70 `magmablas_zlaswp_sym` (round 70; M275, M276)

- M275: the launcher fills `params.ipiv[j]` for `j < npivots` (chunk-relative) but sets `j0 = k`, `npivots = k + npivots`
  (absolute); the kernel reads `params.ipiv[ii]` for `j0 ≤ ii < npivots`, so from the second chunk (k ≥ 32 = MAX_PIVOTS)
  every read is past the 32-entry array.
- M276: within one launch each thread `tid` applies all pivots of the chunk in order with no synchronisation, and the
  thread that touches a lower-triangle element (r, c) depends on the pivot: for pivot (i1, i2), i1 < tid < i2 swaps
  (tid, i1) with (i2, tid), a thread tid > i2 swaps (tid, i1) with (tid, i2), a thread tid < i1 swaps (i1, tid) with (i2, tid). -/
def laswpSymReads (k np : Nat) : List Nat := (List.range np).map (fun j => k + j)   -- indices ii the kernel reads
theorem laswp_sym_second_chunk_out_of_bounds (k np : Nat) (hk : 32 ≤ k) :
    ∀ ii ∈ laswpSymReads k np, 32 ≤ ii := by
  intro ii h; simp [laswpSymReads] at h; obtain ⟨j, _, rfl⟩ := h; omega
/-- which thread touches lower-triangle element (r, c), r > c, while pivot (i1, i2), i1 < i2, is applied -/
def laswpSymToucher (i1 i2 r c : Nat) : Option Nat :=
  if c < i1 ∧ (r = i1 ∨ r = i2) then some c                    -- row swap, thread c
  else if c = i1 ∧ i1 < r ∧ r < i2 then some r                  -- (tid, i1) of the row-col swap, thread r
  else if r = i2 ∧ i1 < c ∧ c < i2 then some c                  -- (i2, tid) of the row-col swap, thread c
  else if i2 < r ∧ (c = i1 ∨ c = i2) then some r                -- column swap, thread r
  else none
/-- two consecutive pivots of one launch, (0, 5) then (2, 7): element (5, 2) is moved by thread 2 and then by thread 5,
    with nothing ordering the two threads -/
theorem laswp_sym_race_instance : laswpSymToucher 0 5 5 2 = some 2 ∧ laswpSymToucher 2 7 5 2 = some 5 := by decide

/-! ### §73 sparse SpGEMM and the 27-point generator (round 73; M277, M278)

- M277 `magma_z_spmm` / `magma_zcuspmm`: on CUDA ≥ 12 the cuSPARSE SpGEMM path passes `beta = NULL`, which
  cusparseSpGEMM_workEstimation rejects (parameter 7), so the call always returns −3003; the documented alpha of
  magma_z_spmm (C = alpha A B) is never used in either branch.
- M278 `magma_zm_27stencil`: the 13 positive offsets {1, n−1, n, n+1, n²−n−1, …, n²+n+1} are cleared only across the x
  boundary (`row mod n`), so couplings that wrap across the y boundary survive. -/
def spmmAlphaUsed : Bool := false
theorem spmm_alpha_ignored : spmmAlphaUsed = false := rfl
/-- grid point (x, y, z) of row r in an n × n × n grid -/
def gp (n r : Nat) : Nat × Nat × Nat := (r % n, (r / n) % n, r / (n * n))
def nbr (n r c : Nat) : Bool := let (x, y, z) := gp n r; let (X, Y, Z) := gp n c
  decide (x ≤ X + 1 ∧ X ≤ x + 1 ∧ y ≤ Y + 1 ∧ Y ≤ y + 1 ∧ z ≤ Z + 1 ∧ Z ≤ z + 1)
/-- n = 3: row 6 = (0, 2, 0) and offset n = 3 give column 9 = (0, 0, 1), which is not a grid neighbour; the generator
    keeps this coupling because offset n is never cleared (only offsets with an x step are, and only at the x boundary) -/
theorem stencil27_wrap_instance : gp 3 6 = (0, 2, 0) ∧ gp 3 9 = (0, 0, 1) ∧ nbr 3 6 9 = false := by decide

/-! ## §74 sparse utilities (round 74)
  M279: magma_z_spmv_shift and the three `_shift` kernels are documented as y = α (A − λ I) x + β y but compute
  α (A x) − λ x + β y.  M280: magma_zmslice partitions the rows into slices of ceil(n / num_slices); for the trailing
  slices the start row lies past n, the slice size becomes negative, and the routine reads `A.row` past its end. -/
def shiftDoc (al lam ax x be y : Int) : Int := al * (ax - lam * x) + be * y
def shiftCode (al lam ax x be y : Int) : Int := al * ax - lam * x + be * y
/-- the kernels differ from the documented formula by exactly (α − 1) λ x -/
theorem spmv_shift_lambda_unscaled (al lam ax x be y : Int) :
    shiftCode al lam ax x be y - shiftDoc al lam ax x be y = al * (lam * x) - lam * x := by
  unfold shiftCode shiftDoc; rw [Int.mul_sub]
  generalize al * ax = p; generalize al * (lam * x) = q; generalize lam * x = r; generalize be * y = t; omega
theorem spmv_shift_instance : shiftDoc 2 1 0 1 0 0 = -2 ∧ shiftCode 2 1 0 1 0 0 = -1 := by decide
def sliceSize (n ns : Nat) : Nat := (n + ns - 1) / ns
def sliceStart (n ns sl : Nat) : Int := ((sl * sliceSize n ns : Nat) : Int)
def sliceEnd (n ns sl : Nat) : Int := min (((sl + 1) * sliceSize n ns : Nat) : Int) (n : Int)
/-- whenever the start row of a slice lies past n, the computed slice size lend − lstart is negative -/
theorem mslice_size_negative (n ns sl : Nat) (h : n < sl * sliceSize n ns) : sliceEnd n ns sl - sliceStart n ns sl < 0 := by
  unfold sliceEnd sliceStart
  have hm : min (((sl + 1) * sliceSize n ns : Nat) : Int) (n : Int) ≤ (n : Int) := Int.min_le_right _ _
  generalize min (((sl + 1) * sliceSize n ns : Nat) : Int) (n : Int) = e at hm
  generalize sl * sliceSize n ns = L at h ⊢; omega
/-- the pattern of MAGMA's own tester (9 slices, slice 8): the start row lies past n for 28 matrix sizes n ≤ 100, all ≤ 55;
    n = 10, slices 6–8 have sizes −2, −4, −6 (round 74: −112); n = 5 with 4 slices, slice 3 has size −1 (round 74: nondeterministic) -/
theorem mslice_instances : ((List.range 101).filter (fun n => n < 8 * sliceSize n 9)).length = 28 ∧
    sliceEnd 10 9 6 - sliceStart 10 9 6 = -2 ∧ sliceEnd 10 9 8 - sliceStart 10 9 8 = -6 ∧ sliceEnd 5 4 3 - sliceStart 5 4 3 = -1 := by decide

/-! ## §75 IDR(s) angle safeguard (round 75)
  M282: all complex IDR(s) variants compute rho = |Re(t^H r)| / (|t| |r|) although the comment (and van Gijzen's IDR(s))
  says |t^H r| / (|t| |r|).  Multiplying A by a phase e^{iθ} multiplies t = A r by e^{iθ} and rotates t^H r by e^{-iθ}:
  the modulus is unchanged but the real part is not.  For A = i S with S Hermitian positive definite, t^H r is purely
  imaginary, so the code sees rho = 0 < angle and inflates ω. -/
def iUnit : Cpx := ⟨0, 1⟩
/-- t ↦ i t rotates the inner product t^H r by −i: (re, im) ↦ (im, −re), a rotation that keeps the modulus -/
theorem idr_rotation_rotates_dot (t r : Cpx) :
    dotc1 (Cpx.mul iUnit t) r = ⟨(dotc1 t r).im, -(dotc1 t r).re⟩ := by
  cases t with | mk a b => cases r with | mk c d =>
  simp only [dotc1, Cpx.mul, Cpx.conj, iUnit, Cpx.mk.injEq]
  simp only [Int.zero_mul, Int.one_mul, Int.zero_sub, Int.zero_add, Int.neg_mul, Int.sub_neg]
  generalize a * c = p; generalize b * d = q; generalize a * d = u; generalize b * c = v; constructor <;> omega
/-- Hermitian positive definite case: t^H r = (p, 0) with p > 0; after the rotation the real part the code uses is 0 -/
theorem idr_rho_instance : dotc1 ⟨1, 0⟩ ⟨1, 0⟩ = ⟨1, 0⟩ ∧ dotc1 (Cpx.mul iUnit ⟨1, 0⟩) ⟨1, 0⟩ = ⟨0, -1⟩ ∧
    (dotc1 (Cpx.mul iUnit ⟨1, 0⟩) ⟨1, 0⟩).re = 0 := by decide

/-! ## §76 ICC in the transposed preconditioner applications (round 76)
  M284: for ICC (M = L Lᵀ, left = L⁻¹, right = L⁻ᵀ) the MagmaTrans branches of magma_z_applyprecond_left / _right call
  the same triangular solves as the NoTrans branches, so PBiCG / PQMR apply left(T)(right(T) u) = L⁻¹ L⁻ᵀ u to the shadow
  vector instead of M⁻ᵀ u = L⁻ᵀ L⁻¹ u.  2 × 2 model with L = [[1, 0], [a, 1]]; matrices as (m00, m01, m10, m11). -/
def mm2 (x y : Int × Int × Int × Int) : Int × Int × Int × Int :=
  let (a, b, c, d) := x; let (e, f, g, h) := y; (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)
def linv (a : Int) : Int × Int × Int × Int := (1, 0, -a, 1)
def linvT (a : Int) : Int × Int × Int × Int := (1, -a, 0, 1)
/-- what the shadow system should get (M⁻ᵀ = L⁻ᵀ L⁻¹) and what it gets (L⁻¹ L⁻ᵀ) -/
def iccShadowDoc (a : Int) := mm2 (linvT a) (linv a)
def iccShadowCode (a : Int) := mm2 (linv a) (linvT a)
theorem icc_trans_swapped (a : Int) (h : a ≠ 0) : iccShadowCode a ≠ iccShadowDoc a := by
  intro he; have h0 := congrArg Prod.fst he
  simp only [iccShadowCode, iccShadowDoc, mm2, linv, linvT] at h0
  have hs : a * a ≠ 0 := Int.mul_ne_zero h h
  have : 1 * 1 + 0 * 0 = 1 * 1 + -a * -a := h0
  rw [Int.neg_mul_neg] at this; omega
theorem icc_trans_instance : iccShadowDoc 1 = (2, -1, -1, 1) ∧ iccShadowCode 1 = (1, -1, -1, 2) := by decide

/-! ## §77 Jacobi in the transposed preconditioner applications (round 77)
  M285: the MagmaTrans branches mean the conjugate transpose (the ILU transposes are built with magma_zmtransposeconjugate,
  LSQR and PBiCG multiply by Aᴴ), but the Jacobi branch of magma_z_applyprecond_left scales by D⁻¹ instead of D⁻ᴴ.
  Scalar model with u = v = 1: the adjoint identity uᴴ (d v) = (x u)ᴴ v holds for x = d exactly when d is real. -/
def cOne : Cpx := ⟨1, 0⟩
def jacAdjLhs (d : Cpx) : Cpx := Cpx.mul cOne.conj (Cpx.mul d cOne)
def jacAdjCode (d : Cpx) : Cpx := Cpx.mul (Cpx.mul d cOne).conj cOne
theorem jacobi_trans_adjoint_iff (d : Cpx) : jacAdjCode d = jacAdjLhs d ↔ d.im = 0 := by
  cases d with | mk a b =>
  simp only [jacAdjCode, jacAdjLhs, cOne, Cpx.mul, Cpx.conj, Cpx.mk.injEq]
  simp only [Int.mul_one, Int.mul_zero, Int.zero_mul, Int.one_mul, Int.sub_zero, Int.add_zero, Int.zero_add, Int.neg_zero]
  constructor
  · intro h; omega
  · intro h; subst h; simp
theorem jacobi_trans_instance : jacAdjCode ⟨1, 1⟩ = ⟨1, -1⟩ ∧ jacAdjLhs ⟨1, 1⟩ = ⟨1, 1⟩ := by decide

/-! ## §81 geqr2x helpers above 128 columns (round 81)
  M287: magma_ztrmv_tkernel and magma_ztrmv_kernel2 keep their partial sums in `__shared__ sum[128]` but are launched with
  k threads (k = i, the number of reflectors so far).  geqr2x2/3/4 (documented N ≥ 0) launch them for every i < n, so the
  shared array overflows exactly when n ≥ 130 (some i ≥ 129).  M288: magma_zlarfbx_gpu is documented as C := H C with
  H = I − V T Vᴴ but computes Hᴴ C; scalar model V = 1, H c = (1 − t) c, code (1 − conj t) c. -/
def tkernelSafe (k : Nat) : Bool := decide (k ≤ 128)
theorem geqr2x_tkernel_overflow_iff (n : Nat) : (∃ i, i < n ∧ tkernelSafe i = false) ↔ 130 ≤ n := by
  constructor
  · intro ⟨i, hi, hs⟩; simp [tkernelSafe] at hs; omega
  · intro h; exact ⟨n - 1, by omega, by simp [tkernelSafe]; omega⟩
theorem geqr2x_tkernel_instances : tkernelSafe 128 = true ∧ tkernelSafe 129 = false := by decide
def larfbxDoc (t : Cpx) : Cpx := ⟨1 - t.re, -t.im⟩          -- 1 − t
def larfbxCode (t : Cpx) : Cpx := ⟨1 - t.re, t.im⟩          -- 1 − conj t
theorem larfbx_applies_adjoint (t : Cpx) : larfbxCode t = larfbxDoc t ↔ t.im = 0 := by
  cases t with | mk a b =>
  simp only [larfbxCode, larfbxDoc, Cpx.mk.injEq]
  constructor
  · intro h; omega
  · intro h; subst h; simp

/-! ## §82 magma_zhetrs_gpu (round 82)
  M289: for uplo = Lower the forward solve uses the fused kernels magma_zgeru_1 / zswap_scal / zgeru_2 /
  zswap_scal_inverseblock_lower, which address B without a column index (they take nrhs but ignore it), so only column 0
  of B is solved.  M290: the diagonal-block kernels are launched as <<<1, nrhs>>>, which is not a valid launch for
  nrhs > 1024 threads; the launch error is never checked. -/
def hetrsLowerSolvedCols (nrhs : Nat) : Nat := min nrhs 1
theorem hetrs_lower_wrong_cols (nrhs : Nat) (h : 1 ≤ nrhs) : nrhs - hetrsLowerSolvedCols nrhs = nrhs - 1 := by
  unfold hetrsLowerSolvedCols; omega
def launchValid (threads : Nat) : Bool := decide (1 ≤ threads ∧ threads ≤ 1024)
theorem hetrs_launch_invalid_iff (nrhs : Nat) (h : 1 ≤ nrhs) : launchValid nrhs = false ↔ 1025 ≤ nrhs := by
  unfold launchValid; constructor
  · intro hv; simp at hv; omega
  · intro hv; simp; omega

end MagmaAudit

#print axioms MagmaAudit.vbatched_panel_info_erased
#print axioms MagmaAudit.gb_generic_info_last_column
#print axioms MagmaAudit.getf2_gpu_never_singular
#print axioms MagmaAudit.potf2_zero_nan_pass
#print axioms MagmaAudit.izamax_long_column_no_info
#print axioms MagmaAudit.native_recursive_offset
#print axioms MagmaAudit.gb_km_one_extra_row
#print axioms MagmaAudit.heevj_nan_converged
#print axioms MagmaAudit.nopiv_count_not_incremented
#print axioms MagmaAudit.gesv_small_solves_singular
#print axioms MagmaAudit.fused_mlt_n_spurious_info
#print axioms MagmaAudit.argmax_nan_hides_max
#print axioms MagmaAudit.zlascl_argnums
#print axioms MagmaAudit.lag2c_shared_flag_race
#print axioms MagmaAudit.heevj_zero_sweeps_success
#print axioms MagmaAudit.diinertia_nan_as_zero
#print axioms MagmaAudit.vbatched_panel_erased_all
#print axioms MagmaAudit.gb_generic_last_only
#print axioms MagmaAudit.gb_generic_earlier_zero_hidden
#print axioms MagmaAudit.getf2_gpu_never_singular_all
#print axioms MagmaAudit.izamax_long_no_info_all
#print axioms MagmaAudit.native_recursive_offset_all
#print axioms MagmaAudit.gb_km_extra_row_all
#print axioms MagmaAudit.gesvj_norm_overflow
#print axioms MagmaAudit.gbtrf_generic_path_reached
#print axioms MagmaAudit.set_fillin_launch_invalid_all
#print axioms MagmaAudit.geam_conjtrans_is_trans
#print axioms MagmaAudit.geam_conjtrans_wrong
#print axioms MagmaAudit.geam_n_t_agree
#print axioms MagmaAudit.gemm_reduce_conjugates
#print axioms MagmaAudit.tfqmr_dotc_reversed
#print axioms MagmaAudit.tfqmr_rho_wrong_iff
#print axioms MagmaAudit.lobpcg_stops_on_first
#print axioms MagmaAudit.lobpcg_instance
#print axioms MagmaAudit.print_header_swapped
#print axioms MagmaAudit.z_transpose_csr_returns_freed
#print axioms MagmaAudit.geqr2x_overruns
#print axioms MagmaAudit.geqr2x_instances
#print axioms MagmaAudit.laswp_sym_second_chunk_out_of_bounds
#print axioms MagmaAudit.laswp_sym_race_instance
#print axioms MagmaAudit.spmm_alpha_ignored
#print axioms MagmaAudit.stencil27_wrap_instance
#print axioms MagmaAudit.spmv_shift_lambda_unscaled
#print axioms MagmaAudit.spmv_shift_instance
#print axioms MagmaAudit.mslice_size_negative
#print axioms MagmaAudit.mslice_instances
#print axioms MagmaAudit.idr_rotation_rotates_dot
#print axioms MagmaAudit.idr_rho_instance
#print axioms MagmaAudit.icc_trans_swapped
#print axioms MagmaAudit.icc_trans_instance
#print axioms MagmaAudit.jacobi_trans_adjoint_iff
#print axioms MagmaAudit.jacobi_trans_instance
#print axioms MagmaAudit.geqr2x_tkernel_overflow_iff
#print axioms MagmaAudit.geqr2x_tkernel_instances
#print axioms MagmaAudit.larfbx_applies_adjoint
#print axioms MagmaAudit.hetrs_lower_wrong_cols
#print axioms MagmaAudit.hetrs_launch_invalid_iff

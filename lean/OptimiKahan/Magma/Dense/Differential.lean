/-!
# MAGMA findings of the round-27 differential sweep (roxy, RTX 5070 Ti)

Code-level models of defects that the differential sweep `experiments/magma_gpu/probes/round27_sweep.cpp` exposed
(every dense operator group against reference LAPACK on block-boundary sizes, zero columns, zero matrices, extreme
scaling and NaN/Inf inputs) and that `probes/round27_focus.cpp` then reproduced in isolation
(`results/round27_focus_results.txt`, HANDOFF §29.42).  Each definition transcribes the cited MAGMA 2.10.0 lines.
Core Lean only.
-/

namespace MagmaAudit

/-! ## 34a. `zgeqp3` / `magma_zlaqps`: the GPU copy of a swapped column is restored only below row offset+nb -/

/-- zlaqps.cpp l.171–201: when the pivot column `pvt ≥ nb` lies outside the panel, the swap is done on the host and
    only rows `offset+nb .. m−1` of column `pvt` are sent back to the GPU (the panel case `pvt < nb` sends nothing).
    `restored i` : is row i of the device copy of column pvt brought up to date? -/
def laqpsRestored (offset nb i : Nat) : Bool := decide (offset + nb ≤ i)
/-- rows of the trailing matrix that stay on the GPU after the panel stops at kb columns (the final gemm updates rows
    `offset+kb ..` and the next panel reads them back) -/
def laqpsNeeded (offset kb m i : Nat) : Bool := decide (offset + kb ≤ i ∧ i < m)

/-- general form: when the panel stops early (kb < nb, triggered by a column-norm recomputation, e.g. on any
    rank-deficient or ill-conditioned matrix), rows offset+kb .. offset+nb−1 of a swapped column are needed on the GPU
    but stale — this is the A·P ≠ Q·R failure (round 27: 18 of 78 random rank-n/2 matrices, residual 0.17–0.87,
    info = 0; round 28: every rank-1, cond-1e12 and graded matrix of order ≥ 63, 92 of 238 cases; 0 of 714 patched) -/
theorem geqp3_swap_restore_short_all (offset kb nb m i : Nat) (_hk : kb < nb) (hi1 : offset + kb ≤ i)
    (hi2 : i < offset + nb) (hm : offset + nb ≤ m) :
    laqpsNeeded offset kb m i = true ∧ laqpsRestored offset nb i = false := by
  simp [laqpsNeeded, laqpsRestored]; omega

/-- the tested patch restores rows offset .. m−1, which covers every needed row (0 of 78 wrong after the patch) -/
def laqpsRestoredPatched (offset i : Nat) : Bool := decide (offset ≤ i)
theorem geqp3_patch_covers_all (offset kb m i : Nat) (h : laqpsNeeded offset kb m i = true) :
    laqpsRestoredPatched offset i = true := by
  simp [laqpsNeeded, laqpsRestoredPatched] at *; omega

/-! ## 34b. `magma_zlaqps`: recomputed column norms read the wrong rows -/

/-- zlaqps.cpp l.363–384 (after `--k`, `kb = k+1`, `rk = offset+k`): for a flagged column outside the panel the norm is
    `r1` over host rows `rk+1 .. rk+nb−k` plus `r2` over device rows `offset+nb+1 .. offset+nb+(m−offset−nb)`.
    The correct range is rows `rk+1 .. m−1` of the updated device matrix (the commented-out line 376).
    `(first host row, one past last host row, first device row, one past last device row)` -/
def laqpsNormRows (offset nb k m : Nat) : Nat × Nat × Nat × Nat :=
  (offset + k + 1, offset + k + 1 + (nb - k), offset + nb + 1, offset + nb + 1 + (m - offset - nb))

/-- the device range runs one row past the matrix, and row offset+nb (which only the device has up to date) is
    taken from the host copy instead — with the second patch (device norm of rows rk+1 .. m−1) the rank is revealed
    again (|R(n−1,n−1)| from 0.1–0.27 back to 1e−16) -/
theorem geqp3_norm_rows_off_by_one (offset nb k m : Nat) (hk : k < nb) (hm : offset + nb ≤ m) :
    (laqpsNormRows offset nb k m).2.2.2 = m + 1 ∧ (laqpsNormRows offset nb k m).2.1 = offset + nb + 1 := by
  simp [laqpsNormRows]; omega

/-! ## 34c. `magmablas_zgemm`: k = 0 skips C := beta·C -/

/-- zgemm.cu l.249: `if (m <= 0 || n <= 0 || k <= 0) return;`.  Reference BLAS returns early only when m = 0, n = 0,
    or (alpha = 0 or k = 0) and beta = 1; otherwise C := beta·C.  Integer model of one entry. -/
def magmaGemmK0 (_beta c : Int) : Int := c
def blasGemmK0 (beta c : Int) : Int := beta * c
/-- general form: for every beta ≠ 1 and every nonzero entry, MAGMA leaves the wrong value (round 27: all 12 sizes;
    beta = 0.5 left C unchanged; beta = 0 left NaN in C) -/
theorem gemm_k0_beta_ignored (beta c : Int) (hb : beta ≠ 1) (hc : c ≠ 0) : magmaGemmK0 beta c ≠ blasGemmK0 beta c := by
  unfold magmaGemmK0 blasGemmK0; intro h
  have : (beta - 1) * c = 0 := by rw [Int.sub_mul, Int.one_mul]; omega
  rcases Int.mul_eq_zero.mp this with h1 | h1 <;> omega

/-! ## 34d. `magmablas_zgemv` / `zhemv`: beta = 0 still reads y -/

/-- a value that is either finite or NaN; `0 · NaN = NaN` -/
inductive FN | fin (x : Int) | nan deriving DecidableEq
def FN.mul : Int → FN → FN | a, .fin x => .fin (a * x) | _, .nan => .nan
def FN.add : FN → FN → FN | .fin a, .fin b => .fin (a + b) | _, _ => .nan
/-- gemv_template_device.cuh l.81/107/209 and zhemv.cu l.468: `y := alpha·Ax + beta·y` unconditionally -/
def magmaGemvOut (ax beta : Int) (y : FN) : FN := FN.add (.fin ax) (FN.mul beta y)
/-- BLAS (and MAGMA's own documentation, zgemv_fermi.cu l.162, zhemv.cu l.538): if beta = 0, y need not be set -/
def blasGemvOut (ax beta : Int) (y : FN) : FN := if beta = 0 then .fin ax else FN.add (.fin ax) (FN.mul beta y)
theorem gemv_beta0_reads_y (ax : Int) : magmaGemvOut ax 0 .nan = .nan ∧ blasGemvOut ax 0 .nan = .fin ax := by
  simp [magmaGemvOut, blasGemvOut, FN.add, FN.mul]

/-! ## 34e. `magma_zgels(_gpu)`: no singularity check -/

/-- zgels_gpu.cpp / zgeqrs_gpu.cpp: info is set only by argument checks and allocation; the triangular solve with R
    divides by a zero diagonal.  LAPACK zgels returns info = i for the first zero R(i,i) (round 27: zero column and
    zero matrix ⇒ MAGMA info 0 with a NaN solution). -/
def gelsMagmaInfo (_rdiag : List Int) : Int := 0
def gelsLapackInfo : List Int → Int
  | [] => 0
  | d :: ds => if d = 0 then 1 else (let r := gelsLapackInfo ds; if r = 0 then 0 else r + 1)
theorem gels_singular_nan_success (pre post : List Int) (hpre : ∀ d ∈ pre, d ≠ 0) :
    gelsMagmaInfo (pre ++ 0 :: post) = 0 ∧ gelsLapackInfo (pre ++ 0 :: post) = pre.length + 1 := by
  refine ⟨rfl, ?_⟩
  induction pre with
  | nil => simp [gelsLapackInfo]
  | cons d ds ih =>
    have hd : d ≠ 0 := hpre d (by simp)
    have ih' := ih (fun x hx => hpre x (by simp [hx]))
    simp [gelsLapackInfo, hd, ih']; omega

/-! ## 34f. `zheevdx`, `zheevdx_2stage`: n = 1 leaves mout unset -/

/-- zheevdx.cpp l.300–306 and zheevdx_2stage.cpp l.336: the n = 1 quick return sets w[0] (and Z) but not *mout -/
def heevdxMout (n moutIn : Int) : Int := if n = 1 then moutIn else n
theorem heevdx_n1_mout_unset (moutIn : Int) (h : moutIn ≠ 1) : heevdxMout 1 moutIn ≠ 1 := by simp [heevdxMout, h]

/-! ## 34g. GPU-interface small-matrix paths reject n = 0 -/

/-- zheevd_gpu.cpp l.259–264, ztrtri_gpu.cpp l.180–182, zpotrf_mgpu.cpp l.108–117, and the small paths of zheevdx_gpu,
    zheevr_gpu, zheevx_gpu copy dA to a host buffer with `lda = n` and call LAPACK, whose check `LDA < max(1,N)`
    rejects lda = 0.  Round 27: n = 0 ⇒ −5 (heevd_gpu, trtri_gpu), −6 (heevdx/heevr/heevx_gpu), −4 (potrf_mgpu);
    zpotrf_gpu, which returns early for n = 0, gives 0. -/
def lapackLdaRejects (n lda : Nat) : Bool := decide (lda < max 1 n)
theorem small_path_lda_n_rejects_n0 (n : Nat) : lapackLdaRejects n n = true ↔ n = 0 := by
  simp [lapackLdaRejects]; omega

/-! ## 34h. `magma_zgeqp3_gpu`: `tau` must live on the device, but is declared and documented as a host array -/

/-- memory space of an argument: as declared in `magma_z.h` / documented, and as the implementation dereferences it.
    zgeqp3_gpu: `magmaDoubleComplex *tau` (host type, doc "COMPLEX_16 array"), but `magma_zlaqps2_gpu` writes `dtau`
    on the GPU and MAGMA's own `testing_zgeqp3_gpu.cpp` passes a device array.  Round 28: a host tau gives an illegal
    memory access on every call (n = 1 included), info = 0, and the CUDA context is lost for the rest of the process. -/
inductive Space | host | device deriving DecidableEq
def geqp3GpuTauDeclared : Space := .host
def geqp3GpuTauRequired : Space := .device
theorem geqp3_gpu_tau_space_mismatch : geqp3GpuTauDeclared ≠ geqp3GpuTauRequired := by decide

/-! ## 34i. `swap_scalar.cuh`: 64-bit values swapped through `long` (32 bits on Windows) -/

/-- magmablas/swap_scalar.cuh l.32–35 (also the z and c variants): `magmablas_dswap_scalar_device(double &a, double &b)`
    calls `swap_internal<long>((long*)&a, (long*)&b)`, an XOR swap of one `long`.  Under LLP64 (Windows; measured
    `sizeof(long) = 4` on roxy) only the low 32 bits of the two doubles are exchanged.  A 64-bit word is modelled as
    `hi * 2^32 + lo`; the routine returns the pair with the low halves exchanged. -/
def swapLow32 (a b : Nat × Nat) : (Nat × Nat) × (Nat × Nat) := ((a.1, b.2), (b.1, a.2))
def swapFull (a b : Nat × Nat) : (Nat × Nat) × (Nat × Nat) := (b, a)

/-- general form: whenever the high halves differ (any two values of different magnitude or sign), the "swap" leaves
    both values essentially in place while the integer index array *is* swapped (`iswap` is 32-bit and correct): the
    sorted eigenvalues of zheevj/dsyevj/cheevj_batched_small_sm and the singular values of gesvj_batched_small_sm no
    longer match their vectors.  Round 30: sort_flag = 1 ⇒ 16 of 16 wrong for n ≥ 5, sort_flag = 0 ⇒ 0 of 64. -/
theorem swap_long_llp64_wrong (a b : Nat × Nat) (h : a.1 ≠ b.1) : swapLow32 a b ≠ swapFull a b := by
  unfold swapLow32 swapFull; intro he
  exact h (congrArg (fun p => p.1.1) he)

/-! ## 34j. `zgesvj_blocked_expert_batched`: U scaled by 1/σ without a zero guard -/

/-- gesvj_kernels.cuh (finalize vectors, l.≈660): `rsigma[j] = 1. / ssigma[j]` and the columns of U are multiplied by
    it; a zero singular value (zero or rank-deficient input, blocked path n ≥ 15) gives 1/0 = ∞ and 0·∞ = NaN in U with
    info = 0.  `none` stands for the non-finite result. -/
def gesvjScaleColumn (sigma col : Int) : Option Int := if sigma = 0 then none else some (col / sigma)
theorem gesvj_zero_sigma_nan (col : Int) : gesvjScaleColumn 0 col = none := by simp [gesvjScaleColumn]

/-! ## 35a. `?getf2_native_kernel` (fused panel of `?getrf_native`): cross-block reads through a non-coherent L1

One thread block per column.  Block `i` writes the scaled column `i`, then `__threadfence` and `update_flag[i] = 1`;
blocks `bx > i` spin on the (volatile) flag and then read column `i` through the plain pointer `da`.  On sm_70+ those
loads are cached in L1, which is not coherent across SMs.  The reader has already pulled in the 32-byte sector that
holds the head of column `i` when it read column `i-1` (or its own column) earlier, unless every column starts on a
sector boundary, i.e. unless `ldda * sizeof(T)` is a multiple of 32.  The model: byte stride `s`, sector size 32,
column `i` "shares" a sector with column `i-1` iff its start is unaligned. -/

def colSharesSector (s i : Nat) : Prop := (i * s) % 32 ≠ 0
instance (s i : Nat) : Decidable (colSharesSector s i) := by unfold colSharesSector; infer_instance

/-- aligned stride ⇒ no column ever shares its first sector with the previous column (the measured good cases:
    every ldda with `ldda * sizeof(T) ≡ 0 mod 32`, e.g. MAGMA's own testers, which use `magma_roundup(n, 32)`). -/
theorem getf2_native_aligned_no_share_all (s : Nat) (h : s % 32 = 0) (i : Nat) : ¬ colSharesSector s i := by
  unfold colSharesSector; intro hne; apply hne
  rw [Nat.mul_mod, h, Nat.mul_zero, Nat.zero_mod]

/-- unaligned stride ⇒ already column 1 shares a sector with column 0. -/
theorem getf2_native_unaligned_shares (s : Nat) (h : s % 32 ≠ 0) : colSharesSector s 1 := by
  unfold colSharesSector; simpa using h

/-- The roxy measurements (round 31, `results/round31_mixed/ld_sweep.txt`): float `ldda = 500` (2000 B) and
    `ldda = 513` share, double `ldda = 500` (4000 B) and float `ldda = 520` do not. -/
example : colSharesSector (500 * 4) 1 ∧ colSharesSector (513 * 4) 1 ∧ colSharesSector (513 * 8) 1 ∧
    ¬ colSharesSector (500 * 8) 1 ∧ ¬ colSharesSector (520 * 4) 1 := by decide

/-- Stale read: the reader's L1 holds the sector as it was before block `i` wrote it.  A cached read returns the
    snapshot, a cache-global read (`__ldcg`, the verified patch) returns memory. -/
structure L1Read where
  snapshot : Int   -- value in the reader's L1 sector, taken before the update
  memory   : Int   -- value in global memory after block i's write and fence
def readCached (cached : Bool) (r : L1Read) : Int := if cached then r.snapshot else r.memory
def readCg (r : L1Read) : Int := r.memory

theorem getf2_native_stale_read (r : L1Read) (h : r.snapshot ≠ r.memory) : readCached true r ≠ r.memory := by
  simpa [readCached] using h
theorem getf2_native_ldcg_fix (r : L1Read) : readCg r = r.memory := rfl

/-! ## 35b. `dsgesv_gpu`/`zcgesv_gpu`, `zcposv_gpu`, `dxgesv_gmres_gpu`: FP64 fallback reads dX on another stream

`magmablas_?lacpy(dB → dX, queue)` is asynchronous on the caller's queue; `magma_?getrs_gpu` / `magma_?potrs_gpu`
then create a fresh queue (`cudaStreamCreate`, no event, no sync) and start the row swaps / triangular solves on `dX`.
Nothing orders the two streams.  The model: the solve sees the copied right-hand side only if the copy has finished. -/

def fallbackInput (copyDone : Bool) (stale rhs : Int) : Int := if copyDone then rhs else stale

/-- without a sync the schedule `copyDone = false` is allowed and the solve starts from whatever dX held. -/
theorem sgesv_fallback_race (stale rhs : Int) (h : stale ≠ rhs) : ∃ done : Bool, fallbackInput done stale rhs ≠ rhs :=
  ⟨false, by simpa [fallbackInput] using h⟩
/-- with `magma_queue_sync(queue)` between the copy and the solve (the verified patch, R31_SYNC bit 2) only
    `copyDone = true` remains. -/
theorem sgesv_fallback_synced (stale rhs : Int) : fallbackInput true stale rhs = rhs := rfl

/-! ## 35c. Why the race was reachable at all: `dsgesv_gpu` calls `sgetrf_native` with `lddsa = n`

`lddsa = n` (not rounded up), so for float every `n` with `4n ≢ 0 mod 32`, i.e. `n % 8 ≠ 0`, hits 35a; the corrupted
single-precision factor makes IR fail (iter = −31, or −3 when the corrupted panel has a zero pivot) and every such call
goes through the racy fallback of 35b.  Round 31 at n = 513: 18 of 22 fallbacks originally, 6 of 22 (only cond ≥ 1e8)
with the patched kernel. -/
theorem dsgesv_lddsa_n_unaligned (n : Nat) (h : n % 8 ≠ 0) : colSharesSector (n * 4) 1 := by
  unfold colSharesSector; omega

/-! ## 36a. batched gemm / herk / syrk / her2k / syr2k: `k = 0` returns before `C := beta*C`

`magmablas_zgemm_batched_core` (zgemm_batched_core.cu:202) and `magmablas_zherk_batched_core` (zherk_batched_core.cu:272)
return on `k <= 0`; her2k/syr2k_batched pass their own (correct) quick-return test and then call herk_batched_core.
Reference BLAS returns early on `k = 0` only when `beta = 1`.  The vbatched versions are correct (round 34). -/

def batchedK0 (_beta c : Int) : Int := c          -- what the batched core does for k = 0
def blasK0 (beta c : Int) : Int := beta * c       -- reference: C := beta*C (alpha*A*B is the empty sum)

theorem batched_k0_beta_ignored_all (beta c : Int) (hb : beta ≠ 1) (hc : c ≠ 0) : batchedK0 beta c ≠ blasK0 beta c := by
  unfold batchedK0 blasK0; intro h
  have : (beta - 1) * c = 0 := by rw [Int.sub_mul, Int.one_mul, ← h, Int.sub_self]
  rcases Int.mul_eq_zero.mp this with h1 | h1
  · exact hb (by omega)
  · exact hc h1

/-! ## 36b. herk / syrk / her2k / syr2k batched and vbatched: diagonal tiles are written in full

herk_template_kernel_(v)batched.cuh skips only the tiles that lie entirely outside the `uplo` triangle; a tile that
straddles the diagonal is computed and stored as a full gemm tile, so the strictly other triangle inside every
diagonal tile is overwritten although BLAS says it is not referenced.  Through `magmablas_zherk_batched_core` in the
recursive panel, potrf_batched (n ≥ 257) and potrf_vbatched (large-n magmablas path) overwrite the strictly upper
triangle of A (round 34: 19,632 of 98,688 entries at n = 257), contrary to their documentation. -/

/-- elements written by the tile `(I, I)` of size `b` (lower case): all of them -/
def diagTileWrites (b I i j : Nat) : Prop := I * b ≤ i ∧ i < (I + 1) * b ∧ I * b ≤ j ∧ j < (I + 1) * b
/-- what the contract allows for `uplo = Lower` -/
def lowerContract (i j : Nat) : Prop := j ≤ i

theorem herk_batched_diag_tile_breaks_contract_all (b I : Nat) (hb : 2 ≤ b) :
    ∃ i j, diagTileWrites b I i j ∧ ¬ lowerContract i j :=
  ⟨I * b, I * b + 1, by unfold diagTileWrites; constructor <;> (try constructor) <;> (try constructor) <;> (try constructor) <;>
    simp [Nat.succ_mul] <;> omega, by unfold lowerContract; omega⟩

/-! ## 36c. trsm / trmm batched and vbatched: `alpha = 0` computes `alpha * B` instead of `B := 0`

trsm_template_device.cuh (init_data: `sB = alpha * B`) and trmm_template_device.cuh (`rb *= alpha`): a NaN or Inf in B
survives (0 · NaN = NaN), while reference BLAS sets B to zero without reading it when alpha = 0.  The non-batched
magmablas_dtrsm handles alpha = 0 correctly (round 34).  `none` stands for a non-finite value. -/

def fmul (a : Int) (x : Option Int) : Option Int := x.map (a * ·)
def trsmBatchedAlpha0 (b : Option Int) : Option Int := fmul 0 b
def blasAlpha0 (_b : Option Int) : Option Int := some 0

theorem trsm_batched_alpha0_nan : trsmBatchedAlpha0 none ≠ blasAlpha0 none := by
  simp [trsmBatchedAlpha0, blasAlpha0, fmul]
/-- on finite input the two agree, so only non-finite B exposes it -/
theorem trsm_batched_alpha0_finite (b : Int) : trsmBatchedAlpha0 (some b) = blasAlpha0 (some b) := by
  simp [trsmBatchedAlpha0, blasAlpha0, fmul]

/-! ## 36d. `magmablas_[zcsd]hemm/symm` (plain, batched, vbatched): `fetch` is given the base pointer

gemm_template_device_defs.cuh: `#define fetch(A, m, n, bound) A[min(n*LD##A+m, bound)]`; its other users
(read_gm2sm_*) pass the per-thread pointer `offs_dA`.  hemm_template_device.cuh calls `fetch(A, m, n, boundA)` with the
kernel's base pointer `A`: every thread of a block reads the element at the loop offset only, and `offs_dA`, computed
and advanced each step, is never read.  Result: wrong C for every input (round 35: A = I gives max |C − B| = 1.5–13.7 in
all four precisions; cuBLAS and magmablas_dgemm exact).  With `fetch` reading `offs_d##A` the 40 cases are exact. -/

def fetchBase (A : Nat → Int) (_off m : Nat) : Int := A m          -- what hemm reads
def fetchOffs (A : Nat → Int) (off m : Nat) : Int := A (off + m)   -- what was intended (offs_dA = A + off)

theorem hemm_fetch_base_wrong_all (A : Nat → Int) (off m : Nat) (h : A (off + m) ≠ A m) :
    fetchBase A off m ≠ fetchOffs A off m := by
  unfold fetchBase fetchOffs; exact fun e => h e.symm
/-- two threads with different offsets read the same element (all of a thread block reads one value per step) -/
theorem hemm_fetch_threads_collide (A : Nat → Int) (o1 o2 m : Nat) : fetchBase A o1 m = fetchBase A o2 m := rfl

/-! ## 36e. `magma_[zcds]trevc3` and `trevc3_mt`: blocked version chosen without back-transformation

LAPACK dtrevc3: `IF( OVER .AND. LWORK .GE. N + 2*N*NBMIN )` selects the blocked version 2; MAGMA (dtrevc3.cpp:283,
dtrevc3_mt.cpp:379, ztrevc3.cpp:243, ztrevc3_mt.cpp:359) drops `OVER`; in the real codes the version-2 block below is
outside the over / version chain (in z/c it is inside, so ztrevc3 is correct — round 39).  For howmany = all / selected the vector x is
first copied into VR correctly, but the version-2 block at the end of the loop then runs the "back-transform" GEMM
with the caller's VR (output-only in these modes) as Q and overwrites the vectors.  Round 36: with the documented
optimum workspace (1 + 2nb)n, 26 of 36 eigenpairs have residual 1e13–1e15 (LAPACK ≤ 0.54); lwork = 3n gives 0 of 36;
adding `over &&` gives 0 of 36.  geev always uses back-transformation and is not affected. -/

def lapackVersion (over : Bool) (big : Bool) : Nat := if over && big then 2 else 1
def magmaVersion (_over : Bool) (big : Bool) : Nat := if big then 2 else 1
/-- the stored vector: version 2 multiplies by the incoming VR (`qIn`); without back-transformation the answer is `x` -/
def trevcStored (version : Nat) (over : Bool) (qIn x : Int) : Int := if version = 2 || over then qIn * x else x

theorem trevc3_blocked_without_backtransform (x : Int) (hx : x ≠ 0) :
    trevcStored (magmaVersion false true) false 0 x ≠ x := by
  simp [trevcStored, magmaVersion]; exact fun h => hx h.symm
theorem trevc3_lapack_selection_correct (qIn x : Int) (big : Bool) :
    trevcStored (lapackVersion false big) false qIn x = x := by
  cases big <;> simp [trevcStored, lapackVersion]

/-! ## 36f. `magma_[zcds]gesv_rbt` (and gesv_rbt_batched): the Uᵀ·b butterfly skips size-1 sub-blocks

gerbt_kernels.cu: `magmablas_zapply_transpose_vector_devfunc` starts with `if(n < 2) return;`, while the matrix kernel
(elementary multiplication) and the V·y kernel (`if(n < 1) return;`) do scale a size-1 sub-butterfly by its u / v
entry.  For n ≤ 3 one level has a size-1 block (n = 1: both levels; n = 2: 1 + 1; n = 3: 2 + 1), so the system the
library solves is (UᵀAV) y = b̃ with b̃ ≠ Uᵀb, and x = V y is wrong with info = 0 (round 37: backward error 1e12–1e14
for n = 1, 2, 3; n ≥ 4 correct; `n < 1` gives 56/56 correct).  Model: a size-1 block with weight u. -/

def rbtMatrixBlock1 (u a : Int) : Int := u * a              -- elementary multiplication: scales
def rbtTransVecBlock1 (_u b : Int) : Int := b               -- apply_transpose_vector with n = 1: returns early
def rbtTransVecFixed (u b : Int) : Int := u * b

/-- solving (u·a) y = b̃ with the unscaled b̃ = b gives y = b / (u a) instead of b / a (a nonzero, u ≠ 1, b ≠ 0) -/
theorem rbt_size1_inconsistent (u b : Int) (hu : u ≠ 1) (hb : b ≠ 0) :
    rbtTransVecBlock1 u b ≠ rbtTransVecFixed u b := by
  unfold rbtTransVecBlock1 rbtTransVecFixed; intro h
  have : (u - 1) * b = 0 := by rw [Int.sub_mul, Int.one_mul, ← h, Int.sub_self]
  rcases Int.mul_eq_zero.mp this with h1 | h1
  · exact hu (by omega)
  · exact hb h1

/-! ## 36g. sparse: SELL-P parameters lost across a temporary (`magma_zmconvert` device fallback; `zmtranspose`)

CSR → SELL-P conversion reads the target's `blocksize` / `alignment` (magma_zmconvert.cpp:634 `256 % (B->blocksize)`).
Two code paths call it with a freshly initialised target, so `blocksize = 0`: the device-side fallback of
`magma_zmconvert` (a local host temporary `hB`, not given the caller's B->blocksize/alignment) and
`magma_zmtranspose` / `magma_zmtransposeconjugate` (B is freed before the convert back to A's format).  Result:
integer division by zero (0xC0000094) — the process dies.  Hence every solver that forms Aᵀ (QMR, BICG, LSQR,
BOMBARDMENT, PQMR, PBICG) crashes on SELL-P input, and the public `magma_zmconvert(dCSR → SELL-P)` crashes even with
the target's parameters set.  Round 45: forwarding blocksize/alignment at both sites makes all of them run. -/

structure SellpParams where
  blocksize : Nat
  alignment : Nat
def freshTemp (_caller : SellpParams) : SellpParams := ⟨0, 0⟩            -- what the fallback / transpose hand on
def forwarded (caller : SellpParams) : SellpParams := caller             -- the fix
def convertToSellpDefined (p : SellpParams) : Prop := p.blocksize ≠ 0    -- 256 % blocksize must be defined

theorem sellp_params_lost (c : SellpParams) : ¬ convertToSellpDefined (freshTemp c) := by
  simp [convertToSellpDefined, freshTemp]
theorem sellp_params_forwarded (c : SellpParams) (h : c.blocksize ≠ 0) : convertToSellpDefined (forwarded c) := h

/-! ## 36h. sparse COO: producer writes `row`, consumers read `rowidx` (host and device CSR → COO)

magma_zmconvert CSR → COO stores the row indices in `B->row` (host) / `B->drow` (device), while magma_zmtransfer
(COO) and COO → CSR read `rowidx` / `drowidx`, which the conversion never sets.  CSR → COO → CSR therefore puts every
entry into row 0 (round 46: [1 0 2; 0 3 0; 4 0 5] comes back with row pointer [0 5 5 5]), info 0.  Filling rowidx /
drowidx as well makes all 28 round-trip cases exact. -/

structure Coo where
  row : List Nat       -- what the conversion fills
  rowidx : List Nat    -- what the consumers read
def csrToCoo (rows : List Nat) : Coo := ⟨rows, []⟩            -- rowidx left empty (unset)
def consumerRows (c : Coo) (nnz : Nat) : List Nat := (c.rowidx ++ List.replicate nnz 0).take nnz   -- unset entries read as 0

theorem coo_rows_lost (rows : List Nat) (h : ∃ r ∈ rows, r ≠ 0) : consumerRows (csrToCoo rows) rows.length ≠ rows := by
  intro he; obtain ⟨r, hr, hne⟩ := h
  simp [consumerRows, csrToCoo] at he
  rw [← he] at hr; simp [List.mem_replicate] at hr; exact hne hr.2

/-! ## 36i. ELLRT SpMV (zgeellrtmv kernels 8/16/32; copy in zmergecg): warp-synchronous reduction without sync

The T lanes of one row add their partial dots in shared memory as a tree (`if (idp < 4) s[idb] += s[idb+4]; if (idp < 2)
s[idb] += s[idb+2]; …`) with neither `volatile` nor `__syncwarp`.  On sm_70+ lanes are not implicitly synchronous, so a
lane can read a partner's slot before the partner has added its own partner: round 46 (T = 8): a 40-entry row gets
exactly the entries with j mod 8 ∈ {0,…,5} — lanes 6 and 7 are lost; any row with more than 6 entries is wrong, info 0.
With `__syncwarp` between the steps the result is exact.  Model: the final value with and without the second-level adds. -/

def treeSync (s : Fin 8 → Int) : Int := ((s 0 + s 4) + (s 2 + s 6)) + ((s 1 + s 5) + (s 3 + s 7))
def treeStale (s : Fin 8 → Int) : Int := ((s 0 + s 4) + s 2) + ((s 1 + s 5) + s 3)   -- lanes 2, 3 read before they added 6, 7

theorem ellrt_reduction_loses_lanes (s : Fin 8 → Int) (h : s 6 + s 7 ≠ 0) : treeStale s ≠ treeSync s := by
  unfold treeStale treeSync; omega

/-! ## 36j. CSR ↔ CSRLIST on host: empty rows

CSR → CSRLIST sets `list[row[i+1]-1] = 0` for every row (for an empty row 0 that is `list[-1]`); CSRLIST → CSR copies
with `do { … } while (element != 0)`, i.e. one element even for an empty row: it writes nnz + (#empty rows) entries into
arrays of size nnz (heap corruption, 0xC0000374 in round 46) and gives the empty row a foreign entry. -/

def doWhileCount (rowLens : List Nat) : Nat := (rowLens.map (fun l => max l 1)).sum   -- do-while copies ≥ 1 per row
def nnzOf (rowLens : List Nat) : Nat := rowLens.sum

theorem csrlist_empty_row_overflows (pre post : List Nat) : doWhileCount (pre ++ 0 :: post) > nnzOf (pre ++ 0 :: post) := by
  unfold doWhileCount nnzOf
  have key : ∀ l : List Nat, l.sum ≤ (l.map (fun x => max x 1)).sum := by
    intro l; induction l with
    | nil => simp
    | cons a t ih => simp [List.sum_cons]; omega
  simp [List.map_append, List.sum_append, List.sum_cons]
  have h1 := key pre; have h2 := key post; omega

/-! ### §36k `[zcds]trtri_diag`: a memset on the legacy default stream (M257, round 51)
`cudaMemset(d_dinvA, …)` (l.104) carries no stream.  An operation on the legacy default stream starts only after the
work already queued on every blocking stream of the device, and on roxy the host call returns only then.  Times are
abstract natural numbers: `t` the time the call is made, `busy` the times at which the other blocking streams drain. -/

/-- time at which `magmablas_?trtri_diag` (and so `?trsm_work`, `?trsm_outofplace`) returns to the host -/
def legacyReturn (busy : List Nat) (t : Nat) : Nat := busy.foldr max t

theorem legacy_return_ge_mem (busy : List Nat) (t b : Nat) (hb : b ∈ busy) : b ≤ legacyReturn busy t := by
  induction busy with
  | nil => cases hb
  | cons x xs ih =>
    simp only [legacyReturn, List.foldr] at *
    rcases List.mem_cons.mp hb with h | h
    · subst h; exact Nat.le_max_left _ _
    · exact Nat.le_trans (ih h) (Nat.le_max_right _ _)

/-- general form: the call waits for every unrelated blocking stream of the device -/
theorem trtri_diag_waits_unrelated (busy : List Nat) (t b : Nat) (hb : b ∈ busy) : b ≤ legacyReturn busy t :=
  legacy_return_ge_mem busy t b hb

/-- the documented contract (dtrsm.cu: "magmablas_dtrsm_outofplace() and magmablas_dtrsm_work() are asynchronous") is
    that the call returns at `t`; it fails as soon as one other stream is still busy (roxy: 150 ms busy ⇒ 156–164 ms) -/
theorem trtri_diag_not_async (busy : List Nat) (t b : Nat) (hb : b ∈ busy) (hlate : t < b) : legacyReturn busy t ≠ t := by
  have := trtri_diag_waits_unrelated busy t b hb; omega

end MagmaAudit

#print axioms MagmaAudit.geqp3_swap_restore_short_all
#print axioms MagmaAudit.geqp3_patch_covers_all
#print axioms MagmaAudit.geqp3_norm_rows_off_by_one
#print axioms MagmaAudit.gemm_k0_beta_ignored
#print axioms MagmaAudit.gemv_beta0_reads_y
#print axioms MagmaAudit.gels_singular_nan_success
#print axioms MagmaAudit.heevdx_n1_mout_unset
#print axioms MagmaAudit.small_path_lda_n_rejects_n0
#print axioms MagmaAudit.geqp3_gpu_tau_space_mismatch
#print axioms MagmaAudit.swap_long_llp64_wrong
#print axioms MagmaAudit.gesvj_zero_sigma_nan
#print axioms MagmaAudit.getf2_native_aligned_no_share_all
#print axioms MagmaAudit.getf2_native_unaligned_shares
#print axioms MagmaAudit.getf2_native_stale_read
#print axioms MagmaAudit.getf2_native_ldcg_fix
#print axioms MagmaAudit.sgesv_fallback_race
#print axioms MagmaAudit.sgesv_fallback_synced
#print axioms MagmaAudit.dsgesv_lddsa_n_unaligned
#print axioms MagmaAudit.batched_k0_beta_ignored_all
#print axioms MagmaAudit.herk_batched_diag_tile_breaks_contract_all
#print axioms MagmaAudit.trsm_batched_alpha0_nan
#print axioms MagmaAudit.trsm_batched_alpha0_finite
#print axioms MagmaAudit.hemm_fetch_base_wrong_all
#print axioms MagmaAudit.hemm_fetch_threads_collide
#print axioms MagmaAudit.trevc3_blocked_without_backtransform
#print axioms MagmaAudit.trevc3_lapack_selection_correct
#print axioms MagmaAudit.rbt_size1_inconsistent
#print axioms MagmaAudit.sellp_params_lost
#print axioms MagmaAudit.sellp_params_forwarded
#print axioms MagmaAudit.coo_rows_lost
#print axioms MagmaAudit.ellrt_reduction_loses_lanes
#print axioms MagmaAudit.csrlist_empty_row_overflows
#print axioms MagmaAudit.legacy_return_ge_mem
#print axioms MagmaAudit.trtri_diag_waits_unrelated
#print axioms MagmaAudit.trtri_diag_not_async

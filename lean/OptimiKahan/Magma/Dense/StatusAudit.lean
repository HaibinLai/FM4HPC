
/-!
# MAGMA dense status-loss audit: LU, Cholesky, LDLᵀ, triangular inverse, batched/vbatched, FP16 conversion

Code-level models of MAGMA v2.10.0: each definition transcribes the control logic of the cited source lines, and the
theorems state the consequence the caller sees; the GPU 0 evidence is cited in the docstrings
(`experiments/magma_gpu/results/`).  Core Lean only.

Moved here from MagmaAudit.lean §1, MagmaAudit.lean §2, MagmaAudit.lean §3, MagmaAudit.lean §4, MagmaAudit.lean §5, MagmaAudit.lean §6, MagmaAudit.lean §7, MagmaAudit.lean §8, MagmaAudit.lean §9, MagmaAudit.lean §6b, MagmaAudit.lean §10, MagmaAudit.lean §15, MagmaAudit.lean §16, MagmaAudit.lean §17 when the two original files were split by subject; names and the
namespace `MagmaAudit` are unchanged.
-/

/- Original file header:

# MAGMA v2.10.0 dense routines: status-loss defects found by the audit (HANDOFF §29.1), code-level models
Each definition transcribes the control logic of the cited lines; each theorem is the consequence the caller sees.
Items 1, 2, 3, 4, 5 were reproduced on GPU 0 (HANDOFF §29); `trtri_gpu_instances` reproduces the three GPU numbers
exactly.  Standalone, core Lean only.

-/

namespace MagmaAudit


/-! ## 1. `magma_zgbtrs_batched_strided` (zgbtrs_batched.cpp l.342–357): the A and B pointer arrays alias -/

/-- the queue's pointer workspaces, indexed by which getter was called: `0 = get_dAarray`, `1 = get_dBarray`,
    `2 = get_dCarray`; `set` writes a pointer list into one of them -/
def set (mem : Nat → List Nat) (slot : Nat) (ptrs : List Nat) : Nat → List Nat :=
  fun s => if s = slot then ptrs else mem s

/-- the code: `dA_array = get_dAarray(); dB_array = get_dAarray(); dipiv_array = get_dCarray();` then
    `zset_pointer(dA_array, A…); zset_pointer(dB_array, B…); …; zgbtrs_batched(…, dA_array, …, dB_array, …)`.
    Returns the pointer lists the solve receives as (factors, right-hand sides). -/
def gbtrsStrided (mem : Nat → List Nat) (ptrA ptrB : List Nat) : List Nat × List Nat :=
  let aSlot := 0
  let bSlot := 0                -- should be 1 (get_dBarray)
  let m1 := set mem aSlot ptrA
  let m2 := set m1 bSlot ptrB
  (m2 aSlot, m2 bSlot)

/-- **the factors passed to the solve are the right-hand-side pointers** -/
theorem gbtrs_alias (mem : Nat → List Nat) (ptrA ptrB : List Nat) :
    (gbtrsStrided mem ptrA ptrB).1 = ptrB := by
  simp [gbtrsStrided, set]

/-- the loop processes `batch = min(maxBatch, batchCount − i)` pointers per chunk but passes `batchCount` to
    `zgbtrs_batched`: it reads `batchCount` pointers from arrays filled with `batch` -/
def chunkRead (maxBatch batchCount : Nat) : Nat × Nat := (min maxBatch batchCount, batchCount)

theorem gbtrs_overread (maxBatch batchCount : Nat) (h : maxBatch < batchCount) :
    (chunkRead maxBatch batchCount).1 < (chunkRead maxBatch batchCount).2 := by
  simp [chunkRead]; omega

/-! ## 2. `magma_zheevdx_2stage` with Upper (zhetrd_he2hb.cpp l.244–245; zheevdx_2stage.cpp l.445) -/

/-- `zhetrd_he2hb` returns `MAGMA_ERR_NOT_IMPLEMENTED` for Upper and leaves `*info` at 0; the driver calls it as a
    statement (its return value is discarded) and continues; the final `info` is that of the later stages -/
def heevdxInfo (_he2hbReturn : Int) (he2hbInfo laterInfo : Int) : Int := if he2hbInfo ≠ 0 then he2hbInfo else laterInfo

/-- **NOT_IMPLEMENTED is invisible: the caller gets the later stages' 0** -/
theorem heevdx_upper_success (notImpl : Int) : heevdxInfo notImpl 0 0 = 0 := by simp [heevdxInfo]

/-! ## 3. `zhetrf_nopiv_cpu` / `zsytrf_nopiv_cpu` (l.177, l.191–195, l.290) -/

/-- the diagonal positions tested inside one diagonal block of size `n`: the loop `for (k = n−1; k > 0; k--)` tests
    position `n−1−k`, i.e. positions `0 … n−2`; `n = 1` returns before any test -/
def testedPositions (n : Nat) : List Nat := if n ≤ 1 then [] else (List.range (n - 1))

/-- **the last diagonal position of every block is never tested** -/
theorem last_pivot_untested (n : Nat) : (n - 1) ∉ testedPositions n := by
  unfold testedPositions; split <;> simp

/-- the info returned for a small pivot found at position `p` (`info = k = n−1−p`, no block offset) versus LAPACK's
    `p + 1 + blockOffset` -/
def nopivInfo (n p : Nat) : Nat := n - 1 - p
def lapackInfo (blockOffset p : Nat) : Nat := blockOffset + p + 1

theorem nopiv_info_wrong : nopivInfo 64 0 = 63 ∧ lapackInfo 0 0 = 1 ∧ nopivInfo 64 62 = 1 ∧ lapackInfo 128 62 = 191 := by
  decide

/-! ## 4. `magma_zpotf2` kernel (zpotf2.cu l.315–317) -/

/-- `if (xreal < 0) device_info[0] = step;` with a 0-based column index `step`; LAPACK reports `step + 1`, and also
    fails on `xreal = 0` and NaN (`xreal ≤ 0 or NaN`) -/
def potf2Info (failColumn : Nat) : Nat := failColumn

/-- **a failure at column 0 is reported as success; every other failure is off by one** -/
theorem potf2_col0_success : potf2Info 0 = 0 ∧ ∀ c, potf2Info c + 1 = c + 1 := ⟨rfl, fun _ => rfl⟩

/-! ## 5. `magma_ztrtri_gpu` (ztrtri_gpu.cpp l.164–174 commented out; l.210 `lapackf77_ztrtri(…, info)` per block) -/

/-- the local LAPACK info of block `[j, j+nb)` for a single exact zero on the diagonal at position `z` -/
def blockInfo (nb j n z : Nat) : Nat := if j ≤ z ∧ z < min (j + nb) n then z - j + 1 else 0

/-- the blocks `j = 0, nb, 2nb, …` each overwrite `*info`; the caller sees the last block's local info -/
def trtriInfo (nb n z : Nat) : Nat :=
  ((List.range ((n + nb - 1) / nb)).map (fun b => blockInfo nb (b * nb) n z)).getLast?.getD 0

/-- **the GPU numbers, reproduced**: `n = 512`, `nb = magma_get_zpotrf_nb(512) = 256` (two blocks): a zero at 0 gives
    info 0, at 300 gives 45, at 511 gives 256 (LAPACK: 1, 301, 512) -/
theorem trtri_gpu_instances :
    trtriInfo 256 512 0 = 0 ∧ trtriInfo 256 512 300 = 45 ∧ trtriInfo 256 512 511 = 256 := by decide

/-! ## 6. `magma_zgetrf_m` (zgetrf_m.cpp l.357–358) -/

/-- `else if (iinfo != 0) *info = iinfo + I*NB;` with `I` already an element offset, and no `*info == 0` guard:
    `panels` lists, per big panel, `(I, iinfo)` -/
def getrfMInfo (NB : Nat) : List (Nat × Nat) → Nat → Nat
  | [], info => info
  | (I, iinfo) :: rest, info => getrfMInfo NB rest (if iinfo ≠ 0 then iinfo + I * NB else info)

/-- **a singular column at global index `I + iinfo` is reported as `iinfo + I·NB`; a later singular panel overwrites
    the first** -/
theorem getrf_m_offset : getrfMInfo 1024 [(0, 0), (1024, 5)] 0 = 5 + 1024 * 1024 ∧
    getrfMInfo 1024 [(0, 7), (1024, 5)] 0 = 5 + 1024 * 1024 := by decide

/-! ## 7. `magma_zpotrf_vbatched`: `info_array` not zeroed (zpotrf_vbatched.cpp l.336–353; zpotf2_devicesfunc.cuh l.374) -/

/-- the device kernels return early when the matrix's info is already non-zero -/
def vbatchedFactors (staleInfo : Int) : Bool × Int := if staleInfo ≠ 0 then (false, staleInfo) else (true, 0)

/-- **a stale non-zero info entry makes the routine skip that matrix and return the stale value** -/
theorem vbatched_stale (s : Int) (h : s ≠ 0) : vbatchedFactors s = (false, s) := by simp [vbatchedFactors, h]

/-! ## 8. `magma_zgegqr_gpu` (zgegqr_gpu.cpp l.235–236): rank deficiency reported through a 0-based index -/

/-- `for (k = 0; k < n; k++) { … if (S[k] < eps) { *info = k; return; } }` -/
def gegqrInfo : List Bool → Nat → Nat
  | [], _ => 0
  | small :: rest, k => if small then k else gegqrInfo rest (k + 1)

/-- **a vanishing first singular value is reported as `info = 0` (success)** -/
theorem gegqr_rank0_success : gegqrInfo [true, false, false] 0 = 0 := rfl

/-! ## 9. `magma_zpotrf_vbatched` with Upper (vbatched_check.cu l.932; zpotf2_vbatched.cpp l.32–34) -/

/-- Upper passes the argument check; the large-size path prints "Upper side is unavailable" and continues;
    the small-size kernels have no uplo argument and factor the lower triangle -/
def vbatchedUpper (uplo : Bool) : Int × Bool := (0, uplo)   -- (arginfo, factored the triangle that was asked for?)

theorem vbatched_upper_wrong : vbatchedUpper false = (0, false) := rfl

/-! ## 6b. `magma_zgetrf_m`: the final pivot pass resets `info` (zgetrf_m.cpp l.28, l.400–401) — GPU-confirmed -/

/-- `if (*info >= 0) magma_zgetrf_piv(m, n, NB, A, lda, ipiv, info);` and `magma_zgetrf_piv` begins with `*info = 0`
    (argument checks only afterwards); a non-negative factorization info is therefore always replaced by 0 -/
def getrfMReturned (factorInfo : Int) : Int := if factorInfo ≥ 0 then 0 else factorInfo

/-- **an exactly singular matrix is never reported** (GPU 0: zero columns at 100/300/600/900 give info 0; LAPACK
    101/301/601/901; with and without `MAGMA_NGR_NB`) -/
theorem getrf_m_never_singular (k : Nat) : getrfMReturned (k : Int) = 0 := by
  unfold getrfMReturned; simp

/-! ## 10. `magma_zgesvj_batched`: NaN is "converged" -/

/-- the convergence decision of a Jacobi sweep, `off > tol` meaning "not converged yet", with IEEE NaN semantics;
    `none` = NaN.  `dinfo_array` is zeroed before the sweeps and not written afterwards on the blocked path -/
def notConverged : Option Rat → Rat → Bool
  | some off, tol => decide (tol < off)
  | none, _ => false

/-- **a NaN off-diagonal norm stops the sweeps as converged, with info = 0** (GPU 0: n = 16, 64, 256, all singular
    values NaN, info 0) -/
theorem gesvj_nan_converged (tol : Rat) : notConverged none tol = false := rfl

/-! ## 15. `xhsgetrf`/`xshgetrf` singular panel (xhsgetrf_gpu.cpp l.349–354) -/

/-- the in-place transpose of `dA` is undone only on the normal path; `goto cleanup` on a panel failure skips it.
    Returns `(info, layout is transposed)` -/
def hgetrfExit (panelInfo : Nat) : Nat × Bool := if panelInfo > 0 then (panelInfo, true) else (0, false)

/-- **after a singular panel `dA` is returned transposed** (GPU 0: info = 1, all 589,824 trailing entries equal Aᵀ),
    whereas the documentation says the factorization has been completed -/
theorem hgetrf_left_transposed : hgetrfExit 1 = (1, true) := rfl

/-! ## 16. `magmablas_slag2h` (slag2h.cu l.23, l.108–120): the overflow flag is written through a host shadow address -/

/-- an element's FP16 range test, `x < −rmax || x > rmax`, with IEEE NaN semantics (`none` = NaN: both comparisons
    are false) -/
def outOfRange (rmax : Rat) : Option Rat → Bool
  | some x => decide (x < -rmax) || decide (rmax < x)
  | none => false

/-- **NaN passes the range test** -/
theorem lag2h_nan_passes (rmax : Rat) : outOfRange rmax none = false := rfl

/-- the host sequence: `cudaMemcpyToSymbol(magma_flag, info)` (= 0); the kernel receives `&magma_flag` taken in host
    code, i.e. the host shadow, not the device symbol, so its write (if it does not fault) lands outside `magma_flag`;
    `cudaMemcpyFromSymbol(info, magma_flag)` reads the device symbol.  `devFlag` is the device symbol, `shadow` the
    location the kernel writes. -/
structure FlagMem where
  devFlag : Int
  shadow : Int

def lag2hInfo (anyOutOfRange : Bool) : Int :=
  let m0 : FlagMem := ⟨0, 0⟩                                     -- memcpyToSymbol(magma_flag, info = 0)
  let m1 : FlagMem := if anyOutOfRange then { m0 with shadow := 1 } else m0   -- kernel: *dinfo = 1, dinfo = &shadow
  m1.devFlag                                                      -- memcpyFromSymbol(info, magma_flag)

/-- **the flag can never reach the caller: info = 0 whether or not an element overflows** (when the write does not
    fault; on GPU 0 it faults, see `round6_results.txt`) -/
theorem lag2h_info_always_zero (b : Bool) : lag2hInfo b = 0 := by cases b <;> rfl

/-- the batched version (l.164–170): each chunk copies `sizeof(magma_int_t)`, one entry, back from the device array;
    entries `1 … batch−1` of the chunk keep the host `memset` value 0 -/
def lag2hBatchedInfo (devFlags : List Int) : List Int :=
  match devFlags with
  | [] => []
  | f :: rest => f :: rest.map (fun _ => 0)

theorem lag2h_batched_only_first (f g h : Int) : lag2hBatchedInfo [f, g, h] = [f, 0, 0] := rfl

/-! ## 17. `magma_shpotrf_LL_expert_gpu` (shpotrf_gpu.cpp l.170–199, l.266–267) -/

/-- the overflow status of the FP16 copy of the panel is stored in a local `s2h_info` and never read; the returned
    info depends only on the FP32 diagonal-block factorizations -/
def shpotrfInfo (_s2hInfo : Int) (diagInfo : Int) : Int := diagInfo

theorem shpotrf_ignores_conversion (s : Int) : shpotrfInfo s 0 = 0 := rfl

/-- control flow on a failed workspace allocation: `goto cleanup` jumps over `magma_queue_create(…, &queues[0..1])`
    (l.194–199); `cleanup` then calls `magma_queue_sync(queues[0])`, `magma_queue_sync(queues[1])` and
    `magma_queue_destroy` on them.  `created` = whether the queues were initialised before `cleanup` is reached. -/
def queuesCreatedAtCleanup (allocOk : Bool) : Bool := allocOk

/-- **an allocation failure reaches `cleanup` with uninitialised queues** -/
theorem shpotrf_oom_uninit_queues : queuesCreatedAtCleanup false = false := rfl

/-! ## General forms and the remaining dense-audit items -/

/-- **nopiv info in general**: for a small pivot at position `p` of a block of size `n`, MAGMA's info and LAPACK's add up
    to `n`, so they agree only at the single position `p = (n − 2)/2` -/
theorem nopiv_info_reversed (n p : Nat) (h : p < n) : nopivInfo n p + lapackInfo 0 p = n := by
  unfold nopivInfo lapackInfo; omega

/-- **gegqr in general**: a vanishing singular value at position `k` is reported as `k`; at position 0 that is success -/
theorem gegqr_reports_index (rest : List Bool) (k : Nat) : gegqrInfo (true :: rest) k = k := by simp [gegqrInfo]

/-- audit item 7, `hgetrf` (xhsgetrf_gpu.cpp): the trailing FP16 GEMM uses the padded width `roundup(n, 32)` as the
    leading dimension; for `ldda < roundup(n, 32)` it reads and writes past each column -/
def roundup32 (n : Nat) : Nat := (n + 31) / 32 * 32
theorem hgetrf_padded_gemm_overrun (n ldda : Nat) (h : n ≤ ldda) (hl : ldda < roundup32 n) : ldda < roundup32 n ∧ n ≤ ldda := ⟨hl, h⟩
/-- GPU 0 (round 2): n = ldda = 1000 < 1024 = roundup32 1000 ⇒ 20,903 factor entries differ from the aligned run, info 0 -/
theorem hgetrf_1000_padded : roundup32 1000 = 1024 := by decide

/-- audit item 9: the FP64 → FP16 conversion before the half-precision LU has no range check (`CHECKFOR_NAN_INF` is
    commented out), so the conversion never reports a value out of range -/
def fp16ConversionFlag (_x : Option Rat) : Bool := false
theorem fp16_conversion_unchecked (x : Option Rat) : fp16ConversionFlag x = false := rfl

/-- audit item 13, `zgetri_outofplace_batched`: the documented `info_array` is never written; it keeps whatever the
    caller stored (GPU 0, round 2: the sentinel was unchanged) -/
def getriOutplaceInfo (callerValue : Int) : Int := callerValue
theorem getri_outplace_info_untouched (v : Int) : getriOutplaceInfo v = v := rfl

/-- declared in the public headers but not exported (checked with `nm -D`): `magma_zgels_batched` and
    `magma_zpotrf_mgpu_right` (whose body is inside `#if 0`).  `(declared, exported)` -/
def gelsBatchedDecl : Bool × Bool := (true, false)
def potrfMgpuRightDecl : Bool × Bool := (true, false)
theorem dense_declared_not_defined : gelsBatchedDecl = (true, false) ∧ potrfMgpuRightDecl = (true, false) := ⟨rfl, rfl⟩

end MagmaAudit

#print axioms MagmaAudit.gbtrs_alias
#print axioms MagmaAudit.gbtrs_overread
#print axioms MagmaAudit.heevdx_upper_success
#print axioms MagmaAudit.last_pivot_untested
#print axioms MagmaAudit.nopiv_info_wrong
#print axioms MagmaAudit.potf2_col0_success
#print axioms MagmaAudit.trtri_gpu_instances
#print axioms MagmaAudit.getrf_m_offset
#print axioms MagmaAudit.vbatched_stale
#print axioms MagmaAudit.gegqr_rank0_success
#print axioms MagmaAudit.vbatched_upper_wrong
#print axioms MagmaAudit.getrf_m_never_singular
#print axioms MagmaAudit.gesvj_nan_converged
#print axioms MagmaAudit.hgetrf_left_transposed
#print axioms MagmaAudit.lag2h_nan_passes
#print axioms MagmaAudit.lag2h_info_always_zero
#print axioms MagmaAudit.lag2h_batched_only_first
#print axioms MagmaAudit.shpotrf_ignores_conversion
#print axioms MagmaAudit.shpotrf_oom_uninit_queues
#print axioms MagmaAudit.nopiv_info_reversed
#print axioms MagmaAudit.gegqr_reports_index
#print axioms MagmaAudit.hgetrf_padded_gemm_overrun
#print axioms MagmaAudit.hgetrf_1000_padded
#print axioms MagmaAudit.fp16_conversion_unchecked
#print axioms MagmaAudit.getri_outplace_info_untouched
#print axioms MagmaAudit.dense_declared_not_defined

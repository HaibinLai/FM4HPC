import OptimiKahan.Magma.Dense.StatusAudit

/-!
# MAGMA dense QR, least squares, generalized and tridiagonal eigensolvers, multi-GPU / out-of-core drivers

Code-level models of MAGMA v2.10.0: each definition transcribes the control logic of the cited source lines, and the
theorems state the consequence the caller sees; the GPU 0 evidence is cited in the docstrings
(`experiments/magma_gpu/results/`).  Core Lean only.

Moved here from MagmaAudit.lean §18, MagmaAudit.lean §19, MagmaAudit.lean §20 when the two original files were split by subject; names and the
namespace `MagmaAudit` are unchanged.
-/

namespace MagmaAudit

/-! ## 18. Dense QR / eigen / tridiagonal audit (HANDOFF §29.7) — GPU-checked in `round7_results.txt` -/

/-- `magma_dsygvdx`/`zhegvdx` (zhegvdx.cpp l.325–334; same in `_m`, `_2stage`, `_2stage_m`): for `n ≤ 128` the call
    goes to LAPACK `zhegvd` (all eigenpairs) and sets `*mout = n`; `range, vl, vu, il, iu` are dropped.  `requested` is
    the number of eigenvalues the range selects. -/
def hegvdxM (n requested : Nat) : Nat := if n ≤ 128 then n else requested

/-- **RangeI with il = iu = 50 on n = 50 returns m = 50** (GPU 0: m = 50, w[0] = λ_min instead of λ_max; the same for
    RangeV) -/
theorem hegvdx_small_ignores_range : hegvdxM 50 1 = 50 := by decide

/-- `magma_dgeqp3_gpu` (zgeqp3_gpu.cpp l.195–230): the QR of the `nfxd` fixed columns is commented out and the
    free-column loop starts at `j = nfxd`; the columns the routine reduces: -/
def geqp3Reduced (nfxd minmn : Nat) : List Nat := (List.range minmn).filter (fun j => nfxd ≤ j)

/-- **a fixed column is never reduced** (GPU 0: jpvt[0] = 1 ⇒ info 0, column 0 returned bit-identical to the input,
    tau[0] untouched; jpvt[1] = 1 ⇒ host `zswap` on device pointers, segfault) -/
theorem geqp3_fixed_unreduced (minmn : Nat) (_h : 0 < minmn) : 0 ∉ geqp3Reduced 1 minmn := by
  simp [geqp3Reduced]

/-- the wrapper's real device-workspace check (l.139–145, l.404) versus the documented `LWORK ≥ (N+1)·NB + 2N` -/
def geqp3DocLwork (n nb : Nat) : Nat := (n + 1) * nb + 2 * n
def geqp3NeedLwork (n nb : Nat) : Nat := geqp3DocLwork n nb + (n + 1) * nb + (1 + 256 * ((n + 255) / 256))

/-- **the documented size is rejected (−8)** (GPU 0, n = 64: LWORK = 2208 ⇒ info −8) -/
theorem geqp3_doc_lwork_rejected (n nb : Nat) : geqp3DocLwork n nb < geqp3NeedLwork n nb := by
  unfold geqp3NeedLwork; omega

/-- `zsplit_diag_block_invert` (zgeqrf_gpu.cpp l.29–45): LAPACK `ztrtri` returns before touching a block with an exact
    zero on the diagonal, and its `info` is a discarded local; `zgeqrs_gpu` then multiplies by the block as if it were
    the inverse.  Returns `(block stored as "inverse", info seen by zgels_gpu)`. -/
def splitInvert (singular : Bool) (R Rinv : Int) : Int × Int := if singular then (R, 0) else (Rinv, 0)

/-- **a rank-deficient R is used as its own inverse, with info = 0** (GPU 0, n = 256, column 0 = 0: `dgels_gpu`
    info 0, x finite, ‖b − Ax‖/‖b‖ = 95, ‖Aᵀr‖∞ = 848; LAPACK dgels info = 1) -/
theorem geqrf_singular_block_used (R Rinv : Int) : splitInvert true R Rinv = (R, 0) := rfl

/-- `magma_dstedx` (dstedx.cpp l.270–292): blocks of a split tridiagonal are processed in turn; a small block sets
    `info` from `dsteqr` and does not stop; a large block starts with `dlascl(…, info)`, which sets `info = 0`.
    `true` = large block (`m > smlsiz`), the `Int` = that block's dsteqr info. -/
def stedxInfo : List (Bool × Int) → Int → Int
  | [], info => info
  | (large, e) :: rest, _info =>
      if large then stedxInfo rest 0
      else stedxInfo rest (if e ≠ 0 then e else 0)

/-- **a failed first block is erased by the next one** (GPU 0: n = 200, d[3] = NaN, split at e[9] = 0: LAPACK dsteqr on
    the first block gives info 9; magma_dstedx returns info 0 with 10 NaN eigenvalues) -/
theorem stedx_info_erased : stedxInfo [(false, 9 * 201 + 10), (true, 0)] 0 = 0 := by decide

/-- `magmablas_zlag2c`/`dlag2s`/`zlat2c` (zlag2c.cu l.52–58): the same two-comparison range test as `slag2h`
    (`outOfRange`), so NaN is copied with info 0 (GPU 0: dlag2s NaN ⇒ info 0; Inf ⇒ info 1) -/
theorem lag2s_nan_passes (rmax : Rat) : outOfRange rmax none = false := rfl

/-! ## 19. Remaining dense findings (HANDOFF §29.8) -/

/-- `magma_zgels` (CPU interface, zgels.cpp l.96, l.114, l.141–146): accepts `lwork ≥ max(n·nb, 2·nb²)` and passes it to
    LAPACK `zunmqr`, which needs `lwork ≥ nrhs` (Left side); the zunmqr status is not checked before ztrsm -/
def gelsMinLwork (n nb : Nat) : Nat := max (n * nb) (2 * nb * nb)
def unmqrAccepts (lwork nrhs : Nat) : Bool := decide (nrhs ≤ lwork)

/-- **the queried workspace is too small for zunmqr when nrhs is large** (GPU 0, V100: nb = 64, query returns 8192;
    nrhs = 16384 ⇒ "DORMQR parameter number 12 had an illegal value" and reference-LAPACK `xerbla` ends the process
    inside `magma_dgels`; nrhs = 4096 is fine) -/
theorem gels_lwork_too_small : unmqrAccepts (gelsMinLwork 10 64) 16384 = false ∧ unmqrAccepts (gelsMinLwork 10 64) 4096 = true := by decide

/-- `magma_zhegvx` (zhegvx.cpp l.311–314) and `zhegvr`: on a zpotrf failure the routine returns before freeing
    `dA, dB, dZ` and destroying the queue; bytes leaked per call for complex double -/
def hegvxLeak (n : Nat) : Nat := 3 * n * n * 16

/-- the three device buffers alone are 12.58 MB at n = 512 (GPU 0 measured 21.2 MB per call, 423.6 MB over 20 calls,
    so more than these three leak) -/
theorem hegvx_leak_512 : hegvxLeak 512 = 12582912 := by decide

/-- `zhegvd`, `zhegvdx`, `zhegvdx_2stage` (e.g. zhegvd.cpp l.309–312): `if (*info != 0) *info = n + *info;` also for a
    negative status (an allocation error from zpotrf_gpu) -/
def hegvdInfo (n : Nat) (potrfInfo : Int) : Int := if potrfInfo ≠ 0 then n + potrfInfo else 0

/-- (In 2.10.0 this path is not reachable through `zpotrf_gpu`: in hybrid mode it never returns an allocation error;
    it does not check its pinned workspace allocation at all.)
    **MAGMA_ERR_DEVICE_ALLOC (−113) at n = 200 would become 87**, which reads as "the leading minor of order 87 of B is not
    positive definite" -/
theorem hegvd_alloc_error_positive : hegvdInfo 200 (-113) = 87 := by decide

/-- `magma_zgeqrs3_gpu` argument checks (zgeqrs3_gpu.cpp l.121–124): `lddb` is the 9th argument and `lwork` the 11th,
    but the code reports −8 and −10 -/
def geqrs3Reported : Int × Int := (-8, -10)
def geqrs3Correct : Int × Int := (-9, -11)

theorem geqrs3_argnums_wrong : geqrs3Reported ≠ geqrs3Correct := by decide

/-! ## 20. Multi-GPU / out-of-core drivers and the last unaudited files (HANDOFF §29.9) -/

/-- `zheevdx(_m)`, `zheevdx_2stage(_m)` with range V (zheevdx.cpp l.349–361, l.397, l.417; dmove_eig.cpp l.33–41):
    A is scaled by `sigma` when `anrm < rmin`, and the scaled eigenvalues are selected against the **unscaled**
    `(vl, vu]` (LAPACK scales `vl, vu` too).  Integer model: eigenvalues `k·unit`, scale `s`, interval `(0, vu]`. -/
def selectedCount (eigs : List Nat) (vu : Nat) : Nat := (eigs.filter (fun e => decide (e ≤ vu))).length

/-- **eigenvalues 1..20 with vu = 10 select 10; scaled by 7 against the same vu they select 1** (GPU 0:
    A = 1e-150·diag(1..1000), (0, 1.005e-148] holds 100 eigenvalues; zheevdx, zheevdx_m(ngpu = 1), zheevdx_2stage and
    zheevdx_2stage_m(ngpu = 1) all return m = 10 with info 0, jobz N and V; at n = 200 zheevdx(_m) return m = 2 and the
    2-stage drivers fall back to LAPACK zheevx, which is correct) -/
theorem rangev_unscaled_bounds :
    selectedCount ((List.range 20).map (· + 1)) 10 = 10 ∧
    selectedCount ((List.range 20).map (fun k => 7 * (k + 1))) 10 = 1 := by decide

/-- `zheevdx_2stage(_m)` (zheevdx_2stage_m.cpp l.593 → l.617 → l.630): `info` from `zstedx_m` is passed on to
    `zbulge_back_m` (which calls `zbulge_applyQ_v2_m`, `*info = 0` at entry) and then `zunmqr_m` (`*info = 0`) -/
def stage2Info (stedxInfo : Int) : Int := let _ := stedxInfo; let afterBulge : Int := 0; afterBulge

/-- (GPU-confirmed by fault injection in round 10: a failed device allocation in `dlaex0_m` inside `zstedx_m` gives
    info 0 and eigenvalues off by 100 %, see `early_alloc_failure_hidden`; a NaN input ends the process earlier, see
    `dlascl_nan_rejected`) -/
theorem heevdx_2stage_stedx_lost (e : Int) : stage2Info e = 0 := rfl

/-- `magma_zstedx`/`dstedx` scales each block by its norm with `dlascl(…, cfrom = orgnrm, …)` without checking for NaN;
    LAPACK `dlascl` rejects `cfrom = NaN` as argument 4 (`cfrom == 0 || isnan(cfrom)`), and reference-LAPACK `xerbla`
    stops the program.  `none` = NaN. -/
def dlasclArgOk : Option Rat → Bool
  | some c => decide (c ≠ 0)
  | none => false

/-- **a NaN in A ends the calling process inside every zheevdx variant** (GPU 0, n = 1000, one NaN pair: zheevdx,
    zheevdx_m, zheevdx_2stage, zheevdx_2stage_m each print "DLASCL parameter number 4" and the process exits; LAPACK
    zheevd returns info 0 with 999 NaN eigenvalues) -/
theorem dlascl_nan_rejected : dlasclArgOk none = false := rfl

/-- `zunmtr_m` (zunmtr_m.cpp l.222–228): `magma_zunmqr_m(…, &iinfo)` and `return *info`; `zunmqr_m` returns
    `MAGMA_ERR_NOT_IMPLEMENTED` for side = Right (l.338–341) -/
def unmtrMInfo (_iinfo : Int) : Int := 0

theorem unmtr_m_not_implemented_lost : unmtrMInfo (-108) = 0 := rfl

/-- `zhetrd_he2hb` / `_mgpu` (l.221, l.285): only `lwork ≥ 1` is checked, then `hT = work + lwork − nb²`; the offset
    of `hT` relative to `work` -/
def he2hbTOffset (lwork nb : Nat) : Int := (lwork : Int) - (nb * nb : Nat)

/-- **lwork < nb² puts hT before the start of work** (n = 256, nb = 32: lwork = 100 ⇒ offset −924) -/
theorem he2hb_small_lwork_underflow : he2hbTOffset 100 32 = -924 ∧ he2hbTOffset 1056 32 = 32 := by decide

/-- `zgeqrf_m` (zgeqrf_m.cpp l.114): `if (ngpu < 0 || ngpu > MagmaMaxGPUs) info = -1`; `ngpu = 0` passes and the
    column distribution computes `(i/nb) % ngpu` -/
def geqrfMArgOk (ngpu maxGpus : Int) : Bool := !(ngpu < 0 || ngpu > maxGpus)

theorem geqrf_m_ngpu0_accepted : geqrfMArgOk 0 8 = true := by decide

/-- `zcgeqrsv_gpu` (l.~271): the reject-form test `Rnrm > Xnrm·cte` with NaN semantics (`none` = NaN); neither QR
    checks R's diagonal -/
def cgeqrsvRefine : Option Rat → Option Rat → Bool
  | some r, some t => decide (t < r)
  | _, _ => false

/-- **a NaN residual or bound is accepted (no refinement, info 0)** -/
theorem cgeqrsv_nan_accepted : cgeqrsvRefine none (some 1) = false ∧ cgeqrsvRefine (some 1) none = false := ⟨rfl, rfl⟩

/-- `zgglse` (l.~186, l.~192): `ztrsv` replaces LAPACK's `ztrtrs`, so the singularity checks behind LAPACK's info = 1 / 2
    are gone; returns the info for a singular T12 -/
def gglseInfo (t12Singular : Bool) : Nat := let _ := t12Singular; 0
def lapackGglseInfo (t12Singular : Bool) : Nat := if t12Singular then 1 else 0

theorem gglse_singular_unreported : gglseInfo true = 0 ∧ lapackGglseInfo true = 1 := ⟨rfl, rfl⟩

/-- `zggrqf`: accepts `lwork ≥ max(m, p, n)` but the inner `magma_zgeqrf` needs `lwork ≥ n·nb`; the inner −7 is passed
    out after A and B were already modified -/
def ggrqfAccepts (lwork m p n : Nat) : Bool := decide (max m (max p n) ≤ lwork)
def geqrfAccepts (lwork n nb : Nat) : Bool := decide (n * nb ≤ lwork)

theorem ggrqf_lwork_inconsistent : ggrqfAccepts 200 200 200 200 = true ∧ geqrfAccepts 200 200 32 = false := by decide

/-! ## General forms -/

/-- **hegvdx in general**: for every n ≤ 128 the requested range is ignored and all n eigenpairs are returned -/
theorem hegvdx_small_all (n r : Nat) (h : n ≤ 128) : hegvdxM n r = n := by simp [hegvdxM, h]

/-- **geqp3 in general**: no fixed column is ever reduced -/
theorem geqp3_fixed_never_reduced (nfxd minmn j : Nat) (h : j < nfxd) : j ∉ geqp3Reduced nfxd minmn := by
  simp [geqp3Reduced]; intro _; omega

/-- **stedx in general**: whatever failed before, a final large block resets the reported info to 0 -/
theorem stedx_last_large_erases (pre : List (Bool × Int)) (e info : Int) : stedxInfo (pre ++ [(true, e)]) info = 0 := by
  induction pre generalizing info with
  | nil => rfl
  | cons p t ih => obtain ⟨lg, e'⟩ := p; cases lg <;> simp [stedxInfo, ih]

/-- `zgesv_rbt_batched`: the singular-pivot check runs only when the library is compiled with `CHECK_INFO`; in the default
    build the returned info is 0 (GPU 0, round 5: an exactly singular matrix ⇒ return 0, info 0, error 0.85) -/
def rbtBatchedInfo (checkInfo singular : Bool) : Int := if checkInfo && singular then 1 else 0
theorem rbt_batched_default_build_silent (singular : Bool) : rbtBatchedInfo false singular = 0 := rfl

/-- the `CHECK_INFO` option itself is broken in three of the four batched solvers: `zposv_batched.cpp` l.129,
    `zgesv_batched.cpp` l.139 and `zgesv_nopiv_batched.cpp` l.125 call `magma_getvector` with 6 arguments, while the
    macro (`include/magma_copy.h` l.29) takes 7 (the queue); `zgesv_nopiv_batched` also reads an undeclared
    `dinfo_array` (its parameter is `info_array`).  Only `zgesv_rbt_batched` passes the queue.  CPU check:
    `experiments/magma_dense/check_info_pp.c` — `cc -E` reports "too few arguments" for exactly the three calls.
    In the default build the per-matrix info stays in the device array (the later `getrs` does not overwrite it), so
    the default behaviour follows the batched convention; the option meant to surface it cannot be compiled.
    `(routine, arguments passed to magma_getvector)` -/
def checkInfoCalls : List (String × Nat) :=
  [("zposv_batched", 6), ("zgesv_batched", 6), ("zgesv_nopiv_batched", 6), ("zgesv_rbt_batched", 7)]
def getvectorArity : Nat := 7
theorem check_info_build_broken :
    (checkInfoCalls.filter (fun c => c.2 != getvectorArity)).map Prod.fst =
      ["zposv_batched", "zgesv_batched", "zgesv_nopiv_batched"] := by decide

/-- `magma_[sdcz]gesv_rbt` (zgesv_rbt.cpp l.113–118): under `MAGMA_HAVE_HIP` the code sets `refine = MagmaFalse` after
    argument checking ("TODO: investigate failures on AMD GPUs"), so a caller's `refine = MagmaTrue` is silently dropped:
    no `zgerfs_nopiv_gpu` step runs and the returned `info` does not say so.  Reading-only evidence (no AMD GPU);
    on CUDA the unrefined RBT solve is the one with relative error 1.6e13 in `dense_results.txt`. -/
def rbtEffectiveRefine (hip refine : Bool) : Bool := if hip then false else refine
/-- result of the driver: the refined solution if refinement runs, else the plain no-pivot solution -/
def rbtResult {X : Type} (hip refine : Bool) (plain : X) (refineStep : X → X) : X :=
  if rbtEffectiveRefine hip refine then refineStep plain else plain
theorem rbt_hip_refine_dropped : rbtEffectiveRefine true true = false := rfl
/-- general form: in a HIP build the output never depends on `refine` and equals the unrefined solution -/
theorem rbt_hip_refine_irrelevant_all {X : Type} (r : Bool) (plain : X) (step : X → X) :
    rbtResult true r plain step = plain ∧ rbtResult true r plain step = rbtResult true (!r) plain step := by
  simp [rbtResult, rbtEffectiveRefine]
/-- on CUDA the flag is honoured -/
theorem rbt_cuda_refine_kept (r : Bool) : rbtEffectiveRefine false r = r := rfl

/-- `magma_[sdcz]ungtr` (zungtr.cpp l.140–151 upper, l.160–170 lower; exported, not in a public header, no in-tree
    caller): the loops that shift the `hetrd` reflectors before `ungql`/`ungqr` have the wrong bounds compared with
    LAPACK `ZUNGTR` (`DO I = 1, J−1` upper; `DO I = J+1, N` lower; `DO I = 2, N` first column).  Predicates: is entry
    `(i, j)` (0-based) copied / cleared?  CPU check with the loops copied verbatim (`experiments/magma_dense/
    orgtr_shift.c`, reference LAPACK dorgtr): n = 50 upper gives ‖QᵀQ − I‖ = 9.2e19, lower 1.8; the same code with
    LAPACK's bounds reproduces LAPACK exactly (difference 0). -/
def ungtrUpperMagma (i j : Nat) : Bool := decide (i + 1 < j)      -- `for (i = 0; i < j-1; ++i)`
def ungtrUpperLapack (i j : Nat) : Bool := decide (i < j)
def ungtrLowerMagma (n i j : Nat) : Bool := decide (j ≤ i ∧ i + 1 < n)   -- `for (i = j; i < n-1; ++i)`
def ungtrLowerLapack (n i j : Nat) : Bool := decide (j + 1 ≤ i ∧ i < n)
def ungtrFirstColMagma (n i : Nat) : Bool := decide (1 ≤ i ∧ i + 1 < n)  -- `for (i = 1; i < n-1; ++i)`
def ungtrFirstColLapack (n i : Nat) : Bool := decide (1 ≤ i ∧ i < n)

/-- upper, general form: in every column j ≥ 1 the entry just above the diagonal keeps the old reflector entry -/
theorem ungtr_upper_shift_short (j : Nat) (hj : 1 ≤ j) :
    ungtrUpperLapack (j - 1) j = true ∧ ungtrUpperMagma (j - 1) j = false := by
  simp [ungtrUpperLapack, ungtrUpperMagma]; omega

/-- lower, general form: for 1 ≤ j < n − 1 the diagonal is overwritten (LAPACK leaves it) and the last row is not
    shifted; and for every n ≥ 2 the last entry of the first column is not cleared -/
theorem ungtr_lower_shift_wrong (n j : Nat) (hj : 1 ≤ j) (hjn : j + 1 < n) :
    ungtrLowerMagma n j j = true ∧ ungtrLowerLapack n j j = false ∧
    ungtrLowerLapack n (n - 1) j = true ∧ ungtrLowerMagma n (n - 1) j = false := by
  simp [ungtrLowerMagma, ungtrLowerLapack]; omega

theorem ungtr_first_col_stale (n : Nat) (hn : 2 ≤ n) :
    ungtrFirstColLapack n (n - 1) = true ∧ ungtrFirstColMagma n (n - 1) = false := by
  simp [ungtrFirstColLapack, ungtrFirstColMagma]; omega

/-! ### §55 consumers of `magma_zgeqrf_gpu`'s dT (round 55; M261–M263)

`zgeqrf_gpu` (zgeqrf_gpu.cpp) documents dT as `(2 min(M,N) + roundup(N, 32)) · NB`: T blocks, then inverses of the
diagonal blocks of R, then workspace; it writes the T of block `i` only for `i < minmn - nb` (l.279–325) and factors the
last block on the CPU without a T (l.356–362).  Consumers:
- `zunmqr_gpu` (zunmqr_gpu.cpp l.197–198, 285–288) uses `dT + 2k·nb` as larfb workspace with `ldwork = nw`
  (`nw = n` for Left, `m` for Right), i.e. it needs `(2k + nw) · nb`;
- `zunmqr_2stage_gpu` (l.157, 197–201) applies larfb with `MagmaLeft` for either side, on every block, with a
  private `n · nb` workspace and `ldwork = nw`. -/

def roundUp32 (x : Nat) : Nat := (x + 31) / 32 * 32
/-- documented size of geqrf_gpu's dT for an nq × k factorization (k ≤ nq) -/
def geqrfDTDoc (k nb : Nat) : Nat := (2 * k + roundUp32 k) * nb
/-- what zunmqr_gpu touches: T and R-inverse blocks, then nw × nb of larfb workspace -/
def unmqrDTUsed (k nw nb : Nat) : Nat := (2 * k + nw) * nb

/-- M261: whenever the other dimension of C exceeds `roundUp32 k`, zunmqr_gpu writes past a dT of the documented size,
    by exactly `(nw - roundUp32 k) · nb` elements (round 55 guard region: 72/72 cases, equal to the prediction) -/
theorem unmqr_gpu_dT_overflow (k nw nb : Nat) (h : roundUp32 k < nw) (hnb : 0 < nb) :
    geqrfDTDoc k nb < unmqrDTUsed k nw nb ∧
    unmqrDTUsed k nw nb - geqrfDTDoc k nb = (nw - roundUp32 k) * nb := by
  unfold geqrfDTDoc unmqrDTUsed
  constructor
  · exact Nat.mul_lt_mul_of_pos_right (by omega) hnb
  · rw [← Nat.sub_mul]; congr 1; omega

/-- the allocation MAGMA's own testing_zunmqr_gpu uses, `(2k + roundup(max(m, n), 32)) · nb`, always suffices -/
theorem unmqr_gpu_tester_size_suffices (k m n nb : Nat) (side_left : Bool) :
    unmqrDTUsed k (if side_left then n else m) nb ≤ (2 * k + roundUp32 (max m n)) * nb := by
  unfold unmqrDTUsed roundUp32
  apply Nat.mul_le_mul_right
  have h1 : max m n ≤ (max m n + 31) / 32 * 32 := by
    have := Nat.div_add_mod (max m n + 31) 32; have := Nat.mod_lt (max m n + 31) (show 0 < 32 by decide); omega
  cases side_left <;> simp <;> omega

/-- the measured instance m = 600, n = 70, Right (nw = 600), nb = 32, k = 35 (k = 37 gives the same): 17,152 elements written past the end -/
theorem unmqr_gpu_overflow_instance : unmqrDTUsed 35 600 32 - geqrfDTDoc 35 32 = 17152 := by decide

/-- M262: `zunmqr_2stage_gpu` hands larfb `MagmaLeft` whatever `side` is -/
def unmqr2sLarfbSide (_side : Bool) : Bool := true   -- true = Left
theorem unmqr_2stage_right_applied_left : unmqr2sLarfbSide false ≠ false := by decide
/-- … and its private workspace `n · nb` is read with `ldwork = nw = m` for Right: short whenever m > n -/
theorem unmqr_2stage_right_dwork_short (m n nb : Nat) (h : n < m) (hnb : 0 < nb) : n * nb < m * nb :=
  Nat.mul_lt_mul_of_pos_right h hnb

/-- M263: zgeqrf_gpu writes the T of block `i` only when `i + nb < minmn`; zunmqr_2stage_gpu reads the T of every block
    `i < k`.  For `k = minmn ≥ 1` the last block `(k - 1) / nb · nb` is read but never written. -/
def geqrfWritesT (minmn nb i : Nat) : Prop := i + nb < minmn
theorem unmqr_2stage_reads_unwritten_T (k nb : Nat) (hk : 0 < k) (hnb : 0 < nb) :
    (k - 1) / nb * nb < k ∧ ¬ geqrfWritesT k nb ((k - 1) / nb * nb) := by
  unfold geqrfWritesT
  have h1 := Nat.div_add_mod (k - 1) nb; have h2 := Nat.mod_lt (k - 1) hnb
  rw [Nat.mul_comm] at h1
  constructor <;> omega

/-! ### §56 `magma_[zcds]hetrf_aasen` / `sytrf_aasen` (round 56; M264)

The documentation describes `uplo = MagmaUpper` (A = U*D*U**H) next to Lower, but the argument check (zhetrf_aasen.cpp
l.109–112) is `if (uplo != MagmaLower) *info = MAGMA_ERR_NOT_IMPLEMENTED` (−117); the documented `−1` check is commented out. -/
def aasenStatus (upper : Bool) : Int := if upper then -117 else 0
theorem aasen_rejects_documented_upper : aasenStatus true = -117 ∧ aasenStatus false = 0 := by decide

/-! ### §57 `magma_[zcds]unmqr_batched` (round 57; M265)

The documentation lists SIDE = MagmaRight (C*Q, C*Q**H; "if SIDE = MagmaRight, N >= K >= 0", "LDDA >= max(1,N)"), but after
the argument checks the routine returns `-1` for any side other than Left (zunmqr_batched.cpp l.235–237); for small
sizes the internal `zunm2r_reg_batched` / `zunm2r_sm_batched` reject Right as well. -/
def unmqrBatchedStatus (left : Bool) : Int := if left then 0 else -1
theorem unmqr_batched_rejects_documented_right : unmqrBatchedStatus false = -1 ∧ unmqrBatchedStatus true = 0 := by decide

/-! ### §61 `magma_[zcds]potrf_vbatched` (round 61; M266, M267)

`magma_zpotrf_vbatched_max_nocheck` takes the large path `magma_zpotrf_lg_vbatched` when `max_n > crossover`
(Z 448, C 384, D 480, S 704, get_batched_crossover.cpp l.133–136), else `magma_zpotrf_lpout_vbatched`.
- M266: the large path calls the panel with `gbstep = 0` (zpotrf_vbatched.cpp l.173) instead of the panel offset `j`
  (the fixed-size zpotrf_batched passes `j`, l.154), so a non-positive pivot at global column c is reported as
  `c mod 128 + 1` (POTRF_NB = 128).
- M267: lpout needs `esize · (max_n + 8) · 8` bytes of shared memory and returns −33 if that exceeds 47000
  (zpotf2_kernels_var.cu l.102–108, POTF2_NB = 8) before any launch; for Z (16 bytes) that happens from max_n = 360,
  below the Z crossover 448, so 360 ≤ max_n ≤ 448 factors nothing and leaves info_array as it was. -/

def potrfLgReported (c : Nat) : Nat := c % 128 + 1
theorem potrf_vbatched_lg_info_local (c : Nat) (h : 128 ≤ c) : potrfLgReported c ≠ c + 1 := by
  unfold potrfLgReported; have := Nat.mod_lt c (show 0 < 128 by decide); omega
theorem potrf_vbatched_lg_info_first_panel (c : Nat) (h : c < 128) : potrfLgReported c = c + 1 := by
  unfold potrfLgReported; rw [Nat.mod_eq_of_lt h]
/-- the measured values: column 500 of n = 1000 gives 117; column 150 of a 200 x 200 matrix batched with n = 1000 gives 23 -/
theorem potrf_vbatched_lg_instances : potrfLgReported 500 = 117 ∧ potrfLgReported 150 = 23 ∧ potrfLgReported 999 = 104 := by decide

def lpoutShmem (esize n : Nat) : Nat := esize * (n + 8) * 8
def lpoutRejects (esize n : Nat) : Prop := 47000 < lpoutShmem esize n
instance (e n : Nat) : Decidable (lpoutRejects e n) := by unfold lpoutRejects; infer_instance
/-- Z (16-byte elements): lpout rejects exactly from n = 360 on, and the crossover 448 leaves 360..448 to it -/
theorem potrf_vbatched_z_gap : ¬ lpoutRejects 16 359 ∧ lpoutRejects 16 360 ∧ lpoutRejects 16 448 := by decide
theorem potrf_vbatched_z_gap_all (n : Nat) (h1 : 360 ≤ n) : lpoutRejects 16 n := by
  unfold lpoutRejects lpoutShmem; omega
/-- C (8 bytes, crossover 384), D (8, 480), S (4, 704): lpout accepts every n up to the crossover -/
theorem potrf_vbatched_cds_no_gap : ¬ lpoutRejects 8 384 ∧ ¬ lpoutRejects 8 480 ∧ ¬ lpoutRejects 4 704 := by decide

end MagmaAudit

#print axioms MagmaAudit.hegvdx_small_ignores_range
#print axioms MagmaAudit.geqp3_fixed_unreduced
#print axioms MagmaAudit.geqp3_doc_lwork_rejected
#print axioms MagmaAudit.geqrf_singular_block_used
#print axioms MagmaAudit.stedx_info_erased
#print axioms MagmaAudit.lag2s_nan_passes
#print axioms MagmaAudit.gels_lwork_too_small
#print axioms MagmaAudit.hegvx_leak_512
#print axioms MagmaAudit.hegvd_alloc_error_positive
#print axioms MagmaAudit.geqrs3_argnums_wrong
#print axioms MagmaAudit.rangev_unscaled_bounds
#print axioms MagmaAudit.heevdx_2stage_stedx_lost
#print axioms MagmaAudit.unmtr_m_not_implemented_lost
#print axioms MagmaAudit.he2hb_small_lwork_underflow
#print axioms MagmaAudit.geqrf_m_ngpu0_accepted
#print axioms MagmaAudit.cgeqrsv_nan_accepted
#print axioms MagmaAudit.gglse_singular_unreported
#print axioms MagmaAudit.ggrqf_lwork_inconsistent
#print axioms MagmaAudit.dlascl_nan_rejected
#print axioms MagmaAudit.hegvdx_small_all
#print axioms MagmaAudit.geqp3_fixed_never_reduced
#print axioms MagmaAudit.stedx_last_large_erases
#print axioms MagmaAudit.rbt_batched_default_build_silent
#print axioms MagmaAudit.rbt_hip_refine_dropped
#print axioms MagmaAudit.rbt_hip_refine_irrelevant_all
#print axioms MagmaAudit.rbt_cuda_refine_kept
#print axioms MagmaAudit.ungtr_upper_shift_short
#print axioms MagmaAudit.ungtr_lower_shift_wrong
#print axioms MagmaAudit.ungtr_first_col_stale
#print axioms MagmaAudit.check_info_build_broken
#print axioms MagmaAudit.unmqr_gpu_dT_overflow
#print axioms MagmaAudit.unmqr_gpu_tester_size_suffices
#print axioms MagmaAudit.unmqr_gpu_overflow_instance
#print axioms MagmaAudit.unmqr_2stage_right_applied_left
#print axioms MagmaAudit.unmqr_2stage_right_dwork_short
#print axioms MagmaAudit.unmqr_2stage_reads_unwritten_T
#print axioms MagmaAudit.aasen_rejects_documented_upper
#print axioms MagmaAudit.unmqr_batched_rejects_documented_right
#print axioms MagmaAudit.potrf_vbatched_lg_info_local
#print axioms MagmaAudit.potrf_vbatched_lg_info_first_panel
#print axioms MagmaAudit.potrf_vbatched_lg_instances
#print axioms MagmaAudit.potrf_vbatched_z_gap
#print axioms MagmaAudit.potrf_vbatched_z_gap_all
#print axioms MagmaAudit.potrf_vbatched_cds_no_gap

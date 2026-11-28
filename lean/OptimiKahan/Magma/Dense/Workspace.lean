/-!
# MAGMA workspace sizes: queried size vs. accepted minimum vs. size actually used

Code-level models of MAGMA v2.10.0 from a systematic read of every `src/z*.cpp` driver with an `lwork`, `lrwork` or
`liwork` argument (83 files; HANDOFF §29.30).  For each driver three quantities were compared: the size returned by
a workspace query (Q), the minimum the argument check accepts (C), and the workspace the code and its sub-calls
actually need (U).  Each definition transcribes the cited source lines; the theorems state what the caller sees.
Evidence is code reading only (the GPU node was unavailable); the concrete numbers use the `nb` functions of
`control/get_nb.cpp`.  Core Lean only.

The five lwork items found earlier (`geqp3_doc_lwork_rejected`, `gels_lwork_too_small`,
`he2hb_small_lwork_underflow`, `ggrqf_lwork_inconsistent`, `geqrf_ooc_lwork_short`) live in `StatusAudit` and
`QrEigen`; this module holds the new ones.
-/

namespace MagmaAudit

/-! ## 32a. `zhegvdx_2stage(_m)`: its own minimum is smaller than what the inner `zheevdx_2stage` requires -/

/-- `zhegvdx_2stage.cpp` l.308–319 (complex, no vectors): `lwmin = lq2 + n + n*nb`, with `lq2 = 2*n*2` for
    `wantz = 0` (`magma_get_zbulge_lq2`).  The value is also what the workspace query returns (l.331). -/
def hegvdxLwminNoVec (n nb : Nat) : Nat := 4 * n + n + n * nb
/-- `zheevdx_2stage.cpp` l.283–297 (complex, no vectors): `lwmin = lwstg2 + n + lwstg1`, with `lwstg2 = 2n + 0 + 2n`
    (`magma_zbulge_getstg2size`, `wantz = 0`) and `lwstg1 = 2*nb*n` (`magma_bulge_getlwstg1`). -/
def heevdxLwminNoVec (n nb : Nat) : Nat := 4 * n + n + 2 * nb * n
/-- With vectors both use the same stage-2 size `L` (`lq2 = lwstg2` when `wantz ≠ 0`), but the outer routine adds
    `n*n` and the inner one adds `max(2*nb*n, n*n)`. -/
def hegvdxLwminVec (L n : Nat) : Nat := L + 2 * n + n * n
def heevdxLwminVec (L n nb : Nat) : Nat := L + 2 * n + max (2 * nb * n) (n * n)

/-- `zhegvdx_2stage` passes its own `lwork` unchanged to `zheevdx_2stage` (l.427) after `zpotrf` and `zhegst` have
    overwritten B and A.  For `n > 128` (smaller n go to LAPACK) the inner check fails and its code −14 (argument 14 of
    `zheevdx_2stage`) is returned as the outer `info`; the outer routine's own lwork is argument 17.
    `(returned info, A and B already modified)` -/
def hegvdxRun (n nb lwork : Nat) : Int × Bool :=
  if n ≤ 128 then (0, true)
  else if lwork < heevdxLwminNoVec n nb then (-14, true) else (0, true)

/-- smallest instance: n = 129, nb = 64 — the queried size 8,901 is accepted by the outer check and rejected by the
    inner one, which needs 17,157 -/
theorem hegvdx_2stage_query_rejected :
    hegvdxLwminNoVec 129 64 = 8901 ∧ heevdxLwminNoVec 129 64 = 17157 ∧
    hegvdxRun 129 64 (hegvdxLwminNoVec 129 64) = (-14, true) := by decide

/-- general form (no vectors): for every n > 128 and nb > 0, calling with the size the routine itself reports
    fails, with the inner routine's argument number, after A and B have been overwritten -/
theorem hegvdx_2stage_query_rejected_all (n nb : Nat) (hn : 128 < n) (hnb : 0 < nb) :
    hegvdxLwminNoVec n nb < heevdxLwminNoVec n nb ∧
    hegvdxRun n nb (hegvdxLwminNoVec n nb) = (-14, true) := by
  have h1 : n * nb < 2 * nb * n := by
    have : 0 < n * nb := Nat.mul_pos (by omega) hnb
    rw [Nat.mul_assoc, Nat.mul_comm nb n]; omega
  have hlt : hegvdxLwminNoVec n nb < heevdxLwminNoVec n nb := by
    unfold hegvdxLwminNoVec heevdxLwminNoVec; omega
  refine ⟨hlt, ?_⟩
  have hn' : ¬ n ≤ 128 := by omega
  simp [hegvdxRun, hn', hlt]

/-- general form (with vectors): the outer minimum is short exactly when `n < 2*nb`; with `nb = 128`
    (`magma_get_zbulge_nb` when more than 14 threads) that is every n in [129, 255] -/
theorem hegvdx_2stage_vec_short_iff (L n nb : Nat) (hn : 0 < n) :
    hegvdxLwminVec L n < heevdxLwminVec L n nb ↔ n < 2 * nb := by
  unfold hegvdxLwminVec heevdxLwminVec
  constructor
  · intro h
    by_cases hc : 2 * nb * n ≤ n * n
    · rw [Nat.max_eq_right hc] at h; omega
    · exact Nat.lt_of_mul_lt_mul_right (Nat.lt_of_not_le (fun h' => hc (by rw [Nat.mul_comm n n] at h' ⊢; exact h')))
  · intro h
    have : n * n < 2 * nb * n := Nat.mul_lt_mul_of_pos_right h hn
    rw [Nat.max_eq_left (Nat.le_of_lt this)]; omega

theorem hegvdx_2stage_vec_instance : hegvdxLwminVec 0 129 < heevdxLwminVec 0 129 128 := by decide

/-- real precision (`dsygvdx_2stage.cpp` l.326 vs `dsyevdx_2stage.cpp` l.309, no vectors): the outer minimum is
    `2n + n*nb` (no stage-2 term at all, although the documentation says `LQ2 + 2N + N*NB`), the inner one
    `lwstg2 + 2n + lwstg1 = 4n + 2n + 2*nb*n`; the shortfall `4n + nb*n` holds for every n (it is reached for n > 128,
    below which the outer routine calls LAPACK) -/
def sygvdxLwminNoVec (n nb : Nat) : Nat := 2 * n + n * nb
def syevdxLwminNoVec (n nb : Nat) : Nat := 4 * n + 2 * n + 2 * nb * n
theorem sygvdx_2stage_real_short_all (n nb : Nat) (hn : 0 < n) :
    syevdxLwminNoVec n nb = sygvdxLwminNoVec n nb + 4 * n + nb * n ∧ sygvdxLwminNoVec n nb < syevdxLwminNoVec n nb := by
  unfold syevdxLwminNoVec sygvdxLwminNoVec
  have e : 2 * nb * n = n * nb + nb * n := by rw [Nat.mul_assoc, Nat.two_mul, Nat.mul_comm nb n]
  constructor <;> omega

/-! ## 32b. `zgesdd` path 6to-slow: the chunked VT buffer takes the space `zungbr` needs -/

/-- `zgesdd.cpp` l.2132–2145, 2193 (path 6to, `jobz = O`, m < n, slow branch taken when `lwork < m*n + wrkbl`):
    `chunk = (lwork − wrkbl)/m`, VT occupies `m*chunk` after the 2m tau entries, and `magma_zungbr(P)` gets the rest:
    `lnwork = lwork − 2m − m*chunk`.  The minimum (l.703–704) is `m*m + max(wrkbl, 2m + lwork_ungbr_P)`, but the
    chunk computation leaves `wrkbl − 2m + (lwork − wrkbl) mod m` regardless. -/
def gesddSlowLnwork (m wrkbl lwork : Nat) : Nat := lwork - 2 * m - m * ((lwork - wrkbl) / m)

/-- what remains for `zungbr` never exceeds `wrkbl − m − 1` -/
theorem gesdd_slow_lnwork_bound (m wrkbl lwork : Nat) (hm : 0 < m) (hw : wrkbl ≤ lwork) :
    gesddSlowLnwork m wrkbl lwork + m < wrkbl ∨ wrkbl < 2 * m := by
  unfold gesddSlowLnwork
  have hdm := Nat.div_add_mod (lwork - wrkbl) m
  have hlt := Nat.mod_lt (lwork - wrkbl) hm
  by_cases h2 : wrkbl < 2 * m
  · exact Or.inr h2
  · left; omega

/-- general form: whenever `wrkbl ≤ m + lwork_ungbr_P` (the gebrd-based `wrkbl` uses nb = 32, `zungbr(P)` needs
    m·`zgelqf_nb` = 128m once min(m,n) ≥ 1024), every lwork in the slow range gives `zungbr` too little, so it
    returns −9; `ierr` is never read and `zlarcm` then multiplies unformed reflectors, with `info = 0` -/
theorem gesdd_slow_ungbr_short_all (m wrkbl P lwork : Nat) (hm : 0 < m) (hw : wrkbl ≤ lwork)
    (h2 : 2 * m ≤ wrkbl) (hP : wrkbl ≤ m + P) : gesddSlowLnwork m wrkbl lwork < P := by
  rcases gesdd_slow_lnwork_bound m wrkbl lwork hm hw with h | h <;> omega

/-- instance: m = 1024, n = 1087, nb_gebrd = 32, nb_lq = 128: wrkbl = 2m + (m+n)·32 = 69,600, the minimum is
    1,181,696 < m·n + wrkbl = 1,182,688 (slow path), and `zungbr` gets 67,552 + r < 131,072 -/
theorem gesdd_6to_slow_instance :
    2 * 1024 + (1024 + 1087) * 32 = 69600 ∧ 1024 * 1024 + max 69600 (2 * 1024 + 1024 * 128) = 1181696 ∧
    1181696 < 1024 * 1087 + 69600 ∧ gesddSlowLnwork 1024 69600 1181696 < 1024 * 128 := by decide

/-! ## 32c. `zhetrd_he2hb_mgpu`: same unchecked lwork as the single-GPU twin -/

/-- `zhetrd_he2hb_mgpu.cpp` l.221: only `lwork ≥ 1` is checked; `hT = work + lwork − nb*nb` (l.285–287) and LAPACK
    `zgeqrf` gets `lwork − nb*nb` (l.347).  `(accepted, offset of hT, lwork given to zgeqrf)` -/
def he2hbMgpu (nb lwork : Nat) : Bool × Int × Int :=
  (1 ≤ lwork, (lwork : Int) - nb * nb, (lwork : Int) - nb * nb)

/-- general form: every accepted lwork below nb² puts hT before the start of `work` (host underrun) -/
theorem he2hb_mgpu_underrun_all (nb lwork : Nat) (h1 : 1 ≤ lwork) (h2 : lwork < nb * nb) :
    (he2hbMgpu nb lwork).1 = true ∧ (he2hbMgpu nb lwork).2.1 < 0 := by
  unfold he2hbMgpu; refine ⟨by simp; omega, by simp; omega⟩

/-- lwork = nb² − 1 hands zgeqrf lwork = −1, i.e. a workspace query: the panel is silently not factored -/
theorem he2hb_mgpu_silent_query (nb : Nat) (h : 2 ≤ nb) : (he2hbMgpu nb (nb * nb - 1)).2.2 = -1 := by
  unfold he2hbMgpu
  have : 1 ≤ nb * nb := Nat.mul_pos (by omega) (by omega)
  simp; omega

/-- independent review (HANDOFF §29.35): the workspace query itself is unsafe.  The query returns `n*nb`
    (l.228 mgpu, l.193 single-GPU, which does `lwork -= nb*nb; hT = work + lwork` at l.234–236), and the only early
    exit is `n = 0`; so for every 0 < n < nb the size the routine reports puts hT before `work` -/
theorem he2hb_query_underrun_all (n nb : Nat) (h0 : 0 < n) (h : n < nb) : (he2hbMgpu nb (n * nb)).2.1 < 0 := by
  have : n * nb < nb * nb := Nat.mul_lt_mul_of_pos_right h (by omega)
  unfold he2hbMgpu; simp; omega

/-! ## 32d. `zgbtrf_native(_work)`, `zgbsv_native(_work)`: `*info` is never cleared on entry -/

/-- `zgbtrf_gpu.cpp` l.27–38 and l.95–106 (and `zgbsv_gpu.cpp` l.26–39, 106): the argument checks assign `*info`
    only when an argument is bad, then `if (*info != 0) return`.  A caller's info variable that still holds a value
    from an earlier call makes the routine return at once.  `(returned info, factorization performed)` -/
def gbtrfNative (infoIn : Int) (argsOk : Bool) (argErr : Int) : Int × Bool :=
  let info := if argsOk then infoIn else argErr
  if info ≠ 0 then (info, false) else (0, true)

/-- general form: a stale nonzero info (e.g. a singular pivot index from the previous matrix) is returned again and
    nothing is factored, for every valid argument list -/
theorem gbtrf_native_stale_info_all (stale argErr : Int) (h : stale ≠ 0) :
    gbtrfNative stale true argErr = (stale, false) := by
  simp [gbtrfNative, h]

/-- `zgbsv_native_work` reports a short lwork as −10, but lwork is its 12th argument (−10 is `info`) -/
def gbsvNativeWorkLworkCode : Int × Nat := (-10, 12)
theorem gbsv_native_work_argnum : gbsvNativeWorkLworkCode.1 ≠ -(gbsvNativeWorkLworkCode.2 : Int) := by decide

/-! ## 32e. `zgbtrf_native_work` / `zgbsv_batched_work`: the query is skipped on the quick return -/

/-- `zgbtrf_batched.cpp` l.150–160: `if (m == 0 || n == 0 || batchCount == 0) return` comes before the size query,
    so a query with `lwork = −1` leaves −1 in place; `zgbtrf_native_work` then adds `batchCount*sizeof(magma_int_t)`
    and reports 7 bytes (ILP64) where its own info slot is placed at `device_work + (−1)`. -/
def gbtrfBatchedQuery (m n batch : Nat) (lworkIn need : Int) : Int :=
  if m = 0 ∨ n = 0 ∨ batch = 0 then lworkIn else need

theorem gbtrf_batched_query_unanswered (n batch : Nat) (need : Int) :
    gbtrfBatchedQuery 0 n batch (-1) need = -1 := by simp [gbtrfBatchedQuery]

/-! ## 32f. `zgeqlf`: the queried size is rejected when min(m,n) = 0 -/

/-- `zgeqlf.cpp` l.128–138: the query returns 1 when `min(m,n) = 0`, but the check `lwork < max(max(1,n), 2*nb*nb)`
    is applied regardless.  `(queried size, accepted?)` -/
def geqlfQuery (m n nb : Nat) : Nat := if min m n = 0 then 1 else max (n * nb) (2 * nb * nb)
def geqlfAccepts (n nb lwork : Nat) : Bool := decide (max (max 1 n) (2 * nb * nb) ≤ lwork)

/-- general form: for every nb > 0 and every empty shape, the size the routine reports is rejected (info = −7) -/
theorem geqlf_query_rejected_all (m n nb : Nat) (h0 : min m n = 0) (hnb : 0 < nb) :
    geqlfAccepts n nb (geqlfQuery m n nb) = false := by
  have h1 : 1 ≤ nb * nb := Nat.mul_pos hnb hnb
  have h2 : 2 ≤ 2 * nb * nb := by rw [Nat.mul_assoc]; omega
  simp [geqlfQuery, geqlfAccepts, h0]; omega

theorem geqlf_query_instance : geqlfQuery 0 5 64 = 1 ∧ geqlfAccepts 5 64 1 = false := by decide

/-! ## 32g. Documented minima that the argument check rejects -/

/-- each entry: routine, documented minimum, minimum accepted by the code, for the smallest triggering case
    (doc comments vs. checks; `nb` from `control/get_nb.cpp`):
    zgesvd m=10,n=100,jobvt=A (l.130–160 vs 752–799);  zgesdd path 4t m=1,n=2 (l.137–153 vs 603–722);
    zgehrd_m n=100,ngpu=1 (l.62 vs 153);  zungbr VECT=P m=2,n=3,k=2 (l.88 vs 153–161);
    zheevx n=1 (l.138);  zheevr n=1 (l.183);  zhegvx n=1 (l.163);  zhegvr n=1 (l.211);
    zstedx lrwork n=2 (l.84–105 vs 176–191);  zhegvdx_2stage n=1 (l.139 vs 313–317, no N ≤ 1 branch). -/
def docVsCheck : List (String × Nat × Nat) :=
  [("zgesvd", 3310, 6410), ("zgesdd", 67, 130), ("zgehrd_m", 3200, 6400), ("zungbr", 2, 128),
   ("zheevx", 1, 65), ("zheevr", 2, 65), ("zhegvx", 2, 65), ("zhegvr", 2, 65),
   ("zstedx", 2, 1 + 4 * 2 + 2 * 2 * 2), ("zhegvdx_2stage", 1, 3)]

/-- a caller who allocates exactly the documented size is rejected with a negative info in all ten routines -/
theorem doc_minimum_rejected_all : docVsCheck.all (fun e => decide (e.2.1 < e.2.2)) = true := by decide

/-! ## 32h. `zhegvr`: only `lwork = −1` counts as a query -/

/-- `zhegvr.cpp` l.301: `lquery = (lwork == −1)`, while the documentation also promises a query for `lrwork = −1` or
    `liwork = −1`; those are rejected (−23 / −25) -/
def hegvrIsQuery (lwork _lrwork _liwork : Int) : Bool := lwork == -1
theorem hegvr_lrwork_query_ignored : hegvrIsQuery 1000 (-1) 1000 = false := rfl

/-! ## 32i. `zgbtrf_native_work`: the device info slot is 4-byte aligned, `magma_int_t` may be 8 bytes -/

/-- `zgbtrf_batched.cpp` l.157–158: the generic workspace is `batchCount * sizeof(int)` bytes ("no need for
    magma_int_t here"); `zgbtrf_gpu.cpp` l.66 places `device_info` (a `magma_int_t*`, written by the kernels as
    `info_array[batchid]`) right after it.  For the native routine `batchCount = 1`, so the slot is at byte offset 4.
    In an ILP64 build (`sizeof(magma_int_t) = 8`) that 8-byte access is misaligned; the default LP64 build is fine. -/
def gbtrfInfoOffset (batch : Nat) : Nat := batch * 4
def alignedFor (bytes off : Nat) : Bool := off % bytes == 0

theorem gbtrf_native_info_misaligned_ilp64 : alignedFor 8 (gbtrfInfoOffset 1) = false := rfl
/-- general form: misaligned under ILP64 for every odd batch count, aligned under LP64 for every batch count -/
theorem gbtrf_info_alignment_all (b : Nat) :
    alignedFor 8 (gbtrfInfoOffset (2 * b + 1)) = false ∧ alignedFor 4 (gbtrfInfoOffset b) = true := by
  simp [alignedFor, gbtrfInfoOffset]; omega

end MagmaAudit

#print axioms MagmaAudit.hegvdx_2stage_query_rejected
#print axioms MagmaAudit.hegvdx_2stage_query_rejected_all
#print axioms MagmaAudit.hegvdx_2stage_vec_short_iff
#print axioms MagmaAudit.hegvdx_2stage_vec_instance
#print axioms MagmaAudit.gesdd_slow_lnwork_bound
#print axioms MagmaAudit.gesdd_slow_ungbr_short_all
#print axioms MagmaAudit.gesdd_6to_slow_instance
#print axioms MagmaAudit.he2hb_mgpu_underrun_all
#print axioms MagmaAudit.he2hb_mgpu_silent_query
#print axioms MagmaAudit.gbtrf_native_stale_info_all
#print axioms MagmaAudit.gbsv_native_work_argnum
#print axioms MagmaAudit.gbtrf_batched_query_unanswered
#print axioms MagmaAudit.geqlf_query_rejected_all
#print axioms MagmaAudit.geqlf_query_instance
#print axioms MagmaAudit.doc_minimum_rejected_all
#print axioms MagmaAudit.hegvr_lrwork_query_ignored
#print axioms MagmaAudit.gbtrf_native_info_misaligned_ilp64
#print axioms MagmaAudit.gbtrf_info_alignment_all
#print axioms MagmaAudit.he2hb_query_underrun_all
#print axioms MagmaAudit.sygvdx_2stage_real_short_all

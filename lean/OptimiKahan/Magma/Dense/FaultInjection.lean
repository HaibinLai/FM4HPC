import OptimiKahan.Magma.Dense.QrEigen

/-!
# MAGMA allocation-failure paths found by fault injection, and helpers that drop a sub-call status

Code-level models of MAGMA v2.10.0: each definition transcribes the control logic of the cited source lines, and the
theorems state the consequence the caller sees; the GPU 0 evidence is cited in the docstrings
(`experiments/magma_gpu/results/`).  Core Lean only.

Moved here from MagmaAudit.lean §21, MagmaAudit.lean §22, MagmaAudit.lean §26 when the two original files were split by subject; names and the
namespace `MagmaAudit` are unchanged.
-/

namespace MagmaAudit

/-! ## 21. Allocation-failure paths, found by fault injection (HANDOFF §29.10, `round10_results.txt`)

`experiments/magma_gpu/fail_malloc.c` makes the k-th `cudaMalloc`/`cudaHostAlloc` fail; the backtrace names the site. -/

/-- the common shape: stages run in sequence and each writes the shared `info` (or its own ignored `ierr`/`iinfo`);
    the caller sees the last stage's value.  `stages` = the statuses the stages would report, in order. -/
def pipelineInfo : List Int → Int
  | [] => 0
  | [s] => s
  | _ :: rest => pipelineInfo rest

/-- **an allocation failure in an early stage is replaced by the next stage's 0**.  Confirmed sites (GPU 0, all info 0
    with a wrong result):
    `zhegvd` — `zhegst_gpu`'s pinned workspace (zhegst_gpu.cpp l.135–137 return −112/−113; zhegvd.cpp l.324–325 calls
    zheevd next with the same `info`): eigenvalues of A instead of (A, B), relative error 599;
    `zhegvd_m` — `zhegst_m`'s device buffer, same consequence;
    `zheevdx_2stage` — `zhetrd_he2hb`'s device buffer (l.445 then the next stages), error 0.498;
    `zheevdx_2stage_m` — `dlaex0_m`'s device buffer inside `zstedx_m`, then `zbulge_back_m`/`zunmqr_m` reset `info`
    (this confirms `heevdx_2stage_stedx_lost`), error 1;
    `zgeev`, `zgeev_m` — `zgehrd(_m)`'s device buffer, `ierr` never read; zhseqr runs on a non-Hessenberg matrix, error
    0.067. -/
theorem early_alloc_failure_hidden : pipelineInfo [-113, 0] = 0 ∧ pipelineInfo [-112, 0, 0] = 0 := by decide

/-- `zheevd(_m)`, `zhegvd(_m)`: `zhetrd`'s `iinfo` (−113 from its device buffer, zhetrd.cpp l.214–216) is not read;
    dsterf/zstedx then run on the uninitialised tridiagonal, whose norm is NaN, and `dlascl` stops the process
    (`dlascl_nan_rejected`).  GPU 0: "DLASCL parameter number 4" and exit for zheevd k = 1, zheevd_m k = 1–3,
    zhegvd k = 6, zhegvd_m k = 3–5. -/
def hetrdThenScale (hetrdInfo : Int) : Option Rat := if hetrdInfo = 0 then some 1 else none   -- norm of d, e
theorem hetrd_failure_reaches_dlascl : dlasclArgOk (hetrdThenScale (-113)) = false := rfl

/-- `zgetrf_gpu` (l.508/573/638) and `zpotrf_gpu` (l.417) call `magma_malloc_pinned` without checking the result and
    then use the workspace: a failed pinned allocation is a NULL dereference, not an error code (GPU 0: segfault in
    zgesv, zposv and zhegvd at the pinned allocation).  `(returns an error code, dereferences the NULL workspace)`: -/
def pinnedUnchecked (allocOk : Bool) : Bool × Bool := if allocOk then (false, false) else (false, true)
theorem pinned_alloc_unchecked : pinnedUnchecked false = (false, true) := rfl

/-- `zgeqrf_ooc` (l.124): `lwork ≥ max(1, n)` is accepted, the documentation says `LWORK ≥ N·NB`, and the out-of-core
    loop uses `2·nb²` of it.  GPU 0 (m = 20000, n = 4096, nb = 128, 298 MB free so NB = 640 < n): lwork = 4096 ⇒
    segfault.  (With 19 MB free NB = 0, but the routine returns −113 instead of looping.) -/
theorem geqrf_ooc_lwork_short : 4096 < 2 * 128 * 128 ∧ 4096 < 4096 * 128 := by decide

/-! ## 22. Helpers that discard the sub-call's `iinfo` (HANDOFF §29.11, `round11_results.txt`) -/

/-- `zunmtr` (l.201/214), `zunmbr` (l.243–285), `zunghr` (l.133), and `zungtr` (l.154/172, not in the public header):
    the inner `zunmqr`/`zungqr` writes `iinfo`, the helper returns its own `*info`, which only carries argument errors -/
def helperInfo (_iinfo : Int) (argInfo : Int) : Int := argInfo

/-- **a failed allocation inside the inner routine is reported as success** (GPU 0, fault injection, n = 1000:
    zunmtr and zunmbr return C bit-identical to the input, zunghr returns a wrong Q, all with info 0; the failing
    allocation is in zunmqr resp. zungqr) -/
theorem helper_iinfo_dropped : helperInfo (-113) 0 = 0 := rfl
-- Also `zungbr` (l.195, 215, 224, 244: `magma_zungqr2(…, &iinfo)` / `magma_zunglq(…, &iinfo)`, never read), found by the
-- independent review; GPU 0 (round 19): an injected failure in zungqr2 (Q) / zunglq (P) ⇒ info 0, relative error 52 / 8.6.

/-! ## 26. `magma_zunglq`: header and definition disagree (HANDOFF §29.18, `round18_results.txt`) -/

/-- parameters 7 and 8 as `include/magma_z.h` l.1764 declares them and as `src/zunglq.cpp` l.83 defines them -/
def unglqDeclared : String × String := ("magmaDoubleComplex_ptr dT", "magma_int_t nb")
def unglqDefined : String × String := ("magmaDoubleComplex *work", "magma_int_t lwork")

/-- **a caller that follows the header passes a device buffer and a block size where the code expects a host workspace
    and its length** (GPU 0: the header-style call segfaults; the definition-style call is correct and reports an
    injected allocation failure as −113) -/
theorem unglq_header_mismatch : unglqDeclared ≠ unglqDefined := by decide

/-! ## General forms -/

theorem pipelineInfo_cons2 (a b : Int) (rest : List Int) : pipelineInfo (a :: b :: rest) = pipelineInfo (b :: rest) := rfl

/-- **in general**: whatever the earlier stages report, a final stage that succeeds makes the driver report its 0 -/
theorem early_failure_hidden_all (earlier : List Int) : pipelineInfo (earlier ++ [0]) = 0 := by
  induction earlier with
  | nil => rfl
  | cons a t ih =>
    obtain ⟨b, rest, h⟩ : ∃ b rest, t ++ [(0 : Int)] = b :: rest := by
      cases t with
      | nil => exact ⟨0, [], rfl⟩
      | cons b u => exact ⟨b, u ++ [0], rfl⟩
    rw [List.cons_append, h, pipelineInfo_cons2, ← h]; exact ih

/-- **helpers in general**: the helper's info never depends on the inner routine's status -/
theorem helper_iinfo_dropped_all (iinfo argInfo : Int) : helperInfo iinfo argInfo = argInfo := rfl

end MagmaAudit

#print axioms MagmaAudit.early_alloc_failure_hidden
#print axioms MagmaAudit.hetrd_failure_reaches_dlascl
#print axioms MagmaAudit.pinned_alloc_unchecked
#print axioms MagmaAudit.geqrf_ooc_lwork_short
#print axioms MagmaAudit.helper_iinfo_dropped
#print axioms MagmaAudit.unglq_header_mismatch
#print axioms MagmaAudit.early_failure_hidden_all
#print axioms MagmaAudit.helper_iinfo_dropped_all

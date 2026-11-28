
/-!
# MAGMA sparse Krylov solvers and preconditioner setup: acceptance logic

Code-level models of MAGMA v2.10.0: each definition transcribes the control logic of the cited source lines, and the
theorems state the consequence the caller sees; the GPU 0 evidence is cited in the docstrings
(`experiments/magma_gpu/results/`).  Core Lean only.

Moved here from MagmaSparse.lean §1, MagmaSparse.lean §2, MagmaSparse.lean §3, MagmaSparse.lean §4, MagmaSparse.lean §5, MagmaSparse.lean §6, MagmaSparse.lean §7, MagmaSparse.lean §8, MagmaSparse.lean §9, MagmaSparse.lean §10, MagmaSparse.lean §11, MagmaSparse.lean §12 when the two original files were split by subject; names and the
namespace `MagmaSparse` are unchanged.
-/

/- Original file header:

# MAGMA v2.10.0 sparse iterative solvers: acceptance logic, code-level models
Source: `sparse/src/*.cpp`, `sparse/blas/zmergecg.cu` (MAGMA v2.10.0).  Read-only audit; each theorem is a prediction.
* `zbicg.cpp` l.145–147: `if (magma_z_isnan_inf(rho_new)) { info = MAGMA_DIVERGENCE; break; }`, then l.197–198
  `if (numiter < maxiter) info = MAGMA_SUCCESS;` — the breakdown status is overwritten (same in `zpbicg`, `zptfqmr`,
  `zptfqmr_merge`).
* `zcg_merge.cpp` with `zmergecg.cu`: the merged kernels form `Σ d_i z_i` (l.232) and `Σ r_i r_i` (l.1258) without
  complex conjugation; `betanom = sqrt(|Σ r_i r_i|)` (l.155) is the stopping quantity, `betanom/nomb < rtol` (l.167).
  The initial `beta` uses the conjugated `zdotc`.  CG_MERGE is the default solver (`magma_zparse_opts`).
* `magma_z_solver_wrapper.cpp` l.141–142, 153–154: `default: printf(...); break;` and the wrapper returns 0.
* `zcg_res.cpp` l.106 (also `zbicgstab`, `zbicgstab_merge`, `zpbicgstab_merge`): early exit `if (nomb < r0)` tests ‖b‖,
  not the initial residual ‖b − A x0‖, with `r0 = max(rtol·‖b‖, dlamch('E'))`.
Complex numbers are modelled as Gaussian integers (enough for the counterexample); other values are rationals.

-/

namespace MagmaSparse


/-! ## 1. BiCG: a detected breakdown is reported as success -/

/-- the status logic of `zbicg` after the loop: `brk` = the NaN/Inf breakdown at l.145 fired (sets DIVERGENCE and
    breaks), `numiter` = iterations done.  `-202` = MAGMA_DIVERGENCE, `-201` = MAGMA_SLOW_CONVERGENCE, `0` = success.
    `decreased` = `init_res > final_res`. -/
def bicgInfo (brk : Bool) (numiter maxiter : Nat) (decreased : Bool) : Int :=
  let info : Int := if brk then -202 else 0
  if numiter < maxiter then 0                       -- l.197–198 overwrites whatever info was
  else if decreased then -201 else (if brk then info else -202)

/-- **a breakdown detected before the iteration limit returns MAGMA_SUCCESS** -/
theorem bicg_breakdown_success (numiter maxiter : Nat) (h : numiter < maxiter) (d : Bool) :
    bicgInfo true numiter maxiter d = 0 := by
  unfold bicgInfo; simp [h]

/-! ## 2. CG_MERGE in complex precision: an unconjugated "norm" -/

/-- Gaussian integers suffice for the counterexample (all intermediate values are integral) -/
structure C where
  re : Int
  im : Int
deriving DecidableEq

def C.add (a b : C) : C := ⟨a.re + b.re, a.im + b.im⟩
def C.sub (a b : C) : C := ⟨a.re - b.re, a.im - b.im⟩
def C.mul (a b : C) : C := ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩
def C.conj (a : C) : C := ⟨a.re, -a.im⟩
def C.ofInt (q : Int) : C := ⟨q, 0⟩

/-- `Σ a_i b_i` (unconjugated, as in the merged kernels) and `Σ conj(a_i) b_i` (`zdotc`) for 2-vectors -/
def dotu (a b : C × C) : C := C.add (C.mul a.1 b.1) (C.mul a.2 b.2)
def dotc (a b : C × C) : C := C.add (C.mul (C.conj a.1) b.1) (C.mul (C.conj a.2) b.2)

/-- `A = diag(a1, a2)` acting on a 2-vector -/
def diag2 (a1 a2 : Int) (x : C × C) : C × C := (C.mul (C.ofInt a1) x.1, C.mul (C.ofInt a2) x.2)
def axpy (s : C) (x y : C × C) : C × C := (C.add y.1 (C.mul s x.1), C.add y.2 (C.mul s x.2))

/-- the first iteration of `magma_zcg_merge` from `x = 0`, `r = d = b`, `beta = zdotc(b, b)`, on `A = diag(a1, a2)`,
    given the scalar `rho` that solves `rho·tmp1 = beta` (checked in the theorem):
    `z = A d; tmp1 = Σ d_i z_i; x += rho d; r −= rho z; beta' = Σ r_i r_i`.  Returns `(beta, tmp1, x, r, beta')`. -/
def cgMergeStep1 (a1 a2 : Int) (b : C × C) (rho : C) : C × C × (C × C) × (C × C) × C :=
  let z := diag2 a1 a2 b
  let x := axpy rho b (⟨0, 0⟩, ⟨0, 0⟩)
  let r := axpy (C.sub ⟨0, 0⟩ rho) z b
  (dotc b b, dotu b z, x, r, dotu r r)

/-- **false convergence**: `A = diag(2, 3)` (Hermitian positive definite), `b = (2, i)`.  `beta = zdotc(b,b) = 5`,
    `tmp1 = Σ b_i (A b)_i = 5` (unconjugated), so `rho = 1`, `x = b = (2, i)`, `r = (−2, −2i)`, and the unconjugated
    `Σ r_i² = 0`: `betanom = 0 < atol`, the solver stops with `numiter = 1 < maxiter`, i.e. `info = 0`.  The true
    residual has `‖r‖² = 8 > ‖b‖² = 5`, and the exact solution is `(1, i/3)` (`3·(i/3) = i`). -/
theorem cg_merge_false_convergence :
    cgMergeStep1 2 3 (⟨2, 0⟩, ⟨0, 1⟩) ⟨1, 0⟩ =
      (⟨5, 0⟩, ⟨5, 0⟩, (⟨2, 0⟩, ⟨0, 1⟩), (⟨-2, 0⟩, ⟨0, -2⟩), ⟨0, 0⟩) ∧
    C.mul ⟨1, 0⟩ ⟨5, 0⟩ = ⟨5, 0⟩ ∧
    dotc (⟨-2, 0⟩, ⟨0, -2⟩) (⟨-2, 0⟩, ⟨0, -2⟩) = ⟨8, 0⟩ ∧
    dotc (⟨2, 0⟩, ⟨0, 1⟩) (⟨2, 0⟩, ⟨0, 1⟩) = ⟨5, 0⟩ := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide

/-- the same recurrence on real data stored in complex format is ordinary CG (conjugation does nothing on reals) -/
theorem dotu_eq_dotc_real (a b c d : Int) :
    dotu (⟨a, 0⟩, ⟨b, 0⟩) (⟨c, 0⟩, ⟨d, 0⟩) = dotc (⟨a, 0⟩, ⟨b, 0⟩) (⟨c, 0⟩, ⟨d, 0⟩) := by
  simp [dotu, dotc, C.add, C.mul, C.conj]

/-! ## 3. Wrapper and early exits -/

inductive Solver where
  | cg | cgmerge | pcg | bicgstab | gmres | idr | ba | baorder | iterref | pardiso
deriving DecidableEq

/-- `magma_z_solver` (single RHS): the solver classes handled by the `switch`; others fall to `default:` -/
def handled : Solver → Bool
  | .ba | .baorder | .iterref | .pardiso => false
  | _ => true

/-- the wrapper's return value and whether `x` was touched -/
def wrapper (s : Solver) (solverInfo : Int) : Int × Bool :=
  if handled s then (solverInfo, true) else (0, false)   -- default: printf; break; return 0

/-- **options that `magma_zparse_opts` accepts but the wrapper does not handle return 0 and leave `x = x0`** -/
theorem wrapper_unhandled_success (s : Solver) (h : handled s = false) (i : Int) : wrapper s i = (0, false) := by
  unfold wrapper; simp [h]

/-- early exit of `zcg_res` / `zbicgstab*` (l.106/112/117/126): `if (nomb < r0) { info = 0; goto cleanup; }` with
    `r0 = max(rtol·nomb, atolerance)`; returns `(info, x_is_x0)`.  `nomb` = ‖b‖, not the initial residual. -/
def earlyExit (nomb rtol atolerance : Rat) : Option (Int × Bool) :=
  let r0 := if rtol * nomb < atolerance then atolerance else rtol * nomb
  if nomb < r0 then some (0, true) else none

/-- **‖b‖ below the floor ⇒ success with `x = x0`, whatever `x0` is** (the residual ‖b − A x0‖ is never looked at) -/
theorem early_exit_returns_x0 (nomb rtol atol : Rat) (hb : nomb < atol) (hr : rtol * nomb < atol) :
    earlyExit nomb rtol atol = some (0, true) := by
  unfold earlyExit; simp [hr, hb]


/-! ## 4. Post-loop status: the true residual is computed but not compared -/

/-- the post-loop status of the common template (e.g. `zcg_res.cpp` l.170–198): `final_res = ‖b − A x‖` is computed;
    success is decided by `numiter < maxiter` alone, or at the limit by the recurrence `iterRes` -/
def templateInfo (numiter maxiter : Nat) (initRes finalRes iterRes rtolNomb atol : Rat) : Int :=
  if numiter < maxiter then 0
  else if finalRes < initRes then (if iterRes < rtolNomb ∨ iterRes < atol then 0 else -201)
  else -202

/-- **stopping before the limit is success whatever the true residual is** -/
theorem template_ignores_final_res (numiter maxiter : Nat) (h : numiter < maxiter) (i f r t a : Rat) :
    templateInfo numiter maxiter i f r t a = 0 := by
  unfold templateInfo; simp [h]

/-- `magma_zjacobi` (l.133–136): success iff the true residual decreased, by any amount -/
def jacobiInfo (initRes finalRes : Rat) : Int := if finalRes < initRes then 0 else -202

/-- **Jacobi reports success for any decrease**, e.g. from `10^10` to `10^10 − 1` -/
theorem jacobi_any_decrease : jacobiInfo (10 ^ 10) (10 ^ 10 - 1) = 0 := by
  unfold jacobiInfo
  have : (10 : Rat) ^ 10 - 1 < 10 ^ 10 := by
    have h : (0 : Rat) < 1 := by decide
    grind
  simp [this]

/-! ## 5. LOBPCG: a NaN residual norm deactivates the vector as if converged -/

inductive Fv where
  | nan : Fv
  | num : Rat → Fv

def Fv.gt : Fv → Rat → Bool
  | .num a, t => decide (t < a)
  | .nan, _ => false

/-- `zcompact.cu` l.36: a vector stays active iff `dnorms[j] > tol && active[j]` -/
def stillActive (dnorm : Fv) (tol : Rat) (active : Bool) : Bool := dnorm.gt tol && active

/-- **NaN residual ⇒ the eigenvector is removed from the active set (treated as converged)** -/
theorem lobpcg_nan_converged (tol : Rat) (active : Bool) : stillActive .nan tol active = false := rfl

/-! ## 6. More status logic (each checked against the source) -/

/-- `zcg_res.cpp` l.111–114: after the early exit, `if (|den| <= 0) info = MAGMA_NONSPD` with `den = d^H A d`,
    `d = r0 = b − A x0`.  Returns `some info` for an early return. -/
def cgresSetup (nombSmall : Bool) (den : Rat) : Option Int :=
  if nombSmall then some 0 else if (if den < 0 then -den else den) ≤ 0 then some (-203) else none

/-- **an exact initial guess (r0 = 0, so den = 0) is reported as "matrix not SPD"**, even for SPD `A` -/
theorem cgres_exact_x0_nonspd : cgresSetup false 0 = some (-203) := by decide

/-- `zpcg_merge.cpp` l.225–236: at the iteration limit, SLOW is upgraded to SUCCESS if `iterRes/initRes < rtol`
    (relative to the INITIAL residual), whereas the loop test is relative to ‖b‖.  Residuals are given in units of
    `10⁻¹⁶` (natural numbers); `rtol = 1/rtolInv`, so `iterRes/initRes < rtol` is `iterRes·rtolInv < initRes`. -/
def pcgMergeInfo (numiter maxiter : Nat) (initRes finalRes iterRes atol rtolInv : Nat) : Int :=
  if numiter < maxiter then 0
  else if finalRes < initRes then (if iterRes < atol ∨ iterRes * rtolInv < initRes then 0 else -201)
  else -202

/-- the loop test of `zpcg_merge` (l.210): `res/nomb < rtol`, i.e. `res·rtolInv < nomb` -/
def pcgMergeLoopStops (res nomb rtolInv : Nat) : Bool := decide (res * rtolInv < nomb)

/-- **success with a residual 10⁵ times the requested tolerance**: `‖b‖ = 1`, a large `x0` with `‖r0‖ = 10⁶`; after
    `maxiter = 20` iterations `‖r‖ = 10⁻⁵`; default `rtol = 10⁻¹⁰`, `atol = 10⁻¹⁶`.  The loop never stopped
    (`10⁻⁵/‖b‖ > rtol`), but `10⁻⁵/10⁶ = 10⁻¹¹ < rtol` turns the result into SUCCESS. -/
theorem pcg_merge_upgrade :
    pcgMergeInfo 20 20 (10 ^ 22) (10 ^ 11) (10 ^ 11) 1 (10 ^ 10) = 0 ∧
    pcgMergeLoopStops (10 ^ 11) (10 ^ 16) (10 ^ 10) = false := by decide

/-- `zbpcg.cpp` l.224: the multi-RHS PCG stops on column 0 only -/
def bpcgStop (res0 nom0 rtol : Rat) (_resOthers : List Rat) : Bool := decide (res0 / nom0 < rtol)

/-- **column 0 converged ⇒ stop, whatever the other columns' residuals are** -/
theorem bpcg_col0 (res0 nom0 rtol : Rat) (others : List Rat) (h : res0 / nom0 < rtol) :
    bpcgStop res0 nom0 rtol others = true := by
  unfold bpcgStop; simp [h]

/-- `zidr.cpp` l.128–140: shadow-space size and iteration limit -/
def idrParams (restart maxiter n : Nat) : Nat × Nat :=
  let s := if restart != 50 then (if restart > n then n else restart) else 1
  (s, min (2 * n) maxiter)

/-- **asking for a shadow space of 50 gives 1** (50 is the sentinel for "unset"), and `maxiter` is silently capped at
    `2n` and written back into the caller's `solver_par` -/
theorem idr_params : idrParams 50 1000 1000 = (1, 1000) ∧ idrParams 4 1000 100 = (4, 200) := by decide

/-- `zbombard.cpp` l.622–625: `if (T_res < res) { res = Q_res; flag = 2; }` (copy-paste of the QMR line).
    Residuals in arbitrary integer units. -/
def bombardStep (res Q_res T_res : Nat) : Nat × Nat :=
  let (r1, f1) := if Q_res < res then (Q_res, 1) else (res, 0)
  if T_res < r1 then (Q_res, 2) else (r1, f1)

/-- **TFQMR is selected (`flag = 2`) but the recorded residual is QMR's, larger than the previous best** -/
theorem bombard_wrong_res : bombardStep 5 10 1 = (10, 2) := by decide

/-! ## 7. Preconditioner setup -/

/-- `blas/zilu.cpp` (`magma_zcumilusetup`, CUDA path): `cusparseXcsrilu02_zeroPivot` is called after the ANALYSIS
    and before the numerical factorization `cusparseZcsrilu02`, so it can only report a structural zero (it returns
    `CUSPARSE_STATUS_ZERO_PIVOT` and `CHECK_CUSPARSE` aborts the setup); a zero pivot that appears during the numerical
    factorization is never queried.  Returns the setup `info` (0 = success). -/
def iluSetup (structuralZero _numericalZero : Bool) : Int := if structuralZero then -1 else 0

/-- **a numerical zero pivot passes the ILU setup with `info = 0`** -/
theorem ilu_numerical_zero_unchecked : iluSetup false true = 0 := rfl

/-- `magma_z_precondsetup` (precond wrapper l.160–184): for PQMR, PQMRMERGE, PBICG, LSQR with ILU/ICC, the setup
    `info` is overwritten by the transpose setup -/
def precondSetupInfo (iluInfo transposeInfo : Int) (needsTranspose : Bool) : Int :=
  if needsTranspose then transposeInfo else iluInfo

/-- **an ILU setup failure is lost when the transpose setup succeeds** -/
theorem precond_setup_info_lost (iluInfo : Int) : precondSetupInfo iluInfo 0 true = 0 := rfl

/-- (GPU 0, 2026-11-25: for `A = [[0,1],[1,0]]` with the zero diagonal stored, `magma_z_precondsetup` crashed the
    process instead, so this composition was not observed end to end; HANDOFF §29.)
    **composition**: ILU with a numerical zero pivot passes setup; preconditioning then yields Inf/NaN, so PBiCG's
    first `rho` is NaN and it sets DIVERGENCE and breaks at `numiter = 1`; since `1 < maxiter`, the status is
    overwritten to SUCCESS (`zpbicg.cpp` l.155 → l.207, same logic as `bicgInfo`). -/
theorem ilu_zero_pivot_pbicg_success (maxiter : Nat) (hm : 1 < maxiter) (d : Bool) :
    iluSetup false true = 0 ∧ bicgInfo true 1 maxiter d = 0 :=
  ⟨rfl, bicg_breakdown_success 1 maxiter hm d⟩

/-! ## 8. Post-loop variants across all solver files (classification by `grep`, HANDOFF §28.8) -/

/-- the "fixed" post-loop (zcgs, zcgs_merge, zpcgs, zbicgstab_merge, zpbicgstab(_merge), zqmr(_merge), zpqmr(_merge),
    zfgmres, zbombard(_merge)): success before the limit requires `info == MAGMA_SUCCESS` from inside the loop; otherwise
    SLOW (−201) upgraded to SUCCESS only if the true residual decreased and the recurrence residual is small.
    `finalDecreased = (init_res > final_res)`, false when `final_res` is NaN. -/
def fixedInfo (numiter maxiter : Nat) (loopInfo : Int) (finalDecreased iterSmall : Bool) : Int :=
  if numiter < maxiter ∧ loopInfo = 0 then 0
  else if finalDecreased then (if iterSmall then 0 else -201) else -202

/-- **in the fixed variant a NaN breakdown stays DIVERGENCE** (contrast `bicg_breakdown_success`) -/
theorem fixed_breakdown_divergence (numiter maxiter : Nat) (b : Bool) :
    fixedInfo numiter maxiter (-202) false b = -202 := by
  unfold fixedInfo; simp

/-- the post-loop of the overwrite group that also detects breakdowns inside the loop: exactly
    `zbicg`, `zpbicg`, `zptfqmr`, `zptfqmr_merge` (the other overwrite-variant files have no in-loop breakdown test) -/
def overwriteWithBreakdown : List String := ["zbicg", "zpbicg", "zptfqmr", "zptfqmr_merge"]

-- GPU 0 (2026-11-25, all 32 solver classes with b[0] = NaN): exactly these four report info = 0 (HANDOFF §29.4)
theorem overwrite_group_size : overwriteWithBreakdown.length = 4 := rfl

/-! ## 9. LOBPCG: the parameter check is dead code (`zlobpcg.cpp` l.170–178) -/

/-- `solver_par->info = MAGMA_SUCCESS; if (m < 2) info = DIVERGENCE; else if (n > m) info = SLOW;
    if (solver_par->info != 0) { … goto cleanup; }` — the local `info` is set, the struct field is tested -/
def lobpcgQuickExit (m n : Nat) : Bool × Int :=
  let solverInfo : Int := 0
  let info : Int := if m < 2 then -202 else if n > m then -201 else 0
  (solverInfo != 0, info)

/-- **invalid sizes (a 1×1 matrix, or more eigenvectors than rows) never take the exit** (GPU 0: n_ev = 8 > m = 6 runs,
    the illegal size reaches `gegqr`, eigenvalues are NaN, status −201) -/
theorem lobpcg_check_dead (m n : Nat) : (lobpcgQuickExit m n).1 = false := rfl

/-! ## 10. Jacobi and IC setup -/

/-- the diagonal entry of a CSR row as found by `magma_zjacobisetup_diagscal` (`zjacobi.cpp`) -/
inductive DiagEntry where
  | missing : DiagEntry
  | stored : Rat → DiagEntry

/-- `diag.val[row] = 1.0/A.val[i]` for the stored entry (initial value 0); `1/0` is `+∞` -/
inductive Inv where
  | zero : Inv
  | inf : Inv
  | num : Rat → Inv

def invOf : DiagEntry → Inv
  | .missing => .zero
  | .stored a => if a = 0 then .inf else .num (1 / a)

/-- the setup rejects only `diag.val[row] == 0` -/
def jacobiRejects (d : DiagEntry) : Bool := match invOf d with | .zero => true | _ => false

/-- **a stored zero on the diagonal passes the check and yields an infinite scaling** (GPU 0, 2026-11-25: setup
    info = 0 and the scaling entry is NaN in complex precision, where `1/(0+0i)` is NaN; HANDOFF §29) -/
theorem jacobi_stored_zero_passes : jacobiRejects (.stored 0) = false ∧ invOf (.stored 0) = .inf := by
  simp [jacobiRejects, invOf]

/-- `magma_zcumiccsetup` (`blas/zilu.cpp`): `cusparseXcsric02_zeroPivot` before the numerical `cusparseZcsric02`,
    whose status is not checked at all; an incomplete-Cholesky breakdown (a non-positive pivot, which can happen for
    SPD matrices that are not M-matrices) is never reported -/
def iccSetup (structuralZero _numericalBreakdown : Bool) : Int := if structuralZero then -1 else 0

-- GPU 0 (2026-11-25): for a 6×6 SPD matrix whose IC(0) has a negative 6th pivot on the CPU, the setup returned 0
-- but PCG still converged (4 iterations, 1e-15): the unchecked status had no visible effect for that input.
theorem icc_breakdown_unchecked : iccSetup false true = 0 := rfl

/-! ## 11. Sparse FGMRES: a sticky success flag (`zfgmres.cpp` l.187–319) -/

/-- one restart cycle as seen by the status logic -/
inductive Cycle where
  | betaNaN : Cycle          -- l.199: NaN at the restart residual ⇒ info = DIVERGENCE, break out of the outer loop
  | atolHit : Cycle          -- l.286: betanom ≤ atol ⇒ info = SUCCESS (outer loop continues if rel_resid > rtol)
  | relHit : Cycle           -- l.286: rel_resid ≤ rtol ⇒ info = SUCCESS, outer loop ends
  | midNaN : Cycle           -- NaN inside the cycle ⇒ rel_resid = NaN, the outer test `rel_resid > rtol` is false: exit
  | plain : Cycle            -- no test fired, the outer loop restarts

/-- the `info` after the outer loop, starting from `MAGMA_NOTCONVERGED` (−205); `none` = keep restarting -/
def fgmresInfo : Int → List Cycle → Int
  | info, [] => info
  | _, .betaNaN :: _ => -202
  | _, .relHit :: _ => 0
  | info, .midNaN :: _ => info
  | _, .atolHit :: rest => fgmresInfo 0 rest
  | info, .plain :: rest => fgmresInfo info rest

/-- **an absolute-tolerance hit followed by a NaN inside a later cycle ends with SUCCESS** (and then the post-loop
    `numiter < maxiter && info == MAGMA_SUCCESS` reports success), while `x` was updated with NaN coefficients -/
theorem fgmres_sticky_success : fgmresInfo (-205) [.atolHit, .midNaN] = 0 := rfl

/-- without the earlier absolute hit the same NaN is not reported as success -/
theorem fgmres_nan_alone : fgmresInfo (-205) [.plain, .midNaN] = -205 := rfl

/-! ## 12. `magma_zmtransposeconjugate` recursion (control/magma_zmtranspose.cpp) — the ILU crash, GPU-confirmed -/

/-- the branch structure: `CSR ∧ DEV` ⇒ done; `CPU` ⇒ transfer to DEV and recurse; otherwise ⇒ convert to CSR (location
    unchanged) and recurse.  `Magma_CPU = 571`, `Magma_DEV = 572`.  With fuel: `none` = still recursing when fuel runs out. -/
def transposeConj (fuel : Nat) (csr : Bool) (loc : Nat) : Option Unit :=
  match fuel with
  | 0 => none
  | f + 1 =>
    if csr ∧ loc = 572 then some ()
    else if loc = 571 then transposeConj f true 572
    else transposeConj f true loc

/-- **a matrix whose `memory_location` is neither CPU nor DEV (e.g. 0, an un-built ILU factor) recurses forever**
    (GPU 0: `magma_z_precondsetup` with ILU and outer PBICG on a stored-zero diagonal ⇒ ILU setup fails (−3009, factor
    left with location 0), the failure is overwritten (`precond_setup_info_lost`), the transpose recurses until the
    stack overflows: segfault) -/
theorem transpose_unbuilt_diverges (fuel : Nat) (csr : Bool) (loc : Nat) (h1 : loc ≠ 571) (h2 : loc ≠ 572) :
    transposeConj fuel csr loc = none := by
  induction fuel generalizing csr with
  | zero => rfl
  | succ f ih => simp only [transposeConj, h1, h2, and_false, ite_false]; exact ih true

/-- the normal cases terminate within two calls -/
theorem transpose_cpu_terminates (csr : Bool) : transposeConj 2 csr 571 = some () := by
  simp [transposeConj]

/-! ## General forms -/

/-- **Jacobi in general**: every strict decrease of the true residual, however small, is reported as success -/
theorem jacobi_any_decrease_all (initRes finalRes : Rat) (h : finalRes < initRes) : jacobiInfo initRes finalRes = 0 := by
  simp [jacobiInfo, h]

/-- **IDR in general**: a requested shadow space of 50 always becomes 1, and the iteration limit is always capped at 2n -/
theorem idr_params_all (maxiter n : Nat) : idrParams 50 maxiter n = (1, min (2 * n) maxiter) := by
  simp [idrParams]

/-- **FGMRES in general**: once an absolute-tolerance hit has set info to SUCCESS, any later sequence of plain cycles
    ending in a NaN inside a cycle keeps SUCCESS, whatever the starting status was -/
theorem fgmres_sticky_success_all (start : Int) (plains : Nat) :
    fgmresInfo start (.atolHit :: (List.replicate plains .plain ++ [.midNaN])) = 0 := by
  show fgmresInfo 0 (List.replicate plains .plain ++ [.midNaN]) = 0
  induction plains with
  | zero => rfl
  | succ k ih => simpa [List.replicate_succ, fgmresInfo] using ih

end MagmaSparse

#print axioms MagmaSparse.bicg_breakdown_success
#print axioms MagmaSparse.cg_merge_false_convergence
#print axioms MagmaSparse.dotu_eq_dotc_real
#print axioms MagmaSparse.wrapper_unhandled_success
#print axioms MagmaSparse.early_exit_returns_x0
#print axioms MagmaSparse.template_ignores_final_res
#print axioms MagmaSparse.jacobi_any_decrease
#print axioms MagmaSparse.lobpcg_nan_converged
#print axioms MagmaSparse.cgres_exact_x0_nonspd
#print axioms MagmaSparse.pcg_merge_upgrade
#print axioms MagmaSparse.bpcg_col0
#print axioms MagmaSparse.idr_params
#print axioms MagmaSparse.bombard_wrong_res
#print axioms MagmaSparse.ilu_numerical_zero_unchecked
#print axioms MagmaSparse.precond_setup_info_lost
#print axioms MagmaSparse.ilu_zero_pivot_pbicg_success
#print axioms MagmaSparse.fixed_breakdown_divergence
#print axioms MagmaSparse.overwrite_group_size
#print axioms MagmaSparse.lobpcg_check_dead
#print axioms MagmaSparse.jacobi_stored_zero_passes
#print axioms MagmaSparse.icc_breakdown_unchecked
#print axioms MagmaSparse.fgmres_sticky_success
#print axioms MagmaSparse.fgmres_nan_alone
#print axioms MagmaSparse.transpose_unbuilt_diverges
#print axioms MagmaSparse.transpose_cpu_terminates
#print axioms MagmaSparse.jacobi_any_decrease_all
#print axioms MagmaSparse.idr_params_all
#print axioms MagmaSparse.fgmres_sticky_success_all

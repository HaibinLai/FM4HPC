import OptimiKahan.Magma.Fgmres

/-!
# MAGMA v2.10.0: the remaining mixed-precision dense solvers — code-level models and where they deviate

Source: MAGMA v2.10.0 (tag 752e63c), `src/`.  Line numbers refer to those files.

## Part A. The classical refinement driver shared by six routines

`zcgesv_gpu.cpp` (`magma_dsgesv_gpu`), `zcposv_gpu.cpp`, `zchesv_gpu.cpp`, `zcgeqrsv_gpu.cpp`, `zgerfs_nopiv_gpu.cpp`
and the classical branch of `dshposv_gmres_gpu.cpp` all follow the same skeleton:

* `cte = Anrm·eps·√n`; convert to single precision (range check → `iter = -2`, fallback); factor
  (failure → `iter = -3`, fallback); solve; FP64 residual;
* **norms by `i = i?amax(v); |v[i]|`** and the reject-form test **`if (Rnrm > Xnrm*cte) goto refinement`**;
* `for (iiter = 1; iiter < ITERMAX; )` refine, retest, `*iter = iiter` on success;
* otherwise `*iter = -ITERMAX-1` and fall back.

They differ in the fallback (Part A.3) and in which range checks exist (Part A.4):
* `zcgesv`, `zcposv`, `zcgeqrsv`, `dshposv` (classical): a real FP64 factorization + solve;
* `zchesv` l.371–381: the FP64 factorization and solve are **commented out**; only `zlacpy(B → X)` remains;
* `zgerfs_nopiv` l.263–266: the label `fallback:` is followed by nothing.
* `zcgesv` l.231–235: the conversion of B is commented out, so the following `if (*info != 0) { *iter = -2; … }` is
  dead; B is converted inside `zcgetrs_gpu` by `zclaswp` (`magmablas/zclaswp.cu`, "TODO check precision"), which has
  no range check.  `zcposv`, `zchesv`, `zcgeqrsv` do check B with `zlag2c`.

## Part B. `dfgmres_spd_gpu` (the FP16-Cholesky + FGMRES path of `magma_dshposv_gpu` / `_native`)

* Both public entry points call the expert routine with `use_gmres = 1`, and **return 0** (not `*info`),
  `dshposv_gmres_gpu.cpp` l.469, l.497.
* The GMRES branch (l.292–308) calls `magma_dfgmres_spd_gpu` and then **`goto cleanup`**: no residual recheck, no
  fallback, `info` stays 0, `iter = niters ≥ 0`.
* `dfgmres_spd_gpu.cpp` l.15/97: early exit **`if (fabs(ro) <= EPSILON)`** with `EPSILON = dlamch("Epsilon") = 2^-53`
  (the GMSTRS solver uses `1e-18`); l.110 `eps1 = tol·ro`; l.195 inner exit
  `i+1 >= restrt || ro <= eps1 || iters >= maxiter`; l.220 `*niters = iters`.
* l.133/142: each inner step converts `nrhs` columns starting at column `i` of `d_VV` and writes `nrhs` columns starting
  at column `i` of `d_w`, which has only `restrt` columns (l.66).

Modelling: exact rationals; `F` adds NaN and ±∞ with IEEE comparison semantics; FP32 conversion and the 1×1 FP32 solve
model **only overflow** (finite rounding is irrelevant to the statements below).  Each theorem is a prediction about
the library; GPU confirmation is listed in HANDOFF.
-/

namespace MagmaDense
open IrsContract

/-! ## A.1 Values with NaN and infinities -/

inductive F where
  | nan : F
  | pinf : F
  | ninf : F
  | num : Rat → F
deriving DecidableEq

/-- IEEE `a > b` (false whenever an operand is NaN, and `∞ > ∞` is false) -/
def F.gt : F → F → Bool
  | .num a, .num b => decide (b < a)
  | .pinf, .num _ => true
  | .pinf, .ninf => true
  | .num _, .ninf => true
  | _, _ => false

/-- `Xnrm*cte` with `cte > 0` -/
def F.scale (c : Rat) : F → F
  | .num a => .num (a * c)
  | x => x

/-- `|x|` -/
def F.abs : F → F
  | .num a => .num (ab a)
  | .ninf => .pinf
  | x => x

/-- the stopping test of every classical driver: `Rnrm > Xnrm*cte` means "not yet, keep refining" -/
def cont (cte : Rat) (Rnrm Xnrm : F) : Bool := F.gt Rnrm (F.scale cte Xnrm)

/-! ## A.2 The driver -/

/-- what the caller finds in `X` -/
inductive Out where
  | iterate : Nat → Out   -- the mixed-precision iterate after that many corrections
  | fp64 : Out            -- a full FP64 factorization and solve
  | copyB : Out           -- `X := B` (zchesv)
  | untouched : Out
deriving DecidableEq

/-- the three fallbacks found in the code -/
inductive Fb where
  | real : Fb   -- zcgesv, zcposv, zcgeqrsv, dshposv (classical)
  | stub : Fb   -- zchesv: the FP64 solve is commented out, only zlacpy(B → X) is left
  | none : Fb   -- zgerfs_nopiv: empty `fallback:` label
deriving DecidableEq

structure Res where
  iter : Int
  info : Int
  out : Out
deriving DecidableEq

/-- refinement loop `for (iiter = k; iiter < itermax; )`: the first `iiter` whose test passes -/
def loop (itermax : Nat) (cte : Rat) (R X : Nat → F) : Nat → Nat → Option Nat
  | 0, _ => none
  | fuel + 1, k =>
    if k < itermax then (if cont cte (R k) (X k) then loop itermax cte R X fuel (k + 1) else some k) else none

/-- the driver after a successful low-precision factorization.  `R k`, `X k`: the norms computed after `k`
    corrections; `fp64info`: the `info` of the FP64 factorization in the real fallback.  On entry to the fallback after
    the loop, `*info = 0` (it is reset at the top of every iteration). -/
def run (itermax : Nat) (fb : Fb) (fp64info : Int) (cte : Rat) (R X : Nat → F) : Res :=
  if !cont cte (R 0) (X 0) then ⟨0, 0, .iterate 0⟩ else
  match loop itermax cte R X itermax 1 with
  | some k => ⟨k, 0, .iterate k⟩
  | none =>
    match fb with
    | .real => ⟨-((itermax + 1 : Nat) : Int), fp64info, if fp64info = 0 then .fp64 else .untouched⟩
    | .stub => ⟨-((itermax + 1 : Nat) : Int), 0, .copyB⟩
    | .none => ⟨-((itermax + 1 : Nat) : Int), 0, .iterate (itermax - 1)⟩

theorem loop_none (itermax : Nat) (cte : Rat) (R X : Nat → F) (h : ∀ k, cont cte (R k) (X k) = true) :
    ∀ fuel k, loop itermax cte R X fuel k = none
  | 0, _ => rfl
  | fuel + 1, k => by
    simp only [loop, h k, ↓reduceIte]
    split
    · exact loop_none itermax cte R X h fuel (k + 1)
    · rfl

theorem loop_bound (itermax : Nat) (cte : Rat) (R X : Nat → F) :
    ∀ fuel k j, loop itermax cte R X fuel k = some j → k ≤ j ∧ j < itermax
  | 0, _, _, h => by simp [loop] at h
  | fuel + 1, k, j, h => by
    simp only [loop] at h
    split at h
    · split at h
      · have := loop_bound itermax cte R X fuel (k + 1) j h; omega
      · simp only [Option.some.injEq] at h; omega
    · simp at h

/-! ## A.3 Reject-form test: NaN and `∞ > ∞` are accepted -/

/-- the loop stops at the first index whose test passes -/
theorem loop_first (itermax : Nat) (cte : Rat) (R X : Nat → F) (j : Nat) (hj : j < itermax)
    (hcont : ∀ i, i < j → cont cte (R i) (X i) = true) (hstop : cont cte (R j) (X j) = false) :
    ∀ fuel k, k ≤ j → j - k < fuel → loop itermax cte R X fuel k = some j
  | 0, _, _, h => by omega
  | fuel + 1, k, hk, hf => by
    simp only [loop]
    have hki : k < itermax := by omega
    simp only [hki, ↓reduceIte]
    by_cases e : k = j
    · subst e; simp [hstop]
    · have hlt : k < j := by omega
      simp only [hcont k hlt, ↓reduceIte]
      exact loop_first itermax cte R X j hj hcont hstop fuel (k + 1) (by omega) (by omega)

/-- **a diverging refinement is reported as converged**: if every test up to correction `j − 1` says "keep refining"
    and the iterate then overflows (`Rnrm = Xnrm = +∞`), the driver returns `iter = j > 0` ("successfully used"),
    `info = 0`, with the non-finite iterate in `X`.  `zchesv_port.py` (seed 2) reaches this at `j = 29`. -/
theorem diverge_accepted (itermax : Nat) (fb : Fb) (f : Int) (cte : Rat) (R X : Nat → F) (j : Nat)
    (hj1 : 1 ≤ j) (hj : j < itermax) (hcont : ∀ i, i < j → cont cte (R i) (X i) = true)
    (hR : R j = .pinf) (hX : X j = .pinf) :
    run itermax fb f cte R X = ⟨j, 0, .iterate j⟩ := by
  have hstop : cont cte (R j) (X j) = false := by simp [cont, hR, hX, F.scale, F.gt]
  have h0 := hcont 0 (by omega)
  unfold run
  simp only [h0, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  rw [loop_first itermax cte R X j hj hcont hstop itermax 1 hj1 (by omega)]

/-- **NaN in the residual norm is accepted at the first test** (iter = 0, info = 0) -/
theorem accept_nan_R (itermax : Nat) (fb : Fb) (f : Int) (cte : Rat) (R X : Nat → F) (h : R 0 = .nan) :
    run itermax fb f cte R X = ⟨0, 0, .iterate 0⟩ := by
  unfold run; simp [cont, h, F.gt]

/-- **NaN in the solution norm is accepted at the first test** -/
theorem accept_nan_X (itermax : Nat) (fb : Fb) (f : Int) (cte : Rat) (R X : Nat → F) (h : X 0 = .nan) :
    run itermax fb f cte R X = ⟨0, 0, .iterate 0⟩ := by
  unfold run; simp only [cont, h, F.scale]; cases R 0 <;> simp [F.gt]

/-- **an infinite solution is accepted**: `Rnrm = Xnrm = +∞` makes the test `∞ > ∞·cte` false -/
theorem accept_inf (itermax : Nat) (fb : Fb) (f : Int) (cte : Rat) (R X : Nat → F)
    (hR : R 0 = .pinf) (hX : X 0 = .pinf) :
    run itermax fb f cte R X = ⟨0, 0, .iterate 0⟩ := by
  unfold run; simp [cont, hR, hX, F.scale, F.gt]

/-- a successful return never reports `iter = ITERMAX`: at most `ITERMAX − 1 = 29` corrections are tried
    (`for (iiter=1; iiter < ITERMAX; )`) -/
theorem success_iter_lt (itermax : Nat) (fb : Fb) (f : Int) (cte : Rat) (R X : Nat → F)
    (h : 0 ≤ (run itermax fb f cte R X).iter) : (run itermax fb f cte R X).iter < itermax ∨ itermax = 0 := by
  unfold run at h ⊢
  by_cases h0 : (!cont cte (R 0) (X 0)) = true
  · simp only [h0, ↓reduceIte]
    rcases Nat.eq_zero_or_pos itermax with e | e
    · right; exact e
    · left; simp; omega
  · simp only [h0, Bool.false_eq_true, ↓reduceIte] at h ⊢
    cases hl : loop itermax cte R X itermax 1 with
    | some j =>
      have := loop_bound itermax cte R X itermax 1 j hl
      simp only; left; omega
    | none =>
      rw [hl] at h
      cases fb <;> simp only at h <;> omega

/-- **zchesv: the stub fallback returns `X = B` with `info = 0`** when refinement does not converge.  The
    documentation says `iter < 0` means "double precision factorization has been performed". -/
theorem zchesv_stub (itermax : Nat) (f : Int) (cte : Rat) (R X : Nat → F) (h : ∀ k, cont cte (R k) (X k) = true) :
    run itermax .stub f cte R X = ⟨-((itermax + 1 : Nat) : Int), 0, .copyB⟩ := by
  unfold run; simp only [h 0, Bool.not_true, Bool.false_eq_true, ↓reduceIte, loop_none itermax cte R X h]

/-- **zgerfs_nopiv: no fallback at all**; the non-converged iterate is left in `X` with `info = 0` -/
theorem gerfs_nopiv_silent (itermax : Nat) (f : Int) (cte : Rat) (R X : Nat → F)
    (h : ∀ k, cont cte (R k) (X k) = true) :
    run itermax .none f cte R X = ⟨-((itermax + 1 : Nat) : Int), 0, .iterate (itermax - 1)⟩ := by
  unfold run; simp only [h 0, Bool.not_true, Bool.false_eq_true, ↓reduceIte, loop_none itermax cte R X h]

theorem run_none_info (itermax : Nat) (f : Int) (cte : Rat) (R X : Nat → F) :
    (run itermax .none f cte R X).info = 0 := by
  unfold run
  split
  · rfl
  · cases loop itermax cte R X itermax 1 <;> rfl

/-- `magma_zgesv_rbt(refine = true)` (`zgesv_rbt.cpp` l.155–163): the `info` of `zgesv_nopiv_gpu` (`info1`, e.g. an
    exactly zero pivot of the non-pivoted LU) is overwritten because `zgerfs_nopiv` starts with `*info = 0`; `iter` is a
    local variable, so the caller sees only the returned `info`. -/
def rbt (_info1 : Int) (r : Res) : Int := r.info

/-- **RBT hides both a failed refinement and a failed factorization**: whatever the non-pivoted LU reported, the
    caller gets `info = 0` -/
theorem rbt_info_lost (info1 : Int) (itermax : Nat) (f : Int) (cte : Rat) (R X : Nat → F) :
    rbt info1 (run itermax .none f cte R X) = 0 := run_none_info itermax f cte R X

/-! ## A.4 Concrete 1×1 instances: overflow of the single-precision solution or right-hand side -/

theorem p2_lt {a b : Int} (h : a < b) : p2 a < p2 b := by
  have h1 : p2 (a + 1) ≤ p2 b := p2_mono (by omega)
  have h2 : p2 (a + 1) = p2 a * 2 := by
    have e : p2 1 = 2 := by
      show p2 ((1 : Nat) : Int) = 2
      rw [p2_nat]; exact Rat.pow_one 2
    rw [p2_add, e]
  have := p2_pos a
  grind

/-- FP32 range: `2^128` is the overflow threshold (round-to-nearest overflows at `2^128·(1 − 2^-25)`; we only use
    values that are exact powers of two, so `≥ 2^128` overflows and `< 2^128` does not). -/
def ovf32 (q : Rat) : Bool := decide (p2 128 ≤ q)

/-- the single-precision 1×1 solve `x = b/a` for `a, b > 0`, modelling only overflow -/
def solve32 (a b : Rat) : F := if ovf32 b then .pinf else if p2 128 * a ≤ b then .pinf else .num (b / a)

/-- FP64 residual `b − a·x` for `a > 0`, and its absolute value -/
def resid (a b : Rat) : F → F
  | .num x => .num (b - a * x)
  | .pinf => .ninf
  | .ninf => .pinf
  | .nan => .nan

/-- the first test of a classical driver on a 1×1 system, `checkB` = whether the routine checks B's FP32 range
    (`zcposv`, `zchesv`, `zcgeqrsv`: yes; `zcgesv`: no).  Out-of-range B with the check ⇒ `iter = -2`, real fallback. -/
def drv1 (checkB : Bool) (a b cte : Rat) : Res :=
  if checkB && ovf32 b then ⟨-2, 0, .fp64⟩ else
  let x := solve32 a b
  run 30 .real 0 cte (fun _ => F.abs (resid a b x)) (fun _ => F.abs x)

/-- **a finite, FP32-representable system whose solution exceeds the FP32 range is accepted with `x = +∞`**:
    `a = 2^-100`, `b = 2^28` (both normal FP32 numbers, `2^-126 ≤ a`, `b < 2^128`), FP64 solution `2^128` (finite,
    `< 2^1024`).  This passes every range check in zcgesv and zcposv, because those checks look only at A and B. -/
theorem x_overflow_accepted (checkB : Bool) (cte : Rat) :
    drv1 checkB (p2 (-100)) (p2 28) cte = ⟨0, 0, .iterate 0⟩ ∧ solve32 (p2 (-100)) (p2 28) = .pinf ∧
    p2 (-126) ≤ p2 (-100) ∧ p2 28 < p2 128 ∧ p2 28 / p2 (-100) = p2 128 ∧ p2 128 < p2 1024 := by
  have hb : ovf32 (p2 28) = false := by
    unfold ovf32; simp only [decide_eq_false_iff_not]; exact fun h => absurd (p2_lt (show (28 : Int) < 128 by omega))
      (fun h' => by grind)
  have hq : p2 128 * p2 (-100) ≤ p2 28 := by rw [← p2_add]; exact Rat.le_refl
  have hs : solve32 (p2 (-100)) (p2 28) = .pinf := by unfold solve32; simp [hb, hq]
  refine ⟨?_, hs, p2_mono (by omega), p2_lt (by omega), ?_, p2_lt (by omega)⟩
  · unfold drv1; simp only [hb, Bool.and_false, Bool.false_eq_true, ↓reduceIte, hs]
    exact accept_inf 30 .real 0 cte _ _ rfl rfl
  · rw [Rat.div_def]
    have : (p2 (-100))⁻¹ = p2 100 := by
      have h := p2_inv 100
      have hn := p2_ne (-100)
      have : p2 100 * p2 (-100) * (p2 (-100))⁻¹ = (p2 (-100))⁻¹ := by rw [h, Rat.one_mul]
      rw [Rat.mul_assoc, Rat.mul_inv_cancel _ hn, Rat.mul_one] at this; exact this.symm
    rw [this, ← p2_add]; exact congrArg p2 (by decide)

/-- **zcgesv does not check B**: `a = 1`, `b = 2^200` (finite in FP64, out of FP32 range) is cast to `+∞` by
    `zclaswp`, the solve returns `+∞`, and the result is accepted with `iter = 0`, `info = 0` -/
theorem zcgesv_b_overflow_accepted (cte : Rat) : drv1 false 1 (p2 200) cte = ⟨0, 0, .iterate 0⟩ := by
  have hb : ovf32 (p2 200) = true := by unfold ovf32; simp only [decide_eq_true_eq]; exact p2_mono (by omega)
  unfold drv1; simp only [Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  have hs : solve32 1 (p2 200) = .pinf := by unfold solve32; simp [hb]
  rw [hs]; exact accept_inf 30 .real 0 cte _ _ rfl rfl

/-- **zcposv on the same input falls back correctly** (`zlag2c` on B, l.211–215) -/
theorem zcposv_b_overflow_fallback (cte : Rat) : drv1 true 1 (p2 200) cte = ⟨-2, 0, .fp64⟩ := by
  have hb : ovf32 (p2 200) = true := by unfold ovf32; simp only [decide_eq_true_eq]; exact p2_mono (by omega)
  unfold drv1; simp [hb]

/-! ## B. `dfgmres_spd_gpu` and `magma_dshposv_gpu` -/

open MagmaFgmres

/-- `magma_dfgmres_spd_gpu` from `x = 0`: same inner loop as the GMSTRS solver (`MagmaFgmres.inner`), early exit with
    `EPSILON = dlamch('Epsilon') = 2^-53` (l.15, l.97) -/
def fgmresSpd (maxiter : Nat) (tol : Rat) (ro0 : V) (ro : Nat → V) : Nat × Bool :=
  if V.le ro0 (.num (p2 (-53))) ∧ V.le (.num (-(p2 (-53)))) ro0 then (0, true)
  else (inner maxiter (V.mul (.num tol) ro0) ro maxiter 0, false)

/-- the public `magma_dshposv_gpu` / `magma_dshposv_native`: `(iter, info, return value)`; the expert routine's GMRES
    branch leaves `info = 0` and sets `iter = niters`; the wrappers `return 0` -/
def dshposv (maxiter : Nat) (tol : Rat) (ro0 : V) (ro : Nat → V) : Int × Int × Int :=
  ((fgmresSpd maxiter tol ro0 ro).1, 0, 0)

/-- **no failure path**: every call of the public `dshposv` reports success -/
theorem dshposv_always_success (maxiter : Nat) (tol : Rat) (ro0 : V) (ro : Nat → V) :
    0 ≤ (dshposv maxiter tol ro0 ro).1 ∧ (dshposv maxiter tol ro0 ro).2 = (0, 0) := by
  unfold dshposv; exact ⟨Int.natCast_nonneg _, rfl⟩

/-- **`0 ≤ ‖b‖₂ ≤ 2^-53` ⇒ `x = 0`, `iter = 0`** -/
theorem spd_tiny_rhs_returns_zero (maxiter : Nat) (tol r : Rat) (ro : Nat → V) (h1 : r ≤ p2 (-53)) (h2 : 0 ≤ r) :
    fgmresSpd maxiter tol (.num r) ro = (0, true) ∧ (dshposv maxiter tol (.num r) ro).1 = 0 := by
  have hc : V.le (.num r) (.num (p2 (-53))) ∧ V.le (.num (-(p2 (-53)))) (.num r) := by
    simp only [V.le]; have := p2_pos (-53); constructor <;> grind
  exact ⟨by simp [fgmresSpd, hc], by simp [dshposv, fgmresSpd, hc]⟩

/-- the SPD path's absolute threshold `2^-53 ≈ 1.1e-16` is larger than the GMSTRS path's `1e-18` -/
theorem spd_threshold_larger : 1 / 10 ^ 18 < p2 (-53) := by
  have h1 : (2 : Rat) ^ 53 < 10 ^ 18 := by
    have : ((2 ^ 53 : Nat) : Rat) < ((10 ^ 18 : Nat) : Rat) := Rat.natCast_lt_natCast.2 (by decide)
    rw [Rat.natCast_pow, Rat.natCast_pow] at this; exact this
  have hp : p2 53 = (2 : Rat) ^ 53 := p2_nat 53
  have h10 : (0 : Rat) < 10 ^ 18 := by
    have : ((0 : Nat) : Rat) < ((10 ^ 18 : Nat) : Rat) := Rat.natCast_lt_natCast.2 (by decide)
    rw [Rat.natCast_pow] at this; simpa using this
  have hinv := p2_inv 53
  -- 1/10^18 < p2(-53)  ⟺  1 < 10^18 · p2(-53), and 10^18·p2(-53) > 2^53·p2(-53) = 1
  have key : 1 < 10 ^ 18 * p2 (-53) := by
    have := Rat.mul_lt_mul_of_pos_right h1 (p2_pos (-53))
    rw [← hp, hinv] at this; exact this
  rw [Rat.div_def, Rat.one_mul]
  have hne : (10 : Rat) ^ 18 ≠ 0 := fun h => by rw [h] at h10; exact Rat.lt_irrefl h10
  have := Rat.mul_lt_mul_of_pos_left key (Rat.inv_pos.2 h10)
  rw [Rat.mul_one, ← Rat.mul_assoc, Rat.inv_mul_cancel _ hne, Rat.one_mul] at this
  exact this

/-- **NaN runs to the iteration limit and is reported as success** -/
theorem spd_nan_runs_to_maxiter (maxiter : Nat) (hm : 0 < maxiter) (tol : Rat) (ro : Nat → V) :
    dshposv maxiter tol .nan ro = ((maxiter : Int), 0, 0) := by
  have hne : ¬ (V.le V.nan (.num (p2 (-53))) ∧ V.le (.num (-(p2 (-53)))) V.nan) := by simp [V.le]
  have hall : ∀ i, ¬ V.le (ro i) (V.mul (.num tol) .nan) := by intro i; cases ro i <;> simp [V.le, V.mul]
  unfold dshposv fgmresSpd
  simp only [hne, ↓reduceIte]
  rw [inner_all_false maxiter _ ro hall maxiter 0 (by omega) hm]

/-! ## B.2 `nrhs > 1`: out-of-bounds device accesses -/

/-- columns of `d_w` written at inner step `i` by `slag2d(n, nrhs, dSX, n, &d_w[i*n2], n2)` (l.142) -/
def wcols (i nrhs : Nat) : List Nat := (List.range nrhs).map (i + ·)

/-- some column written is outside the `restrt` columns allocated for `d_w` (l.66) -/
def oobW (restrt i nrhs : Nat) : Prop := ∃ c ∈ wcols i nrhs, restrt ≤ c

/-- columns of `d_VV` read at step `i` by `dlag2s(n, nrhs, &d_VV[i*n2], n2, …)` (l.133); `d_VV` has `restrt + 1` -/
def oobV (restrt i nrhs : Nat) : Prop := ∃ c ∈ wcols i nrhs, restrt + 1 ≤ c

theorem oobW_last (restrt nrhs : Nat) (hr : 1 ≤ restrt) (hn : 2 ≤ nrhs) : oobW restrt (restrt - 1) nrhs :=
  ⟨restrt - 1 + 1, by simp [wcols]; omega, by omega⟩

theorem oobV_last (restrt nrhs : Nat) (hr : 1 ≤ restrt) (hn : 3 ≤ nrhs) : oobV restrt (restrt - 1) nrhs :=
  ⟨restrt - 1 + 2, by simp [wcols]; omega, by omega⟩

/-- **with `nrhs ≥ 2`, a run that does not converge before the restart length writes past `d_w`**: if no Givens
    estimate passes the test (e.g. they are NaN), the inner loop executes step `i = maxiter − 1` (the caller passes
    `restrt = maxiter`), whose write covers column `maxiter`, outside the `restrt` allocated columns -/
theorem nonconverged_oob (maxiter nrhs : Nat) (hm : 1 ≤ maxiter) (hn : 2 ≤ nrhs) (eps1 : V) (ro : Nat → V)
    (h : ∀ i, ¬ V.le (ro i) eps1) :
    inner maxiter eps1 ro maxiter 0 = maxiter ∧ oobW maxiter (maxiter - 1) nrhs :=
  ⟨inner_all_false maxiter eps1 ro h maxiter 0 (by omega) (by omega), oobW_last maxiter nrhs hm hn⟩

end MagmaDense

#print axioms MagmaDense.accept_nan_R
#print axioms MagmaDense.accept_nan_X
#print axioms MagmaDense.accept_inf
#print axioms MagmaDense.diverge_accepted
#print axioms MagmaDense.success_iter_lt
#print axioms MagmaDense.zchesv_stub
#print axioms MagmaDense.gerfs_nopiv_silent
#print axioms MagmaDense.rbt_info_lost
#print axioms MagmaDense.x_overflow_accepted
#print axioms MagmaDense.zcgesv_b_overflow_accepted
#print axioms MagmaDense.zcposv_b_overflow_fallback
#print axioms MagmaDense.dshposv_always_success
#print axioms MagmaDense.spd_tiny_rhs_returns_zero
#print axioms MagmaDense.spd_threshold_larger
#print axioms MagmaDense.spd_nan_runs_to_maxiter
#print axioms MagmaDense.nonconverged_oob

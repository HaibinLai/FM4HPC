import IrsContract

/-!
# MAGMA v2.10.0 mixed-precision FGMRES refinement (GMSTRS path): a code-level model and where it can deviate

Source (tag v2.10.0, 752e63c):
* `src/dxgesv_gmres_gpu.cpp`: `magma_dxgesv_gmres_gpu`.  The public `magma_dsgesv_iteref_gpu` (FP32 LU) and
  `magma_dhgesv_iteref_gpu` (FP16 tensor-core LU) both call it with `solver_type = Magma_REFINE_GMSTRS`.
* `src/dgmres_plu_gpu.cpp`: `magma_dfgmres_plu_gpu`.

What the GMSTRS branch does (line numbers of those files):
* l.406–446  `dlag2s` (overflow ⇒ `iter = -2`, fallback), low-precision `getrf` (`info ≠ 0` ⇒ `iter = -3`, fallback);
* l.490–493  `cte = Anrm·eps·√n` with `Anrm = ‖A‖∞`, `eps = dlamch('Epsilon')`, and `gmtol = cte`;
* l.570–600  `x := 0`; `magma_dfgmres_plu_gpu(…, maxiter = restrt = solver_outer_itermax = min(n, 500), tol = gmtol,
             …, iter)`; then **`goto cleanup`**: no residual check on the result, no fallback, `info` stays 0;
* FGMRES l.146–157 `ro = ‖b − A x‖₂`; **`if (fabs(ro − 0) <= EPSILON) break`** with `EPSILON = 1e-18` (l.36);
  l.161–165 at the first pass `eps1 = tol·ro`;
  l.172–275 inner loop, `iters++` each step, `ro = |rs[i+1]|` (Givens recurrence, l.247–261),
  **`if (i+1 >= restrt || ro <= eps1 || iters >= maxiter) in_flag = FALSE`** (l.272);
  l.293 `if (ro<=eps1 || iters >= maxiter) out_flag = FALSE`; l.297 `*niters = iters`.

The documented contract (same file, l.855–866 and l.941–952): stop when `RNRM < √N·XNRM·ANRM·EPS·BWDMAX`;
`iter > 0`: "iterative refinement has been successfully used", `iter < 0`: failure, double precision was used.

Proved here, each a candidate deviation derived from the code (exact `Rat` arithmetic; NaN modelled by `V.nan`,
on which every comparison is false as in IEEE 754; infinities are not modelled):
* `single_cycle`      : with `restrt = maxiter` (as passed by the caller) the outer FGMRES loop runs exactly once.
* `gmstrs_always_success` : after a successful low-precision factorization, the GMSTRS path returns `info = 0` and
                        `iter ≥ 0` on **every** run — there is no code path that reports failure or falls back.
* `nan_runs_to_maxiter` : NaN in the residual estimates (e.g. an overflowed low-precision LU) ⇒ the loop runs to
                        `maxiter` and returns `iter = maxiter > 0`.
* `enforced_rule`     : when it stops on the test, what holds is `|rs| ≤ (√n·eps·‖A‖∞)·‖b‖₂` — a recurrence estimate in
                        the 2-norm against `‖A‖·‖b‖`, not `‖b − A x‖∞ < √n·eps·‖A‖∞·‖x‖∞` (scales as `s²` vs `s`).
* `tol_ge_one_one_step` : if `‖A‖∞·eps·√n ≥ 1` the solver stops after one inner iteration (`iter = 1`), whatever
                        the preconditioner, because the Givens estimate never increases (`givens_nonincreasing`).
* `one_step_violates` : a 2×2 instance (`A = s·I`, FP16-level preconditioner error `2^-11`, `s = 2^60`) where that
                        one-step result is returned with `iter = 1`, `info = 0` and violates the documented condition.
* `tiny_rhs_returns_zero` : `0 < ‖b‖₂ ≤ 1e-18` ⇒ `x = 0` is returned with `iter = 0`; `tiny_violates` instantiates it
                        (`A = [[2^-70]]`, `b = [2^-70]`) and shows the documented postcondition fails.
These are predictions about the library; GPU confirmation is listed in HANDOFF.
-/

namespace MagmaFgmres
open IrsContract

/-! ## FP values with NaN -/

inductive V where
  | nan : V
  | num : Rat → V

/-- IEEE `a <= b`: false if either operand is NaN -/
def V.le : V → V → Prop
  | .num a, .num b => a ≤ b
  | _, _ => False

instance (a b : V) : Decidable (V.le a b) := by
  cases a <;> cases b <;> simp only [V.le] <;> infer_instance

def V.mul : V → V → V
  | .num a, .num b => .num (a * b)
  | _, _ => .nan

/-! ## The FGMRES inner loop (first cycle) -/

/-- inner loop of `magma_dfgmres_plu_gpu`, first cycle, with `restrt = maxiter`: index `i`, `iters = i + 1`;
    returns `iters` at exit.  `ro i` = `|rs[i+1]|` computed at inner step `i` (l.261);
    exit test l.272: `i+1 >= restrt || ro <= eps1 || iters >= maxiter`. -/
def inner (maxiter : Nat) (eps1 : V) (ro : Nat → V) : Nat → Nat → Nat
  | 0, i => i + 1
  | fuel + 1, i => if maxiter ≤ i + 1 ∨ V.le (ro i) eps1 then i + 1 else inner maxiter eps1 ro fuel (i + 1)

/-- the run of `magma_dfgmres_plu_gpu` on the GMSTRS path from `x = 0`:
    `ro0` = computed `‖b‖₂` (l.151–152), returns `(niters, early_exit)` -/
def fgmres (maxiter : Nat) (tol : Rat) (ro0 : V) (ro : Nat → V) : Nat × Bool :=
  if V.le ro0 (.num (1 / 10 ^ 18)) ∧ V.le (.num (-(1 / 10 ^ 18))) ro0 then (0, true)  -- fabs(ro-0) <= EPSILON
  else (inner maxiter (V.mul (.num tol) ro0) ro maxiter 0, false)

/-- the GMSTRS branch of `magma_dxgesv_gmres_gpu` after a successful factorization: `(iter, info)` -/
def gmstrs (maxiter : Nat) (tol : Rat) (ro0 : V) (ro : Nat → V) : Int × Int :=
  ((fgmres maxiter tol ro0 ro).1, 0)

theorem inner_bounds (maxiter : Nat) (eps1 : V) (ro : Nat → V) :
    ∀ fuel i, i + 1 ≤ inner maxiter eps1 ro fuel i ∧ inner maxiter eps1 ro fuel i ≤ max (i + 1) maxiter
  | 0, i => by simp only [inner]; omega
  | fuel + 1, i => by
    simp only [inner]
    split
    · omega
    · have := inner_bounds maxiter eps1 ro fuel (i + 1); omega

/-- the outer loop runs once: at the exit of the first inner loop, the outer test l.293 holds -/
theorem single_cycle (maxiter : Nat) (eps1 : V) (ro : Nat → V) :
    ∀ fuel i, i + fuel = maxiter →
      let k := inner maxiter eps1 ro fuel i
      V.le (ro (k - 1)) eps1 ∨ maxiter ≤ k
  | 0, i, h => by simp only [inner]; right; omega
  | fuel + 1, i, h => by
    simp only [inner]
    by_cases hc : maxiter ≤ i + 1 ∨ V.le (ro i) eps1
    · simp only [hc, ↓reduceIte]
      rcases hc with hc | hc
      · right; exact hc
      · left; simpa using hc
    · simp only [hc, ↓reduceIte]
      exact single_cycle maxiter eps1 ro fuel (i + 1) (by omega)

/-- **no failure path**: the GMSTRS branch reports success (`info = 0`, `iter ≥ 0`) on every run -/
theorem gmstrs_always_success (maxiter : Nat) (tol : Rat) (ro0 : V) (ro : Nat → V) :
    0 ≤ (gmstrs maxiter tol ro0 ro).1 ∧ (gmstrs maxiter tol ro0 ro).2 = 0 := by
  unfold gmstrs; exact ⟨Int.natCast_nonneg _, rfl⟩

theorem inner_all_false (maxiter : Nat) (eps1 : V) (ro : Nat → V) (h : ∀ i, ¬ V.le (ro i) eps1) :
    ∀ fuel i, i + fuel = maxiter → i < maxiter → inner maxiter eps1 ro fuel i = maxiter
  | 0, i, h1, h2 => by omega
  | fuel + 1, i, h1, h2 => by
    simp only [inner]
    by_cases hm : maxiter ≤ i + 1
    · simp only [hm, true_or, ↓reduceIte]; omega
    · have : ¬ (maxiter ≤ i + 1 ∨ V.le (ro i) eps1) := fun hc => hc.elim hm (h i)
      simp only [this, ↓reduceIte]
      exact inner_all_false maxiter eps1 ro h fuel (i + 1) (by omega) (by omega)

/-- the early exit (l.154) is not taken when `ro0` is NaN or larger than `1e-18` -/
theorem no_early (ro0 : V) (h : ro0 = .nan ∨ ∃ r, ro0 = .num r ∧ 1 / 10 ^ 18 < r) :
    ¬ (V.le ro0 (.num (1 / 10 ^ 18)) ∧ V.le (.num (-(1 / 10 ^ 18))) ro0) := by
  rcases h with h | ⟨r, h, hr⟩
  · subst h; simp [V.le]
  · subst h; simp only [V.le]; intro ⟨h1, _⟩; grind

/-- **NaN runs to the iteration limit and is reported as success**: if the computed initial residual norm is NaN
    (then `eps1 = tol·NaN = NaN`), or it is a number above `1e-18` and every recurrence estimate is NaN, the GMSTRS
    branch returns `iter = maxiter`, `info = 0` -/
theorem nan_runs_to_maxiter (maxiter : Nat) (hm : 0 < maxiter) (tol : Rat) (ro0 : V) (ro : Nat → V)
    (h : ro0 = .nan ∨ (∃ r, ro0 = .num r ∧ 1 / 10 ^ 18 < r) ∧ ∀ i, ro i = .nan) :
    gmstrs maxiter tol ro0 ro = ((maxiter : Int), 0) := by
  have hne : ¬ (V.le ro0 (.num (1 / 10 ^ 18)) ∧ V.le (.num (-(1 / 10 ^ 18))) ro0) := by
    rcases h with h | ⟨h, _⟩
    · exact no_early ro0 (Or.inl h)
    · exact no_early ro0 (Or.inr h)
  have hall : ∀ i, ¬ V.le (ro i) (V.mul (.num tol) ro0) := by
    intro i
    rcases h with h | ⟨_, h⟩
    · subst h; cases ro i <;> simp [V.le, V.mul]
    · rw [h i]; simp [V.le]
  unfold gmstrs fgmres
  simp only [hne, ↓reduceIte]
  rw [inner_all_false maxiter _ ro hall maxiter 0 (by omega) hm]

/-! ## What the stopping test actually enforces, and the `tol ≥ 1` case -/

/-- **the enforced rule**: on the normal path, FGMRES stops either at the iteration limit or with the Givens estimate
    `|rs| ≤ tol·‖b‖₂` where `tol = ‖A‖∞·eps·√n` (l.493, l.164) — i.e. `|rs| ≤ √n·eps·‖A‖∞·‖b‖₂`, compared with the
    documented `‖b − A x‖∞ < √n·eps·‖A‖∞·‖x‖∞`.  Under `(A, b) ↦ (s A, s b)` the left side scales by `s`, the enforced
    bound by `s²`, the documented bound by `s`. -/
theorem enforced_rule (maxiter : Nat) (tol : Rat) (ro0 : V) (ro : Nat → V)
    (hne : ¬ (V.le ro0 (.num (1 / 10 ^ 18)) ∧ V.le (.num (-(1 / 10 ^ 18))) ro0)) :
    let k := (fgmres maxiter tol ro0 ro).1
    V.le (ro (k - 1)) (V.mul (.num tol) ro0) ∨ maxiter ≤ k := by
  unfold fgmres; simp only [hne, ↓reduceIte]
  exact single_cycle maxiter _ ro maxiter 0 (by omega)

theorem ab_mul_self (x : Rat) : ab x * ab x = x * x := by unfold ab; split <;> grind

theorem ab_le_of_sq_le {x g : Rat} (hg : 0 < g) (h : x * x ≤ g * g) : ab x ≤ g := by
  by_cases hc : ab x ≤ g
  · exact hc
  · have h1 : g < ab x := by grind
    have h2 : g * g < ab x * ab x := by
      have := Rat.mul_lt_mul_of_pos_left h1 hg
      have := Rat.mul_lt_mul_of_pos_left h1 (show 0 < ab x by grind)
      grind
    rw [ab_mul_self] at h2; grind

theorem sq_nonneg' (x : Rat) : 0 ≤ x * x := by
  rw [← ab_mul_self]; exact mul_nonneg' (ab_nonneg x) (ab_nonneg x)

/-- **the Givens estimate never increases** (l.250–261): `rs[i+1] = −s_i·rs[i]` with `s_i = Hii1/gam`, and the `gam`
    actually used (`√(Hii²+Hii1²)`, or `EPSMAC = 1e-16` when that is `≤ 1e-18`) satisfies `gam² ≥ Hii1²` -/
theorem givens_nonincreasing {Hii1 gam rs : Rat} (hg : 0 < gam) (h : Hii1 * Hii1 ≤ gam * gam) :
    ab (-(Hii1 / gam) * rs) ≤ ab rs := by
  rw [Rat.neg_mul, ab_neg]
  generalize hq : Hii1 / gam = q
  have hqq : q * q ≤ 1 := by
    have hi := Rat.inv_pos.2 hg
    have hii : 0 ≤ gam⁻¹ * gam⁻¹ := mul_nonneg' (Rat.le_of_lt hi) (Rat.le_of_lt hi)
    have h1 := Rat.mul_le_mul_of_nonneg_left h hii
    have e1 : q * q = gam⁻¹ * gam⁻¹ * (Hii1 * Hii1) := by
      rw [← hq, Rat.div_def]; grind
    have hc := Rat.inv_mul_cancel gam (fun h0 => by rw [h0] at hg; exact Rat.lt_irrefl hg)
    have e2 : gam⁻¹ * gam⁻¹ * (gam * gam) = 1 := by
      calc gam⁻¹ * gam⁻¹ * (gam * gam) = (gam⁻¹ * gam) * (gam⁻¹ * gam) := by grind
        _ = 1 := by rw [hc, Rat.one_mul]
    rw [e1]; rw [e2] at h1; exact h1
  by_cases hrs : rs = 0
  · subst hrs; rw [Rat.mul_zero]; exact Rat.le_refl
  have hpos : 0 < ab rs := by
    have := ab_nonneg rs
    rcases Rat.le_iff_lt_or_eq.1 this with h' | h'
    · exact h'
    · exact absurd (ab_eq_zero h'.symm) hrs
  apply ab_le_of_sq_le hpos
  rw [ab_mul_self]
  have h3 := Rat.mul_le_mul_of_nonneg_left hqq (sq_nonneg' rs)
  have e3 : q * rs * (q * rs) = rs * rs * (q * q) := by grind
  rw [e3]; rw [Rat.mul_one] at h3; exact h3

/-- **`tol ≥ 1` ⇒ one step**: if `‖A‖∞·eps·√n ≥ 1` (e.g. n = 1024 and `‖A‖∞ ≥ 2^48`), the first estimate
    `|rs[1]| ≤ ‖b‖₂ ≤ tol·‖b‖₂ = eps1`, so FGMRES stops with `iter = 1`, whatever the low-precision factors are -/
theorem tol_ge_one_one_step (maxiter : Nat) (hm : 0 < maxiter) {tol r ρ : Rat} (ro : Nat → V)
    (htol : 1 ≤ tol) (hr : 1 / 10 ^ 18 < r) (h0 : ro 0 = .num ρ) (hρ : ρ ≤ r) :
    gmstrs maxiter tol (.num r) ro = (1, 0) := by
  have hne := no_early (.num r) (Or.inr ⟨r, rfl, hr⟩)
  unfold gmstrs fgmres
  simp only [hne, ↓reduceIte]
  obtain ⟨m, rfl⟩ : ∃ m, maxiter = m + 1 := ⟨maxiter - 1, by omega⟩
  have hle : V.le (ro 0) (V.mul (.num tol) (.num r)) := by
    rw [h0]; simp only [V.le, V.mul]
    have hr0 : 0 ≤ r := by grind
    have := Rat.mul_le_mul_of_nonneg_left htol hr0
    grind
  simp only [inner, hle, or_true, ↓reduceIte]
  rfl

/-! ## A concrete instance: the one-step result violates the documented condition -/

def d2 (a b : List Rat) : Rat := dot a b

/-- one FGMRES step from `x = 0`, written with the code's quantities in exact arithmetic (n = 2):
    `v0 = b/ro0` (l.160), `w0 = M⁻¹ v0` (l.179, the low-precision solve), `y = A w0` (l.213), `h00 = ⟨v0, y⟩`,
    `u = y − h00 v0` (l.216–222), `t² = ⟨u, u⟩` (l.224), Givens with `gam² = h00² + t²` (l.250),
    `rs[0] ← c0·ro0` (l.258), back substitution `rs[0]/HH[0][0]` with `HH[0][0] = c0·h00 + s0·t = gam` (l.260, 277),
    `x1 = (c0·ro0/gam)·w0 = (h00·ro0/gam²)·w0` (l.290).  The square roots cancel, so everything stays rational. -/
def oneStep (Minv : List (List Rat)) (A : List (List Rat)) (b : List Rat) (ro0 : Rat) : List Rat :=
  let v0 := smul (1 / ro0) b
  let w0 := mv Minv v0
  let y := mv A w0
  let h00 := d2 v0 y
  let u := vsub y (smul h00 v0)
  let t2 := d2 u u
  smul (h00 * ro0 / (h00 * h00 + t2)) w0

/-- the instance: `A = I`, `b = (3, 4)` (so `‖b‖₂ = 5` exactly), and a low-precision inverse with relative error
    `δ = 2^-11` (FP16 unit roundoff) in its second diagonal entry -/
def Aex : List (List Rat) := [[1, 0], [0, 1]]
def bex : List Rat := [3, 4]
def Mex : List (List Rat) := [[1, 0], [0, 1 + 1 / 2048]]

theorem oneStep_value : oneStep Mex Aex bex 5 = ([2185216 / 728633, 2915044 / 728633] : List Rat) := by
  simp only [oneStep, Mex, Aex, bex, d2, mv, smul, vsub, dot, List.map]
  congr 1
  · grind
  · congr 1; grind

theorem ab_of_nonneg {x : Rat} (h : 0 ≤ x) : ab x = x := by
  unfold ab; have : ¬ x < 0 := by grind
  simp only [this, ↓reduceIte]

theorem ab_of_neg {x : Rat} (h : x < 0) : ab x = -x := by unfold ab; simp only [h, ↓reduceIte]

theorem max_of_le' {x y : Rat} (h : x ≤ y) : max x y = y := by rw [Rat.max_def]; simp only [h, ↓reduceIte]

theorem max_of_ge' {x y : Rat} (h : y < x) : max x y = x := by
  rw [Rat.max_def]; have : ¬ x ≤ y := by grind
  simp only [this, ↓reduceIte]

/-- the one-step iterate `x1 = (2185216, 2915044)/728633` (true solution `(3, 4)`, relative error 2.3e-4) has residual
    `(683, −512)/728633`, so `‖r‖∞ = 683/728633 ≈ 9.4e-4` while `c·‖x1‖∞·‖A‖∞ ≤ 2^-51·4.0 ≈ 1.8e-15`: the documented
    condition fails for every `c ≤ 2^-51` (`√2·2^-53 < 2^-51`) -/
theorem oneStep_not_accept {c : Rat} (hc : c ≤ 1 / 2 ^ 51) : ¬ Accept c Aex bex (oneStep Mex Aex bex 5) := by
  rw [oneStep_value]
  unfold Accept
  have hr : ninf (vsub bex (mv Aex [2185216 / 728633, 2915044 / 728633])) = 683 / 728633 := by
    simp only [Aex, bex, mv, dot, vsub, ninf, List.map]
    have e1 : (3 : Rat) - (1 * (2185216 / 728633) + (0 * (2915044 / 728633) + 0)) = 683 / 728633 := by grind
    have e2 : (4 : Rat) - (0 * (2185216 / 728633) + (1 * (2915044 / 728633) + 0)) = -(512 / 728633) := by grind
    rw [e1, e2, ab_of_nonneg (by grind), ab_of_neg (by grind),
      show (-(-(512 / 728633)) : Rat) = 512 / 728633 by grind,
      max_of_ge' (show (0 : Rat) < 512 / 728633 by grind), max_of_ge' (show (512 : Rat) / 728633 < 683 / 728633 by grind)]
  have hx : ninf [2185216 / 728633, 2915044 / 728633] = 2915044 / 728633 := by
    simp only [ninf]
    rw [ab_of_nonneg (by grind), ab_of_nonneg (by grind),
      max_of_ge' (show (0 : Rat) < 2915044 / 728633 by grind), max_of_le' (by grind)]
  have ha : anorm Aex = 1 := by
    simp only [Aex, anorm, asum]
    rw [ab_of_nonneg (show (0 : Rat) ≤ 1 by grind), ab_of_nonneg (show (0 : Rat) ≤ 0 by grind),
      max_of_ge' (show (0 : Rat) < 0 + (1 + 0) by grind), max_of_le' (show (1 + (0 + 0) : Rat) ≤ 0 + (1 + 0) by grind)]
    grind
  rw [hr, hx, ha]
  intro h
  have hX : (0 : Rat) ≤ 2915044 / 728633 := by grind
  have := Rat.mul_le_mul_of_nonneg_left hc hX
  have h51 : (1 : Rat) / 2 ^ 51 = 1 / 2251799813685248 := by grind
  grind

/-- **the whole story for `A = s·I`, `s = 2^60`, n = 2**: `tol = ‖A‖∞·eps·√2 ≥ 2^60·2^-53 ≥ 1`, so GMSTRS stops after
    one step and returns `iter = 1`, `info = 0`; the returned `x1` (independent of `s`, since `w0` scales by `1/s`)
    violates the documented condition, by `accept_scale` from the unscaled instance -/
theorem one_step_violates {c : Rat} (hc : c ≤ 1 / 2 ^ 51) (iterRet : Int) (hiter : 0 ≤ iterRet) :
    ¬ Post c (smulM (p2 60) Aex) (smul (p2 60) bex) (oneStep Mex Aex bex 5) iterRet := by
  intro hpost
  have := (accept_scale c Aex bex (oneStep Mex Aex bex 5) (p2_pos 60)).1 (hpost hiter)
  exact oneStep_not_accept hc this

/-- and GMSTRS does return after one step there: `tol ≥ 1` (`2^60·2^-53 = 128`), so with any first estimate `≤ ‖b‖₂`
    the result is `(iter, info) = (1, 0)` -/
theorem one_step_returned (ro : Nat → V) {ρ : Rat} (h0 : ro 0 = .num ρ) (hρ : ρ ≤ 5 * p2 60) :
    gmstrs 2 (p2 60 * p2 (-53)) (.num (5 * p2 60)) ro = (1, 0) := by
  have h128 : p2 60 * p2 (-53) = 128 := by
    rw [← p2_add]; show p2 ((7 : Nat) : Int) = 128; rw [p2_nat]; grind
  have hbig : 1 / 10 ^ 18 < 5 * p2 60 := by
    have : p2 60 = p2 ((60 : Nat) : Int) := rfl
    rw [this, p2_nat]
    have := one_le_pow_nat 60; grind
  rw [h128]
  exact tol_ge_one_one_step 2 (by decide) ro (by grind) hbig h0 hρ

/-! ## Tiny right-hand sides: `x = 0` with `iter = 0` -/

/-- **`0 < ‖b‖₂ ≤ 1e-18` ⇒ `x = 0`, `iter = 0`**: the early `break` at l.154 leaves `x` as set by `dlaset` (l.570) -/
theorem tiny_rhs_returns_zero (maxiter : Nat) (tol r : Rat) (ro : Nat → V)
    (h1 : r ≤ 1 / 10 ^ 18) (h2 : 0 ≤ r) :
    fgmres maxiter tol (.num r) ro = (0, true) ∧ gmstrs maxiter tol (.num r) ro = (0, 0) := by
  have hc : V.le (.num r) (.num (1 / 10 ^ 18)) ∧ V.le (.num (-(1 / 10 ^ 18))) (.num r) := by
    simp only [V.le]; constructor <;> grind
  unfold gmstrs fgmres; simp only [hc]; exact ⟨rfl, rfl⟩

theorem p2_m70_tiny : p2 (-70) ≤ 1 / 10 ^ 18 := by
  apply le_div_of_mul_le (by grind)
  have h1 : (10 : Rat) ^ 18 ≤ p2 70 := by
    have : p2 70 = p2 ((70 : Nat) : Int) := rfl
    rw [this, p2_nat]
    have : ((10 ^ 18 : Nat) : Rat) ≤ ((2 ^ 70 : Nat) : Rat) := Rat.natCast_le_natCast.2 (by decide)
    rw [Rat.natCast_pow, Rat.natCast_pow] at this; exact this
  have h2 := p2_inv 70
  have := Rat.mul_le_mul_of_nonneg_left h1 (Rat.le_of_lt (p2_pos (-70)))
  grind

/-- the instance `A = [[2^-70]]`, `b = [2^-70]` (true solution 1): GMSTRS returns `x = 0`, `iter = 0`, and the
    documented postcondition fails for every `c` -/
theorem tiny_violates (c : Rat) (ro : Nat → V) :
    gmstrs 1 (p2 (-70) * p2 (-53)) (.num (p2 (-70))) ro = (0, 0) ∧
    ¬ Post c [[p2 (-70)]] [p2 (-70)] [0] 0 := by
  constructor
  · exact (tiny_rhs_returns_zero 1 _ _ ro p2_m70_tiny (Rat.le_of_lt (p2_pos _))).2
  · intro h
    have h' := h (Int.le_refl 0)
    unfold Accept at h'
    simp only [mv, dot, vsub, ninf, anorm, asum, List.map, ab] at h'
    have := p2_pos (-70)
    split at h' <;> grind

end MagmaFgmres

#print axioms MagmaFgmres.single_cycle
#print axioms MagmaFgmres.gmstrs_always_success
#print axioms MagmaFgmres.nan_runs_to_maxiter
#print axioms MagmaFgmres.enforced_rule
#print axioms MagmaFgmres.givens_nonincreasing
#print axioms MagmaFgmres.tol_ge_one_one_step
#print axioms MagmaFgmres.oneStep_not_accept
#print axioms MagmaFgmres.one_step_violates
#print axioms MagmaFgmres.one_step_returned
#print axioms MagmaFgmres.tiny_violates

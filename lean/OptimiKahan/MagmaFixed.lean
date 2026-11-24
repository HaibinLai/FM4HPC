import MagmaFgmres

/-!
# A fixed GMSTRS path for MAGMA's mixed-precision FGMRES refinement, and its proof

`MagmaFgmres.lean` models the GMSTRS branch of MAGMA v2.10.0 (`magma_dsgesv_iteref_gpu`, `magma_dhgesv_iteref_gpu`) and
shows the documented postcondition can fail: the FGMRES result is returned without a residual check, and the stopping
test is scale-dependent, NaN-unsafe and has an absolute threshold.  This file models the patch in
`experiments/magma_fgmres/fix/gmstrs_fix.patch` and proves it meets the documented contract.

**Design: verify the result, not the iteration.**  The patch keeps FGMRES as a black box and adds a tail:
1. reject a result with a non-finite entry;
2. compute the residual `b − A x` with a compensated dot product (Dot2, Ogita–Rump–Oishi 2005), and `‖A‖∞`, `‖x‖∞`,
   `‖b‖∞` (the last two are exact in floating point);
3. evaluate a test whose left side over-approximates, and right side under-approximates, the exact quantities in
   `IrsContract.wrapper_sound`; written `L < R`, so NaN makes it false;
4. pass ⇒ return `x` with `iter = niters ≥ 0`; fail ⇒ `goto fallback` (FP64 LU) with `iter < 0`, as documented.
The other changes (scale-free FGMRES tolerance, `ro == 0` instead of `ro ≤ 1e-18`, an FP16 range check before the
half-precision LU) only decide **how often** the test passes.  Soundness does not depend on them.

Proved here:
* `fixed_post`         : for **every** FGMRES outcome (any `niters`, any returned vector, NaN included) and every
                         computed `L`, `R` satisfying the error-bound hypothesis `Faithful`, the returned pair satisfies
                         the documented postcondition `Post c A b x iter` (`iter ≥ 0 ⇒ ‖b − A x‖∞ < c·‖x‖∞·‖A‖∞`).
* `fixed_nan_fallback` : a NaN in the returned vector, or in `L`/`R`, ⇒ `iter < 0` (contrast `nan_runs_to_maxiter`).
* `fixed_rejects`, `fixed_rejects_tiny`, `fixed_rejects_one_step` : an `x` that violates the documented condition is
                         never returned with `iter ≥ 0`; in particular not the two outputs of the original code
                         constructed in `MagmaFgmres` (`tiny_violates`, `one_step_violates`).
* `fixed_accepts_example` : the accepting branch is reachable (the proof is not vacuous): with the bounds for n = 1024,
                         an exact solution with a small computed residual is returned with `iter = niters`;
                         `example_faithful`, `example_post`: the hypothesis `Faithful` holds there, so `fixed_post` is
                         not vacuous either.
* `fgmresFixed_scale`  : with the scale-free tolerance and the exact-zero break, the FGMRES exit decision (`niters`)
                         is invariant under `(A, b) ↦ (s A, s b)` (the estimates scale by `s` in exact arithmetic);
                         the original's threshold scales by `s²` (`enforced_rule`).

Assumed, not proved here (hypothesis `Faithful`):
* the published error bound of Dot2: `|r_i − r̂_i| ≤ u|r_i| + γ_{n+1}²·(|b_i| + (|A||x|)_i)`, hence
  `‖r‖∞ ≤ (1+ε)·R̂ + γR·(‖b‖∞ + ‖A‖∞‖x‖∞)` with `ε = u/(1−u)`, `γR = γ_{n+1}²/(1−u)`;
* the FP64 row-sum bound `|Â − ‖A‖∞| ≤ γ_n·‖A‖∞`;
* the rounding of the few operations that form `L` and `R`, covered in the patch by the factors `(1 ± 16u)`.
Underflow is excluded by these bounds (as in the papers); Inf is not modelled (step 1 rejects it in the patch).
-/

namespace MagmaFixed
open IrsContract MagmaFgmres

/-- IEEE `a < b`: false if either operand is NaN -/
def V.lt : V → V → Prop
  | .num a, .num b => a < b
  | _, _ => False

instance (a b : V) : Decidable (V.lt a b) := by
  cases a <;> cases b <;> simp only [V.lt] <;> infer_instance

/-- step 1: the returned vector, if all entries are finite -/
def finite : List V → Option (List Rat)
  | [] => some []
  | .nan :: _ => none
  | .num q :: vs => (finite vs).map (q :: ·)

/-- constants of the test: `ε`, `γR`, `γA` (error bounds) and the documented `c = √n·eps·BWDMAX` -/
structure Consts where
  ε : Rat
  γR : Rat
  γA : Rat
  c : Rat

/-- the left and right sides of the certified test in exact arithmetic, from the computed residual norm `rh` and the
    computed `‖A‖∞` `ah` -/
def lhs (k : Consts) (B X rh ah : Rat) : Rat := (1 + k.ε) * rh + k.γR * (B + X * (ah / (1 - k.γA)))
def rhs (k : Consts) (X ah : Rat) : Rat := k.c * X * (ah / (1 + k.γA))

/-- **what the computed test values must satisfy**: whenever `L`, `R` are numbers, there are computed `rh`, `ah` with
    the Dot2 and row-sum error bounds, and `L`, `R` bound the exact sides from above and below -/
def Faithful (k : Consts) (A : List (List Rat)) (b x : List Rat) (L R : V) : Prop :=
  ∀ l r, L = .num l → R = .num r → ∃ rh ah,
    ninf (vsub b (mv A x)) ≤ (1 + k.ε) * rh + k.γR * (ninf b + anorm A * ninf x) ∧
    (1 - k.γA) * anorm A ≤ ah ∧ ah ≤ (1 + k.γA) * anorm A ∧
    lhs k (ninf b) (ninf x) rh ah ≤ l ∧ r ≤ rhs k (ninf x) ah

/-- **the patched tail** (after FGMRES returned `xf` and `niters`): finite and `L < R` ⇒ return `x`, `iter = niters`;
    otherwise the FP64 fallback result `xd` with `iter = −(niters + 1)` -/
def tail (niters : Nat) (xf : List V) (L R : V) (xd : List Rat) : List Rat × Int :=
  match finite xf with
  | some x => if V.lt L R then (x, niters) else (xd, -((niters : Int) + 1))
  | none => (xd, -((niters : Int) + 1))

theorem neg_succ_lt (n : Nat) : -((n : Int) + 1) < 0 := by omega

theorem ninf_nonneg : ∀ x : List Rat, 0 ≤ ninf x
  | [] => Rat.le_refl
  | a :: as => by simp only [ninf]; have := ab_nonneg a; grind

/-- the accepting branch: what it returns, and when -/
theorem tail_accept {niters : Nat} {xf : List V} {L R : V} {xd x : List Rat} {iter : Int}
    (h : tail niters xf L R xd = (x, iter)) (hi : 0 ≤ iter) :
    finite xf = some x ∧ V.lt L R ∧ iter = niters := by
  unfold tail at h
  split at h
  · rename_i x' hx
    by_cases hc : V.lt L R
    · simp only [hc, ↓reduceIte, Prod.mk.injEq] at h
      obtain ⟨h1, h2⟩ := h
      subst h1; exact ⟨hx, hc, h2.symm⟩
    · simp only [hc, ↓reduceIte, Prod.mk.injEq] at h
      have := neg_succ_lt niters; omega
  · simp only [Prod.mk.injEq] at h
    have := neg_succ_lt niters; omega

/-- **soundness of the fix**: whatever FGMRES did, the returned pair satisfies the documented postcondition -/
theorem fixed_post (k : Consts) (A : List (List Rat)) (b : List Rat) (niters : Nat) (xf : List V) (L R : V)
    (xd : List Rat) (hc : 0 ≤ k.c) (hγR : 0 ≤ k.γR) (hγA0 : 0 ≤ k.γA) (hγA1 : k.γA < 1)
    (hF : ∀ x, finite xf = some x → Faithful k A b x L R) :
    Post k.c A b (tail niters xf L R xd).1 (tail niters xf L R xd).2 := by
  intro hi
  have hacc := tail_accept (x := (tail niters xf L R xd).1) (iter := (tail niters xf L R xd).2) rfl hi
  obtain ⟨hx, hlt, _⟩ := hacc
  generalize (tail niters xf L R xd).1 = x at hx
  revert hlt
  cases hL : L <;> cases hR : R <;> simp only [V.lt, false_imp_iff]
  rename_i l r
  intro hlr
  obtain ⟨rh, ah, h1, h2, h3, h4, h5⟩ := hF x hx l r hL hR
  have htest : (1 + k.ε) * rh + k.γR * (ninf b + ninf x * (ah / (1 - k.γA))) < k.c * ninf x * (ah / (1 + k.γA)) := by
    unfold lhs at h4; unfold rhs at h5; grind
  exact wrapper_sound (ninf_nonneg x) hc hγR hγA1 h1 h2 h3 hγA0 htest

/-- **NaN is never reported as success** -/
theorem fixed_nan_fallback (niters : Nat) (xf : List V) (L R : V) (xd : List Rat)
    (h : finite xf = none ∨ L = .nan ∨ R = .nan) : (tail niters xf L R xd).2 < 0 := by
  have hn := neg_succ_lt niters
  unfold tail
  rcases h with h | h | h
  · simp only [h]; exact hn
  · split
    · subst h; simp only [V.lt, ↓reduceIte]; exact hn
    · exact hn
  · split
    · subst h; have : ¬ V.lt L .nan := by cases L <;> simp [V.lt]
      simp only [this, ↓reduceIte]; exact hn
    · exact hn

/-- **a wrong `x` is never returned with `iter ≥ 0`** -/
theorem fixed_rejects (k : Consts) (A : List (List Rat)) (b : List Rat) (niters : Nat) (xf : List V) (L R : V)
    (xd : List Rat) (hc : 0 ≤ k.c) (hγR : 0 ≤ k.γR) (hγA0 : 0 ≤ k.γA) (hγA1 : k.γA < 1)
    (hF : ∀ x, finite xf = some x → Faithful k A b x L R) {x : List Rat}
    (hx : finite xf = some x) (hbad : ¬ Accept k.c A b x) : (tail niters xf L R xd).2 < 0 := by
  have hp := fixed_post k A b niters xf L R xd hc hγR hγA0 hγA1 hF
  by_cases hi : 0 ≤ (tail niters xf L R xd).2
  · have := tail_accept (x := (tail niters xf L R xd).1) (iter := (tail niters xf L R xd).2) rfl hi
    rw [hx] at this
    obtain ⟨hxx, _, _⟩ := this
    simp only [Option.some.injEq] at hxx
    rw [hxx] at hbad
    exact absurd (hp hi) hbad
  · omega

/-- the original code's output for `A = [[2^-70]]`, `b = [2^-70]` (`x = 0`, `tiny_violates`) is rejected -/
theorem fixed_rejects_tiny (k : Consts) (niters : Nat) (L R : V) (xd : List Rat)
    (hc : 0 ≤ k.c) (hγR : 0 ≤ k.γR) (hγA0 : 0 ≤ k.γA) (hγA1 : k.γA < 1)
    (hF : Faithful k [[p2 (-70)]] [p2 (-70)] [0] L R) :
    (tail niters [.num 0] L R xd).2 < 0 := by
  have hfin : finite [V.num 0] = some [0] := rfl
  refine fixed_rejects k [[p2 (-70)]] [p2 (-70)] niters _ L R xd hc hγR hγA0 hγA1 (fun x hx => ?_) hfin ?_
  · rw [hfin] at hx; simp only [Option.some.injEq] at hx; subst hx; exact hF
  · intro h
    have := (tiny_violates k.c (fun _ => .nan)).2
    exact this (fun _ => h)

/-- the original code's one-step output for `A = 2^60·I`, `b = 2^60·(3, 4)` (`one_step_violates`) is rejected -/
theorem fixed_rejects_one_step (k : Consts) (niters : Nat) (L R : V) (xd : List Rat)
    (hc0 : 0 ≤ k.c) (hc : k.c ≤ 1 / 2 ^ 51) (hγR : 0 ≤ k.γR) (hγA0 : 0 ≤ k.γA) (hγA1 : k.γA < 1)
    (hF : Faithful k (smulM (p2 60) Aex) (smul (p2 60) bex) (oneStep Mex Aex bex 5) L R) :
    (tail niters ((oneStep Mex Aex bex 5).map .num) L R xd).2 < 0 := by
  have hfin : finite ((oneStep Mex Aex bex 5).map .num) = some (oneStep Mex Aex bex 5) := by
    rw [oneStep_value]; rfl
  refine fixed_rejects k (smulM (p2 60) Aex) (smul (p2 60) bex) niters _ L R xd hc0 hγR hγA0 hγA1
    (fun x hx => ?_) hfin ?_
  · rw [hfin] at hx; simp only [Option.some.injEq] at hx; subst hx; exact hF
  · intro h
    exact one_step_violates hc 0 (Int.le_refl 0) (fun _ => h)

/-! ## The accepting branch is reachable -/

/-- bounds for n = 1024 rounded up: `u = 2^-53`, `ε ≤ 2^-52`, `γR = γ_1025²/(1−u) ≤ 2^-84`, `γA = γ_1024 ≤ 2^-42`,
    `c = √1024·2^-53 = 2^-48` -/
def k1024 : Consts := { ε := 1 / 2 ^ 52, γR := 1 / 2 ^ 84, γA := 1 / 2 ^ 42, c := 1 / 2 ^ 48 }

/-- `A = I`, `b = (3, 4)`, FGMRES returned the exact solution, computed residual `0`, computed `‖A‖∞ = 1`; the code
    forms `L`, `R` and rounds them outward by `(1 ± 2^-49)` (= `1 ± 16u`): the test passes, `x` is returned with
    `iter = niters` -/
theorem fixed_accepts_example (niters : Nat) (xd : List Rat) :
    let L := lhs k1024 4 4 0 1 * (1 + 1 / 2 ^ 49)
    let R := rhs k1024 4 1 * (1 - 1 / 2 ^ 49)
    tail niters [.num 3, .num 4] (.num L) (.num R) xd = ([3, 4], (niters : Int)) := by
  intro L R
  have hlt : L < R := by
    simp only [L, R, lhs, rhs, k1024]
    grind
  have hfin : finite [V.num 3, V.num 4] = some [3, 4] := rfl
  unfold tail; rw [hfin]
  have : V.lt (.num L) (.num R) := hlt
  simp only [this, ↓reduceIte]

/-- and the hypothesis `Faithful` is satisfiable there (with the exact `rh = 0`, `ah = 1`), so `fixed_post` applies:
    the returned `x = (3, 4)` meets the documented postcondition -/
theorem example_faithful :
    Faithful k1024 Aex bex [3, 4] (.num (lhs k1024 4 4 0 1 * (1 + 1 / 2 ^ 49))) (.num (rhs k1024 4 1 * (1 - 1 / 2 ^ 49))) := by
  intro l r hl hr
  simp only [V.num.injEq] at hl hr
  subst hl hr
  have hr0 : ninf (vsub bex (mv Aex [3, 4])) = 0 := by
    simp only [Aex, bex, mv, dot, vsub, ninf, List.map]
    have e1 : (3 : Rat) - (1 * 3 + (0 * 4 + 0)) = 0 := by grind
    have e2 : (4 : Rat) - (0 * 3 + (1 * 4 + 0)) = 0 := by grind
    rw [e1, e2, ab_of_nonneg (by grind), max_of_le' (by grind), max_of_le' (by grind)]
  have hb : ninf bex = 4 := by
    simp only [bex, ninf]
    rw [ab_of_nonneg (by grind), ab_of_nonneg (by grind), max_of_ge' (show (0 : Rat) < 4 by grind), max_of_le' (by grind)]
  have hx : ninf [3, 4] = 4 := hb
  have ha : anorm Aex = 1 := by
    simp only [Aex, anorm, asum]
    rw [ab_of_nonneg (show (0 : Rat) ≤ 1 by grind), ab_of_nonneg (show (0 : Rat) ≤ 0 by grind),
      max_of_ge' (show (0 : Rat) < 0 + (1 + 0) by grind), max_of_le' (show (1 + (0 + 0) : Rat) ≤ 0 + (1 + 0) by grind)]
    grind
  refine ⟨0, 1, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hr0, hb, hx, ha]; simp only [k1024]; grind
  · rw [ha]; simp only [k1024]; grind
  · rw [ha]; simp only [k1024]; grind
  · rw [hb, hx]; simp only [lhs, k1024]; grind
  · rw [hx]; simp only [rhs, k1024]; grind

theorem example_post (niters : Nat) (xd : List Rat) :
    Post k1024.c Aex bex
      (tail niters [.num 3, .num 4] (.num (lhs k1024 4 4 0 1 * (1 + 1 / 2 ^ 49)))
        (.num (rhs k1024 4 1 * (1 - 1 / 2 ^ 49))) xd).1
      (tail niters [.num 3, .num 4] (.num (lhs k1024 4 4 0 1 * (1 + 1 / 2 ^ 49)))
        (.num (rhs k1024 4 1 * (1 - 1 / 2 ^ 49))) xd).2 :=
  fixed_post k1024 Aex bex niters _ _ _ xd (by simp only [k1024]; grind) (by simp only [k1024]; grind)
    (by simp only [k1024]; grind) (by simp only [k1024]; grind)
    (fun x hx => by
      have : x = [3, 4] := by simp only [finite, Option.map, Option.some.injEq] at hx; exact hx.symm
      subst this; exact example_faithful)

/-! ## The scale-free FGMRES exit decision -/

def V.smul (s : Rat) : V → V
  | .num a => .num (s * a)
  | .nan => .nan

theorem le_smul_iff {s : Rat} (hs : 0 < s) (a b : V) : V.le (V.smul s a) (V.smul s b) ↔ V.le a b := by
  cases a <;> cases b <;> simp only [V.smul, V.le] <;> first | exact mul_le_mul_left_iff hs | exact Iff.rfl

theorem inner_scale {s : Rat} (hs : 0 < s) (maxiter : Nat) (eps1 : V) (ro : Nat → V) :
    ∀ fuel i, inner maxiter (V.smul s eps1) (fun j => V.smul s (ro j)) fuel i = inner maxiter eps1 ro fuel i
  | 0, i => rfl
  | fuel + 1, i => by
    simp only [inner, le_smul_iff hs]
    rw [inner_scale hs maxiter eps1 ro fuel (i + 1)]

/-- the exact-zero test `ro == 0` (false on NaN) -/
def V.isZero : V → Prop
  | .num a => a = 0
  | .nan => False

instance (a : V) : Decidable (V.isZero a) := by cases a <;> simp only [V.isZero] <;> infer_instance

/-- FGMRES with the patch: `eps1 = tolRel·ro0` with a scale-free `tolRel` (`√n·eps·BWDMAX`), and a break only when
    the computed `‖b‖₂` is exactly zero (`x = 0` is then exact) -/
def fgmresFixed (maxiter : Nat) (tolRel : Rat) (ro0 : V) (ro : Nat → V) : Nat × Bool :=
  if V.isZero ro0 then (0, true) else (inner maxiter (V.mul (.num tolRel) ro0) ro maxiter 0, false)

theorem mul_smul_comm (t s : Rat) (v : V) : V.mul (.num t) (V.smul s v) = V.smul s (V.mul (.num t) v) := by
  cases v <;> simp only [V.smul, V.mul]; congr 1; grind

theorem isZero_smul {s : Rat} (hs : 0 < s) (v : V) : V.isZero (V.smul s v) ↔ V.isZero v := by
  cases v with
  | nan => exact Iff.rfl
  | num a =>
    simp only [V.smul, V.isZero]
    constructor
    · intro h
      rcases Rat.mul_eq_zero.1 h with h | h
      · subst h; exact absurd hs (Rat.lt_irrefl)
      · exact h
    · intro h; subst h; exact Rat.mul_zero s

/-- **scale invariance of the patched exit decision**: if `(A, b) ↦ (s A, s b)` scales the initial residual norm and
    every Givens estimate by `s > 0` (true in exact arithmetic, and with IEEE rounding for `s = 2^k` absent
    over/underflow, `IrsContract.rnd_scale`), the patched FGMRES takes the same number of steps -/
theorem fgmresFixed_scale {s : Rat} (hs : 0 < s) (maxiter : Nat) (tolRel : Rat) (ro0 : V) (ro : Nat → V) :
    fgmresFixed maxiter tolRel (V.smul s ro0) (fun j => V.smul s (ro j)) = fgmresFixed maxiter tolRel ro0 ro := by
  unfold fgmresFixed
  by_cases hz : V.isZero ro0
  · have : V.isZero (V.smul s ro0) := (isZero_smul hs ro0).2 hz
    simp only [hz, this, ↓reduceIte]
  · have : ¬ V.isZero (V.smul s ro0) := fun h => hz ((isZero_smul hs ro0).1 h)
    simp only [hz, this, ↓reduceIte, mul_smul_comm, inner_scale hs]

end MagmaFixed

#print axioms MagmaFixed.fixed_post
#print axioms MagmaFixed.fixed_nan_fallback
#print axioms MagmaFixed.fixed_rejects
#print axioms MagmaFixed.fixed_rejects_tiny
#print axioms MagmaFixed.fixed_rejects_one_step
#print axioms MagmaFixed.fixed_accepts_example
#print axioms MagmaFixed.example_faithful
#print axioms MagmaFixed.example_post
#print axioms MagmaFixed.fgmresFixed_scale

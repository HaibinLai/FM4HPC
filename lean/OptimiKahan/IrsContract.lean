/-!
# The convergence contract of mixed-precision iterative refinement, a proof-backed scaling oracle, and a sound post-check

Context: HANDOFF §22.  cuSOLVER documents that `cusolverDn<t1><t2>gesv` / `cusolverDnIRSXgesv` report convergence
(`iter ≥ 0`) only when `RNRM < √N · XNRM · ANRM · EPS` (∞-norms).  We observed, on its GMRES refinement path, `iter ≥ 0`
with backward errors up to 10¹¹× that threshold.  This file makes three things precise (core Lean, `Rat`, no Mathlib).

**Part 1 — the contract.**  `Accept c A b x : ‖b − A x‖ < c·‖x‖·‖A‖` and `Post` (`iter ≥ 0 → Accept`), with exact
rational arithmetic.  `accept_scale` : the rule is invariant under `(A, b) ↦ (s A, s b)` for every `s > 0`.
`absolute_rule_flips` : a rule with an absolute tolerance (`‖r‖ < tol`) is not.

**Part 2 — why the scaling test is an oracle.**
* `rnd p E q` : round-to-nearest, ties-to-even, `p`-bit significand, unbounded exponent (no overflow/underflow); `E`
  is any exponent function with `2^E(q) ≤ |q| < 2^(E(q)+1)` (`exp_spec`, one exists: `expOf`, `expOf_spec`).
* `rnd_scale` : `rnd (2^k q) = 2^k · rnd q` for every integer `k`.   `rnd_rel_err` : `|rnd q − q| ≤ 2^-p · |q|`.
* `Ex` : straight-line programs with branches over rounded `+ − × ÷`, exact `neg`, `abs`, `max`, comparisons and
  constants.  `deg` assigns a scaling degree; adding or comparing values of different degree is ill-typed.
* `eval_scale` : a well-typed program evaluated on inputs scaled by `2^(k·δᵢ)` returns `2^(k·d)` times its result —
  same branches, same rounding decisions.  `iterate_scale` lifts this to iterations (refinement loops).
* Hence any implementation built from such operations (LU, triangular solves, GMRES with ∞-norms, the documented stopping
  test), while no value overflows or becomes subnormal, returns the *same* `x` and the same iteration count for `(A, b)`
  and `(2^k A, 2^k b)` (`x` has degree 0).  A change of `x` under exact power-of-two scaling therefore proves that the
  implementation uses a scale-dependent constant somewhere, or over/underflows.  (It does not by itself prove a contract
  violation; Part 1's postcondition, checked directly, does that.)

**Part 3 — a post-check that makes any solver output satisfy the contract.**
`wrapper_sound` : if the computed residual norm `R̂` and matrix norm `Â` obey the stated error bounds and the test
`(1+ε)·R̂ + γR·(B + X·Â/(1−γA)) < c·X·Â/(1+γA)` holds, then the exact `‖b − A x‖ < c·‖x‖·‖A‖`.  With an FP64 residual
(`γR ≈ n·u`) the test can never pass, because `n·u > √n·u = c`: the documented threshold is below the worst-case
rounding error of an FP64 residual.  A compensated residual (Ogita–Rump–Oishi Dot2, `γR ≈ n²u²`) makes it usable;
`experiments/cusolver_irs/wrapper_demo.py` runs it on the failing cases.
-/

namespace IrsContract

/-! ## 0. Helpers on `Rat`: powers of two and absolute value -/

def p2 (k : Int) : Rat := (2 : Rat) ^ k

theorem two_ne : (2 : Rat) ≠ 0 := by decide

theorem p2_add (a b : Int) : p2 (a + b) = p2 a * p2 b := Rat.zpow_add two_ne a b

theorem p2_zero : p2 0 = 1 := by unfold p2; exact Rat.zpow_natCast 2 0

theorem one_le_pow_nat : ∀ n : Nat, (1 : Rat) ≤ (2 : Rat) ^ n
  | 0 => by rw [Rat.pow_zero]; exact Rat.le_refl
  | n + 1 => by
    rw [Rat.pow_succ]
    have := one_le_pow_nat n
    grind

theorem p2_nat (n : Nat) : p2 (n : Int) = (2 : Rat) ^ n := Rat.zpow_natCast 2 n

theorem p2_pos (k : Int) : 0 < p2 k := by
  rcases Int.le_total 0 k with h | h
  · obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le h
    rw [p2_nat]; have := one_le_pow_nat n; grind
  · obtain ⟨n, hn⟩ := Int.eq_ofNat_of_zero_le (Int.neg_nonneg_of_nonpos h)
    have e : k = -(n : Int) := by omega
    rw [e]; unfold p2; rw [Rat.zpow_neg, Rat.zpow_natCast]
    have := one_le_pow_nat n
    exact Rat.inv_pos.2 (by grind)

theorem p2_ne (k : Int) : p2 k ≠ 0 := fun h => by have := p2_pos k; grind

theorem p2_mono {a b : Int} (h : a ≤ b) : p2 a ≤ p2 b := by
  obtain ⟨n, hn⟩ := Int.eq_ofNat_of_zero_le (Int.sub_nonneg_of_le h)
  have e : b = a + n := by omega
  rw [e, p2_add, p2_nat]
  have h1 := one_le_pow_nat n
  have h2 := p2_pos a
  have := Rat.mul_le_mul_of_nonneg_left h1 (Rat.le_of_lt h2)
  grind

/-- absolute value -/
def ab (x : Rat) : Rat := if x < 0 then -x else x

theorem ab_nonneg (x : Rat) : 0 ≤ ab x := by unfold ab; split <;> grind

theorem le_ab (x : Rat) : x ≤ ab x := by unfold ab; split <;> grind

theorem neg_le_ab (x : Rat) : -x ≤ ab x := by unfold ab; split <;> grind

theorem ab_le {x y : Rat} (h1 : x ≤ y) (h2 : -x ≤ y) : ab x ≤ y := by unfold ab; split <;> grind

theorem ab_neg (x : Rat) : ab (-x) = ab x := by unfold ab; split <;> split <;> grind

theorem ab_triangle (x y : Rat) : ab (x + y) ≤ ab x + ab y := by
  apply ab_le
  · have := le_ab x; have := le_ab y; grind
  · have := neg_le_ab x; have := neg_le_ab y; grind

theorem ab_eq_zero {x : Rat} (h : ab x = 0) : x = 0 := by unfold ab at h; split at h <;> grind

/-- `|s·x| = s·|x|` for `s > 0` -/
theorem ab_smul {s : Rat} (hs : 0 < s) (x : Rat) : ab (s * x) = s * ab x := by
  unfold ab
  by_cases hx : x < 0
  · have : s * x < 0 := by have := Rat.mul_lt_mul_of_pos_left hx hs; grind
    simp only [this, hx, ↓reduceIte]; grind
  · have : ¬ s * x < 0 := by
      have h0 : 0 ≤ x := by grind
      have := Rat.mul_le_mul_of_nonneg_left h0 (Rat.le_of_lt hs); grind
    simp only [this, hx, ↓reduceIte]

theorem mul_le_mul_left_iff {s a b : Rat} (hs : 0 < s) : s * a ≤ s * b ↔ a ≤ b := by
  constructor
  · intro h
    by_cases h' : a ≤ b
    · exact h'
    · have h'' : b < a := by grind
      have := Rat.mul_lt_mul_of_pos_left h'' hs; grind
  · intro h; exact Rat.mul_le_mul_of_nonneg_left h (Rat.le_of_lt hs)

theorem max_smul {s : Rat} (hs : 0 < s) (x y : Rat) : max (s * x) (s * y) = s * max x y := by
  rw [Rat.max_def, Rat.max_def]
  by_cases h : x ≤ y
  · have h' : s * x ≤ s * y := (mul_le_mul_left_iff hs).2 h
    simp only [h, h', ↓reduceIte]
  · have h' : ¬ s * x ≤ s * y := fun h2 => h ((mul_le_mul_left_iff hs).1 h2)
    simp only [h, h', ↓reduceIte]

theorem mul_lt_mul_left_iff {s a b : Rat} (hs : 0 < s) : s * a < s * b ↔ a < b := by
  constructor
  · intro h
    by_cases h' : a < b
    · exact h'
    · have h'' : b ≤ a := by grind
      have := Rat.mul_le_mul_of_nonneg_left h'' (Rat.le_of_lt hs); grind
  · intro h; exact Rat.mul_lt_mul_of_pos_left h hs

/-! ## 1. The contract -/

def dot : List Rat → List Rat → Rat
  | a :: as, x :: xs => a * x + dot as xs
  | _, _ => 0

def mv (A : List (List Rat)) (x : List Rat) : List Rat := A.map (fun row => dot row x)

def vsub : List Rat → List Rat → List Rat
  | a :: as, b :: bs => (a - b) :: vsub as bs
  | _, _ => []

def smul (s : Rat) (v : List Rat) : List Rat := v.map (s * ·)

def smulM (s : Rat) (A : List (List Rat)) : List (List Rat) := A.map (smul s)

/-- ∞-norm of a vector -/
def ninf : List Rat → Rat
  | [] => 0
  | a :: as => max (ab a) (ninf as)

def asum : List Rat → Rat
  | [] => 0
  | a :: as => ab a + asum as

/-- ∞-operator norm: largest absolute row sum -/
def anorm : List (List Rat) → Rat
  | [] => 0
  | r :: rs => max (asum r) (anorm rs)

/-- the documented acceptance test `RNRM < c·XNRM·ANRM` (c = √N·EPS) -/
def Accept (c : Rat) (A : List (List Rat)) (b x : List Rat) : Prop :=
  ninf (vsub b (mv A x)) < c * ninf x * anorm A

/-- the documented postcondition: `iter ≥ 0` (reported convergence) implies the acceptance test -/
def Post (c : Rat) (A : List (List Rat)) (b x : List Rat) (iter : Int) : Prop := 0 ≤ iter → Accept c A b x

theorem dot_smul (s : Rat) : ∀ r x : List Rat, dot (smul s r) x = s * dot r x
  | a :: as, y :: ys => by simp only [smul, List.map, dot]; have := dot_smul s as ys; simp only [smul] at this; rw [this]; grind
  | [], _ => by simp [smul, dot]
  | _ :: _, [] => by simp [smul, dot]

theorem mv_smulM (s : Rat) (A : List (List Rat)) (x : List Rat) : mv (smulM s A) x = smul s (mv A x) := by
  induction A with
  | nil => rfl
  | cons r rs ih =>
    simp only [mv, smulM, smul, List.map] at ih ⊢
    rw [ih]; congr 1; exact dot_smul s r x

theorem vsub_smul (s : Rat) : ∀ b v : List Rat, vsub (smul s b) (smul s v) = smul s (vsub b v)
  | a :: as, y :: ys => by
    simp only [smul, List.map, vsub]; have := vsub_smul s as ys; simp only [smul] at this; rw [this]
    congr 1; grind
  | [], _ => by simp [smul, vsub]
  | _ :: _, [] => by simp [smul, vsub]

theorem ninf_smul {s : Rat} (hs : 0 < s) : ∀ v : List Rat, ninf (smul s v) = s * ninf v
  | [] => by simp [smul, ninf]
  | a :: as => by
    simp only [smul, List.map, ninf]; have := ninf_smul hs as; simp only [smul] at this
    rw [this, ab_smul hs, max_smul hs]

theorem asum_smul {s : Rat} (hs : 0 < s) : ∀ v : List Rat, asum (smul s v) = s * asum v
  | [] => by simp [smul, asum]
  | a :: as => by
    simp only [smul, List.map, asum]; have := asum_smul hs as; simp only [smul] at this
    rw [this, ab_smul hs]; grind

theorem anorm_smulM {s : Rat} (hs : 0 < s) : ∀ A : List (List Rat), anorm (smulM s A) = s * anorm A
  | [] => by simp [smulM, anorm]
  | r :: rs => by
    simp only [smulM, List.map, anorm]; have := anorm_smulM hs rs; simp only [smulM] at this
    rw [this, asum_smul hs, max_smul hs]

/-- **the documented rule is scale invariant** -/
theorem accept_scale (c : Rat) (A : List (List Rat)) (b x : List Rat) {s : Rat} (hs : 0 < s) :
    Accept c (smulM s A) (smul s b) x ↔ Accept c A b x := by
  unfold Accept
  rw [mv_smulM, vsub_smul, ninf_smul hs, anorm_smulM hs]
  have e : c * ninf x * (s * anorm A) = s * (c * ninf x * anorm A) := by grind
  rw [e]
  exact mul_lt_mul_left_iff hs

theorem post_scale (c : Rat) (A : List (List Rat)) (b x : List Rat) (iter : Int) {s : Rat} (hs : 0 < s) :
    Post c (smulM s A) (smul s b) x iter ↔ Post c A b x iter := by
  unfold Post; rw [accept_scale c A b x hs]

/-- an absolute tolerance is not scale invariant: `A = [[1]]`, `b = [1]`, `x = [0]`, `tol = 2`:
    `‖b − A x‖ = 1 < 2` is accepted, the same system scaled by 4 (`‖r‖ = 4`) is rejected -/
theorem absolute_rule_flips :
    ninf (vsub [1] (mv [[1]] [0])) < 2 ∧ ¬ ninf (vsub (smul 4 [1]) (mv (smulM 4 [[1]]) [0])) < 2 := by
  simp only [ninf, vsub, mv, dot, smul, smulM, List.map, ab]
  constructor <;> grind

/-! ## 2. Rounding to nearest-even with `p` significand bits commutes with `2^k` -/

/-- `e` is the exponent of `q`: `2^e ≤ |q| < 2^(e+1)` -/
def IsExp (q : Rat) (e : Int) : Prop := p2 e ≤ ab q ∧ ab q < p2 (e + 1)

theorem p2_inv (k : Int) : p2 k * p2 (-k) = 1 := by rw [← p2_add, Int.add_right_neg, p2_zero]

theorem exp_unique {q : Rat} {e e' : Int} (h : IsExp q e) (h' : IsExp q e') : e = e' := by
  unfold IsExp at h h'
  rcases Int.lt_trichotomy e e' with hl | he | hg
  · have := p2_mono (show e + 1 ≤ e' by omega); grind
  · exact he
  · have := p2_mono (show e' + 1 ≤ e by omega); grind

theorem exp_scale (k : Int) {q : Rat} {e : Int} (h : IsExp q e) : IsExp (p2 k * q) (e + k) := by
  unfold IsExp at h ⊢
  rw [ab_smul (p2_pos k), p2_add, show e + k + 1 = (e + 1) + k by omega, p2_add]
  have hk := p2_pos k
  constructor
  · have := Rat.mul_le_mul_of_nonneg_left h.1 (Rat.le_of_lt hk); grind
  · have := Rat.mul_lt_mul_of_pos_left h.2 hk; grind

/-- round half to even, on an exact rational -/
def rhe (y : Rat) : Int :=
  if y - y.floor < 1/2 then y.floor
  else if 1/2 < y - y.floor then y.floor + 1
  else if y.floor % 2 = 0 then y.floor else y.floor + 1

theorem rhe_err (y : Rat) : ab ((rhe y : Rat) - y) ≤ 1/2 := by
  have h1 := Rat.floor_le y
  have h2 := Rat.lt_floor_add_one y
  rw [Rat.intCast_add, Rat.intCast_one] at h2
  unfold rhe
  apply ab_le <;> (split <;> (try split) <;> (try split) <;> (try rw [Rat.intCast_add, Rat.intCast_one]) <;> grind)

/-- `p`-bit round-to-nearest-even with unbounded exponent; `E` supplies the exponent -/
def rnd (p : Int) (E : Rat → Int) (q : Rat) : Rat :=
  if q = 0 then 0 else (rhe (q * p2 (p - 1 - E q)) : Rat) * p2 (E q + 1 - p)

def ExpFn (E : Rat → Int) : Prop := ∀ q, q ≠ 0 → IsExp q (E q)

theorem smul_ne_zero (k : Int) {q : Rat} (hq : q ≠ 0) : p2 k * q ≠ 0 := by
  intro h
  rcases Rat.mul_eq_zero.1 h with h | h
  · exact p2_ne k h
  · exact hq h

/-- **rounding commutes with scaling by any power of two** (no overflow/underflow in this model) -/
theorem rnd_scale (p : Int) {E : Rat → Int} (hE : ExpFn E) (k : Int) (q : Rat) :
    rnd p E (p2 k * q) = p2 k * rnd p E q := by
  by_cases hq : q = 0
  · subst hq; unfold rnd; simp
  have hq' := smul_ne_zero k hq
  have hEk : E (p2 k * q) = E q + k := exp_unique (hE _ hq') (exp_scale k (hE q hq))
  unfold rnd
  simp only [hq, hq', ↓reduceIte]
  rw [hEk]
  have harg : p2 k * q * p2 (p - 1 - (E q + k)) = q * p2 (p - 1 - E q) := by
    rw [show p - 1 - (E q + k) = (p - 1 - E q) + (-k) by omega, p2_add]
    have := p2_inv k
    grind
  rw [harg, show E q + k + 1 - p = k + (E q + 1 - p) by omega, p2_add]
  grind

/-- the unit roundoff: `|rnd q − q| ≤ 2^-p · |q|` -/
theorem rnd_rel_err (p : Int) {E : Rat → Int} (hE : ExpFn E) (q : Rat) :
    ab (rnd p E q - q) ≤ p2 (-p) * ab q := by
  by_cases hq : q = 0
  · subst hq; unfold rnd; simp only [↓reduceIte]
    have : ab (0 - 0) = 0 := by unfold ab; grind
    rw [this]; have := ab_nonneg 0; have := p2_pos (-p); grind
  have hx := hE q hq
  unfold IsExp at hx
  unfold rnd; simp only [hq, ↓reduceIte]
  generalize hy : q * p2 (p - 1 - E q) = y
  have hP := p2_pos (E q + 1 - p)
  have hqy : q = y * p2 (E q + 1 - p) := by
    rw [← hy, Rat.mul_assoc, ← p2_add, show p - 1 - E q + (E q + 1 - p) = 0 by omega, p2_zero, Rat.mul_one]
  have e1 : (rhe y : Rat) * p2 (E q + 1 - p) - q = p2 (E q + 1 - p) * ((rhe y : Rat) - y) := by rw [hqy]; grind
  rw [e1, ab_smul hP]
  have h1 := rhe_err y
  have h2 : p2 (E q + 1 - p) * ab ((rhe y : Rat) - y) ≤ p2 (E q + 1 - p) * (1/2) :=
    Rat.mul_le_mul_of_nonneg_left h1 (Rat.le_of_lt hP)
  have h3 : p2 (E q + 1 - p) * (1/2) = p2 (-p) * p2 (E q) := by
    rw [← p2_add, show -p + E q = (E q + 1 - p) + (-1) by omega, p2_add]
    have : p2 (-1) = 1/2 := by
      unfold p2; rw [Rat.zpow_neg, show ((1 : Int)) = ((1 : Nat) : Int) by rfl, Rat.zpow_natCast, Rat.pow_one]; grind
    rw [this]
  have h4 : p2 (-p) * p2 (E q) ≤ p2 (-p) * ab q := Rat.mul_le_mul_of_nonneg_left hx.1 (Rat.le_of_lt (p2_pos _))
  grind

/-! ### an exponent function exists (so the statements above are not vacuous) -/

theorem p2_natcast (n : Nat) : ((2 ^ n : Nat) : Rat) = p2 (n : Int) := by
  rw [p2_nat, Rat.natCast_pow]; rfl

/-- `floor(log2 |q|)` computed from the numerator and denominator of `|q|` -/
def expOf (q : Rat) : Int :=
  let r := ab q
  let a : Int := Nat.log2 r.num.toNat
  let b : Int := Nat.log2 r.den
  if p2 (a - b) ≤ r then a - b else a - b - 1

theorem expOf_spec : ExpFn expOf := by
  intro q hq
  have hr : 0 < ab q := by
    have := ab_nonneg q
    rcases Rat.le_iff_lt_or_eq.1 this with h | h
    · exact h
    · exact absurd (ab_eq_zero h.symm) hq
  generalize hrdef : ab q = r at hr
  have hnum : 0 < r.num := by
    have h0 : 0 ≤ r.num := Rat.num_nonneg.2 (Rat.le_of_lt hr)
    have h1 : r.num ≠ 0 := fun h => by have := Rat.num_eq_zero.1 h; grind
    omega
  obtain ⟨n, hn⟩ := Int.eq_ofNat_of_zero_le (Int.le_of_lt hnum)
  have hn0 : n ≠ 0 := by omega
  have htoNat : r.num.toNat = n := by omega
  have hd0 : r.den ≠ 0 := Rat.den_nz r
  have hrr : r * (r.den : Rat) = (n : Rat) := by
    have e := Rat.mkRat_self r
    rw [Rat.mkRat_eq_div] at e
    have hdq : (r.den : Rat) ≠ 0 := by
      intro h; exact hd0 (by exact_mod_cast h)
    have h3 : r * (r.den : Rat) = (r.num : Rat) / (r.den : Rat) * (r.den : Rat) := by rw [e]
    rw [h3, Rat.div_def, Rat.mul_assoc, Rat.inv_mul_cancel _ hdq, Rat.mul_one, hn]
    rfl
  have hdpos : (0 : Rat) < (r.den : Rat) := Rat.natCast_pos.2 (Rat.den_pos r)
  have na1 := Nat.log2_self_le hn0
  have na2 := Nat.lt_log2_self (n := n)
  have nb1 := Nat.log2_self_le hd0
  have nb2 := Nat.lt_log2_self (n := r.den)
  have ra1 : p2 (Nat.log2 n : Int) ≤ (n : Rat) := by rw [← p2_natcast]; exact_mod_cast na1
  have ra2 : (n : Rat) < p2 ((Nat.log2 n + 1 : Nat) : Int) := by rw [← p2_natcast]; exact_mod_cast na2
  have rb1 : p2 (Nat.log2 r.den : Int) ≤ (r.den : Rat) := by rw [← p2_natcast]; exact_mod_cast nb1
  have rb2 : (r.den : Rat) < p2 ((Nat.log2 r.den + 1 : Nat) : Int) := by rw [← p2_natcast]; exact_mod_cast nb2
  generalize ha : (Nat.log2 n : Int) = a at ra1 ra2
  generalize hb : (Nat.log2 r.den : Int) = b at rb1 rb2
  have ra2' : (n : Rat) < p2 (a + 1) := by rw [show a + 1 = ((Nat.log2 n + 1 : Nat) : Int) by omega]; exact ra2
  have rb2' : (r.den : Rat) < p2 (b + 1) := by rw [show b + 1 = ((Nat.log2 r.den + 1 : Nat) : Int) by omega]; exact rb2
  -- r ≥ 2^(a-b-1) and r < 2^(a-b+1)
  have lo : p2 (a - b - 1) ≤ r := by
    have e : p2 (a - b - 1) * p2 (b + 1) = p2 a := by rw [← p2_add]; congr 1; omega
    have h1 : p2 (a - b - 1) * (r.den : Rat) ≤ p2 (a - b - 1) * p2 (b + 1) :=
      Rat.mul_le_mul_of_nonneg_left (Rat.le_of_lt rb2') (Rat.le_of_lt (p2_pos _))
    have h2 : p2 (a - b - 1) * (r.den : Rat) ≤ r * (r.den : Rat) := by grind
    rw [Rat.mul_comm (p2 _), Rat.mul_comm r] at h2
    exact (mul_le_mul_left_iff hdpos).1 h2
  have hi : r < p2 (a - b + 1) := by
    have e : p2 (a - b + 1) * p2 b = p2 (a + 1) := by rw [← p2_add]; congr 1; omega
    have h1 : p2 (a - b + 1) * p2 b ≤ p2 (a - b + 1) * (r.den : Rat) :=
      Rat.mul_le_mul_of_nonneg_left rb1 (Rat.le_of_lt (p2_pos _))
    have h2 : r * (r.den : Rat) < p2 (a - b + 1) * (r.den : Rat) := by grind
    rw [Rat.mul_comm r, Rat.mul_comm (p2 _)] at h2
    exact (mul_lt_mul_left_iff hdpos).1 h2
  unfold expOf
  simp only [hrdef, htoNat, ha, hb]
  by_cases hc : p2 (a - b) ≤ r
  · simp only [hc, ↓reduceIte]
    unfold IsExp; rw [hrdef]; exact ⟨hc, hi⟩
  · simp only [hc, ↓reduceIte]
    unfold IsExp; rw [hrdef, show a - b - 1 + 1 = a - b by omega]; exact ⟨lo, by grind⟩

/-! ### programs with scaling degrees -/

/-- straight-line programs with branches: rounded `+ − × ÷`, exact `neg`, `abs`, `max`, `if a < b then t else f` -/
inductive Ex where
  | var : Nat → Ex
  | cst : Rat → Ex
  | add : Ex → Ex → Ex
  | sub : Ex → Ex → Ex
  | mul : Ex → Ex → Ex
  | div : Ex → Ex → Ex
  | neg : Ex → Ex
  | abs : Ex → Ex
  | max : Ex → Ex → Ex
  | ite : Ex → Ex → Ex → Ex → Ex

/-- evaluation with a rounding function `R` applied after every arithmetic operation -/
def eval (R : Rat → Rat) (env : Nat → Rat) : Ex → Rat
  | .var i => env i
  | .cst c => c
  | .add a b => R (eval R env a + eval R env b)
  | .sub a b => R (eval R env a - eval R env b)
  | .mul a b => R (eval R env a * eval R env b)
  | .div a b => R (eval R env a / eval R env b)
  | .neg a => -(eval R env a)
  | .abs a => ab (eval R env a)
  | .max a b => max (eval R env a) (eval R env b)
  | .ite a b t f => if eval R env a < eval R env b then eval R env t else eval R env f

/-- scaling degrees: inputs `i` have degree `δ i`; `+`, `−`, `max` and comparisons need equal degrees;
    constants have degree 0 (so `‖r‖ < tol` with an absolute `tol` is ill-typed unless `‖r‖` has degree 0) -/
inductive HasDeg (δ : Nat → Int) : Ex → Int → Prop where
  | var (i : Nat) : HasDeg δ (.var i) (δ i)
  | cst (c : Rat) : HasDeg δ (.cst c) 0
  | add {a b : Ex} {d : Int} : HasDeg δ a d → HasDeg δ b d → HasDeg δ (.add a b) d
  | sub {a b : Ex} {d : Int} : HasDeg δ a d → HasDeg δ b d → HasDeg δ (.sub a b) d
  | mul {a b : Ex} {d d' : Int} : HasDeg δ a d → HasDeg δ b d' → HasDeg δ (.mul a b) (d + d')
  | div {a b : Ex} {d d' : Int} : HasDeg δ a d → HasDeg δ b d' → HasDeg δ (.div a b) (d - d')
  | neg {a : Ex} {d : Int} : HasDeg δ a d → HasDeg δ (.neg a) d
  | abs {a : Ex} {d : Int} : HasDeg δ a d → HasDeg δ (.abs a) d
  | max {a b : Ex} {d : Int} : HasDeg δ a d → HasDeg δ b d → HasDeg δ (.max a b) d
  | ite {a b t f : Ex} {d d' : Int} : HasDeg δ a d → HasDeg δ b d → HasDeg δ t d' → HasDeg δ f d' →
      HasDeg δ (.ite a b t f) d'

/-- inputs scaled by `2^(k·δᵢ)` -/
def senv (k : Int) (δ : Nat → Int) (env : Nat → Rat) : Nat → Rat := fun i => p2 (k * δ i) * env i

theorem inv_mul_distrib {t b : Rat} (ht : t ≠ 0) : (t * b)⁻¹ = t⁻¹ * b⁻¹ := by
  by_cases hb : b = 0
  · subst hb; rw [Rat.mul_zero, Rat.inv_zero, Rat.mul_zero]
  have htb : t * b ≠ 0 := fun h => by rcases Rat.mul_eq_zero.1 h with h | h <;> contradiction
  have h1 : (t * b) * (t⁻¹ * b⁻¹) = 1 := by
    have := Rat.mul_inv_cancel t ht; have := Rat.mul_inv_cancel b hb
    calc (t * b) * (t⁻¹ * b⁻¹) = (t * t⁻¹) * (b * b⁻¹) := by grind
      _ = 1 := by grind
  calc (t * b)⁻¹ = (t * b)⁻¹ * ((t * b) * (t⁻¹ * b⁻¹)) := by rw [h1, Rat.mul_one]
    _ = ((t * b)⁻¹ * (t * b)) * (t⁻¹ * b⁻¹) := (Rat.mul_assoc _ _ _).symm
    _ = t⁻¹ * b⁻¹ := by rw [Rat.inv_mul_cancel _ htb, Rat.one_mul]

theorem p2_inv' (m : Int) : (p2 m)⁻¹ = p2 (-m) := by unfold p2; rw [Rat.zpow_neg]

/-- **covariance**: a well-typed program commutes with scaling, provided the rounding does (`rnd_scale`) -/
theorem eval_scale {R : Rat → Rat} (hR : ∀ m q, R (p2 m * q) = p2 m * R q) (k : Int) {δ : Nat → Int}
    (env : Nat → Rat) {e : Ex} {d : Int} (h : HasDeg δ e d) :
    eval R (senv k δ env) e = p2 (k * d) * eval R env e := by
  induction h with
  | var i => rfl
  | cst c => simp only [eval]; rw [Int.mul_zero, p2_zero, Rat.one_mul]
  | add _ _ iha ihb => simp only [eval]; rw [iha, ihb, ← hR]; congr 1; grind
  | sub _ _ iha ihb => simp only [eval]; rw [iha, ihb, ← hR]; congr 1; grind
  | @mul a b d d' _ _ iha ihb =>
    simp only [eval]; rw [iha, ihb, ← hR, Int.mul_add, p2_add]; congr 1; grind
  | @div a b d d' _ _ iha ihb =>
    simp only [eval]; rw [iha, ihb, ← hR, Rat.div_def, Rat.div_def, inv_mul_distrib (p2_ne _), p2_inv',
      show k * (d - d') = k * d + -(k * d') by rw [Int.mul_sub]; omega, p2_add]
    congr 1; grind
  | neg _ ih => simp only [eval]; rw [ih]; grind
  | abs _ ih => simp only [eval]; rw [ih, ab_smul (p2_pos _)]
  | max _ _ iha ihb => simp only [eval]; rw [iha, ihb, max_smul (p2_pos _)]
  | @ite a b t f d d' _ _ _ _ iha ihb iht ihf =>
    simp only [eval]; rw [iha, ihb, iht, ihf]
    by_cases hc : eval R env a < eval R env b
    · have : p2 (k * d) * eval R env a < p2 (k * d) * eval R env b := (mul_lt_mul_left_iff (p2_pos _)).2 hc
      simp only [hc, this, ↓reduceIte]
    · have : ¬ p2 (k * d) * eval R env a < p2 (k * d) * eval R env b :=
        fun h => hc ((mul_lt_mul_left_iff (p2_pos _)).1 h)
      simp only [hc, this, ↓reduceIte]

/-- degree-0 outputs (e.g. the solution `x` of `A x = b` when `A`, `b` have degree 1) are unchanged by scaling -/
theorem eval_scale_deg0 {R : Rat → Rat} (hR : ∀ m q, R (p2 m * q) = p2 m * R q) (k : Int) {δ : Nat → Int}
    (env : Nat → Rat) {e : Ex} (h : HasDeg δ e 0) : eval R (senv k δ env) e = eval R env e := by
  rw [eval_scale hR k env h, Int.mul_zero, p2_zero, Rat.one_mul]

/-- one step of an iterative method: every state variable `i` is recomputed by a program of degree `δ i` -/
def stepOf (R : Rat → Rat) (es : Nat → Ex) (env : Nat → Rat) : Nat → Rat := fun i => eval R env (es i)

def iter {α : Type} (f : α → α) : Nat → α → α
  | 0, a => a
  | n + 1, a => iter f n (f a)

theorem stepOf_scale {R : Rat → Rat} (hR : ∀ m q, R (p2 m * q) = p2 m * R q) (k : Int) {δ : Nat → Int}
    {es : Nat → Ex} (hes : ∀ i, HasDeg δ (es i) (δ i)) (env : Nat → Rat) :
    stepOf R es (senv k δ env) = senv k δ (stepOf R es env) := by
  funext i; exact eval_scale hR k env (hes i)

/-- **iterations commute with scaling**: `n` refinement steps (stopping tests and iteration counters included, as
    state variables of degree 0 updated through `ite`) give the scaled state of the unscaled run -/
theorem iterate_scale {R : Rat → Rat} (hR : ∀ m q, R (p2 m * q) = p2 m * R q) (k : Int) {δ : Nat → Int}
    {es : Nat → Ex} (hes : ∀ i, HasDeg δ (es i) (δ i)) :
    ∀ (n : Nat) (env : Nat → Rat), iter (stepOf R es) n (senv k δ env) = senv k δ (iter (stepOf R es) n env)
  | 0, env => rfl
  | n + 1, env => by
    simp only [iter]; rw [stepOf_scale hR k hes env]; exact iterate_scale hR k hes n _

/-- the oracle, for the concrete rounding `rnd p expOf` -/
theorem oracle (p : Int) (k : Int) {δ : Nat → Int} {es : Nat → Ex} (hes : ∀ i, HasDeg δ (es i) (δ i))
    (n : Nat) (env : Nat → Rat) (i : Nat) (h0 : δ i = 0) :
    iter (stepOf (rnd p expOf) es) n (senv k δ env) i = iter (stepOf (rnd p expOf) es) n env i := by
  rw [iterate_scale (fun m q => rnd_scale p expOf_spec m q) k hes n env]
  simp only [senv]; rw [h0, Int.mul_zero, p2_zero, Rat.one_mul]

/-- an absolute constant in a comparison breaks it: `if r < 1 then 1 else 0` with `r` an input of degree 1 -/
theorem absolute_tol_breaks :
    eval id (senv 2 (fun _ => 1) (fun _ => 1/2)) (.ite (.var 0) (.cst 1) (.cst 1) (.cst 0)) = 0 ∧
    eval id (fun _ => 1/2) (.ite (.var 0) (.cst 1) (.cst 1) (.cst 0)) = 1 := by
  have : p2 (2 * 1) = 4 := by
    unfold p2; rw [show (2 : Int) * 1 = ((2 : Nat) : Int) by rfl, Rat.zpow_natCast]; grind
  simp only [eval, senv, this]
  constructor
  · split <;> grind
  · simp only [show (1 / 2 : Rat) < 1 by grind, ↓reduceIte]

/-! ## 3. A sound post-check (wrapper) -/

theorem le_div_of_mul_le {a b c : Rat} (hc : 0 < c) (h : c * a ≤ b) : a ≤ b / c := by
  have hi : 0 < c⁻¹ := Rat.inv_pos.2 hc
  have := Rat.mul_le_mul_of_nonneg_left h (Rat.le_of_lt hi)
  rw [← Rat.mul_assoc, Rat.inv_mul_cancel c (by grind), Rat.one_mul] at this
  rw [Rat.div_def]; grind

theorem div_le_of_le_mul {a b c : Rat} (hc : 0 < c) (h : a ≤ c * b) : a / c ≤ b := by
  have hi : 0 < c⁻¹ := Rat.inv_pos.2 hc
  have := Rat.mul_le_mul_of_nonneg_left h (Rat.le_of_lt hi)
  rw [← Rat.mul_assoc, Rat.inv_mul_cancel c (by grind), Rat.one_mul] at this
  rw [Rat.div_def]; grind

theorem mul_nonneg' {a b : Rat} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a * b := by
  have := Rat.mul_le_mul_of_nonneg_left hb ha; grind

/-- **soundness of the post-check.**  `R`, `A`, `X`, `B`: exact `‖b − A x‖∞`, `‖A‖∞`, `‖x‖∞`, `‖b‖∞`;
    `Rh`, `Ah`: their computed values, with the error bounds `hR`, `hA1`, `hA2` (hypotheses: they are the standard
    bounds of the residual and row-sum computations, e.g. `γR = γ_{n+1}` for an FP64 residual or `≈ γ_n²` for a
    compensated (Dot2) residual, and `γA = γ_n`).  If the test passes, the documented condition holds exactly. -/
theorem wrapper_sound {R Rh B A Ah X c ε γR γA : Rat}
    (hX : 0 ≤ X) (hc : 0 ≤ c) (hγR : 0 ≤ γR) (hγA1 : γA < 1)
    (hR : R ≤ (1 + ε) * Rh + γR * (B + A * X))
    (hA1 : (1 - γA) * A ≤ Ah) (hA2 : Ah ≤ (1 + γA) * A) (hγA0 : 0 ≤ γA)
    (htest : (1 + ε) * Rh + γR * (B + X * (Ah / (1 - γA))) < c * X * (Ah / (1 + γA))) :
    R < c * X * A := by
  have h1 : A ≤ Ah / (1 - γA) := le_div_of_mul_le (by grind) hA1
  have h2 : X * A ≤ X * (Ah / (1 - γA)) := Rat.mul_le_mul_of_nonneg_left h1 hX
  have h3 : γR * (B + A * X) ≤ γR * (B + X * (Ah / (1 - γA))) :=
    Rat.mul_le_mul_of_nonneg_left (by grind) hγR
  have h4 : Ah / (1 + γA) ≤ A := div_le_of_le_mul (by grind) hA2
  have h5 : c * X * (Ah / (1 + γA)) ≤ c * X * A := Rat.mul_le_mul_of_nonneg_left h4 (mul_nonneg' hc hX)
  grind

/-- the wrapper's guarantee in terms of Part 1: whatever `x` the solver returned, if the check passes, the documented
    postcondition holds (for any reported `iter`) -/
theorem wrapper_post (c : Rat) (A : List (List Rat)) (b x : List Rat) (iter : Int) {Rh Ah ε γR γA : Rat}
    (hc : 0 ≤ c) (hγR : 0 ≤ γR) (hγA0 : 0 ≤ γA) (hγA1 : γA < 1)
    (hR : ninf (vsub b (mv A x)) ≤ (1 + ε) * Rh + γR * (ninf b + anorm A * ninf x))
    (hA1 : (1 - γA) * anorm A ≤ Ah) (hA2 : Ah ≤ (1 + γA) * anorm A)
    (htest : (1 + ε) * Rh + γR * (ninf b + ninf x * (Ah / (1 - γA))) < c * ninf x * (Ah / (1 + γA))) :
    Post c A b x iter := by
  intro _
  have hX : 0 ≤ ninf x := by
    cases x with
    | nil => exact Rat.le_refl
    | cons a as => simp only [ninf]; have := ab_nonneg a; grind
  exact wrapper_sound hX hc hγR hγA1 hR hA1 hA2 hγA0 htest

/-- **an FP64 residual cannot certify the documented threshold**: if the residual error bound `γR` is at least the
    threshold constant `c` (FP64: `γR ≥ (n+1)·u > √n·u = c`), the test never passes -/
theorem fp64_cannot_certify {Rh B Ah X c ε γR γA : Rat}
    (hRh : 0 ≤ (1 + ε) * Rh) (hB : 0 ≤ B) (hX : 0 ≤ X) (hAh : 0 ≤ Ah) (hc : 0 ≤ c)
    (hγA0 : 0 ≤ γA) (hγA1 : γA < 1) (hcR : c ≤ γR) :
    ¬ ((1 + ε) * Rh + γR * (B + X * (Ah / (1 - γA))) < c * X * (Ah / (1 + γA))) := by
  intro h
  have hd1 : Ah ≤ Ah / (1 - γA) := le_div_of_mul_le (by grind) (by
    have := Rat.mul_le_mul_of_nonneg_left (show 1 - γA ≤ 1 by grind) hAh; grind)
  have hd2 : Ah / (1 + γA) ≤ Ah := div_le_of_le_mul (by grind) (by
    have := Rat.mul_le_mul_of_nonneg_left (show 1 ≤ 1 + γA by grind) hAh; grind)
  have hXA : 0 ≤ X * Ah := mul_nonneg' hX hAh
  have e1 : c * X * (Ah / (1 + γA)) ≤ c * (X * Ah) := by
    have := Rat.mul_le_mul_of_nonneg_left hd2 (mul_nonneg' hc hX); grind
  have e2 : c * (X * Ah) ≤ γR * (X * Ah) := by
    have := Rat.mul_le_mul_of_nonneg_left hcR hXA; grind
  have e3 : X * Ah ≤ X * (Ah / (1 - γA)) := Rat.mul_le_mul_of_nonneg_left hd1 hX
  have e4 : γR * (X * Ah) ≤ γR * (B + X * (Ah / (1 - γA))) :=
    Rat.mul_le_mul_of_nonneg_left (by grind) (by grind)
  grind

/-- the numbers for n = 1024: `c = √1024·2^-53 = 32·2^-53`, while an FP64 residual has `γR ≥ 1025·2^-53` -/
theorem fp64_n1024 : (32 : Rat) * p2 (-53) ≤ 1025 * p2 (-53) := by
  have := p2_pos (-53)
  have := Rat.mul_le_mul_of_nonneg_left (show (32 : Rat) ≤ 1025 by grind) (Rat.le_of_lt this); grind

end IrsContract

#print axioms IrsContract.accept_scale
#print axioms IrsContract.absolute_rule_flips
#print axioms IrsContract.rnd_scale
#print axioms IrsContract.rnd_rel_err
#print axioms IrsContract.expOf_spec
#print axioms IrsContract.eval_scale
#print axioms IrsContract.iterate_scale
#print axioms IrsContract.oracle
#print axioms IrsContract.absolute_tol_breaks
#print axioms IrsContract.wrapper_sound
#print axioms IrsContract.wrapper_post
#print axioms IrsContract.fp64_cannot_certify
#print axioms IrsContract.fp64_n1024

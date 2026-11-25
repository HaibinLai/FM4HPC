/-!
# D11: max reductions that drop NaN, and when an acceptance test built on them can see a NaN

Defect class D11 of `operators/IR_SURVEY_zh.md`: the componentwise backward error, or a column norm, is computed as
a running maximum, and the way the maximum is written decides whether a NaN survives.

The three ways seen in the surveyed code (`v` = next entry, `s` = running value):
* `smax`  — SuperLU / SuperLU_DIST `SUPERLU_MAX(s, v) = (s > v) ? s : v`
            (`SRC/slu_util.h:95`, `SRC/include/util_dist.h:63`; used in `dgsrfs.c`, `pdgsrfs.c`, `psgsrfs_d2.c`);
* `pstep` — PLASMA `core_dzamax.c:35` `if (tmp > values[j]) values[j] = tmp;` (same shape as reference IDAMAX's
            `IF (DABS(DX(I)).GT.DMAX)`), started from the first entry;
* `nmax`  — a NaN-propagating maximum (the fix).

Proved (core Lean, exact `Rat` values plus a NaN; every IEEE comparison with NaN is false; infinities not modelled):
* `smax_fold_isNaN`  : the SuperLU fold is NaN **iff the last entry is NaN** (`lastOr`); entries before it are lost.
* `smax_fold_after_nan` / `smax_fold_value` : if the last NaN is followed by finite entries, the result is exactly the
                       maximum of the entries after it — a clean, finite number.
* `masked_equals_baseline` : the shape of the SuperLU_DIST run in `experiments/ir_survey/sludist_d2_results.txt`
                       (first block NaN, second block finite and equal to the first block's clean values) returns
                       **exactly the clean baseline value** — as measured (berr 2.96e-16 in both runs).
* `pstep_fold_isNaN` : the PLASMA / IDAMAX fold is NaN **iff the first entry is NaN**; later NaNs are lost.
* `nmax_fold_isNaN`  : the propagating fold is NaN iff some entry (or the start value) is NaN.
* `reject_accepts_nan`, `accept_rejects_nan` : `!(r > thr)` (the LAPACK `RNRM.GT.XNRM*CTE` / `BERR.GT.EPS` form)
                       accepts a NaN `r`; `r < thr` rejects it.
* `detects_iff` : over the 2 test forms × 3 reductions, the acceptance test rejects every input that contains a NaN
                  **iff** it is the `<` form **and** the reduction is `nmax`. Each of the other five combinations has an
                  explicit two-entry counterexample.
-/

namespace MaxFold

/-! ## Values -/

inductive V where
  | nan : V
  | num : Rat → V
  deriving DecidableEq

def V.isNaN : V → Bool
  | .nan => true
  | .num _ => false

/-- IEEE `a > b`, false when either operand is NaN -/
def gt : V → V → Bool
  | .num a, .num b => decide (b < a)
  | _, _ => false

/-- IEEE `a < b`, false when either operand is NaN -/
def lt : V → V → Bool
  | .num a, .num b => decide (a < b)
  | _, _ => false

/-! ## The three reductions -/

/-- SuperLU `SUPERLU_MAX(s, v) = (s > v) ? s : v` -/
def smax (s v : V) : V := if gt s v then s else v

/-- PLASMA `core_dzamax` / IDAMAX step: `if (v > s) s = v` -/
def pstep (s v : V) : V := if gt v s then v else s

/-- NaN-propagating maximum -/
def nmax : V → V → V
  | .num a, .num b => .num (max a b)
  | _, _ => .nan

/-- the last entry of `xs`, or `d` if `xs` is empty -/
def lastOr : V → List V → V
  | d, [] => d
  | _, v :: vs => lastOr v vs

/-! ## SuperLU: only the last entry decides NaN-ness -/

theorem smax_isNaN (s v : V) : (smax s v).isNaN = v.isNaN := by
  unfold smax
  split
  · rename_i h; cases s <;> cases v <;> simp_all [gt, V.isNaN]
  · rfl

theorem smax_fold_isNaN : ∀ (xs : List V) (s : V), (xs.foldl smax s).isNaN = (lastOr s xs).isNaN
  | [], _ => rfl
  | v :: vs, s => by
    rw [List.foldl_cons, smax_fold_isNaN vs (smax s v)]
    cases vs with
    | nil => simp only [lastOr]; exact smax_isNaN s v
    | cons w ws => simp only [lastOr]

theorem lastOr_append_single (d v : V) : ∀ pre : List V, lastOr d (pre ++ [v]) = v
  | [] => rfl
  | p :: ps => by simp only [List.cons_append, lastOr]; exact lastOr_append_single p v ps

/-- the SuperLU fold is NaN iff the last entry is NaN (for a nonempty list) -/
theorem smax_fold_nan_iff_last (s : V) (pre : List V) (v : V) :
    ((pre ++ [v]).foldl smax s).isNaN = v.isNaN := by
  rw [smax_fold_isNaN, lastOr_append_single]

/-- whatever came before the last NaN is forgotten -/
theorem smax_fold_after_nan (s : V) (pre suf : List V) :
    (pre ++ V.nan :: suf).foldl smax s = suf.foldl smax V.nan := by
  rw [List.foldl_append, List.foldl_cons]
  have : smax (pre.foldl smax s) V.nan = V.nan := by
    cases h : pre.foldl smax s <;> simp [smax, gt]
  rw [this]

def maxR (a : Rat) : List Rat → Rat
  | [] => a
  | x :: xs => maxR (max a x) xs

theorem smax_num_num (a b : Rat) : smax (.num a) (.num b) = .num (max a b) := by
  simp only [smax, gt]
  by_cases h : b < a
  · simp only [h, decide_true, ↓reduceIte, V.num.injEq]; rw [Rat.max_def]; split <;> grind
  · simp only [h, decide_false, Bool.false_eq_true, ↓reduceIte, V.num.injEq]; rw [Rat.max_def]; split <;> grind

theorem smax_fold_nums (a : Rat) : ∀ xs : List Rat, (xs.map V.num).foldl smax (.num a) = .num (maxR a xs)
  | [] => rfl
  | x :: xs => by
    simp only [List.map_cons, List.foldl_cons, smax_num_num, maxR]; exact smax_fold_nums (max a x) xs

/-- after the last NaN, a finite tail `x :: xs` gives exactly its maximum -/
theorem smax_fold_value (s : V) (pre : List V) (x : Rat) (xs : List Rat) :
    (pre ++ V.nan :: (x :: xs).map V.num).foldl smax s = .num (maxR x xs) := by
  rw [smax_fold_after_nan, List.map_cons, List.foldl_cons]
  have : smax V.nan (V.num x) = V.num x := by simp [smax, gt]
  rw [this]; exact smax_fold_nums x xs

theorem maxR_ge_init : ∀ (a : Rat) (xs : List Rat), a ≤ maxR a xs
  | _, [] => Rat.le_refl
  | a, x :: xs => by
    simp only [maxR]
    exact Rat.le_trans (by rw [Rat.max_def]; split <;> grind) (maxR_ge_init (max a x) xs)

theorem maxR_mono {a b : Rat} (h : a ≤ b) : ∀ xs : List Rat, maxR a xs ≤ maxR b xs
  | [] => h
  | x :: xs => by
    simp only [maxR]
    exact maxR_mono (by rw [Rat.max_def, Rat.max_def]; split <;> split <;> grind) xs

theorem maxR_le_of {a c : Rat} (ha : a ≤ c) : ∀ xs : List Rat, (∀ y ∈ xs, y ≤ c) → maxR a xs ≤ c
  | [], _ => ha
  | x :: xs, h => by
    simp only [maxR]
    have hx : x ≤ c := h x (by simp)
    have hm : max a x ≤ c := by rw [Rat.max_def]; split <;> grind
    exact maxR_le_of hm xs (fun y hy => h y (List.mem_cons_of_mem x hy))

theorem mem_le_maxR (a : Rat) : ∀ (xs : List Rat), ∀ y ∈ xs, y ≤ maxR a xs
  | [], _, h => by cases h
  | x :: xs, y, h => by
    simp only [maxR]
    cases h with
    | head =>
      have h1 : x ≤ max a x := by rw [Rat.max_def]; split <;> grind
      exact Rat.le_trans h1 (maxR_ge_init (max a x) xs)
    | tail _ h' => exact mem_le_maxR (max a x) xs y h'

theorem maxR_append (a : Rat) (xs ys : List Rat) : maxR a (xs ++ ys) = maxR (maxR a xs) ys := by
  induction xs generalizing a with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append, maxR]; exact ih (max a x)

/-- taking the maximum over the same entries twice changes nothing -/
theorem maxR_twice (a : Rat) (xs : List Rat) : maxR (maxR a xs) xs = maxR a xs := by
  apply Rat.le_antisymm
  · exact maxR_le_of Rat.le_refl xs (fun y hy => mem_le_maxR a xs y hy)
  · exact maxR_ge_init _ xs

/-- The SuperLU_DIST experiment: two identical decoupled blocks, the first block's per-row values are NaN, the second
    block's are the clean values `x :: xs` (≥ 0, as backward errors are).  The masked result equals the clean baseline
    (all 2·n rows finite) exactly. -/
theorem masked_equals_baseline (x : Rat) (xs : List Rat) (hx : 0 ≤ x) (k : Nat) :
    List.foldl smax (.num 0) (List.replicate (k + 1) V.nan ++ (x :: xs).map V.num)
      = List.foldl smax (.num 0) (((x :: xs) ++ (x :: xs)).map V.num) := by
  have hrep : List.replicate (k + 1) V.nan = List.replicate k V.nan ++ [V.nan] := by
    rw [List.replicate_succ']
  rw [hrep, List.append_assoc, List.singleton_append, smax_fold_value, smax_fold_nums, maxR_append, maxR_twice]
  congr 1
  simp only [maxR]
  have : max 0 x = x := by rw [Rat.max_def]; split <;> grind
  rw [this]

/-! ## PLASMA / IDAMAX: only the first entry decides NaN-ness -/

theorem pstep_isNaN (s v : V) : (pstep s v).isNaN = s.isNaN := by
  unfold pstep
  split
  · rename_i h; cases s <;> cases v <;> simp_all [gt, V.isNaN]
  · rfl

theorem pstep_fold_isNaN : ∀ (xs : List V) (s : V), (xs.foldl pstep s).isNaN = s.isNaN
  | [], _ => rfl
  | v :: vs, s => by rw [List.foldl_cons, pstep_fold_isNaN vs, pstep_isNaN]

/-! ## The NaN-propagating fold -/

theorem nmax_fold_isNaN : ∀ (xs : List V) (s : V), (xs.foldl nmax s).isNaN = (s.isNaN || xs.any V.isNaN)
  | [], s => by simp
  | v :: vs, s => by
    rw [List.foldl_cons, nmax_fold_isNaN vs]
    cases s <;> cases v <;> simp [nmax, V.isNaN]

/-! ## Acceptance tests built on a reduction -/

/-- the two ways the acceptance test is written -/
inductive Form where
  | reject  -- `if (r > thr) keep iterating / fail else accept`  (LAPACK dsgesv, dgsrfs, PLASMA, SLATE, MAGMA zcgesv)
  | accept  -- `if (r < thr) accept`                              (cuSOLVER IRS documentation, rocHPL-MxP)
  deriving DecidableEq

def Form.test : Form → V → V → Bool
  | .reject, thr, r => !(gt r thr)
  | .accept, thr, r => lt r thr

theorem reject_accepts_nan (thr : V) : Form.reject.test thr V.nan = true := by
  cases thr <;> rfl

theorem accept_rejects_nan (thr : V) : Form.accept.test thr V.nan = false := by
  cases thr <;> rfl

/-- the three reductions as the libraries start them: SuperLU from `s = 0`, PLASMA `core_dzamax` from the first
    entry (`values[j] = |A[0,j]|`), the propagating fix from `0` -/
inductive Red where
  | superlu
  | plasma
  | propagate
  deriving DecidableEq

def Red.run : Red → List V → V
  | .superlu, xs => xs.foldl smax (.num 0)
  | .plasma, [] => .num 0
  | .plasma, x :: xs => xs.foldl pstep x
  | .propagate, xs => xs.foldl nmax (.num 0)

/-- the test rejects every list of per-row values that contains a NaN, for every positive threshold -/
def Detects (f : Form) (r : Red) : Prop :=
  ∀ (t : Rat), 0 < t → ∀ xs : List V, xs.any V.isNaN = true → f.test (.num t) (r.run xs) = false

theorem lt_nan_false (thr : V) : lt V.nan thr = false := by cases thr <;> rfl

theorem accept_propagate_detects : Detects .accept .propagate := by
  intro t _ xs hx
  have h : (Red.propagate.run xs).isNaN = true := by
    simp only [Red.run]; rw [nmax_fold_isNaN]; simp [V.isNaN, hx]
  cases hr : Red.run .propagate xs with
  | nan => rfl
  | num _ => rw [hr] at h; simp [V.isNaN] at h

/-- the five failing combinations, each with a two-entry witness and threshold 1 -/
theorem reject_superlu_misses : Form.reject.test (.num 1) (Red.superlu.run [V.nan, V.num 0]) = true := by decide
theorem accept_superlu_misses : Form.accept.test (.num 1) (Red.superlu.run [V.nan, V.num 0]) = true := by decide
theorem reject_plasma_misses : Form.reject.test (.num 1) (Red.plasma.run [V.num 0, V.nan]) = true := by decide
theorem accept_plasma_misses : Form.accept.test (.num 1) (Red.plasma.run [V.num 0, V.nan]) = true := by decide
theorem reject_propagate_misses : Form.reject.test (.num 1) (Red.propagate.run [V.nan]) = true := by decide

/-- **D11 + D1.** Over the surveyed design space, the acceptance test sees every NaN iff it is written in the
    `<` form and the reduction propagates NaN. -/
theorem detects_iff (f : Form) (r : Red) : Detects f r ↔ f = .accept ∧ r = .propagate := by
  constructor
  · intro h
    have one : (0 : Rat) < 1 := by decide
    cases f <;> cases r
    · exact absurd (h 1 one [V.nan, V.num 0] (by decide)) (by rw [reject_superlu_misses]; decide)
    · exact absurd (h 1 one [V.num 0, V.nan] (by decide)) (by rw [reject_plasma_misses]; decide)
    · exact absurd (h 1 one [V.nan] (by decide)) (by rw [reject_propagate_misses]; decide)
    · exact absurd (h 1 one [V.nan, V.num 0] (by decide)) (by rw [accept_superlu_misses]; decide)
    · exact absurd (h 1 one [V.num 0, V.nan] (by decide)) (by rw [accept_plasma_misses]; decide)
    · exact ⟨rfl, rfl⟩
  · rintro ⟨rfl, rfl⟩; exact accept_propagate_detects

end MaxFold

#print axioms MaxFold.smax_fold_isNaN
#print axioms MaxFold.smax_fold_nan_iff_last
#print axioms MaxFold.smax_fold_value
#print axioms MaxFold.masked_equals_baseline
#print axioms MaxFold.pstep_fold_isNaN
#print axioms MaxFold.nmax_fold_isNaN
#print axioms MaxFold.detects_iff

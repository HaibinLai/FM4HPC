import MaxFold

/-!
# The acceptance contract of mixed-precision iterative refinement: necessary and sufficient conditions

`operators/IR_SURVEY_zh.md` lists eleven defect classes (D1–D11) found in about 25 implementations. This file shows
that, at the level the caller observes, they are failures of **four** conditions, and that the four are together
**necessary and sufficient** for the documented contract.

## The abstract driver (`Impl`, `run`)

Every surveyed routine has this shape once the low-precision factorization has been done:
* the refinement (classical IR or an inner Krylov solver) produces a candidate `x`. It is treated as **adversarial**:
  any value can come out (garbage from an overflowed FP16/FP32 LU, NaN, a stagnated iterate, x = 0 …);
* an outer acceptance test `check p x` is evaluated (it may be a constant: no check at all);
* if it fails and the fallback is reachable (`fbReach`), the FP64 path writes `fb p`, with its own success flag;
* the routine, or a wrapper around it, may report success regardless (`force`).

## The documented contract
* `Sound`    : a reported success means the answer the caller reads is good (`Good p x`, e.g. finite and
               `‖b − A x‖ < c ‖A‖ ‖x‖`);
* `Complete` : if the problem is solvable in FP64 (`Solvable p`), the routine reports success (“if the approach
               fails the method switches to a double precision factorization and solve”).

## Main theorem (`contract_iff`)
Under three modelling hypotheses — the inner solver can return a bad value (`HBad`), a good answer only exists for a
solvable problem (`HGood`), and both solvable and unsolvable problems occur (`HSolv`, `HUnsolv`) —
`Sound ∧ Complete` holds **iff**
* **C1** the outer check is sound: `check p x = true → Good p x`;
* **C2** the fallback is reachable after a failed check;
* **C3** the fallback is correct: its success flag implies a good answer, and it succeeds on every solvable problem;
* **C4** success is not forced.

## Corollaries (each surveyed failure is a violated condition; observations in `experiments/ir_survey/`)
* `trust_inner_unsound` (C1): no outer check (MAGMA GMSTRS `goto cleanup`, MAGMA `dshposv_gmres`, PaStiX GMRES,
  STRUMPACK mixed): unsound.
* `forced_unsound` (C4): success forced (SLATE LAPACK API `*info = 0`, STRUMPACK mixed `return SUCCESS`, PaStiX
  `return PASTIX_SUCCESS`): unsound.
* `stub_fallback_fails` (C3): a fallback that reports success with a wrong answer (MAGMA `zchesv`: `zlacpy(B → X)`):
  the contract fails.
* `skipped_fallback_incomplete` (C2): the fallback is not reached (PLASMA: a failed task cancels the sequence):
  incomplete.
* `c1_forces_accept_propagate` (C1 → D1, D11): if the check is `test(reduction(per-row values))` and every NaN
  pattern is realisable by a bad input, C1 forces the `<` form **and** a NaN-propagating reduction
  (`MaxFold.detects_iff`).

What makes C1 achievable numerically is proved elsewhere: `IrsContract.wrapper_post` and `MagmaFixed.fixed_post`
(a Dot2 residual with an outward-rounded `<` test is sound for any refinement output), and
`IrsContract.fp64_cannot_certify` (a plain FP64 residual cannot certify the documented `√n·eps` threshold).
Core Lean only.
-/

namespace IrDriver
open MaxFold

/-- one implementation, seen from the caller -/
structure Impl (P X : Type) where
  check   : P → X → Bool
  fbReach : Bool
  fb      : P → X × Bool
  force   : Bool

/-- what the caller receives when the refinement produced `x`: (answer, reported success) -/
def run {P X : Type} (I : Impl P X) (p : P) (x : X) : X × Bool :=
  if I.check p x then (x, true || I.force)
  else if I.fbReach then ((I.fb p).1, (I.fb p).2 || I.force)
  else (x, false || I.force)

variable {P X : Type} (Good : P → X → Prop) (Solvable : P → Prop)

def Sound (I : Impl P X) : Prop := ∀ p x, (run I p x).2 = true → Good p (run I p x).1
def Complete (I : Impl P X) : Prop := ∀ p x, Solvable p → (run I p x).2 = true

def C1 (I : Impl P X) : Prop := ∀ p x, I.check p x = true → Good p x
def C2 (I : Impl P X) : Prop := I.fbReach = true
def C3 (I : Impl P X) : Prop := ∀ p, ((I.fb p).2 = true → Good p (I.fb p).1) ∧ (Solvable p → (I.fb p).2 = true)
def C4 (I : Impl P X) : Prop := I.force = false

theorem run_check {I : Impl P X} {p : P} {x : X} (h : I.check p x = true) : run I p x = (x, true) := by
  simp [run, h]

theorem run_fb {I : Impl P X} {p : P} {x : X} (h : I.check p x = false) (hr : I.fbReach = true)
    (hf : I.force = false) : run I p x = I.fb p := by
  simp [run, h, hr, hf]

theorem run_nofb {I : Impl P X} {p : P} {x : X} (h : I.check p x = false) (hr : I.fbReach = false)
    (hf : I.force = false) : run I p x = (x, false) := by
  simp [run, h, hr, hf]

/-- sufficiency: no modelling hypotheses needed -/
theorem contract_of_conditions (I : Impl P X) (h1 : C1 Good I) (h2 : C2 I) (h3 : C3 Good Solvable I) (h4 : C4 I) :
    Sound Good I ∧ Complete Solvable I := by
  constructor
  · intro p x hok
    cases hc : I.check p x
    · rw [run_fb hc h2 h4] at hok ⊢; exact (h3 p).1 hok
    · rw [run_check hc]; exact h1 p x hc
  · intro p x hs
    cases hc : I.check p x
    · rw [run_fb hc h2 h4]; exact (h3 p).2 hs
    · rw [run_check hc]

section necessity
variable (HBad : ∀ p, ∃ x, ¬ Good p x) (HGood : ∀ p x, Good p x → Solvable p)

theorem c1_of_sound {I : Impl P X} (hS : Sound Good I) : C1 Good I := by
  intro p x hc
  have := hS p x (by rw [run_check hc])
  rwa [run_check hc] at this

include HBad HGood in
theorem c4_of_sound {I : Impl P X} (hS : Sound Good I) (HUnsolv : ∃ p, ¬ Solvable p) : C4 I := by
  obtain ⟨p, hp⟩ := HUnsolv
  obtain ⟨x, _⟩ := HBad p
  cases hf : I.force
  · exact hf
  · exfalso
    have hok : (run I p x).2 = true := by
      unfold run; split
      · simp [hf]
      · split <;> simp [hf]
    exact hp (HGood p _ (hS p x hok))

include HBad in
theorem c2_of_contract {I : Impl P X} (hS : Sound Good I) (hC : Complete Solvable I) (h4 : C4 I)
    (HSolv : ∃ p, Solvable p) : C2 I := by
  obtain ⟨p, hp⟩ := HSolv
  obtain ⟨x, hx⟩ := HBad p
  have hc : I.check p x = false := by
    cases h : I.check p x
    · rfl
    · exact absurd (c1_of_sound Good hS p x h) hx
  cases hr : I.fbReach
  · have := hC p x hp
    rw [run_nofb hc hr h4] at this; simp at this
  · exact hr

include HBad in
theorem c3_of_contract {I : Impl P X} (hS : Sound Good I) (hC : Complete Solvable I) (h2 : C2 I) (h4 : C4 I) :
    C3 Good Solvable I := by
  intro p
  obtain ⟨x, hx⟩ := HBad p
  have hc : I.check p x = false := by
    cases h : I.check p x
    · rfl
    · exact absurd (c1_of_sound Good hS p x h) hx
  have hrun := run_fb (x := x) hc h2 h4
  refine ⟨fun hok => ?_, fun hs => ?_⟩
  · have := hS p x (by rw [hrun]; exact hok); rwa [hrun] at this
  · have := hC p x hs; rwa [hrun] at this

include HBad HGood in
/-- **Main theorem.** Under the modelling hypotheses, the documented contract holds iff C1–C4 hold. -/
theorem contract_iff (I : Impl P X) (HSolv : ∃ p, Solvable p) (HUnsolv : ∃ p, ¬ Solvable p) :
    (Sound Good I ∧ Complete Solvable I) ↔ (C1 Good I ∧ C2 I ∧ C3 Good Solvable I ∧ C4 I) := by
  constructor
  · rintro ⟨hS, hC⟩
    have h4 := c4_of_sound Good Solvable HBad HGood hS HUnsolv
    have h2 := c2_of_contract Good Solvable HBad hS hC h4 HSolv
    exact ⟨c1_of_sound Good hS, h2, c3_of_contract Good Solvable HBad hS hC h2 h4, h4⟩
  · rintro ⟨h1, h2, h3, h4⟩; exact contract_of_conditions Good Solvable I h1 h2 h3 h4

/-! ## Corollaries: the surveyed failure shapes -/

include HBad in
/-- C1 violated — no outer check (MAGMA GMSTRS / `dshposv_gmres`, PaStiX GMRES, STRUMPACK mixed) -/
theorem trust_inner_unsound {I : Impl P X} (h : ∀ p x, I.check p x = true) (p0 : P) : ¬ Sound Good I := by
  intro hS
  obtain ⟨x, hx⟩ := HBad p0
  exact hx (c1_of_sound Good hS p0 x (h p0 x))

include HBad HGood in
/-- C4 violated — success forced (SLATE LAPACK API, STRUMPACK mixed, PaStiX) -/
theorem forced_unsound {I : Impl P X} (h : I.force = true) (HUnsolv : ∃ p, ¬ Solvable p) : ¬ Sound Good I := by
  intro hS
  have := c4_of_sound Good Solvable HBad HGood hS HUnsolv
  rw [C4, h] at this; exact absurd this (by decide)

include HBad in
/-- C3 violated — the fallback reports success with a wrong answer (MAGMA `zchesv` copies B into X) -/
theorem stub_fallback_fails {I : Impl P X} (h2 : C2 I) (h4 : C4 I) (p0 : P) (hok : (I.fb p0).2 = true)
    (hbad : ¬ Good p0 (I.fb p0).1) : ¬ (Sound Good I ∧ Complete Solvable I) := by
  rintro ⟨hS, hC⟩
  exact hbad ((c3_of_contract Good Solvable HBad hS hC h2 h4 p0).1 hok)

include HBad in
/-- C2 violated — the fallback is skipped (PLASMA: the failed sequence cancels the FP64 tasks) -/
theorem skipped_fallback_incomplete {I : Impl P X} (hS : Sound Good I) (hr : I.fbReach = false) (h4 : C4 I)
    (HSolv : ∃ p, Solvable p) : ¬ Complete Solvable I := by
  intro hC
  have := c2_of_contract Good Solvable HBad hS hC h4 HSolv
  rw [C2, hr] at this; exact absurd this (by decide)

end necessity

/-! ## C1 forces the `<` form and a NaN-propagating reduction (link to `MaxFold`) -/

/-- If the outer check is `form.test thr (reduction (per-row values))`, and every NaN pattern of per-row values at
    every positive threshold is produced by some bad input (a NaN placed in `b` at chosen rows of a block-diagonal
    matrix, scaled to the threshold), then C1 forces `form = accept` and `reduction = propagate`. -/
theorem c1_forces_accept_propagate {I : Impl P X} (f : Form) (r : Red) (rows : P → X → List V) (thr : P → Rat)
    (hcheck : ∀ p x, I.check p x = f.test (.num (thr p)) (r.run (rows p x)))
    (hreal : ∀ t : Rat, 0 < t → ∀ xs : List V, xs.any V.isNaN = true →
      ∃ p x, thr p = t ∧ rows p x = xs ∧ ¬ Good p x)
    (h1 : C1 Good I) : f = .accept ∧ r = .propagate := by
  rw [← detects_iff]
  intro t ht xs hxs
  obtain ⟨p, x, hthr, hrows, hbad⟩ := hreal t ht xs hxs
  cases h : f.test (.num t) (r.run xs)
  · rfl
  · exfalso; apply hbad; apply h1 p x; rw [hcheck, hthr, hrows, h]

/-! ## Non-vacuity: a toy instance where all hypotheses hold

`P = Bool` (`true` = FP64-solvable), `X = Option Rat` (`none` = a garbage/NaN answer), the good answer is `some 1`. -/

def tGood (p : Bool) (x : Option Rat) : Prop := p = true ∧ x = some 1
def tSolv (p : Bool) : Prop := p = true

theorem t_bad : ∀ p, ∃ x, ¬ tGood p x := fun _ => ⟨none, fun h => by cases h.2⟩
theorem t_good : ∀ p x, tGood p x → tSolv p := fun _ _ h => h.1

/-- a correct implementation: certified check, reachable and correct FP64 fallback, status not forced -/
def tOk : Impl Bool (Option Rat) where
  check p x := p && x == some 1
  fbReach := true
  fb p := if p then (some 1, true) else (none, false)
  force := false

theorem tOk_contract : Sound tGood tOk ∧ Complete tSolv tOk := by
  apply contract_of_conditions
  · intro p x h
    simp only [tOk, Bool.and_eq_true, beq_iff_eq] at h
    exact ⟨h.1, h.2⟩
  · rfl
  · intro p; cases p <;> simp [tOk, tGood, tSolv]
  · rfl

/-- the same implementation with success forced (the SLATE LAPACK-API shape) is unsound -/
theorem tForced_unsound : ¬ Sound tGood { tOk with force := true } :=
  forced_unsound tGood tSolv t_bad t_good rfl ⟨false, by simp [tSolv]⟩

/-- and with the check removed (the MAGMA GMSTRS shape) it is unsound -/
theorem tTrust_unsound : ¬ Sound tGood { tOk with check := fun _ _ => true } :=
  trust_inner_unsound tGood t_bad (fun _ _ => rfl) true

end IrDriver

#print axioms IrDriver.contract_of_conditions
#print axioms IrDriver.contract_iff
#print axioms IrDriver.trust_inner_unsound
#print axioms IrDriver.forced_unsound
#print axioms IrDriver.stub_fallback_fails
#print axioms IrDriver.skipped_fallback_incomplete
#print axioms IrDriver.c1_forces_accept_propagate
#print axioms IrDriver.tOk_contract
#print axioms IrDriver.tForced_unsound
#print axioms IrDriver.tTrust_unsound

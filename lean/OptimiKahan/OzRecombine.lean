import OptimiKahan.Kahan
open OptimiKahan

/-!
# OzBLAS D_H FP64 recombination (integer-lattice model of binary64)

Models the `sumMode = 2` accumulation of `cuozblasLocalFsumKernel` (pinned OzBLAS a6f6198): for each output element the terms
T_k = scalbn(c_k, seA + seB) are accumulated in double in a fixed order: the first term is ASSIGNED, the others are added
(`devCtmp += c`).  binary64 = `lattice 53` (nearest rounding, any tie-breaking, no exponent upper bound), integers = units of the
smallest quantum.  Scaling by 2^e is exact (no overflow/underflow assumed): the terms are taken as representable lattice points.

PROVED (no `sorry`; axioms printed at the end; Kahan.lean is imported):
* `relErr53`              : 2^53·|z - rd z| ≤ |z| for the 53-bit lattice (binary64 nearest rounding, any tie-breaking).
* `recombine_gen`, `recombine_error` : for ANY ordering of N ≤ 2^52 terms, sequential double accumulation (first term assigned, then `+=`)
                            satisfies 2^53·|result - exact sum| ≤ 2(N-1)·Σ|T_k|.
* `accum_rep`, `final_axpby_exact`, `recombine_alpha1_beta0` : with alpha = 1 and beta·C = 0 the final `fma(alpha, t, beta*C)` adds no rounding,
                            so the stored result is the accumulation itself and obeys the same bound.
* `rep_mul_pow2`          : scaling a representable term UP by 2^e is exact (`scalbn`).

NOT proved here (see HANDOFF section 15): general alpha/beta (needs a finer lattice for the exact product alpha·t), scaling DOWN (terms are taken as
representable, i.e. no underflow/overflow), the use of the level order (the bound holds for every order and is stated in Σ|T_k|, not in the input
magnitudes), the truncation tail and the composition with the slicing lemma into an end-to-end GEMM bound, non-finite C, signed zeros, and that the
compiled kernel performs exactly one binary64 rounding per addition.
-/
namespace OzRecombine

/-! ## Relative rounding error 2^-53 of the 53-bit lattice -/

theorem close_nat53 (a : Nat) : ∃ b : Nat, Rep 53 (b : Int) ∧ 9007199254740992 * ((a : Int) - b).natAbs ≤ a := by
  by_cases ha : a < 9007199254740992
  · exact ⟨a, ⟨a, 0, by simp, by omega⟩, by omega⟩
  · have hapos : a ≠ 0 := by omega
    have hk1 : 2 ^ a.log2 ≤ a := Nat.log2_self_le hapos
    have hk2 : a < 2 ^ (a.log2 + 1) := Nat.lt_log2_self
    have hk53 : 53 ≤ a.log2 := by
      by_cases h : a.log2 < 53
      · have : 2 ^ (a.log2 + 1) ≤ 2 ^ 53 := Nat.pow_le_pow_right (by decide) (by omega)
        omega
      · omega
    obtain ⟨s, hs⟩ : ∃ s, a.log2 = s + 52 := ⟨a.log2 - 52, by omega⟩
    rw [hs] at hk1 hk2
    have e1 : 2 ^ (s + 52) = 2 ^ s * 4503599627370496 := by rw [Nat.pow_add]
    have e2 : 2 ^ (s + 52 + 1) = 2 ^ s * 9007199254740992 := by rw [Nat.pow_succ, e1]; omega
    have e3 : 2 ^ (s + 1) = 2 ^ s * 2 := by rw [Nat.pow_succ]
    have hqpos : 0 < 2 ^ s := Nat.two_pow_pos s
    obtain ⟨q, hq⟩ : ∃ q, q = 2 ^ s := ⟨_, rfl⟩
    rw [← hq] at e1 e2 e3 hqpos
    rw [e1] at hk1
    rw [e2] at hk2
    have hdm := Nat.div_add_mod a q
    have hr : a % q < q := Nat.mod_lt _ hqpos
    have hm : a / q < 9007199254740992 := (Nat.div_lt_iff_lt_mul hqpos).2 (by omega)
    obtain ⟨m, hmdef⟩ : ∃ m, m = a / q := ⟨_, rfl⟩
    obtain ⟨r, hrdef⟩ : ∃ r, r = a % q := ⟨_, rfl⟩
    rw [← hmdef, ← hrdef] at hdm
    rw [← hrdef] at hr
    rw [← hmdef] at hm
    obtain ⟨Q, hQ⟩ : ∃ Q, Q = q * m := ⟨_, rfl⟩
    rw [← hQ] at hdm
    have hQq : Q + q = (m + 1) * q := by
      rw [hQ, Nat.add_mul, Nat.one_mul, Nat.mul_comm]
    have hrepQ : Rep 53 (Q : Int) := ⟨m, s, by rw [Int.natAbs_natCast, hQ, hq, Nat.mul_comm], by omega⟩
    have hrepQ' : Rep 53 ((Q + q : Nat) : Int) := by
      by_cases hm1 : m + 1 < 9007199254740992
      · exact ⟨m + 1, s, by rw [Int.natAbs_natCast, hQq, hq], by omega⟩
      · refine ⟨4503599627370496, s + 1, ?_, by omega⟩
        have hm53 : m + 1 = 9007199254740992 := by omega
        rw [Int.natAbs_natCast, hQq, hm53, e3, hq]
        omega
    by_cases hc : 2 * r ≤ q
    · exact ⟨Q, hrepQ, by omega⟩
    · exact ⟨Q + q, hrepQ', by omega⟩

theorem exists_close53 (z : Int) : ∃ y, Rep 53 y ∧ 9007199254740992 * (z - y).natAbs ≤ z.natAbs := by
  obtain ⟨a, rfl | rfl⟩ := Int.eq_nat_or_neg z
  · obtain ⟨b, hb, h⟩ := close_nat53 a
    exact ⟨b, hb, by simpa using h⟩
  · obtain ⟨b, ⟨m, e, hm, hlt⟩, h⟩ := close_nat53 a
    refine ⟨-(b : Int), ⟨m, e, by simpa using hm, hlt⟩, ?_⟩
    omega

/-- Relative error of binary64 nearest rounding: |z - rd z| ≤ 2^-53 |z|. -/
theorem relErr53 (z : Int) : 9007199254740992 * (z - rdNear 53 z).natAbs ≤ z.natAbs := by
  obtain ⟨y, hy, hle⟩ := exists_close53 z
  have hn : (z - rdNear 53 z).natAbs ≤ (z - y).natAbs := (lattice 53).rd_nearest z y hy
  omega


/-! ## Sequential accumulation in double (first term assigned, then `+=`) -/

/-- The kernel's accumulation: `devCtmp = c` for the first term, then `devCtmp += c` for the others (each addition rounded to binary64). -/
noncomputable def accum : List Int → Int
  | [] => 0
  | t :: ts => ts.foldl (fun a x => rdNear 53 (a + x)) t

/-- Error invariant: 2^53·|acc - S| ≤ 2(k-1)·A where S is the exact partial sum and A the sum of absolute values. -/
theorem recombine_gen (xs : List Int) :
    ∀ (acc S : Int) (A k : Nat), 1 ≤ k → S.natAbs ≤ A →
      9007199254740992 * (acc - S).natAbs ≤ 2 * ((k - 1) * A) →
      k + xs.length ≤ 4503599627370497 →
      9007199254740992 * (xs.foldl (fun a x => rdNear 53 (a + x)) acc - (S + xs.sum)).natAbs
        ≤ 2 * ((k - 1 + xs.length) * (A + (xs.map Int.natAbs).sum)) := by
  induction xs with
  | nil => intro acc S A k _ _ hI _; simpa using hI
  | cons x xs ih =>
    intro acc S A k hk hS hI hlen
    have hrel := relErr53 (acc + x)
    have h1 : (k - 1) * A ≤ 4503599627370496 * A := Nat.mul_le_mul_right A (by simp at hlen; omega)
    have hEA : (acc - S).natAbs ≤ A := by omega
    have hmono : (k - 1) * A ≤ (k - 1) * (A + x.natAbs) := Nat.mul_le_mul_left (k - 1) (by omega)
    have hkA : (k - 1 + 1) * (A + x.natAbs) = (k - 1) * (A + x.natAbs) + (A + x.natAbs) := Nat.succ_mul _ _
    have hk1 : k - 1 + 1 = k := by omega
    rw [hk1] at hkA
    have hI' : 9007199254740992 * (rdNear 53 (acc + x) - (S + x)).natAbs
        ≤ 2 * ((k + 1 - 1) * (A + x.natAbs)) := by
      rw [Nat.add_sub_cancel]
      omega
    have hlen' : (k + 1) + xs.length ≤ 4503599627370497 := by simp at hlen; omega
    have := ih (rdNear 53 (acc + x)) (S + x) (A + x.natAbs) (k + 1) (by omega) (by omega) hI' hlen'
    simp only [List.foldl_cons, List.sum_cons, List.map_cons, List.length_cons]
    have e1 : k - 1 + (xs.length + 1) = k + 1 - 1 + xs.length := by omega
    have e2 : A + (x.natAbs + (xs.map Int.natAbs).sum) = A + x.natAbs + (xs.map Int.natAbs).sum := by omega
    have e3 : S + (x + xs.sum) = S + x + xs.sum := by omega
    rw [e1, e2, e3]
    exact this

/-- **FP64 recombination bound**: for ANY ordering of N ≤ 2^52 terms, sequential double accumulation with the first term assigned has
    2^53·|result - exact sum| ≤ 2(N-1)·Σ|T_k|. -/
theorem recombine_error (t : Int) (ts : List Int) (hlen : ts.length ≤ 4503599627370496) :
    9007199254740992 * (accum (t :: ts) - (t + ts.sum)).natAbs
      ≤ 2 * (ts.length * (t.natAbs + (ts.map Int.natAbs).sum)) := by
  unfold accum
  have := recombine_gen ts t t t.natAbs 1 (by omega) (by omega) (by simp) (by omega)
  simpa using this

theorem foldl_rep (xs : List Int) : ∀ acc : Int, Rep 53 acc → Rep 53 (xs.foldl (fun a x => rdNear 53 (a + x)) acc) := by
  induction xs with
  | nil => intro acc h; exact h
  | cons x xs ih => intro acc _; exact ih _ ((lattice 53).rd_rep _)

theorem accum_rep (t : Int) (ts : List Int) (ht : Rep 53 t) : Rep 53 (accum (t :: ts)) := foldl_rep ts t ht

/-- Final `C = fma(alpha, t, beta*C)` with alpha = 1 and beta*C = 0 (beta = 0, finite C): no further rounding. -/
theorem final_axpby_exact (t : Int) (ht : Rep 53 t) : rdNear 53 (t + rdNear 53 0) = t := by
  rw [rd_self (rep_zero 53), Int.add_zero]
  exact rd_self ht

/-- Scaling a representable term by 2^e (e ≥ 0) is exact (`scalbn` up; scaling down needs the no-underflow assumption). -/
theorem rep_mul_pow2 {p : Nat} {z : Int} (h : Rep p z) (e : Nat) : Rep p (z * ((2 ^ e : Nat) : Int)) := by
  obtain ⟨m, k, h1, h2⟩ := h
  refine ⟨m, k + e, ?_, h2⟩
  rw [Int.natAbs_mul, Int.natAbs_natCast, h1, Nat.pow_add]
  ac_rfl

/-- Recombination for alpha = 1, beta = 0: the stored result IS the accumulation, and it is within 2(N-1)·2^-53·Σ|T_k| of the exact sum. -/
theorem recombine_alpha1_beta0 (t : Int) (ts : List Int) (ht : Rep 53 t) (hlen : ts.length ≤ 4503599627370496) :
    rdNear 53 (accum (t :: ts) + rdNear 53 0) = accum (t :: ts) ∧
    9007199254740992 * (rdNear 53 (accum (t :: ts) + rdNear 53 0) - (t + ts.sum)).natAbs
      ≤ 2 * (ts.length * (t.natAbs + (ts.map Int.natAbs).sum)) := by
  have hfin := final_axpby_exact _ (accum_rep t ts ht)
  refine ⟨hfin, ?_⟩
  rw [hfin]
  exact recombine_error t ts hlen

#print axioms OzRecombine.relErr53
#print axioms OzRecombine.recombine_gen
#print axioms OzRecombine.recombine_error
#print axioms OzRecombine.accum_rep
#print axioms OzRecombine.final_axpby_exact
#print axioms OzRecombine.rep_mul_pow2
#print axioms OzRecombine.recombine_alpha1_beta0

end OzRecombine

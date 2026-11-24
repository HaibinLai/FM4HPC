/-!
# The coarsened Exponent Span Capacity (ESC) of cuBLAS ADP, with zeros: when is it safe, what does it cost?

Model of arXiv 2511.13778 §4 (the paper behind cuBLAS FP64 emulation, "ADP").  For one dot product `x·y`, ADP
reserves `ESC = exp(x_p) + exp(y_q) − ẑ_r (+1)` extra bits, where `z_r` is the largest product exponent
`max_l (exp x_l + exp y_l)`.  To make this cheap, `ẑ_r` is estimated block-wise: for a block,
`est = max(Max(x_b) + Min(y_b), Min(x_b) + Max(y_b))`, and `ẑ_r` is the max over blocks.

The estimate is SAFE iff it never exceeds the true `z_r` (then ESC is not underestimated).  The paper's
argument implicitly assumes no zeros.  A zero has no exponent; an implementation must code it by some
integer `Z`.  We model an entry exponent as `Option Int` (`none` = zero, coded as `Z`).

Proved:
* `coarse_safe`     : if every real exponent lies in `[lo, hi]`, `Z ≤ lo` and `Z + hi < 2·lo`, then for any
                      block containing at least one pair of nonzeros, some real pair `(x, y)` in the block has
                      `est ≤ x + y`.  (Contract: the zero code must satisfy `Z + hi < 2·lo`.)
* `coarse_unsafe`   : conversely, if `2·lo < Z + hi` (and `Z ≤ lo ≤ hi`), the two-entry block
                      `x = [hi, lo]`, `y = [0, lo]` has `est > lo + lo`, its only real product: the
                      estimate exceeds the true `z_r`, i.e. ESC is underestimated (an accuracy hazard).
* `zero_block_gap`  : if a block contains a zero in `x` and a zero in `y`, then `est ≤ hi + Z`; so
                      whenever every block has such zeros, `ẑ_r ≤ hi + Z` and the coarsened ESC exceeds the
                      exact ESC by at least `z_r − (hi + Z)` bits: with a very negative `Z` this is
                      thousands of bits (the observed 4248-bit requests and fallbacks to native FP64).
For binary64, real exponents lie in `[-1074, 1023]`, so safety needs `Z ≤ -3172`; the observed 4248-bit
request on uniform data (exponents ≈ -1) is consistent with `Z ≈ -4195`, which satisfies it.
-/

namespace CublasEsc

/-- exponents of `(x_l, y_l)`; `none` encodes the value zero -/
abbrev Entry := Option Int × Option Int

def code (Z : Int) (o : Option Int) : Int := o.getD Z

/-- every nonzero exponent lies in `[lo, hi]` -/
def InRange (lo hi : Int) (b : List Entry) : Prop :=
  ∀ p ∈ b, (∀ e, p.1 = some e → lo ≤ e ∧ e ≤ hi) ∧ (∀ e, p.2 = some e → lo ≤ e ∧ e ≤ hi)

theorem code_le_hi (Z lo hi : Int) (hZ : Z ≤ lo) (hlh : lo ≤ hi) (o : Option Int)
    (h : ∀ e, o = some e → lo ≤ e ∧ e ≤ hi) : code Z o ≤ hi := by
  cases o with
  | none => simp [code]; omega
  | some e => simp [code]; exact (h e rfl).2

theorem code_eq_Z_of_none (Z : Int) : code Z none = Z := rfl

/-- one half of the estimate: `Max(x_b) + Min(y_b)` is dominated by a real pair -/
theorem half_safe (Z lo hi : Int) (b : List Entry) (hR : InRange lo hi b) (hZ : Z ≤ lo) (hlh : lo ≤ hi)
    (hZb : Z + hi < 2 * lo) (MX mY : Int)
    (hMX : ∃ p ∈ b, code Z p.1 = MX) (hmY : ∀ p ∈ b, mY ≤ code Z p.2)
    (x0 y0 : Int) (h0 : (some x0, some y0) ∈ b) :
    ∃ x y, (some x, some y) ∈ b ∧ MX + mY ≤ x + y := by
  obtain ⟨p1, hp1, hc1⟩ := hMX
  have hx0 := (hR _ h0).1 x0 rfl
  have hy0 := (hR _ h0).2 y0 rfl
  rcases p1 with ⟨o1, o2⟩
  cases o2 with
  | none =>
    -- the partner of the x-maximum is a zero, so Min(y) ≤ Z
    have hm := hmY _ hp1
    simp [code] at hm
    have hle : MX ≤ hi := by
      rw [← hc1]; exact code_le_hi Z lo hi hZ hlh o1 ((hR _ hp1).1)
    exact ⟨x0, y0, h0, by omega⟩
  | some y1 =>
    cases o1 with
    | none =>
      -- the x-maximum is a zero itself: MX = Z
      simp [code] at hc1
      have hm := hmY _ h0
      simp [code] at hm
      exact ⟨x0, y0, h0, by omega⟩
    | some x1 =>
      simp [code] at hc1
      have hm := hmY _ hp1
      simp [code] at hm
      exact ⟨x1, y1, hp1, by omega⟩

/-- the other half: `Min(x_b) + Max(y_b)` -/
theorem half_safe' (Z lo hi : Int) (b : List Entry) (hR : InRange lo hi b) (hZ : Z ≤ lo) (hlh : lo ≤ hi)
    (hZb : Z + hi < 2 * lo) (mX MY : Int)
    (hmX : ∀ p ∈ b, mX ≤ code Z p.1) (hMY : ∃ p ∈ b, code Z p.2 = MY)
    (x0 y0 : Int) (h0 : (some x0, some y0) ∈ b) :
    ∃ x y, (some x, some y) ∈ b ∧ mX + MY ≤ x + y := by
  obtain ⟨p1, hp1, hc1⟩ := hMY
  have hx0 := (hR _ h0).1 x0 rfl
  have hy0 := (hR _ h0).2 y0 rfl
  rcases p1 with ⟨o1, o2⟩
  cases o1 with
  | none =>
    have hm := hmX _ hp1
    simp [code] at hm
    have hle : MY ≤ hi := by
      rw [← hc1]; exact code_le_hi Z lo hi hZ hlh o2 ((hR _ hp1).2)
    exact ⟨x0, y0, h0, by omega⟩
  | some x1 =>
    cases o2 with
    | none =>
      simp [code] at hc1
      have hm := hmX _ h0
      simp [code] at hm
      exact ⟨x0, y0, h0, by omega⟩
    | some y1 =>
      simp [code] at hc1
      have hm := hmX _ hp1
      simp [code] at hm
      exact ⟨x1, y1, hp1, by omega⟩

/-- **safety contract**: with `Z ≤ lo` and `Z + hi < 2·lo`, the block estimate never exceeds a real
    product exponent of the block (so ESC is never underestimated) -/
theorem coarse_safe (Z lo hi : Int) (b : List Entry) (hR : InRange lo hi b) (hZ : Z ≤ lo) (hlh : lo ≤ hi)
    (hZb : Z + hi < 2 * lo) (MX mX MY mY : Int)
    (hMX : ∃ p ∈ b, code Z p.1 = MX) (hmX : ∀ p ∈ b, mX ≤ code Z p.1)
    (hMY : ∃ p ∈ b, code Z p.2 = MY) (hmY : ∀ p ∈ b, mY ≤ code Z p.2)
    (x0 y0 : Int) (h0 : (some x0, some y0) ∈ b) :
    ∃ x y, (some x, some y) ∈ b ∧ max (MX + mY) (mX + MY) ≤ x + y := by
  obtain ⟨x1, y1, h1, e1⟩ := half_safe Z lo hi b hR hZ hlh hZb MX mY hMX hmY x0 y0 h0
  obtain ⟨x2, y2, h2, e2⟩ := half_safe' Z lo hi b hR hZ hlh hZb mX MY hmX hMY x0 y0 h0
  by_cases hc : MX + mY ≤ mX + MY
  · exact ⟨x2, y2, h2, by rw [Int.max_eq_right hc]; exact e2⟩
  · exact ⟨x1, y1, h1, by rw [Int.max_eq_left (by omega)]; exact e1⟩

/-- **the contract is necessary**: if `2·lo < Z + hi`, the block `x = [hi, lo]`, `y = [0, lo]` has
    `Max(x) = hi`, `Min(y) = Z`, hence `est ≥ hi + Z > lo + lo`, while its only real product exponent is
    `lo + lo`: the estimate overshoots the true `z_r` and ESC is underestimated -/
theorem coarse_unsafe (Z lo hi : Int) (hZ : Z ≤ lo) (hlh : lo ≤ hi) (hbad : 2 * lo < Z + hi) :
    let b : List Entry := [(some hi, none), (some lo, some lo)]
    (∀ p ∈ b, code Z p.1 ≤ hi) ∧ (∃ p ∈ b, code Z p.1 = hi) ∧
    (∀ p ∈ b, Z ≤ code Z p.2) ∧ (∃ p ∈ b, code Z p.2 = Z) ∧
    (∀ x y, (some x, some y) ∈ b → x + y = lo + lo) ∧
    lo + lo < max (hi + Z) (Z + lo) := by
  intro b
  refine ⟨?_, ⟨(some hi, none), by simp [b], by simp [code]⟩, ?_, ⟨(some hi, none), by simp [b], by simp [code]⟩, ?_, ?_⟩
  · intro p hp
    simp [b] at hp
    rcases hp with rfl | rfl <;> simp [code] <;> omega
  · intro p hp
    simp [b] at hp
    rcases hp with rfl | rfl <;> simp [code] <;> omega
  · intro x y h
    simp [b] at h
    omega
  · omega

/-- **the cost of safety**: a block with a zero in `x` and a zero in `y` has estimate at most `hi + Z` -/
theorem zero_block_gap (Z lo hi : Int) (b : List Entry) (hR : InRange lo hi b) (hZ : Z ≤ lo) (hlh : lo ≤ hi)
    (MX mX MY mY : Int)
    (hMX : ∃ p ∈ b, code Z p.1 = MX) (hmY : ∀ p ∈ b, mY ≤ code Z p.2)
    (hmX : ∀ p ∈ b, mX ≤ code Z p.1) (hMY : ∃ p ∈ b, code Z p.2 = MY)
    (zx : ∃ p ∈ b, p.1 = none) (zy : ∃ p ∈ b, p.2 = none) :
    max (MX + mY) (mX + MY) ≤ hi + Z := by
  obtain ⟨p, hp, hcx⟩ := hMX
  obtain ⟨q, hq, hcy⟩ := hMY
  obtain ⟨u, hu, hu1⟩ := zx
  obtain ⟨v, hv, hv2⟩ := zy
  have h1 : MX ≤ hi := by rw [← hcx]; exact code_le_hi Z lo hi hZ hlh p.1 ((hR _ hp).1)
  have h2 : MY ≤ hi := by rw [← hcy]; exact code_le_hi Z lo hi hZ hlh q.2 ((hR _ hq).2)
  have h3 : mY ≤ Z := by have := hmY _ hv; rw [hv2] at this; simpa [code] using this
  have h4 : mX ≤ Z := by have := hmX _ hu; rw [hu1] at this; simpa [code] using this
  omega

/-- binary64: real exponents in `[-1074, 1023]`; the safety contract becomes `Z ≤ -3172` -/
theorem binary64_contract (Z : Int) : (Z ≤ -1074 ∧ Z + 1023 < 2 * (-1074)) ↔ Z ≤ -3172 := by
  omega

#print axioms coarse_safe
#print axioms coarse_unsafe
#print axioms zero_block_gap
#print axioms binary64_contract

end CublasEsc

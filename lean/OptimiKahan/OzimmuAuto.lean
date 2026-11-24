/-!
# What ozIMMU's `fp64_int8_auto` mantissa-loss counter certifies (ozIMMU 08eea92, `src/split.cu`)

Integer model (core Lean only).  A nonzero FP64 entry of one row of A (or one column of B) has magnitude
`v = M * 2^q` with a 53-bit significand `2^52 ≤ M` (normal number), so its leading bit is at position `q + 52`
(`mask_exponent(a)` in the code).  `max_exp = x2(get_exp_max_element(row))` has its single bit at position `E`
(`E ≥ q + 53` for every entry of the row).

`cut_int8_core` aligns the significand so that position `E` is bit 127 of a 128-bit word and emits `s` slices
of `b = get_bits_per_int8(k)` bits from the top.  With `S = s * b`, the slices keep exactly the bits at
positions `≥ g = E + 1 - S` (sign-magnitude, i.e. truncation of `v` toward zero): `kept v g = v / 2^g * 2^g`.

`calculate_mantissa_loss_core` computes, per entry,
`required = (E - (q + 52)) + 53` and `loss = required - S` (0 if negative); the auto mode sums `loss` over
all `m*k + k*n` entries and picks the first `s` whose AVERAGE is `≤ threshold`.

Proved below:
* `cut_exact`        : `loss = 0` (i.e. `required ≤ S`)  ⇒ the entry is represented exactly.
* `cut_rel_err`      : in general, truncation error `< 2^(loss-52) · |entry|`.
* `cut_total_loss`   : if the entry lies wholly below the grid, all of it is lost (`kept = 0`).
* `dot_bound`        : for one output entry, with every A-entry loss `≤ La` and every B-entry loss `≤ Lb`,
                       `|Σ a b − Σ â b̂| · 2^52 ≤ (2^La + 2^Lb) · Σ |a||b|`  (componentwise, DGEMM-style).
                       So the guarantee is governed by the MAXIMUM loss on the row/column, not the average.
* `avg_not_max`      : an average `≤ θ` over `N` entries admits one entry with loss `θ·N` (`kept = 0` then).
* `int32_exact`      : `b = min 7 ((31 - c)/2)` with `k ≤ 2^c` keeps every INT8 slice dot product in int32.
Not modelled: the FP64 accumulation of the int32 slice products (gemm.cu), nor which slice pairs are summed.
-/

namespace OzimmuAuto

/-- magnitude kept by `s` slices when the lowest kept bit position is `g` -/
def kept (v g : Nat) : Nat := v / 2 ^ g * 2 ^ g

theorem kept_le (v g : Nat) : kept v g ≤ v := Nat.div_mul_le_self v (2 ^ g)

theorem err_eq (v g : Nat) : v - kept v g = v % 2 ^ g := by
  unfold kept
  have h := Nat.div_add_mod v (2 ^ g)
  rw [Nat.mul_comm] at h
  omega

/-- the code's `required` / `loss` for an entry with leading bit `q + 52`, row max bit `E`, `S = s*b` bits -/
def required (E q : Nat) : Nat := (E - (q + 52)) + 53
def loss (E q S : Nat) : Nat := required E q - S

/-- loss = 0 ⇒ exact -/
theorem cut_exact (M q E S : Nat) (hq : q + 52 ≤ E) (h : loss E q S = 0) :
    kept (M * 2 ^ q) (E + 1 - S) = M * 2 ^ q := by
  unfold loss required at h
  have hg : E + 1 - S ≤ q := by omega
  have hd : 2 ^ (E + 1 - S) ∣ M * 2 ^ q :=
    Nat.dvd_trans (Nat.pow_dvd_pow 2 hg) (Nat.dvd_mul_left (2 ^ q) M)
  unfold kept
  exact Nat.div_mul_cancel hd

/-- truncation error `(v mod 2^g)` is `< 2^(loss-52) · v` -/
theorem cut_rel_err (M q E S : Nat) (hM : 2 ^ 52 ≤ M) (hq : q + 52 ≤ E) :
    (M * 2 ^ q) % 2 ^ (E + 1 - S) * 2 ^ 52 < 2 ^ loss E q S * (M * 2 ^ q) := by
  have hg : E + 1 - S ≤ q + loss E q S := by unfold loss required; omega
  have h1 : (M * 2 ^ q) % 2 ^ (E + 1 - S) < 2 ^ (q + loss E q S) :=
    Nat.lt_of_lt_of_le (Nat.mod_lt _ (Nat.two_pow_pos _))
      (Nat.pow_le_pow_right (by decide) hg)
  have h2 : 2 ^ (q + loss E q S) * 2 ^ 52 ≤ 2 ^ loss E q S * (M * 2 ^ q) := by
    rw [Nat.pow_add]
    calc 2 ^ q * 2 ^ loss E q S * 2 ^ 52 = 2 ^ loss E q S * (2 ^ 52 * 2 ^ q) := by
          rw [Nat.mul_comm (2 ^ q), Nat.mul_assoc, Nat.mul_comm (2 ^ q)]
      _ ≤ 2 ^ loss E q S * (M * 2 ^ q) :=
          Nat.mul_le_mul_left _ (Nat.mul_le_mul_right _ hM)
  exact Nat.lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right h1 (Nat.two_pow_pos _)) h2

/-- an entry wholly below the grid is lost completely -/
theorem cut_total_loss (v g : Nat) (h : v < 2 ^ g) : kept v g = 0 := by
  unfold kept
  rw [Nat.div_eq_of_lt h, Nat.zero_mul]

/-! ## One output entry: componentwise bound in terms of the maximum loss -/

structure Q where
  a : Int
  ah : Int
  b : Int
  bh : Int

def exactDot : List Q → Int
  | [] => 0
  | q :: t => q.a * q.b + exactDot t

def slicedDot : List Q → Int
  | [] => 0
  | q :: t => q.ah * q.bh + slicedDot t

def absDot : List Q → Nat
  | [] => 0
  | q :: t => q.a.natAbs * q.b.natAbs + absDot t

/-- per-term hypotheses: relative truncation error `2^(L-52)`, and truncation toward zero for `â` -/
def Good (La Lb : Nat) (q : Q) : Prop :=
  (q.a - q.ah).natAbs * 2 ^ 52 ≤ 2 ^ La * q.a.natAbs ∧ q.ah.natAbs ≤ q.a.natAbs ∧
  (q.b - q.bh).natAbs * 2 ^ 52 ≤ 2 ^ Lb * q.b.natAbs

theorem term_bound (La Lb : Nat) (q : Q) (h : Good La Lb q) :
    (q.a * q.b - q.ah * q.bh).natAbs * 2 ^ 52 ≤ (2 ^ La + 2 ^ Lb) * (q.a.natAbs * q.b.natAbs) := by
  obtain ⟨ha, hah, hb⟩ := h
  have e : q.a * q.b - q.ah * q.bh = (q.a - q.ah) * q.b + q.ah * (q.b - q.bh) := by
    rw [Int.sub_mul, Int.mul_sub]; omega
  have h1 : (q.a * q.b - q.ah * q.bh).natAbs ≤
      (q.a - q.ah).natAbs * q.b.natAbs + q.ah.natAbs * (q.b - q.bh).natAbs := by
    rw [e]
    have := Int.natAbs_add_le ((q.a - q.ah) * q.b) (q.ah * (q.b - q.bh))
    rw [Int.natAbs_mul, Int.natAbs_mul] at this
    exact this
  calc (q.a * q.b - q.ah * q.bh).natAbs * 2 ^ 52
      ≤ ((q.a - q.ah).natAbs * q.b.natAbs + q.ah.natAbs * (q.b - q.bh).natAbs) * 2 ^ 52 :=
        Nat.mul_le_mul_right _ h1
    _ = (q.a - q.ah).natAbs * 2 ^ 52 * q.b.natAbs + q.ah.natAbs * ((q.b - q.bh).natAbs * 2 ^ 52) := by
        rw [Nat.add_mul, Nat.mul_right_comm]; simp only [Nat.mul_assoc]
    _ ≤ 2 ^ La * q.a.natAbs * q.b.natAbs + q.a.natAbs * (2 ^ Lb * q.b.natAbs) :=
        Nat.add_le_add (Nat.mul_le_mul_right _ ha) (Nat.mul_le_mul hah hb)
    _ = (2 ^ La + 2 ^ Lb) * (q.a.natAbs * q.b.natAbs) := by
        rw [Nat.add_mul, Nat.mul_assoc, Nat.mul_left_comm q.a.natAbs]

theorem dot_bound (La Lb : Nat) : ∀ l : List Q, (∀ q ∈ l, Good La Lb q) →
    (exactDot l - slicedDot l).natAbs * 2 ^ 52 ≤ (2 ^ La + 2 ^ Lb) * absDot l
  | [], _ => by simp [exactDot, slicedDot, absDot]
  | q :: t, h => by
    have hq := term_bound La Lb q (h q (List.mem_cons_self ..))
    have ht := dot_bound La Lb t (fun x hx => h x (List.mem_cons_of_mem _ hx))
    have e : exactDot (q :: t) - slicedDot (q :: t) =
        (q.a * q.b - q.ah * q.bh) + (exactDot t - slicedDot t) := by
      simp only [exactDot, slicedDot]; omega
    rw [e]
    have := Int.natAbs_add_le (q.a * q.b - q.ah * q.bh) (exactDot t - slicedDot t)
    calc ((q.a * q.b - q.ah * q.bh) + (exactDot t - slicedDot t)).natAbs * 2 ^ 52
        ≤ ((q.a * q.b - q.ah * q.bh).natAbs + (exactDot t - slicedDot t).natAbs) * 2 ^ 52 :=
          Nat.mul_le_mul_right _ this
      _ = (q.a * q.b - q.ah * q.bh).natAbs * 2 ^ 52 + (exactDot t - slicedDot t).natAbs * 2 ^ 52 :=
          Nat.add_mul ..
      _ ≤ (2 ^ La + 2 ^ Lb) * (q.a.natAbs * q.b.natAbs) + (2 ^ La + 2 ^ Lb) * absDot t :=
          Nat.add_le_add hq ht
      _ = (2 ^ La + 2 ^ Lb) * absDot (q :: t) := by simp only [absDot]; rw [Nat.mul_add]

/-- the sliced entry (sign `σ`, magnitude truncated) satisfies the A-side hypotheses of `Good`
    with `La` any bound on its loss -/
theorem good_side (M q E S L : Nat) (hM : 2 ^ 52 ≤ M) (hq : q + 52 ≤ E) (hL : loss E q S ≤ L)
    (neg : Bool) :
    let v := M * 2 ^ q
    let a : Int := if neg then -(v : Int) else v
    let ah : Int := if neg then -((kept v (E + 1 - S) : Nat) : Int) else (kept v (E + 1 - S) : Nat)
    (a - ah).natAbs * 2 ^ 52 ≤ 2 ^ L * a.natAbs ∧ ah.natAbs ≤ a.natAbs := by
  intro v a ah
  have hk := kept_le v (E + 1 - S)
  have he := err_eq v (E + 1 - S)
  have hr := cut_rel_err M q E S hM hq
  have hp : 2 ^ loss E q S ≤ 2 ^ L := Nat.pow_le_pow_right (by decide) hL
  have habs : (a - ah).natAbs = v - kept v (E + 1 - S) ∧ a.natAbs = v ∧ ah.natAbs = kept v (E + 1 - S) := by
    cases neg <;> simp [a, ah] <;> omega
  obtain ⟨h1, h2, h3⟩ := habs
  rw [h1, h2, h3, he]
  refine ⟨?_, hk⟩
  exact Nat.le_of_lt (Nat.lt_of_lt_of_le hr (Nat.mul_le_mul_right _ hp))

/-! ## Average versus maximum -/

def sumL : List Nat → Nat
  | [] => 0
  | x :: t => x + sumL t

theorem sumL_replicate_zero (n : Nat) : sumL (List.replicate n 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [List.replicate, sumL, ih]

/-- For any threshold `θ` and count `N ≥ 1`, the average criterion `Σ loss ≤ θ·N` admits one entry
    with loss `θ·N` (all other entries exact).  By `cut_total_loss`, an entry whose loss exceeds its
    own 53 significant bits has `kept = 0`. -/
theorem avg_not_max (θ N : Nat) (hN : 1 ≤ N) :
    ∃ ls : List Nat, ls.length = N ∧ sumL ls ≤ θ * N ∧ θ * N ∈ ls := by
  refine ⟨(θ * N) :: List.replicate (N - 1) 0, ?_, ?_, List.mem_cons_self ..⟩
  · simp; omega
  · simp [sumL, sumL_replicate_zero]

/-! ## The INT8 slice dot products are exact in int32 -/

/-- `get_bits_per_int8`: `b = min 7 ((31 - c)/2)` with `c = ceil(log2 k)`, i.e. `k ≤ 2^c`.
    Slices have magnitude `< 2^b`, so a length-`k` slice dot product has magnitude `≤ k (2^b-1)^2 < 2^31`. -/
theorem int32_exact (k c : Nat) (hk : k ≤ 2 ^ c) (hc : c ≤ 31) :
    k * (2 ^ (min 7 ((31 - c) / 2)) * 2 ^ (min 7 ((31 - c) / 2))) ≤ 2 ^ 31 := by
  have hb : c + (min 7 ((31 - c) / 2) + min 7 ((31 - c) / 2)) ≤ 31 := by omega
  calc k * (2 ^ (min 7 ((31 - c) / 2)) * 2 ^ (min 7 ((31 - c) / 2)))
      ≤ 2 ^ c * (2 ^ (min 7 ((31 - c) / 2)) * 2 ^ (min 7 ((31 - c) / 2))) := Nat.mul_le_mul_right _ hk
    _ = 2 ^ (c + (min 7 ((31 - c) / 2) + min 7 ((31 - c) / 2))) := by rw [Nat.pow_add, Nat.pow_add]
    _ ≤ 2 ^ 31 := Nat.pow_le_pow_right (by decide) hb

theorem slice_prod_lt (b : Nat) (x y : Nat) (hx : x < 2 ^ b) (hy : y < 2 ^ b) :
    x * y < 2 ^ b * 2 ^ b :=
  Nat.lt_of_lt_of_le (Nat.mul_lt_mul_of_lt_of_le hx (Nat.le_of_lt hy) (Nat.two_pow_pos _))
    (Nat.le_refl _)

#print axioms cut_exact
#print axioms cut_rel_err
#print axioms cut_total_loss
#print axioms dot_bound
#print axioms good_side
#print axioms avg_not_max
#print axioms int32_exact
#print axioms slice_prod_lt

end OzimmuAuto

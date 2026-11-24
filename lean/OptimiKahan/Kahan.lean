/-!
# Kahan-compensated low-precision accumulation (optimi) -- abstract rounding layer

Scope.  Core Lean only (no Mathlib).  All quantities are integers: multiples of the smallest quantum of the format,
so floating-point values with a fixed exponent range live on the integer lattice `Int`.  Rounding is an abstract
function `rd` constrained only by "result is representable" and "no representable value is strictly closer".

What is PROVED here (no `sorry`, no extra axioms; see `#print axioms` at the end):
* `step_value` / `run_value`  : exact conservation law  V' = V + u - δ + η  (η = renormalisation defect), summed over N steps.
* `run_bound`                 : under a relative-error hypothesis, 256·|Σδ| ≤ Σ|c_k + u_k|.
* `naive_stagnates`, `naive_run_stagnates` : plain low-precision accumulation drops every update smaller than half a gap.
* Concrete 8-significant-bit lattice `lattice 8` (bf16 without overflow, subnormals included):
  `relErr8` (relative error ≤ 2^-8), `run_bound_bf16`, `gap_2pow20`, `bf16_stagnation`.
* `fast2sum_bf16` : Fast2Sum is exact on `lattice 8` for ANY nearest rounding (ties arbitrary) when |b| ≤ |a|.
  Hence `run_value_bf16`, `kahan_error_bf16` (valid when every step satisfies |c1| ≤ |w|).
* **Violation case** (optimi does not enforce |c1| ≤ |w|):
  `eta_bound_violation` : for ANY rounding with relative error 2^-8 and |w| ≤ |c1|, 251·|η| ≤ |c1|  (≈ 1.02·2^-8·|c1|; observed max 2^-8·|c1|).
  `eta_bound_bf16`      : for `lattice 8` and representable w, 251·|η| ≤ |c1| for ALL states (η = 0 when |c1| ≤ |w|).
  `kahan_error_bf16_unconditional` : NO per-step precondition:  125·|(w_N + c_N) - (w + c + Σu)| ≤ Σ|c_k + u_k|.

What is NOT proved here (see HANDOFF section 13):
* The link from this model to PyTorch bf16 / float32-intermediate arithmetic is empirical (validated bit-for-bit on the CPU
  single-tensor path); the model uses single rounding and has no exponent upper bound (no overflow).
* Only p = 8 (bf16) is instantiated; fp16 (p = 11) and the Triton/foreach code paths are not modelled.
* The constants are not optimal for the multi-step bound (observed 125·err/S ≤ 0.08 on adversarial trajectories).
-/
namespace OptimiKahan

/-- Abstract round-to-nearest on the integer lattice. -/
structure Rounding where
  R : Int → Prop
  rd : Int → Int
  rd_rep : ∀ z, R (rd z)
  rd_nearest : ∀ z y, R y → (z - rd z).natAbs ≤ (z - y).natAbs

/-- Unit roundoff 2^-8 (bf16: 8 significant bits): relative rounding error bound. -/
def RelErr (rb : Rounding) : Prop := ∀ z, 256 * (z - rb.rd z).natAbs ≤ z.natAbs

/-- Fast2Sum exactness: for representable a b with |b| ≤ |a|, the three-operation sequence returns the exact rounding error. -/
def Fast2SumExact (rb : Rounding) : Prop :=
  ∀ a b, rb.R a → rb.R b → b.natAbs ≤ a.natAbs →
    rb.rd (b + rb.rd (a - rb.rd (a + b))) = (a + b) - rb.rd (a + b)

/-- optimi's compensated update (the part after the Adam direction `u` is formed):
    c ← rd(c+u); w ← rd(w+c); c ← rd(c + rd(t - w)) with t the old w. -/
def step (rb : Rounding) (w c u : Int) : Int × Int :=
  (rb.rd (w + rb.rd (c + u)),
   rb.rd (rb.rd (c + u) + rb.rd (w - rb.rd (w + rb.rd (c + u)))))

/-- Rounding error of the compensation accumulate `c + u`. -/
def delta (rb : Rounding) (c u : Int) : Int := (c + u) - rb.rd (c + u)

/-- Renormalisation defect of one step (zero iff the represented value w + c1 is preserved by the Fast2Sum part). -/
def eta (rb : Rounding) (w c u : Int) : Int :=
  (step rb w c u).1 + (step rb w c u).2 - (w + rb.rd (c + u))

/-- One-step conservation law (pure algebra): V' = V + u - δ + η. -/
theorem step_value (rb : Rounding) (w c u : Int) :
    (step rb w c u).1 + (step rb w c u).2 = (w + c) + u - delta rb c u + eta rb w c u := by
  unfold eta delta; omega

/-- Under Fast2Sum exactness and |c1| ≤ |w| the defect vanishes. -/
theorem eta_zero (rb : Rounding) (hF : Fast2SumExact rb) (w c u : Int)
    (hw : rb.R w) (hle : (rb.rd (c + u)).natAbs ≤ w.natAbs) : eta rb w c u = 0 := by
  have h := hF w (rb.rd (c + u)) hw (rb.rd_rep _) hle
  unfold eta step
  simp only
  omega

/-- N-step run: returns (w, c, Σδ, Σ|c_k+u_k|). -/
def run (rb : Rounding) : Int → Int → List Int → Int × Int × Int × Nat
  | w, c, [] => (w, c, 0, 0)
  | w, c, u :: us =>
    let s := step rb w c u
    let r := run rb s.1 s.2 us
    (r.1, r.2.1, delta rb c u + r.2.2.1, (c + u).natAbs + r.2.2.2)

/-- Every step satisfies the Fast2Sum precondition. -/
def GoodAll (rb : Rounding) : Int → Int → List Int → Prop
  | _, _, [] => True
  | w, c, u :: us => (rb.rd (c + u)).natAbs ≤ w.natAbs ∧ GoodAll rb (step rb w c u).1 (step rb w c u).2 us

theorem repr_step (rb : Rounding) (w c u : Int) : rb.R (step rb w c u).1 ∧ rb.R (step rb w c u).2 :=
  ⟨rb.rd_rep _, rb.rd_rep _⟩

/-- N-step conservation: with exact Fast2Sum and representable start, w_N + c_N = w + c + Σu - Σδ exactly. -/
theorem run_value (rb : Rounding) (hF : Fast2SumExact rb) :
    ∀ (us : List Int) (w c : Int), rb.R w → GoodAll rb w c us →
      (run rb w c us).1 + (run rb w c us).2.1 = (w + c) + us.sum - (run rb w c us).2.2.1 := by
  intro us
  induction us with
  | nil => intro w c _ _; simp [run]
  | cons u us ih =>
    intro w c hw hg
    obtain ⟨hle, hrest⟩ := hg
    have hz := eta_zero rb hF w c u hw hle
    have hv := step_value rb w c u
    have hw1 : rb.R (step rb w c u).1 := (repr_step rb w c u).1
    have := ih (step rb w c u).1 (step rb w c u).2 hw1 hrest
    simp only [run, List.sum_cons]
    rw [hz] at hv
    omega

/-- Error bound: with relative rounding error 2^-8, 256·|Σδ| ≤ Σ|c_k + u_k|. -/
theorem run_bound (rb : Rounding) (hE : RelErr rb) :
    ∀ (us : List Int) (w c : Int), 256 * (run rb w c us).2.2.1.natAbs ≤ (run rb w c us).2.2.2 := by
  intro us
  induction us with
  | nil => intro w c; simp [run]
  | cons u us ih =>
    intro w c
    have h1 := hE (c + u)
    have h2 := ih (step rb w c u).1 (step rb w c u).2
    simp only [run]
    unfold delta
    omega

/-- Plain low-precision accumulation stagnates: if the nearest other representable value is at least q away and
    the update is below q/2, rounding returns w unchanged. -/
theorem naive_stagnates (rb : Rounding) (w u : Int) (q : Nat) (hw : rb.R w)
    (hgap : ∀ y, rb.R y → y ≠ w → q ≤ (y - w).natAbs) (hu : 2 * u.natAbs < q) :
    rb.rd (w + u) = w := by
  have hn := rb.rd_nearest (w + u) w hw
  by_cases heq : rb.rd (w + u) = w
  · exact heq
  · have hg := hgap _ (rb.rd_rep (w + u)) heq
    omega

/-- Naive accumulation w ← rd(w + u_k) over a whole list leaves w unchanged when every u_k is below q/2. -/
theorem naive_run_stagnates (rb : Rounding) (w : Int) (q : Nat) (hw : rb.R w)
    (hgap : ∀ y, rb.R y → y ≠ w → q ≤ (y - w).natAbs) :
    ∀ us : List Int, (∀ u ∈ us, 2 * u.natAbs < q) → us.foldl (fun acc u => rb.rd (acc + u)) w = w := by
  intro us
  induction us with
  | nil => intro _; rfl
  | cons u us ih =>
    intro h
    have h0 := naive_stagnates rb w u q hw hgap (h u (by simp))
    simp only [List.foldl_cons, h0]
    exact ih (fun v hv => h v (by simp [hv]))

/-! ## Concrete nearest rounding on the p-significant-bit lattice (bf16 for p = 8, no overflow, subnormals included) -/

/-- Lattice points with at most p significant bits: |z| = m * 2^e with m < 2^p. -/
def Rep (p : Nat) (z : Int) : Prop := ∃ m e : Nat, z.natAbs = m * 2 ^ e ∧ m < 2 ^ p

theorem rep_zero (p : Nat) : Rep p 0 := ⟨0, 0, by simp, Nat.two_pow_pos p⟩

/-- A nearest representable point always exists (strong induction on the distance). -/
theorem exists_nearest (p : Nat) (z : Int) :
    ∃ y, Rep p y ∧ ∀ y', Rep p y' → (z - y).natAbs ≤ (z - y').natAbs := by
  have key : ∀ n : Nat, ∀ y : Int, Rep p y → (z - y).natAbs = n →
      ∃ y0, Rep p y0 ∧ ∀ y', Rep p y' → (z - y0).natAbs ≤ (z - y').natAbs := by
    intro n
    induction n using Nat.strongRecOn with
    | _ n ih =>
      intro y hy hn
      by_cases h : ∀ y', Rep p y' → (z - y).natAbs ≤ (z - y').natAbs
      · exact ⟨y, hy, h⟩
      · have h2 : ∃ y', Rep p y' ∧ (z - y').natAbs < (z - y).natAbs := by
          refine Classical.byContradiction (fun hc => h ?_)
          intro y' hy'
          by_cases hlt : (z - y').natAbs < (z - y).natAbs
          · exact absurd ⟨y', hy', hlt⟩ hc
          · omega
        obtain ⟨y', hy', hlt⟩ := h2
        exact ih _ (by omega) y' hy' rfl
  exact key _ 0 (rep_zero p) rfl

noncomputable def rdNear (p : Nat) (z : Int) : Int := Classical.choose (exists_nearest p z)

/-- The concrete rounding: nearest point of the p-bit lattice (ties broken arbitrarily by choice). -/
noncomputable def lattice (p : Nat) : Rounding where
  R := Rep p
  rd := rdNear p
  rd_rep := fun z => (Classical.choose_spec (exists_nearest p z)).1
  rd_nearest := fun z y hy => (Classical.choose_spec (exists_nearest p z)).2 y hy

/-- Every natural number has an 8-significant-bit lattice point within relative distance 2^-8 (error ≤ half a gap ≤ a/256). -/
theorem close_nat (a : Nat) : ∃ b : Nat, Rep 8 (b : Int) ∧ 256 * ((a : Int) - b).natAbs ≤ a := by
  by_cases ha : a < 256
  · exact ⟨a, ⟨a, 0, by simp, by omega⟩, by omega⟩
  · have hapos : a ≠ 0 := by omega
    have hk1 : 2 ^ a.log2 ≤ a := Nat.log2_self_le hapos
    have hk2 : a < 2 ^ (a.log2 + 1) := Nat.lt_log2_self
    have hk8 : 8 ≤ a.log2 := by
      by_cases h : a.log2 < 8
      · have : 2 ^ (a.log2 + 1) ≤ 2 ^ 8 := Nat.pow_le_pow_right (by decide) (by omega)
        omega
      · omega
    obtain ⟨s, hs⟩ : ∃ s, a.log2 = s + 7 := ⟨a.log2 - 7, by omega⟩
    rw [hs] at hk1 hk2
    have e1 : 2 ^ (s + 7) = 2 ^ s * 128 := by rw [Nat.pow_add]
    have e2 : 2 ^ (s + 7 + 1) = 2 ^ s * 256 := by rw [Nat.pow_succ, e1]; omega
    have e3 : 2 ^ (s + 1) = 2 ^ s * 2 := by rw [Nat.pow_succ]
    have hqpos : 0 < 2 ^ s := Nat.two_pow_pos s
    obtain ⟨q, hq⟩ : ∃ q, q = 2 ^ s := ⟨_, rfl⟩
    rw [← hq] at e1 e2 e3 hqpos
    rw [e1] at hk1
    rw [e2] at hk2
    have hdm := Nat.div_add_mod a q
    have hr : a % q < q := Nat.mod_lt _ hqpos
    have hm : a / q < 256 := (Nat.div_lt_iff_lt_mul hqpos).2 (by omega)
    obtain ⟨m, hmdef⟩ : ∃ m, m = a / q := ⟨_, rfl⟩
    obtain ⟨r, hrdef⟩ : ∃ r, r = a % q := ⟨_, rfl⟩
    rw [← hmdef, ← hrdef] at hdm
    rw [← hrdef] at hr
    rw [← hmdef] at hm
    obtain ⟨Q, hQ⟩ : ∃ Q, Q = q * m := ⟨_, rfl⟩
    rw [← hQ] at hdm
    have hQq : Q + q = (m + 1) * q := by
      rw [hQ, Nat.add_mul, Nat.one_mul, Nat.mul_comm]
    have hrepQ : Rep 8 (Q : Int) := ⟨m, s, by rw [Int.natAbs_natCast, hQ, hq, Nat.mul_comm], by omega⟩
    have hrepQ' : Rep 8 ((Q + q : Nat) : Int) := by
      by_cases hm1 : m + 1 < 256
      · exact ⟨m + 1, s, by rw [Int.natAbs_natCast, hQq, hq], by omega⟩
      · refine ⟨128, s + 1, ?_, by omega⟩
        have hm256 : m + 1 = 256 := by omega
        rw [Int.natAbs_natCast, hQq, hm256, e3, hq]
        omega
    by_cases hc : 2 * r ≤ q
    · exact ⟨Q, hrepQ, by omega⟩
    · exact ⟨Q + q, hrepQ', by omega⟩

theorem exists_close (z : Int) : ∃ y, Rep 8 y ∧ 256 * (z - y).natAbs ≤ z.natAbs := by
  obtain ⟨a, rfl | rfl⟩ := Int.eq_nat_or_neg z
  · obtain ⟨b, hb, h⟩ := close_nat a
    exact ⟨b, hb, by simpa using h⟩
  · obtain ⟨b, ⟨m, e, hm, hlt⟩, h⟩ := close_nat a
    refine ⟨-(b : Int), ⟨m, e, by simpa using hm, hlt⟩, ?_⟩
    omega

/-- The relative-error hypothesis holds for the concrete 8-bit nearest rounding (bf16 without overflow). -/
theorem relErr8 : RelErr (lattice 8) := by
  intro z
  obtain ⟨y, hy, hle⟩ := exists_close z
  have hn := (lattice 8).rd_nearest z y hy
  show 256 * (z - rdNear 8 z).natAbs ≤ z.natAbs
  have hn' : (z - rdNear 8 z).natAbs ≤ (z - y).natAbs := hn
  omega

/-- Headline corollary for the concrete rounding: the accumulated compensation-rounding error is bounded by 2^-8 Σ|c_k+u_k|. -/
theorem run_bound_bf16 (us : List Int) (w c : Int) :
    256 * (run (lattice 8) w c us).2.2.1.natAbs ≤ (run (lattice 8) w c us).2.2.2 :=
  run_bound (lattice 8) relErr8 us w c

/-- Gap around 2^20 in the 8-bit lattice: every other representable point is at least 2^12 away. -/
theorem gap_2pow20 : ∀ y : Int, Rep 8 y → y ≠ 1048576 → 4096 ≤ (y - 1048576).natAbs := by
  intro y ⟨m, e, hy, hm⟩ hne
  by_cases he : 12 ≤ e
  · obtain ⟨e', rfl⟩ : ∃ e', e = 12 + e' := ⟨e - 12, by omega⟩
    have hpow : 2 ^ (12 + e') = 4096 * 2 ^ e' := by rw [Nat.pow_add]
    obtain ⟨P, hP⟩ : ∃ P, P = m * 2 ^ e' := ⟨_, rfl⟩
    have hyP : y.natAbs = 4096 * P := by
      rw [hy, hpow, hP]; rw [Nat.mul_left_comm]
    omega
  · have hlt : e ≤ 11 := by omega
    have h1 : 2 ^ e ≤ 2 ^ 11 := Nat.pow_le_pow_right (by decide) hlt
    have h2 : m * 2 ^ e ≤ 255 * 2 ^ 11 := Nat.mul_le_mul (by omega) h1
    have h3 : y.natAbs ≤ 255 * 2048 := by rw [hy]; simpa using h2
    omega

/-- Concrete stagnation in bf16 (p = 8): starting from w = 2^20 (the value 1.0 on this lattice), ANY sequence of updates
    each below 2^11 is dropped entirely by plain low-precision accumulation w ← rd(w + u). -/
theorem bf16_stagnation (us : List Int) (h : ∀ u ∈ us, u.natAbs < 2048) :
    us.foldl (fun acc u => (lattice 8).rd (acc + u)) 1048576 = 1048576 :=
  naive_run_stagnates (lattice 8) 1048576 4096 ⟨1, 20, by decide, by decide⟩ gap_2pow20 us
    (fun u hu => by have := h u hu; omega)

/-! ## Fast2Sum exactness on the 8-bit lattice -/

/-- Binade quantum exponent: q(z) = 2^(ex p z) is the spacing of the p-bit lattice at |z|. -/
def ex (p : Nat) (z : Int) : Nat := Nat.log2 z.natAbs + 1 - p

theorem lt_two_pow_ex (p : Nat) (z : Int) : z.natAbs < 2 ^ p * 2 ^ ex p z := by
  have h := Nat.lt_log2_self (n := z.natAbs)
  unfold ex
  by_cases hp : p ≤ z.natAbs.log2 + 1
  · have e : p + (z.natAbs.log2 + 1 - p) = z.natAbs.log2 + 1 := by omega
    rw [← Nat.pow_add, e]; exact h
  · have h0 : z.natAbs.log2 + 1 - p = 0 := by omega
    rw [h0, Nat.pow_zero, Nat.mul_one]
    have : 2 ^ (z.natAbs.log2 + 1) ≤ 2 ^ p := Nat.pow_le_pow_right (by decide) (by omega)
    omega

theorem dvd_cast_iff (K : Nat) (y : Int) : ((K : Nat) : Int) ∣ y ↔ K ∣ y.natAbs := by
  rw [← Int.natAbs_dvd_natAbs, Int.natAbs_natCast]

theorem two_pow_ex_dvd {p : Nat} (hp : 1 ≤ p) {z : Int} (h : Rep p z) : ((2 ^ ex p z : Nat) : Int) ∣ z := by
  obtain ⟨m, e, hz, hm⟩ := h
  rw [dvd_cast_iff]
  by_cases hex : ex p z ≤ e
  · rw [hz]; exact Nat.dvd_trans (Nat.pow_dvd_pow 2 hex) (Nat.dvd_mul_left _ _)
  · exfalso
    have hz0 : z.natAbs ≠ 0 := by
      intro h0; apply hex; unfold ex; rw [h0]; simp; omega
    have hlow := Nat.log2_self_le hz0
    have h1 : e + 1 ≤ z.natAbs.log2 + 1 - p := by unfold ex at hex; omega
    have h2 : m * 2 ^ e < 2 ^ p * 2 ^ e := Nat.mul_lt_mul_of_pos_right hm (Nat.two_pow_pos e)
    have h3 : 2 ^ (p + e) ≤ 2 ^ z.natAbs.log2 := Nat.pow_le_pow_right (by decide) (by omega)
    rw [Nat.pow_add] at h3
    omega

theorem ex_mono {p : Nat} {y z : Int} (h : y.natAbs ≤ z.natAbs) : ex p y ≤ ex p z := by
  unfold ex
  have : y.natAbs.log2 ≤ z.natAbs.log2 := by
    by_cases hy : y.natAbs = 0
    · rw [hy]; simp
    · have hz : z.natAbs ≠ 0 := by omega
      exact (Nat.le_log2 hz).2 (Nat.le_trans (Nat.log2_self_le hy) h)
  omega

theorem ex_ge {p k : Nat} {z : Int} (h : 2 ^ (p + k) ≤ z.natAbs) : k + 1 ≤ ex p z := by
  unfold ex
  have hz : z.natAbs ≠ 0 := by
    have := Nat.two_pow_pos (p + k); omega
  have := (Nat.le_log2 hz).2 h
  omega

/-- Criterion: a multiple of 2^k with |y| ≤ 2^p·2^k is representable (the boundary 2^p·2^k = 2^(p-1)·2^(k+1) is a power of two). -/
theorem rep_of_dvd {p : Nat} (hp : 1 ≤ p) (k : Nat) (y : Int)
    (hd : ((2 ^ k : Nat) : Int) ∣ y) (hb : y.natAbs ≤ 2 ^ p * 2 ^ k) : Rep p y := by
  rw [dvd_cast_iff] at hd
  obtain ⟨c, hc⟩ := hd
  have hcle : c ≤ 2 ^ p := by
    have : 2 ^ k * c ≤ 2 ^ k * 2 ^ p := by rw [← hc, Nat.mul_comm (2 ^ k)]; exact hb
    exact Nat.le_of_mul_le_mul_left this (Nat.two_pow_pos k)
  by_cases hlt : c < 2 ^ p
  · exact ⟨c, k, by rw [hc, Nat.mul_comm], hlt⟩
  · have hceq : c = 2 ^ p := by omega
    obtain ⟨p', rfl⟩ : ∃ p', p = p' + 1 := ⟨p - 1, by omega⟩
    refine ⟨2 ^ p', k + 1, ?_, Nat.pow_lt_pow_right (by decide) (by omega)⟩
    rw [hc, hceq, Nat.pow_succ, Nat.pow_succ]
    ac_rfl

theorem rd_self {p : Nat} {y : Int} (h : Rep p y) : rdNear p y = y := by
  have h1 : (y - rdNear p y).natAbs ≤ (y - y).natAbs := (lattice p).rd_nearest y y h
  have h2 : (y - rdNear p y).natAbs ≤ 0 := by simpa using h1
  omega

theorem rd_abs_ge {p : Nat} {T z : Int} (hT : Rep p T) (h : T.natAbs ≤ z.natAbs) :
    T.natAbs ≤ (rdNear p z).natAbs := by
  by_cases hz : 0 ≤ z
  · have hy : Rep p ((T.natAbs : Nat) : Int) := by
      obtain ⟨m, e, h1, h2⟩ := hT; exact ⟨m, e, by rw [Int.natAbs_natCast]; exact h1, h2⟩
    have := (lattice p).rd_nearest z _ hy
    have h' : (z - rdNear p z).natAbs ≤ (z - (T.natAbs : Int)).natAbs := this
    omega
  · have hy : Rep p (-((T.natAbs : Nat) : Int)) := by
      obtain ⟨m, e, h1, h2⟩ := hT; exact ⟨m, e, by simpa using h1, h2⟩
    have := (lattice p).rd_nearest z _ hy
    have h' : (z - rdNear p z).natAbs ≤ (z - (-(T.natAbs : Int))).natAbs := this
    omega

theorem rd_abs_le {p : Nat} {T z : Int} (hT : Rep p T) (h : z.natAbs ≤ T.natAbs) :
    (rdNear p z).natAbs ≤ T.natAbs := by
  by_cases hz : 0 ≤ z
  · have hy : Rep p ((T.natAbs : Nat) : Int) := by
      obtain ⟨m, e, h1, h2⟩ := hT; exact ⟨m, e, by rw [Int.natAbs_natCast]; exact h1, h2⟩
    have := (lattice p).rd_nearest z _ hy
    have h' : (z - rdNear p z).natAbs ≤ (z - (T.natAbs : Int)).natAbs := this
    omega
  · have hy : Rep p (-((T.natAbs : Nat) : Int)) := by
      obtain ⟨m, e, h1, h2⟩ := hT; exact ⟨m, e, by simpa using h1, h2⟩
    have := (lattice p).rd_nearest z _ hy
    have h' : (z - rdNear p z).natAbs ≤ (z - (-(T.natAbs : Int))).natAbs := this
    omega

theorem rep_of_dvd' {p : Nat} (hp : 1 ≤ p) {K k : Nat} (hK : K = 2 ^ k) (y : Int)
    (hd : (K : Int) ∣ y) (hb : y.natAbs ≤ 2 ^ p * K) : Rep p y := by
  subst hK; exact rep_of_dvd hp k y hd hb

/-- A multiple of K that is below c·K is at most c·K - K. -/
theorem dvd_lt_le {K c : Nat} {y : Int} (hd : (K : Int) ∣ y) (h : y.natAbs < K * c) :
    y.natAbs + K ≤ K * c := by
  rw [dvd_cast_iff] at hd
  obtain ⟨t, ht⟩ := hd
  rw [ht] at h ⊢
  have ht' : t < c := Nat.lt_of_mul_lt_mul_left h
  have h2 : K * (t + 1) ≤ K * c := Nat.mul_le_mul_left K ht'
  rw [Nat.mul_succ] at h2
  exact h2

/-- Every multiple of K is within K of a multiple of 2K. -/
theorem exists_even_near (K : Nat) (z : Int) (hd : (K : Int) ∣ z) :
    ∃ y : Int, ((2 * K : Nat) : Int) ∣ y ∧ (z - y).natAbs ≤ K := by
  obtain ⟨t, rfl⟩ := hd
  have h2K : ((2 * K : Nat) : Int) = 2 * (K : Int) := by omega
  rcases (by omega : t % 2 = 0 ∨ t % 2 = 1) with h | h
  · obtain ⟨u, hu⟩ : ∃ u, t = 2 * u := ⟨t / 2, by omega⟩
    refine ⟨(K : Int) * t, ⟨u, ?_⟩, by simp⟩
    rw [hu, h2K]; simp [Int.mul_comm, Int.mul_left_comm]
  · obtain ⟨u, hu⟩ : ∃ u, t = 2 * u + 1 := ⟨t / 2, by omega⟩
    refine ⟨(K : Int) * (t + 1), ⟨u + 1, ?_⟩, ?_⟩
    · rw [hu, h2K]
      have : 2 * u + 1 + 1 = 2 * (u + 1) := by omega
      rw [this]; simp [Int.mul_comm, Int.mul_left_comm]
    · have : (K : Int) * t - (K : Int) * (t + 1) = -(K : Int) := by
        rw [Int.mul_add, Int.mul_one]; omega
      rw [this]; simp

theorem rep_neg {p : Nat} {y : Int} (h : Rep p y) : Rep p (-y) := by
  obtain ⟨m, e, h1, h2⟩ := h; exact ⟨m, e, by simpa using h1, h2⟩

/-- **Fast2Sum is exact on the 8-bit (bf16-like) lattice**: for representable a, b with |b| ≤ |a|, the sequence
    s = rd(a+b); d = rd(a - s); e = rd(b + d) returns the exact rounding error e = (a+b) - s. -/
theorem fast2sum_bf16 : Fast2SumExact (lattice 8) := by
  intro a b ha hb hle
  have ha' : Rep 8 a := ha
  have hb' : Rep 8 b := hb
  show rdNear 8 (b + rdNear 8 (a - rdNear 8 (a + b))) = (a + b) - rdNear 8 (a + b)
  -- K = quantum of b's binade; everything below is measured in units of K
  obtain ⟨K, hKdef⟩ : ∃ K, K = 2 ^ ex 8 b := ⟨_, rfl⟩
  have hKpos : 0 < K := by rw [hKdef]; exact Nat.two_pow_pos _
  have hK2 : 2 * K = 2 ^ (ex 8 b + 1) := by rw [hKdef, Nat.pow_succ]; omega
  have hKb : (K : Int) ∣ b := by rw [hKdef]; exact two_pow_ex_dvd (by decide) hb'
  have hexle : ex 8 b ≤ ex 8 a := ex_mono hle
  have hKa : (K : Int) ∣ a := by
    have h2 := two_pow_ex_dvd (by decide) ha'
    have h3 : ((2 ^ ex 8 b : Nat) : Int) ∣ ((2 ^ ex 8 a : Nat) : Int) :=
      Int.natCast_dvd_natCast.2 (Nat.pow_dvd_pow 2 hexle)
    rw [hKdef]; exact Int.dvd_trans h3 h2
  have hbK : b.natAbs < 256 * K := by
    have := lt_two_pow_ex 8 b; rw [← hKdef] at this; omega
  have hdz : (K : Int) ∣ a + b := Int.dvd_add hKa hKb
  by_cases hc1 : (a + b).natAbs ≤ 256 * K
  · -- case 1: a + b is representable, everything is exact
    have hrz : Rep 8 (a + b) := rep_of_dvd' (by decide) hKdef _ hdz (by omega)
    rw [rd_self hrz]
    have e1 : a - (a + b) = -b := by omega
    rw [e1, rd_self (rep_neg hb')]
    have e2 : b + -b = 0 := by omega
    rw [e2, rd_self (rep_zero 8)]
    omega
  · -- case 2: |a+b| > 2^8 K
    obtain ⟨s, hs⟩ : ∃ s, rdNear 8 (a + b) = s := ⟨_, rfl⟩
    rw [hs]
    have hsrep : Rep 8 s := by rw [← hs]; exact (lattice 8).rd_rep _
    have hnear : ∀ y, Rep 8 y → (a + b - s).natAbs ≤ (a + b - y).natAbs := by
      intro y hy; rw [← hs]; exact (lattice 8).rd_nearest (a + b) y hy
    -- T = 256 K is representable and |a+b| ≥ T, hence |s| ≥ T
    have hT : Rep 8 ((256 * K : Nat) : Int) :=
      rep_of_dvd' (by decide) hKdef _ ⟨256, by omega⟩ (by omega)
    have hsge : 256 * K ≤ s.natAbs := by
      have := rd_abs_ge hT (by simp; omega : ((256 * K : Nat) : Int).natAbs ≤ (a + b).natAbs)
      rw [hs] at this; omega
    -- 2K ∣ s
    have hexs : ex 8 b + 1 ≤ ex 8 s := by
      apply ex_ge (p := 8) (k := ex 8 b)
      rw [Nat.pow_add, ← hKdef]; omega
    have h2Ks : ((2 * K : Nat) : Int) ∣ s := by
      have h1 := two_pow_ex_dvd (by decide) hsrep
      have h3 : ((2 ^ (ex 8 b + 1) : Nat) : Int) ∣ ((2 ^ ex 8 s : Nat) : Int) :=
        Int.natCast_dvd_natCast.2 (Nat.pow_dvd_pow 2 hexs)
      rw [hK2]; exact Int.dvd_trans h3 h1
    have hKs : (K : Int) ∣ s := Int.dvd_trans ⟨2, by omega⟩ h2Ks
    -- error bound |e| ≤ |b| (candidate a) and e representable
    have heb : (a + b - s).natAbs ≤ b.natAbs := by
      have := hnear a ha'
      have e3 : a + b - a = b := by omega
      rw [e3] at this; exact this
    have hKe : (K : Int) ∣ a + b - s := Int.dvd_sub hdz hKs
    have hrepe : Rep 8 (a + b - s) := rep_of_dvd' (by decide) hKdef _ hKe (by omega)
    -- a - s representable
    have hrepd : Rep 8 (a - s) := by
      by_cases h2Ka : ((2 * K : Nat) : Int) ∣ a
      · exact rep_of_dvd' (k := ex 8 b + 1) (by decide) hK2 _ (Int.dvd_sub h2Ka h2Ks) (by omega)
      · have hexa : ex 8 a = ex 8 b := by
          by_cases hlt : ex 8 b < ex 8 a
          · exfalso; apply h2Ka
            have h2 := two_pow_ex_dvd (by decide) ha'
            have h3 : ((2 ^ (ex 8 b + 1) : Nat) : Int) ∣ ((2 ^ ex 8 a : Nat) : Int) :=
              Int.natCast_dvd_natCast.2 (Nat.pow_dvd_pow 2 (by omega))
            rw [hK2]; exact Int.dvd_trans h3 h2
          · omega
        have haK : a.natAbs < 256 * K := by
          have := lt_two_pow_ex 8 a; rw [hexa, ← hKdef] at this; omega
        have hzlt : (a + b).natAbs < K * 512 := by omega
        have hz512 := dvd_lt_le hdz hzlt
        obtain ⟨y, hy2, hyd⟩ := exists_even_near K (a + b) hdz
        have hyrep : Rep 8 y :=
          rep_of_dvd' (k := ex 8 b + 1) (by decide) hK2 y hy2 (by omega)
        have heK : (a + b - s).natAbs ≤ K := by
          have := hnear y hyrep; omega
        have hb2 := dvd_lt_le hKb (by omega : b.natAbs < K * 256)
        exact rep_of_dvd' (by decide) hKdef _ (Int.dvd_sub hKa hKs) (by omega)
    rw [rd_self hrepd]
    have e4 : b + (a - s) = a + b - s := by omega
    rw [e4, rd_self hrepe]

/-- N-step conservation on the concrete bf16 lattice, with NO remaining hypothesis except the per-step precondition |c1| ≤ |w|. -/
theorem run_value_bf16 (us : List Int) (w c : Int) (hw : Rep 8 w) (hg : GoodAll (lattice 8) w c us) :
    (run (lattice 8) w c us).1 + (run (lattice 8) w c us).2.1
      = (w + c) + us.sum - (run (lattice 8) w c us).2.2.1 :=
  run_value (lattice 8) fast2sum_bf16 us w c hw hg

/-- Headline: the represented value w_N + c_N differs from the exact w + c + Σu by at most 2^-8 · Σ|c_k + u_k|. -/
theorem kahan_error_bf16 (us : List Int) (w c : Int) (hw : Rep 8 w) (hg : GoodAll (lattice 8) w c us) :
    256 * ((run (lattice 8) w c us).1 + (run (lattice 8) w c us).2.1 - (w + c + us.sum)).natAbs
      ≤ (run (lattice 8) w c us).2.2.2 := by
  have h1 := run_value_bf16 us w c hw hg
  have h2 := run_bound_bf16 us w c
  omega

/-! ## The violation case |c1| > |w|: a bound on the renormalisation defect (needs only RelErr) -/

/-- If |w| ≤ |c1| (Fast2Sum precondition violated or tight), the defect is at most |c1|/251 ≈ 1.02·2^-8·|c1|. -/
theorem eta_bound_violation (rb : Rounding) (hE : RelErr rb) (w c u : Int)
    (hv : w.natAbs ≤ (rb.rd (c + u)).natAbs) :
    251 * (eta rb w c u).natAbs ≤ (rb.rd (c + u)).natAbs := by
  have h1 := hE (w + rb.rd (c + u))
  have h2 := hE (w - rb.rd (w + rb.rd (c + u)))
  have h3 := hE (rb.rd (c + u) + rb.rd (w - rb.rd (w + rb.rd (c + u))))
  unfold eta step
  simp only
  omega

/-- Unconditional on the precondition: for the concrete bf16 lattice and representable w, the defect is always ≤ |c1|/251
    (zero when |c1| ≤ |w| by Fast2Sum exactness; ≤ |c1|/251 otherwise). -/
theorem eta_bound_bf16 (w c u : Int) (hw : Rep 8 w) :
    251 * (eta (lattice 8) w c u).natAbs ≤ ((lattice 8).rd (c + u)).natAbs := by
  by_cases hle : ((lattice 8).rd (c + u)).natAbs ≤ w.natAbs
  · have h0 := eta_zero (lattice 8) fast2sum_bf16 w c u hw hle
    rw [h0]; simp
  · exact eta_bound_violation (lattice 8) relErr8 w c u (by omega)

/-- **Unconditional error bound for the concrete bf16 Kahan scheme** (no |c1| ≤ |w| hypothesis):
    125·|(w_N + c_N) - (w + c + Σu)| ≤ Σ|c_k + u_k|, i.e. relative error ≲ 2^-7 per accumulated |c+u|. -/
theorem kahan_error_bf16_unconditional (us : List Int) :
    ∀ w c : Int, Rep 8 w →
      125 * ((run (lattice 8) w c us).1 + (run (lattice 8) w c us).2.1 - (w + c + us.sum)).natAbs
        ≤ (run (lattice 8) w c us).2.2.2 := by
  induction us with
  | nil => intro w c _; simp [run]
  | cons u us ih =>
    intro w c hw
    have hv := step_value (lattice 8) w c u
    have hη := eta_bound_bf16 w c u hw
    have hδ := relErr8 (c + u)
    have ih' := ih (step (lattice 8) w c u).1 (step (lattice 8) w c u).2 (repr_step (lattice 8) w c u).1
    simp only [run, List.sum_cons]
    unfold delta at hv
    have hδ' : 256 * (c + u - (lattice 8).rd (c + u)).natAbs ≤ (c + u).natAbs := hδ
    omega

#print axioms step_value
#print axioms eta_zero
#print axioms run_value
#print axioms run_bound
#print axioms naive_stagnates
#print axioms naive_run_stagnates
#print axioms exists_nearest
#print axioms relErr8
#print axioms run_bound_bf16
#print axioms gap_2pow20
#print axioms bf16_stagnation
#print axioms fast2sum_bf16
#print axioms run_value_bf16
#print axioms kahan_error_bf16
#print axioms eta_bound_violation
#print axioms eta_bound_bf16
#print axioms kahan_error_bf16_unconditional

end OptimiKahan

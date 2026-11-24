/-!
# GEMMul8 (Ozaki scheme II, INT8 backend): the scaled product never leaves the CRT range

GEMMul8 (RIKEN-RCCS/GEMMul8 d8d3901) scales row `i` of A by `2^{sa_i}` and column `j` of B by `2^{sb_j}`,
truncates toward zero to integers `x`, `y`, computes `x·y` modulo each modulus `m_t` with INT8 GEMMs, and
reconstructs the integer `c = Σ_l x_l y_l` from the residues by the CRT, using the symmetric residue modulo
`P = Π m_t`.  This is exact iff `|c| ≤ (P-1)/2` (and the moduli are pairwise coprime).

What the shifts guarantee (the floating-point parts are checked separately, exhaustively on the GPU, see
`experiments/gemmul8/sft_exhaustive.cu`; `S = 2^{2·log2P} ≤ (P-1)/2` is checked for the table):
* fast mode: `Σ x_l^2 ≤ S` and `Σ y_l^2 ≤ S`  (row/column 2-norms after scaling and truncation);
* accurate mode: `|x_l| ≤ 2^{ca} hA_l`, `|y_l| ≤ 2^{cb} hB_l` with `H = Σ hA_l hB_l` (an INT8 GEMM of rounded-up
  magnitudes), and `4^{ca}·R ≤ S`, `4^{cb}·Q ≤ S` where `R`, `Q` are the row / column maxima of `H` (so `H ≤ R`,
  `H ≤ Q`).

Proved here (core Lean, Int/Nat only):
* `fast_range`      : `Σx² ≤ S ∧ Σy² ≤ S ⇒ |Σxy| ≤ S`   (via `2|ab| ≤ a²+b²`, no square roots)
* `accu_range`      : the accurate-mode conditions ⇒ `|Σxy| ≤ S`
* `crt_unique`      : `P ∣ c - c'`, `2|c| ≤ P-1`, `2|c'| ≤ P-1` ⇒ `c = c'`
* `dvd_prod_of_pairwise` : pairwise coprime moduli, each dividing `d` ⇒ their product divides `d`
* `crt_exact`       : combining the two: residues agree modulo every `m_t` and both values are in the
                      symmetric range ⇒ the reconstructed integer equals the true one.
-/

namespace Gemmul8Range

/-! ## Sums over a list of pairs `(x_l, y_l)` -/

def dot : List (Int × Int) → Int
  | [] => 0
  | p :: t => p.1 * p.2 + dot t

def sqx : List (Int × Int) → Nat
  | [] => 0
  | p :: t => p.1.natAbs * p.1.natAbs + sqx t

def sqy : List (Int × Int) → Nat
  | [] => 0
  | p :: t => p.2.natAbs * p.2.natAbs + sqy t

def absdot : List (Int × Int) → Nat
  | [] => 0
  | p :: t => p.1.natAbs * p.2.natAbs + absdot t

theorem natAbs_dot_le (l : List (Int × Int)) : (dot l).natAbs ≤ absdot l := by
  induction l with
  | nil => simp [dot, absdot]
  | cons p t ih =>
    simp only [dot, absdot]
    have h := Int.natAbs_add_le (p.1 * p.2) (dot t)
    rw [Int.natAbs_mul] at h
    omega

theorem two_mul_le_aux (a d : Nat) : 2 * (a * (a + d)) ≤ a * a + (a + d) * (a + d) := by
  have e : (a + d) * (a + d) = a * a + 2 * (a * d) + d * d := by
    simp only [Nat.add_mul, Nat.mul_add]; rw [Nat.mul_comm d a]; omega
  rw [e, Nat.mul_add]
  have := Nat.zero_le (d * d)
  omega

/-- `2ab ≤ a² + b²` on naturals -/
theorem two_mul_le_sq_add_sq (a b : Nat) : 2 * (a * b) ≤ a * a + b * b := by
  rcases Nat.le_total a b with h | h
  · obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
    exact two_mul_le_aux a d
  · obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
    have := two_mul_le_aux b d
    rw [Nat.mul_comm (b + d) b]
    omega

theorem two_absdot_le (l : List (Int × Int)) : 2 * absdot l ≤ sqx l + sqy l := by
  induction l with
  | nil => simp [absdot, sqx, sqy]
  | cons p t ih =>
    simp only [absdot, sqx, sqy]
    have := two_mul_le_sq_add_sq p.1.natAbs p.2.natAbs
    omega

/-- fast mode: both scaled 2-norms squared `≤ S` ⇒ the integer product is in `[-S, S]` -/
theorem fast_range (l : List (Int × Int)) (S : Nat) (hx : sqx l ≤ S) (hy : sqy l ≤ S) :
    (dot l).natAbs ≤ S := by
  have h1 := natAbs_dot_le l
  have h2 := two_absdot_le l
  omega

/-! ## Accurate mode -/

/-- `|x_l| ≤ 2^ca · hA_l` and `|y_l| ≤ 2^cb · hB_l` termwise -/
def Dominated (ca cb : Nat) : List ((Int × Int) × (Nat × Nat)) → Prop
  | [] => True
  | ((x, y), (ha, hb)) :: t => x.natAbs ≤ 2 ^ ca * ha ∧ y.natAbs ≤ 2 ^ cb * hb ∧ Dominated ca cb t

def dotq : List ((Int × Int) × (Nat × Nat)) → Int
  | [] => 0
  | ((x, y), _) :: t => x * y + dotq t

def hsum : List ((Int × Int) × (Nat × Nat)) → Nat
  | [] => 0
  | (_, (ha, hb)) :: t => ha * hb + hsum t

theorem dotq_le (ca cb : Nat) : ∀ l, Dominated ca cb l → (dotq l).natAbs ≤ 2 ^ ca * 2 ^ cb * hsum l
  | [], _ => by simp [dotq, hsum]
  | ((x, y), (ha, hb)) :: t, h => by
    obtain ⟨hx, hy, ht⟩ := h
    simp only [dotq, hsum]
    have ih := dotq_le ca cb t ht
    have h1 := Int.natAbs_add_le (x * y) (dotq t)
    rw [Int.natAbs_mul] at h1
    have h2 : x.natAbs * y.natAbs ≤ 2 ^ ca * ha * (2 ^ cb * hb) := Nat.mul_le_mul hx hy
    have h3 : 2 ^ ca * ha * (2 ^ cb * hb) = 2 ^ ca * 2 ^ cb * (ha * hb) := by
      rw [Nat.mul_assoc, Nat.mul_left_comm ha, ← Nat.mul_assoc]
    rw [Nat.mul_add]
    omega

/-- accurate mode: `H ≤ R`, `H ≤ Q`, `4^ca R ≤ S`, `4^cb Q ≤ S` ⇒ `|Σxy| ≤ S` -/
theorem accu_range (ca cb R Q S : Nat) (l : List ((Int × Int) × (Nat × Nat))) (hd : Dominated ca cb l)
    (hR : hsum l ≤ R) (hQ : hsum l ≤ Q) (hRS : 4 ^ ca * R ≤ S) (hQS : 4 ^ cb * Q ≤ S) :
    (dotq l).natAbs ≤ S := by
  have h1 := dotq_le ca cb l hd
  -- t = 2^ca 2^cb H;  t*t = (4^ca H)(4^cb H) ≤ (4^ca R)(4^cb Q) ≤ S*S
  have e4 : ∀ c : Nat, 4 ^ c = 2 ^ c * 2 ^ c := fun c => by
    rw [← Nat.mul_pow]
  have ht : (2 ^ ca * 2 ^ cb * hsum l) * (2 ^ ca * 2 ^ cb * hsum l) ≤ S * S := by
    have a1 : 4 ^ ca * hsum l ≤ S := Nat.le_trans (Nat.mul_le_mul_left _ hR) hRS
    have a2 : 4 ^ cb * hsum l ≤ S := Nat.le_trans (Nat.mul_le_mul_left _ hQ) hQS
    have := Nat.mul_le_mul a1 a2
    calc (2 ^ ca * 2 ^ cb * hsum l) * (2 ^ ca * 2 ^ cb * hsum l)
        = (4 ^ ca * hsum l) * (4 ^ cb * hsum l) := by
          rw [e4, e4]
          simp only [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
      _ ≤ S * S := this
  have h2 : 2 ^ ca * 2 ^ cb * hsum l ≤ S := Nat.mul_self_le_mul_self_iff.mp ht
  omega

/-! ## CRT reconstruction with the symmetric residue -/

theorem crt_unique (P : Nat) (c c' : Int) (hd : (P : Int) ∣ c - c')
    (hc : 2 * c.natAbs ≤ P - 1) (hc' : 2 * c'.natAbs ≤ P - 1) : c = c' := by
  rcases Nat.eq_zero_or_pos P with h0 | hpos
  · subst h0
    have e1 := Int.natAbs_eq_zero.mp (by omega : c.natAbs = 0)
    have e2 := Int.natAbs_eq_zero.mp (by omega : c'.natAbs = 0)
    omega
  have h1 : P ∣ (c - c').natAbs := by
    have := Int.natAbs_dvd_natAbs.mpr hd
    simpa using this
  have h2 : (c - c').natAbs < P := by
    have := Int.natAbs_sub_le c c'
    omega
  have h3 := Int.natAbs_eq_zero.mp (Nat.eq_zero_of_dvd_of_lt h1 h2)
  omega

def prod : List Nat → Nat
  | [] => 1
  | m :: t => m * prod t

def Pairwise : List Nat → Prop
  | [] => True
  | m :: t => (∀ m' ∈ t, Nat.Coprime m m') ∧ Pairwise t

theorem coprime_prod (m : Nat) : ∀ t : List Nat, (∀ m' ∈ t, Nat.Coprime m m') → Nat.Coprime m (prod t)
  | [], _ => by simp [prod, Nat.Coprime]
  | a :: t, h => by
    simp only [prod]
    exact Nat.Coprime.mul_right (h a (List.mem_cons_self ..))
      (coprime_prod m t (fun x hx => h x (List.mem_cons_of_mem _ hx)))

theorem dvd_prod_of_pairwise (d : Nat) : ∀ ms : List Nat, Pairwise ms → (∀ m ∈ ms, m ∣ d) → prod ms ∣ d
  | [], _, _ => by simp [prod]
  | m :: t, ⟨hm, ht⟩, hd => by
    simp only [prod]
    exact Nat.Coprime.mul_dvd_of_dvd_of_dvd (coprime_prod m t hm) (hd m (List.mem_cons_self ..))
      (dvd_prod_of_pairwise d t ht (fun x hx => hd x (List.mem_cons_of_mem _ hx)))

/-- residues agree modulo every modulus and both integers lie in the symmetric range ⇒ they are equal -/
theorem crt_exact (ms : List Nat) (hp : Pairwise ms) (c c' : Int)
    (hres : ∀ m ∈ ms, (m : Int) ∣ c - c')
    (hc : 2 * c.natAbs ≤ prod ms - 1) (hc' : 2 * c'.natAbs ≤ prod ms - 1) : c = c' := by
  apply crt_unique (prod ms) c c' _ hc hc'
  have hn : prod ms ∣ (c - c').natAbs :=
    dvd_prod_of_pairwise _ ms hp (fun m hm => by
      have := Int.natAbs_dvd_natAbs.mpr (hres m hm)
      simpa using this)
  exact Int.natAbs_dvd_natAbs.mp (by simpa using hn)

/-- end-to-end for one entry (fast mode): with `S ≤ (P-1)/2`, the CRT value equals the true integer product -/
theorem fast_crt_exact (ms : List Nat) (hp : Pairwise ms) (l : List (Int × Int)) (S : Nat)
    (hS : 2 * S ≤ prod ms - 1) (hx : sqx l ≤ S) (hy : sqy l ≤ S) (c' : Int)
    (hres : ∀ m ∈ ms, (m : Int) ∣ dot l - c') (hc' : 2 * c'.natAbs ≤ prod ms - 1) : dot l = c' := by
  have := fast_range l S hx hy
  exact crt_exact ms hp (dot l) c' hres (by omega) hc'

#print axioms fast_range
#print axioms accu_range
#print axioms crt_unique
#print axioms dvd_prod_of_pairwise
#print axioms crt_exact
#print axioms fast_crt_exact

end Gemmul8Range

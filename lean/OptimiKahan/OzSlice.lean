import Kahan
open OptimiKahan

/-!
# OzBLAS D_H slicing lemma (integer-lattice model of binary64 / FP16 / FP32)

Models one extraction step of `cuozblasSplitCKernel` (pinned OzBLAS a6f6198):
  piece = (r + sigma) - sigma,   residual = r - piece,   sigma = 0.75 * 2^(rho+tau)   (`CONST * scalbn(1., rho+tau) * splitShift`),
  slice stored as FP16(piece * 2^-tau),  slice products accumulated in FP32.
binary64 = the 53-significant-bit lattice (`lattice 53`: nearest rounding, ANY tie-breaking, no exponent upper bound), all quantities
in units of the smallest quantum; FP16 = `Rep 11`; FP32 = `rdNear 24`.  P = rho + tau, q = 2^(P-53) = ulp(sigma).

PROVED (no `sorry`; axioms printed at the end; Kahan.lean is imported):
* `slice_lemma`   : for 54 ≤ P, 3 ≤ rho ≤ 53, r representable and |r| < 2^tau + q/2 (so also |r| ≤ 2^tau, and tau too SMALL by one
                    near a power of two is covered): the computed piece equals q·n, 2|r - q·n| ≤ q, the computed residual r - q·n is exact,
                    |n| ≤ 2^(53-rho).
* `slice_fp16`    : rho ≥ 42 (s = 53 - rho ≤ 11) ⇒ the slice integer n is an FP16 number.
* `multi_slice`   : T slices with P_k = P - k(54 - rho): exact residuals, each |n_k| ≤ 2^s, final |res| ≤ 2^(P - T(54-rho) - rho).
* `product_exact`, `fl32_foldl_exact`, `tree_exact`, `ozblas_partial_dot_exact` : with slice integers ≤ 2^s (s ≤ 11) and K·4^s ≤ 2^24,
                    every FP16×FP16 product is exact in FP32 and FP32 accumulation of up to K products is exact in ANY order/grouping.
* `IsMaxS`, `isMaxS_unique`, `s_le_11`, `isMaxS_examples`, `k1_needs_special_case` : the library's slice width s = 53 - rho is the largest
                    s with K·4^s ≤ 2^24; K ≥ 2 forces s ≤ 11; K = 1 would allow s = 12 and break FP16 (hence `dim_ = (dim == 1) ? 2 : dim`).

NOT proved here (see HANDOFF section 14): the FP64 recombination order / scale recovery / alpha-beta, the end-to-end GEMM error bound,
overflow (sigma = inf when rho + tau ≥ 1024) and the underflow regime P < 54, FP16/FP32 exponent ranges, the `log2` computation of tau
(only the hypothesis on |r| enters), the termination test / zero rows, and that Tensor Core accumulation behaves like `rdNear 24`.
-/
namespace OzSlice

theorem rep_two_pow {p : Nat} (hp : 1 ≤ p) (k : Nat) : Rep p ((2 ^ k : Nat) : Int) :=
  ⟨1, k, by simp, by have := Nat.pow_le_pow_right (show 2 > 0 by decide) hp; omega⟩

theorem slice_lemma (P rho : Nat) (hP : 54 ≤ P) (hr2 : 3 ≤ rho) (hr53 : rho ≤ 53)
    (r : Int) (hr : Rep 53 r) (hb : 2 * r.natAbs < 2 * 2 ^ (P - rho) + 2 ^ (P - 53)) :
    ∃ n : Int,
      rdNear 53 (rdNear 53 (r + ((3 * 2 ^ (P - 2) : Nat) : Int)) - ((3 * 2 ^ (P - 2) : Nat) : Int))
          = ((2 ^ (P - 53) : Nat) : Int) * n ∧
      2 * (r - ((2 ^ (P - 53) : Nat) : Int) * n).natAbs ≤ 2 ^ (P - 53) ∧
      rdNear 53 (r - ((2 ^ (P - 53) : Nat) : Int) * n) = r - ((2 ^ (P - 53) : Nat) : Int) * n ∧
      n.natAbs ≤ 2 ^ (53 - rho) := by
  obtain ⟨q, hq⟩ : ∃ q, q = 2 ^ (P - 53) := ⟨_, rfl⟩
  obtain ⟨H, hH⟩ : ∃ H, H = 2 ^ (P - 2) := ⟨_, rfl⟩
  obtain ⟨S, hS⟩ : ∃ S, S = 2 ^ (53 - rho) := ⟨_, rfl⟩
  rw [← hq, ← hH, ← hS]
  have hqpos : 0 < q := by rw [hq]; exact Nat.two_pow_pos _
  obtain ⟨h, hh⟩ : ∃ h, q = 2 * h :=
    ⟨2 ^ (P - 54), by rw [hq, show P - 53 = (P - 54) + 1 by omega, Nat.pow_succ]; omega⟩
  have hHq : H = q * 2 ^ 51 := by rw [hH, hq, ← Nat.pow_add]; congr 1; omega
  have hSq : 2 ^ (P - rho) = q * S := by rw [hq, hS, ← Nat.pow_add]; congr 1; omega
  have hHle : 2 * 2 ^ (P - rho) ≤ H := by
    have h1 : 2 ^ (P - rho) ≤ 2 ^ (P - 3) := Nat.pow_le_pow_right (by decide) (by omega)
    have h2 : H = 2 * 2 ^ (P - 3) := by
      rw [hH, show P - 2 = (P - 3) + 1 by omega, Nat.pow_succ]; omega
    omega
  obtain ⟨QS, hQS⟩ : ∃ QS, QS = q * S := ⟨_, rfl⟩
  rw [← hQS] at hSq
  have hqH : q ≤ H := by rw [hHq]; exact Nat.le_mul_of_pos_right q (Nat.two_pow_pos 51)
  rw [← hq] at hb
  have hb' : r.natAbs ≤ H := by omega
  have h2H : 2 * H = 2 ^ (P - 1) := by
    rw [hH, show P - 1 = (P - 2) + 1 by omega, Nat.pow_succ]; omega
  have hT1 : Rep 53 ((2 * H : Nat) : Int) := by rw [h2H]; exact rep_two_pow (by decide) _
  have hσrep : Rep 53 ((3 * H : Nat) : Int) :=
    ⟨3, P - 2, by rw [Int.natAbs_natCast, hH], by decide⟩
  obtain ⟨w, hw⟩ : ∃ w, rdNear 53 (r + ((3 * H : Nat) : Int)) = w := ⟨_, rfl⟩
  have hwrep : Rep 53 w := by rw [← hw]; exact (lattice 53).rd_rep _
  have hnear : ∀ y, Rep 53 y →
      (r + ((3 * H : Nat) : Int) - w).natAbs ≤ (r + ((3 * H : Nat) : Int) - y).natAbs := by
    intro y hy; rw [← hw]; exact (lattice 53).rd_nearest _ y hy
  -- x = r + σ lies in [2H, 4H]
  have hn1 := hnear _ hT1
  have hwge : 2 * H ≤ w := by
    have h2 := rd_abs_ge hT1
      (by omega : ((2 * H : Nat) : Int).natAbs ≤ (r + ((3 * H : Nat) : Int)).natAbs)
    rw [hw] at h2
    omega
  -- q divides w
  have hqw : (q : Int) ∣ w := by
    have hge : 2 ^ (53 + (P - 54)) ≤ w.natAbs := by
      have : 2 ^ (53 + (P - 54)) = 2 * H := by rw [h2H, show 53 + (P - 54) = P - 1 by omega]
      omega
    have h1 := ex_ge (p := 53) (k := P - 54) hge
    have h2 := two_pow_ex_dvd (by decide : 1 ≤ 53) hwrep
    have h3 : ((2 ^ (P - 53) : Nat) : Int) ∣ ((2 ^ ex 53 w : Nat) : Int) :=
      Int.natCast_dvd_natCast.2 (Nat.pow_dvd_pow 2 (by omega))
    rw [hq]; exact Int.dvd_trans h3 h2
  -- a multiple of q within h of x, which is representable: hence |x - w| ≤ h
  obtain ⟨t, ht⟩ : ∃ t : Int, t = r + ((3 * H : Nat) : Int) + (h : Int) := ⟨_, rfl⟩
  have hqne : (q : Int) ≠ 0 := by omega
  have hdm := Int.mul_ediv_add_emod t (q : Int)
  have hr0a := Int.emod_nonneg t hqne
  have hr0b := Int.emod_lt_of_pos t (by omega : (0 : Int) < q)
  obtain ⟨j, hj⟩ : ∃ j, j = t / (q : Int) := ⟨_, rfl⟩
  obtain ⟨r0, hr0⟩ : ∃ r0, r0 = t % (q : Int) := ⟨_, rfl⟩
  rw [← hj, ← hr0] at hdm
  rw [← hr0] at hr0a hr0b
  obtain ⟨Y, hY⟩ : ∃ Y, Y = (q : Int) * j := ⟨_, rfl⟩
  rw [← hY] at hdm
  have hj0 : 0 ≤ j := by rw [hj]; exact Int.ediv_nonneg (by omega) (by omega)
  have hjle : j ≤ 2 ^ 53 := by
    by_cases hc : j ≤ 2 ^ 53
    · exact hc
    · exfalso
      have h1 : (q : Int) * (2 ^ 53 + 1) ≤ Y := by
        rw [hY]; exact Int.mul_le_mul_of_nonneg_left (by omega) (by omega)
      omega
  have hYle : Y ≤ (q : Int) * 2 ^ 53 := by
    rw [hY]; exact Int.mul_le_mul_of_nonneg_left hjle (by omega)
  have hY0 : 0 ≤ Y := by rw [hY]; exact Int.mul_nonneg (by omega) hj0
  have hyrep : Rep 53 Y :=
    rep_of_dvd' (by decide) hq _ ⟨j, hY⟩ (by omega)
  have hxY : (r + ((3 * H : Nat) : Int) - Y).natAbs ≤ h := by omega
  have hxw : (r + ((3 * H : Nat) : Int) - w).natAbs ≤ h := by
    have := hnear Y hyrep; omega
  -- candidate σ gives |x - w| ≤ |r|
  have hxe : (r + ((3 * H : Nat) : Int) - w).natAbs ≤ r.natAbs := by
    have := hnear _ hσrep; omega
  -- the piece
  have hq3 : (q : Int) ∣ ((3 * H : Nat) : Int) := ⟨3 * 2 ^ 51, by omega⟩
  obtain ⟨n, hn⟩ := Int.dvd_sub hqw hq3
  obtain ⟨Z, hZ⟩ : ∃ Z, Z = (q : Int) * n := ⟨_, rfl⟩
  rw [← hZ] at hn
  refine ⟨n, ?_, ?_, ?_, ?_⟩
  · -- the computed piece equals q*n (the final subtraction is exact)
    rw [hw]
    have hrep : Rep 53 (w - ((3 * H : Nat) : Int)) := by
      rw [hn]
      exact rep_of_dvd' (by decide) hq _ ⟨n, hZ⟩ (by omega)
    rw [rd_self hrep, hn, ← hZ]
  · rw [← hZ]; omega
  · rw [← hZ]
    have hres : Rep 53 (r - Z) := by
      by_cases hk : ex 53 r ≤ P - 53
      · obtain ⟨K, hK⟩ : ∃ K, K = 2 ^ ex 53 r := ⟨_, rfl⟩
        have hKr : (K : Int) ∣ r := by rw [hK]; exact two_pow_ex_dvd (by decide) hr
        have hKq : (K : Int) ∣ Z := by
          have : K ∣ q := by rw [hK, hq]; exact Nat.pow_dvd_pow 2 hk
          exact Int.dvd_trans (Int.natCast_dvd_natCast.2 this) ⟨n, hZ⟩
        have hlt := lt_two_pow_ex 53 r
        rw [← hK] at hlt
        exact rep_of_dvd' (by decide) hK _ (Int.dvd_sub hKr hKq) (by omega)
      · have hqr : (q : Int) ∣ r := by
          have h1 := two_pow_ex_dvd (by decide : 1 ≤ 53) hr
          have : q ∣ 2 ^ ex 53 r := by rw [hq]; exact Nat.pow_dvd_pow 2 (by omega)
          exact Int.dvd_trans (Int.natCast_dvd_natCast.2 this) h1
        exact rep_of_dvd' (by decide) hq _ (Int.dvd_sub hqr ⟨n, hZ⟩) (by omega)
    exact rd_self hres
  · have hZabs : Z.natAbs = q * n.natAbs := by rw [hZ, Int.natAbs_mul, Int.natAbs_natCast]
    obtain ⟨QM, hQM⟩ : ∃ QM, QM = q * n.natAbs := ⟨_, rfl⟩
    rw [← hQM] at hZabs
    have hlt : QM < q * (S + 1) := by
      have : q * (S + 1) = QS + q := by rw [hQS, Nat.mul_succ]
      omega
    rw [hQM] at hlt
    have := Nat.lt_of_mul_lt_mul_left hlt
    omega



/-! ## FP16 representability of the slice integers -/

/-- Slice integers fit FP16 (11 significant bits) as soon as 53 - rho ≤ 11. -/
theorem slice_fp16 (rho : Nat) (hrho : 42 ≤ rho) (n : Int) (hn : n.natAbs ≤ 2 ^ (53 - rho)) : Rep 11 n := by
  have h1 : 2 ^ (53 - rho) ≤ 2 ^ 11 := Nat.pow_le_pow_right (by decide) (by omega)
  exact rep_of_dvd (by decide) 0 n (by simp) (by simpa using Nat.le_trans hn h1)

/-- Multi-slice decomposition (the library's loop): slice k uses P_k = P - k·(54 - rho). `Dec` records the actual
    arithmetic: the computed piece equals q_k·n_k, the computed residual is exact, and the next row value is the exact residual. -/
def Dec (rho : Nat) : Nat → Nat → Int → Int → Prop
  | 0, _, r, res => res = r
  | T + 1, P, r, res =>
    ∃ n : Int, n.natAbs ≤ 2 ^ (53 - rho) ∧
      rdNear 53 (rdNear 53 (r + ((3 * 2 ^ (P - 2) : Nat) : Int)) - ((3 * 2 ^ (P - 2) : Nat) : Int))
          = ((2 ^ (P - 53) : Nat) : Int) * n ∧
      rdNear 53 (r - ((2 ^ (P - 53) : Nat) : Int) * n) = r - ((2 ^ (P - 53) : Nat) : Int) * n ∧
      Dec rho T (P - (54 - rho)) (r - ((2 ^ (P - 53) : Nat) : Int) * n) res

theorem multi_slice (rho : Nat) (hr2 : 3 ≤ rho) (hr53 : rho ≤ 53) :
    ∀ (T P : Nat) (r : Int), Rep 53 r → 54 + T * (54 - rho) ≤ P → r.natAbs ≤ 2 ^ (P - rho) →
      ∃ res : Int, Dec rho T P r res ∧ res.natAbs ≤ 2 ^ (P - T * (54 - rho) - rho) := by
  intro T
  induction T with
  | zero => intro P r _ _ hb; exact ⟨r, rfl, by simpa using hb⟩
  | succ T ih =>
    intro P r hr hP hb
    rw [Nat.succ_mul] at hP
    have hm1 : 1 ≤ 54 - rho := by omega
    have hqpos' : 0 < 2 ^ (P - 53) := Nat.two_pow_pos _
    obtain ⟨n, h1, h2, h3, h4⟩ := slice_lemma P rho (by omega) (by omega) hr53 r hr (by omega)
    have hrep' : Rep 53 (r - ((2 ^ (P - 53) : Nat) : Int) * n) := by
      have e : rdNear 53 (r - ((2 ^ (P - 53) : Nat) : Int) * n) = r - ((2 ^ (P - 53) : Nat) : Int) * n := h3
      have : Rep 53 (rdNear 53 (r - ((2 ^ (P - 53) : Nat) : Int) * n)) := (lattice 53).rd_rep _
      rw [e] at this; exact this
    have hq2 : 2 ^ (P - 53) = 2 * 2 ^ (P - 54) := by
      rw [show P - 53 = (P - 54) + 1 by omega, Nat.pow_succ]; omega
    have hP' : P - (54 - rho) - rho = P - 54 := by omega
    have hb' : (r - ((2 ^ (P - 53) : Nat) : Int) * n).natAbs ≤ 2 ^ (P - (54 - rho) - rho) := by
      rw [hP']; omega
    obtain ⟨res, hres, hresb⟩ :=
      ih (P - (54 - rho)) (r - ((2 ^ (P - 53) : Nat) : Int) * n) hrep' (by omega) hb'
    refine ⟨res, ⟨n, h4, h1, h3, hres⟩, ?_⟩
    have : P - (T + 1) * (54 - rho) - rho = P - (54 - rho) - T * (54 - rho) - rho := by
      rw [Nat.succ_mul]; omega
    rw [this]; exact hresb

/-! ## FP32 accumulation of slice products is exact in ANY order -/

theorem rep24_of_le (v : Int) (h : v.natAbs ≤ 2 ^ 24) : Rep 24 v :=
  rep_of_dvd (by decide) 0 v (by simp) (by simpa using h)

/-- The product of two slice integers (each ≤ 2^s with s ≤ 11) is exactly representable in FP32. -/
theorem product_exact (s : Nat) (hs : s ≤ 11) (n m : Int) (hn : n.natAbs ≤ 2 ^ s) (hm : m.natAbs ≤ 2 ^ s) :
    Rep 24 (n * m) := by
  apply rep24_of_le
  rw [Int.natAbs_mul]
  have h1 : n.natAbs * m.natAbs ≤ 2 ^ s * 2 ^ s := Nat.mul_le_mul hn hm
  have h2 : 2 ^ s * 2 ^ s ≤ 2 ^ 11 * 2 ^ 11 := Nat.mul_le_mul (Nat.pow_le_pow_right (by decide) hs) (Nat.pow_le_pow_right (by decide) hs)
  have h3 : 2 ^ 11 * 2 ^ 11 ≤ 2 ^ 24 := by decide
  omega

/-- Sequential FP32 accumulation (any list order): exact while the running sum stays within 2^24. -/
theorem fl32_foldl_exact (xs : List Int) :
    ∀ (acc : Int) (B : Nat), (∀ x ∈ xs, x.natAbs ≤ B) → acc.natAbs + xs.length * B ≤ 2 ^ 24 →
      xs.foldl (fun a x => rdNear 24 (a + x)) acc = acc + xs.sum := by
  induction xs with
  | nil => intro acc B _ _; simp
  | cons x xs ih =>
    intro acc B hB hlen
    have hx := hB x (by simp)
    obtain ⟨LB, hLB⟩ : ∃ LB, LB = xs.length * B := ⟨_, rfl⟩
    have hl : (x :: xs).length * B = LB + B := by
      rw [hLB, List.length_cons, Nat.succ_mul]
    rw [hl] at hlen
    have h1 : (acc + x).natAbs ≤ 2 ^ 24 := by omega
    have h2 : rdNear 24 (acc + x) = acc + x := rd_self (rep24_of_le _ h1)
    have h3 := ih (acc + x) B (fun y hy => hB y (by simp [hy])) (by rw [← hLB]; omega)
    simp only [List.foldl_cons, List.sum_cons]
    rw [h2, h3]; omega

/-- Arbitrary summation trees (any grouping and order). -/
inductive Tree where
  | leaf (x : Int)
  | node (l r : Tree)

noncomputable def Tree.eval : Tree → Int
  | .leaf x => x
  | .node l r => rdNear 24 (l.eval + r.eval)

def Tree.sum : Tree → Int
  | .leaf x => x
  | .node l r => l.sum + r.sum

def Tree.size : Tree → Nat
  | .leaf _ => 1
  | .node l r => l.size + r.size

def Tree.bounded (B : Nat) : Tree → Prop
  | .leaf x => x.natAbs ≤ B
  | .node l r => l.bounded B ∧ r.bounded B

theorem tree_exact (B : Nat) : ∀ t : Tree, t.bounded B → t.size * B ≤ 2 ^ 24 →
    t.eval = t.sum ∧ t.sum.natAbs ≤ t.size * B := by
  intro t
  induction t with
  | leaf x => intro hb _; simp [Tree.eval, Tree.sum, Tree.size, Tree.bounded] at *; omega
  | node l r ihl ihr =>
    intro hb hs
    obtain ⟨hbl, hbr⟩ := hb
    simp only [Tree.size] at hs
    rw [Nat.add_mul] at hs
    obtain ⟨el, bl⟩ := ihl hbl (by omega)
    obtain ⟨er, br⟩ := ihr hbr (by omega)
    have h1 : (l.sum + r.sum).natAbs ≤ 2 ^ 24 := by omega
    have h2 : rdNear 24 (l.sum + r.sum) = l.sum + r.sum := rd_self (rep24_of_le _ h1)
    refine ⟨?_, ?_⟩
    · simp only [Tree.eval, Tree.sum]; rw [el, er, h2]
    · simp only [Tree.size, Tree.sum]; rw [Nat.add_mul]; omega

/-- The OzBLAS partial-product theorem: with slice integers ≤ 2^s (s ≤ 11) and K·4^s ≤ 2^24, every FP16×FP16 product is exact
    in FP32 and sequential FP32 accumulation of up to K products equals the exact integer dot product. -/
theorem ozblas_partial_dot_exact (s K : Nat) (hs : s ≤ 11) (hK : K * 4 ^ s ≤ 2 ^ 24)
    (pairs : List (Int × Int)) (hlen : pairs.length ≤ K)
    (hb : ∀ p ∈ pairs, p.1.natAbs ≤ 2 ^ s ∧ p.2.natAbs ≤ 2 ^ s) :
    (∀ p ∈ pairs, Rep 24 (p.1 * p.2)) ∧
    (pairs.map (fun p => p.1 * p.2)).foldl (fun a x => rdNear 24 (a + x)) 0
      = (pairs.map (fun p => p.1 * p.2)).sum := by
  refine ⟨fun p hp => product_exact s hs _ _ (hb p hp).1 (hb p hp).2, ?_⟩
  have h4 : 2 ^ s * 2 ^ s = 4 ^ s := by rw [← Nat.mul_pow]
  have hprod : ∀ x ∈ pairs.map (fun p => p.1 * p.2), x.natAbs ≤ 4 ^ s := by
    intro x hx
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hx
    rw [Int.natAbs_mul, ← h4]
    exact Nat.mul_le_mul (hb p hp).1 (hb p hp).2
  have hl : (pairs.map (fun p => p.1 * p.2)).length * 4 ^ s ≤ 2 ^ 24 := by
    rw [List.length_map]
    exact Nat.le_trans (Nat.mul_le_mul_right _ hlen) hK
  have := fl32_foldl_exact (pairs.map (fun p => p.1 * p.2)) 0 (4 ^ s) hprod (by simpa using hl)
  simpa using this

/-! ## The slice width s = 53 - rho is the largest one for which FP32 accumulation is exact (K ≥ 2); K = 1 needs the special case -/

/-- s is the largest value with K·4^s ≤ 2^24 (equivalently s = floor((24 - log2 K)/2), i.e. rho = ceil(53 - (24 - log2 K)/2)). -/
def IsMaxS (K s : Nat) : Prop := K * 4 ^ s ≤ 2 ^ 24 ∧ 2 ^ 24 < K * 4 ^ (s + 1)

theorem isMaxS_unique {K s s' : Nat} (h : IsMaxS K s) (h' : IsMaxS K s') : s = s' := by
  obtain ⟨a1, a2⟩ := h
  obtain ⟨b1, b2⟩ := h'
  by_cases hlt : s < s'
  · have : K * 4 ^ (s + 1) ≤ K * 4 ^ s' := Nat.mul_le_mul_left K (Nat.pow_le_pow_right (by decide) hlt)
    omega
  · by_cases hgt : s' < s
    · have : K * 4 ^ (s' + 1) ≤ K * 4 ^ s := Nat.mul_le_mul_left K (Nat.pow_le_pow_right (by decide) hgt)
      omega
    · omega

/-- For K ≥ 2 the exactness condition forces s ≤ 11, so slice integers always fit FP16. -/
theorem s_le_11 (K s : Nat) (hK : 2 ≤ K) (h : K * 4 ^ s ≤ 2 ^ 24) : s ≤ 11 := by
  by_cases hs : s ≤ 11
  · exact hs
  · exfalso
    have h1 : 4 ^ 12 ≤ 4 ^ s := Nat.pow_le_pow_right (by decide) (by omega)
    have h2 : 2 * 4 ^ 12 ≤ K * 4 ^ s := Nat.mul_le_mul hK h1
    have h3 : 2 * 4 ^ 12 = 2 ^ 25 := by decide
    have h4 : 2 ^ 25 = 2 * 2 ^ 24 := by decide
    omega

theorem not_rep_2049 : ¬ Rep 11 (2049 : Int) := by
  rintro ⟨m, e, h1, h2⟩
  have h1' : 2049 = m * 2 ^ e := by simpa using h1
  cases e with
  | zero => simp at h1'; omega
  | succ e =>
    obtain ⟨u, hu⟩ : ∃ u, u = m * 2 ^ e := ⟨_, rfl⟩
    have h3 : m * 2 ^ (e + 1) = u * 2 := by rw [Nat.pow_succ, ← Nat.mul_assoc, hu]
    rw [h3] at h1'
    omega

/-- Why the library treats K = 1 as K = 2 (`dim_ = (dim == 1) ? 2 : dim`): with K = 1 the exactness condition allows s = 12,
    but 12-bit slice integers such as 2049 (≤ 2^12) are NOT FP16 numbers. -/
theorem k1_needs_special_case : 1 * 4 ^ 12 ≤ 2 ^ 24 ∧ 2049 ≤ 2 ^ 12 ∧ ¬ Rep 11 (2049 : Int) :=
  ⟨by decide, by decide, not_rep_2049⟩

/-- The slice widths used on the GPU (rho = 42, 43, 44, 46, 48, 49 for K = 2, 16, 64, 1024, 16384, 65536). -/
theorem isMaxS_examples :
    IsMaxS 2 11 ∧ IsMaxS 16 10 ∧ IsMaxS 64 9 ∧ IsMaxS 1024 7 ∧ IsMaxS 16384 5 ∧ IsMaxS 65536 4 := by
  unfold IsMaxS; decide

#print axioms OzSlice.slice_lemma
#print axioms OzSlice.slice_fp16
#print axioms OzSlice.multi_slice
#print axioms OzSlice.product_exact
#print axioms OzSlice.fl32_foldl_exact
#print axioms OzSlice.tree_exact
#print axioms OzSlice.ozblas_partial_dot_exact
#print axioms OzSlice.isMaxS_unique
#print axioms OzSlice.s_le_11
#print axioms OzSlice.k1_needs_special_case
#print axioms OzSlice.isMaxS_examples

end OzSlice

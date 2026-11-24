/-!
# cuBLAS FP64 emulation (Fixed-Point Ozaki, ADP): the coarsened ESC estimate and the per-element error bound

ADP (arXiv 2511.13778 §4) picks the number of retained mantissa bits for `C = A·B` from
`ESC_ij = x_i + y_j − z_ij + 1`, where `x_i`, `y_j` are the largest exponents of row `i` of `A` and column `j`
of `B`, and `z_ij` should be the largest exponent of a product `a_ik b_kj`.  Computing `z_ij` exactly is a
max-plus matrix product, so it is coarsened over k-blocks: for a block,
`est_b = max(MaxA_b + MinB_b, MinA_b + MaxB_b)`, and `est = max_b est_b`.  A zero entry has no exponent; the
implementation gives it a sentinel `L`.

Fitted to cuBLAS 13.10 on sm_120 (`experiments/cublas_emu/esc_model.py`, 38/38 measured bit counts reproduced,
2/2 out-of-sample predictions confirmed): block length 64, true exponents for subnormals, `L = −4195` (in units of
`floor(log2 |v|)`), and `bits = x + y − est + 54`.

Proved here (core Lean, Int/Nat only, exponents are integers, zeros are `none`):
* `paperEst_le`      : if `L + Emax ≤ 2·Emin` (every real exponent in `[Emin, Emax]`) then the coarsened estimate is
                       `≤` the true largest product exponent `z` — so ESC is never underestimated.
* `sentinel_iff`     : that sentinel condition is also necessary: if `L + Emax > 2·Emin`, a two-entry block makes the
                       estimate exceed `z`.  For FP64 (`Emin = −1074`, `Emax = 1023`): safe iff `L ≤ −3171`;
                       the fitted `−4195` is safe (`fp64_sentinel_ok`).
* `skip_zero_unsafe` : the obvious fix "take Min over nonzero entries only" is unsafe (two-entry counterexample).
* `cand_sound`, `fixedEst_le` : adding candidates that are an actual product of the block (e.g. its first or last
                       index), or `MinNZ_A + MinNZ_B` when the block's nonzero counts satisfy `nnzA + nnzB > len`
                       (pigeonhole: `pigeon`), keeps the estimate `≤ z`.  This is the zero-aware fix evaluated in
                       `esc_model.py` (4248 → 55 bits on the zero-block inputs).
* `adp_sound`        : integer fixed-point model.  Every entry of a row of `A` (column of `B`) is `< 2^XA` (`< 2^XB`)
                       in magnitude; entries are truncated toward zero to multiples of `2^{XA−m}` (`2^{XB−m}`), i.e.
                       `m` bits below the row (column) maximum; the truncated integer dot product is exact (INT8 tensor
                       core slices with exact accumulation — not modelled here).  If `est ≤ z` and
                       `XA + XB + t ≤ m + est`, then `|Σ a b − Σ ã b̃| · 2^t ≤ 2·k·Σ|a||b|`.
* `adp_paper_sound`, `adp_fixed_sound` : the two estimates plugged into `adp_sound`.

With cuBLAS's fitted rule (`XA = x+1`, `XB = y+1`, `m = x + y − est + 54`) the bit condition holds with `t = 52`,
so the truncation error is `≤ 2k·2^{-52}·(|A||B|)_ij = 4k·u·(|A||B|)_ij` per element (plus the final rounding to
FP64), versus the classical FP64 bound `≈ k·u·(|A||B|)_ij`.  The constant depends on internals we do not know
(rounding vs truncation of slices, what "retained bits" counts); the per-element form does not.

Scaling: the integer model has all real exponents `≥ 0` (multiply everything by a power of two); shifting every
exponent and `L` by `s` leaves `L + Emax ≤ 2·Emin`, `x + y − est` and the conclusion unchanged.
-/

namespace AdpEsc

/-! ## 1. Exponent pairs, the coarsened estimate, the true maximum -/

/-- exponents of `(a_k, b_k)`; `none` = the entry is zero -/
abbrev EP := Option Int × Option Int

/-- exponent with the zero sentinel `L` -/
def ex (L : Int) (o : Option Int) : Int := o.getD L

def lmax (d : Int) : List Int → Int
  | [] => d
  | a :: t => max a (lmax d t)

def lmin (d : Int) : List Int → Int
  | [] => d
  | a :: t => min a (lmin d t)

theorem lmax_mem (d : Int) : ∀ l : List Int, lmax d l ∈ d :: l
  | [] => by simp [lmax]
  | a :: t => by
    have ih := lmax_mem d t
    simp only [lmax]
    by_cases h : a ≤ lmax d t
    · rw [Int.max_eq_right h]
      simp only [List.mem_cons] at ih ⊢
      rcases ih with h' | h'
      · exact Or.inl h'
      · exact Or.inr (Or.inr h')
    · rw [Int.max_eq_left (by omega)]
      simp

theorem le_lmax (d : Int) : ∀ l : List Int, ∀ x ∈ d :: l, x ≤ lmax d l
  | [], x, hx => by
    rw [List.mem_singleton] at hx; rw [hx]; exact Int.le_refl _
  | a :: t, x, hx => by
    have ih := le_lmax d t
    simp only [lmax]
    simp only [List.mem_cons] at hx
    rcases hx with h | h | h
    · have := ih x (by rw [h]; exact List.mem_cons_self ..); omega
    · omega
    · have := ih x (List.mem_cons_of_mem _ h); omega

theorem lmin_le (d : Int) : ∀ l : List Int, ∀ x ∈ d :: l, lmin d l ≤ x
  | [], x, hx => by
    rw [List.mem_singleton] at hx; rw [hx]; exact Int.le_refl _
  | a :: t, x, hx => by
    have ih := lmin_le d t
    simp only [lmin]
    simp only [List.mem_cons] at hx
    rcases hx with h | h | h
    · have := ih x (by rw [h]; exact List.mem_cons_self ..); omega
    · omega
    · have := ih x (List.mem_cons_of_mem _ h); omega

theorem lmin_mem (d : Int) : ∀ l : List Int, lmin d l ∈ d :: l
  | [] => by simp [lmin]
  | a :: t => by
    have ih := lmin_mem d t
    simp only [lmin]
    by_cases h : lmin d t ≤ a
    · rw [Int.min_eq_right h]
      simp only [List.mem_cons] at ih ⊢
      rcases ih with h' | h'
      · exact Or.inl h'
      · exact Or.inr (Or.inr h')
    · rw [Int.min_eq_left (by omega)]
      simp

/-- the paper's block estimate `max(MaxA + MinB, MinA + MaxB)` (empty block: `2L`) -/
def blockEst (L : Int) : List EP → Int
  | [] => L + L
  | p :: ps =>
    max (lmax (ex L p.1) (ps.map fun q => ex L q.1) + lmin (ex L p.2) (ps.map fun q => ex L q.2))
        (lmin (ex L p.1) (ps.map fun q => ex L q.1) + lmax (ex L p.2) (ps.map fun q => ex L q.2))

/-- the coarsened estimate of `z`: maximum over the blocks -/
def paperEst (L : Int) : List (List EP) → Int
  | [] => L + L
  | b :: bs => max (blockEst L b) (paperEst L bs)

/-- the true largest exponent of a nonzero product (`none` if every product has a zero factor) -/
def zmax : List EP → Option Int
  | [] => none
  | (some a, some b) :: t =>
    some (match zmax t with
      | none => a + b
      | some z => max (a + b) z)
  | _ :: t => zmax t

def InRange (Emin Emax : Int) (o : Option Int) : Prop := ∀ a, o = some a → Emin ≤ a ∧ a ≤ Emax

theorem zmax_ub : ∀ (l : List EP) (z : Int), zmax l = some z →
    ∀ p ∈ l, ∀ a b, p.1 = some a → p.2 = some b → a + b ≤ z
  | [], _, h, _, _, _, _, _, _ => by simp [zmax] at h
  | (o1, o2) :: t, z, h, p, hp, a, b, ha, hb => by
    simp only [List.mem_cons] at hp
    cases o1 with
    | none =>
      simp only [zmax] at h
      rcases hp with rfl | hp
      · simp at ha
      · exact zmax_ub t z h p hp a b ha hb
    | some a1 =>
      cases o2 with
      | none =>
        simp only [zmax] at h
        rcases hp with rfl | hp
        · simp at hb
        · exact zmax_ub t z h p hp a b ha hb
      | some b1 =>
        simp only [zmax] at h
        cases ht : zmax t with
        | none =>
          rw [ht] at h
          simp only [Option.some.injEq] at h
          rcases hp with rfl | hp
          · simp at ha hb; subst ha; subst hb; omega
          · have := zmax_none_aux t ht p hp; rcases this with h1 | h1 <;> simp_all
        | some z' =>
          rw [ht] at h
          simp only [Option.some.injEq] at h
          rcases hp with rfl | hp
          · simp at ha hb; subst ha; subst hb; omega
          · have := zmax_ub t z' ht p hp a b ha hb; omega
where
  zmax_none_aux : ∀ (l : List EP), zmax l = none → ∀ p ∈ l, p.1 = none ∨ p.2 = none
    | [], _, _, hp => by simp at hp
    | (o1, o2) :: t, h, p, hp => by
      simp only [List.mem_cons] at hp
      cases o1 with
      | none =>
        rcases hp with rfl | hp
        · exact Or.inl rfl
        · exact zmax_none_aux t (by simpa [zmax] using h) p hp
      | some a1 =>
        cases o2 with
        | none =>
          rcases hp with rfl | hp
          · exact Or.inr rfl
          · exact zmax_none_aux t (by simpa [zmax] using h) p hp
        | some b1 => simp [zmax] at h

theorem zmax_none (l : List EP) (h : zmax l = none) : ∀ p ∈ l, p.1 = none ∨ p.2 = none :=
  zmax_ub.zmax_none_aux l h

theorem zmax_att : ∀ (l : List EP) (z : Int), zmax l = some z →
    ∃ p ∈ l, ∃ a b, p.1 = some a ∧ p.2 = some b ∧ a + b = z
  | [], _, h => by simp [zmax] at h
  | (o1, o2) :: t, z, h => by
    cases o1 with
    | none =>
      obtain ⟨p, hp, r⟩ := zmax_att t z (by simpa [zmax] using h)
      exact ⟨p, List.mem_cons_of_mem _ hp, r⟩
    | some a1 =>
      cases o2 with
      | none =>
        obtain ⟨p, hp, r⟩ := zmax_att t z (by simpa [zmax] using h)
        exact ⟨p, List.mem_cons_of_mem _ hp, r⟩
      | some b1 =>
        simp only [zmax] at h
        cases ht : zmax t with
        | none =>
          rw [ht] at h
          simp only [Option.some.injEq] at h
          exact ⟨(some a1, some b1), List.mem_cons_self .., a1, b1, rfl, rfl, h⟩
        | some z' =>
          rw [ht] at h
          simp only [Option.some.injEq] at h
          by_cases hc : z' ≤ a1 + b1
          · exact ⟨(some a1, some b1), List.mem_cons_self .., a1, b1, rfl, rfl, by omega⟩
          · obtain ⟨p, hp, a, b, ha, hb, e⟩ := zmax_att t z' ht
            exact ⟨p, List.mem_cons_of_mem _ hp, a, b, ha, hb, by omega⟩

/-! ## 2. The block estimate is a lower bound of `z` iff the sentinel is low enough -/

/-- core case analysis: `MaxA + MinB ≤ z` (the entry attaining `MaxA` either is a real pair, or involves a zero,
    in which case the sum is `≤ L + Emax ≤ 2·Emin ≤ z`) -/
theorem cand_le (L Emin Emax Z MA mB : Int) (bl : List EP)
    (hE : Emin ≤ Emax) (hL : L + Emax ≤ 2 * Emin) (hZ : 2 * Emin ≤ Z)
    (hR : ∀ p ∈ bl, InRange Emin Emax p.1 ∧ InRange Emin Emax p.2)
    (hub : ∀ p ∈ bl, ∀ a b, p.1 = some a → p.2 = some b → a + b ≤ Z)
    (hMA : ∃ p ∈ bl, ex L p.1 = MA) (hmB : ∀ p ∈ bl, mB ≤ ex L p.2) : MA + mB ≤ Z := by
  obtain ⟨p, hp, hpa⟩ := hMA
  have hm := hmB p hp
  obtain ⟨r1, r2⟩ := hR p hp
  cases h1 : p.1 with
  | none =>
    rw [h1] at hpa; simp only [ex, Option.getD] at hpa
    cases h2 : p.2 with
    | none => rw [h2] at hm; simp only [ex, Option.getD] at hm; omega
    | some b => rw [h2] at hm; simp only [ex, Option.getD] at hm; have := r2 b h2; omega
  | some a =>
    rw [h1] at hpa; simp only [ex, Option.getD] at hpa
    have := r1 a h1
    cases h2 : p.2 with
    | none => rw [h2] at hm; simp only [ex, Option.getD] at hm; omega
    | some b => rw [h2] at hm; simp only [ex, Option.getD] at hm; have := hub p hp a b h1 h2; omega

theorem cand_le' (L Emin Emax Z mA MB : Int) (bl : List EP)
    (hE : Emin ≤ Emax) (hL : L + Emax ≤ 2 * Emin) (hZ : 2 * Emin ≤ Z)
    (hR : ∀ p ∈ bl, InRange Emin Emax p.1 ∧ InRange Emin Emax p.2)
    (hub : ∀ p ∈ bl, ∀ a b, p.1 = some a → p.2 = some b → a + b ≤ Z)
    (hMB : ∃ p ∈ bl, ex L p.2 = MB) (hmA : ∀ p ∈ bl, mA ≤ ex L p.1) : mA + MB ≤ Z := by
  obtain ⟨p, hp, hpb⟩ := hMB
  have hm := hmA p hp
  obtain ⟨r1, r2⟩ := hR p hp
  cases h2 : p.2 with
  | none =>
    rw [h2] at hpb; simp only [ex, Option.getD] at hpb
    cases h1 : p.1 with
    | none => rw [h1] at hm; simp only [ex, Option.getD] at hm; omega
    | some a => rw [h1] at hm; simp only [ex, Option.getD] at hm; have := r1 a h1; omega
  | some b =>
    rw [h2] at hpb; simp only [ex, Option.getD] at hpb
    have := r2 b h2
    cases h1 : p.1 with
    | none => rw [h1] at hm; simp only [ex, Option.getD] at hm; omega
    | some a => rw [h1] at hm; simp only [ex, Option.getD] at hm; have := hub p hp a b h1 h2; omega

theorem mem_col_max (L : Int) (f : EP → Option Int) (p : EP) (ps : List EP) :
    ∃ q ∈ p :: ps, ex L (f q) = lmax (ex L (f p)) (ps.map fun q => ex L (f q)) := by
  have h := lmax_mem (ex L (f p)) (ps.map fun q => ex L (f q))
  simp only [List.mem_cons, List.mem_map] at h
  rcases h with h | ⟨q, hq, e⟩
  · exact ⟨p, List.mem_cons_self .., h.symm⟩
  · exact ⟨q, List.mem_cons_of_mem _ hq, e⟩

theorem mem_col_min (L : Int) (f : EP → Option Int) (p : EP) (ps : List EP) :
    ∃ q ∈ p :: ps, ex L (f q) = lmin (ex L (f p)) (ps.map fun q => ex L (f q)) := by
  have h := lmin_mem (ex L (f p)) (ps.map fun q => ex L (f q))
  simp only [List.mem_cons, List.mem_map] at h
  rcases h with h | ⟨q, hq, e⟩
  · exact ⟨p, List.mem_cons_self .., h.symm⟩
  · exact ⟨q, List.mem_cons_of_mem _ hq, e⟩

theorem col_min_le (L : Int) (f : EP → Option Int) (p : EP) (ps : List EP) :
    ∀ q ∈ p :: ps, lmin (ex L (f p)) (ps.map fun q => ex L (f q)) ≤ ex L (f q) := by
  intro q hq
  apply lmin_le
  simp only [List.mem_cons] at hq ⊢
  rcases hq with rfl | hq
  · exact Or.inl rfl
  · exact Or.inr (List.mem_map_of_mem hq)

theorem col_le_max (L : Int) (f : EP → Option Int) (p : EP) (ps : List EP) :
    ∀ q ∈ p :: ps, ex L (f q) ≤ lmax (ex L (f p)) (ps.map fun q => ex L (f q)) := by
  intro q hq
  apply le_lmax
  simp only [List.mem_cons] at hq ⊢
  rcases hq with rfl | hq
  · exact Or.inl rfl
  · exact Or.inr (List.mem_map_of_mem hq)

theorem blockEst_le (L Emin Emax Z : Int) (bl : List EP)
    (hE : Emin ≤ Emax) (hL : L + Emax ≤ 2 * Emin) (hZ : 2 * Emin ≤ Z)
    (hR : ∀ p ∈ bl, InRange Emin Emax p.1 ∧ InRange Emin Emax p.2)
    (hub : ∀ p ∈ bl, ∀ a b, p.1 = some a → p.2 = some b → a + b ≤ Z) : blockEst L bl ≤ Z := by
  cases bl with
  | nil => simp only [blockEst]; omega
  | cons p ps =>
    simp only [blockEst]
    have c1 := cand_le L Emin Emax Z _ _ (p :: ps) hE hL hZ hR hub (mem_col_max L Prod.fst p ps)
      (col_min_le L Prod.snd p ps)
    have c2 := cand_le' L Emin Emax Z _ _ (p :: ps) hE hL hZ hR hub (mem_col_max L Prod.snd p ps)
      (col_min_le L Prod.fst p ps)
    omega

theorem two_emin_le (Emin Emax z : Int) (l : List EP)
    (hR : ∀ p ∈ l, InRange Emin Emax p.1 ∧ InRange Emin Emax p.2) (hz : zmax l = some z) :
    2 * Emin ≤ z := by
  obtain ⟨p, hp, a, b, ha, hb, e⟩ := zmax_att l z hz
  have := ((hR p hp).1 a ha).1
  have := ((hR p hp).2 b hb).1
  omega

theorem maxEst_le (L z : Int) (f : List EP → Int) (bls : List (List EP)) (h0 : L + L ≤ z)
    (hb : ∀ b ∈ bls, f b ≤ z) :
    (bls.foldr (fun b acc => max (f b) acc) (L + L)) ≤ z := by
  induction bls with
  | nil => simpa using h0
  | cons b bs ih =>
    simp only [List.foldr]
    have := hb b (List.mem_cons_self ..)
    have := ih (fun b' hb' => hb b' (List.mem_cons_of_mem _ hb'))
    omega

theorem paperEst_eq (L : Int) : ∀ bls : List (List EP),
    paperEst L bls = bls.foldr (fun b acc => max (blockEst L b) acc) (L + L)
  | [] => rfl
  | b :: bs => by simp only [paperEst, List.foldr]; rw [paperEst_eq L bs]

/-- **soundness of the coarsened estimate**: with `L + Emax ≤ 2·Emin`, `paperEst ≤ z` -/
theorem paperEst_le (L Emin Emax z : Int) (bls : List (List EP))
    (hE : Emin ≤ Emax) (hL : L + Emax ≤ 2 * Emin)
    (hR : ∀ p ∈ bls.flatten, InRange Emin Emax p.1 ∧ InRange Emin Emax p.2)
    (hz : zmax bls.flatten = some z) : paperEst L bls ≤ z := by
  have hZ := two_emin_le Emin Emax z _ hR hz
  rw [paperEst_eq]
  apply maxEst_le L z _ bls (by omega)
  intro b hb
  have mem : ∀ p ∈ b, p ∈ bls.flatten := fun p hp => List.mem_flatten.2 ⟨b, hb, hp⟩
  exact blockEst_le L Emin Emax z b hE hL hZ (fun p hp => hR p (mem p hp))
    (fun p hp a b' ha hb' => zmax_ub _ z hz p (mem p hp) a b' ha hb')

/-- the two-entry block that breaks a too-high sentinel: `A = (0, 2^Emin)`, `B = (2^Emax, 2^Emin)` -/
def badBlock (Emin Emax : Int) : List EP := [(none, some Emax), (some Emin, some Emin)]

theorem badBlock_breaks (L Emin Emax : Int) (hE : Emin ≤ Emax) (hbad : 2 * Emin < L + Emax) :
    zmax (badBlock Emin Emax) = some (2 * Emin) ∧ 2 * Emin < blockEst L (badBlock Emin Emax) := by
  constructor
  · simp only [badBlock, zmax]; congr 1; omega
  · simp only [badBlock, blockEst, ex, Option.getD, List.map, lmax, lmin]
    omega

/-- **the sentinel condition is exactly right**: the coarsened estimate is `≤ z` on every input with exponents in
    `[Emin, Emax]` iff `L + Emax ≤ 2·Emin` -/
theorem sentinel_iff (L Emin Emax : Int) (hE : Emin ≤ Emax) :
    (∀ bls : List (List EP), (∀ p ∈ bls.flatten, InRange Emin Emax p.1 ∧ InRange Emin Emax p.2) →
      ∀ z, zmax bls.flatten = some z → paperEst L bls ≤ z) ↔ L + Emax ≤ 2 * Emin := by
  constructor
  · intro h
    rcases Int.lt_or_le (2 * Emin) (L + Emax) with hbad | hok
    · exfalso
      obtain ⟨hz, hlt⟩ := badBlock_breaks L Emin Emax hE hbad
      have hR : ∀ p ∈ [badBlock Emin Emax].flatten, InRange Emin Emax p.1 ∧ InRange Emin Emax p.2 := by
        intro p hp
        obtain ⟨l, hl, hpl⟩ := List.mem_flatten.1 hp
        rw [List.mem_singleton] at hl
        subst hl
        rcases List.mem_cons.1 hpl with rfl | hpl
        · refine ⟨fun a ha => ?_, fun a ha => ?_⟩
          · cases ha
          · cases ha; exact ⟨hE, Int.le_refl _⟩
        rcases List.mem_cons.1 hpl with rfl | hpl
        · refine ⟨fun a ha => ?_, fun a ha => ?_⟩
          · cases ha; exact ⟨Int.le_refl _, hE⟩
          · cases ha; exact ⟨Int.le_refl _, hE⟩
        · cases hpl
      have hz' : zmax [badBlock Emin Emax].flatten = some (2 * Emin) := by
        show zmax (badBlock Emin Emax ++ []) = _
        rw [List.append_nil]; exact hz
      have := h _ hR _ hz'
      simp only [paperEst] at this
      omega
    · exact hok
  · intro hL bls hR z hz
    exact paperEst_le L Emin Emax z bls hE hL hR hz

/-- FP64: real exponents lie in `[−1074, 1023]`; the fitted cuBLAS sentinel `−4195` is safe, `−3170` would not be -/
theorem fp64_sentinel_ok : (-4195 : Int) + 1023 ≤ 2 * (-1074) := by decide
theorem fp64_threshold (L : Int) : L + 1023 ≤ 2 * (-1074) ↔ L ≤ -3171 := by omega

/-- the naive zero-aware fix — take `MinB` over the *nonzero* entries only — is unsafe: in the block
    `A = (2^0, 2^-1000)`, `B = (0, 2^0)` the entry attaining `MaxA` meets a zero of `B`, so
    `MaxA + MinNZ_B = 0` while the only nonzero product has exponent `−1000` -/
theorem skip_zero_unsafe :
    let bl : List EP := [(some 0, none), (some (-1000), some 0)]
    zmax bl = some (-1000) ∧
    (∃ p ∈ bl, p.1 = some 0) ∧ (∀ p ∈ bl, ∀ b, p.2 = some b → (0 : Int) ≤ b) ∧ (-1000 : Int) < 0 + 0 := by
  refine ⟨rfl, ⟨_, List.mem_cons_self .., rfl⟩, ?_, by decide⟩
  intro p hp b hb
  rcases List.mem_cons.1 hp with rfl | hp
  · cases hb
  rcases List.mem_cons.1 hp with rfl | hp
  · cases hb; exact Int.le_refl _
  · cases hp

/-! ## 3. Zero-aware candidates that keep the estimate sound -/

def cntA : List EP → Nat
  | [] => 0
  | p :: t => (if p.1.isSome then 1 else 0) + cntA t

def cntB : List EP → Nat
  | [] => 0
  | p :: t => (if p.2.isSome then 1 else 0) + cntB t

def cntAB : List EP → Nat
  | [] => 0
  | p :: t => (if p.1.isSome && p.2.isSome then 1 else 0) + cntAB t

/-- pigeonhole: `nnzA + nnzB ≤ len + #(both nonzero)` -/
theorem pigeon : ∀ l : List EP, cntA l + cntB l ≤ l.length + cntAB l
  | [] => by simp [cntA, cntB, cntAB]
  | (o1, o2) :: t => by
    have := pigeon t
    cases o1 <;> cases o2 <;> simp [cntA, cntB, cntAB] <;> omega

theorem exists_real_of_cntAB : ∀ l : List EP, 0 < cntAB l → ∃ p ∈ l, ∃ a b, p.1 = some a ∧ p.2 = some b
  | [], h => by simp [cntAB] at h
  | (o1, o2) :: t, h => by
    cases o1 with
    | none =>
      obtain ⟨p, hp, r⟩ := exists_real_of_cntAB t (by simpa [cntAB] using h)
      exact ⟨p, List.mem_cons_of_mem _ hp, r⟩
    | some a =>
      cases o2 with
      | none =>
        obtain ⟨p, hp, r⟩ := exists_real_of_cntAB t (by simpa [cntAB] using h)
        exact ⟨p, List.mem_cons_of_mem _ hp, r⟩
      | some b => exact ⟨_, List.mem_cons_self .., a, b, rfl, rfl⟩

/-- admissible extra candidates for a block -/
inductive Cand (bl : List EP) : Int → Prop
  /-- an actual nonzero product of the block (e.g. its first or last index) -/
  | rep (p : EP) (hp : p ∈ bl) (a b : Int) (ha : p.1 = some a) (hb : p.2 = some b) : Cand bl (a + b)
  /-- lower bounds of the nonzero exponents of A and of B in the block, when `nnzA + nnzB > len` -/
  | pig (mA mB : Int) (hA : ∀ p ∈ bl, ∀ a, p.1 = some a → mA ≤ a) (hB : ∀ p ∈ bl, ∀ b, p.2 = some b → mB ≤ b)
      (hc : bl.length < cntA bl + cntB bl) : Cand bl (mA + mB)

theorem cand_sound (bl : List EP) (c z : Int) (hc : Cand bl c)
    (hub : ∀ p ∈ bl, ∀ a b, p.1 = some a → p.2 = some b → a + b ≤ z) : c ≤ z := by
  cases hc with
  | rep p hp a b ha hb => exact hub p hp a b ha hb
  | pig mA mB hA hB hc =>
    have := pigeon bl
    obtain ⟨p, hp, a, b, ha, hb⟩ := exists_real_of_cntAB bl (by omega)
    have := hA p hp a ha
    have := hB p hp b hb
    have := hub p hp a b ha hb
    omega

/-- the fixed estimate: per block, the paper's candidate and any list of admissible extra candidates -/
def fixedEst (L : Int) (cands : List EP → List Int) : List (List EP) → Int
  | [] => L + L
  | b :: bs => max (lmax (blockEst L b) (cands b)) (fixedEst L cands bs)

theorem fixedEst_le (L Emin Emax z : Int) (cands : List EP → List Int) (bls : List (List EP))
    (hE : Emin ≤ Emax) (hL : L + Emax ≤ 2 * Emin)
    (hR : ∀ p ∈ bls.flatten, InRange Emin Emax p.1 ∧ InRange Emin Emax p.2)
    (hc : ∀ b ∈ bls, ∀ c ∈ cands b, Cand b c)
    (hz : zmax bls.flatten = some z) : fixedEst L cands bls ≤ z :=
  fixedEst_le_aux L Emin Emax z cands bls hE hL (two_emin_le Emin Emax z _ hR hz) hR hc
    (fun p hp a b ha hb => zmax_ub _ z hz p hp a b ha hb)
where
  fixedEst_le_aux (L Emin Emax z : Int) (cands : List EP → List Int) : ∀ (bls : List (List EP)),
      Emin ≤ Emax → L + Emax ≤ 2 * Emin → 2 * Emin ≤ z →
      (∀ p ∈ bls.flatten, InRange Emin Emax p.1 ∧ InRange Emin Emax p.2) →
      (∀ b ∈ bls, ∀ c ∈ cands b, Cand b c) →
      (∀ p ∈ bls.flatten, ∀ a b, p.1 = some a → p.2 = some b → a + b ≤ z) →
      fixedEst L cands bls ≤ z
    | [], _, _, hZ, _, _, _ => by simp only [fixedEst]; omega
    | b :: bs, hE, hL, hZ, hR, hc, hub => by
      simp only [fixedEst]
      have memb : ∀ p ∈ b, p ∈ (b :: bs).flatten := fun p hp =>
        List.mem_flatten.2 ⟨b, List.mem_cons_self .., hp⟩
      have memt : ∀ p ∈ bs.flatten, p ∈ (b :: bs).flatten := fun p hp => by
        simp only [List.flatten_cons]; exact List.mem_append_right _ hp
      have hubb : ∀ p ∈ b, ∀ a b', p.1 = some a → p.2 = some b' → a + b' ≤ z :=
        fun p hp => hub p (memb p hp)
      have h1 := blockEst_le L Emin Emax z b hE hL hZ (fun p hp => hR p (memb p hp)) hubb
      have h2 : lmax (blockEst L b) (cands b) ≤ z := by
        obtain h | h := List.mem_cons.1 (lmax_mem (blockEst L b) (cands b))
        · omega
        · have := cand_sound b _ z (hc b (List.mem_cons_self ..) _ h) hubb; omega
      have h3 := fixedEst_le_aux L Emin Emax z cands bs hE hL hZ (fun p hp => hR p (memt p hp))
        (fun b' hb' => hc b' (List.mem_cons_of_mem _ hb')) (fun p hp => hub p (memt p hp))
      omega

/-! ## 4. Integer fixed-point model: per-element error bound -/

def dot : List (Int × Int) → Int
  | [] => 0
  | p :: t => p.1 * p.2 + dot t

def absdot : List (Int × Int) → Nat
  | [] => 0
  | p :: t => p.1.natAbs * p.2.natAbs + absdot t

/-- truncation toward zero to a multiple of `2^g` -/
def tr (g : Nat) (a : Int) : Int :=
  if 0 ≤ a then ((a.natAbs / 2 ^ g * 2 ^ g : Nat) : Int) else -((a.natAbs / 2 ^ g * 2 ^ g : Nat) : Int)

theorem tr_natAbs_le (g : Nat) (a : Int) : (tr g a).natAbs ≤ a.natAbs := by
  have := Nat.div_mul_le_self a.natAbs (2 ^ g)
  unfold tr; split <;> omega

theorem tr_err (g : Nat) (a : Int) : (a - tr g a).natAbs < 2 ^ g := by
  have hd := Nat.mod_add_div a.natAbs (2 ^ g)
  have hm := Nat.mod_lt a.natAbs (Nat.two_pow_pos g)
  rw [Nat.mul_comm] at hd
  unfold tr
  generalize a.natAbs / 2 ^ g * 2 ^ g = q at *
  generalize 2 ^ g = P at *
  split <;> omega

theorem tr_zero (g : Nat) : tr g 0 = 0 := by
  unfold tr
  simp only [Int.le_refl, ↓reduceIte, Int.natAbs_zero, Nat.zero_div, Nat.zero_mul]
  rfl

/-- one term: `ab − ãb̃ = a(b − b̃) + b̃(a − ã)` -/
theorem term_err (a b ta tb : Int) (XA XB ga gb : Nat)
    (ha : a.natAbs < 2 ^ XA) (hb : b.natAbs < 2 ^ XB) (htb : tb.natAbs ≤ b.natAbs)
    (ea : (a - ta).natAbs < 2 ^ ga) (eb : (b - tb).natAbs < 2 ^ gb) :
    (a * b - ta * tb).natAbs ≤ 2 ^ XA * 2 ^ gb + 2 ^ XB * 2 ^ ga := by
  have e : a * b - ta * tb = a * (b - tb) + tb * (a - ta) := by
    rw [Int.mul_sub, Int.mul_sub, Int.mul_comm tb a, Int.mul_comm tb ta]; omega
  rw [e]
  have h := Int.natAbs_add_le (a * (b - tb)) (tb * (a - ta))
  rw [Int.natAbs_mul, Int.natAbs_mul] at h
  have h1 : a.natAbs * (b - tb).natAbs ≤ 2 ^ XA * 2 ^ gb := Nat.mul_le_mul (by omega) (by omega)
  have h2 : tb.natAbs * (a - ta).natAbs ≤ 2 ^ XB * 2 ^ ga := Nat.mul_le_mul (by omega) (by omega)
  omega

def trl (ga gb : Nat) (l : List (Int × Int)) : List (Int × Int) := l.map fun p => (tr ga p.1, tr gb p.2)

theorem dot_err (ga gb XA XB : Nat) : ∀ l : List (Int × Int),
    (∀ p ∈ l, p.1.natAbs < 2 ^ XA ∧ p.2.natAbs < 2 ^ XB) →
    (dot l - dot (trl ga gb l)).natAbs ≤ l.length * (2 ^ XA * 2 ^ gb + 2 ^ XB * 2 ^ ga)
  | [], _ => by simp [dot, trl]
  | p :: t, h => by
    have ih := dot_err ga gb XA XB t (fun q hq => h q (List.mem_cons_of_mem _ hq))
    obtain ⟨ha, hb⟩ := h p (List.mem_cons_self ..)
    have hp := term_err p.1 p.2 (tr ga p.1) (tr gb p.2) XA XB ga gb ha hb (tr_natAbs_le gb p.2)
      (tr_err ga p.1) (tr_err gb p.2)
    simp only [trl, List.map, dot, List.length_cons] at ih ⊢
    have e : p.1 * p.2 + dot t - (tr ga p.1 * tr gb p.2 + dot (List.map (fun p => (tr ga p.1, tr gb p.2)) t))
        = (p.1 * p.2 - tr ga p.1 * tr gb p.2) + (dot t - dot (List.map (fun p => (tr ga p.1, tr gb p.2)) t)) := by
      omega
    rw [e, Nat.succ_mul]
    have := Int.natAbs_add_le (p.1 * p.2 - tr ga p.1 * tr gb p.2)
      (dot t - dot (List.map (fun p => (tr ga p.1, tr gb p.2)) t))
    omega

theorem le_absdot : ∀ l : List (Int × Int), ∀ p ∈ l, p.1.natAbs * p.2.natAbs ≤ absdot l
  | [], _, hp => by simp at hp
  | q :: t, p, hp => by
    simp only [absdot]
    rcases List.mem_cons.1 hp with rfl | hp
    · omega
    · have := le_absdot t p hp; omega

/-- exponent of an integer: `floor(log2 |v|)`, `none` for zero -/
def e (v : Int) : Option Int := if v = 0 then none else some (Nat.log2 v.natAbs : Int)

def ep (p : Int × Int) : EP := (e p.1, e p.2)

theorem e_some {v a : Int} (h : e v = some a) : v ≠ 0 ∧ a = (Nat.log2 v.natAbs : Int) := by
  unfold e at h
  by_cases hv : v = 0
  · simp only [hv, ↓reduceIte] at h; cases h
  · simp only [hv, ↓reduceIte] at h; exact ⟨hv, (Option.some.inj h).symm⟩

theorem e_none {v : Int} (h : e v = none) : v = 0 := by
  unfold e at h
  by_cases hv : v = 0
  · exact hv
  · simp only [hv, ↓reduceIte] at h; cases h

theorem dot_zero_of_zmax_none (ga gb : Nat) : ∀ l : List (Int × Int), zmax (l.map ep) = none →
    dot l = 0 ∧ dot (trl ga gb l) = 0
  | [], _ => ⟨rfl, rfl⟩
  | p :: t, h => by
    have hnone := zmax_none _ h
    have h' : zmax (ep p :: t.map ep) = none := h
    have ht : zmax (t.map ep) = none := by
      revert h'
      simp only [ep]
      cases e p.1 <;> cases e p.2 <;> intro h' <;> simp only [zmax] at h' <;> first | exact h' | cases h'
    obtain ⟨ih1, ih2⟩ := dot_zero_of_zmax_none ga gb t ht
    have hp : p.1 = 0 ∨ p.2 = 0 := by
      rcases hnone (ep p) (List.mem_cons_self ..) with h0 | h0
      · exact Or.inl (e_none h0)
      · exact Or.inr (e_none h0)
    simp only [trl, List.map, dot] at ih2 ⊢
    rcases hp with h0 | h0
    · rw [h0, tr_zero, Int.zero_mul, Int.zero_mul, ih1, ih2]; exact ⟨rfl, rfl⟩
    · rw [h0, tr_zero, Int.mul_zero, Int.mul_zero, ih1, ih2]; exact ⟨rfl, rfl⟩

/-- **per-element error bound of the fixed-point model**, for any estimate `est ≤ z` -/
theorem adp_sound (l : List (Int × Int)) (est : Int) (t m XA XB ga gb : Nat)
    (hest : ∀ z, zmax (l.map ep) = some z → est ≤ z)
    (hAB : ∀ p ∈ l, p.1.natAbs < 2 ^ XA ∧ p.2.natAbs < 2 ^ XB)
    (hga : ga + m = XA) (hgb : gb + m = XB)
    (hbits : ((XA + XB + t : Nat) : Int) ≤ m + est) :
    (dot l - dot (trl ga gb l)).natAbs * 2 ^ t ≤ 2 * l.length * absdot l := by
  cases hz : zmax (l.map ep) with
  | none =>
    obtain ⟨h1, h2⟩ := dot_zero_of_zmax_none ga gb l hz
    rw [h1, h2]; simp
  | some z =>
    obtain ⟨q, hq, a, b, ha, hb, hab⟩ := zmax_att _ z hz
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hq
    have hle := hest z hz
    obtain ⟨hp1, ha⟩ := e_some ha
    obtain ⟨hp2, hb⟩ := e_some hb
    -- `Z = est.toNat ≤ log2|p1| + log2|p2|`
    have hZ : est.toNat ≤ Nat.log2 p.1.natAbs + Nat.log2 p.2.natAbs := by omega
    have hpow : 2 ^ est.toNat ≤ p.1.natAbs * p.2.natAbs := by
      have h1 := Nat.log2_self_le (n := p.1.natAbs) (by omega)
      have h2 := Nat.log2_self_le (n := p.2.natAbs) (by omega)
      calc 2 ^ est.toNat ≤ 2 ^ (Nat.log2 p.1.natAbs + Nat.log2 p.2.natAbs) := Nat.pow_le_pow_right (by decide) hZ
        _ = 2 ^ Nat.log2 p.1.natAbs * 2 ^ Nat.log2 p.2.natAbs := Nat.pow_add ..
        _ ≤ p.1.natAbs * p.2.natAbs := Nat.mul_le_mul h1 h2
    have habs := le_absdot l p hp
    have herr := dot_err ga gb XA XB l hAB
    -- `(2^XA·2^gb + 2^XB·2^ga)·2^t ≤ 2·2^Z`
    have hb1 : XA + gb + t ≤ est.toNat := by omega
    have hb2 : XB + ga + t ≤ est.toNat := by omega
    have p1 : 2 ^ XA * 2 ^ gb * 2 ^ t ≤ 2 ^ est.toNat := by
      rw [← Nat.pow_add, ← Nat.pow_add]; exact Nat.pow_le_pow_right (by decide) hb1
    have p2 : 2 ^ XB * 2 ^ ga * 2 ^ t ≤ 2 ^ est.toNat := by
      rw [← Nat.pow_add, ← Nat.pow_add]; exact Nat.pow_le_pow_right (by decide) hb2
    have key : (2 ^ XA * 2 ^ gb + 2 ^ XB * 2 ^ ga) * 2 ^ t ≤ 2 * absdot l := by
      rw [Nat.add_mul]; omega
    calc (dot l - dot (trl ga gb l)).natAbs * 2 ^ t
        ≤ l.length * (2 ^ XA * 2 ^ gb + 2 ^ XB * 2 ^ ga) * 2 ^ t := Nat.mul_le_mul_right _ herr
      _ = l.length * ((2 ^ XA * 2 ^ gb + 2 ^ XB * 2 ^ ga) * 2 ^ t) := Nat.mul_assoc ..
      _ ≤ l.length * (2 * absdot l) := Nat.mul_le_mul_left _ key
      _ = 2 * l.length * absdot l := by rw [← Nat.mul_assoc, Nat.mul_comm l.length 2]

/-- exponents of a nonzero integer `< 2^X` are in `[0, X]` -/
theorem e_range (v : Int) (X : Nat) (h : v.natAbs < 2 ^ X) : InRange 0 X (e v) := by
  intro a ha
  obtain ⟨hv, ha⟩ := e_some ha
  subst ha
  have hlt : Nat.log2 v.natAbs < X := by
    rcases Nat.lt_or_ge (Nat.log2 v.natAbs) X with h' | h'
    · exact h'
    · exfalso
      have := Nat.log2_self_le (n := v.natAbs) (by omega)
      have := Nat.pow_le_pow_right (n := 2) (by decide) h'
      omega
  generalize Nat.log2 v.natAbs = n at hlt ⊢
  exact ⟨Int.natCast_nonneg n, by omega⟩

theorem range_of_bound (vb : List (List (Int × Int))) (XA XB : Nat)
    (hAB : ∀ p ∈ vb.flatten, p.1.natAbs < 2 ^ XA ∧ p.2.natAbs < 2 ^ XB) :
    ∀ p ∈ (vb.map (List.map ep)).flatten, InRange 0 (max XA XB : Nat) p.1 ∧ InRange 0 (max XA XB : Nat) p.2 := by
  intro p hp
  rw [← List.map_flatten] at hp
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hp
  obtain ⟨h1, h2⟩ := hAB q hq
  constructor
  · intro a ha
    have := e_range q.1 XA h1 a ha
    exact ⟨this.1, Int.le_trans this.2 (Int.ofNat_le.2 (Nat.le_max_left XA XB))⟩
  · intro a ha
    have := e_range q.2 XB h2 a ha
    exact ⟨this.1, Int.le_trans this.2 (Int.ofNat_le.2 (Nat.le_max_right XA XB))⟩

/-- **the paper's estimate in the model**: values in k-blocks `vb`, sentinel `L ≤ −max(XA, XB)`
    (the scaled form of `L + Emax ≤ 2·Emin` with `Emin = 0`) -/
theorem adp_paper_sound (vb : List (List (Int × Int))) (L : Int) (t m XA XB ga gb : Nat)
    (hL : L + (max XA XB : Nat) ≤ 0)
    (hAB : ∀ p ∈ vb.flatten, p.1.natAbs < 2 ^ XA ∧ p.2.natAbs < 2 ^ XB)
    (hga : ga + m = XA) (hgb : gb + m = XB)
    (hbits : ((XA + XB + t : Nat) : Int) ≤ m + paperEst L (vb.map (List.map ep))) :
    (dot vb.flatten - dot (trl ga gb vb.flatten)).natAbs * 2 ^ t ≤ 2 * vb.flatten.length * absdot vb.flatten := by
  apply adp_sound _ _ t m XA XB ga gb _ hAB hga hgb hbits
  intro z hz
  rw [List.map_flatten] at hz
  exact paperEst_le L 0 (max XA XB : Nat) z _ (by omega) (by omega) (range_of_bound vb XA XB hAB) hz

/-- **the zero-aware fix in the model**: any admissible extra candidates keep the bound -/
theorem adp_fixed_sound (vb : List (List (Int × Int))) (L : Int) (cands : List EP → List Int)
    (t m XA XB ga gb : Nat)
    (hL : L + (max XA XB : Nat) ≤ 0)
    (hAB : ∀ p ∈ vb.flatten, p.1.natAbs < 2 ^ XA ∧ p.2.natAbs < 2 ^ XB)
    (hc : ∀ b ∈ vb.map (List.map ep), ∀ c ∈ cands b, Cand b c)
    (hga : ga + m = XA) (hgb : gb + m = XB)
    (hbits : ((XA + XB + t : Nat) : Int) ≤ m + fixedEst L cands (vb.map (List.map ep))) :
    (dot vb.flatten - dot (trl ga gb vb.flatten)).natAbs * 2 ^ t ≤ 2 * vb.flatten.length * absdot vb.flatten := by
  apply adp_sound _ _ t m XA XB ga gb _ hAB hga hgb hbits
  intro z hz
  rw [List.map_flatten] at hz
  exact fixedEst_le L 0 (max XA XB : Nat) z cands _ (by omega) (by omega) (range_of_bound vb XA XB hAB) hc hz

/-- cuBLAS's fitted rule `m = x + y − est + 54` with `XA = x + 1`, `XB = y + 1` meets the bit condition with
    `t = 52`, i.e. truncation error `≤ 2k·2^{-52}·Σ|a||b| = 4k·u·Σ|a||b|` -/
theorem cublas_rule (x y est m : Int) (hm : m = x + y - est + 54) : (x + 1) + (y + 1) + 52 ≤ m + est := by
  omega

end AdpEsc

#print axioms AdpEsc.sentinel_iff
#print axioms AdpEsc.fp64_sentinel_ok
#print axioms AdpEsc.skip_zero_unsafe
#print axioms AdpEsc.fixedEst_le
#print axioms AdpEsc.adp_sound
#print axioms AdpEsc.adp_paper_sound
#print axioms AdpEsc.adp_fixed_sound
#print axioms AdpEsc.cublas_rule

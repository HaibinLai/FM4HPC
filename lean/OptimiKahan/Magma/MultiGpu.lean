/-!
# MAGMA v2.10.0 multi-GPU status logic for any number of GPUs (HANDOFF §29.19)

The ngpu ≥ 2 paths could not be run (only GPU 0 may be used).  Their status bookkeeping is proved here for every
`ngpu`, `nb`, `n`: the block-cyclic column distribution and the global info index of the multi-GPU LU and Cholesky.
-/
namespace MagmaMultiGpu

/-- `zgetrf2_mgpu.cpp` l.162–166: `n_local[d] = ((n/nb)/ngpu)*nb`, plus `nb` if `d < (n/nb) % ngpu`, plus `n % nb` if
    `d = (n/nb) % ngpu` -/
def nLocal (n nb ngpu d : Nat) : Nat :=
  (n / nb / ngpu) * nb + (if d < (n / nb) % ngpu then nb else if d = (n / nb) % ngpu then n % nb else 0)

def sumLocal (n nb ngpu : Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => sumLocal n nb ngpu k + nLocal n nb ngpu k

/-- partial sums: for `k > r` the extra terms contribute `r·nb + n % nb` -/
theorem sumLocal_closed (n nb ngpu : Nat) : ∀ k, (n / nb) % ngpu < k →
    sumLocal n nb ngpu k = k * ((n / nb / ngpu) * nb) + ((n / nb) % ngpu) * nb + n % nb := by
  intro k
  induction k with
  | zero => intro h; omega
  | succ k ih =>
    intro h
    simp only [sumLocal, nLocal]
    by_cases hk : (n / nb) % ngpu < k
    · rw [ih hk]
      have h1 : ¬ k < (n / nb) % ngpu := by omega
      have h2 : ¬ k = (n / nb) % ngpu := by omega
      simp only [h1, h2, ite_false, Nat.succ_mul]; omega
    · have hr : (n / nb) % ngpu = k := by omega
      have hs : ∀ j, j ≤ k → sumLocal n nb ngpu j = j * ((n / nb / ngpu) * nb) + j * nb := by
        intro j
        induction j with
        | zero => intro _; simp [sumLocal]
        | succ j ihj =>
          intro hj
          simp only [sumLocal, nLocal]
          have hj' : j < (n / nb) % ngpu := by omega
          rw [ihj (by omega)]; simp only [hj', ite_true, Nat.succ_mul]; omega
      rw [hs k (Nat.le_refl k)]
      have h1 : ¬ k < (n / nb) % ngpu := by omega
      simp only [hr, Nat.lt_irrefl, ite_false, ite_true, Nat.succ_mul]; omega

/-- **the local column counts add up to n, for every n, nb > 0, ngpu > 0** -/
theorem distribution_covers (n nb ngpu : Nat) (hg : 0 < ngpu) : sumLocal n nb ngpu ngpu = n := by
  have hr : (n / nb) % ngpu < ngpu := Nat.mod_lt _ hg
  rw [sumLocal_closed n nb ngpu ngpu hr]
  have e1 : n / nb = ngpu * (n / nb / ngpu) + (n / nb) % ngpu := (Nat.div_add_mod (n / nb) ngpu).symm
  have e2 : n = nb * (n / nb) + n % nb := (Nat.div_add_mod n nb).symm
  have : ngpu * ((n / nb / ngpu) * nb) + ((n / nb) % ngpu) * nb = (n / nb) * nb := by
    calc ngpu * ((n / nb / ngpu) * nb) + ((n / nb) % ngpu) * nb
        = (ngpu * (n / nb / ngpu) + (n / nb) % ngpu) * nb := by rw [Nat.add_mul, Nat.mul_assoc]
      _ = (n / nb) * nb := by rw [← e1]
  rw [this]; rw [Nat.mul_comm] at e2; omega

/-! ### Per-device column counts (round 42 checked them by enumeration; proved here for all n, nb, ngpu)

`magmablas/zbcyclic.cpp` puts global column `j` on device `(j / nb) % ngpu`.  `colsOn nb ngpu d n` counts the columns
`j < n` that land on device `d`; `nlocal_eq_cols_on_all` shows that the formula used by getrf_mgpu, geqrf_mgpu,
getrf2_mgpu, potrf3_mgpu, geqrf_m, getrf_m and potrf_mgpu_right is exactly that count, for every device. -/

def colsOn (nb ngpu d : Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => colsOn nb ngpu d n + (if (n / nb) % ngpu = d then 1 else 0)

theorem succ_div_cases (n nb : Nat) (hb : 0 < nb) :
    (n % nb + 1 < nb ∧ (n + 1) / nb = n / nb ∧ (n + 1) % nb = n % nb + 1) ∨
    (n % nb + 1 = nb ∧ (n + 1) / nb = n / nb + 1 ∧ (n + 1) % nb = 0) := by
  have hr : n % nb < nb := Nat.mod_lt _ hb
  have e : n = nb * (n / nb) + n % nb := (Nat.div_add_mod n nb).symm
  by_cases h : n % nb + 1 < nb
  · left; refine ⟨h, ?_, ?_⟩
    · have : (n + 1) = (n % nb + 1) + nb * (n / nb) := by omega
      rw [this, Nat.add_mul_div_left _ _ hb, Nat.div_eq_of_lt h]; omega
    · have : (n + 1) = (n % nb + 1) + nb * (n / nb) := by omega
      rw [this, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h]
  · right
    have h1 : n % nb + 1 = nb := by omega
    refine ⟨h1, ?_, ?_⟩
    · have : (n + 1) = 0 + nb * (n / nb + 1) := by rw [Nat.mul_succ]; omega
      rw [this, Nat.add_mul_div_left _ _ hb]; simp
    · have : (n + 1) = 0 + nb * (n / nb + 1) := by rw [Nat.mul_succ]; omega
      rw [this, Nat.add_mul_mod_self_left]; simp

theorem nlocal_succ (n nb ngpu d : Nat) (hb : 0 < nb) (hg : 0 < ngpu) (hd : d < ngpu) :
    nLocal (n + 1) nb ngpu d = nLocal n nb ngpu d + (if (n / nb) % ngpu = d then 1 else 0) := by
  unfold nLocal
  rcases succ_div_cases n nb hb with ⟨hlt, hq, hm⟩ | ⟨heq, hq, hm⟩
  · rw [hq, hm]
    (repeat' split) <;> omega
  · rw [hq, hm]
    rcases succ_div_cases (n / nb) ngpu hg with ⟨hlt2, hq2, hm2⟩ | ⟨heq2, hq2, hm2⟩
    · rw [hq2, hm2]
      (repeat' split) <;> omega
    · rw [hq2, hm2, Nat.succ_mul]
      (repeat' split) <;> omega

/-- **every device's local column count is right, for all n, nb > 0, ngpu > 0** -/
theorem nlocal_eq_cols_on_all (n nb ngpu d : Nat) (hb : 0 < nb) (hg : 0 < ngpu) (hd : d < ngpu) :
    nLocal n nb ngpu d = colsOn nb ngpu d n := by
  induction n with
  | zero => simp [nLocal, colsOn]
  | succ k ih => rw [nlocal_succ k nb ngpu d hb hg hd, ih]; rfl

/-- gehrd_m / ungqr_m write the same formula through `min_lblocks = (n/nb)/ngpu`, `last_dev = (n/nb)%ngpu` -/
def nLocalLblocks (n nb ngpu d : Nat) : Nat :=
  let minLb := (n / nb) / ngpu; let lastDev := (n / nb) % ngpu
  minLb * nb + (if d < lastDev then nb else if d = lastDev then n % nb else 0)
theorem lblocks_eq_cols_on_all (n nb ngpu d : Nat) (hb : 0 < nb) (hg : 0 < ngpu) (hd : d < ngpu) :
    nLocalLblocks n nb ngpu d = colsOn nb ngpu d n := by
  rw [← nlocal_eq_cols_on_all n nb ngpu d hb hg hd]; rfl

/-! ### Block ownership with an offset (zhemm_mgpu, zhetrd_mgpu, zhemv_mgpu)

zhemm_mgpu.cpp l.235–276 computes, for a sub-matrix that starts at block `B` (device `s = B % ngpu`), the number of
blocks before it on device `dev` (`myblkoffset = B/ngpu + (B%ngpu > dev ? 1 : 0)`) and the number of blocks among the
next `K` on device `dev` (`myblk = K/ngpu + (K%ngpu > devperm ? 1 : 0)`, `devperm = (dev - s + ngpu) % ngpu`).  Both
formulas are proved equal to the actual counts, for every ngpu, B, K, s, dev.  (Round-42 reading found no error.) -/

/-- blocks b < B owned by device `dev` (block b lives on device b % ngpu) -/
def ownedBefore (ngpu dev : Nat) : Nat → Nat
  | 0 => 0
  | B + 1 => ownedBefore ngpu dev B + (if B % ngpu = dev then 1 else 0)
/-- among K consecutive blocks starting at device s, those owned by `dev` -/
def ownedFrom (ngpu s dev : Nat) : Nat → Nat
  | 0 => 0
  | K + 1 => ownedFrom ngpu s dev K + (if (s + K) % ngpu = dev then 1 else 0)

/-- zhemm_mgpu / zhetrd_mgpu `myblkoffset = B/ngpu + (B%ngpu > dev ? 1 : 0)` -/
theorem owned_before_formula (ngpu dev B : Nat) (hg : 0 < ngpu) (hd : dev < ngpu) :
    ownedBefore ngpu dev B = B / ngpu + (if dev < B % ngpu then 1 else 0) := by
  induction B with
  | zero => simp [ownedBefore]
  | succ k ih =>
    simp only [ownedBefore]; rw [ih]
    rcases succ_div_cases k ngpu hg with ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩
    · rw [h2, h3]; (repeat' split) <;> omega
    · rw [h2, h3]; (repeat' split) <;> omega

theorem mod_lt_two (x g : Nat) (_hg : 0 < g) (h : x < 2 * g) : x % g = if x < g then x else x - g := by
  split
  · exact Nat.mod_eq_of_lt ‹_›
  · rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]

/-- `(s + K) % ngpu = dev` iff `K % ngpu = (dev + ngpu - s) % ngpu` (for s, dev < ngpu) -/
theorem shift_mod (ngpu s dev K : Nat) (hg : 0 < ngpu) (hs : s < ngpu) (hd : dev < ngpu) :
    ((s + K) % ngpu = dev) ↔ (K % ngpu = (dev + ngpu - s) % ngpu) := by
  have hk : K % ngpu < ngpu := Nat.mod_lt _ hg
  rw [Nat.add_mod, Nat.mod_eq_of_lt hs, mod_lt_two (s + K % ngpu) ngpu hg (by omega), mod_lt_two (dev + ngpu - s) ngpu hg (by omega)]
  constructor <;> intro h <;> (repeat' split at h) <;> (repeat' split) <;> omega

/-- zhemm_mgpu `myblk = K/ngpu + (K%ngpu > devperm ? 1 : 0)` with `devperm = (dev - s + ngpu) % ngpu` -/
theorem owned_from_formula (ngpu s dev K : Nat) (hg : 0 < ngpu) (hs : s < ngpu) (hd : dev < ngpu) :
    ownedFrom ngpu s dev K = K / ngpu + (if (dev + ngpu - s) % ngpu < K % ngpu then 1 else 0) := by
  have hp : (dev + ngpu - s) % ngpu < ngpu := Nat.mod_lt _ hg
  induction K with
  | zero => simp [ownedFrom]
  | succ k ih =>
    simp only [ownedFrom]; rw [ih]
    have hsh := shift_mod ngpu s dev k hg hs hd
    rcases succ_div_cases k ngpu hg with ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩
    · rw [h2, h3]
      by_cases hc : (s + k) % ngpu = dev
      · have := hsh.mp hc; simp only [hc, ite_true]; (repeat' split) <;> omega
      · have : ¬ k % ngpu = (dev + ngpu - s) % ngpu := fun h => hc (hsh.mpr h)
        simp only [hc, ite_false]; (repeat' split) <;> omega
    · rw [h2, h3]
      by_cases hc : (s + k) % ngpu = dev
      · have := hsh.mp hc; simp only [hc, ite_true]; (repeat' split) <;> omega
      · have : ¬ k % ngpu = (dev + ngpu - s) % ngpu := fun h => hc (hsh.mpr h)
        simp only [hc, ite_false]; (repeat' split) <;> omega

/-- zgetrf2_mgpu.cpp l.249–259: after panel `j` (on device `id = j % ngpu`), device `d` continues at local block
    `j_local2 = j/ngpu + 1` if `d = id`, else `j/ngpu + (d < id ? 1 : 0)` — exactly the number of blocks `≤ j` it owns -/
theorem getrf_mgpu_next_local_all (ngpu d j : Nat) (hg : 0 < ngpu) (hd : d < ngpu) :
    ownedBefore ngpu d (j + 1) = j / ngpu + (if d = j % ngpu then 1 else if d < j % ngpu then 1 else 0) := by
  rw [owned_before_formula ngpu d (j + 1) hg hd]
  rcases succ_div_cases j ngpu hg with ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩
  · rw [h2, h3]; (repeat' split) <;> omega
  · rw [h2, h3]; (repeat' split) <;> omega

/-- `zgetrf2_mgpu.cpp` l.204–206 and l.382–384: for panel `j`, `if (*info == 0 && iinfo > 0) *info = iinfo + j*nb`.
    `panels` lists the local LAPACK infos of panels 0, 1, …; the fold returns the reported info. -/
def getrfMgpuInfo (nb : Nat) : List Nat → Nat → Nat → Nat
  | [], _, info => info
  | i :: rest, j, info => getrfMgpuInfo nb rest (j + 1) (if info = 0 ∧ i > 0 then i + j * nb else info)

/-- the global index of the first failing column, computed directly -/
def firstFailure (nb : Nat) : List Nat → Nat → Nat
  | [], _ => 0
  | i :: rest, j => if i > 0 then i + j * nb else firstFailure nb rest (j + 1)

/-- **the multi-GPU LU reports the first failing column with its global offset, for any panel sequence** -/
theorem getrf_mgpu_first_global (nb : Nat) : ∀ (panels : List Nat) (j : Nat),
    getrfMgpuInfo nb panels j 0 = firstFailure nb panels j := by
  intro panels
  induction panels with
  | nil => intro j; rfl
  | cons i rest ih =>
    intro j
    by_cases h : i > 0
    · simp only [getrfMgpuInfo, firstFailure, h, and_true, ite_true]
      have keep : ∀ (r : List Nat) (k v : Nat), v ≠ 0 → getrfMgpuInfo nb r k v = v := by
        intro r; induction r with
        | nil => intro k v _; rfl
        | cons x xs ihx => intro k v hv; simp only [getrfMgpuInfo, hv, false_and, ite_false]; exact ihx _ _ hv
      exact keep rest (j + 1) (i + j * nb) (by omega)
    · simp only [getrfMgpuInfo, firstFailure, h, and_false, ite_false]; exact ih (j + 1)

/-- `zpotrf3_mgpu.cpp` l.297–300 / l.553–556: the diagonal block at global offset `j` is factored by LAPACK, and on
    failure `*info = *info + j` and the routine returns — the global index of the failing column -/
def potrfMgpuInfo (j localInfo : Nat) : Nat := if localInfo ≠ 0 then localInfo + j else 0
theorem potrf_mgpu_global (j l : Nat) (h : l ≠ 0) : potrfMgpuInfo j l = j + l := by
  simp [potrfMgpuInfo, h]; omega

/-- `magma_zhtodpo`, `magma_zdtohpo` (zpotrf3_mgpu.cpp l.782–911) and `magma_zhtodhe` (zhetrd_mgpu.cpp l.504–556),
    declared in the public header `magmablas_z.h`: none of them assigns `*info`, yet each ends with
    `return *info;`.  The result is whatever the caller's variable held.  (In-tree callers pass a scratch `iinfo`
    and ignore it, so MAGMA's own drivers are unaffected.)  Found by the 2026-11-26 sweep for functions that read
    `*info` before writing it (HANDOFF §29.32). -/
def htodInfo (infoIn : Int) : Int := infoIn
/-- general form: an external caller gets back its own uninitialised or stale value, whatever the copy did -/
theorem htod_returns_caller_value_all (infoIn : Int) : htodInfo infoIn = infoIn := rfl

/-! ### `zpotrf3_mgpu`: reuse distance of its two rotating buffers (round 50, HANDOFF §29.67)
The happens-before model `scripts/potrf_mgpu_hb.py` replays the host program; these are its arithmetic facts. -/

/-- `(k + i) % g = k % g` exactly when `g ∣ i` -/
theorem add_mod_eq_self_iff (g k i : Nat) (hg : 0 < g) : (k + i) % g = k % g ↔ i % g = 0 := by
  have h1 := Nat.mod_lt k hg
  have h2 := Nat.mod_lt i hg
  rw [Nat.add_mod, mod_lt_two (k % g + i % g) g hg (by omega)]
  split <;> omega

/-- device diagonal buffer `dlP(d,0,0,buf)`, `buf = (j/nb) % ngpu`: refilled exactly `ngpu` block steps later -/
theorem potrf_mgpu_dlp_reuse_at (g k : Nat) : (k + g) % g = k % g := Nat.add_mod_right k g
/-- … and never earlier -/
theorem potrf_mgpu_dlp_not_reused_before (g k i : Nat) (hg : 0 < g) (hi : 0 < i) (hlt : i < g) :
    (k + i) % g ≠ k % g := by
  rw [Ne, add_mod_eq_self_iff g k i hg, Nat.mod_eq_of_lt hlt]; omega

/-- host workspace `Aup(i,j)` (`Alo`): block slot `(j/nb) % h`.  Block rows `0 .. k+h-1` of block column `k+h`,
    copied to the host at step `k+h-1` (l.423 / l.687), land in the slot of block `k` and cover its diagonal rows -/
theorem potrf_mgpu_host_slot_alias (h k : Nat) (hh : 0 < h) : (k + h) % h = k % h ∧ k < k + h :=
  ⟨Nat.add_mod_right k h, by omega⟩

/-- Orderings that can cover the read of block `k`'s diagonal from the host slot (stream-1 copy on device `k % g`)
    before the slot is rewritten at step `k+h-1` by device `(k+h) % g`: a host `magma_queue_sync` of device
    `(k+i) % g`, stream 1, at step `k+i` (`1 ≤ i < h`), or the writer being the reading device (`i = h`, through
    its own stream-1 lookahead).  Cross-device events go only through the rows copy (`events[d][3]`). -/
def hostSlotCovered (g h k : Nat) : Prop := ∃ i, 1 ≤ i ∧ i ≤ h ∧ (k + i) % g = k % g

/-- covered iff `ngpu ≤ h`, for every step `k`; this is exactly the line the model and the GPU emulation give -/
theorem potrf_mgpu_host_slot_covered_iff (g h k : Nat) (hg : 0 < g) : hostSlotCovered g h k ↔ g ≤ h := by
  constructor
  · intro ⟨i, h1, h2, h3⟩
    rw [add_mod_eq_self_iff g k i hg] at h3
    have := Nat.le_of_dvd (by omega) (Nat.dvd_of_mod_eq_zero h3)
    omega
  · intro hgh
    exact ⟨g, hg, hgh, Nat.add_mod_right k g⟩

/-- `zpotrf_mgpu.cpp` fixes `h = 2` ("setting h = 1 produces failures for ngpus >= 4 … TODO"): uncovered for
    every step once `ngpu ≥ 3`; the commented-out alternative `h = ngpu` is the least covered value -/
theorem potrf_mgpu_h2_uncovered (g k : Nat) (hg : 3 ≤ g) : ¬ hostSlotCovered g 2 k := by
  rw [potrf_mgpu_host_slot_covered_iff g 2 k (by omega)]; omega
theorem potrf_mgpu_h_ngpu_covered (g k : Nat) (hg : 0 < g) : hostSlotCovered g g k :=
  (potrf_mgpu_host_slot_covered_iff g g k hg).2 (Nat.le_refl g)

/-! ### `zgetrf_m`: the main loop can use more devices than were set up (M258, round 52)
l.187–192 (before allocation) and l.248–252 (in the loop) both compute
`ngpu = if ngpu0 > ceildiv NB nb then ceildiv NB nb else ngpu0`, but with different `NB`: the first with the panel width
then in force (capped by `MAGMA_NGR_NB` when set), the second with the final width (recomputed without the cap, and
then `NB = n` in core or `ngpu·NB` out of core).  Buffers, queues and events exist only for the first value. -/

def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b
def getrfMNgpu (ngpu0 NB nb : Nat) : Nat := if ngpu0 > ceilDiv NB nb then ceilDiv NB nb else ngpu0

/-- general form: if the first computation reduced the device count and the final panel spans at least `ngpu0`
    blocks, the loop runs on all `ngpu0` devices, more than were set up -/
theorem getrf_m_loop_exceeds_alloc (ngpu0 NBa NBf nb : Nat) (hnb : 0 < nb)
    (hred : ngpu0 > ceilDiv NBa nb) (hwide : ngpu0 * nb ≤ NBf) :
    getrfMNgpu ngpu0 NBf nb = ngpu0 ∧ getrfMNgpu ngpu0 NBa nb < ngpu0 := by
  have h1 : ¬ ngpu0 > ceilDiv NBf nb := by
    unfold ceilDiv; intro h
    have : (NBf + nb - 1) / nb ≥ ngpu0 := by
      apply (Nat.le_div_iff_mul_le hnb).2; omega
    omega
  refine ⟨by simp [getrfMNgpu, h1], ?_⟩
  simp [getrfMNgpu, hred]

/-- the measured case: `MAGMA_NGR_NB = 64`, nb = 32 (sm_120), n = 256 in core: 2 devices set up, 3 used (crash) -/
theorem getrf_m_ngr64_instance : getrfMNgpu 3 64 32 = 2 ∧ getrfMNgpu 3 256 32 = 3 := by decide

/-- the fix used in round 52 (`min` with the allocated count) never exceeds it -/
theorem getrf_m_fix_within_alloc (ngpu0 NBa NBf nb : Nat) :
    min (getrfMNgpu ngpu0 NBf nb) (getrfMNgpu ngpu0 NBa nb) ≤ getrfMNgpu ngpu0 NBa nb := Nat.min_le_right _ _

/-! ### M259: `magma_[zcds]gehrd_m` + `magma_[zcds]unghr_m` (`[sd]orghr_m`) with ilo > 1

`gehrd_m` rounds `ilo - 1` down to a block boundary `ilo_r` and stores the T of the block whose first reflector is
the global index `s` at T column `s`.  `unghr_m` hands `ungqr_m` the submatrix from reflector `ilo - 1` on with T
unshifted, and `ungqr_m` reads the T of its sub-block starting at `c` from T column `c` while it needs the block whose
first reflector is `c + (ilo - 1)`.  The single-GPU `zgehrd` starts its blocks at `ilo - 1` and stores at column
`s - (ilo - 1)`, which is what `unghr` expects.  A reader (sub-problem offset `off`, T offset `tOff`) is consistent
with a storage map `stored : column ↦ first reflector` when every sub-block `k` finds its own block. -/

/-- first reflector of the block whose T `gehrd_m` stored at column `col` -/
def gehrdMStored (col : Nat) : Nat := col
/-- the same for the single-GPU `zgehrd` (blocks start at `ilo0`, stored at `s - ilo0`) -/
def gehrdStored (ilo0 col : Nat) : Nat := col + ilo0
/-- `ungqr_m` on the sub-problem starting at reflector `off`, reading T from column `tOff` on -/
def readerConsistent (stored : Nat → Nat) (nb off tOff : Nat) : Prop := ∀ k, stored (k * nb + tOff) = k * nb + off
/-- block starts of `gehrd_m`: from `ilo0` rounded down, every `nb` -/
def gehrdMStart (ilo0 nb s : Nat) : Prop := ∃ k, s = ilo0 / nb * nb + k * nb

/-- `unghr_m` as written (off = ilo - 1, T unshifted) reads the wrong T for every sub-block once ilo > 1 -/
theorem unghr_m_reads_wrong_T (nb ilo0 : Nat) (h : 0 < ilo0) (k : Nat) :
    gehrdMStored (k * nb + 0) ≠ k * nb + ilo0 := by simp [gehrdMStored]; omega

theorem unghr_m_inconsistent (nb ilo0 : Nat) (h : 0 < ilo0) :
    ¬ readerConsistent gehrdMStored nb ilo0 0 := fun hc => unghr_m_reads_wrong_T nb ilo0 h 0 (hc 0)

/-- with ilo = 1 the same reader is consistent (the only case MAGMA's tester runs) -/
theorem unghr_m_ilo1_consistent (nb : Nat) : readerConsistent gehrdMStored nb 0 0 := by
  intro k; simp [gehrdMStored]

/-- the single-GPU pair is consistent for every ilo -/
theorem unghr_single_consistent (nb ilo0 : Nat) : readerConsistent (gehrdStored ilo0) nb ilo0 0 := by
  intro k; simp [gehrdStored]

/-- the round-53 fix: start the sub-problem at the rounded `ilo_r` and shift T by `ilo_r` columns.  Every sub-block
    then finds its own block, each of its blocks is one `gehrd_m` wrote, and it only adds the `ilo0 % nb < nb`
    reflectors in `ilo_r .. ilo0 - 1`, whose tau is zero under the zgehrd precondition. -/
theorem unghr_m_fix_consistent (nb ilo0 : Nat) :
    readerConsistent gehrdMStored nb (ilo0 / nb * nb) (ilo0 / nb * nb) ∧
    (∀ k, gehrdMStart ilo0 nb (k * nb + ilo0 / nb * nb)) := by
  refine ⟨fun k => by simp [gehrdMStored], fun k => ⟨k, by omega⟩⟩

theorem unghr_m_fix_extra (nb ilo0 : Nat) (hnb : 0 < nb) :
    ilo0 / nb * nb ≤ ilo0 ∧ ilo0 - ilo0 / nb * nb < nb := by
  have h1 := Nat.div_add_mod ilo0 nb; have h2 := Nat.mod_lt ilo0 hnb
  rw [Nat.mul_comm] at h1; omega

/-! ### M260: `magma_[zcds]gehrd_m` reuses `work` in zgehd2 while the last zlahr2_m's `Y = work -> dY` copies are queued

After the blocked loop the code syncs only the devices of the `R` remainder blocks (block indices `b0, b0+1, …,
b0+R-1`, device `b % ngpu`), but the last `zlahr2_m` queued `magma_zsetmatrix_async(Y, dY)` on every device.  Then
`lapackf77_zgehd2` overwrites `work` (= Y).  A device is unsynced when no remainder block lives on it. -/

/-- the devices synced by the remainder loop -/
def gehrdSynced (ngpu b0 R dev : Nat) : Prop := ∃ k, k < R ∧ (b0 + k) % ngpu = dev

/-- the number of remainder blocks with ilo = 1, ihi = n: `i_end` is the first multiple of nb with
    `i_end ≥ n - 1 - nb`, so `n - i_end ∈ [1, nb + 1]` and at most 2 blocks remain -/
theorem gehrd_remainder_le_two (n nb iend : Nat) (hnb : 0 < nb) (h1 : n - 1 - nb ≤ iend) (h2 : iend < n) :
    ceilDiv (n - iend) nb ≤ 2 := by
  unfold ceilDiv
  have : n - iend + nb - 1 < 3 * nb := by omega
  have := (Nat.div_lt_iff_lt_mul hnb).2 this
  omega

/-- whenever fewer blocks remain than there are devices, some device is never synced before zgehd2 -/
theorem gehrd_m_unsynced_device (ngpu b0 R : Nat) (h : R < ngpu) :
    ∃ dev, dev < ngpu ∧ ¬ gehrdSynced ngpu b0 R dev := by
  -- the R synced devices are the images of k < R; a device outside that image exists by counting
  refine ⟨(b0 + R) % ngpu, Nat.mod_lt _ (by omega), ?_⟩
  rintro ⟨k, hk, hkd⟩
  -- (b0 + k) ≡ (b0 + R) mod ngpu with k < R < ngpu forces k = R
  have h1 : (b0 + k) % ngpu = (b0 + R) % ngpu := hkd
  have h2 : (R - k) % ngpu = 0 := by
    have := Nat.sub_mod_eq_zero_of_mod_eq h1.symm
    rwa [show b0 + R - (b0 + k) = R - k by omega] at this
  have h3 : (R - k) % ngpu = R - k := Nat.mod_eq_of_lt (by omega)
  omega

/-- the device that matters is the panel device (block `b0 - 1`, written as `b0 + ngpu - 1` mod ngpu): zlahru_m uses
    dY only to update a device's own columns ≥ k, i.e. the panel block and the remainder blocks.  It is unsynced
    exactly when fewer blocks remain than there are devices (R ≤ 2 by `gehrd_remainder_le_two`). -/
theorem gehrd_m_panel_unsynced (ngpu b0 R : Nat) (h : R < ngpu) :
    ¬ gehrdSynced ngpu b0 R ((b0 + (ngpu - 1)) % ngpu) := by
  rintro ⟨k, hk, hkd⟩
  have h2 := Nat.sub_mod_eq_zero_of_mod_eq hkd.symm
  rw [show b0 + (ngpu - 1) - (b0 + k) = ngpu - 1 - k by omega] at h2
  have h3 : (ngpu - 1 - k) % ngpu = ngpu - 1 - k := Nat.mod_eq_of_lt (by omega)
  omega

theorem gehrd_m_panel_synced_when_full (ngpu b0 : Nat) (h : 0 < ngpu) :
    gehrdSynced ngpu b0 ngpu ((b0 + (ngpu - 1)) % ngpu) := ⟨ngpu - 1, by omega, rfl⟩

/-- the measured instance: n = 96, nb = 32, ngpu = 2: one remainder block (block 2, on device 0), device 1 unsynced -/
theorem gehrd_m_n96_instance : ¬ gehrdSynced 2 2 1 1 := by
  rintro ⟨k, hk, hkd⟩; have : k = 0 := by omega
  subst this; simp at hkd

/-- the round-52 fix syncs every device -/
theorem gehrd_m_fix_syncs_all (ngpu dev : Nat) (h : dev < ngpu) : gehrdSynced ngpu 0 ngpu dev :=
  ⟨dev, h, by rw [Nat.zero_add]; exact Nat.mod_eq_of_lt h⟩

end MagmaMultiGpu

#print axioms MagmaMultiGpu.distribution_covers
#print axioms MagmaMultiGpu.getrf_mgpu_first_global
#print axioms MagmaMultiGpu.potrf_mgpu_global
#print axioms MagmaMultiGpu.htod_returns_caller_value_all
#print axioms MagmaMultiGpu.nlocal_eq_cols_on_all
#print axioms MagmaMultiGpu.lblocks_eq_cols_on_all
#print axioms MagmaMultiGpu.owned_before_formula
#print axioms MagmaMultiGpu.owned_from_formula
#print axioms MagmaMultiGpu.getrf_mgpu_next_local_all
#print axioms MagmaMultiGpu.add_mod_eq_self_iff
#print axioms MagmaMultiGpu.potrf_mgpu_dlp_reuse_at
#print axioms MagmaMultiGpu.potrf_mgpu_dlp_not_reused_before
#print axioms MagmaMultiGpu.potrf_mgpu_host_slot_alias
#print axioms MagmaMultiGpu.potrf_mgpu_host_slot_covered_iff
#print axioms MagmaMultiGpu.potrf_mgpu_h2_uncovered
#print axioms MagmaMultiGpu.potrf_mgpu_h_ngpu_covered
#print axioms MagmaMultiGpu.getrf_m_loop_exceeds_alloc
#print axioms MagmaMultiGpu.getrf_m_ngr64_instance
#print axioms MagmaMultiGpu.getrf_m_fix_within_alloc
#print axioms MagmaMultiGpu.unghr_m_reads_wrong_T
#print axioms MagmaMultiGpu.unghr_m_inconsistent
#print axioms MagmaMultiGpu.unghr_m_ilo1_consistent
#print axioms MagmaMultiGpu.unghr_single_consistent
#print axioms MagmaMultiGpu.unghr_m_fix_consistent
#print axioms MagmaMultiGpu.unghr_m_fix_extra
#print axioms MagmaMultiGpu.gehrd_remainder_le_two
#print axioms MagmaMultiGpu.gehrd_m_unsynced_device
#print axioms MagmaMultiGpu.gehrd_m_panel_unsynced
#print axioms MagmaMultiGpu.gehrd_m_panel_synced_when_full
#print axioms MagmaMultiGpu.gehrd_m_n96_instance
#print axioms MagmaMultiGpu.gehrd_m_fix_syncs_all

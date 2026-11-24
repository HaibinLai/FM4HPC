"""Cross-check of the Lean slice lemma (lean/OptimiKahan/OzSlice.lean) against real binary64 arithmetic (numpy float64).
(1) slice_lemma: piece=(r+sigma)-sigma with sigma=0.75*2^(rho+tau), |r|<=2^tau:  piece multiple of q=2^(tau+rho-53);
    |r-piece|<=q/2; residual r-piece computed in double is exact; |n|=|piece/q| <= 2^(53-rho); FP16-representable when rho>=42.
(2) K=1 with rho=41 (s=12): 12-bit slice integers exist that are not FP16 numbers (n=2049).
(3) IsMaxS: library rho = ceil(53-(24-log2(max(K,2)))/2) (double arithmetic) equals 53 - s, s = max{s: K*4^s <= 2^24}.
"""
import math
from fractions import Fraction as F
import numpy as np

def check_slice(rho, tau, r):
    sigma = 0.75 * math.ldexp(1.0, rho + tau); q = math.ldexp(1.0, tau + rho - 53)
    piece = (r + sigma) - sigma; res = r - piece
    ok = {}
    ok['multiple_of_q'] = (F(piece) / F(q)).denominator == 1
    ok['half_quantum'] = abs(F(r) - F(piece)) <= F(q) / 2
    ok['residual_exact'] = F(res) == F(r) - F(piece)
    n = F(piece) / F(q)
    ok['n_bound'] = abs(n) <= 2 ** (53 - rho)
    ok['fp16'] = (rho < 42) or (np.float16(float(n) * 2.0 ** -(53 - rho)) .astype(np.float64) == float(n) * 2.0 ** -(53 - rho))
    return ok

rng = np.random.default_rng(5)
tot = 0; bad = {}
for rho in (42, 43, 44, 45, 46, 48, 49):
    for tau in (-5, 0, 3, 17, 30):
        q = math.ldexp(1.0, tau + rho - 53); B = math.ldexp(1.0, tau)
        samples = [B, -B, 0.0, q / 2, -q / 2, 3 * q / 2, q, B - q, math.nextafter(B, 0), math.nextafter(-B, 0), q * 0.4999999, q * 0.5000001]
        samples += list(rng.uniform(-B, B, 4000)) + list(np.ldexp(rng.uniform(-1, 1, 2000), rng.integers(tau - 60, tau + 1, 2000)))
        samples += [float((k + 0.5) * q) for k in rng.integers(-2 ** (53 - rho), 2 ** (53 - rho), 300) if abs((k + 0.5) * q) <= B]   # exact ties
        for r in samples:
            if abs(r) > B: continue
            tot += 1
            for k, v in check_slice(rho, tau, float(r)).items():
                if not v: bad[(k, rho)] = bad.get((k, rho), 0) + 1
print(f'(1) slice lemma: {tot} samples (rho 42..49, tau -5..30, ties and boundaries); failures: {bad if bad else "none"}')

# (2) K=1, rho=41 -> s=12: odd 12-bit slice integers
rho = 41; tau = 0; q = math.ldexp(1.0, tau + rho - 53); r = 2049 * q
piece = (r + 0.75 * math.ldexp(1.0, rho + tau)) - 0.75 * math.ldexp(1.0, rho + tau); n = piece / q
scaled = n * 2.0 ** -12
print(f'(2) K=1 (rho=41, s=12): n={n:.0f}, |n|<=2^12 ok={abs(n) <= 2**12}, scaled slice FP16-exact: {np.float16(scaled).astype(np.float64) == scaled}  (expected False)')

# (3) rho formula vs IsMaxS
def smax(K):
    s = 0
    while K * 4 ** (s + 1) <= 2 ** 24: s += 1
    return s
def rho_lib(K):
    dim = 2 if K == 1 else K
    return math.ceil(53 - (24 - math.log2(1.0 * dim)) / 2)      # getRho<double,half> (double arithmetic)
mism = []
Ks = list(range(1, 2 ** 20 + 1)) + [m for e in range(20, 25) for m in (2 ** e - 1, 2 ** e, 2 ** e + 1)]
for K in Ks:
    if K * 4 ** 0 > 2 ** 24: continue
    kk = max(K, 2)
    if rho_lib(K) != 53 - smax(kk): mism.append(K)
print(f'(3) rho formula vs 53 - max{{s: K*4^s <= 2^24}} for {len(Ks)} values of K (K=1 treated as 2): mismatches {mism[:10] if mism else "none"}')

# (4) relaxed hypothesis |r| < 2^tau + q/2 (covers tau too small by 1 near powers of two) -- and the boundary |r| = 2^tau + q/2
bad4 = {}; tot4 = 0
for rho in (42, 43, 44, 46, 49):
    for tau in (-3, 0, 5, 20):
        q = math.ldexp(1.0, tau + rho - 53); B = math.ldexp(1.0, tau)
        for eps in (2.0 ** -45, 2.0 ** -40, 2.0 ** -30, 2.0 ** -20):
            r = B * (1 + eps)                                  # |r| slightly above 2^tau: the tau-too-small situation
            if r - B >= q / 2: continue
            for rr in (r, -r):
                tot4 += 1
                for k, v in check_slice(rho, tau, rr).items():
                    if not v: bad4[(k, rho)] = bad4.get((k, rho), 0) + 1
        # just below the bound
        for rr in (B + q / 2 * 0.999, -(B + q / 2 * 0.999), math.nextafter(B + q / 2, 0)):
            tot4 += 1
            for k, v in check_slice(rho, tau, rr).items():
                if not v: bad4[(k, rho, 'below-bound')] = bad4.get((k, rho, 'below-bound'), 0) + 1
print(f'(4) relaxed bound |r| < 2^tau + q/2: {tot4} samples; failures: {bad4 if bad4 else "none"}')
# boundary |r| = 2^tau + q/2 is a TIE between n = 2^s (even) and n = 2^s + 1 (odd)
even_ok = 0; away_viol = 0; total_at_bound = 0
for rho in (42, 43, 44, 46, 49):
    for tau in (-3, 0, 5, 20):
        q = math.ldexp(1.0, tau + rho - 53); B = math.ldexp(1.0, tau); r = B + q / 2; total_at_bound += 1
        even_ok += check_slice(rho, tau, r)['n_bound']                                    # IEEE ties-to-even (what the GPU does)
        n_away = math.floor(F(r) / F(q) + F(1, 2))                                         # ties-away-from-zero, exact arithmetic
        away_viol += n_away > 2 ** (53 - rho)
print(f'    boundary |r| = 2^tau + q/2: ties-to-even keeps |n| <= 2^(53-rho) in {even_ok}/{total_at_bound} cases; '
      f'a ties-away rule gives n = 2^s + 1 > 2^s in {away_viol}/{total_at_bound}. The Lean lemma allows ANY tie-breaking, hence its strict bound.')

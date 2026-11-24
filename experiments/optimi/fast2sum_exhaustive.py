"""Exhaustive check of Fast2Sum exactness in the integer lattice with p significant bits (= binary FP with subnormals, no overflow).
R_p = {z in Z : |z| = m*2^e, m < 2^p}.  rd = round-to-nearest, ties-to-even-significand (mimics IEEE RNE).
Step under test (optimi's renormalisation):  s = rd(a+b);  d = rd(a - s);  e = rd(b + d).   Claim: s+e == a+b  (e is the exact rounding error).
Usage: python fast2sum_exhaustive.py [--pmax 6] [--range 400]
"""
import argparse

def make_rd(p):
    def rd(z):
        if z == 0: return 0
        s = -1 if z < 0 else 1; a = abs(z); k = a.bit_length() - p
        if k <= 0: return z
        q = 1 << k; base = (a // q) * q; r = a - base
        if 2 * r > q or (2 * r == q and ((base // q) & 1)): base += q
        return s * base
    def rep(z):
        if z == 0: return True
        a = abs(z); a >>= (a & -a).bit_length() - 1
        return a.bit_length() <= p
    return rd, rep

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pmax', type=int, default=6); ap.add_argument('--range', type=int, default=400); a = ap.parse_args()
    for p in range(2, a.pmax + 1):
        rd, rep = make_rd(p); R = [z for z in range(-a.range, a.range + 1) if rep(z)]
        tot = {'|b|<=|a|': [0, 0], '|b|>|a|': [0, 0]}; ex = []
        for x in R:
            for y in R:
                s = rd(x + y); d = rd(x - s); e = rd(y + d); exact = (s + e == x + y)
                k = '|b|<=|a|' if abs(y) <= abs(x) else '|b|>|a|'; tot[k][0] += 1; tot[k][1] += exact
                if k == '|b|<=|a|' and not exact and len(ex) < 3: ex.append((x, y, s, d, e))
        print(f'p={p}: representable values in range {len(R)}; ' + '; '.join(f'{k}: {v[1]}/{v[0]} exact' for k, v in tot.items()) + (f'  COUNTEREXAMPLES {ex}' if ex else ''))
        # also: sub-step exactness a - s (Sterbenz-type) when |b|<=|a|
        bad = sum(1 for x in R for y in R if abs(y) <= abs(x) and rd(x - rd(x + y)) != x - rd(x + y))
        print(f'      a - s computed exactly (|b|<=|a|): violations {bad}')

if __name__ == '__main__': main()

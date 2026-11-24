"""Cross-check of the Lean theorem fast2sum_bf16 (Fast2Sum exact for ANY nearest rounding, ties broken arbitrarily).
Python side: exhaustive over the p-bit integer lattice with (i) ties-to-even, (ii) ties-away, (iii) ties-toward-zero, (iv) random ties.
Negative control: truncation (round toward zero) is NOT nearest and should break Fast2Sum.
"""
import random
import sys

def rep(p, z):
    if z == 0: return True
    a = abs(z); a >>= (a & -a).bit_length() - 1
    return a.bit_length() <= p

def make_rd(p, mode, rng):
    def rd(z):
        if z == 0: return 0
        sg = -1 if z < 0 else 1; a = abs(z); k = a.bit_length() - p
        if k <= 0: return z
        q = 1 << k; base = (a // q) * q; r = a - base
        if mode == 'trunc': return sg * base
        if 2 * r > q: up = True
        elif 2 * r < q: up = False
        else:
            up = {'even': bool((base // q) & 1), 'away': True, 'zero': False, 'random': rng.random() < 0.5}[mode]
        return sg * (base + q if up else base)
    return rd

def check(p, mode, rng, R):
    rd = make_rd(p, mode, rng) if mode != 'random' else None
    bad = tot = 0
    for a in R:
        for b in R:
            if abs(b) > abs(a): continue
            rdf = rd if rd else make_rd(p, 'random', rng)        # fresh random tie choices per evaluation of the sequence
            s = rdf(a + b); d = rdf(a - s); e = rdf(b + d); tot += 1; bad += (e != (a + b) - s)
    return tot, bad

rng = random.Random(1)
for p in (3, 4, 5, 6):
    R = [z for z in range(-300, 301) if rep(p, z)]
    out = []
    for mode in ('even', 'away', 'zero', 'random', 'trunc'):
        tot, bad = check(p, mode, rng, R); out.append(f'{mode}: {tot - bad}/{tot}')
    print(f'p={p}: ' + '  '.join(out))

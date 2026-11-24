"""Numerical properties of optimi's Kahan-compensated bf16 update, using the validated model (kahan_model.py).
Usage: python kahan_experiments.py [--steps 20000]
E1  Renormalisation exactness: after  w1 = rn(w+c1), c2 = rn(c1 + rn(w - w1))  is  w1 + c2 == w + c1  EXACTLY (Fast2Sum)?  When does it fail?
E2  Conservation law: V_k := w_k + c_k satisfies  V_{k+1} = V_k + u_k - delta_k  with delta_k = (c_k+u_k) - rn(c_k+u_k) (exact Fractions).
E3  Long-run drift vs fp64: naive bf16 accumulation (stagnates) vs Kahan bf16, constant and random small updates.
All values are bf16 held in float32; exact arithmetic via fractions.Fraction.
"""
import argparse
from fractions import Fraction as F
import numpy as np
import kahan_model as K

EPS_U = F(1, 256)   # bf16 unit roundoff 2^-8 (8 significant bits, round-to-nearest)

def fr(x): return F(float(x))

def kahan_add(w, c, u):
    """The summation core of the optimi step for a given update u (float32/bf16 value): c<-rn(c+u); w<-rn(w+c); c<-rn(c+rn(t-w))."""
    c1 = K.bf16_add(c, u); w1 = K.bf16_add(w, c1); d = K.bf16_sub(w, w1); c2 = K.bf16_add(c1, d)
    return w1, c2, c1

def e1(n=200000, seed=1):
    rng = np.random.default_rng(seed); ok = fail = 0; cond_fail = {'|c1|>|w|': 0, 'w==0': 0, 'other': 0}; ex = []
    for _ in range(n):
        w = K.to_bf16(np.float32(rng.standard_normal() * 10 ** rng.uniform(-3, 3)))
        c = K.to_bf16(np.float32(rng.standard_normal() * abs(w) * 10 ** rng.uniform(-6, -0.5)))
        u = K.to_bf16(np.float32(rng.standard_normal() * abs(w) * 10 ** rng.uniform(-7, -0.5)))
        w1, c2, c1 = kahan_add(w, c, u)
        if fr(w1) + fr(c2) == fr(w) + fr(c1): ok += 1
        else:
            fail += 1; k = '|c1|>|w|' if abs(fr(c1)) > abs(fr(w)) else ('w==0' if w == 0 else 'other'); cond_fail[k] += 1
            if k == 'other' and len(ex) < 3: ex.append((float(w), float(c), float(u), float(c1), float(w1), float(c2)))
    print(f'E1 exact renormalisation  w1+c2 == w+c1:  {ok}/{n} exact, {fail} not exact; failure kinds {cond_fail}')
    for e in ex: print('   example of "other":', e)

def e2(steps=20000, seed=2):
    rng = np.random.default_rng(seed); w = K.to_bf16(np.float32(1.0)); c = K.to_bf16(np.float32(0.0))
    V = fr(w) + fr(c); bad = 0; worst = F(0); sumbound = F(0); sumdelta = F(0); nonzero = 0
    for k in range(steps):
        u = K.to_bf16(np.float32(rng.standard_normal() * 10 ** rng.uniform(-6, -3)))
        c1 = K.bf16_add(c, u); delta = fr(c) + fr(u) - fr(c1)
        w1, c2, _ = kahan_add(w, c, u)
        V1 = fr(w1) + fr(c2)
        if V1 != V + fr(u) - delta: bad += 1
        bound = EPS_U * abs(fr(c) + fr(u))
        if abs(delta) > bound: nonzero += 1                                # |delta| <= 2^-8 |c+u| (nearest rounding, ignoring double rounding)
        sumdelta += abs(delta); sumbound += bound; w, c, V = w1, c2, V1
    print(f'E2 conservation V_(k+1) = V_k + u_k - delta_k exact: violations {bad}/{steps}; steps with |delta| > 2^-8|c+u|: {nonzero}')
    print(f'   sum|delta| = {float(sumdelta):.3e}   sum of 2^-8|c+u| = {float(sumbound):.3e}')

def e3(steps=20000):
    print('E3 long-run accumulation of small updates (w0 = 1.0): error of represented value vs exact sum of the same updates')
    print('   scenario                         exact sum        naive bf16 w     Kahan w+c        naive err      Kahan err')
    for name, gen in (('const u=-1e-4', lambda r, k: -1e-4), ('const u=+3e-5', lambda r, k: 3e-5), ('random N(0,1e-4)', lambda r, k: r.standard_normal() * 1e-4), ('random drift 1e-4*(N+0.5)', lambda r, k: 1e-4 * (r.standard_normal() + 0.5))):
        r = np.random.default_rng(3); wn = K.to_bf16(np.float32(1.0)); w = wn; c = K.to_bf16(np.float32(0.0)); exact = F(1)
        for k in range(steps):
            u = K.to_bf16(np.float32(gen(r, k))); exact += fr(u)
            wn = K.bf16_add(wn, u)                                         # naive: w <- rn(w + u)
            w, c, _ = kahan_add(w, c, u)
        kv = fr(w) + fr(c); print(f'   {name:30} {float(exact):15.8f} {float(wn):15.8f} {float(kv):15.8f}   {float(abs(fr(wn)-exact)):.3e}   {float(abs(kv-exact)):.3e}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=20000); a = ap.parse_args()
    e1(); e2(a.steps); e3(a.steps)

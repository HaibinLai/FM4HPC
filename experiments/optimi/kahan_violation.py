"""What happens when optimi's Kahan precondition |c1| <= |w| is violated?  (c1 = rn(c+u); w<-rn(w+c1); c<-rn(c1+rn(w-w_new)))
Measures, in exact arithmetic, the invariant defect  |(w1+c2) - (w+c1)| relative to |c1|, over regimes of |c1|/|w|.
"""
from fractions import Fraction as F
import numpy as np
import kahan_model as K
fr = lambda x: F(float(x))
rng = np.random.default_rng(7)
print('regime |c1|/|w|        samples  exact   max defect/|c1|   mean defect/|c1|   defect/|w1| max')
for lo, hi in ((-8, -2), (-2, -0.3), (-0.3, 0), (0, 0.5), (0.5, 2), (2, 6)):
    n = 20000; exact = 0; mx = F(0); sm = F(0); mw = F(0)
    for _ in range(n):
        w = K.to_bf16(np.float32(rng.standard_normal() * 10 ** rng.uniform(-3, 3))); 
        if w == 0: continue
        c1 = K.to_bf16(np.float32(float(w) * 10 ** rng.uniform(lo, hi) * rng.choice([-1, 1])))
        w1 = K.bf16_add(w, c1); d = K.bf16_sub(w, w1); c2 = K.bf16_add(c1, d)
        defect = abs(fr(w1) + fr(c2) - fr(w) - fr(c1))
        if defect == 0: exact += 1
        else:
            r = defect / abs(fr(c1)); mx = max(mx, r); sm += r
            if w1 != 0: mw = max(mw, defect / abs(fr(w1)))
    print(f'10^[{lo:5},{hi:5}]      {n:7d}  {exact:6d}   {float(mx):.3e}         {float(sm / n):.3e}           {float(mw):.3e}')

"""Cross-check of the Lean theorems eta_bound_bf16 / kahan_error_bf16_unconditional with the validated bf16 model.
(1) Per-step defect:  251*|eta| <= |c1|  must hold for ALL states (exact Fractions); report the max of 251*|eta|/|c1| (<=1 expected).
(2) Multi-step:  125*|V_N - (V_0 + sum u)| <= sum |c_k+u_k|  on trajectories where |c1|>|w| happens often.
"""
from fractions import Fraction as F
import numpy as np
import kahan_model as K
fr = lambda x: F(float(x))
rng = np.random.default_rng(11)

# (1) per-step: sample (w, c, u) across regimes including heavy violations
worst = F(0); n = 0; viol = 0
for _ in range(400000):
    w = K.to_bf16(np.float32(rng.standard_normal() * 10 ** rng.uniform(-4, 2)))
    c = K.to_bf16(np.float32(rng.standard_normal() * 10 ** rng.uniform(-6, 2)))
    u = K.to_bf16(np.float32(rng.standard_normal() * 10 ** rng.uniform(-6, 2)))
    c1 = K.bf16_add(c, u)
    if c1 == 0: continue
    w1 = K.bf16_add(w, c1); d = K.bf16_sub(w, w1); c2 = K.bf16_add(c1, d)
    eta = abs(fr(w1) + fr(c2) - fr(w) - fr(c1)); r = 251 * eta / abs(fr(c1)); n += 1
    viol += abs(fr(c1)) > abs(fr(w)); worst = max(worst, r)
print(f'(1) per-step: {n} states ({viol} with |c1|>|w|); max 251*|eta|/|c1| = {float(worst):.4f}  (Lean bound requires <= 1)')

# (2) multi-step on violation-heavy trajectories
bad = 0; worst2 = F(0); trials = 0
for t in range(300):
    w = K.to_bf16(np.float32(rng.standard_normal() * 10 ** rng.uniform(-4, 0))); c = K.to_bf16(np.float32(0.0))
    V0 = fr(w) + fr(c); sum_u = F(0); S = F(0); viol_steps = 0
    for k in range(400):
        u = K.to_bf16(np.float32(rng.standard_normal() * 10 ** rng.uniform(-3, 0.5)))
        c1 = K.bf16_add(c, u); S += abs(fr(c) + fr(u)); sum_u += fr(u)
        w1 = K.bf16_add(w, c1); d = K.bf16_sub(w, w1); c2 = K.bf16_add(c1, d)
        viol_steps += abs(fr(c1)) > abs(fr(w)); w, c = w1, c2
    err = abs(fr(w) + fr(c) - (V0 + sum_u)); trials += 1
    if S > 0: worst2 = max(worst2, 125 * err / S)
    if 125 * err > S: bad += 1
print(f'(2) multi-step: {trials} trajectories of 400 steps; violations of 125*err <= S: {bad}; max 125*err/S = {float(worst2):.4f}  (Lean bound requires <= 1)')

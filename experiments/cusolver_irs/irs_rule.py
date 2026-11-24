"""Which stopping rule does cuSOLVER IRS actually apply?  Per iteration we have RNRM_i (residual history).  Compare the
stopping iteration with the first i satisfying each candidate rule (inf-norms, EPS = 2^-53):
  doc : RNRM < sqrt(n) * XNRM * ANRM * EPS      (documented; XNRM from the returned x, constant to ~1e-10)
  bnrm: RNRM < sqrt(n) * BNRM * ANRM * EPS      (hypothesis: ||b|| in place of ||x||)
  rel : RNRM < sqrt(n) * BNRM * EPS             (hypothesis: relative residual)"""
import math
import numpy as np
import irs_probe as P

rng = np.random.default_rng(53)
n = 2048
A0 = rng.choice([-1.0, 1.0], (n, n)) * rng.uniform(0.5, 1.0, (n, n))
xt = rng.uniform(-1, 1, n)
b0 = A0 @ xt
print(f"cuSOLVER {P.ver.value}; n={n}", flush=True)
for low, rf in (("32F", "CL"), ("32F", "GM"), ("16F", "CL"), ("16F", "GM"), ("16BF", "GM")):
    for k in (-20, -10, 0, 10, 20):
        s = 2.0 ** k
        A, b = A0 * s, b0 * s
        r = P.irs(A, b, low, rf, maxiters=50)
        an, xn, bn = np.abs(A).sum(axis=1).max(), np.abs(r["x"]).max(), np.abs(b).max()
        rules = {"doc": math.sqrt(n) * xn * an * P.EPS, "bnrm": math.sqrt(n) * bn * an * P.EPS, "rel": math.sqrt(n) * bn * P.EPS}
        hist = [h[1] for h in r["hist"]]
        first = {nm: next((i for i, v in enumerate(hist) if v < t), None) for nm, t in rules.items()}
        stop = len(hist) - 1
        match = [nm for nm, f in first.items() if f == stop] if r["niters"] >= 0 else []
        print(f"{low}/{rf} k={k:4d} niters={r['niters']:4d} hist_rows={len(hist)} first-satisfied: " +
              " ".join(f"{nm}={first[nm]}" for nm in rules) + f"  -> consistent with {match}  last RNRM/doc_thr="
              f"{hist[-1] / rules['doc'] if hist else float('nan'):.3g}", flush=True)

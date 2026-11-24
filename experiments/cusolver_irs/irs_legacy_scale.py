"""Legacy one-call API (cusolverDnDHgesv / DBgesv / DSgesv / DXgesv, default settings, fallback on) on A = 2^k * A0,
b = 2^k * b0 (A0 entries +-[0.5,1), x_true uniform): does a 'converged' (niters >= 0) result come back wrong?"""
import ctypes as C
import numpy as np
import irs_probe as P
import irs_scale as S

for nm in ("cusolverDnDSgesv", "cusolverDnDXgesv"):
    getattr(P.cusolver, nm + "_bufferSize").argtypes = getattr(P.cusolver, "cusolverDnDHgesv_bufferSize").argtypes
    getattr(P.cusolver, nm).argtypes = getattr(P.cusolver, "cusolverDnDHgesv").argtypes

rng = np.random.default_rng(41)
print(f"cuSOLVER {P.ver.value}; legacy API, default settings", flush=True)
for n in (1024, 4096):
    S.n = n
    A0 = rng.choice([-1.0, 1.0], (n, n)) * rng.uniform(0.5, 1.0, (n, n))
    xt = rng.uniform(-1, 1, n)
    b0 = A0 @ xt
    for nm in ("cusolverDnDHgesv", "cusolverDnDBgesv", "cusolverDnDSgesv", "cusolverDnDXgesv"):
        for k in (-100, -60, -40, -20, 0, 10, 20, 40, 60, 100):
            if nm == "cusolverDnDHgesv" and abs(k) > 12:
                continue
            s = 2.0 ** k
            r = S.legacy(nm, A0 * s, b0 * s)
            e, q = P.eta(A0 * s, r["x"], b0 * s)
            fe = np.abs(r["x"] - xt).max() / np.abs(xt).max()
            bad = "   <-- converged but WRONG" if r["niters"] >= 0 and fe > 1e-6 else ("   <-- converged, eta > thr" if r["niters"] >= 0 and not q < 1 else "")
            print(f"n={n} {nm} k={k:4d} niters={r['niters']:4d} info={r['info']} eta/thr={q:9.3g} fwd_err={fe:9.3g}{bad}", flush=True)

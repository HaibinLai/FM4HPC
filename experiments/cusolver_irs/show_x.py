"""Print a few solution components next to x_true, to show what 'wrong' looks like (n=1024, seed 0, as repro_dbgesv)."""
import numpy as np
import irs_probe as P
import irs_scale as S
for nm in ("cusolverDnDXgesv", "cusolverDnDSgesv"):
    P.cusolver.__getattr__(nm + "_bufferSize").argtypes = P.cusolver.cusolverDnDHgesv_bufferSize.argtypes
    P.cusolver.__getattr__(nm).argtypes = P.cusolver.cusolverDnDHgesv.argtypes
rng = np.random.default_rng(0)
n = 1024
A0 = rng.choice([-1.0, 1.0], (n, n)) * rng.uniform(0.5, 1.0, (n, n))
xt = rng.uniform(-1, 1, n)
b0 = A0 @ xt
xd = np.linalg.solve(A0, b0)
print("x_true[0:5]          ", np.array2string(xt[:5], precision=10))
print("numpy FP64 solve     ", np.array2string(xd[:5], precision=10), f" max rel err {np.abs(xd - xt).max() / np.abs(xt).max():.2g}")
for nm, k in (("cusolverDnDBgesv", 0), ("cusolverDnDBgesv", 40), ("cusolverDnDXgesv", 40), ("cusolverDnDSgesv", 40), ("cusolverDnDSgesv", -60)):
    r = S.legacy(nm, A0 * 2.0 ** k, b0 * 2.0 ** k)
    x = r["x"]
    err = np.abs(x - xt)
    i = int(err.argmax())
    digits = -np.log10(np.maximum(err / np.abs(xt).max(), 1e-17))
    print(f"{nm} 2^{k:<3d} iter={r['niters']:3d}  x[0:5] = {np.array2string(x[:5], precision=10)}")
    print(f"      worst component i={i}: x={x[i]:.10f} true={xt[i]:.10f}   correct digits: min {digits.min():.1f}, median {np.median(digits):.1f}")

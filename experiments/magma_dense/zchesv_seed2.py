import numpy as np, importlib.util, sys, io, contextlib
spec = importlib.util.spec_from_file_location("p", "zchesv_port.py"); p = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(p)
d = 2.0**-24; g = np.random.default_rng(2); n = 16
c = g.uniform(-1, 1, n - 1); Q, _ = np.linalg.qr(g.standard_normal((n - 1, n - 1)))
M = Q @ np.diag(g.uniform(1, 2, n - 1)) @ Q.T
A = np.zeros((n, n)); A[0, 0] = d; A[0, 1:] = c; A[1:, 0] = c; A[1:, 1:] = M
xt = g.uniform(-1, 1, n); b = A @ xt
# re-run the loop and print norms per iteration
L, D, info = p.ldlt_nopiv32(A); cte = np.max(np.sum(np.abs(A), 1)) * p.u * np.sqrt(n)
x = p.solve32(L, D, b).astype(np.float64); r = b - A @ x
with np.errstate(all='ignore'):
  for it in range(0, 30):
    if it > 0: x = x + p.solve32(L, D, r).astype(np.float64); r = b - A @ x
    Rn, Xn = np.max(np.abs(r)), np.max(np.abs(x))
    print(f"it={it:2d}  Rnrm={Rn:.3e}  Xnrm={Xn:.3e}  Xnrm*cte={Xn*cte:.3e}  Rnrm>Xnrm*cte={Rn > Xn*cte}  "
          f"finite(x)={np.all(np.isfinite(x))}  nan(x)={np.isnan(x).sum()}")
    if not (Rn > Xn * cte): print("  -> accepted with iter =", it); break

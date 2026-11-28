"""Counterexamples for magma_dgesv_rbt(refine=true) built on the port in rbt_port.py.
A = U^{-T} M V^{-1} where M is well conditioned but M[0,0] is 0 (exact zero pivot of the non-pivoted LU) or tiny.
The u, v are those of the FIRST init_butterfly call in a glibc process (rand() unseeded); a different rand() state
gives different u, v, and then this A is just a generic well-conditioned matrix."""
import numpy as np, importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location("p", "rbt_port.py"); p = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(p)
n = 64; u, v, _ = p.init_butterfly(2*n, 0); UT, V = p.dense_UV(n, u, v)
g = np.random.default_rng(7)
for label, m00 in [("exact zero pivot", 0.0), ("pivot 1e-14", 1e-14), ("pivot 1e-8", 1e-8), ("control: pivot 1", None)]:
    Q1, _ = np.linalg.qr(g.standard_normal((n, n))); Q2, _ = np.linalg.qr(g.standard_normal((n, n)))
    M = Q1 @ np.diag(np.linspace(1, 3, n)) @ Q2.T
    if m00 is not None:
        # adjust M[0,0] by a rank-one change that keeps M well conditioned
        M[0, 0] = m00
    A0 = np.linalg.solve(UT, np.linalg.solve(V.T, M.T).T)       # A0 = UT^{-1} M V^{-1}
    xt = g.standard_normal(n); b0 = A0 @ xt
    A = A0.copy(); p.prbt(A, u, v)
    with np.errstate(all='ignore'):
        x, info, _ = p.gesv_rbt(A0, b0, refine=True, state=0)
        err = np.linalg.norm(x - xt) / np.linalg.norm(xt) if np.all(np.isfinite(x)) else float('inf')
    xs = np.linalg.solve(A0, b0)
    print(f"{label:18s} cond(A)={np.linalg.cond(A0):8.2f}  transformed A[0,0]={A[0,0]: .2e}  "
          f"dgesv_rbt: info={info} finite(x)={np.all(np.isfinite(x))} #nan={np.isnan(x).sum():2d} relerr={err:.2e}  "
          f"| pivoted FP64 solve relerr={np.linalg.norm(xs-xt)/np.linalg.norm(xt):.1e}")
    if label == "exact zero pivot": np.save("A_zero_pivot_n64.npy", A0); np.save("b_zero_pivot_n64.npy", b0)

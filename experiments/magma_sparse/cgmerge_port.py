"""Faithful port of the scalar recurrences of MAGMA v2.10.0 sparse magma_zcg_merge (src/zcg_merge.cpp,
blas/zmergecg.cu), complex double.  skp = [alpha|beta|gamma|rho|tmp1|tmp2].
  setup : x = 0; r = b; d = r; beta = zdotc(r, r) (conjugated); nomb = ||b||
  loop  : z = A d; tmp1 = sum d_i z_i          (zmergecg.cu:232, unconjugated)
          rho = beta/tmp1; gamma = beta        (rhokernel :1012)
          x += rho d; r -= rho z; beta = sum r_i r_i   (xrbeta :1258, unconjugated)
          alpha = beta/gamma; d = r + alpha d  (alphabetakernel :1308, d_kernel)
          betanom = sqrt(|beta|); stop if betanom < atol or betanom/nomb < rtol   (zcg_merge.cpp:155,167)
  exit  : numiter < maxiter -> info = 0; final_res = ||b - A x||
Defaults (magma_zparse_opts, double): atol = 1e-16, rtol = 1e-10, maxiter = 1000."""
import numpy as np
def cg_merge(A, b, atol=1e-16, rtol=1e-10, maxiter=1000):
    x = np.zeros_like(b); r = b.copy(); d = r.copy(); beta = np.vdot(r, r); nomb = np.linalg.norm(b) or 1.0
    it = 0
    while True:
        it += 1
        z = A @ d; tmp1 = np.sum(d * z); rho = beta / tmp1; gamma = beta
        x = x + rho * d; r = r - rho * z; beta = np.sum(r * r)
        alpha = beta / gamma; d = r + alpha * d
        betanom = np.sqrt(abs(beta))
        if betanom < atol or betanom / nomb < rtol or it >= maxiter: break
    info = 0 if it < maxiter else -201
    return x, info, it, np.linalg.norm(b - A @ x)
def ref_cg(A, b, tol=1e-10, maxiter=1000):  # textbook CG with conjugated inner products, for contrast
    x = np.zeros_like(b); r = b.copy(); d = r.copy(); rr = np.vdot(r, r).real
    for it in range(1, maxiter+1):
        z = A @ d; a = rr / np.vdot(d, z).real; x += a*d; r -= a*z; rn = np.vdot(r, r).real
        if np.sqrt(rn) / np.linalg.norm(b) < tol: break
        d = r + (rn/rr)*d; rr = rn
    return x, it
cases = {
  "diag(1,1.5), b=(1, 0.5i)": (np.diag([1.0, 1.5]).astype(complex), np.array([1.0, 0.5j])),
  "diag(1,2,3), b=(1, i, 1)": (np.diag([1.0, 2.0, 3.0]).astype(complex), np.array([1.0, 1j, 1.0])),
}
rng = np.random.default_rng(0)
for n in [16, 64]:
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))
    A = Q @ np.diag(np.linspace(1, 10, n)) @ Q.conj().T
    cases[f"random HPD n={n}, cond=10, complex b"] = (A, rng.standard_normal(n) + 1j*rng.standard_normal(n))
    cases[f"random HPD n={n}, cond=10, real b (control)"] = (A.real.copy()+0j if False else A, rng.standard_normal(n)+0j)
for name, (A, b) in cases.items():
    xt = np.linalg.solve(A, b)
    x, info, it, fres = cg_merge(A, b)
    xr, itr = ref_cg(A, b)
    print(f"{name:42s} MAGMA-port: info={info} numiter={it:4d} final_res/||b||={fres/np.linalg.norm(b):.2e} "
          f"relerr={np.linalg.norm(x-xt)/np.linalg.norm(xt):.2e} | textbook CG: iters={itr} relerr={np.linalg.norm(xr-xt)/np.linalg.norm(xt):.1e}")

print("--- control: real SPD data stored as complex (unconjugated == conjugated), port must converge ---")
for n in [16, 64]:
    g = np.random.default_rng(n); Q, _ = np.linalg.qr(g.standard_normal((n, n)))
    A = (Q @ np.diag(np.linspace(1, 10, n)) @ Q.T).astype(complex); b = g.standard_normal(n).astype(complex)
    xt = np.linalg.solve(A, b); x, info, it, fres = cg_merge(A, b)
    print(f"real SPD n={n} in complex storage       MAGMA-port: info={info} numiter={it:4d} "
          f"final_res/||b||={fres/np.linalg.norm(b):.2e} relerr={np.linalg.norm(x-xt)/np.linalg.norm(xt):.2e}")

"""Three further verification rounds of the cuSOLVER GMRES-refinement finding (HANDOFF section 22), several seeds.

Verdicts per call (niters >= 0 means the library reports convergence, so the default fallback is NOT taken):
  WRONG     : niters >= 0 and max rel. error of x vs x_true > 1e-6     (FP64 itself gives ~1e-13 on these systems)
  VIOLATES  : niters >= 0 and eta / thr > 100, eta = ||b - A x||_inf / (||A||_inf ||x||_inf), thr = sqrt(n) * 2^-53
              (100x is far above anything FP64 residual rounding can cause, which is at most ~n*eps*||A|| ||x||)
  ok        : otherwise (converged within ~threshold, or niters < 0 = documented failure + fallback)
Round 1: legacy one-call APIs, A = +-U[0.5,1), n = 1024, seeds 1..10, scales 2^0, 2^20, 2^40, 2^-60.
Round 2: other matrix families (uniform[-1,1), Gaussian, diagonally dominant, nonsymmetric with kappa ~1e3),
         n in {512, 2048}, decimal scales 1e6, 1e12, 1e-18 (realistic units, not powers of two), 3 seeds each.
Round 3 (args "R3 <seed> <api> <k>"): one call in a fresh process (the .bat loops), plus an exact rational residual.
Usage: python irs_verify.py R1 | R2 | R3 seed api k | R3IRS
"""
import math
import sys
from fractions import Fraction

import numpy as np

import irs_probe as P
import irs_scale as S

APIS = ("cusolverDnDHgesv", "cusolverDnDBgesv", "cusolverDnDXgesv", "cusolverDnDSgesv")
for nm in APIS[2:]:
    getattr(P.cusolver, nm + "_bufferSize").argtypes = P.cusolver.cusolverDnDHgesv_bufferSize.argtypes
    getattr(P.cusolver, nm).argtypes = P.cusolver.cusolverDnDHgesv.argtypes


def verdict(niters, fe, q):
    if niters >= 0 and not fe <= 1e-6:
        return "WRONG"
    if niters >= 0 and not q <= 100:
        return "VIOLATES"
    return "ok"


def run_one(nm, A, b, xt):
    r = S.legacy(nm, A, b)
    _, q = P.eta(A, r["x"], b)
    fe = np.abs(r["x"] - xt).max() / np.abs(xt).max()
    return r["niters"], r["info"], r["status"], q, fe, verdict(r["niters"], fe, q), r["x"]


def mat(family, n, rng):
    if family == "pm[.5,1)":
        return rng.choice([-1.0, 1.0], (n, n)) * rng.uniform(0.5, 1.0, (n, n))
    if family == "U[-1,1)":
        return rng.uniform(-1, 1, (n, n))
    if family == "gauss":
        return rng.standard_normal((n, n))
    if family == "diagdom":
        A = rng.uniform(-1, 1, (n, n))
        return A + np.diag(np.abs(A).sum(axis=1))
    if family == "kappa1e3":
        Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
        Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
        return (Q1 * np.logspace(0, -3, n)) @ Q2.T
    raise ValueError(family)


def tally(rows):
    out = {}
    for api, v in rows:
        out.setdefault(api, {"WRONG": 0, "VIOLATES": 0, "ok": 0})[v] += 1
    for api, c in out.items():
        print(f"   {api:18s} WRONG={c['WRONG']:3d} VIOLATES={c['VIOLATES']:3d} ok={c['ok']:3d}", flush=True)


def exact_eta(A, x, b):
    def dec(v):
        m, e = math.frexp(float(v))
        return int(m * (1 << 53)), e - 53
    xd = [dec(v) for v in x]
    worst = Fraction(0)
    for i in range(A.shape[0]):
        mb, eb = dec(b[i])
        terms = [(-mb, eb)] + [(ma * xd[j][0], ea + xd[j][1]) for j, (ma, ea) in enumerate(dec(v) for v in A[i])]
        e0 = min(t[1] for t in terms)
        worst = max(worst, Fraction(abs(sum(t[0] << (t[1] - e0) for t in terms))) * Fraction(2) ** e0)
    # ||A||_inf and ||x||_inf exactly
    an = max(sum(abs(Fraction(float(v))) for v in row) for row in A)
    xn = max(abs(Fraction(float(v))) for v in x)
    return float(worst / (an * xn)) / (math.sqrt(A.shape[0]) * P.EPS)


if __name__ == "__main__":
    which = sys.argv[1]
    print(f"cuSOLVER {P.ver.value}; round {which}", flush=True)
    if which == "R1":
        n, rows = 1024, []
        for seed in range(1, 11):
            rng = np.random.default_rng(seed)
            A0 = mat("pm[.5,1)", n, rng)
            xt = rng.uniform(-1, 1, n)
            b0 = A0 @ xt
            for nm in APIS:
                for k in (0, 20, 40, -60):
                    # 2^20 A would exceed the FP16 range, so DHgesv uses 2^4 instead of the other scales
                    kk = 4 if (nm == "cusolverDnDHgesv" and k != 0) else k
                    s = 2.0 ** kk
                    it, info, st, q, fe, v, _ = run_one(nm, A0 * s, b0 * s, xt)
                    rows.append((f"{nm} 2^{kk}", v))
                    print(f"seed={seed:2d} {nm} 2^{kk:<4d} niters={it:4d} info={info} st={st} eta/thr={q:9.3g} "
                          f"fwd={fe:9.3g} {v}", flush=True)
        print("\nR1 tally (10 seeds per line):")
        tally(rows)
    elif which == "R2":
        rows = []
        for n in (512, 2048):
            for fam in ("U[-1,1)", "gauss", "diagdom", "kappa1e3"):
                for seed in (101, 202, 303):
                    rng = np.random.default_rng(seed)
                    A0 = mat(fam, n, rng)
                    xt = rng.uniform(-1, 1, n)
                    for scale in (1.0, 1e6, 1e12, 1e-18):
                        A = A0 * scale
                        b = A @ xt
                        for nm in APIS:
                            if nm == "cusolverDnDHgesv" and scale != 1.0:
                                continue
                            it, info, st, q, fe, v, _ = run_one(nm, A, b, xt)
                            rows.append((f"{nm} {scale:g}", v))
                            print(f"n={n} {fam:9s} seed={seed} scale={scale:<6g} {nm} niters={it:4d} info={info} "
                                  f"eta/thr={q:9.3g} fwd={fe:9.3g} {v}", flush=True)
        print("\nR2 tally (2 sizes x 4 families x 3 seeds = 24 per line):")
        tally(rows)
    elif which == "R3":
        seed, nm, k = int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
        n = 512
        rng = np.random.default_rng(seed)
        A0 = mat("pm[.5,1)", n, rng)
        xt = rng.uniform(-1, 1, n)
        A, b = A0 * 2.0 ** k, (A0 @ xt) * 2.0 ** k
        it, info, st, q, fe, v, x = run_one(nm, A, b, xt)
        qe = exact_eta(A, x, b) if it >= 0 else float("nan")
        print(f"fresh-process seed={seed} {nm} 2^{k} n={n} niters={it} info={info} eta/thr FP64={q:.4g} exact={qe:.4g} "
              f"fwd={fe:.3g} {v}", flush=True)
    elif which == "R3IRS":
        n, rows = 2048, []
        for seed in (7, 8, 9):
            rng = np.random.default_rng(seed)
            A0 = mat("pm[.5,1)", n, rng)
            xt = rng.uniform(-1, 1, n)
            b0 = A0 @ xt
            for low, rf in (("32F", "CL"), ("32F", "GM"), ("16BF", "GM"), ("TF32", "GM")):
                its = []
                for k in (-10, 0, 10, 20, 40):
                    s = 2.0 ** k
                    r = P.irs(A0 * s, b0 * s, low, rf)
                    _, q = P.eta(A0 * s, r["x"], b0 * s)
                    fe = np.abs(r["x"] - xt).max() / np.abs(xt).max()
                    v = verdict(r["niters"], fe, q)
                    rows.append((f"IRS {low}/{rf}", v))
                    its.append(f"2^{k}:{r['niters']}/{q:.2g}/{fe:.2g}/{v}")
                print(f"seed={seed} {low}/{rf}  (k:niters/eta_over_thr/fwd/verdict)  " + "  ".join(its), flush=True)
        print("\nR3IRS tally (3 seeds x 5 scales = 15 per line):")
        tally(rows)

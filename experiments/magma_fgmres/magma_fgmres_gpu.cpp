// GPU confirmation of the predictions of lean/OptimiKahan/MagmaFgmres.lean (HANDOFF section 24) for MAGMA 2.10.0
// magma_dsgesv_iteref_gpu (FP32 LU + FGMRES, GMSTRS) and magma_dhgesv_iteref_gpu (FP16 tensor-core LU + FGMRES).
// System: A = 2^k * A0, b = 2^k * b0, A0 entries +-U[0.5,1), x_true U[-1,1), b0 = A0 x_true; n = 1024 by default.
// For every call the model's prediction is printed next to the measurement:
//   tiny_rhs_returns_zero : ||b||_2 <= 1e-18            =>  iter = 0, x = 0
//   tol_ge_one_one_step   : tol = ||A||_inf eps sqrt(n) >= 1 (eps = 2^-53)  =>  iter = 1
//   gmstrs_always_success : iter >= 0 and info = 0 (after a successful low-precision factorization)
//   enforced_rule         : the FGMRES estimate <= tol * ||b||_2; we report the true ||r||_2/||b||_2 next to tol,
//                           or iter = maxiter = min(n, 500)
// Backward error eta = ||b - A x||_inf / (||A||_inf ||x||_inf) is computed on the host in long double (80-bit), and
// reported as eta / (sqrt(n) eps), the documented threshold (an FP64 residual would be too coarse: n eps > sqrt(n) eps).
// Build: g++ -O2 -fopenmp magma_fgmres_gpu.cpp -I$MAGMA/include -L$MAGMA/lib -lmagma -llapack -lblas -lcublas
//        -lcusparse -lcudart -o magma_fgmres_gpu      Run: CUDA_VISIBLE_DEVICES=0 ./magma_fgmres_gpu [n]
#include <magma_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

typedef magma_int_t (*iteref_fn)(magma_trans_t, magma_int_t, magma_int_t, magmaDouble_ptr, magma_int_t, magma_int_t*,
                                 magmaInt_ptr, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t,
                                 magmaDouble_ptr, magmaFloat_ptr, magma_int_t*, magma_int_t*);

struct Res { long iter; long info; std::vector<double> x; };

static Res run_iteref(iteref_fn f, magma_int_t n, const std::vector<double>& A, const std::vector<double>& b,
                      magma_queue_t q) {
  double *dA, *dB, *dX, *dworkd;
  float* dworks;
  magma_int_t *dipiv, iter = -12345, info = -12345;
  std::vector<magma_int_t> ipiv(n);
  magma_dmalloc(&dA, (size_t)n * n);
  magma_dmalloc(&dB, n);
  magma_dmalloc(&dX, n);
  magma_dmalloc(&dworkd, n);
  magma_smalloc(&dworks, (size_t)n * (n + 1));
  magma_imalloc(&dipiv, n);
  magma_dsetmatrix(n, n, A.data(), n, dA, n, q);
  magma_dsetvector(n, b.data(), 1, dB, 1, q);
  f(MagmaNoTrans, n, 1, dA, n, ipiv.data(), dipiv, dB, n, dX, n, dworkd, dworks, &iter, &info);
  Res r{(long)iter, (long)info, std::vector<double>(n)};
  magma_dgetvector(n, dX, 1, r.x.data(), 1, q);
  magma_free(dA); magma_free(dB); magma_free(dX); magma_free(dworkd); magma_free(dworks); magma_free(dipiv);
  return r;
}

int main(int argc, char** argv) {
  const magma_int_t n = argc > 1 ? atoi(argv[1]) : 1024;
  magma_init();
  magma_queue_t q;
  magma_queue_create(0, &q);
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> U(0.5, 1.0), V(-1.0, 1.0);
  std::bernoulli_distribution sgn(0.5);
  std::vector<double> A0((size_t)n * n), xt(n), b0(n, 0.0);
  for (auto& a : A0) a = (sgn(rng) ? 1.0 : -1.0) * U(rng);
  for (auto& v : xt) v = V(rng);
  for (magma_int_t j = 0; j < n; j++)
    for (magma_int_t i = 0; i < n; i++) b0[i] += A0[(size_t)j * n + i] * xt[j];
  const double eps = std::ldexp(1.0, -53), thr = std::sqrt((double)n) * eps;
  const long maxiter = n < 500 ? n : 500;
  struct R { const char* name; iteref_fn f; } routines[] = {{"dsgesv_iteref (FP32 LU)", magma_dsgesv_iteref_gpu},
                                                            {"dhgesv_iteref (FP16-TC LU)", magma_dhgesv_iteref_gpu}};
  printf("MAGMA %d.%d.%d, n = %lld, maxiter = min(n,500) = %ld, documented threshold sqrt(n)*eps = %.3g\n",
         MAGMA_VERSION_MAJOR, MAGMA_VERSION_MINOR, MAGMA_VERSION_MICRO, (long long)n, maxiter, thr);
  const int ks[] = {-80, -70, -60, -40, -20, 0, 10, 20, 30, 40, 50, 60};
  int mismatches = 0, total = 0;
  for (auto& rt : routines) {
    printf("\n%s\n", rt.name);
    for (int k : ks) {
      const double s = std::ldexp(1.0, k);
      std::vector<double> A(A0), b(b0);
      for (auto& a : A) a *= s;
      for (auto& v : b) v *= s;
      double an = 0, b2 = 0;
      for (magma_int_t i = 0; i < n; i++) {
        double rs = 0;
        for (magma_int_t j = 0; j < n; j++) rs += std::fabs(A[(size_t)j * n + i]);
        an = rs > an ? rs : an;
        b2 += b[i] * b[i];
      }
      b2 = std::sqrt(b2);
      const double tol = an * eps * std::sqrt((double)n);
      Res r = run_iteref(rt.f, n, A, b, q);
      // long double residual
      long double rinf = 0, r2 = 0, xinf = 0, fe = 0, xtinf = 0;
      bool finite = true;
      for (magma_int_t i = 0; i < n; i++) {
        if (!std::isfinite(r.x[i])) finite = false;
        long double acc = b[i];
        for (magma_int_t j = 0; j < n; j++) acc -= (long double)A[(size_t)j * n + i] * r.x[j];
        rinf = fabsl(acc) > rinf ? fabsl(acc) : rinf;
        r2 += acc * acc;
        xinf = fabsl(r.x[i]) > xinf ? fabsl(r.x[i]) : xinf;
        fe = fabsl(r.x[i] - xt[i]) > fe ? fabsl(r.x[i] - xt[i]) : fe;
        xtinf = fabsl(xt[i]) > xtinf ? fabsl(xt[i]) : xtinf;
      }
      const double bwd = (xinf > 0 && finite) ? (double)(rinf / ((long double)an * xinf)) / thr : INFINITY;
      const double rel = finite ? (double)(sqrtl(r2) / b2) : NAN;
      // the model's prediction
      const char* pred;
      bool ok;
      if (b2 <= 1e-18) { pred = "iter=0,x=0 [tiny_rhs]"; ok = (r.iter == 0 && xinf == 0); }
      else if (tol >= 1) { pred = "iter=1 [tol>=1]"; ok = (r.iter == 1); }
      else { pred = "rel<=tol or iter=maxiter [enforced]"; ok = (rel <= tol * 1.01) || (r.iter == maxiter); }
      ok = ok && r.iter >= 0 && r.info == 0;  // gmstrs_always_success
      total++;
      mismatches += !ok;
      printf("  k=%4d ||b||_2=%9.3g tol=%9.3g | iter=%4ld info=%ld | rel.res=%9.3g bwd/thr=%9.3g fwd_err=%9.3g "
             "finite=%d | predicted %-34s %s\n", k, b2, tol, r.iter, r.info, rel, bwd, (double)(fe / xtinf), finite,
             pred, ok ? "MATCH" : "MISMATCH");
      fflush(stdout);
    }
  }
  printf("\n%d / %d calls match the model's prediction\n", total - mismatches, total);
  magma_queue_destroy(q);
  magma_finalize();
  return 0;
}

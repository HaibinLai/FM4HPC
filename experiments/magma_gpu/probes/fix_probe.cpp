// Overhead of the GMSTRS fix, and the trans / nrhs predictions (HANDOFF §24.x), MAGMA 2.10.0, GPU 0.
// The same binary is run against the original and the patched libmagma.so (switch LD_LIBRARY_PATH).
#include "magma_v2.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstdlib>
typedef magma_int_t (*iteref_fn)(magma_trans_t, magma_int_t, magma_int_t, magmaDouble_ptr, magma_int_t, magma_int_t*,
                                 magmaInt_ptr, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t,
                                 magmaDouble_ptr, magmaFloat_ptr, magma_int_t*, magma_int_t*);
static magma_queue_t Q;

struct Out { long iter, info; double t; std::vector<double> x; };
static Out call(iteref_fn f, magma_trans_t tr, int n, int nrhs, const std::vector<double>& A, const std::vector<double>& B) {
  double *dA, *dB, *dX, *dwd; float* dws; magma_int_t* dipiv; std::vector<magma_int_t> ipiv(n);
  magma_dmalloc(&dA, (size_t)n * n); magma_dmalloc(&dB, (size_t)n * nrhs); magma_dmalloc(&dX, (size_t)n * nrhs);
  magma_dmalloc(&dwd, (size_t)n * nrhs); magma_smalloc(&dws, (size_t)n * (n + nrhs)); magma_imalloc(&dipiv, n);
  magma_dsetmatrix(n, n, A.data(), n, dA, n, Q); magma_dsetmatrix(n, nrhs, B.data(), n, dB, n, Q);
  magma_int_t iter = -12345, info = -12345;
  magma_queue_sync(Q); double t0 = magma_sync_wtime(Q);
  f(tr, n, nrhs, dA, n, ipiv.data(), dipiv, dB, n, dX, n, dwd, dws, &iter, &info);
  double t1 = magma_sync_wtime(Q);
  Out o{(long)iter, (long)info, t1 - t0, std::vector<double>((size_t)n * nrhs)};
  magma_dgetmatrix(n, nrhs, dX, n, o.x.data(), n, Q);
  magma_free(dA); magma_free(dB); magma_free(dX); magma_free(dwd); magma_free(dws); magma_free(dipiv);
  return o;
}
static void make(int n, std::vector<double>& A, std::vector<double>& xt, unsigned seed) {
  std::mt19937_64 g(seed); std::uniform_real_distribution<double> u(0.5, 1.0), s(-1.0, 1.0);
  A.resize((size_t)n * n); for (auto& a : A) a = (s(g) < 0 ? -1 : 1) * u(g);
  for (int i = 0; i < n; i++) A[i + (size_t)i * n] += n;           // diagonally dominant: well conditioned
  xt.resize(n); for (auto& v : xt) v = s(g);
}
static double relerr(const double* x, const double* xt, int n) { double e = 0, m = 0;
  for (int i = 0; i < n; i++) { if (!std::isfinite(x[i])) return INFINITY; e = fmax(e, fabs(x[i] - xt[i])); m = fmax(m, fabs(xt[i])); } return e / m; }

int main() {
  magma_init(); magma_queue_create(0, &Q);
  struct R { const char* nm; iteref_fn f; } rs[] = {{"dsgesv_iteref", magma_dsgesv_iteref_gpu}, {"dhgesv_iteref", magma_dhgesv_iteref_gpu}};
  printf("== timing (well-conditioned, k = 0), best of 3 after one warm-up ==\n");
  std::vector<int> ns = {1024, 2048, 4096, 8192}; if (getenv("FIXN")) ns = {atoi(getenv("FIXN"))};
  for (int n : ns) {
    std::vector<double> A, xt; make(n, A, xt, 1); std::vector<double> b(n);
    for (int i = 0; i < n; i++) { double t = 0; for (int j = 0; j < n; j++) t += A[i + (size_t)j * n] * xt[j]; b[i] = t; }
    for (auto& r : rs) { call(r.f, MagmaNoTrans, n, 1, A, b); double best = 1e9; Out o;
      for (int k = 0; k < 3; k++) { o = call(r.f, MagmaNoTrans, n, 1, A, b); best = fmin(best, o.t); }
      printf("%-14s n=%5d  time=%8.4f s  iter=%4ld info=%ld  relerr=%.2e\n", r.nm, n, best, o.iter, o.info, relerr(o.x.data(), xt.data(), n)); }
  }
  if (getenv("FIXN")) { magma_queue_destroy(Q); magma_finalize(); return 0; }
  printf("== trans = MagmaTrans (A nonsymmetric): solution should satisfy A^T x = b ==\n");
  { int n = 1024; std::vector<double> A, xt; make(n, A, xt, 2); std::vector<double> b(n);
    for (int i = 0; i < n; i++) { double t = 0; for (int j = 0; j < n; j++) t += A[j + (size_t)i * n] * xt[j]; b[i] = t; }   // b = A^T x_true
    for (auto& r : rs) { Out o = call(r.f, MagmaTrans, n, 1, A, b);
      printf("%-14s trans=T n=%d  iter=%4ld info=%ld  relerr vs A^T-solution=%.2e\n", r.nm, n, o.iter, o.info, relerr(o.x.data(), xt.data(), n)); } }
  printf("== nrhs = 2 ==\n");
  { int n = 1024; std::vector<double> A, xt; make(n, A, xt, 3); std::vector<double> X2(2 * n), B(2 * n);
    for (int c = 0; c < 2; c++) for (int i = 0; i < n; i++) X2[i + c * n] = (c == 0 ? xt[i] : -2 * xt[i]);
    for (int c = 0; c < 2; c++) for (int i = 0; i < n; i++) { double t = 0; for (int j = 0; j < n; j++) t += A[i + (size_t)j * n] * X2[j + c * n]; B[i + c * n] = t; }
    for (auto& r : rs) { Out o = call(r.f, MagmaNoTrans, n, 2, A, B);
      double m1 = 0; for (int i = 0; i < n; i++) m1 = fmax(m1, fabs(o.x[n + i]));
      printf("%-14s nrhs=2 n=%d  iter=%4ld info=%ld  relerr col0=%.2e  col1=%.2e  max|x col1|=%.3g\n", r.nm, n, o.iter, o.info,
             relerr(o.x.data(), X2.data(), n), relerr(o.x.data() + n, X2.data() + n, n), m1); } }
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

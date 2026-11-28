// Nineteenth GPU round: findings of the independent review (HANDOFF §29.20).  GPU 0 only.
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z;
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 31u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static double diff(const std::vector<Z>& a, const std::vector<Z>& b) { double e = 0, m = 0; for (size_t i = 0; i < a.size(); i++) { if (!std::isfinite(MAGMA_Z_REAL(a[i]))) return INFINITY;
  e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(a[i], b[i]))); m = fmax(m, MAGMA_Z_ABS(b[i])); } return e / m; }
static void ungbr(long k) {
  void (*fi_arm)(long) = (void (*)(long))dlsym(RTLD_DEFAULT, "fi_arm"); void (*fi_disarm)() = (void (*)())dlsym(RTLD_DEFAULT, "fi_disarm"); long (*fi_count)() = (long (*)())dlsym(RTLD_DEFAULT, "fi_count");
  int n = 400; magma_int_t nn = n, inf, lw = n * 256; std::vector<Z> A((size_t)n * n), tauq(n), taup(n), wk(lw); std::vector<double> d(n), e(n); for (auto& a : A) a = C(rnd(), rnd());
  lapackf77_zgebrd(&nn, &nn, A.data(), &nn, d.data(), e.data(), tauq.data(), taup.data(), wk.data(), &lw, &inf);
  for (magma_vect_t v : {MagmaQ, MagmaP}) { std::vector<Z>& tau = v == MagmaQ ? tauq : taup; std::vector<Z> ref = A, out = A; magma_int_t info = -999;
    magma_zungbr(v, n, n, n, ref.data(), n, tau.data(), wk.data(), lw, &inf);
    if (fi_arm) fi_arm(k); magma_zungbr(v, n, n, n, out.data(), n, tau.data(), wk.data(), lw, &info); long c = fi_count ? fi_count() : -1; if (fi_disarm) fi_disarm();
    double er = diff(out, ref); printf("ungbr(%s) k=%3ld nalloc=%3ld info=%-6lld diff_to_unfailed=%-10.3g %s\n", v == MagmaQ ? "Q" : "P", k, c, (long long)info, er,
           (info == 0 && !(er <= 1e-12)) ? "SILENT-WRONG" : (info != 0 ? "reported" : "ok")); }
}
static void lascl_2x2() { Z *dW, *dA; magma_zmalloc(&dW, 64); magma_zmalloc(&dA, 64); magma_int_t info = 0;
  magmablas_zlascl_2x2(MagmaLower, 4, dW, 4, dA, 2, Q, &info); printf("zlascl_2x2 ldda = 2 < m = 4 (argument 6): info=%lld\n", (long long)info);
  info = 0; magmablas_zlascl_2x2(MagmaLower, 4, dW, 1, dA, 4, Q, &info); printf("zlascl_2x2 lddw = 1 < m (argument 4; W is read up to row m+1): info=%lld\n", (long long)info); }
static void residual(bool zero) {
  if (zero) { magma_z_matrix A = {Magma_CSR}, b = {Magma_CSR}, x = {Magma_CSR}; std::vector<magma_index_t> r = {0}; magma_zcsrset(0, 0, r.data(), NULL, NULL, &A, Q);
    magma_z_matrix dA = {Magma_CSR}; magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q); magma_zvinit(&b, Magma_DEV, 0, 1, C(0), Q); magma_zvinit(&x, Magma_DEV, 0, 1, C(0), Q);
    double res = -1; printf("zresidual with A 0x0 ...\n"); fflush(stdout); magma_int_t ret = magma_zresidual(dA, b, x, &res, Q); printf("  ret=%lld res=%g\n", (long long)ret, res); return; }
  int n = 4, k = 2; std::vector<magma_index_t> r = {0, 1, 2, 3, 4}, c = {0, 1, 2, 3}; std::vector<Z> v(4, C(1)); magma_z_matrix A = {Magma_CSR}, dA = {Magma_CSR};
  magma_zcsrset(n, n, r.data(), c.data(), v.data(), &A, Q); magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q);
  std::vector<Z> B(n * k), X(n * k, C(0)); for (int i = 0; i < n * k; i++) B[i] = C(i < n ? 1.0 : 2.0); magma_z_matrix hb = {Magma_CSR}, hx = {Magma_CSR}, db = {Magma_CSR}, dx = {Magma_CSR};
  magma_zvset(n, k, B.data(), &hb, Q); magma_zvset(n, k, X.data(), &hx, Q); magma_zmtransfer(hb, &db, Magma_CPU, Magma_DEV, Q); magma_zmtransfer(hx, &dx, Magma_CPU, Magma_DEV, Q);
  double res[2] = {-1, -1}; magma_int_t ret = magma_zresidual(dA, db, dx, res, Q);
  printf("zresidual, A = I (4x4), b 4x2 with columns of 1s and 2s, x = 0: ret=%lld res = %g %g (expected 2, 4)\n", (long long)ret, res[0], res[1]); }
static void csrsplit() { std::vector<magma_index_t> r = {0, 1, 3}, c = {0, 1, 0}; std::vector<Z> v = {C(1), C(2), C(3)};   // row 1 = {(1,1)=2, (1,0)=3}: diagonal before lower
  magma_z_matrix A = {Magma_CSR}, D = {Magma_CSR}, R = {Magma_CSR}; magma_zcsrset(2, 2, r.data(), c.data(), v.data(), &A, Q);
  printf("zcsrsplit(offset 0, bsize 1) of a 2x2 CSR whose last row is unsorted ...\n"); fflush(stdout); magma_int_t ret = magma_zcsrsplit(0, 1, A, &D, &R, Q);
  printf("  ret=%lld D nnz=%lld R nnz=%lld; freeing ...\n", (long long)ret, (long long)D.nnz, (long long)R.nnz); fflush(stdout); magma_zmfree(&D, Q); magma_zmfree(&R, Q); printf("  freed\n"); }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; long k = argc > 2 ? atol(argv[2]) : 0; magma_init(); magma_queue_create(0, &Q);
  if (m == "ungbr") ungbr(k); else if (m == "lascl_2x2") lascl_2x2(); else if (m == "residual0") residual(true); else if (m == "residual") residual(false); else if (m == "csrsplit") csrsplit();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

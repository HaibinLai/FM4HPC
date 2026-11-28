// Ninth GPU round: multi-GPU / out-of-core drivers run with ngpu = 1, the three last unaudited dense files, and
// zmscale_generate (HANDOFF §29.9).  GPU 0 only; one mode per process.
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z;
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 99u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }

typedef magma_int_t (*heevdx_fn)(magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Z*, magma_int_t, double, double, magma_int_t, magma_int_t,
                                 magma_int_t*, double*, Z*, magma_int_t, double*, magma_int_t, magma_int_t*, magma_int_t, magma_int_t*);
static magma_int_t heevdx_m1(magma_vec_t j, magma_range_t r, magma_uplo_t u, magma_int_t n, Z* A, magma_int_t lda, double vl, double vu, magma_int_t il, magma_int_t iu,
                             magma_int_t* m, double* w, Z* wk, magma_int_t lw, double* rw, magma_int_t lrw, magma_int_t* iw, magma_int_t liw, magma_int_t* info) {
  return magma_zheevdx_m(1, j, r, u, n, A, lda, vl, vu, il, iu, m, w, wk, lw, rw, lrw, iw, liw, info); }
static magma_int_t heevdx2_m1(magma_vec_t j, magma_range_t r, magma_uplo_t u, magma_int_t n, Z* A, magma_int_t lda, double vl, double vu, magma_int_t il, magma_int_t iu,
                              magma_int_t* m, double* w, Z* wk, magma_int_t lw, double* rw, magma_int_t lrw, magma_int_t* iw, magma_int_t liw, magma_int_t* info) {
  return magma_zheevdx_2stage_m(1, j, r, u, n, A, lda, vl, vu, il, iu, m, w, wk, lw, rw, lrw, iw, liw, info); }
static void run_heevdx(const char* nm, heevdx_fn f, magma_vec_t jobz, magma_range_t rg, int n, const std::vector<Z>& A0, double vl, double vu,
                       magma_int_t* m_out, std::vector<double>& w, magma_int_t* info_out) {
  std::vector<Z> A = A0; w.assign(n, NAN); magma_int_t m = -1, info = -1, lw = -1, lrw = -1, liw = -1; Z q; double rq; magma_int_t iq;
  f(jobz, rg, MagmaLower, n, A.data(), n, vl, vu, 1, n, &m, w.data(), &q, lw, &rq, lrw, &iq, liw, &info);
  lw = (magma_int_t)MAGMA_Z_REAL(q); lrw = (magma_int_t)rq; liw = iq;
  std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); A = A0;
  f(jobz, rg, MagmaLower, n, A.data(), n, vl, vu, 1, n, &m, w.data(), wk.data(), lw, rw.data(), lrw, iw.data(), liw, &info);
  *m_out = m; *info_out = info; (void)nm;
}
static void rangev() {
  int n = getenv("RN") ? atoi(getenv("RN")) : 200; std::vector<Z> A((size_t)n * n, C(0)); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = C(1e-150 * (i + 1));
  double vl = 0, vu = 1.005e-148;
  printf("== Range V on A = 1e-150*diag(1..%d) (anrm < rmin => internal scaling), (vl, vu] = (0, 1.005e-148]: 100 eigenvalues ==\n", n);
  struct { const char* nm; heevdx_fn f; } fs[] = {{"zheevdx", magma_zheevdx}, {"zheevdx_m(ngpu=1)", heevdx_m1}, {"zheevdx_2stage", magma_zheevdx_2stage}, {"zheevdx_2stage_m(ngpu=1)", heevdx2_m1}};
  for (auto& F : fs) for (magma_vec_t jz : {MagmaNoVec, MagmaVec}) { magma_int_t m, info; std::vector<double> w;
    run_heevdx(F.nm, F.f, jz, MagmaRangeV, n, A, vl, vu, &m, w, &info);
    printf("%-26s jobz=%s: info=%lld m=%lld (expected 100)  w[0]=%g\n", F.nm, jz == MagmaVec ? "V" : "N", (long long)info, (long long)m, w[0]); }
  // control: same spectrum scaled up by 1e150 (no internal scaling), vu = 100.5
  std::vector<Z> B((size_t)n * n, C(0)); for (int i = 0; i < n; i++) B[i + (size_t)i * n] = C(i + 1.0);
  magma_int_t m, info; std::vector<double> w; run_heevdx("zheevdx", magma_zheevdx, MagmaNoVec, MagmaRangeV, n, B, 0, 100.5, &m, w, &info);
  printf("control zheevdx, A = diag(1..200), (0, 100.5]: info=%lld m=%lld\n", (long long)info, (long long)m);
}
static void eig_nan() {
  int n = 1000; std::vector<Z> A((size_t)n * n);
  for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { double v = rnd(); A[i + (size_t)j * n] = C(v); A[j + (size_t)i * n] = C(v); }
  A[7 + (size_t)3 * n] = C(NAN); A[3 + (size_t)7 * n] = C(NAN);
  printf("== Hermitian n = 1000 with one NaN pair, jobz = V, range All ==\n");
  if (!getenv("EIGF") || atoi(getenv("EIGF")) == 9) { std::vector<Z> Al = A; std::vector<double> w(n); magma_int_t nn = n, lw = -1, lrw = -1, liw = -1, info; Z q; double rq; magma_int_t iq;
    lapackf77_zheevd("V", "L", &nn, Al.data(), &nn, w.data(), &q, &lw, &rq, &lrw, &iq, &liw, &info);
    lw = (magma_int_t)MAGMA_Z_REAL(q); lrw = (magma_int_t)rq; liw = iq; std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw);
    lapackf77_zheevd("V", "L", &nn, Al.data(), &nn, w.data(), wk.data(), &lw, rw.data(), &lrw, iw.data(), &liw, &info);
    int nf = 0; for (double v : w) if (!std::isfinite(v)) nf++; printf("LAPACK zheevd: info=%lld non-finite eigenvalues=%d\n", (long long)info, nf); }
  struct { const char* nm; heevdx_fn f; } fs[] = {{"zheevdx", magma_zheevdx}, {"zheevdx_m(ngpu=1)", heevdx_m1}, {"zheevdx_2stage", magma_zheevdx_2stage}, {"zheevdx_2stage_m(ngpu=1)", heevdx2_m1}};
  int only = getenv("EIGF") ? atoi(getenv("EIGF")) : -1; int idx = -1;
  for (auto& F : fs) { idx++; if (only >= 0 && idx != only) continue; magma_int_t m, info; std::vector<double> w;
    printf("calling %s ...\n", F.nm); fflush(stdout); run_heevdx(F.nm, F.f, MagmaVec, MagmaRangeAll, n, A, 0, 0, &m, w, &info);
    int nf = 0; for (double v : w) if (!std::isfinite(v)) nf++;
    printf("%-26s info=%lld m=%lld  non-finite eigenvalues=%d\n", F.nm, (long long)info, (long long)m, nf); }
}
static void unmtr_m() {
  int m = 10, n = 200; std::vector<Z> A((size_t)n * n), tau(n), Cm((size_t)m * n), C0;
  for (auto& a : A) a = C(rnd()); for (auto& t : tau) t = C(1.0 + 0.1 * rnd()); for (auto& c : Cm) c = C(rnd()); C0 = Cm;
  magma_int_t info = -999, lwork = -1; Z q; magma_zunmtr_m(1, MagmaRight, MagmaLower, MagmaNoTrans, m, n, A.data(), n, tau.data(), Cm.data(), m, &q, lwork, &info);
  lwork = std::max((magma_int_t)MAGMA_Z_REAL(q), (magma_int_t)(n * 256)); std::vector<Z> wk(lwork); info = -999;
  magma_zunmtr_m(1, MagmaRight, MagmaLower, MagmaNoTrans, m, n, A.data(), n, tau.data(), Cm.data(), m, wk.data(), lwork, &info);
  int same = 0; for (size_t i = 0; i < Cm.size(); i++) if (MAGMA_Z_EQUAL(Cm[i], C0[i])) same++;
  printf("zunmtr_m(ngpu=1, Right, Lower, m=10, n=200): info=%lld  entries of C unchanged: %d of %zu\n", (long long)info, same, Cm.size());
  std::vector<Z> Cs = C0; info = -999; magma_zunmtr(MagmaRight, MagmaLower, MagmaNoTrans, m, n, A.data(), n, tau.data(), Cs.data(), m, wk.data(), lwork, &info);
  same = 0; for (size_t i = 0; i < Cs.size(); i++) if (MAGMA_Z_EQUAL(Cs[i], C0[i])) same++;
  printf("zunmtr  (single GPU, same call):           info=%lld  entries of C unchanged: %d of %zu\n", (long long)info, same, Cs.size());
}
static void he2hb() {
  int n = 256, nb = 32; std::vector<Z> A0((size_t)n * n);
  for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { double v = rnd(); A0[i + (size_t)j * n] = C(v); A0[j + (size_t)i * n] = C(v); }
  Z* dT; magma_zmalloc(&dT, (size_t)n * n);
  std::vector<std::vector<Z>> out;
  for (magma_int_t lwork : {(magma_int_t)(n * nb * 4), (magma_int_t)1056, (magma_int_t)100}) {
    std::vector<Z> A = A0, tau(n), wk(std::max<magma_int_t>(lwork, 1)); magma_int_t info = -999;
    printf("zhetrd_he2hb n=256 nb=32 lwork=%lld ...\n", (long long)lwork); fflush(stdout);
    magma_zhetrd_he2hb(MagmaLower, n, nb, A.data(), n, tau.data(), wk.data(), lwork, dT, &info);
    double d = 0; if (!out.empty()) for (int j = 0; j < n; j++) for (int i = j; i < std::min(n, j + nb + 1); i++) d = fmax(d, MAGMA_Z_ABS(MAGMA_Z_SUB(A[i + (size_t)j * n], out[0][i + (size_t)j * n])));
    printf("  info=%lld  max |band - band(lwork = 4 n nb)| = %.3g\n", (long long)info, d); out.push_back(A);
  }
}
static void geqrf_m0() {
  int m = 1024, n = 1024; std::vector<Z> A((size_t)m * n); for (auto& a : A) a = C(rnd()); std::vector<Z> tau(n);
  magma_int_t info = -999, lwork = -1; Z q; magma_zgeqrf_m(1, m, n, A.data(), m, tau.data(), &q, lwork, &info); lwork = (magma_int_t)MAGMA_Z_REAL(q);
  std::vector<Z> wk(lwork); printf("zgeqrf_m with ngpu = 0 (the check is ngpu < 0) ...\n"); fflush(stdout);
  magma_zgeqrf_m(0, m, n, A.data(), m, tau.data(), wk.data(), lwork, &info); printf("  returned info=%lld\n", (long long)info);
}
static void cgeqrsv() {
  int n = 64; std::vector<Z> A((size_t)n * n), B(n); for (auto& a : A) a = C(rnd()); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = MAGMA_Z_ADD(A[i + (size_t)i * n], C(4));
  for (int i = 0; i < n; i++) A[i] = C(0); for (auto& b : B) b = C(rnd());
  { std::vector<Z> Al = A, Bl = B; magma_int_t nn = n, one = 1, lw = -1, info; Z q; lapackf77_zgels("N", &nn, &nn, &one, Al.data(), &nn, Bl.data(), &nn, &q, &lw, &info);
    lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> wk(lw); lapackf77_zgels("N", &nn, &nn, &one, Al.data(), &nn, Bl.data(), &nn, wk.data(), &lw, &info);
    printf("LAPACK zgels, column 0 = 0: info=%lld\n", (long long)info); }
  Z *dA, *dB, *dX; magma_zmalloc(&dA, (size_t)n * n); magma_zmalloc(&dB, n); magma_zmalloc(&dX, n);
  magma_zsetmatrix(n, n, A.data(), n, dA, n, Q); magma_zsetmatrix(n, 1, B.data(), n, dB, n, Q); magma_int_t iter = -999, info = -999;
  magma_zcgeqrsv_gpu(n, n, 1, dA, n, dB, n, dX, n, &iter, &info);
  std::vector<Z> X(n); magma_zgetmatrix(n, 1, dX, n, X.data(), n, Q); int nf = 0; for (auto& x : X) if (!std::isfinite(MAGMA_Z_REAL(x)) || !std::isfinite(MAGMA_Z_IMAG(x))) nf++;
  printf("zcgeqrsv_gpu n=64, column 0 = 0: iter=%lld info=%lld  non-finite x entries=%d of %d\n", (long long)iter, (long long)info, nf, n);
  std::vector<Z> Z0(1, C(0)), B1(1, C(1)); magma_zsetmatrix(1, 1, Z0.data(), 1, dA, 1, Q); magma_zsetmatrix(1, 1, B1.data(), 1, dB, 1, Q); iter = info = -999;
  magma_zcgeqrsv_gpu(1, 1, 1, dA, 1, dB, 1, dX, 1, &iter, &info); magma_zgetmatrix(1, 1, dX, 1, X.data(), 1, Q);
  printf("zcgeqrsv_gpu A = [0], b = [1]: iter=%lld info=%lld x=(%g,%g)\n", (long long)iter, (long long)info, MAGMA_Z_REAL(X[0]), MAGMA_Z_IMAG(X[0]));
}
static void gglse() {
  magma_int_t m = 1, n = 2, p = 1; std::vector<Z> A = {C(1), C(1)}, B = {C(0), C(0)}, c = {C(1)}, d = {C(1)}, x(2);
  { std::vector<Z> A1 = A, B1 = B, c1 = c, d1 = d, x1(2), wk(64); magma_int_t lw = 64, info; lapackf77_zgglse(&m, &n, &p, A1.data(), &m, B1.data(), &p, c1.data(), d1.data(), x1.data(), wk.data(), &lw, &info);
    printf("LAPACK zgglse, B = [0 0]: info=%lld\n", (long long)info); }
  magma_int_t info = -999, lwork = -1; Z q; std::vector<Z> A1 = A, B1 = B, c1 = c, d1 = d;
  magma_zgglse(m, n, p, A1.data(), m, B1.data(), p, c1.data(), d1.data(), x.data(), &q, lwork, &info); lwork = (magma_int_t)MAGMA_Z_REAL(q) + 64;
  std::vector<Z> wk(lwork); info = -999; magma_zgglse(m, n, p, A1.data(), m, B1.data(), p, c1.data(), d1.data(), x.data(), wk.data(), lwork, &info);
  printf("magma_zgglse, B = [0 0]: info=%lld  x = (%g,%g) (%g,%g)\n", (long long)info, MAGMA_Z_REAL(x[0]), MAGMA_Z_IMAG(x[0]), MAGMA_Z_REAL(x[1]), MAGMA_Z_IMAG(x[1]));
}
static void ggrqf() {
  int n = 200; std::vector<Z> A((size_t)n * n), B((size_t)n * n), ta(n), tb(n); for (auto& a : A) a = C(rnd()); for (auto& b : B) b = C(rnd());
  std::vector<Z> A0 = A, B0 = B, wk(n); magma_int_t info = -999;
  magma_zggrqf(n, n, n, A.data(), n, ta.data(), B.data(), n, tb.data(), wk.data(), n, &info);
  int chA = 0, chB = 0; for (size_t i = 0; i < A.size(); i++) { if (!MAGMA_Z_EQUAL(A[i], A0[i])) chA++; if (!MAGMA_Z_EQUAL(B[i], B0[i])) chB++; }
  printf("zggrqf m=p=n=200 with lwork = 200 (documented minimum max(m,p,n)): info=%lld; entries modified: A %d, B %d\n", (long long)info, chA, chB);
}
static void mscale_gen() {
  int n = 4; std::vector<magma_index_t> row = {0, 2, 5, 8, 10}, col = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3}; std::vector<Z> val = {C(4), C(-1), C(-1), C(4), C(-1), C(-1), C(4), C(-1), C(-1), C(4)};
  magma_z_matrix h = {Magma_CSR}, d = {Magma_CSR}; magma_zcsrset(n, n, row.data(), col.data(), val.data(), &h, Q); magma_zmtransfer(h, &d, Magma_CPU, Magma_DEV, Q);
  magma_z_matrix sf = {Magma_CSR}; magma_zvinit(&sf, Magma_CPU, n, 1, C(0), Q);
  magma_scale_t sc = Magma_UNITCOL; magma_side_t sd = MagmaLeft;
  printf("magma_zmscale_generate(UNITCOL, Left) on a 4x4 device CSR ...\n"); fflush(stdout);
  magma_int_t r = magma_zmscale_generate(1, &sc, &sd, &d, &sf, Q);
  printf("  ret=%lld  factors = %g %g %g %g\n", (long long)r, MAGMA_Z_REAL(sf.val[0]), MAGMA_Z_REAL(sf.val[1]), MAGMA_Z_REAL(sf.val[2]), MAGMA_Z_REAL(sf.val[3]));
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : "";
  magma_init(); magma_queue_create(0, &Q);
  if (m == "rangev") rangev(); else if (m == "eig_nan") eig_nan(); else if (m == "unmtr_m") unmtr_m(); else if (m == "he2hb") he2hb();
  else if (m == "geqrf_m0") geqrf_m0(); else if (m == "cgeqrsv") cgeqrsv(); else if (m == "gglse") gglse(); else if (m == "ggrqf") ggrqf();
  else if (m == "mscale_gen") mscale_gen();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

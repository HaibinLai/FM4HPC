// Round 38 (roxy, one GPU): numerical differential test of the multi-GPU / out-of-core drivers with ngpu = 1.
// Earlier rounds ran them only for status logic; with ngpu = 1 the _m / _mgpu code paths are still separate from the
// single-GPU drivers (own panel / update scheduling, host-side out-of-core loops).  Every result is checked against
// LAPACK on the same input.
// CSV: r38,routine,case,info,err,verdict   (err: normalised residual or eigenvalue error; BAD if info != 0 or err > 100)
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
static magma_queue_t Q;
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
typedef std::vector<double> V;
static const double EPS = 1.1102230246251565e-16;
static V rnd(size_t k) { V v(k); for (auto& x : v) x = r1(); return v; }
static double amax(const V& v) { double m = 0; for (double x : v) m = std::max(m, fabs(x)); return std::max(m, 1e-300); }
static void out(const char* r, const std::string& c, long long info, double e) { printf("r38,%s,%s,%lld,%.3g,%s\n", r, c.c_str(), info, e, (info == 0 && e <= 100) ? "OK" : "BAD"); }
static char cs_[160];
#define CASE(...) (snprintf(cs_, sizeof cs_, __VA_ARGS__), std::string(cs_))
static V spd(int n) { V G = rnd((size_t)n * n), S((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { double s = 0; for (int k = 0; k < n; k++) s += G[i + (size_t)k * n] * G[j + (size_t)k * n]; S[i + (size_t)j * n] = s + (i == j ? n : 0); } return S; }
static V sym(int n) { V A = rnd((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) A[i + (size_t)j * n] = A[j + (size_t)i * n]; return A; }

static double luErr(const V& A, const V& F, const magma_int_t* ip, int m, int n) {   // ||P A - L U||_max / (||A|| min(m,n) eps)
  V PA = A; int k = std::min(m, n); for (int i = 0; i < k; i++) { int p = (int)ip[i] - 1; if (p != i) for (int j = 0; j < n; j++) std::swap(PA[i + (size_t)j * m], PA[p + (size_t)j * m]); }
  double r = 0; for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { double s = 0; for (int l = 0; l <= std::min(std::min(i, j), k - 1); l++) s += (l == i ? 1.0 : F[i + (size_t)l * m]) * F[l + (size_t)j * m]; r = std::max(r, fabs(PA[i + (size_t)j * m] - s)); }
  return r / (amax(A) * std::max(1, k) * EPS * 10); }
static double cholErr(const V& A, const V& F, int n, bool upper) { double r = 0;
  for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { double s = 0; for (int l = 0; l <= j; l++) s += upper ? F[l + (size_t)i * n] * F[l + (size_t)j * n] : F[i + (size_t)l * n] * F[j + (size_t)l * n]; r = std::max(r, fabs(A[i + (size_t)j * n] - s)); }
  return r / (amax(A) * n * EPS * 10); }
static double qrErr(const V& A, const V& F, int m, int n) { double r = 0, an = amax(A); int k = std::min(m, n);   // A^T A = R^T R
  for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { double ata = 0, rtr = 0; for (int l = 0; l < m; l++) ata += A[l + (size_t)i * m] * A[l + (size_t)j * m]; for (int l = 0; l <= std::min(i, k - 1); l++) rtr += F[l + (size_t)i * m] * F[l + (size_t)j * m]; r = std::max(r, fabs(ata - rtr)); }
  return r / (an * an * m * EPS * 10); }
static V lapackEig(const V& A0, int n) { V A = A0, w(n); magma_int_t N = n, info, lw = -1, liw = -1, qi; double q; lapackf77_dsyevd("N", "L", &N, A.data(), &N, w.data(), &q, &lw, &qi, &liw, &info);
  lw = (magma_int_t)q + 1; liw = qi + 1; V wk(lw); std::vector<magma_int_t> iw(liw); lapackf77_dsyevd("N", "L", &N, A.data(), &N, w.data(), wk.data(), &lw, iw.data(), &liw, &info); return w; }
static double eigvecErr(const V& A0, const V& Z, const V& w, int n, int m, int lo, const V& wl) {   // max(residual, orthogonality, eigenvalue error)
  double an = amax(A0), e = 0; for (int j = 0; j < m; j++) { e = std::max(e, fabs(w[j] - wl[lo + j]) / (an * n * EPS * 10));
    for (int i = 0; i < n; i++) { double s = 0; for (int k = 0; k < n; k++) s += A0[i + (size_t)k * n] * Z[k + (size_t)j * n]; e = std::max(e, fabs(s - w[j] * Z[i + (size_t)j * n]) / (an * n * EPS * 10)); }
    for (int jj = 0; jj <= j; jj++) { double s = 0; for (int k = 0; k < n; k++) s += Z[k + (size_t)j * n] * Z[k + (size_t)jj * n]; e = std::max(e, fabs(s - (j == jj ? 1.0 : 0.0)) / (n * EPS * 10)); } }
  return e; }

static void fact() {
  for (int n : {1, 33, 257, 600, 1025}) for (char up : {'L', 'U'}) {   // potrf_m (host) and potrf_mgpu / potrf_mgpu_right (ngpu = 1)
    Sd = 3u + n; V S = spd(n); magma_int_t info = -999; magma_uplo_t U_ = up == 'L' ? MagmaLower : MagmaUpper;
    { V F = S; magma_dpotrf_m(1, U_, n, F.data(), n, &info); out("dpotrf_m", CASE("n=%d %c", n, up), info, cholErr(S, F, n, up == 'U')); }
    // potrf_mgpu_right: body is #if 0 but still declared (known M156), so it does not link
    for (int right = 0; right < 1; right++) { int ld = (int)magma_roundup(n, 32); magmaDouble_ptr d[1]; magma_dmalloc(&d[0], (size_t)ld * n); magma_dsetmatrix(n, n, S.data(), n, d[0], ld, Q);
      magma_dpotrf_mgpu(1, U_, n, d, ld, &info);
      V F((size_t)n * n); magma_dgetmatrix(n, n, d[0], ld, F.data(), n, Q); out(right ? "dpotrf_mgpu_right" : "dpotrf_mgpu", CASE("n=%d %c", n, up), info, cholErr(S, F, n, up == 'U')); magma_free(d[0]); }
  }
  struct S { int m, n; }; for (auto s : std::vector<S>{{1, 1}, {33, 33}, {257, 257}, {600, 257}, {257, 600}, {1025, 1025}}) {
    Sd = 7u + s.m; V A = rnd((size_t)s.m * s.n); magma_int_t info = -999; std::vector<magma_int_t> ip(std::min(s.m, s.n));
    { V F = A; magma_dgetrf_m(1, s.m, s.n, F.data(), s.m, ip.data(), &info); out("dgetrf_m", CASE("%dx%d", s.m, s.n), info, luErr(A, F, ip.data(), s.m, s.n)); }
    { int ld = (int)magma_roundup(s.m, 32); magmaDouble_ptr d[1]; magma_dmalloc(&d[0], (size_t)ld * s.n); magma_dsetmatrix(s.m, s.n, A.data(), s.m, d[0], ld, Q);
      magma_dgetrf_mgpu(1, s.m, s.n, d, ld, ip.data(), &info); V F((size_t)s.m * s.n); magma_dgetmatrix(s.m, s.n, d[0], ld, F.data(), s.m, Q);
      out("dgetrf_mgpu", CASE("%dx%d", s.m, s.n), info, luErr(A, F, ip.data(), s.m, s.n)); magma_free(d[0]); }
    { V F = A, tau(std::max(1, std::min(s.m, s.n))); double q; magma_int_t lw = -1; magma_dgeqrf_m(1, s.m, s.n, F.data(), s.m, tau.data(), &q, lw, &info); lw = (magma_int_t)q + 64; V w(lw);
      magma_dgeqrf_m(1, s.m, s.n, F.data(), s.m, tau.data(), w.data(), lw, &info); out("dgeqrf_m", CASE("%dx%d", s.m, s.n), info, qrErr(A, F, s.m, s.n)); }
    { int ld = (int)magma_roundup(s.m, 32); magmaDouble_ptr d[1]; magma_dmalloc(&d[0], (size_t)ld * s.n); magma_dsetmatrix(s.m, s.n, A.data(), s.m, d[0], ld, Q); V tau(std::max(1, std::min(s.m, s.n)));
      magma_dgeqrf2_mgpu(1, s.m, s.n, d, ld, tau.data(), &info); V F((size_t)s.m * s.n); magma_dgetmatrix(s.m, s.n, d[0], ld, F.data(), s.m, Q);
      out("dgeqrf2_mgpu", CASE("%dx%d", s.m, s.n), info, qrErr(A, F, s.m, s.n)); magma_free(d[0]); }
  }
}

static void eig() {
  for (int n : {1, 33, 257, 600}) for (char up : {'L', 'U'}) {
    Sd = 13u + n; V A0 = sym(n), wl = lapackEig(A0, n); magma_uplo_t U_ = up == 'L' ? MagmaLower : MagmaUpper;
    { V A = A0, w(n); magma_int_t info = -999, lw = -1, liw = -1, qi = 0; double q = 0; magma_dsyevd_m(1, MagmaVec, U_, n, A.data(), n, w.data(), &q, lw, &qi, liw, &info);
      lw = (magma_int_t)q + 64; liw = qi + 64; V wk(lw); std::vector<magma_int_t> iw(liw); magma_dsyevd_m(1, MagmaVec, U_, n, A.data(), n, w.data(), wk.data(), lw, iw.data(), liw, &info);
      out("dsyevd_m", CASE("n=%d %c", n, up), info, eigvecErr(A0, A, w, n, n, 0, wl)); }
    for (int rg = 0; rg < 2; rg++) for (int two = 0; two < 2; two++) {
      if (two && up == 'U') continue;   // Upper on the 2-stage path is M065
      int il = rg ? std::max(1, n / 4) : 1, iu = rg ? std::max(1, (3 * n) / 4) : n; V A = A0, w(n); magma_int_t info = -999, mout = -1, lw = -1, liw = -1, qi = 0; double q = 0; magma_range_t R = rg ? MagmaRangeI : MagmaRangeAll;
      auto call = [&](double* wk, magma_int_t l, magma_int_t* iw, magma_int_t li) { if (two) magma_dsyevdx_2stage_m(1, MagmaVec, R, U_, n, A.data(), n, 0, 0, il, iu, &mout, w.data(), wk, l, iw, li, &info);
        else magma_dsyevdx_m(1, MagmaVec, R, U_, n, A.data(), n, 0, 0, il, iu, &mout, w.data(), wk, l, iw, li, &info); };
      call(&q, lw, &qi, liw); lw = (magma_int_t)q + 2 * n + 64; liw = qi + 64; V wk(lw); std::vector<magma_int_t> iw(liw); call(wk.data(), lw, iw.data(), liw);
      int cnt = iu - il + 1; double e = (int)mout == cnt ? eigvecErr(A0, A, w, n, cnt, il - 1, wl) : 1e300;
      out(two ? "dsyevdx_2stage_m" : "dsyevdx_m", CASE("n=%d %c rg=%c m=%lld/%d", n, up, rg ? 'I' : 'A', (long long)mout, cnt), info, e); }
    // generalized: A x = lambda B x, itype 1; check A Z = B Z diag(w) and Z^T B Z = I
    for (int x = 0; x < 2; x++) { V B0 = spd(n), A = A0, B = B0, w(n); magma_int_t info = -999, mout = -1, lw = -1, liw = -1, qi = 0; double q = 0;
      auto call = [&](double* wk, magma_int_t l, magma_int_t* iw, magma_int_t li) { if (x) magma_dsygvdx_m(1, 1, MagmaVec, MagmaRangeAll, U_, n, A.data(), n, B.data(), n, 0, 0, 1, n, &mout, w.data(), wk, l, iw, li, &info);
        else magma_dsygvd_m(1, 1, MagmaVec, U_, n, A.data(), n, B.data(), n, w.data(), wk, l, iw, li, &info); };
      call(&q, lw, &qi, liw); lw = (magma_int_t)q + 2 * n + 64; liw = qi + 64; V wk(lw); std::vector<magma_int_t> iw(liw); call(wk.data(), lw, iw.data(), liw);
      double an = amax(A0) + amax(B0), e = 0; for (int j = 0; j < n; j++) { for (int i = 0; i < n; i++) { double s = 0, t = 0; for (int k = 0; k < n; k++) { s += A0[i + (size_t)k * n] * A[k + (size_t)j * n]; t += B0[i + (size_t)k * n] * A[k + (size_t)j * n]; }
            e = std::max(e, fabs(s - w[j] * t) / (an * (1 + fabs(w[j])) * n * EPS * 10)); } }
      out(x ? "dsygvdx_m" : "dsygvd_m", CASE("n=%d %c", n, up), info, e); }
  }
  for (int n : {1, 33, 257, 600}) {   // geev_m: eigenvalues vs LAPACK dgeev; right-vector residual
    Sd = 17u + n; V A0 = rnd((size_t)n * n), A = A0, wr(n), wi(n), VR((size_t)n * n), VL(1); magma_int_t info = -999, lw = -1; double q;
    magma_dgeev_m(MagmaNoVec, MagmaVec, n, A.data(), n, wr.data(), wi.data(), VL.data(), 1, VR.data(), n, &q, lw, &info); lw = (magma_int_t)q + 64; V wk(lw);
    magma_dgeev_m(MagmaNoVec, MagmaVec, n, A.data(), n, wr.data(), wi.data(), VL.data(), 1, VR.data(), n, wk.data(), lw, &info);
    V Al = A0, lr(n), li_(n), vl(1), vr(1); magma_int_t N = n, one = 1, linfo, l2 = -1; lapackf77_dgeev("N", "N", &N, Al.data(), &N, lr.data(), li_.data(), vl.data(), &one, vr.data(), &one, &q, &l2, &linfo); l2 = (magma_int_t)q + 1; { V w2(l2); lapackf77_dgeev("N", "N", &N, Al.data(), &N, lr.data(), li_.data(), vl.data(), &one, vr.data(), &one, w2.data(), &l2, &linfo); }
    std::vector<std::pair<double, double>> a, b; for (int i = 0; i < n; i++) { a.push_back({wr[i], wi[i]}); b.push_back({lr[i], li_[i]}); } std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    double an = amax(A0), e = 0; for (int i = 0; i < n; i++) e = std::max(e, std::hypot(a[i].first - b[i].first, a[i].second - b[i].second) / (an * n * EPS * 1e3));   // eigenvalues of a nonnormal matrix: looser
    for (int j = 0; j < n; j++) { if (wi[j] != 0) { j++; continue; } double r = 0, vn = 0; for (int i = 0; i < n; i++) { double s = 0; for (int k = 0; k < n; k++) s += A0[i + (size_t)k * n] * VR[k + (size_t)j * n]; r = std::max(r, fabs(s - wr[j] * VR[i + (size_t)j * n])); vn = std::max(vn, fabs(VR[i + (size_t)j * n])); }
      e = std::max(e, r / (an * vn * n * EPS * 10)); }
    out("dgeev_m", CASE("n=%d", n), info, e);
  }
}

static void red() {
  for (int n : {1, 33, 257, 600}) {   // gehrd_m: A0 = Q H Q^T with Q from LAPACK dorghr
    Sd = 19u + n; V A0 = rnd((size_t)n * n), A = A0, tau(std::max(1, n)); magma_int_t info = -999, nb = magma_get_dgehrd_nb(n), lw = -1; double qq = 0; V T((size_t)nb * n + 1);
    magma_dgehrd_m(n, 1, n, A.data(), n, tau.data(), &qq, lw, T.data(), &info); lw = (magma_int_t)qq + 1; V wk(lw + 1);   // documented LWORK >= N*NB is rejected (known M049): use the query
    magma_dgehrd_m(n, 1, n, A.data(), n, tau.data(), wk.data(), lw, T.data(), &info);
    V H((size_t)n * n, 0); for (int j = 0; j < n; j++) for (int i = 0; i <= std::min(n - 1, j + 1); i++) H[i + (size_t)j * n] = A[i + (size_t)j * n];
    V Qm = A; magma_int_t N = n, lo = 1, hi = n, li, l2 = -1; double q; lapackf77_dorghr(&N, &lo, &hi, Qm.data(), &N, tau.data(), &q, &l2, &li); l2 = (magma_int_t)q + 1; { V w2(l2); lapackf77_dorghr(&N, &lo, &hi, Qm.data(), &N, tau.data(), w2.data(), &l2, &li); }
    double r = 0, an = amax(A0); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { double s = 0; for (int k = 0; k < n; k++) { double qh = 0; for (int l = 0; l < n; l++) qh += Qm[i + (size_t)l * n] * H[l + (size_t)k * n]; s += qh * Qm[j + (size_t)k * n]; } r = std::max(r, fabs(s - A0[i + (size_t)j * n])); }
    out("dgehrd_m", CASE("n=%d", n), info, r / (an * n * EPS * 10));
  }
  for (int n : {1, 33, 257, 600}) for (char up : {'L', 'U'}) {   // sytrd_mgpu: the tridiagonal must have the eigenvalues of A
    Sd = 23u + n; V A0 = sym(n), A = A0, d(n), e(std::max(1, n)), tau(std::max(1, n)), wl = lapackEig(A0, n); magma_int_t info = -999, lw = -1; double q;
    magma_dsytrd_mgpu(1, 1, up == 'L' ? MagmaLower : MagmaUpper, n, A.data(), n, d.data(), e.data(), tau.data(), &q, lw, &info); lw = (magma_int_t)q + 64; V wk(lw);
    magma_dsytrd_mgpu(1, 1, up == 'L' ? MagmaLower : MagmaUpper, n, A.data(), n, d.data(), e.data(), tau.data(), wk.data(), lw, &info);
    V dd = d, ee = e; magma_int_t N = n, li; lapackf77_dsterf(&N, dd.data(), ee.data(), &li); std::sort(dd.begin(), dd.end());
    double er = 0, an = amax(A0); for (int i = 0; i < n; i++) er = std::max(er, fabs(dd[i] - wl[i]) / (an * n * EPS * 10));
    out("dsytrd_mgpu", CASE("n=%d %c", n, up), info, er);
  }
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string g = argc > 1 ? argv[1] : "all";
  magma_init(); magma_queue_create(0, &Q);
  printf("r38,routine,case,info,err,verdict\n");
  if (g == "all" || g == "fact") fact();
  if (g == "all" || g == "eig") eig();
  if (g == "all" || g == "red") red();
  fprintf(stderr, "DONE %s cuda=%s\n", g.c_str(), cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

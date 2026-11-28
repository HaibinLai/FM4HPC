// Round 37 (roxy): the two-stage tridiagonalisation chain (sy2sb -> hetrd_hb2st bulge chasing -> bulge_back / applyQ
// back-transformation) through magma_dsyevdx_2stage, and the random butterfly helper gerbt through magma_dgesv_rbt.
//   eig  dsyevdx_2stage: n across the 1-stage/2-stage switch and the bulge nb; spectra random / clustered / repeated /
//        zero / diagonal / tridiagonal; range A, I, V; uplo L, U.  Checks: mout, eigenvalues vs LAPACK dsyevd,
//        ||A Z - Z diag(w)|| / (||A|| n eps), ||Z^T Z - I|| / (n eps), info.
//   rbt  dgesv_rbt(refine false/true): n not a multiple of 4 or of the block size; normwise backward error.
// CSV: r37,routine,case,info,metric1,metric2,metric3,verdict
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

static V orth(int n) { V X((size_t)n * n); for (auto& x : X) x = r1(); magma_int_t nn = n, lw = 64 * n + 64, il; V tau(n), w(lw);
  lapackf77_dgeqrf(&nn, &nn, X.data(), &nn, tau.data(), w.data(), &lw, &il); lapackf77_dorgqr(&nn, &nn, &nn, X.data(), &nn, tau.data(), w.data(), &lw, &il); return X; }
static V gen(int n, int kind) {   // symmetric test matrices
  V A((size_t)n * n, 0);
  if (kind == 3) return A;                                                          // zero
  if (kind == 4) { for (int i = 0; i < n; i++) A[i + (size_t)i * n] = (i % 7) - 3.0; return A; }   // diagonal with repeats
  if (kind == 5) { for (int i = 0; i < n; i++) { A[i + (size_t)i * n] = 2; if (i + 1 < n) { A[i + 1 + (size_t)i * n] = -1; A[i + (size_t)(i + 1) * n] = -1; } } return A; }   // tridiagonal
  V U = orth(n), lam(n); for (int i = 0; i < n; i++) lam[i] = kind == 0 ? r1() * 10 : kind == 1 ? 1.0 + 1e-10 * i : (i < n / 2 ? 1.0 : 2.0);   // random / clustered / two values
  for (int l = 0; l < n; l++) for (int j = 0; j < n; j++) { double v = lam[l] * U[j + (size_t)l * n]; for (int i = 0; i < n; i++) A[i + (size_t)j * n] += U[i + (size_t)l * n] * v; }
  for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) { double a = 0.5 * (A[i + (size_t)j * n] + A[j + (size_t)i * n]); A[i + (size_t)j * n] = a; A[j + (size_t)i * n] = a; }
  return A; }

static void eig() {
  const char* kn[] = {"random", "clustered", "twovalues", "zero", "diagonal", "tridiag"};
  for (int n : {1, 2, 63, 64, 65, 127, 128, 129, 200, 255, 256, 257, 511, 513, 1000}) for (int kind = 0; kind < 6; kind++) {
    if (n >= 511 && kind != 0 && kind != 2) continue;
    Sd = 3u + n + 11 * kind; V A0 = gen(n, kind);
    V wl(n), Al = A0; { magma_int_t N = n, info, lw = -1, liw = -1, qi; double q; lapackf77_dsyevd("N", "L", &N, Al.data(), &N, wl.data(), &q, &lw, &qi, &liw, &info);
      lw = (magma_int_t)q + 1; liw = qi + 1; V w(lw); std::vector<magma_int_t> iw(liw); lapackf77_dsyevd("N", "L", &N, Al.data(), &N, wl.data(), w.data(), &lw, iw.data(), &liw, &info); }
    for (int rg = 0; rg < 3; rg++) for (char up : {'L', 'U'}) {
      if (n >= 511 && up == 'U') continue;
      int il = 1, iu = n; double vl = 0, vu = 0;
      if (rg == 1) { il = std::max(1, n / 4); iu = std::max(il, (3 * n) / 4); }
      if (rg == 2) { int a = std::max(0, n / 3 - 1), b = std::min(n - 1, (2 * n) / 3); vl = 0.5 * (wl[a] + (a > 0 ? wl[a - 1] : wl[a] - 1)); vu = 0.5 * (wl[b] + (b + 1 < n ? wl[b + 1] : wl[b] + 1)); if (!(vl < vu)) { vl = wl[0] - 1; vu = wl[n - 1] + 1; } }
      V A = A0, w(n); magma_int_t N = n, mout = -1, info = -999, lw = -1, liw = -1, qi = 0; double q = 0;
      magma_range_t R = rg == 0 ? MagmaRangeAll : rg == 1 ? MagmaRangeI : MagmaRangeV; magma_uplo_t U_ = up == 'L' ? MagmaLower : MagmaUpper;
      magma_dsyevdx_2stage(MagmaVec, R, U_, N, A.data(), N, vl, vu, il, iu, &mout, w.data(), &q, lw, &qi, liw, &info);
      lw = (magma_int_t)q + 2 * n + 64; liw = qi + 64; V work(lw); std::vector<magma_int_t> iw(liw);
      magma_dsyevdx_2stage(MagmaVec, R, U_, N, A.data(), N, vl, vu, il, iu, &mout, w.data(), work.data(), lw, iw.data(), liw, &info);
      // expected eigenvalue window
      int lo = 0, cnt = n; if (rg == 1) { lo = il - 1; cnt = iu - il + 1; } if (rg == 2) { lo = 0; while (lo < n && !(wl[lo] > vl)) lo++; int hi = lo; while (hi < n && wl[hi] <= vu) hi++; cnt = hi - lo; }
      double ew = 0, an = 0; for (double x : A0) an = std::max(an, fabs(x)); an = std::max(an, 1e-300);
      int m = (int)mout; for (int j = 0; j < std::min(m, cnt); j++) ew = std::max(ew, fabs(w[j] - wl[lo + j]) / (an * n * EPS));
      double res = 0, orth_ = 0;
      for (int j = 0; j < m && j < n; j++) { for (int i = 0; i < n; i++) { double s = 0; for (int k = 0; k < n; k++) s += A0[i + (size_t)k * n] * A[k + (size_t)j * n]; res = std::max(res, fabs(s - w[j] * A[i + (size_t)j * n]) / (an * n * EPS)); }
        for (int jj = 0; jj <= j; jj++) { double s = 0; for (int k = 0; k < n; k++) s += A[k + (size_t)j * n] * A[k + (size_t)jj * n]; orth_ = std::max(orth_, fabs(s - (j == jj ? 1.0 : 0.0)) / (n * EPS)); } }
      bool ok = info == 0 && m == cnt && ew <= 100 && res <= 100 && orth_ <= 100;
      printf("r37,dsyevdx_2stage,n=%d %s rg=%c%c m=%d/%d,%lld,%.3g,%.3g,%.3g,%s\n", n, kn[kind], "AIV"[rg], up, m, cnt, (long long)info, ew, res, orth_, ok ? "OK" : "BAD");
    }
  }
}

static void rbt() {
  for (int n : {1, 2, 3, 5, 7, 31, 33, 63, 65, 127, 129, 255, 513, 1001}) for (int ref = 0; ref < 2; ref++) for (int nrhs : {1, 3}) {
    Sd = 9u + n; V A((size_t)n * n), X((size_t)n * nrhs), B((size_t)n * nrhs, 0); for (auto& a : A) a = r1(); for (int i = 0; i < n; i++) A[i + (size_t)i * n] += 0.5 * sqrt((double)n); for (auto& x : X) x = r1();
    for (int c = 0; c < nrhs; c++) for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) B[i + (size_t)c * n] += A[i + (size_t)j * n] * X[j + (size_t)c * n];
    V Ac = A, Bc = B; magma_int_t info = -999; magma_dgesv_rbt(ref ? MagmaTrue : MagmaFalse, n, nrhs, Ac.data(), n, Bc.data(), n, &info);
    double an = 0; for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += fabs(A[i + (size_t)j * n]); an = std::max(an, s); }
    double w = 0, fe = 0; for (int c = 0; c < nrhs; c++) { double rn = 0, xn = 0, bn = 0; for (int i = 0; i < n; i++) { double s = B[i + (size_t)c * n]; for (int j = 0; j < n; j++) s -= A[i + (size_t)j * n] * Bc[j + (size_t)c * n];
        if (!(fabs(s) <= rn)) rn = fabs(s); xn = std::max(xn, fabs(Bc[i + (size_t)c * n])); bn = std::max(bn, fabs(B[i + (size_t)c * n])); fe = std::max(fe, fabs(Bc[i + (size_t)c * n] - X[i + (size_t)c * n])); }
      double e = rn / ((an * xn + bn) * n * EPS); if (!(e <= w)) w = e; }
    printf("r37,dgesv_rbt,n=%d refine=%d nrhs=%d,%lld,%.3g,%.3g,0,%s\n", n, ref, nrhs, (long long)info, w, fe, (info == 0 && w <= (ref ? 100 : 1e6)) ? "OK" : "BAD");
  }
}

// isolate: gesv_nopiv_gpu alone, and gerbt_gpu + nopiv + prbt_mv by hand, for small n
static void rbtiso() {
  for (int n : {1, 2, 3, 4, 5, 8}) {
    Sd = 9u + n; V A((size_t)n * n), b(n), x(n); for (auto& a : A) a = r1(); for (int i = 0; i < n; i++) A[i + (size_t)i * n] += 0.5 * sqrt((double)n) + 1; for (auto& v : x) v = r1();
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) b[i] += A[i + (size_t)j * n] * x[j];
    magmaDouble_ptr dA, dB; magma_dmalloc(&dA, (size_t)n * n); magma_dmalloc(&dB, n); magma_dsetmatrix(n, n, A.data(), n, dA, n, Q); magma_dsetmatrix(n, 1, b.data(), n, dB, n, Q);
    magma_int_t info; magma_dgesv_nopiv_gpu(n, 1, dA, n, dB, n, &info); V y(n); magma_dgetmatrix(n, 1, dB, n, y.data(), n, Q);
    double fe = 0; for (int i = 0; i < n; i++) fe = std::max(fe, fabs(y[i] - x[i]));
    // gerbt on the device, then an explicit host check: A' = U^T A V and b' = U^T b must give the same solution after x = V y
    magma_dsetmatrix(n, n, A.data(), n, dA, n, Q); magma_dsetmatrix(n, 1, b.data(), n, dB, n, Q); V hu(2 * n), hv(2 * n);
    magma_dgerbt_gpu(MagmaTrue, n, 1, dA, n, dB, n, hu.data(), hv.data(), &info); V Ap((size_t)n * n), bp(n); magma_dgetmatrix(n, n, dA, n, Ap.data(), n, Q); magma_dgetmatrix(n, 1, dB, n, bp.data(), n, Q);
    // solve A' y = b' on the host (LAPACK), then x = V y with the library's own prbt_mv
    V Al = Ap, yl = bp; std::vector<magma_int_t> ip(n); magma_int_t N = n, one = 1, li; lapackf77_dgesv(&N, &one, Al.data(), &N, ip.data(), yl.data(), &N, &li);
    magmaDouble_ptr dv; magma_dmalloc(&dv, 2 * n); magma_dsetvector(2 * n, hv.data(), 1, dv, 1, Q); magma_dsetmatrix(n, 1, yl.data(), n, dB, n, Q); magmablas_dprbt_mv(n, 1, dv, dB, n, Q); V xr(n); magma_dgetmatrix(n, 1, dB, n, xr.data(), n, Q);
    double fe2 = 0; for (int i = 0; i < n; i++) fe2 = std::max(fe2, fabs(xr[i] - x[i]));
    printf("r37,rbtiso,n=%d,0,gesv_nopiv_ferr=%.3g,rbt_roundtrip_ferr=%.3g,0,%s\n", n, fe, fe2, (fe < 1e-12 && fe2 < 1e-12) ? "OK" : "BAD");
    magma_free(dA); magma_free(dB); magma_free(dv); }
}
// range V at n = 256: 1-stage dsyevdx vs dsyevdx_2stage on tridiag / clustered, plain and rescaled (stedx scales by max|T|)
static void v256() {
  const char* kn[] = {"tridiag", "tridiag*0.5", "clustered", "clustered*3", "random"};
  for (int n : {200, 256, 257}) for (int kk = 0; kk < 5; kk++) for (int two = 0; two < 2; two++) {
    Sd = 3u + n + 11 * (kk == 0 || kk == 1 ? 5 : kk == 4 ? 0 : 1); V A0 = gen(n, kk <= 1 ? 5 : kk == 4 ? 0 : 1); double sc = kk == 1 ? 0.5 : kk == 3 ? 3.0 : 1.0; for (auto& a : A0) a *= sc;
    V wl(n), Al = A0; { magma_int_t N = n, info, lw = -1, liw = -1, qi; double q; lapackf77_dsyevd("N", "L", &N, Al.data(), &N, wl.data(), &q, &lw, &qi, &liw, &info);
      lw = (magma_int_t)q + 1; liw = qi + 1; V w(lw); std::vector<magma_int_t> iw(liw); lapackf77_dsyevd("N", "L", &N, Al.data(), &N, wl.data(), w.data(), &lw, iw.data(), &liw, &info); }
    int a = n / 3 - 1, b = (2 * n) / 3; double vl = 0.5 * (wl[a] + wl[a - 1]), vu = 0.5 * (wl[b] + wl[b + 1]); int lo = a, cnt = b - a + 1;
    V A = A0, w(n); magma_int_t N = n, mout = -1, info = -999, lw = -1, liw = -1, qi = 0; double q = 0;
    if (two) magma_dsyevdx_2stage(MagmaVec, MagmaRangeV, MagmaLower, N, A.data(), N, vl, vu, 1, n, &mout, w.data(), &q, lw, &qi, liw, &info);
    else magma_dsyevdx(MagmaVec, MagmaRangeV, MagmaLower, N, A.data(), N, vl, vu, 1, n, &mout, w.data(), &q, lw, &qi, liw, &info);
    lw = (magma_int_t)q + 2 * n + 64; liw = qi + 64; V work(lw); std::vector<magma_int_t> iw(liw);
    if (two) magma_dsyevdx_2stage(MagmaVec, MagmaRangeV, MagmaLower, N, A.data(), N, vl, vu, 1, n, &mout, w.data(), work.data(), lw, iw.data(), liw, &info);
    else magma_dsyevdx(MagmaVec, MagmaRangeV, MagmaLower, N, A.data(), N, vl, vu, 1, n, &mout, w.data(), work.data(), lw, iw.data(), liw, &info);
    double an = 0; for (double x : A0) an = std::max(an, fabs(x)); double res = 0, orth_ = 0; int m = (int)mout;
    for (int j = 0; j < m; j++) { for (int i = 0; i < n; i++) { double s2 = 0; for (int k = 0; k < n; k++) s2 += A0[i + (size_t)k * n] * A[k + (size_t)j * n]; res = std::max(res, fabs(s2 - w[j] * A[i + (size_t)j * n]) / (an * n * EPS)); }
      for (int jj = 0; jj <= j; jj++) { double s2 = 0; for (int k = 0; k < n; k++) s2 += A[k + (size_t)j * n] * A[k + (size_t)jj * n]; orth_ = std::max(orth_, fabs(s2 - (j == jj ? 1.0 : 0.0)) / (n * EPS)); } }
    printf("r37,v256,%s n=%d %s m=%d/%d,%lld,0,%.3g,%.3g,%s\n", two ? "2stage" : "1stage", n, kn[kk], m, cnt, (long long)info, res, orth_, (m == cnt && res < 100 && orth_ < 100) ? "OK" : "BAD");
  }
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string g = argc > 1 ? argv[1] : "all";
  magma_init(); magma_queue_create(0, &Q);
  printf("r37,routine,case,info,eigerr_or_berr,resid_or_ferr,orth,verdict\n");
  if (g == "all" || g == "eig") eig();
  if (g == "all" || g == "rbt") rbt();
  if (g == "rbtiso") rbtiso();
  if (g == "v256") v256();
  fprintf(stderr, "DONE %s cuda=%s\n", g.c_str(), cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

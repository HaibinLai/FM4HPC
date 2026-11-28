// Twenty-fifth GPU round: the multi-GPU drivers with ngpu = 2 (GPUs 0 and 1, authorised 2026-11-26), each compared with
// ngpu = 1 and with LAPACK (HANDOFF §29.27).  Run with CUDA_VISIBLE_DEVICES=0,1.
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
typedef magmaDoubleComplex Z;
extern "C" void zhegv_(const magma_int_t*, const char*, const char*, const magma_int_t*, Z*, const magma_int_t*, Z*, const magma_int_t*,
                       double*, Z*, const magma_int_t*, double*, magma_int_t*);
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 2026u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static std::vector<Z> randm(int m, int n) { std::vector<Z> A((size_t)m * n); for (auto& a : A) a = C(rnd(), rnd()); return A; }
static std::vector<Z> herm(int n) { std::vector<Z> A((size_t)n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { Z v = i == j ? C(rnd()) : C(rnd(), rnd());
  A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = MAGMA_Z_CONJ(v); } return A; }
static std::vector<Z> hpd(int n) { auto A = herm(n); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = MAGMA_Z_ADD(A[i + (size_t)i * n], C(n)); return A; }
static double maxrel(const std::vector<Z>& a, const std::vector<Z>& b, bool lowerOnly = false, int n = 0) { double e = 0, m = 0;
  for (size_t k = 0; k < a.size(); k++) { if (lowerOnly) { size_t i = k % n, j = k / n; if (i < j) continue; }
    if (!std::isfinite(MAGMA_Z_REAL(a[k]))) return INFINITY; e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(a[k], b[k]))); m = fmax(m, MAGMA_Z_ABS(b[k])); } return e / m; }
static double relw(std::vector<double> a, std::vector<double> b) { std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); double e = 0, m = 0;
  for (size_t i = 0; i < a.size(); i++) { if (!std::isfinite(a[i])) return INFINITY; e = fmax(e, fabs(a[i] - b[i])); m = fmax(m, fabs(b[i])); } return e / m; }
static void report(const char* nm, int ng, magma_int_t info, double t, const char* what, double err) {
  printf("%-22s ngpu=%d  info=%-7lld time=%7.3f s  %s=%.3g\n", nm, ng, (long long)info, t, what, err); }

static void potrf_m() { int n = 4000; auto A = hpd(n); std::vector<Z> L = A; { magma_int_t nn = n, i; lapackf77_zpotrf("L", &nn, L.data(), &nn, &i); }
  for (int ng : {1, 2}) { std::vector<Z> F = A; magma_int_t info = -999; double t = magma_wtime(); magma_zpotrf_m(ng, MagmaLower, n, F.data(), n, &info); t = magma_wtime() - t;
    report("zpotrf_m", ng, info, t, "rel err vs LAPACK (lower)", maxrel(F, L, true, n)); }
  for (int zc : {100, 3000}) { auto B = hpd(n); B[zc + (size_t)zc * n] = C(-1e6);
    magma_int_t li; { std::vector<Z> T = B; magma_int_t nn = n; lapackf77_zpotrf("L", &nn, T.data(), &nn, &li); }
    for (int ng : {1, 2}) { std::vector<Z> F = B; magma_int_t info = -999; magma_zpotrf_m(ng, MagmaLower, n, F.data(), n, &info);
      printf("zpotrf_m               ngpu=%d  A(%d,%d) = -1e6: info=%lld (LAPACK %lld)\n", ng, zc + 1, zc + 1, (long long)info, (long long)li); } } }
static void getrf_m() { int n = 4000; auto A = randm(n, n); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = MAGMA_Z_ADD(A[i + (size_t)i * n], C(8));
  std::vector<Z> R = A; std::vector<magma_int_t> ipr(n); { magma_int_t nn = n, i; lapackf77_zgetrf(&nn, &nn, R.data(), &nn, ipr.data(), &i); }
  for (int ng : {1, 2}) { std::vector<Z> F = A; std::vector<magma_int_t> ip(n); magma_int_t info = -999; double t = magma_wtime(); magma_zgetrf_m(ng, n, n, F.data(), n, ip.data(), &info); t = magma_wtime() - t;
    int pd = 0; for (int i = 0; i < n; i++) if (ip[i] != ipr[i]) pd++; report("zgetrf_m", ng, info, t, "rel err vs LAPACK", maxrel(F, R)); printf("                        pivots different from LAPACK: %d\n", pd); }
  auto B = A; for (int i = 0; i < n; i++) B[i + (size_t)2500 * n] = C(0);
  for (int ng : {1, 2}) { std::vector<Z> F = B; std::vector<magma_int_t> ip(n); magma_int_t info = -999; magma_zgetrf_m(ng, n, n, F.data(), n, ip.data(), &info);
    printf("zgetrf_m               ngpu=%d  column 2501 = 0: info=%lld (LAPACK 2501)\n", ng, (long long)info); } }
static void geqrf_m() { int n = 4000; auto A = randm(n, n); std::vector<Z> R = A, tr(n); { magma_int_t nn = n, lw = 64 * n, i; std::vector<Z> w(lw); lapackf77_zgeqrf(&nn, &nn, R.data(), &nn, tr.data(), w.data(), &lw, &i); }
  for (int ng : {1, 2}) { std::vector<Z> F = A, tau(n); magma_int_t lw = -1, info; Z q; magma_zgeqrf_m(ng, n, n, F.data(), n, tau.data(), &q, lw, &info);
    lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> w(lw); info = -999; double t = magma_wtime(); magma_zgeqrf_m(ng, n, n, F.data(), n, tau.data(), w.data(), lw, &info); t = magma_wtime() - t;
    double e = 0, m = 0; for (int i = 0; i < n; i++) { e = fmax(e, fabs(MAGMA_Z_ABS(F[i + (size_t)i * n]) - MAGMA_Z_ABS(R[i + (size_t)i * n]))); m = fmax(m, MAGMA_Z_ABS(R[i + (size_t)i * n])); }
    report("zgeqrf_m", ng, info, t, "max ||R_ii| - |R_ii,lapack|| / max", e / m); } }
static void trsm_m() { int m = 3000, n = 1000; auto A = randm(m, m); for (int i = 0; i < m; i++) A[i + (size_t)i * m] = MAGMA_Z_ADD(A[i + (size_t)i * m], C(m)); auto B0 = randm(m, n);
  std::vector<Z> R = B0; { Z one = C(1); magma_int_t mm = m, nn = n; blasf77_ztrsm("L", "L", "N", "N", &mm, &nn, &one, A.data(), &mm, R.data(), &mm); }
  for (int ng : {1, 2}) { std::vector<Z> B = B0; double t = magma_wtime(); magma_ztrsm_m(ng, MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, m, n, C(1), A.data(), m, B.data(), m); t = magma_wtime() - t;
    report("ztrsm_m", ng, 0, t, "rel err vs BLAS", maxrel(B, R)); } }
static void eig() { int n = 2000; auto A = herm(n); auto Bm = hpd(n); std::vector<double> wref(n), wgen(n);
  { std::vector<Z> T = A; magma_int_t nn = n, lw = -1, lrw = -1, liw = -1, i; Z q; double rq; magma_int_t iq;
    lapackf77_zheevd("N", "L", &nn, T.data(), &nn, wref.data(), &q, &lw, &rq, &lrw, &iq, &liw, &i); lw = (magma_int_t)MAGMA_Z_REAL(q); lrw = (magma_int_t)rq; liw = iq;
    std::vector<Z> w(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); lapackf77_zheevd("N", "L", &nn, T.data(), &nn, wref.data(), w.data(), &lw, rw.data(), &lrw, iw.data(), &liw, &i); }
  { std::vector<Z> T = A, U = Bm; magma_int_t nn = n, one = 1, lw = -1, i; Z q; std::vector<double> rw(3 * n); zhegv_(&one, "N", "L", &nn, T.data(), &nn, U.data(), &nn, wgen.data(), &q, &lw, rw.data(), &i);
    lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> w(lw); T = A; U = Bm; zhegv_(&one, "N", "L", &nn, T.data(), &nn, U.data(), &nn, wgen.data(), w.data(), &lw, rw.data(), &i); }
  for (int which = 0; which < 5; which++) for (int ng : {1, 2}) {
    std::vector<Z> T = A, U = Bm; std::vector<double> w(n, NAN); magma_int_t info = -999, m = n, lw = -1, lrw = -1, liw = -1; Z q; double rq; magma_int_t iq;
    auto call = [&](Z* wk, magma_int_t a, double* rw, magma_int_t b, magma_int_t* iw, magma_int_t c) {
      if (which == 0) magma_zheevd_m(ng, MagmaVec, MagmaLower, n, T.data(), n, w.data(), wk, a, rw, b, iw, c, &info);
      else if (which == 1) magma_zheevdx_m(ng, MagmaVec, MagmaRangeAll, MagmaLower, n, T.data(), n, 0, 0, 1, n, &m, w.data(), wk, a, rw, b, iw, c, &info);
      else if (which == 2) magma_zheevdx_2stage_m(ng, MagmaVec, MagmaRangeAll, MagmaLower, n, T.data(), n, 0, 0, 1, n, &m, w.data(), wk, a, rw, b, iw, c, &info);
      else if (which == 3) magma_zhegvd_m(ng, 1, MagmaVec, MagmaLower, n, T.data(), n, U.data(), n, w.data(), wk, a, rw, b, iw, c, &info);
      else magma_zhegvdx_m(ng, 1, MagmaVec, MagmaRangeAll, MagmaLower, n, T.data(), n, U.data(), n, 0, 0, 1, n, &m, w.data(), wk, a, rw, b, iw, c, &info); };
    call(&q, lw, &rq, lrw, &iq, liw); lw = (magma_int_t)MAGMA_Z_REAL(q); lrw = (magma_int_t)rq; liw = iq;
    std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); T = A; U = Bm; info = -999; double t = magma_wtime(); call(wk.data(), lw, rw.data(), lrw, iw.data(), liw); t = magma_wtime() - t;
    // eigenvector check on column 0: ||A v - lambda B v|| / ||A||  (B = I for the standard problems)
    double res = 0, an = 0; for (auto& a : A) an = fmax(an, MAGMA_Z_ABS(a)); bool gen = which >= 3;
    for (int i = 0; i < n; i++) { Z s = C(0), s2 = C(0); for (int k = 0; k < n; k++) { s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(A[i + (size_t)k * n], T[k])); if (gen) s2 = MAGMA_Z_ADD(s2, MAGMA_Z_MUL(Bm[i + (size_t)k * n], T[k])); }
      if (!gen) s2 = T[i]; res = fmax(res, MAGMA_Z_ABS(MAGMA_Z_SUB(s, MAGMA_Z_MUL(C(w[0]), s2)))); }
    const char* nm[] = {"zheevd_m", "zheevdx_m", "zheevdx_2stage_m", "zhegvd_m", "zhegvdx_m"};
    printf("%-22s ngpu=%d  info=%-7lld time=%7.3f s  eigenvalue rel err=%.3g  ||A v0 - l0 B v0||/||A||=%.3g\n", nm[which], ng, (long long)info, t, relw(w, gen ? wgen : wref), res / an); } }
static void geev_m() { int n = 1500; auto A = randm(n, n); std::vector<Z> wref(n);
  { std::vector<Z> T = A; magma_int_t nn = n, lw = 8 * n, i, one = 1; std::vector<Z> wk(lw); std::vector<double> rw(2 * n); Z d; lapackf77_zgeev("N", "N", &nn, T.data(), &nn, wref.data(), &d, &one, &d, &one, wk.data(), &lw, rw.data(), &i); }
  for (int ng : {1, 2}) { char e[8]; snprintf(e, 8, "%d", ng); setenv("MAGMA_NUM_GPUS", e, 1);
    std::vector<Z> T = A, w(n), VR((size_t)n * n), VL(1); std::vector<double> rw(2 * n); magma_int_t lw = -1, info; Z q;
    magma_zgeev_m(MagmaNoVec, MagmaVec, n, T.data(), n, w.data(), VL.data(), 1, VR.data(), n, &q, lw, rw.data(), &info); lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> wk(lw);
    T = A; info = -999; double t = magma_wtime(); magma_zgeev_m(MagmaNoVec, MagmaVec, n, T.data(), n, w.data(), VL.data(), 1, VR.data(), n, wk.data(), lw, rw.data(), &info); t = magma_wtime() - t;
    double e2 = 0, mx = 0; for (auto& a : w) { double dmin = 1e300; for (auto& b : wref) dmin = fmin(dmin, MAGMA_Z_ABS(MAGMA_Z_SUB(a, b))); e2 = fmax(e2, dmin); } for (auto& b : wref) mx = fmax(mx, MAGMA_Z_ABS(b));
    printf("%-22s ngpu=%lld  info=%-7lld time=%7.3f s  eigenvalue set distance=%.3g\n", "zgeev_m", (long long)magma_num_gpus(), (long long)info, t, e2 / mx); } }
static void potrf_mgpu() { int n = 4096, nb = magma_get_zpotrf_nb(n); auto A = hpd(n); std::vector<Z> L = A; { magma_int_t nn = n, i; lapackf77_zpotrf("L", &nn, L.data(), &nn, &i); }
  for (int zc : {-1, 300, 3500}) for (int ng : {1, 2}) {
    std::vector<Z> B = A; if (zc >= 0) B[zc + (size_t)zc * n] = C(-1e6); magma_int_t li = 0; if (zc >= 0) { std::vector<Z> T = B; magma_int_t nn = n; lapackf77_zpotrf("L", &nn, T.data(), &nn, &li); }
    magma_queue_t q[2]; std::vector<Z*> d(ng); int ldda = magma_roundup(n, 32); int nloc = ((n / nb) / ng + 1) * nb;
    for (int g = 0; g < ng; g++) { magma_setdevice(g); magma_queue_create(g, &q[g]); magma_zmalloc(&d[g], (size_t)ldda * nloc); }
    // Lower: distribute rows block-cyclically = columns of A^H; MAGMA's tester transposes for Lower; use the 1D column distribution of A^H
    std::vector<Z> BH((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) BH[j + (size_t)i * n] = MAGMA_Z_CONJ(B[i + (size_t)j * n]);
    magma_zsetmatrix_1D_col_bcyclic(ng, n, n, nb, BH.data(), n, d.data(), ldda, q);
    magma_int_t info = -999; double t = magma_wtime(); magma_zpotrf_mgpu(ng, MagmaUpper, n, d.data(), ldda, &info); t = magma_wtime() - t;
    std::vector<Z> FH((size_t)n * n); magma_zgetmatrix_1D_col_bcyclic(ng, n, n, nb, d.data(), ldda, FH.data(), n, q);
    double e = 0, m = 0; if (zc < 0) for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { Z f = MAGMA_Z_CONJ(FH[j + (size_t)i * n]); e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(f, L[i + (size_t)j * n]))); m = fmax(m, MAGMA_Z_ABS(L[i + (size_t)j * n])); }
    if (zc < 0) report("zpotrf_mgpu (Upper)", ng, info, t, "rel err vs LAPACK", e / m);
    else printf("zpotrf_mgpu (Upper)    ngpu=%d  A(%d,%d) = -1e6 (block on GPU %d): info=%lld (LAPACK %lld)\n", ng, zc + 1, zc + 1, (zc / nb) % ng, (long long)info, (long long)li);
    for (int g = 0; g < ng; g++) { magma_setdevice(g); magma_free(d[g]); magma_queue_destroy(q[g]); } magma_setdevice(0); } }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; magma_init(); int nv = 0; cudaGetDeviceCount(&nv);
  printf("visible devices: %d\n", nv);
  if (m == "potrf_m") potrf_m(); else if (m == "getrf_m") getrf_m(); else if (m == "geqrf_m") geqrf_m(); else if (m == "trsm_m") trsm_m();
  else if (m == "eig") eig(); else if (m == "geev_m") geev_m(); else if (m == "potrf_mgpu") potrf_mgpu();
  cudaError_t e = cudaGetLastError(); printf("cudaGetLastError at exit: %s\n", cudaGetErrorString(e)); magma_finalize(); return 0;
}

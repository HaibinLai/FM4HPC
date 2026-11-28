// Twentieth GPU round: findings of the audit of the non-z status files (HANDOFF §29.21).  GPU 0 only.
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
typedef magmaDoubleComplex Z;
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 2024u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
extern "C" void dsytri_(const char*, const magma_int_t*, double*, const magma_int_t*, const magma_int_t*, double*, magma_int_t*);
static std::vector<Z> herm(int n) { std::vector<Z> A((size_t)n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { Z v = i == j ? C(4 * rnd()) : C(rnd(), rnd());
  A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = MAGMA_Z_CONJ(v); } return A; }
static void eigcheck(const char* tag, int n, const std::vector<Z>& A, double vl, double vu, bool twostage) {
  std::vector<Z> W = A; std::vector<double> w(n); magma_int_t m = -1, info = -1, lw = -1, lrw = -1, liw = -1; Z q; double rq; magma_int_t iq;
  auto call = [&](Z* wk, magma_int_t l1, double* rw, magma_int_t l2, magma_int_t* iw, magma_int_t l3) {
    if (twostage) magma_zheevdx_2stage(MagmaVec, MagmaRangeV, MagmaLower, n, W.data(), n, vl, vu, 1, n, &m, w.data(), wk, l1, rw, l2, iw, l3, &info);
    else magma_zheevdx(MagmaVec, MagmaRangeV, MagmaLower, n, W.data(), n, vl, vu, 1, n, &m, w.data(), wk, l1, rw, l2, iw, l3, &info); };
  call(&q, lw, &rq, lrw, &iq, liw); lw = (magma_int_t)MAGMA_Z_REAL(q); lrw = (magma_int_t)rq; liw = iq;
  std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); W = A; call(wk.data(), lw, rw.data(), lrw, iw.data(), liw);
  // LAPACK count in (vl, vu]
  std::vector<Z> Al = A; std::vector<double> wl(n); { magma_int_t nn = n, lw2 = -1, inf; Z q2; std::vector<double> rw2(3 * n);
    lapackf77_zheev("N", "L", &nn, Al.data(), &nn, wl.data(), &q2, &lw2, rw2.data(), &inf); lw2 = (magma_int_t)MAGMA_Z_REAL(q2); std::vector<Z> wk2(lw2);
    lapackf77_zheev("N", "L", &nn, Al.data(), &nn, wl.data(), wk2.data(), &lw2, rw2.data(), &inf); }
  int cnt = 0; for (double x : wl) if (x > vl && x <= vu) cnt++;
  double res = 0, orth = 0, an = 0; for (auto& a : A) an = fmax(an, MAGMA_Z_ABS(a));
  for (int c = 0; c < m; c++) { for (int i = 0; i < n; i++) { Z s = C(0); for (int k = 0; k < n; k++) s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(A[i + (size_t)k * n], W[k + (size_t)c * n]));
      s = MAGMA_Z_SUB(s, MAGMA_Z_MUL(C(w[c]), W[i + (size_t)c * n])); res = fmax(res, MAGMA_Z_ABS(s)); }
    for (int d = 0; d < m; d++) { Z s = C(0); for (int i = 0; i < n; i++) s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(MAGMA_Z_CONJ(W[i + (size_t)c * n]), W[i + (size_t)d * n]));
      if (c == d) s = MAGMA_Z_SUB(s, C(1)); orth = fmax(orth, MAGMA_Z_ABS(s)); } }
  printf("%s: info=%lld m=%lld (LAPACK count in (vl,vu] = %d)  max|AV - V Lambda|/max|A| = %.3g  max|V^H V - I| = %.3g\n", tag, (long long)info, (long long)m, cnt, res / an, orth);
}
static void f1() { int n = 300; auto A = herm(n); eigcheck("F1 zheevdx n=300 random, jobz=V, range V (0.5, 30]", n, A, 0.5, 30, false);
  eigcheck("   zheevdx_2stage, same", n, A, 0.5, 30, true); }
static void f7() { int n = 300; std::vector<Z> A((size_t)n * n, C(0)); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = C(i + 1.0);
  eigcheck("F7 zheevdx diag(1..300), range V (10, 20] (10 eigenvalues; 10 itself excluded)", n, A, 10, 20, false); }
static void f3() { for (int nb : {0, 600}) { int n = 2400; std::vector<Z> A((size_t)n * n); for (auto& a : A) a = C(rnd(), rnd());
    Z* d; magma_zmalloc(&d, (size_t)n * n); magma_zsetmatrix(n, n, A.data(), n, d, n, Q); std::vector<magma_int_t> ip(n); magma_int_t info = -999;
    if (nb == 0) magma_zgetrf_native(n, n, d, n, ip.data(), &info);
    else { magma_queue_t qs[2]; magma_event_t ev[2]; magma_queue_create(0, &qs[0]); magma_queue_create(0, &qs[1]); magma_event_create(&ev[0]); magma_event_create(&ev[1]);
      magma_int_t lh = -1, ld = -1; magma_zgetrf_expert_gpu_work(n, n, NULL, n, NULL, &info, MagmaNative, nb, 32, NULL, &lh, NULL, &ld, ev, qs);
      void *hw = NULL, *dw = NULL; if (lh > 0) magma_malloc_pinned(&hw, lh); if (ld > 0) magma_malloc(&dw, ld); info = -999;
      magma_zgetrf_expert_gpu_work(n, n, d, n, ip.data(), &info, MagmaNative, nb, 32, hw, &lh, dw, &ld, ev, qs); magma_queue_sync(qs[0]); magma_queue_sync(qs[1]); }
    std::vector<Z> F((size_t)n * n); magma_zgetmatrix(n, n, d, n, F.data(), n, Q); int badp = 0; for (int i = 0; i < n; i++) if (ip[i] < i + 1 || ip[i] > n) badp++;
    // ||P A - L U|| / ||A||
    std::vector<Z> L((size_t)n * n, C(0)), U((size_t)n * n, C(0)), LU((size_t)n * n);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { if (i > j) L[i + (size_t)j * n] = F[i + (size_t)j * n]; else U[i + (size_t)j * n] = F[i + (size_t)j * n]; if (i == j) L[i + (size_t)j * n] = C(1); }
    Z one = C(1), zero = C(0); magma_int_t nn = n; blasf77_zgemm("N", "N", &nn, &nn, &nn, &one, L.data(), &nn, U.data(), &nn, &zero, LU.data(), &nn);
    std::vector<Z> PA = A; magma_int_t i1 = 1, ione = 1; std::vector<magma_int_t> ipc = ip; for (auto& p : ipc) p = std::min<magma_int_t>(std::max<magma_int_t>(p, 1), n);
    lapackf77_zlaswp(&nn, PA.data(), &nn, &i1, &nn, ipc.data(), &ione);
    double e = 0, an = 0; for (size_t k = 0; k < PA.size(); k++) { e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(PA[k], LU[k]))); an = fmax(an, MAGMA_Z_ABS(A[k])); }
    printf("F3 zgetrf native n=2400, %s: info=%lld  invalid ipiv entries (ipiv[i] < i+1 or > n): %d  max|PA - LU|/max|A| = %.3g\n",
           nb ? "expert nb=600 recnb=32 (panel split with n2 = 1200 > 1024)" : "default nb (control)", (long long)info, badp, e / an); } }
static void gesvj(const char* tag, int n, double scale, bool zerocol) {
  int m = n, b = 1; std::vector<Z> A((size_t)m * n); for (auto& a : A) a = C(scale * rnd(), scale * rnd()); if (zerocol) for (int i = 0; i < m; i++) A[i] = C(0);
  Z *dA, *dU, *dV; double* dS; magma_zmalloc(&dA, (size_t)m * n); magma_zmalloc(&dU, (size_t)m * n); magma_zmalloc(&dV, (size_t)n * n); magma_dmalloc(&dS, n);
  magma_zsetmatrix(m, n, A.data(), m, dA, m, Q); Z **dAa, **dUa, **dVa; double** dSa; magma_int_t* dinfo;
  magma_malloc((void**)&dAa, sizeof(Z*)); magma_malloc((void**)&dUa, sizeof(Z*)); magma_malloc((void**)&dVa, sizeof(Z*)); magma_malloc((void**)&dSa, sizeof(double*)); magma_imalloc(&dinfo, 1);
  magma_setvector(1, sizeof(Z*), &dA, 1, dAa, 1, Q); magma_setvector(1, sizeof(Z*), &dU, 1, dUa, 1, Q); magma_setvector(1, sizeof(Z*), &dV, 1, dVa, 1, Q); magma_setvector(1, sizeof(double*), &dS, 1, dSa, 1, Q);
  magma_zgesvj_batched(MagmaSomeVec, MagmaSomeVec, m, n, dAa, m, dSa, dUa, m, dVa, n, dinfo, b, Q);
  magma_int_t info; magma_getvector(1, sizeof(magma_int_t), dinfo, 1, &info, 1, Q); std::vector<Z> U((size_t)m * n), V((size_t)n * n); std::vector<double> Sv(n);
  magma_zgetmatrix(m, n, dU, m, U.data(), m, Q); magma_zgetmatrix(n, n, dV, n, V.data(), n, Q); magma_dgetvector(n, dS, 1, Sv.data(), 1, Q);
  int nfU = 0, nfV = 0, nfS = 0, infS = 0; for (auto& x : U) if (!std::isfinite(MAGMA_Z_REAL(x))) nfU++; for (auto& x : V) if (!std::isfinite(MAGMA_Z_REAL(x))) nfV++;
  for (double s : Sv) { if (std::isnan(s)) nfS++; if (std::isinf(s)) infS++; }
  printf("%s (n=%d, %s path): info=%lld  non-finite U entries=%d, V entries=%d; S: NaN %d, Inf %d, S[0]=%g\n", tag, n,
         magma_zgesvj_batched_use_fused(MagmaSomeVec, MagmaSomeVec, m, n) ? "fused" : "blocked", (long long)info, nfU, nfV, nfS, infS, Sv[0]);
}
static void f4() { for (int n : {32, 64, 128, 256}) gesvj("F4 gesvj_batched, column 0 = 0", n, 1, true); gesvj("   gesvj_batched, A = 0", 128, 0, false); }
static void f9() { for (int n : {64, 128}) { gesvj("F9 gesvj_batched, entries ~1e160", n, 1e160, false); gesvj("   gesvj_batched, entries ~1e-170", n, 1e-170, false); } }
static void f5() { int n = 8; std::vector<double> A0(n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { double v = rnd() + (i == j ? (i % 2 ? -3 : 3) : 0); A0[i + j * n] = v; A0[j + i * n] = v; }
  for (int lower = 0; lower < 2; lower++) { const char* ul = lower ? "L" : "U"; std::vector<double> F = A0, R = A0; std::vector<magma_int_t> ip(n), ip2(n); magma_int_t nn = n, lw = 64 * n, inf, inf2;
    std::vector<double> wk(lw); lapackf77_dsytrf(ul, &nn, F.data(), &nn, ip.data(), wk.data(), &lw, &inf); R = F; ip2 = ip;
    dsytri_(ul, &nn, R.data(), &nn, ip2.data(), wk.data(), &inf2);
    double det[2]; magma_int_t inert[3], info = -999; magma_dsidi(lower ? MagmaLower : MagmaUpper, F.data(), n, n, ip.data(), det, inert, wk.data(), 1, &info);
    double e = 0, m = 0; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { bool tri = lower ? (i >= j) : (i <= j); if (!tri) continue;
      e = fmax(e, fabs(F[i + j * n] - R[i + j * n])); m = fmax(m, fabs(R[i + j * n])); }
    printf("F5 dsidi(job=1, inverse) uplo=%s after LAPACK dsytrf: info=%lld  max|inv - dsytri inv| / max = %.3g (on the %s triangle)\n", ul, (long long)info, e / m, ul); } }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; magma_init(); magma_queue_create(0, &Q);
  if (m == "f1") f1(); else if (m == "f7") f7(); else if (m == "f3") f3(); else if (m == "f4") f4(); else if (m == "f9") f9(); else if (m == "f5") f5();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

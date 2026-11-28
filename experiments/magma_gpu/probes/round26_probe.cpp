// Twenty-sixth GPU round, on roxy (RTX 5070 Ti, sm_120, Windows; MAGMA 2.10.0 built with MSVC 14.44 + nvcc 13.5 +
// OpenBLAS 0.3.34): library-level checks of the code-reading findings of HANDOFF §29.29-§29.38.
// One mode per process:  round26_probe <mode>
//   hegvdx2  zhegvdx_2stage: the size it reports is rejected by the inner zheevdx_2stage (expect -14, A/B modified)
//   gesdd    zgesdd path 6to-slow: at the minimum lwork zungbr(P) gets too little (expect -9 printed, info 0, bad VT)
//   ungtr    magma_zungtr (undeclared, exported) upper path vs LAPACK zungtr
//   gbstale  zgbtrf_native with a stale nonzero info (expect: returns it, no factorization)
//   argnum   reported -k of zhegvx (jobz/range), zgesv_batched / zgetrs_nopiv_batched (lddb), zcposv_gpu bad uplo
//   he2hb    zhetrd_he2hb with n < nb and the queried lwork: writes before work (canary)
//   geqlf    zgeqlf with min(m,n) = 0: the queried lwork is rejected
//   gbbatch  zgbtrf_batched (pointer-array) with 65536 band matrices kl = ku = 32, n = 65 (generic path, set_fillin)
//   gbalias  the same launch with 65536 pointers onto 1024 matrices (for GPUs with < 7 GB free)
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
extern "C" magma_int_t magma_get_parallel_numthreads();
extern "C" magma_int_t magma_zungtr(magma_uplo_t uplo, magma_int_t n, Z* A, magma_int_t lda, Z* tau, Z* work,
                                    magma_int_t lwork, Z* dT, magma_int_t nb, magma_int_t* info);
extern "C" void zhegv_(const magma_int_t*, const char*, const char*, const magma_int_t*, Z*, const magma_int_t*, Z*,
                       const magma_int_t*, double*, Z*, const magma_int_t*, double*, magma_int_t*);
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 2026u;
static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static std::vector<Z> randm(int m, int n) { std::vector<Z> A((size_t)m * n); for (auto& a : A) a = C(rnd(), rnd()); return A; }
static std::vector<Z> herm(int n) { std::vector<Z> A((size_t)n * n);
  for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { Z v = i == j ? C(rnd()) : C(rnd(), rnd());
    A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = MAGMA_Z_CONJ(v); } return A; }
static std::vector<Z> hpd(int n) { auto A = herm(n); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = MAGMA_Z_ADD(A[i + (size_t)i * n], C(n)); return A; }
static double maxdiff(const std::vector<Z>& a, const std::vector<Z>& b) { double e = 0;
  for (size_t k = 0; k < a.size(); k++) e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(a[k], b[k]))); return e; }
static double relw(std::vector<double> a, std::vector<double> b) { std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
  double e = 0, m = 0; for (size_t i = 0; i < a.size(); i++) { if (!std::isfinite(a[i])) return INFINITY; e = fmax(e, fabs(a[i] - b[i])); m = fmax(m, fabs(b[i])); }
  return e / m; }
// max |R R^H - I| over the rows of an m x n matrix (lda = m)
static double roworth(const Z* R, int m, int n) { double e = 0;
  for (int i = 0; i < m; i++) for (int k = 0; k < m; k++) { Z s = C(0);
    for (int j = 0; j < n; j++) s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(R[i + (size_t)j * m], MAGMA_Z_CONJ(R[k + (size_t)j * m])));
    e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(s, C(i == k)))); } return e; }

static magma_queue_t Q;

static void hegvdx2() {
  magma_int_t th = magma_get_parallel_numthreads();
  for (int n : {128, 129, 200}) {
    magma_int_t nb = magma_get_zbulge_nb(n, th);
    auto A0 = herm(n), B0 = hpd(n);
    std::vector<double> wref(n); { auto A = A0, B = B0; magma_int_t one = 1, nn = n, lw = 4 * n, inf; std::vector<Z> wk(lw); std::vector<double> rw(3 * n);
      zhegv_(&one, "N", "L", &nn, A.data(), &nn, B.data(), &nn, wref.data(), wk.data(), &lw, rw.data(), &inf); }
    auto A = A0, B = B0; std::vector<double> w(n); magma_int_t mout = 0, info = -999, iq = 0; Z wq; double rq;
    magma_zhegvdx_2stage(1, MagmaNoVec, MagmaRangeAll, MagmaLower, n, A.data(), n, B.data(), n, 0, 0, 0, 0, &mout, w.data(),
                         &wq, -1, &rq, -1, &iq, -1, &info);
    magma_int_t lwork = (magma_int_t)MAGMA_Z_REAL(wq), lrwork = (magma_int_t)rq, liwork = iq;
    for (int pass = 0; pass < 2; pass++) {
      magma_int_t lw = pass == 0 ? lwork : 5 * n + 2 * nb * n;     // pass 1: the inner routine's own minimum
      A = A0; B = B0; std::vector<Z> work(lw); std::vector<double> rwork(lrwork); std::vector<magma_int_t> iwork(liwork);
      info = -999; mout = 0; std::fill(w.begin(), w.end(), NAN);
      magma_zhegvdx_2stage(1, MagmaNoVec, MagmaRangeAll, MagmaLower, n, A.data(), n, B.data(), n, 0, 0, 0, 0, &mout, w.data(),
                           work.data(), lw, rwork.data(), lrwork, iwork.data(), liwork, &info);
      printf("zhegvdx_2stage n=%d nb=%lld threads=%lld  lwork=%lld (%s)  info=%lld  A modified=%s B modified=%s  eig err=%.3g\n",
             n, (long long)nb, (long long)th, (long long)lw, pass == 0 ? "its own query" : "5n+2nb*n", (long long)info,
             maxdiff(A, A0) > 0 ? "yes" : "no", maxdiff(B, B0) > 0 ? "yes" : "no", info == 0 ? relw(w, wref) : NAN);
    }
  }
}

static void gesdd() {
  const int m = 1024, n = 1087; auto A0 = randm(m, n);
  std::vector<double> s(m), rwork(5 * (size_t)m * m + 7 * m); std::vector<Z> U((size_t)m * m), VT(1); std::vector<magma_int_t> iwork(8 * m);
  magma_int_t info; Z wq; auto A = A0;
  magma_zgesdd(MagmaOverwriteVec, m, n, A.data(), m, s.data(), U.data(), m, VT.data(), 1, &wq, -1, rwork.data(), iwork.data(), &info);
  magma_int_t maxwrk = (magma_int_t)MAGMA_Z_REAL(wq);
  // smallest accepted lwork: argument rejection is immediate, so bisect on it
  magma_int_t lo = (magma_int_t)m * m, hi = maxwrk;
  while (lo < hi) { magma_int_t mid = lo + (hi - lo) / 2; std::vector<Z> w(1); info = 0;
    // a too-small lwork returns -12 before touching anything; probe acceptance with a query-free call only if rejected
    A = A0; std::vector<Z> work(mid);
    fprintf(stderr, "[bisect lwork=%lld]\n", (long long)mid);
    magma_zgesdd(MagmaOverwriteVec, m, n, A.data(), m, s.data(), U.data(), m, VT.data(), 1, work.data(), mid, rwork.data(), iwork.data(), &info);
    if (info == -12) lo = mid + 1; else hi = mid; }
  printf("zgesdd m=%d n=%d jobz=O: queried lwork (maxwrk) = %lld, smallest accepted lwork = %lld\n", m, n, (long long)maxwrk, (long long)lo);
  for (magma_int_t lw : {lo, maxwrk}) {
    A = A0; std::vector<Z> work(lw); info = -999;
    fprintf(stderr, "[run lwork=%lld]\n", (long long)lw);
    magma_zgesdd(MagmaOverwriteVec, m, n, A.data(), m, s.data(), U.data(), m, VT.data(), 1, work.data(), lw, rwork.data(), iwork.data(), &info);
    printf("zgesdd lwork=%-8lld (%s)  info=%lld  max|VT VT^H - I| (rows of A on exit) = %.3g\n", (long long)lw,
           lw == maxwrk ? "fast path" : "slow path", (long long)info, roworth(A.data(), m, n));
  }
}

static void ungtr() {
  for (int n : {8, 50}) {
    auto A = herm(n); magma_int_t nn = n, info, lw = 64 * n; std::vector<double> d(n), e(n); std::vector<Z> tau(n), work(lw);
    lapackf77_zhetrd("U", &nn, A.data(), &nn, d.data(), e.data(), tau.data(), work.data(), &lw, &info);
    auto Qref = A, Qm = A;
    lapackf77_zungtr("U", &nn, Qref.data(), &nn, tau.data(), work.data(), &lw, &info);
    magma_int_t minfo = -999;
    magma_zungtr(MagmaUpper, n, Qm.data(), n, tau.data(), work.data(), lw, NULL, 32, &minfo);
    printf("magma_zungtr upper n=%d info=%lld  max|Q_magma - Q_lapack| = %.3g  orth(Q_lapack)=%.2g  orth(Q_magma)=%.3g\n",
           n, (long long)minfo, maxdiff(Qm, Qref), roworth(Qref.data(), n, n), roworth(Qm.data(), n, n));
  }
}

static void gbstale() {
  const int n = 200, kl = 2, ku = 2, ldab = 2 * kl + ku + 1;
  std::vector<Z> AB((size_t)ldab * n, C(0));
  for (int j = 0; j < n; j++) for (int i = std::max(0, j - ku); i <= std::min(n - 1, j + kl); i++)
    AB[kl + ku + i - j + (size_t)j * ldab] = i == j ? C(10 + rnd()) : C(rnd(), rnd());
  Z* dAB; magma_int_t* dipiv; magma_zmalloc(&dAB, (size_t)ldab * n); magma_imalloc(&dipiv, n);
  for (magma_int_t start : {7, 0}) {
    magma_zsetmatrix(ldab, n, AB.data(), ldab, dAB, ldab, Q);
    magma_int_t info = start; magma_zgbtrf_native(n, n, kl, ku, dAB, ldab, dipiv, &info); magma_queue_sync(Q);
    std::vector<Z> out(AB.size()); magma_zgetmatrix(ldab, n, dAB, ldab, out.data(), ldab, Q);
    printf("zgbtrf_native n=%d: info on entry=%lld -> info=%lld, band modified=%s\n", n, (long long)start, (long long)info,
           maxdiff(out, AB) > 0 ? "yes (factored)" : "no (nothing done)");
  }
  magma_free(dAB); magma_free(dipiv);
}

static void argnum() {
  const int n = 4; auto A = herm(n), B = hpd(n); std::vector<double> w(n), rwork(7 * n); std::vector<Z> Zm(n * n), work(1000);
  std::vector<magma_int_t> iwork(5 * n), ifail(n); magma_int_t mout, info;
  magma_zhegvx(1, (magma_vec_t)300, MagmaRangeAll, MagmaLower, n, A.data(), n, B.data(), n, 0, 0, 1, n, 0, &mout, w.data(),
               Zm.data(), n, work.data(), 1000, rwork.data(), iwork.data(), ifail.data(), &info);
  printf("zhegvx invalid jobz  (argument 2): info=%lld\n", (long long)info);
  magma_zhegvx(1, MagmaNoVec, (magma_range_t)0, MagmaLower, n, A.data(), n, B.data(), n, 0, 0, 1, n, 0, &mout, w.data(),
               Zm.data(), n, work.data(), 1000, rwork.data(), iwork.data(), ifail.data(), &info);
  printf("zhegvx invalid range (argument 3): info=%lld\n", (long long)info);
  // batched: one 4x4 system, lddb = n-1 (invalid)
  Z *dA, *dB; magma_int_t *dpiv, *dinf; magma_zmalloc(&dA, n * n); magma_zmalloc(&dB, n); magma_imalloc(&dpiv, n); magma_imalloc(&dinf, 1);
  Z **dAa, **dBa; magma_int_t** dPa; magma_malloc((void**)&dAa, sizeof(Z*)); magma_malloc((void**)&dBa, sizeof(Z*)); magma_malloc((void**)&dPa, sizeof(magma_int_t*));
  cudaMemcpy(dAa, &dA, sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dBa, &dB, sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dPa, &dpiv, sizeof(void*), cudaMemcpyHostToDevice);
  info = magma_zgesv_batched(n, 1, dAa, n, dPa, dBa, n - 1, dinf, 1, Q);
  printf("zgesv_batched lddb < n (lddb is argument 7): returned %lld\n", (long long)info);
  info = magma_zgetrs_nopiv_batched(MagmaNoTrans, n, 1, dAa, n, dBa, n - 1, dinf, 1, Q);
  printf("zgetrs_nopiv_batched lddb < n (lddb is argument 7): returned %lld\n", (long long)info);
  // zcposv_gpu with an invalid uplo
  auto H = hpd(n); Z *dH, *dX, *dW; magmaFloatComplex* dS; magma_zmalloc(&dH, n * n); magma_zmalloc(&dX, n); magma_zmalloc(&dW, n);
  magma_cmalloc(&dS, n * (n + 1)); magma_zsetmatrix(n, n, H.data(), n, dH, n, Q); std::vector<Z> b(n, C(1)); magma_zsetvector(n, b.data(), 1, dB, 1, Q);
  magma_int_t iter = -999; info = -999;
  magma_zcposv_gpu((magma_uplo_t)0, n, 1, dH, n, dB, n, dX, n, dW, dS, &iter, &info);
  printf("zcposv_gpu invalid uplo (argument 1): iter=%lld info=%lld\n", (long long)iter, (long long)info);
  magma_free(dA); magma_free(dB); magma_free(dpiv); magma_free(dinf); magma_free(dAa); magma_free(dBa); magma_free(dPa);
  magma_free(dH); magma_free(dX); magma_free(dW); magma_free(dS);
}

static void he2hb() {
  const int n = 16, nb = 32; auto A = herm(n); std::vector<Z> tau(n); magma_int_t info; Z wq;
  Z* dT; magma_zmalloc(&dT, (size_t)n * nb);
  magma_zhetrd_he2hb(MagmaLower, n, nb, A.data(), n, tau.data(), &wq, -1, dT, &info);
  magma_int_t lwork = (magma_int_t)MAGMA_Z_REAL(wq);
  const Z canary = C(12345.0, -6789.0); std::vector<Z> big((size_t)nb * nb + lwork + nb * nb, canary);
  Z* work = big.data() + (size_t)nb * nb;
  info = -999; magma_zhetrd_he2hb(MagmaLower, n, nb, A.data(), n, tau.data(), work, lwork, dT, &info);
  size_t before = 0, after = 0;
  for (size_t k = 0; k < (size_t)nb * nb; k++) if (!MAGMA_Z_EQUAL(big[k], canary)) before++;
  for (size_t k = (size_t)nb * nb + lwork; k < big.size(); k++) if (!MAGMA_Z_EQUAL(big[k], canary)) after++;
  printf("zhetrd_he2hb n=%d nb=%d: queried lwork=%lld, info=%lld; elements written before work: %zu, after work+lwork: %zu\n",
         n, nb, (long long)lwork, (long long)info, before, after);
  magma_free(dT);
}

static void geqlf() {
  for (int m : {0, 7}) { int n = m == 0 ? 5 : 0; std::vector<Z> A(std::max(1, m * n) + 1), tau(1); Z wq; magma_int_t info;
    magma_zgeqlf(m, n, A.data(), std::max(1, m), tau.data(), &wq, -1, &info);
    magma_int_t lw = (magma_int_t)MAGMA_Z_REAL(wq); std::vector<Z> work(std::max<magma_int_t>(lw, 1));
    magma_zgeqlf(m, n, A.data(), std::max(1, m), tau.data(), work.data(), lw, &info);
    printf("zgeqlf m=%d n=%d: queried lwork=%lld, call with it -> info=%lld\n", m, n, (long long)lw, (long long)info); }
}

static void gbbatch() {
  const int m = 65, n = 65, kl = 32, ku = 32, ldab = 2 * kl + ku + 1; const size_t per = (size_t)ldab * n;
  size_t fr, tot; cudaMemGetInfo(&fr, &tot);
  auto gen = [&](int k, std::vector<Z>& AB) { unsigned s = 977u * (unsigned)k + 13u; auto r = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffffff) / (double)0x1000000 - 0.5; };
    std::fill(AB.begin(), AB.end(), C(0));
    for (int j = 0; j < n; j++) for (int i = std::max(0, j - ku); i <= std::min(m - 1, j + kl); i++) AB[kl + ku + i - j + (size_t)j * ldab] = C(r(), r()); };
  const int samples[] = {0, 1, 2, 65532, 65533, 65534, 65535};
  std::vector<std::vector<Z>> ref(7); std::vector<std::vector<magma_int_t>> refp(7);
  for (int t = 0; t < 7; t++) { ref[t].resize(per); gen(samples[t], ref[t]); refp[t].resize(n); magma_int_t mm = m, nn = n, a = kl, b = ku, ld = ldab, inf;
    lapackf77_zgbtrf(&mm, &nn, &a, &b, ref[t].data(), &ld, refp[t].data(), &inf); }
  for (int batch : {65535, 65536}) {
    size_t need = (size_t)batch * (per * sizeof(Z) + n * sizeof(magma_int_t) + sizeof(magma_int_t) + 2 * sizeof(void*));
    cudaMemGetInfo(&fr, &tot);
    if (need + (256u << 20) > fr) { printf("gbbatch batch=%d: needs %.2f GB, only %.2f GB free - skipped\n", batch, need / 1e9, fr / 1e9); continue; }
    Z* dAB; magma_int_t *dpiv, *dinfo; magma_zmalloc(&dAB, (size_t)batch * per); magma_imalloc(&dpiv, (size_t)batch * n); magma_imalloc(&dinfo, batch);
    std::vector<Z*> hA(batch); std::vector<magma_int_t*> hP(batch); for (int k = 0; k < batch; k++) { hA[k] = dAB + (size_t)k * per; hP[k] = dpiv + (size_t)k * n; }
    Z** dAa; magma_int_t** dPa; magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dPa, batch * sizeof(void*));
    cudaMemcpy(dAa, hA.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dPa, hP.data(), batch * sizeof(void*), cudaMemcpyHostToDevice);
    std::vector<Z> chunk(per * 1024);
    for (int k0 = 0; k0 < batch; k0 += 1024) { int c = std::min(1024, batch - k0); std::vector<Z> one(per);
      for (int k = 0; k < c; k++) { gen(k0 + k, one); std::copy(one.begin(), one.end(), chunk.begin() + (size_t)k * per); }
      cudaMemcpy(dAB + (size_t)k0 * per, chunk.data(), (size_t)c * per * sizeof(Z), cudaMemcpyHostToDevice); }
    cudaGetLastError();
    double t = magma_wtime();
    magma_int_t ret = magma_zgbtrf_batched(m, n, kl, ku, dAa, ldab, dPa, dinfo, batch, Q);
    magma_queue_sync(Q); t = magma_wtime() - t; cudaError_t e = cudaGetLastError();
    printf("zgbtrf_batched batch=%d m=n=%d kl=ku=%d: returned %lld, cudaGetLastError=%s, time %.2f s\n", batch, m, kl, (long long)ret, cudaGetErrorString(e), t);
    for (int t2 = 0; t2 < 7; t2++) { int k = samples[t2]; if (k >= batch) continue; std::vector<Z> out(per); std::vector<magma_int_t> p(n); magma_int_t inf;
      cudaMemcpy(out.data(), dAB + (size_t)k * per, per * sizeof(Z), cudaMemcpyDeviceToHost); cudaMemcpy(p.data(), dpiv + (size_t)k * n, n * sizeof(magma_int_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(&inf, dinfo + k, sizeof(magma_int_t), cudaMemcpyDeviceToHost);
      int pd = 0; for (int i = 0; i < n; i++) if (p[i] != refp[t2][i]) pd++;
      printf("    matrix %5d: info=%lld  max|LU - LAPACK|=%.3g  pivots different=%d\n", k, (long long)inf, maxdiff(out, ref[t2]), pd); }
    magma_free(dAB); magma_free(dpiv); magma_free(dinfo); magma_free(dAa); magma_free(dPa);
  }
}

// same launch question with little memory: 65536 pointers onto D = 1024 distinct band matrices (aliased, so the
// numbers are meaningless); what is checked is whether the factorization launches report an error
static void gbalias() {
  const int m = 65, n = 65, kl = 32, ku = 32, ldab = 2 * kl + ku + 1, D = 1024; const size_t per = (size_t)ldab * n;
  for (int batch : {65535, 65536}) {
    Z* dAB; magma_int_t *dpiv, *dinfo; magma_zmalloc(&dAB, (size_t)D * per); magma_imalloc(&dpiv, (size_t)D * n); magma_imalloc(&dinfo, batch);
    std::vector<Z> h(per * D); for (auto& v : h) v = C(rnd(), rnd()); cudaMemcpy(dAB, h.data(), h.size() * sizeof(Z), cudaMemcpyHostToDevice);
    std::vector<Z*> hA(batch); std::vector<magma_int_t*> hP(batch); for (int k = 0; k < batch; k++) { hA[k] = dAB + (size_t)(k % D) * per; hP[k] = dpiv + (size_t)(k % D) * n; }
    Z** dAa; magma_int_t** dPa; magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dPa, batch * sizeof(void*));
    cudaMemcpy(dAa, hA.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dPa, hP.data(), batch * sizeof(void*), cudaMemcpyHostToDevice);
    cudaGetLastError();
    magma_int_t ret = magma_zgbtrf_batched(m, n, kl, ku, dAa, ldab, dPa, dinfo, batch, Q); cudaError_t e1 = cudaGetLastError();
    magma_queue_sync(Q); cudaError_t e2 = cudaGetLastError();
    printf("zgbtrf_batched (aliased) batch=%d m=n=%d kl=ku=%d: returned %lld, launch error=%s, after sync=%s\n", batch, m, kl, (long long)ret,
           cudaGetErrorString(e1), cudaGetErrorString(e2));
    magma_free(dAB); magma_free(dpiv); magma_free(dinfo); magma_free(dAa); magma_free(dPa);
  }
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : "";
  magma_init(); magma_queue_create(0, &Q);
  cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0); printf("device: %s, sm_%d%d, shared mem opt-in %zu B\n", pr.name, pr.major, pr.minor, pr.sharedMemPerBlockOptin);
  if (m == "hegvdx2") hegvdx2(); else if (m == "gesdd") gesdd(); else if (m == "ungtr") ungtr(); else if (m == "gbstale") gbstale();
  else if (m == "argnum") argnum(); else if (m == "he2hb") he2hb(); else if (m == "geqlf") geqlf(); else if (m == "gbbatch") gbbatch(); else if (m == "gbalias") gbalias();
  else printf("unknown mode %s\n", m.c_str());
  printf("cudaGetLastError at exit: %s\n", cudaGetErrorString(cudaGetLastError()));
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

// GPU checks of the dense status-loss audit (HANDOFF §28.9). Complex (z) routines, real-valued data. GPU 0 only.
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
// defined in src/zgbtrs_batched.cpp but not declared in the public headers of v2.10.0
extern "C" magma_int_t magma_zgbtrs_batched_strided(magma_trans_t transA, magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA, magma_int_t* dipiv, magma_int_t stride_piv,
    magmaDoubleComplex* dB, magma_int_t lddb, magma_int_t strideB, magma_int_t* info_array, magma_int_t batchCount, magma_queue_t queue);
static magma_queue_t Q;
typedef magmaDoubleComplex Z;
static Z R(double x) { return MAGMA_Z_MAKE(x, 0.0); }
static unsigned S = 4242u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static int nonfin(const std::vector<Z>& v) { int c = 0; for (auto z : v) if (!std::isfinite(MAGMA_Z_REAL(z)) || !std::isfinite(MAGMA_Z_IMAG(z))) c++; return c; }

// T1: gbtrs_batched_strided after a correct gbtrf_batched_strided
static void t1() {
  int n = 8, kl = 1, ku = 1, ldab = 2 * kl + ku + 1, batch = 4, nrhs = 1;
  std::vector<Z> AB(ldab * n * batch, R(0)), B(n * batch), X(n * batch);
  for (int b = 0; b < batch; b++) {
    std::vector<double> xt(n); for (int i = 0; i < n; i++) xt[i] = i + 1 + 10 * b;
    auto A = [&](int i, int j) -> double { if (i == j) return 4.0 + b; if (abs(i - j) == 1) return -1.0 - 0.1 * b; return 0.0; };
    for (int j = 0; j < n; j++) for (int i = std::max(0, j - ku); i <= std::min(n - 1, j + kl); i++) AB[b * ldab * n + (kl + ku + i - j) + j * ldab] = R(A(i, j));
    for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += A(i, j) * xt[j]; B[b * n + i] = R(s); X[b * n + i] = R(xt[i]); }
  }
  magmaDoubleComplex_ptr dAB, dB; magma_int_t *dipiv, *dinfo;
  magma_zmalloc(&dAB, ldab * n * batch); magma_zmalloc(&dB, n * batch); magma_imalloc(&dipiv, n * batch); magma_imalloc(&dinfo, batch);
  magma_zsetvector(ldab * n * batch, AB.data(), 1, dAB, 1, Q); magma_zsetvector(n * batch, B.data(), 1, dB, 1, Q);
  magma_int_t r1 = magma_zgbtrf_batched_strided(n, n, kl, ku, dAB, ldab, ldab * n, dipiv, n, dinfo, batch, Q);
  std::vector<magma_int_t> inf1(batch); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, inf1.data(), 1, Q);
  magma_int_t r2 = magma_zgbtrs_batched_strided(MagmaNoTrans, n, kl, ku, nrhs, dAB, ldab, ldab * n, dipiv, n, dB, n, n, dinfo, batch, Q);
  std::vector<Z> got(n * batch); magma_zgetvector(n * batch, dB, 1, got.data(), 1, Q);
  for (int b = 0; b < batch; b++) { double e = 0, m = 0;
    for (int i = 0; i < n; i++) { e = fmax(e, fabs(MAGMA_Z_REAL(got[b * n + i]) - MAGMA_Z_REAL(X[b * n + i]))); m = fmax(m, fabs(MAGMA_Z_REAL(X[b * n + i]))); }
    printf("T1 gbtrs_batched_strided batch %d: gbtrf ret=%lld info=%lld  gbtrs ret=%lld  relerr=%.2e  x[0]=%g (true %g)\n", b, (long long)r1,
           (long long)inf1[b], (long long)r2, e / m, MAGMA_Z_REAL(got[b * n]), MAGMA_Z_REAL(X[b * n])); }
  magma_free(dAB); magma_free(dB); magma_free(dipiv); magma_free(dinfo);
}

// T2: heevdx_2stage Upper vs Lower vs LAPACK zheev
static void t2() {
  int n = 512; std::vector<Z> A0(n * n);
  for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { double v = rnd(); A0[i + j * n] = R(v); A0[j + i * n] = R(v); }
  // reference eigenvalues
  std::vector<Z> Ar = A0, wq(1); std::vector<double> wr(n), rw(3 * n); magma_int_t lw = -1, inf, nn = n;
  lapackf77_zheev("N", "L", &nn, Ar.data(), &nn, wr.data(), wq.data(), &lw, rw.data(), &inf);
  lw = (magma_int_t)MAGMA_Z_REAL(wq[0]); std::vector<Z> w2(lw); lapackf77_zheev("N", "L", &nn, Ar.data(), &nn, wr.data(), w2.data(), &lw, rw.data(), &inf);
  for (magma_uplo_t up : {MagmaLower, MagmaUpper}) {
    std::vector<Z> A = A0; std::vector<double> W(n); magma_int_t m = 0, info = -999;
    Z wkq; double rwq; magma_int_t iwq, lwork = -1, lrwork = -1, liwork = -1;
    magma_zheevdx_2stage(MagmaVec, MagmaRangeAll, up, n, A.data(), n, 0, 0, 0, 0, &m, W.data(), &wkq, lwork, &rwq, lrwork, &iwq, liwork, &info);
    lwork = (magma_int_t)MAGMA_Z_REAL(wkq); lrwork = (magma_int_t)rwq; liwork = iwq;
    std::vector<Z> wk(lwork); std::vector<double> rwk(lrwork); std::vector<magma_int_t> iwk(liwork);
    info = -999;
    magma_zheevdx_2stage(MagmaVec, MagmaRangeAll, up, n, A.data(), n, 0, 0, 0, 0, &m, W.data(), wk.data(), lwork, rwk.data(), lrwork, iwk.data(), liwork, &info);
    double e = 0; for (int i = 0; i < n; i++) e = fmax(e, fabs(W[i] - wr[i]));
    printf("T2 heevdx_2stage uplo=%s n=%d: info=%lld m=%lld  max|lambda - lambda_LAPACK|=%.2e  (lambda range %.3f..%.3f)\n",
           up == MagmaUpper ? "Upper" : "Lower", n, (long long)info, (long long)m, e, wr[0], wr[n - 1]);
  }
}

// T3: hesv_nopiv on a matrix whose non-pivoted LDL^T has a zero (last) pivot
static void t3() {
  struct C { const char* nm; int n; std::vector<double> A; std::vector<double> b; };
  std::vector<C> cs = { {"[[1,1],[1,1]] (singular, D2 = 0)", 2, {1, 1, 1, 1}, {1, 1}},
                        {"[[0]] n=1", 1, {0}, {1}},
                        {"[[2,1],[1,2]] (control)", 2, {2, 1, 1, 2}, {3, 3}} };
  for (auto& c : cs) {
    int n = c.n; std::vector<Z> A(n * n), b(n); for (int i = 0; i < n * n; i++) A[i] = R(c.A[i]); for (int i = 0; i < n; i++) b[i] = R(c.b[i]);
    magmaDoubleComplex_ptr dA, dB; magma_zmalloc(&dA, n * n); magma_zmalloc(&dB, n);
    magma_zsetmatrix(n, n, A.data(), n, dA, n, Q); magma_zsetmatrix(n, 1, b.data(), n, dB, n, Q);
    magma_int_t info = -999; magma_zhesv_nopiv_gpu(MagmaLower, n, 1, dA, n, dB, n, &info);
    std::vector<Z> x(n); magma_zgetmatrix(n, 1, dB, n, x.data(), n, Q);
    printf("T3 hesv_nopiv_gpu %-34s info=%lld  nonfinite(x)=%d  x[0]=%g\n", c.nm, (long long)info, nonfin(x), MAGMA_Z_REAL(x[0]));
    magma_free(dA); magma_free(dB);
  }
}

// T4: potrf native Upper and potf2_gpu with A(0,0) < 0
static void t4() {
  for (int n : {8, 256}) for (magma_uplo_t up : {MagmaUpper, MagmaLower}) {
    std::vector<Z> A(n * n, R(0)); for (int i = 0; i < n; i++) A[i + i * n] = R(4); for (int i = 0; i + 1 < n; i++) { A[i + (i + 1) * n] = R(-1); A[i + 1 + i * n] = R(-1); }
    A[0] = R(-1);   // not positive definite: LAPACK zpotrf returns info = 1
    std::vector<Z> Al = A; magma_int_t nn = n, linfo; lapackf77_zpotrf(up == MagmaUpper ? "U" : "L", &nn, Al.data(), &nn, &linfo);
    magmaDoubleComplex_ptr dA; magma_zmalloc(&dA, n * n);
    magma_zsetmatrix(n, n, A.data(), n, dA, n, Q); magma_int_t info = -999; magma_zpotrf_native(up, n, dA, n, &info);
    magma_zsetmatrix(n, n, A.data(), n, dA, n, Q); magma_int_t info2 = -999; magma_zpotf2_gpu(up, n, dA, n, Q, &info2);
    std::vector<Z> F(n * n); magma_zgetmatrix(n, n, dA, n, F.data(), n, Q);
    printf("T4 n=%-3d uplo=%s A(0,0)=-1: LAPACK zpotrf info=%lld | magma_zpotrf_native info=%lld | magma_zpotf2_gpu info=%lld (nonfinite factor entries %d)\n",
           n, up == MagmaUpper ? "Upper" : "Lower", (long long)linfo, (long long)info, (long long)info2, nonfin(F));
    magma_free(dA);
  }
}

// T5: trtri_gpu on an upper-triangular matrix with a zero diagonal entry
static void t5() {
  int n = 512;
  for (int zpos : {0, 300, 511}) {
    std::vector<Z> A(n * n, R(0)); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) A[i + j * n] = R(i == j ? 2.0 : 0.01 * rnd());
    A[zpos + zpos * n] = R(0);
    std::vector<Z> Al = A; magma_int_t nn = n, linfo; lapackf77_ztrtri("U", "N", &nn, Al.data(), &nn, &linfo);
    magmaDoubleComplex_ptr dA; magma_zmalloc(&dA, n * n); magma_zsetmatrix(n, n, A.data(), n, dA, n, Q);
    magma_int_t info = -999; magma_ztrtri_gpu(MagmaUpper, MagmaNonUnit, n, dA, n, &info);
    std::vector<Z> F(n * n); magma_zgetmatrix(n, n, dA, n, F.data(), n, Q);
    printf("T5 trtri_gpu n=%d, A(%d,%d)=0: LAPACK ztrtri info=%lld | magma_ztrtri_gpu info=%lld  (nonfinite entries in returned inverse: %d)\n",
           n, zpos, zpos, (long long)linfo, (long long)info, nonfin(F));
    magma_free(dA);
  }
}

int main() {
  magma_init(); magma_queue_create(0, &Q);
  t1(); t2(); t3(); t4(); t5();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

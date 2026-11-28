// Round 27 follow-up (roxy): focused reproductions of sweep hits that survived triage (HANDOFF §29.42).
//   round27_focus geqp3   zgeqp3 on a matrix whose last column duplicates the first: is jpvt a permutation, A P = Q R?
//   round27_focus blas    magmablas_zgemm k = 0 (C := beta C expected), zgemv / zhemv beta = 0 with NaN in y
//   round27_focus gels    magma_zgels / zgels_gpu on a zero matrix and on a matrix with a zero column, vs LAPACK zgels
//   round27_focus trtri0  magma_ztrtri_gpu with n = 0
//   round27_focus heevdgpu  zheevd_gpu(N) on scaled Hermitian matrices
//   round27_focus small     n = 1 (mout) in the range eigensolvers; zheevd_gpu with n = 0
//   round27_focus nzero     n = 0 on GPU-interface routines whose small path passes lda = n to LAPACK
//   round27_focus svdinf <lapack|magma> <gesvd|gesdd> <N|S>   3 x 3 SVD with an Inf entry (hang search)
//   round27_focus geqp3rank  zgeqp3 on random low-rank matrices (rank n/2 and n-1)
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <complex>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
typedef std::complex<double> cd;
static magmaDoubleComplex* zp(std::vector<cd>& v) { return (magmaDoubleComplex*)v.data(); }
static magma_queue_t Q;
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static cd rc() { double a = r1(); return cd(a, r1()); }
static bool fin(const std::vector<cd>& v) { for (auto& a : v) if (!std::isfinite(a.real()) || !std::isfinite(a.imag())) return false; return true; }

static void geqp3() {
  for (int n : {31, 32, 63, 64, 65, 127, 128, 255, 256}) {
    int m = n; Sd = 3000u + 3 * m + n; std::vector<cd> A0((size_t)m * n); for (auto& a : A0) a = rc();   // same seed as the sweep
    for (int i = 0; i < m; i++) A0[i + (size_t)(n - 1) * m] = A0[i];
    for (int impl = 0; impl < 2; impl++) {
      auto A = A0; std::vector<magma_int_t> jp(n, 0); std::vector<cd> tau(n); magma_int_t lw = (n + 1) * 64 + 4096, info = -999; std::vector<cd> w(lw); std::vector<double> rw(2 * n);
      magma_int_t mm = m, nn = n, lda = m;
      if (impl == 0) lapackf77_zgeqp3(&mm, &nn, zp(A), &lda, jp.data(), zp(tau), zp(w), &lw, rw.data(), &info);
      else magma_zgeqp3(m, n, zp(A), lda, jp.data(), zp(tau), zp(w), lw, rw.data(), &info);
      std::vector<int> seen(n, 0); int dup = 0, bad = 0; for (int j = 0; j < n; j++) { int p = (int)jp[j] - 1; if (p < 0 || p >= n) bad++; else if (seen[p]++) dup++; }
      // A P = Q R with Q from zungqr
      std::vector<cd> Qm = A; magma_int_t il, lwq = 64 * n; std::vector<cd> wq(lwq); lapackf77_zungqr(&mm, &nn, &nn, zp(Qm), &lda, zp(tau), zp(wq), &lwq, &il);
      double err = 0, nrm = 0; int worstcol = -1; double worst = 0;
      for (int j = 0; j < n; j++) { int p = (int)jp[j] - 1; if (p < 0 || p >= n) continue; double ce = 0;
        for (int i = 0; i < m; i++) { cd s = 0; for (int l = 0; l <= std::min(j, m - 1); l++) s += Qm[i + (size_t)l * m] * A[l + (size_t)j * m]; ce = std::max(ce, std::abs(s - A0[i + (size_t)p * m])); nrm = std::max(nrm, std::abs(A0[i + (size_t)p * m])); }
        if (ce > worst) { worst = ce; worstcol = j; } err = std::max(err, ce); }
      printf("%-6s zgeqp3 n=%3d info=%lld  jpvt: out-of-range=%d duplicates=%d  max|AP-QR|/max|A|=%.3g (worst column %d)  |R(n-1,n-1)|=%.3g  last pivots:",
             impl ? "MAGMA" : "LAPACK", n, (long long)info, bad, dup, err / nrm, worstcol, std::abs(A[(n - 1) + (size_t)(n - 1) * m]));
      for (int j = std::max(0, n - 4); j < n; j++) printf(" %lld", (long long)jp[j]); printf("\n");
    }
  }
}

static void blas() {
  for (int n : {1, 32, 65}) { // gemm with k = 0: C := beta C
    std::vector<cd> C((size_t)n * n); Sd = 5u; for (auto& c : C) c = rc(); magmaDoubleComplex* dC; magma_zmalloc(&dC, (size_t)n * n); magma_zsetmatrix(n, n, zp(C), n, dC, n, Q);
    magmaDoubleComplex *dA, *dB; magma_zmalloc(&dA, n); magma_zmalloc(&dB, n);
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(1, 0), beta = MAGMA_Z_MAKE(0.5, 0);
    magmablas_zgemm(MagmaNoTrans, MagmaNoTrans, n, n, 0, alpha, dA, n, dB, 1, beta, dC, n, Q);
    std::vector<cd> R((size_t)n * n); magma_zgetmatrix(n, n, dC, n, zp(R), n, Q);
    printf("magmablas_zgemm m=n=%d k=0 beta=0.5: C[0] before %.4f%+.4fi, after %.4f%+.4fi (expected %.4f%+.4fi)\n", n, C[0].real(), C[0].imag(), R[0].real(), R[0].imag(), 0.5 * C[0].real(), 0.5 * C[0].imag());
    std::vector<cd> Cn((size_t)n * n, cd(NAN, NAN)); magma_zsetmatrix(n, n, zp(Cn), n, dC, n, Q); beta = MAGMA_Z_ZERO;
    magmablas_zgemm(MagmaNoTrans, MagmaNoTrans, n, n, 0, alpha, dA, n, dB, 1, beta, dC, n, Q); magma_zgetmatrix(n, n, dC, n, zp(R), n, Q);
    printf("magmablas_zgemm m=n=%d k=0 beta=0, C = NaN: C[0] after %.4f%+.4fi (expected 0)\n", n, R[0].real(), R[0].imag());
    magma_free(dA); magma_free(dB); magma_free(dC);
  }
  for (int n : {1, 64}) for (int herm = 0; herm < 2; herm++) { // beta = 0, y = NaN
    std::vector<cd> A((size_t)n * n, cd(0)), x(n, cd(1)), y(n, cd(NAN, NAN)); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = 2;
    magmaDoubleComplex *dA, *dx, *dy; magma_zmalloc(&dA, (size_t)n * n); magma_zmalloc(&dx, n); magma_zmalloc(&dy, n);
    magma_zsetmatrix(n, n, zp(A), n, dA, n, Q); magma_zsetvector(n, zp(x), 1, dx, 1, Q); magma_zsetvector(n, zp(y), 1, dy, 1, Q);
    if (herm) magmablas_zhemv(MagmaLower, n, MAGMA_Z_ONE, dA, n, dx, 1, MAGMA_Z_ZERO, dy, 1, Q);
    else magmablas_zgemv(MagmaNoTrans, n, n, MAGMA_Z_ONE, dA, n, dx, 1, MAGMA_Z_ZERO, dy, 1, Q);
    std::vector<cd> R(n); magma_zgetvector(n, dy, 1, zp(R), 1, Q);
    printf("magmablas_%s n=%d A=2I x=1 beta=0 y=NaN: y[0] = %.4f%+.4fi (expected 2)\n", herm ? "zhemv" : "zgemv", n, R[0].real(), R[0].imag());
    magma_free(dA); magma_free(dx); magma_free(dy);
  }
}

static void gels() {
  for (int n : {1, 8, 64}) for (int kind = 0; kind < 2; kind++) {
    std::vector<cd> A0((size_t)n * n), B0(n); Sd = 9u + n; for (auto& a : A0) a = rc(); for (auto& b : B0) b = rc();
    if (kind == 0) std::fill(A0.begin(), A0.end(), cd(0)); else for (int i = 0; i < n; i++) A0[i] = 0;
    magma_int_t nn = n, one = 1, lda = n, lw = 64 * n + 4096, il, im = -999; std::vector<cd> w(lw);
    auto Ar = A0, Br = B0; lapackf77_zgels("N", &nn, &nn, &one, zp(Ar), &lda, zp(Br), &lda, zp(w), &lw, &il);
    auto A = A0, B = B0; magma_zgels(MagmaNoTrans, n, n, 1, zp(A), lda, zp(B), lda, zp(w), lw, &im);
    printf("zgels n=%d %-10s LAPACK info=%lld x finite=%d | MAGMA info=%lld x finite=%d x[0]=%.3g%+.3gi\n", n, kind ? "zero col 1" : "zero matrix",
           (long long)il, fin(Br), (long long)im, fin(B), B[0].real(), B[0].imag());
  }
}

static void trtri0() {
  magmaDoubleComplex* d; magma_zmalloc(&d, 1); magma_int_t info = -999;
  magma_ztrtri_gpu(MagmaUpper, MagmaNonUnit, 0, d, 1, &info); printf("magma_ztrtri_gpu n=0 ldda=1: info=%lld\n", (long long)info);
  std::vector<cd> A(1); info = -999; magma_ztrtri(MagmaUpper, MagmaNonUnit, 0, zp(A), 1, &info); printf("magma_ztrtri     n=0 lda=1:  info=%lld\n", (long long)info);
  magma_int_t z = 0, one = 1, il; lapackf77_ztrtri("U", "N", &z, zp(A), &one, &il); printf("LAPACK ztrtri    n=0 lda=1:  info=%lld\n", (long long)il);
  magma_free(d);
}

// low-rank matrices A = U V^H with rank r = n/2 (and r = n-1): A P = Q R residual and the rank gap |R(r,r)| / |R(0,0)|
static void geqp3rank() {
  int fails = 0, cases = 0;
  for (int n : {48, 63, 64, 65, 96, 100, 127, 128, 129, 200, 255, 256, 300}) for (int rr = 0; rr < 2; rr++) for (unsigned seed = 1; seed <= 3; seed++) {
    int m = n, r = rr == 0 ? n / 2 : n - 1; Sd = 777u * seed + n; std::vector<cd> U((size_t)m * r), V((size_t)n * r); for (auto& u : U) u = rc(); for (auto& v : V) v = rc();
    std::vector<cd> A0((size_t)m * n); for (int j = 0; j < n; j++) for (int l = 0; l < r; l++) { cd vv = std::conj(V[j + (size_t)l * n]); for (int i = 0; i < m; i++) A0[i + (size_t)j * m] += U[i + (size_t)l * m] * vv; }
    auto A = A0; std::vector<magma_int_t> jp(n, 0); std::vector<cd> tau(n); magma_int_t lw = (n + 1) * 64 + 4096, info = -999; std::vector<cd> w(lw); std::vector<double> rw(2 * n);
    magma_zgeqp3(m, n, zp(A), m, jp.data(), zp(tau), zp(w), lw, rw.data(), &info);
    magma_int_t mm = m, nn = n, lda = m, il, lwq = 64 * n; std::vector<cd> Qm = A, wq(lwq); lapackf77_zungqr(&mm, &nn, &nn, zp(Qm), &lda, zp(tau), zp(wq), &lwq, &il);
    double err = 0, nrm = 0; for (int j = 0; j < n; j++) { int p = (int)jp[j] - 1; for (int i = 0; i < m; i++) { cd sum = 0; for (int l = 0; l <= std::min(j, m - 1); l++) sum += Qm[i + (size_t)l * m] * A[l + (size_t)j * m];
        err = std::max(err, std::abs(sum - A0[i + (size_t)p * m])); nrm = std::max(nrm, std::abs(A0[i + (size_t)p * m])); } }
    double gap = std::abs(A[r + (size_t)r * m]) / std::abs(A[0]); bool bad = err / nrm > 1e-10 || gap > 1e-10; fails += bad; cases++;
    if (bad) printf("zgeqp3 n=%3d rank=%3d seed=%u: max|AP-QR|/max|A|=%.3g  |R(r,r)|/|R(0,0)|=%.3g\n", n, r, seed, err / nrm, gap);
  }
  printf("zgeqp3 low-rank: %d of %d cases wrong (residual > 1e-10 or rank not revealed)\n", fails, cases);
}

static std::vector<cd> hermA(int n, unsigned seed, double scale) { Sd = seed; std::vector<cd> A((size_t)n * n);
  for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { cd v = i == j ? cd(r1(), 0) : rc(); A[i + (size_t)j * n] = v * scale; A[j + (size_t)i * n] = std::conj(v) * scale; } return A; }
// zheevd_gpu(jobz = N) on scaled Hermitian matrices; zheevd (CPU interface) and LAPACK zheev as references
static void heevdgpu() {
  for (int n : {129, 130, 160, 200, 256, 257}) for (int e : {0, 300, 500, 510}) {
    double sc = ldexp(1.0, e); auto A0 = hermA(n, 9000u + n, sc); magma_int_t nn = n, lda = n, il; std::vector<double> wr(n);
    { auto R = A0; magma_int_t lw = 4 * n; std::vector<cd> w(lw); std::vector<double> rw(3 * n); lapackf77_zheev("N", "L", &nn, zp(R), &lda, wr.data(), zp(w), &lw, rw.data(), &il); }
    auto A = A0; std::vector<double> w(n); magma_int_t lw = n + n * 64 + 4096, lrw = n + 4096, liw = 64, info = -999; std::vector<cd> wk(lw), wA((size_t)n * n); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw);
    magmaDoubleComplex* d; magma_int_t ldd = magma_roundup(n, 32); magma_zmalloc(&d, (size_t)ldd * n); magma_zsetmatrix(n, n, zp(A), n, d, ldd, Q);
    magma_zheevd_gpu(MagmaNoVec, MagmaLower, n, d, ldd, w.data(), zp(wA), n, zp(wk), lw, rw.data(), lrw, iw.data(), liw, &info); magma_free(d);
    double err = 0, mx = 0; for (int i = 0; i < n; i++) { err = std::max(err, std::fabs(w[i] - wr[i])); mx = std::max(mx, std::fabs(wr[i])); }
    printf("zheevd_gpu(N) n=%3d scale=2^%-3d info=%lld  max|w-w_LAPACK|/max|w| = %.3g   w[0]=%.4g (LAPACK %.4g)\n", n, e, (long long)info, err / mx, w[0], wr[0]);
  }
}
// n = 1: is mout set?  n = 0 on the GPU interface
static void small() {
  std::vector<cd> A(1, cd(3, 0)), wk(4096), Zm(1); std::vector<double> w(1), rw(4096); std::vector<magma_int_t> iw(4096), ifl(1), isup(2);
  for (int v = 0; v < 4; v++) { A[0] = cd(3, 0); magma_int_t mo = -77, info = -999; w[0] = NAN;
    const char* nm[] = {"zheevdx", "zheevdx_2stage", "zheevr", "zheevx"};
    if (v == 0) magma_zheevdx(MagmaVec, MagmaRangeAll, MagmaLower, 1, zp(A), 1, 0, 0, 1, 1, &mo, w.data(), zp(wk), 4096, rw.data(), 4096, iw.data(), 4096, &info);
    if (v == 1) magma_zheevdx_2stage(MagmaVec, MagmaRangeAll, MagmaLower, 1, zp(A), 1, 0, 0, 1, 1, &mo, w.data(), zp(wk), 4096, rw.data(), 4096, iw.data(), 4096, &info);
    if (v == 2) magma_zheevr(MagmaVec, MagmaRangeAll, MagmaLower, 1, zp(A), 1, 0, 0, 1, 1, 0, &mo, w.data(), zp(Zm), 1, isup.data(), zp(wk), 4096, rw.data(), 4096, iw.data(), 4096, &info);
    if (v == 3) magma_zheevx(MagmaVec, MagmaRangeAll, MagmaLower, 1, zp(A), 1, 0, 0, 1, 1, 0, &mo, w.data(), zp(Zm), 1, zp(wk), 4096, rw.data(), iw.data(), ifl.data(), &info);
    printf("%-15s n=1 range=All: info=%lld  mout=%lld (caller set -77; LAPACK sets 1)  w[0]=%g\n", nm[v], (long long)info, (long long)mo, w[0]); }
  for (magma_int_t ld : {1, 32}) { magmaDoubleComplex* d; magma_zmalloc(&d, 32); magma_int_t info = -999;
    magma_zheevd_gpu(MagmaNoVec, MagmaLower, 0, d, ld, w.data(), zp(wk), 1, zp(wk), 4096, rw.data(), 4096, iw.data(), 4096, &info);
    printf("zheevd_gpu n=0 ldda=%lld ldwa=1: info=%lld\n", (long long)ld, (long long)info); magma_free(d); }
}

// repeat the sweep's exact zheevd_gpu(N) call (same input, same workspace sizes) to see whether the n = 129 failure recurs
static void heevdrep() {
  for (int n : {129, 200}) for (int e : {0, 500}) {
    int bad = 0; double worst = 0; const int R = 30;
    auto A0 = hermA(n, 9000u + n, ldexp(1.0, e)); magma_int_t nn = n, lda = n, il; std::vector<double> wr(n);
    { auto Rr = A0; magma_int_t lw = 4 * n; std::vector<cd> w(lw); std::vector<double> rw(3 * n); lapackf77_zheev("N", "L", &nn, zp(Rr), &lda, wr.data(), zp(w), &lw, rw.data(), &il); }
    double mx = 0; for (double v : wr) mx = std::max(mx, std::fabs(v));
    for (int rep = 0; rep < R; rep++) {
      magma_int_t lw = 2 * n + n * n + n * 64 + 4096 + 400 * n, lrw = 1 + 5 * n + 2 * n * n + 4096, liw = 3 + 5 * n + 64, info = -999;
      std::vector<cd> wk(lw), wA((size_t)n * n); std::vector<double> rw(lrw), w(n); std::vector<magma_int_t> iw(liw);
      int ld = std::max(1, (int)magma_roundup(n, 32)); magmaDoubleComplex* d; magma_zmalloc(&d, (size_t)ld * n); auto A = A0; magma_zsetmatrix(n, n, zp(A), n, d, ld, Q);
      magma_zheevd_gpu(MagmaNoVec, MagmaLower, n, d, ld, w.data(), zp(wA), lda, zp(wk), lw, rw.data(), lrw, iw.data(), liw, &info); magma_free(d);
      double err = 0; for (int i = 0; i < n; i++) err = std::max(err, std::fabs(w[i] - wr[i])); err /= mx; worst = std::max(worst, err); if (!(err < 1e-12) || info) bad++;
    }
    printf("zheevd_gpu(N) n=%d scale=2^%d, sweep workspace sizes, %d runs: %d wrong, worst max|w-w_LAPACK|/max|w| = %.3g\n", n, e, R, bad, worst);
  }
}

// n = 0 on GPU-interface routines whose small-matrix path calls LAPACK with lda = n
static void nzero() {
  magmaDoubleComplex *d, *dZ; magma_zmalloc(&d, 32); magma_zmalloc(&dZ, 32); std::vector<cd> wk(8192), wA(32), wZ(32); std::vector<double> w(1), rw(8192);
  std::vector<magma_int_t> iw(8192), ifl(1), isup(2); magma_int_t mo = -77, info;
  info = -999; magma_zheevdx_gpu(MagmaVec, MagmaRangeAll, MagmaLower, 0, d, 1, 0, 0, 1, 0, &mo, w.data(), zp(wA), 1, zp(wk), 8192, rw.data(), 8192, iw.data(), 8192, &info);
  printf("zheevdx_gpu n=0: info=%lld mout=%lld\n", (long long)info, (long long)mo);
  info = -999; mo = -77; magma_zheevr_gpu(MagmaVec, MagmaRangeAll, MagmaLower, 0, d, 1, 0, 0, 1, 0, 0, &mo, w.data(), dZ, 1, isup.data(), zp(wA), 1, zp(wZ), 1, zp(wk), 8192, rw.data(), 8192, iw.data(), 8192, &info);
  printf("zheevr_gpu  n=0: info=%lld mout=%lld\n", (long long)info, (long long)mo);
  info = -999; mo = -77; magma_zheevx_gpu(MagmaVec, MagmaRangeAll, MagmaLower, 0, d, 1, 0, 0, 1, 0, 0, &mo, w.data(), dZ, 1, zp(wA), 1, zp(wZ), 1, zp(wk), 8192, rw.data(), iw.data(), ifl.data(), &info);
  printf("zheevx_gpu  n=0: info=%lld mout=%lld\n", (long long)info, (long long)mo);
  magmaDoubleComplex* dl[1] = {d}; info = -999; magma_zpotrf_mgpu(1, MagmaLower, 0, dl, 1, &info); printf("zpotrf_mgpu ngpu=1 n=0: info=%lld\n", (long long)info);
  info = -999; magma_zpotrf_gpu(MagmaLower, 0, d, 1, &info); printf("zpotrf_gpu  n=0: info=%lld\n", (long long)info);
  magma_free(d); magma_free(dZ);
}

// which SVD hangs on a 3 x 3 matrix with A(0,0) = Inf?  one call per process: svdinf <lapack|magma> <gesvd|gesdd> <N|S>
static void svdinf(const std::string& who, const std::string& rt, const std::string& job) {
  const int n = 3; Sd = 11000u + 5 * n + n; std::vector<cd> A((size_t)n * n); for (auto& a : A) a = rc(); A[0] = cd(INFINITY, 0);
  std::vector<double> s(n), rw(5 * n * n + 7 * n + 64); std::vector<cd> U(n * n), VT(n * n), w(65536); std::vector<magma_int_t> iw(8 * n);
  magma_int_t nn = n, lda = n, lw = 65536, info = -999; char jb = job[0]; magma_vec_t mj = jb == 'N' ? MagmaNoVec : MagmaSomeVec;
  printf("start %s %s job=%c\n", who.c_str(), rt.c_str(), jb);
  if (who == "lapack" && rt == "gesvd") lapackf77_zgesvd(&jb, &jb, &nn, &nn, zp(A), &lda, s.data(), zp(U), &lda, zp(VT), &lda, zp(w), &lw, rw.data(), &info);
  if (who == "lapack" && rt == "gesdd") lapackf77_zgesdd(&jb, &nn, &nn, zp(A), &lda, s.data(), zp(U), &lda, zp(VT), &lda, zp(w), &lw, rw.data(), iw.data(), &info);
  if (who == "magma" && rt == "gesvd") magma_zgesvd(mj, mj, n, n, zp(A), lda, s.data(), zp(U), lda, zp(VT), lda, zp(w), lw, rw.data(), &info);
  if (who == "magma" && rt == "gesdd") magma_zgesdd(mj, n, n, zp(A), lda, s.data(), zp(U), lda, zp(VT), lda, zp(w), lw, rw.data(), iw.data(), &info);
  printf("done  %s %s job=%c info=%lld s0=%g\n", who.c_str(), rt.c_str(), jb, (long long)info, s[0]);
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : "";
  magma_init(); magma_queue_create(0, &Q);
  if (m == "geqp3") geqp3(); else if (m == "heevdgpu") heevdgpu(); else if (m == "heevdrep") heevdrep(); else if (m == "nzero") nzero(); else if (m == "svdinf" && argc > 4) svdinf(argv[2], argv[3], argv[4]); else if (m == "small") small(); else if (m == "geqp3rank") geqp3rank(); else if (m == "blas") blas(); else if (m == "gels") gels(); else if (m == "trtri0") trtri0(); else printf("unknown\n");
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

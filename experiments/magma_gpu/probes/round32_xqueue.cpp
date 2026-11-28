// Round 32 (roxy): GPU tests of the cross-queue hand-offs found by the static scan (scripts/xqueue_scan.py).
// Pattern: asynchronous work on the caller's queue, then a callee that creates a fresh queue (cudaStreamCreate, no event,
// no sync) and touches the same data.  magma_queue_destroy syncs the device (cublasDestroy), so only caller -> callee
// hand-offs can race.  Run once normally and once with CUDA_LAUNCH_BLOCKING=1.
//     round32_xqueue heev  [R]   zheevd_gpu / dsyevd_gpu / zheevdx_gpu: magmablas_?lascl (iscale = 1 when ||A|| < ~1.5e-146
//                                or > ~7e145) on the driver's queue, then ?hetrd_gpu / dsytrd2_gpu copies dA to the host on its own queue
//     round32_xqueue rbt   [R]   zgesv_rbt(refine): zcopymatrix(dA -> dAo, dB -> dBo) on its queue, then zgesv_nopiv_gpu overwrites dA, dB
// CSV heev: heev,routine,n,scale,rep,info,relerr   (max |w - w_true| / max |w_true|; w_true known by construction)
// CSV rbt:  rbt,n,rep,info,berr_ratio
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
typedef std::complex<double> cd;
static magma_queue_t Q;
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static std::vector<double> orth(int n) { std::vector<double> X((size_t)n * n); for (auto& x : X) x = r1(); magma_int_t nn = n, lw = 64 * n, il; std::vector<double> tau(n), w(lw);
  lapackf77_dgeqrf(&nn, &nn, X.data(), &nn, tau.data(), w.data(), &lw, &il); lapackf77_dorgqr(&nn, &nn, &nn, X.data(), &nn, tau.data(), w.data(), &lw, &il); return X; }

static void heev(int R) {
  printf("heev,routine,n,scale,rep,info,relerr\n");
  for (int n : {64, 300, 1000}) {
    Sd = 9u + n; auto U = orth(n); std::vector<double> lam(n); for (int i = 0; i < n; i++) lam[i] = 1.0 + i;   // eigenvalues 1..n
    std::vector<double> A0((size_t)n * n, 0); for (int l = 0; l < n; l++) for (int j = 0; j < n; j++) { double v = lam[l] * U[j + (size_t)l * n]; for (int i = 0; i < n; i++) A0[i + (size_t)j * n] += U[i + (size_t)l * n] * v; }
    for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) { double a = 0.5 * (A0[i + (size_t)j * n] + A0[j + (size_t)i * n]); A0[i + (size_t)j * n] = a; A0[j + (size_t)i * n] = a; }
    for (double sc : {1.0, 1e-150, 1e150}) for (int rt = 0; rt < 3; rt++) for (int r = 0; r < R; r++) {
      int ld = (int)magma_roundup(n, 32); std::vector<double> w(n); magma_int_t info = -999, m = 0;
      std::vector<magma_int_t> iw(5 * n + 16); magma_int_t liw = (magma_int_t)iw.size();
      if (rt == 1) {   // dsyevd_gpu
        magmaDouble_ptr dA; magma_dmalloc(&dA, (size_t)ld * n); std::vector<double> A = A0; for (auto& a : A) a *= sc; magma_dsetmatrix(n, n, A.data(), n, dA, ld, Q);
        double q; magma_int_t qi, lw = -1, lq = -1; magma_dsyevd_gpu(MagmaNoVec, MagmaLower, n, dA, ld, w.data(), NULL, n, &q, lw, &qi, lq, &info);
        lw = (magma_int_t)q + 64; std::vector<double> work(lw), wA((size_t)n * n); magma_dsyevd_gpu(MagmaNoVec, MagmaLower, n, dA, ld, w.data(), wA.data(), n, work.data(), lw, iw.data(), liw, &info);
        magma_free(dA);
      } else {          // zheevd_gpu (rt 0) / zheevdx_gpu range all (rt 2)
        magmaDoubleComplex_ptr dA; magma_zmalloc(&dA, (size_t)ld * n); std::vector<cd> A(A0.begin(), A0.end()); for (auto& a : A) a *= sc;
        magma_zsetmatrix(n, n, (magmaDoubleComplex*)A.data(), n, dA, ld, Q);
        magmaDoubleComplex q; double rq; magma_int_t qi;
        if (rt == 0) magma_zheevd_gpu(MagmaNoVec, MagmaLower, n, dA, ld, w.data(), NULL, n, &q, -1, &rq, -1, &qi, -1, &info);
        else magma_zheevdx_gpu(MagmaNoVec, MagmaRangeAll, MagmaLower, n, dA, ld, 0, 0, 1, n, &m, w.data(), NULL, n, &q, -1, &rq, -1, &qi, -1, &info);
        magma_int_t lw = (magma_int_t)MAGMA_Z_REAL(q) + 64, lrw = (magma_int_t)rq + 64; std::vector<magmaDoubleComplex> work(lw), wA((size_t)n * n); std::vector<double> rw(lrw);
        if (rt == 0) magma_zheevd_gpu(MagmaNoVec, MagmaLower, n, dA, ld, w.data(), wA.data(), n, work.data(), lw, rw.data(), lrw, iw.data(), liw, &info);
        else magma_zheevdx_gpu(MagmaNoVec, MagmaRangeAll, MagmaLower, n, dA, ld, 0, 0, 1, n, &m, w.data(), wA.data(), n, work.data(), lw, rw.data(), lrw, iw.data(), liw, &info);
        magma_free(dA);
      }
      double e = 0; for (int i = 0; i < n; i++) { double d = fabs(w[i] / sc - lam[i]); if (!(d <= e)) e = d; }   // NaN-propagating max
      const char* nm[] = {"zheevd_gpu", "dsyevd_gpu", "zheevdx_gpu"};
      printf("heev,%s,%d,%.0e,%d,%lld,%.3g\n", nm[rt], n, sc, r, (long long)info, e / n);
    }
  }
}

static void rbt(int R) {
  printf("rbt,n,rep,info,berr_ratio\n");
  for (int n : {256, 1000, 2000}) for (int r = 0; r < R; r++) {
    Sd = 3u + n; std::vector<cd> A((size_t)n * n), X((size_t)n), B((size_t)n, 0); for (auto& a : A) a = cd(r1(), r1()); for (int i = 0; i < n; i++) A[i + (size_t)i * n] += 2.0 * sqrt((double)n);
    for (auto& x : X) x = cd(r1(), r1()); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) B[i] += A[i + (size_t)j * n] * X[j];
    auto Ac = A; auto Bc = B; magma_int_t info = -999;
    magma_zgesv_rbt(MagmaTrue, n, 1, (magmaDoubleComplex*)Ac.data(), n, (magmaDoubleComplex*)Bc.data(), n, &info);
    double an = 0; for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += std::abs(A[i + (size_t)j * n]); an = std::max(an, s); }
    double rn = 0, xn = 0, bn = 0; for (int i = 0; i < n; i++) { cd s = B[i]; for (int j = 0; j < n; j++) s -= A[i + (size_t)j * n] * Bc[j]; double a = std::abs(s); if (!(a <= rn)) rn = a; xn = std::max(xn, std::abs(Bc[i])); bn = std::max(bn, std::abs(B[i])); }
    printf("rbt,%d,%d,%lld,%.3g\n", n, r, (long long)info, rn / ((an * xn + bn) * n * 1.1102230246251565e-16));
  }
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : "heev"; int R = argc > 2 ? atoi(argv[2]) : 5;
  magma_init(); magma_queue_create(0, &Q);
  printf("# CUDA_LAUNCH_BLOCKING=%s\n", getenv("CUDA_LAUNCH_BLOCKING") ? getenv("CUDA_LAUNCH_BLOCKING") : "unset");
  if (m == "heev") heev(R); else if (m == "rbt") rbt(R); else printf("unknown mode\n");
  fprintf(stderr, "DONE %s cuda=%s\n", m.c_str(), cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

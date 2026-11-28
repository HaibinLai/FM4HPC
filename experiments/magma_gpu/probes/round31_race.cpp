// Round 31 follow-up (roxy): are the dsgesv_gpu / zcgesv_gpu failures at n = 513 a cross-queue race?
// zcgesv_gpu runs its own work on queue q1, while zcgetrs_gpu and zgetrs_gpu each create a fresh queue
// (cudaStreamCreate, with no event or sync against q1). Data written on one stream is read on another without
// ordering, both in the IR loop (zcgetrs writes dX, then q1 runs gemv on dX) and in the fallback
// (zlacpy dB->dX on q1, then zgetrs runs laswp/trsm on dX on its own stream).
//     round31_race rep   cond 10, repeated R times, n in {200, 513, 1024}: count non-convergence (iter < 0) and wrong x
//     round31_race fb    A scaled by 2^200 so dlag2s overflows: always the pure FP64 fallback; count wrong x
// Run once normally and once with CUDA_LAUNCH_BLOCKING=1, which serialises every launch. If the failures disappear, it is a race.
// CSV: mode,solver,n,nrhs,rep,iter,info,berr_ratio,verdict
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
static const double EPS = 1.1102230246251565e-16;
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static std::vector<double> orth(int n) { std::vector<double> X((size_t)n * n); for (auto& x : X) x = r1(); magma_int_t nn = n, lw = 64 * n, il; std::vector<double> tau(n), w(lw);
  lapackf77_dgeqrf(&nn, &nn, X.data(), &nn, tau.data(), w.data(), &lw, &il); lapackf77_dorgqr(&nn, &nn, &nn, X.data(), &nn, tau.data(), w.data(), &lw, &il); return X; }
static std::vector<double> gen(int n, double cond, unsigned seed) { Sd = seed; auto U = orth(n), V = orth(n); std::vector<double> A((size_t)n * n);
  for (int l = 0; l < n; l++) { double s = pow(1.0 / cond, (double)l / (n - 1)); for (int j = 0; j < n; j++) { double v = s * V[j + (size_t)l * n]; for (int i = 0; i < n; i++) A[i + (size_t)j * n] += U[i + (size_t)l * n] * v; } }
  return A; }
template <class T> static double berr(const std::vector<double>& A, const std::vector<T>& X, const std::vector<T>& B, int n, int nrhs, bool& finX) {
  double an = 0; for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += fabs(A[i + (size_t)j * n]); an = std::max(an, s); }
  double worst = 0; finX = true;
  for (int c = 0; c < nrhs; c++) { double rn = 0, xn = 0, bn = 0; for (int i = 0; i < n; i++) { T s = B[i + (size_t)c * n]; for (int j = 0; j < n; j++) s -= A[i + (size_t)j * n] * X[j + (size_t)c * n];
      if (!std::isfinite(std::abs(X[i + (size_t)c * n]))) finX = false;
      rn = std::max(rn, std::abs(s)); xn = std::max(xn, std::abs(X[i + (size_t)c * n])); bn = std::max(bn, std::abs(B[i + (size_t)c * n])); }
    worst = std::max(worst, rn / ((an * xn + bn) * n * EPS)); }
  return worst; }

static void one(const char* mode, bool z, int n, int nrhs, int rep, const std::vector<double>& A) {
  int ld = (int)magma_roundup(n, 32); magma_int_t iter = -999, info = -999; std::vector<magma_int_t> ip(n); magma_int_t* dipiv; magma_imalloc(&dipiv, n);
  Sd = 7u + n; double br; bool fx;
  if (!z) {
    std::vector<double> Xt((size_t)n * nrhs), B((size_t)n * nrhs, 0), X((size_t)n * nrhs); for (auto& x : Xt) x = r1();
    for (int k = 0; k < nrhs; k++) for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) B[i + (size_t)k * n] += A[i + (size_t)j * n] * Xt[j + (size_t)k * n];
    magmaDouble_ptr dA, dB, dX, dwd; magmaFloat_ptr dws; magma_dmalloc(&dA, (size_t)ld * n); magma_dmalloc(&dB, (size_t)ld * nrhs); magma_dmalloc(&dX, (size_t)ld * nrhs);
    magma_dmalloc(&dwd, (size_t)n * nrhs + 64); magma_smalloc(&dws, (size_t)n * (n + nrhs) + 4096);
    magmablas_dlaset(MagmaFull, ld, nrhs, 7.0, 7.0, dX, ld, Q);   // poison: a wrong x shows as 7s, not as stale zeros
    magma_dsetmatrix(n, n, A.data(), n, dA, ld, Q); magma_dsetmatrix(n, nrhs, B.data(), n, dB, ld, Q); magma_queue_sync(Q);
    magma_dsgesv_gpu(MagmaNoTrans, n, nrhs, dA, ld, ip.data(), dipiv, dB, ld, dX, ld, dwd, dws, &iter, &info);
    magma_dgetmatrix(n, nrhs, dX, ld, X.data(), n, Q); br = berr(A, X, B, n, nrhs, fx);
    magma_free(dA); magma_free(dB); magma_free(dX); magma_free(dwd); magma_free(dws);
  } else {
    std::vector<cd> Ac(A.begin(), A.end()), Xt((size_t)n * nrhs), B((size_t)n * nrhs, 0), X((size_t)n * nrhs); for (auto& x : Xt) x = cd(r1(), r1());
    for (int k = 0; k < nrhs; k++) for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) B[i + (size_t)k * n] += Ac[i + (size_t)j * n] * Xt[j + (size_t)k * n];
    magmaDoubleComplex_ptr dA, dB, dX, dwd; magmaFloatComplex_ptr dws; magma_zmalloc(&dA, (size_t)ld * n); magma_zmalloc(&dB, (size_t)ld * nrhs); magma_zmalloc(&dX, (size_t)ld * nrhs);
    magma_zmalloc(&dwd, (size_t)n * nrhs + 64); magma_cmalloc(&dws, (size_t)n * (n + nrhs) + 4096);
    magmaDoubleComplex seven = MAGMA_Z_MAKE(7, 0); magmablas_zlaset(MagmaFull, ld, nrhs, seven, seven, dX, ld, Q);
    magma_zsetmatrix(n, n, (magmaDoubleComplex*)Ac.data(), n, dA, ld, Q); magma_zsetmatrix(n, nrhs, (magmaDoubleComplex*)B.data(), n, dB, ld, Q); magma_queue_sync(Q);
    magma_zcgesv_gpu(MagmaNoTrans, n, nrhs, dA, ld, ip.data(), dipiv, dB, ld, dX, ld, dwd, dws, &iter, &info);
    magma_zgetmatrix(n, nrhs, dX, ld, (magmaDoubleComplex*)X.data(), n, Q); br = berr(A, X, B, n, nrhs, fx);
    magma_free(dA); magma_free(dB); magma_free(dX); magma_free(dwd); magma_free(dws);
  }
  magma_free(dipiv);
  const char* v = info != 0 ? "INFO" : (!fx || !(br <= 100)) ? "WRONG" : (iter < 0 && std::string(mode) == "rep") ? "NOCONV" : "OK";
  printf("%s,%s,%d,%d,%d,%lld,%lld,%.3g,%s\n", mode, z ? "zcgesv_gpu" : "dsgesv_gpu", n, nrhs, rep, (long long)iter, (long long)info, br, v);
}

// lu: backward error ||P A - L U||_1 / (||A||_1 n eps) of sgetrf_native / sgetrf_gpu / dgetrf_native for n near 512, ldda = n or roundup(n,32)
static void lu() {
  printf("lu,routine,n,ldda,info,berr_ratio\n");
  for (int n : {64, 256, 500, 511, 512, 513, 514, 520, 544, 600, 1000, 1024, 1025}) for (int padded = 0; padded < 2; padded++) for (int r = 0; r < 3; r++) {
    int ld = padded ? (int)magma_roundup(n, 32) : n; Sd = 11u + n; std::vector<double> A((size_t)n * n); for (auto& a : A) a = r1();
    double an = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < n; i++) s += fabs(A[i + (size_t)j * n]); an = std::max(an, s); }
    std::vector<magma_int_t> ip(n); magma_int_t info = -999; std::vector<double> F((size_t)n * n);
    if (r < 2) { std::vector<float> As(A.begin(), A.end()), Fs((size_t)n * n); magmaFloat_ptr dS; magma_smalloc(&dS, (size_t)ld * n);
      magma_ssetmatrix(n, n, As.data(), n, dS, ld, Q); if (r == 0) magma_sgetrf_native(n, n, dS, ld, ip.data(), &info); else magma_sgetrf_gpu(n, n, dS, ld, ip.data(), &info);
      magma_sgetmatrix(n, n, dS, ld, Fs.data(), n, Q); magma_free(dS); for (size_t k = 0; k < F.size(); k++) F[k] = Fs[k]; }
    else { magmaDouble_ptr dD; magma_dmalloc(&dD, (size_t)ld * n); magma_dsetmatrix(n, n, A.data(), n, dD, ld, Q); magma_dgetrf_native(n, n, dD, ld, ip.data(), &info);
      magma_dgetmatrix(n, n, dD, ld, F.data(), n, Q); magma_free(dD); }
    // apply the row interchanges to A, then R = PA - L U
    std::vector<double> PA = A; for (int i = 0; i < n; i++) { int p = (int)ip[i] - 1; if (p != i && p >= 0 && p < n) for (int j = 0; j < n; j++) std::swap(PA[i + (size_t)j * n], PA[p + (size_t)j * n]); }
    double rn = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < n; i++) { double lu = 0; int kmax = std::min(i, j); for (int k = 0; k <= kmax; k++) { double l = k == i ? 1.0 : F[i + (size_t)k * n]; if (k < i || k == i) lu += l * F[k + (size_t)j * n] * (k <= j ? 1 : 0); }
        s += fabs(PA[i + (size_t)j * n] - lu); } rn = std::max(rn, s); }
    double eps = r < 2 ? 5.96e-8 : EPS; const char* nm[] = {"sgetrf_native", "sgetrf_gpu", "dgetrf_native"};
    printf("lu,%s,%d,%d,%lld,%.3g\n", nm[r], n, ld, (long long)info, rn / (an * n * eps)); }
}
// ld: separate n from ldda for getrf_native, and test the in-place / out-of-place transpose kernels it calls
static double luErr(bool dbl, int n, int ld, bool native, magma_int_t& info) {
  Sd = 11u + n; std::vector<double> A((size_t)n * n); for (auto& a : A) a = r1(); std::vector<magma_int_t> ip(n); std::vector<double> F((size_t)n * n);
  if (!dbl) { std::vector<float> As(A.begin(), A.end()); magmaFloat_ptr dS; magma_smalloc(&dS, (size_t)ld * n); magma_ssetmatrix(n, n, As.data(), n, dS, ld, Q);
    if (native) magma_sgetrf_native(n, n, dS, ld, ip.data(), &info); else magma_sgetrf_gpu(n, n, dS, ld, ip.data(), &info);
    magma_sgetmatrix(n, n, dS, ld, As.data(), n, Q); magma_free(dS); for (size_t k = 0; k < F.size(); k++) F[k] = As[k]; }
  else { magmaDouble_ptr dD; magma_dmalloc(&dD, (size_t)ld * n); magma_dsetmatrix(n, n, A.data(), n, dD, ld, Q);
    if (native) magma_dgetrf_native(n, n, dD, ld, ip.data(), &info); else magma_dgetrf_gpu(n, n, dD, ld, ip.data(), &info);
    magma_dgetmatrix(n, n, dD, ld, F.data(), n, Q); magma_free(dD); }
  double an = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < n; i++) s += fabs(A[i + (size_t)j * n]); an = std::max(an, s); }
  for (int i = 0; i < n; i++) { int p = (int)ip[i] - 1; if (p != i && p >= 0 && p < n) for (int j = 0; j < n; j++) std::swap(A[i + (size_t)j * n], A[p + (size_t)j * n]); }
  double rn = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < n; i++) { double lu = 0; for (int k = 0; k <= std::min(i, j); k++) lu += (k == i ? 1.0 : F[i + (size_t)k * n]) * F[k + (size_t)j * n]; s += fabs(A[i + (size_t)j * n] - lu); } rn = std::max(rn, s); }
  return rn / (an * n * (dbl ? EPS : 5.96e-8)); }
static void ldsweep() {
  printf("ld,routine,n,ldda,info,berr_ratio\n");
  struct T { int n, ld; }; std::vector<T> ts = {{512, 512}, {512, 513}, {512, 514}, {512, 516}, {512, 520}, {512, 524}, {512, 528}, {513, 516}, {513, 520}, {513, 524}, {513, 528},
    {100, 100}, {100, 101}, {100, 104}, {200, 201}, {200, 204}, {300, 301}, {300, 302}, {300, 304}, {384, 385}, {384, 388}, {384, 392}, {500, 502}, {500, 504}, {2000, 2001}, {2000, 2004}};
  for (auto t : ts) for (int dbl = 0; dbl < 2; dbl++) { magma_int_t info = -999; double e = luErr(dbl, t.n, t.ld, true, info);
    printf("ld,%s,%d,%d,%lld,%.3g\n", dbl ? "dgetrf_native" : "sgetrf_native", t.n, t.ld, (long long)info, e); }
  // transpose kernels: in-place n x n with lda, and out-of-place; compare with the host transpose
  printf("tr,kernel,n,ldda,maxerr\n");
  for (auto t : std::vector<T>{{513, 513}, {513, 544}, {511, 511}, {500, 500}, {514, 514}, {512, 513}, {64, 65}, {200, 201}}) {
    Sd = 3u; std::vector<float> A((size_t)t.ld * t.n, 0); for (int j = 0; j < t.n; j++) for (int i = 0; i < t.n; i++) A[i + (size_t)j * t.ld] = (float)r1();
    magmaFloat_ptr d, d2; magma_smalloc(&d, (size_t)t.ld * t.n); magma_smalloc(&d2, (size_t)t.ld * t.n); magma_ssetmatrix(t.ld, t.n, A.data(), t.ld, d, t.ld, Q);
    magmablas_stranspose_inplace(t.n, d, t.ld, Q); std::vector<float> B((size_t)t.ld * t.n); magma_sgetmatrix(t.ld, t.n, d, t.ld, B.data(), t.ld, Q);
    double e = 0; for (int j = 0; j < t.n; j++) for (int i = 0; i < t.n; i++) e = std::max(e, (double)fabs(B[i + (size_t)j * t.ld] - A[j + (size_t)i * t.ld]));
    printf("tr,stranspose_inplace,%d,%d,%.3g\n", t.n, t.ld, e);
    magma_ssetmatrix(t.ld, t.n, A.data(), t.ld, d, t.ld, Q); magmablas_stranspose(t.n, t.n, d, t.ld, d2, t.ld, Q); magma_sgetmatrix(t.ld, t.n, d2, t.ld, B.data(), t.ld, Q);
    e = 0; for (int j = 0; j < t.n; j++) for (int i = 0; i < t.n; i++) e = std::max(e, (double)fabs(B[i + (size_t)j * t.ld] - A[j + (size_t)i * t.ld]));
    printf("tr,stranspose,%d,%d,%.3g\n", t.n, t.ld, e); magma_free(d); magma_free(d2); }
}
// qd: does magma_queue_create / magma_queue_destroy of another queue wait for work already queued on Q?
// A ~100 ms dgemm is enqueued on Q, then the other queue is created / destroyed and Q is polled with cudaStreamQuery.
static void qd() {
  int n = 4096; magmaDouble_ptr a, b, c; magma_dmalloc(&a, (size_t)n * n); magma_dmalloc(&b, (size_t)n * n); magma_dmalloc(&c, (size_t)n * n);
  magmablas_dlaset(MagmaFull, n, n, 0.5, 1.0, a, n, Q); magmablas_dlaset(MagmaFull, n, n, 0.5, 1.0, b, n, Q); magma_queue_sync(Q);
  cudaStream_t s = magma_queue_get_cuda_stream(Q);
  auto busy = [&]() { return cudaStreamQuery(s) == cudaErrorNotReady; };
  for (int t = 0; t < 3; t++) {
    for (int r = 0; r < 4; r++) magma_dgemm(MagmaNoTrans, MagmaNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, c, n, Q);
    bool b0 = busy(); magma_queue_t q2; magma_queue_create(0, &q2); bool b1 = busy(); magma_queue_destroy(q2); bool b2 = busy();
    magma_queue_sync(Q);
    printf("qd,trial %d: Q busy after enqueue=%d, after create(q2)=%d, after destroy(q2)=%d\n", t, b0, b1, b2); }
  magma_free(a); magma_free(b); magma_free(c);
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : "rep"; int R = argc > 2 ? atoi(argv[2]) : 10;
  magma_init(); magma_queue_create(0, &Q);
  if (m == "qd") { qd(); magma_queue_destroy(Q); magma_finalize(); return 0; }
  if (m == "ld") { ldsweep(); magma_queue_destroy(Q); magma_finalize(); return 0; }
  if (m == "lu") { lu(); magma_queue_destroy(Q); magma_finalize(); return 0; }
  printf("mode,solver,n,nrhs,rep,iter,info,berr_ratio,verdict  CUDA_LAUNCH_BLOCKING=%s\n", getenv("CUDA_LAUNCH_BLOCKING") ? getenv("CUDA_LAUNCH_BLOCKING") : "unset");
  for (int n : {200, 513, 1024}) {
    auto A = gen(n, 10.0, 100u + n); if (m == "fb") for (auto& a : A) a = ldexp(a, 200);
    for (int nrhs : {1, 2}) for (bool z : {false, true}) for (int r = 0; r < R; r++) one(m.c_str(), z, n, nrhs, r, A);
  }
  fprintf(stderr, "DONE %s cuda=%s\n", m.c_str(), cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

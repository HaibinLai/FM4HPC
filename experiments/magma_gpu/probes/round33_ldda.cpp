// Round 33 (roxy): factorizations with an unaligned leading dimension.  M242 showed that getrf_native is wrong whenever
// ldda*sizeof(T) is not a multiple of 32 bytes; MAGMA's own testers always use roundup(n, 32), so this input dimension is
// untested library-wide.  For s and d precision, ldda in {n, n+1, n+3, n+5, roundup(n,32)}:
//   LU       getrf_gpu, getrf_native, getrf_batched, getrf_nopiv_batched     ||PA - LU||_1 / (||A||_1 n eps)
//   Cholesky potrf_gpu, potrf_native, potrf_batched                         ||A - L L^T||_1 / (||A||_1 n eps)
//   QR       geqrf2_gpu, geqrf_batched                                      ||A^T A - R^T R||_1 / (||A||_1^2 n eps)
//   solve    getrs_batched after getrf_batched                              normwise backward error of x
// Batched calls use 3 matrices stored back to back with stride ldda*n, so the matrix starts are unaligned too.
// CSV: ldda,prec,routine,n,ldda,info,err,verdict   (BAD if err > 50 or non-finite, with info = 0)
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

// precision wrappers
static void setm(int m, int n, const float* h, int ld, float* d, int ldd) { magma_ssetmatrix(m, n, h, ld, d, ldd, Q); }
static void setm(int m, int n, const double* h, int ld, double* d, int ldd) { magma_dsetmatrix(m, n, h, ld, d, ldd, Q); }
static void getm(int m, int n, const float* d, int ldd, float* h, int ld) { magma_sgetmatrix(m, n, d, ldd, h, ld, Q); }
static void getm(int m, int n, const double* d, int ldd, double* h, int ld) { magma_dgetmatrix(m, n, d, ldd, h, ld, Q); }
static magma_int_t getrf_gpu(int n, float* d, int ld, magma_int_t* ip) { magma_int_t i; magma_sgetrf_gpu(n, n, d, ld, ip, &i); return i; }
static magma_int_t getrf_gpu(int n, double* d, int ld, magma_int_t* ip) { magma_int_t i; magma_dgetrf_gpu(n, n, d, ld, ip, &i); return i; }
static magma_int_t getrf_nat(int n, float* d, int ld, magma_int_t* ip) { magma_int_t i; magma_sgetrf_native(n, n, d, ld, ip, &i); return i; }
static magma_int_t getrf_nat(int n, double* d, int ld, magma_int_t* ip) { magma_int_t i; magma_dgetrf_native(n, n, d, ld, ip, &i); return i; }
static magma_int_t potrf_gpu(int n, float* d, int ld) { magma_int_t i; magma_spotrf_gpu(MagmaLower, n, d, ld, &i); return i; }
static magma_int_t potrf_gpu(int n, double* d, int ld) { magma_int_t i; magma_dpotrf_gpu(MagmaLower, n, d, ld, &i); return i; }
static magma_int_t potrf_nat(int n, float* d, int ld) { magma_int_t i; magma_spotrf_native(MagmaLower, n, d, ld, &i); return i; }
static magma_int_t potrf_nat(int n, double* d, int ld) { magma_int_t i; magma_dpotrf_native(MagmaLower, n, d, ld, &i); return i; }
static magma_int_t geqrf2(int n, float* d, int ld, float* tau) { magma_int_t i; magma_sgeqrf2_gpu(n, n, d, ld, tau, &i); return i; }
static magma_int_t geqrf2(int n, double* d, int ld, double* tau) { magma_int_t i; magma_dgeqrf2_gpu(n, n, d, ld, tau, &i); return i; }
static void getrf_b(int n, float** a, int ld, magma_int_t** p, magma_int_t* inf, int b) { magma_sgetrf_batched(n, n, a, ld, p, inf, b, Q); }
static void getrf_b(int n, double** a, int ld, magma_int_t** p, magma_int_t* inf, int b) { magma_dgetrf_batched(n, n, a, ld, p, inf, b, Q); }
static void nopiv_b(int n, float** a, int ld, magma_int_t* inf, int b) { magma_sgetrf_nopiv_batched(n, n, a, ld, inf, b, Q); }
static void nopiv_b(int n, double** a, int ld, magma_int_t* inf, int b) { magma_dgetrf_nopiv_batched(n, n, a, ld, inf, b, Q); }
static void potrf_b(int n, float** a, int ld, magma_int_t* inf, int b) { magma_spotrf_batched(MagmaLower, n, a, ld, inf, b, Q); }
static void potrf_b(int n, double** a, int ld, magma_int_t* inf, int b) { magma_dpotrf_batched(MagmaLower, n, a, ld, inf, b, Q); }
static void geqrf_b(int n, float** a, int ld, float** t, magma_int_t* inf, int b) { magma_sgeqrf_batched(n, n, a, ld, t, inf, b, Q); }
static void geqrf_b(int n, double** a, int ld, double** t, magma_int_t* inf, int b) { magma_dgeqrf_batched(n, n, a, ld, t, inf, b, Q); }
static void getrs_b(int n, float** a, int ld, magma_int_t** p, float** x, int ldx, int b) { magma_sgetrs_batched(MagmaNoTrans, n, 1, a, ld, p, x, ldx, b, Q); }
static void getrs_b(int n, double** a, int ld, magma_int_t** p, double** x, int ldx, int b) { magma_dgetrs_batched(MagmaNoTrans, n, 1, a, ld, p, x, ldx, b, Q); }
template <class T> static T* dalloc(size_t k) { void* p; magma_malloc(&p, k * sizeof(T) + 256); return (T*)p; }
template <class T> static T** ptrs(T* base, size_t stride, int b) { std::vector<T*> h(b); for (int i = 0; i < b; i++) h[i] = base + i * stride; T** d; magma_malloc((void**)&d, b * sizeof(T*)); cudaMemcpy(d, h.data(), b * sizeof(T*), cudaMemcpyHostToDevice); return d; }

static double norm1(const std::vector<double>& A, int n) { double mx = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < n; i++) s += fabs(A[i + (size_t)j * n]); mx = std::max(mx, s); } return mx; }
static double luErr(const std::vector<double>& A, const std::vector<double>& F, const magma_int_t* ip, int n, double eps) {
  std::vector<double> PA = A; if (ip) for (int i = 0; i < n; i++) { int p = (int)ip[i] - 1; if (p != i && p >= 0 && p < n) for (int j = 0; j < n; j++) std::swap(PA[i + (size_t)j * n], PA[p + (size_t)j * n]); }
  double rn = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < n; i++) { double lu = 0; for (int k = 0; k <= std::min(i, j); k++) lu += (k == i ? 1.0 : F[i + (size_t)k * n]) * F[k + (size_t)j * n]; s += fabs(PA[i + (size_t)j * n] - lu); } if (!(s <= rn)) rn = s; }
  return rn / (norm1(A, n) * n * eps); }
static double cholErr(const std::vector<double>& A, const std::vector<double>& F, int n, double eps) {
  double rn = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = j; i < n; i++) { double l = 0; for (int k = 0; k <= j; k++) l += F[i + (size_t)k * n] * F[j + (size_t)k * n]; s += fabs(A[i + (size_t)j * n] - l); } if (!(s <= rn)) rn = s; }
  return rn / (norm1(A, n) * n * eps); }
static double qrErr(const std::vector<double>& A, const std::vector<double>& F, int n, double eps) {
  double rn = 0, an = norm1(A, n); for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < n; i++) { double ata = 0, rtr = 0; for (int k = 0; k < n; k++) ata += A[k + (size_t)i * n] * A[k + (size_t)j * n];
      for (int k = 0; k <= std::min(i, j); k++) rtr += F[k + (size_t)i * n] * F[k + (size_t)j * n]; s += fabs(ata - rtr); } if (!(s <= rn)) rn = s; }
  return rn / (an * an * n * eps); }
static void out(const char* prec, const char* r, int n, int ld, long long info, double e) {
  printf("ldda,%s,%s,%d,%d,%lld,%.3g,%s\n", prec, r, n, ld, info, e, (info == 0 && !(e <= 50)) ? "BAD" : "OK"); }

template <class T> static void run(const char* prec, double eps, bool batched) {
  std::vector<int> ns = batched ? std::vector<int>{5, 13, 31, 33, 64, 100, 257} : std::vector<int>{100, 257, 513, 1000};
  for (int n : ns) {
    Sd = 5u + n; std::vector<double> A((size_t)n * n), S((size_t)n * n, 0), D = A; for (auto& a : A) a = r1();
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { double s = 0; for (int k = 0; k < n; k++) s += A[i + (size_t)k * n] * A[j + (size_t)k * n]; S[i + (size_t)j * n] = s + (i == j ? n : 0); }   // SPD
    D = A; for (int i = 0; i < n; i++) D[i + (size_t)i * n] += n;   // diagonally dominant (no pivoting needed)
    auto toT = [&](const std::vector<double>& v) { return std::vector<T>(v.begin(), v.end()); };
    auto back = [&](const std::vector<T>& v) { return std::vector<double>(v.begin(), v.end()); };
    // the matrices as seen in precision T (so the reference uses the same rounded input)
    std::vector<double> At = back(toT(A)), St = back(toT(S)), Dt = back(toT(D));
    for (int ld : {n, n + 1, n + 3, n + 5, (int)magma_roundup(n, 32)}) {
      if (ld == (int)magma_roundup(n, 32) && ld <= n + 5 && ld != n) continue;   // already covered
      std::vector<T> h((size_t)n * n), F((size_t)n * n); std::vector<magma_int_t> ip(n);
      if (!batched) {
        T* d = dalloc<T>((size_t)ld * n); magma_int_t info;
        h = toT(A); setm(n, n, h.data(), n, d, ld); info = getrf_gpu(n, d, ld, ip.data()); getm(n, n, d, ld, F.data(), n); out(prec, "getrf_gpu", n, ld, info, luErr(At, back(F), ip.data(), n, eps));
        h = toT(A); setm(n, n, h.data(), n, d, ld); info = getrf_nat(n, d, ld, ip.data()); getm(n, n, d, ld, F.data(), n); out(prec, "getrf_native", n, ld, info, luErr(At, back(F), ip.data(), n, eps));
        h = toT(S); setm(n, n, h.data(), n, d, ld); info = potrf_gpu(n, d, ld); getm(n, n, d, ld, F.data(), n); out(prec, "potrf_gpu", n, ld, info, cholErr(St, back(F), n, eps));
        h = toT(S); setm(n, n, h.data(), n, d, ld); info = potrf_nat(n, d, ld); getm(n, n, d, ld, F.data(), n); out(prec, "potrf_native", n, ld, info, cholErr(St, back(F), n, eps));
        std::vector<T> tau(n); h = toT(A); setm(n, n, h.data(), n, d, ld); info = geqrf2(n, d, ld, tau.data()); getm(n, n, d, ld, F.data(), n); out(prec, "geqrf2_gpu", n, ld, info, qrErr(At, back(F), n, eps));
        magma_free(d);
      } else {
        const int B = 3; size_t st = (size_t)ld * n; T* d = dalloc<T>(st * B); T** da = ptrs(d, st, B);
        magma_int_t *dinfo, *dpiv; magma_imalloc(&dinfo, B); magma_imalloc(&dpiv, (size_t)n * B); magma_int_t** dpa = ptrs(dpiv, n, B);
        T* dtau = dalloc<T>((size_t)n * B); T** dta = ptrs(dtau, n, B); T* dx = dalloc<T>((size_t)ld * B); T** dxa = ptrs(dx, ld, B);
        std::vector<magma_int_t> info(B); std::vector<magma_int_t> ipb((size_t)n * B);
        auto load = [&](const std::vector<double>& M) { auto t = toT(M); for (int b = 0; b < B; b++) setm(n, n, t.data(), n, d + b * st, ld); };
        auto each = [&](const char* nm, auto err) { magma_getvector(B, sizeof(magma_int_t), dinfo, 1, info.data(), 1, Q); double w = 0; long long bad = 0;
          for (int b = 0; b < B; b++) { getm(n, n, d + b * st, ld, F.data(), n); double e = err(b); if (!(e <= w)) w = e; if (info[b]) bad = info[b]; } out(prec, nm, n, ld, bad, w); };
        load(A); getrf_b(n, da, ld, dpa, dinfo, B); magma_getvector((magma_int_t)n * B, sizeof(magma_int_t), dpiv, 1, ipb.data(), 1, Q);
        each("getrf_batched", [&](int b) { return luErr(At, back(F), ipb.data() + (size_t)b * n, n, eps); });
        {   // getrs_batched on the factors just computed: b = A * ones
          std::vector<T> rhs((size_t)ld * B, 0); for (int b = 0; b < B; b++) for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += At[i + (size_t)j * n]; rhs[i + (size_t)b * ld] = (T)s; }
          setm(ld, B, rhs.data(), ld, dx, ld); getrs_b(n, da, ld, dpa, dxa, ld, B); std::vector<T> x((size_t)ld * B); getm(ld, B, dx, ld, x.data(), ld);
          double w = 0, an = 0; for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += fabs(At[i + (size_t)j * n]); an = std::max(an, s); }
          for (int b = 0; b < B; b++) { double rn = 0, xn = 0, bn = 0; for (int i = 0; i < n; i++) { double s = rhs[i + (size_t)b * ld]; for (int j = 0; j < n; j++) s -= At[i + (size_t)j * n] * x[j + (size_t)b * ld];
              if (!(fabs(s) <= rn)) rn = fabs(s); xn = std::max(xn, (double)fabs(x[i + (size_t)b * ld])); bn = std::max(bn, (double)fabs(rhs[i + (size_t)b * ld])); }
            double e = rn / ((an * xn + bn) * n * eps); if (!(e <= w)) w = e; }
          out(prec, "getrs_batched", n, ld, 0, w); }
        load(D); nopiv_b(n, da, ld, dinfo, B); each("getrf_nopiv_batched", [&](int) { return luErr(Dt, back(F), NULL, n, eps); });
        load(S); potrf_b(n, da, ld, dinfo, B); each("potrf_batched", [&](int) { return cholErr(St, back(F), n, eps); });
        load(A); geqrf_b(n, da, ld, dta, dinfo, B); each("geqrf_batched", [&](int) { return qrErr(At, back(F), n, eps); });
        magma_free(d); magma_free(da); magma_free(dinfo); magma_free(dpiv); magma_free(dpa); magma_free(dtau); magma_free(dta); magma_free(dx); magma_free(dxa);
      }
    }
  }
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : "single";
  magma_init(); magma_queue_create(0, &Q);
  printf("ldda,prec,routine,n,ldda,info,err,verdict\n");
  bool b = m == "batched";
  run<float>("s", 5.96e-8, b); run<double>("d", 1.1102230246251565e-16, b);
  fprintf(stderr, "DONE %s cuda=%s\n", m.c_str(), cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

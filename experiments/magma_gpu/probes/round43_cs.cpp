// Round 43 (roxy): M245, M246, M247, M250 in single precisions s and c (round 39 covered z, rounds 34-37 covered d).
// CSV: r43,finding,routine,case,observed,expected_or_control,verdict   (REPRODUCED | NOT)
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <complex>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
typedef std::complex<float> cf;
static magma_queue_t Q; static unsigned Sd = 1u;
static float r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (float)0x1000000 - 0.5f; }
static void rep(const char* f, const char* r, const std::string& c, double obs, double ctl, bool trig, bool ok) { printf("r43,%s,%s,%s,%.3g,%.3g,%s\n", f, r, c.c_str(), obs, ctl, (trig && ok) ? "REPRODUCED" : "NOT"); }

// precision traits: T = float or cf; M = MAGMA element type
template <class T> struct P;
template <> struct P<float> { typedef float M; static const char* n() { return "s"; } static float mk(float a, float) { return a; }
  static void gemmb(int m, int n, int k, M al, M** a, int lda, M be, M** c, int ldc, int B) { magmablas_sgemm_batched(MagmaNoTrans, MagmaNoTrans, m, n, k, al, (M const* const*)a, lda, (M const* const*)a, lda, be, c, ldc, B, Q); }
  static void herkb(int n, int k, float al, M** a, int lda, float be, M** c, int ldc, int B) { magmablas_ssyrk_batched(MagmaLower, MagmaNoTrans, n, k, al, (M const* const*)a, lda, be, c, ldc, B, Q); }
  static void trsmb(int m, int n, M al, M** a, int lda, M** b, int ldb, int B) { magmablas_strsm_batched(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, m, n, al, a, lda, b, ldb, B, Q); }
  static void trmm(int m, int n, M al, M* a, int lda, M* b, int ldb) { magmablas_strmm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, m, n, al, a, lda, b, ldb, Q); }
  static void rbt(int n, M* A, M* B, magma_int_t* info) { magma_sgesv_rbt(MagmaFalse, n, 1, A, n, B, n, info); } };
template <> struct P<cf> { typedef magmaFloatComplex M; typedef cf T; static const char* n() { return "c"; } static cf mk(float a, float b) { return cf(a, b); }
  static void gemmb(int m, int n, int k, M al, M** a, int lda, M be, M** c, int ldc, int B) { magmablas_cgemm_batched(MagmaNoTrans, MagmaNoTrans, m, n, k, al, (M const* const*)a, lda, (M const* const*)a, lda, be, c, ldc, B, Q); }
  static void herkb(int n, int k, float al, M** a, int lda, float be, M** c, int ldc, int B) { magmablas_cherk_batched(MagmaLower, MagmaNoTrans, n, k, al, (M const* const*)a, lda, be, c, ldc, B, Q); }
  static void trsmb(int m, int n, M al, M** a, int lda, M** b, int ldb, int B) { magmablas_ctrsm_batched(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, m, n, al, a, lda, b, ldb, B, Q); }
  static void trmm(int m, int n, M al, M* a, int lda, M* b, int ldb) { magmablas_ctrmm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, m, n, al, a, lda, b, ldb, Q); }
  static void rbt(int n, M* A, M* B, magma_int_t* info) { magma_cgesv_rbt(MagmaFalse, n, 1, A, n, B, n, info); } };

template <class T> static T* dput(const std::vector<T>& h) { void* d; magma_malloc(&d, h.size() * sizeof(T)); magma_setvector((magma_int_t)h.size(), sizeof(T), h.data(), 1, d, 1, Q); return (T*)d; }
template <class T> static std::vector<T> dget(T* d, size_t k) { std::vector<T> h(k); magma_getvector((magma_int_t)k, sizeof(T), d, 1, h.data(), 1, Q); return h; }
template <class T> static T** parr(T* base, size_t stride, int B) { std::vector<T*> h(B); for (int i = 0; i < B; i++) h[i] = base + i * stride; T** d; magma_malloc((void**)&d, B * sizeof(T*)); cudaMemcpy(d, h.data(), B * sizeof(T*), cudaMemcpyHostToDevice); return d; }
template <class T> static typename P<T>::M asM(T v) { typename P<T>::M m; memcpy(&m, &v, sizeof v); return m; }

template <class T> static void run() {
  typedef typename P<T>::M M; std::string pre = P<T>::n(); const int B = 3;
  {   // M245 gemm_batched k = 0, beta = 0.5
    int m = 40, n = 7; std::vector<T> C((size_t)m * n * B); for (auto& c : C) c = P<T>::mk(r1(), r1()); T* dC = dput(C); T** pc = parr(dC, (size_t)m * n, B);
    P<T>::gemmb(m, n, 0, asM(P<T>::mk(1, 0)), (M**)pc, m, asM(P<T>::mk(0.5f, 0)), (M**)pc, m, B); magma_queue_sync(Q); auto g = dget(dC, C.size());
    double e = 0; for (size_t i = 0; i < C.size(); i++) e = std::max(e, (double)std::abs(g[i] - T(0.5f) * C[i])); rep("M245", (pre + "gemm_batched").c_str(), "k=0 beta=0.5 max|C-0.5C0|", e, 0, e > 1e-3, true);
    magma_free(dC); magma_free(pc); }
  {   // M245 herk/syrk_batched k = 0 and M246 other triangle with k = 17
    int n = 31, k = 17; std::vector<T> A((size_t)n * k * B), C((size_t)n * n * B); for (auto& a : A) a = P<T>::mk(r1(), r1());
    for (int b = 0; b < B; b++) for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) C[b * (size_t)n * n + i + (size_t)j * n] = i < j ? T(7) : P<T>::mk(r1(), 0);
    T *dA = dput(A), *dC = dput(C); T **pa = parr(dA, (size_t)n * k, B), **pc = parr(dC, (size_t)n * n, B);
    P<T>::herkb(n, 0, 1.0f, (M**)pa, n, 0.5f, (M**)pc, n, B); magma_queue_sync(Q); auto g = dget(dC, C.size()); double e = 0;
    for (int b = 0; b < B; b++) for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) { size_t p = b * (size_t)n * n + i + (size_t)j * n; e = std::max(e, (double)std::abs(g[p] - T(0.5f) * C[p])); }
    rep("M245", (pre + (pre == "s" ? "syrk_batched" : "herk_batched")).c_str(), "k=0 beta=0.5 strict lower max|C-0.5C0|", e, 0, e > 1e-3, true);
    magma_setvector((magma_int_t)C.size(), sizeof(T), C.data(), 1, dC, 1, Q); P<T>::herkb(n, k, 1.0f, (M**)pa, n, 0.5f, (M**)pc, n, B); magma_queue_sync(Q); g = dget(dC, C.size());
    long long ch = 0; for (int b = 0; b < B; b++) for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) if (g[b * (size_t)n * n + i + (size_t)j * n] != T(7)) ch++;
    rep("M246", (pre + (pre == "s" ? "syrk_batched" : "herk_batched")).c_str(), "n=31 k=17 L strictly-upper changed (of 1395)", (double)ch, 0, ch > 0, true);
    magma_free(dA); magma_free(dC); magma_free(pa); magma_free(pc); }
  {   // M247 trsm_batched and non-batched trmm, alpha = 0, B = NaN
    int m = 31, n = 5; std::vector<T> A((size_t)m * m * B, T(0)), X((size_t)m * n * B, P<T>::mk(NAN, NAN)); for (int b = 0; b < B; b++) for (int i = 0; i < m; i++) A[b * (size_t)m * m + i + (size_t)i * m] = T(2);
    T *dA = dput(A), *dX = dput(X); T **pa = parr(dA, (size_t)m * m, B), **px = parr(dX, (size_t)m * n, B);
    P<T>::trsmb(m, n, asM(T(0)), (M**)pa, m, (M**)px, m, B); magma_queue_sync(Q); auto g = dget(dX, X.size()); long long nz = 0; for (auto& v : g) if (!(v == T(0))) nz++;
    rep("M247", (pre + "trsm_batched").c_str(), "alpha=0 B=NaN nonzero (of 465)", (double)nz, 0, nz > 0, true);
    magma_setvector((magma_int_t)(m * n), sizeof(T), X.data(), 1, dX, 1, Q); P<T>::trmm(m, n, asM(T(0)), (M*)dA, m, (M*)dX, m); magma_queue_sync(Q); g = dget(dX, (size_t)m * n); nz = 0; for (auto& v : g) if (!(v == T(0))) nz++;
    rep("M247", ("magmablas_" + pre + "trmm").c_str(), "alpha=0 B=NaN nonzero (of 155)", (double)nz, 0, nz > 0, true);
    magma_free(dA); magma_free(dX); magma_free(pa); magma_free(px); }
  {   // M250 gesv_rbt n = 1, 2, 3 vs 5
    auto solve = [&](int n) { Sd = 9u + n; std::vector<T> A((size_t)n * n), X(n), Bv(n, T(0)); for (auto& a : A) a = P<T>::mk(r1(), r1()); for (int i = 0; i < n; i++) A[i + (size_t)i * n] += T(2); for (auto& x : X) x = P<T>::mk(r1(), r1());
      for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) Bv[i] += A[i + (size_t)j * n] * X[j]; magma_int_t info; P<T>::rbt(n, (M*)A.data(), (M*)Bv.data(), &info);
      double e = 0; for (int i = 0; i < n; i++) e = std::max(e, (double)std::abs(Bv[i] - X[i])); return e; };
    double e5 = solve(5); for (int n : {1, 2, 3}) { char c[64]; snprintf(c, sizeof c, "n=%d max|x-x_true| vs n=5", n); double e = solve(n); rep("M250", (pre + "gesv_rbt").c_str(), c, e, e5, e > 1e-3, e5 < 1e-4); } }
}

int main() {
  setvbuf(stdout, NULL, _IONBF, 0); magma_init(); magma_queue_create(0, &Q); printf("r43,finding,routine,case,observed,control,verdict\n");
  run<float>(); run<cf>();
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

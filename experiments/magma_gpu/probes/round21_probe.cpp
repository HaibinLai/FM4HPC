// Twenty-first GPU round: #2 (dstedx sub-block info) via an interposed dsteqr_ failure, #5 dsidi inverse, #8 gesvj batch
// chunking, #10 dsort with NaN (HANDOFF §29.23).  GPU 0 only.
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
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 777u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static void stedx() {   // n = 300, e[149] = 0: two blocks of 150 rows, both handled by dlaex0 (m > smlsiz)
  int n = 300; std::vector<double> d(n), e(n - 1); for (int i = 0; i < n; i++) d[i] = 2 + rnd(); for (auto& x : e) x = 0.5 + 0.1 * rnd(); e[149] = 0;
  magma_int_t lrw = 1 + 4 * n + 2 * n * n, liw = 3 + 5 * n; std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); std::vector<Z> Zm((size_t)n * n);
  double* dw; magma_dmalloc(&dw, 3 * n * n / 2 + 3 * n); magma_int_t info = -999;
  magma_zstedx(MagmaRangeAll, n, 0, 0, 0, 0, d.data(), e.data(), Zm.data(), n, rw.data(), lrw, iw.data(), liw, dw, &info);
  printf("zstedx n=300 (blocks rows 1-150, 151-300): info=%lld", (long long)info);
  if (info > 0) printf("  decoded with base n+1=301: rows %lld..%lld | decoded with block base m+1=151: rows %lld..%lld (block-relative)",
                       (long long)(info / 301), (long long)(info % 301), (long long)(info / 151), (long long)(info % 151));
  printf("\n"); }
static void dsidi() {
  for (int spd = 0; spd < 2; spd++) for (int lower = 0; lower < 2; lower++) {
    int n = 8; std::vector<double> A(n * n); S = 55u; for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { double v = rnd() + (i == j ? (spd ? 8.0 : (i % 2 ? -3.0 : 3.0)) : 0); A[i + j * n] = v; A[j + i * n] = v; }
    std::vector<double> F = A, wk(64 * n); std::vector<magma_int_t> ip(n); magma_int_t nn = n, lw = 64 * n, inf; const char* ul = lower ? "L" : "U";
    lapackf77_dsytrf(ul, &nn, F.data(), &nn, ip.data(), wk.data(), &lw, &inf); int twobytwo = 0; for (auto p : ip) if (p < 0) twobytwo++;
    double det[2]; magma_int_t inert[3], info = -999; magma_dsidi(lower ? MagmaLower : MagmaUpper, F.data(), n, n, ip.data(), det, inert, wk.data(), 111, &info);
    std::vector<double> X(n * n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) X[i + j * n] = (i <= j) ? F[i + j * n] : F[j + i * n];   // documented: upper triangle of the inverse
    double e = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { double s = 0; for (int k = 0; k < n; k++) s += A[i + k * n] * X[k + j * n]; e = fmax(e, fabs(s - (i == j ? 1 : 0))); }
    int pos = 0; { std::vector<double> W = A, w(n), wk2(3 * n); magma_int_t lw2 = 3 * n, i2; lapackf77_dsyev("N", "U", &nn, W.data(), &nn, w.data(), wk2.data(), &lw2, &i2); for (double x : w) if (x > 0) pos++; }
    printf("dsidi(job=111) %s, uplo=%s, 2x2 pivots=%d: info=%lld  max|A X - I| = %.3g  inertia +/-/0 = %lld/%lld/%lld (true positive count %d)\n", spd ? "SPD" : "indefinite", ul,
           twobytwo, (long long)info, e, (long long)inert[0], (long long)inert[1], (long long)inert[2], pos); } }
static void gesvj_chunks() {
  int n = 64, b = getenv("NB_BATCH") ? atoi(getenv("NB_BATCH")) : 70000; size_t fr, tot; cudaMemGetInfo(&fr, &tot);
  printf("gesvj batch %d of %dx%d: free %.1f GB\n", b, n, n, fr / 1e9);
  std::vector<Z*> dA(1), dU(1), dV(1); Z *A, *U, *V; double* Sd; size_t mn = (size_t)n * n;
  if (magma_zmalloc(&A, mn * b) || magma_zmalloc(&U, mn * b) || magma_zmalloc(&V, mn * b) || magma_dmalloc(&Sd, (size_t)n * b)) { printf("allocation failed\n"); return; }
  std::vector<Z> h(mn * b, C(0)); for (int k = 0; k < b; k++) { Z* M = h.data() + k * mn; if (k < 65534) for (int i = 0; i < n; i++) M[i + i * n] = C(1); else for (size_t t = 0; t < mn; t++) M[t] = C(rnd(), rnd()); }
  magma_zsetvector(mn * b, h.data(), 1, A, 1, Q);
  std::vector<Z*> pa(b), pu(b), pv(b); std::vector<double*> ps(b); for (int k = 0; k < b; k++) { pa[k] = A + k * mn; pu[k] = U + k * mn; pv[k] = V + k * mn; ps[k] = Sd + (size_t)k * n; }
  Z **dAa, **dUa, **dVa; double** dSa; magma_int_t* dinfo; magma_malloc((void**)&dAa, b * sizeof(Z*)); magma_malloc((void**)&dUa, b * sizeof(Z*)); magma_malloc((void**)&dVa, b * sizeof(Z*));
  magma_malloc((void**)&dSa, b * sizeof(double*)); magma_imalloc(&dinfo, b);
  magma_setvector(b, sizeof(Z*), pa.data(), 1, dAa, 1, Q); magma_setvector(b, sizeof(Z*), pu.data(), 1, dUa, 1, Q); magma_setvector(b, sizeof(Z*), pv.data(), 1, dVa, 1, Q); magma_setvector(b, sizeof(double*), ps.data(), 1, dSa, 1, Q);
  magma_int_t r = magma_zgesvj_batched(MagmaSomeVec, MagmaSomeVec, n, n, dAa, n, dSa, dUa, n, dVa, n, dinfo, b, Q); cudaError_t ce = cudaDeviceSynchronize();
  std::vector<magma_int_t> info(b); magma_getvector(b, sizeof(magma_int_t), dinfo, 1, info.data(), 1, Q); std::vector<double> Sh((size_t)n * b); magma_dgetvector((size_t)n * b, Sd, 1, Sh.data(), 1, Q);
  int nz = 0; for (auto x : info) if (x) nz++; int checked = 0, wrong = 0; double worst = 0;
  for (int k = 65534; k < b; k += 97) { std::vector<Z> M(h.begin() + k * mn, h.begin() + (k + 1) * mn); std::vector<double> s(n), rw(5 * n); magma_int_t nn = n, lw = 8 * n, inf, one = 1; std::vector<Z> wk(lw); Z dum;
    lapackf77_zgesvd("N", "N", &nn, &nn, M.data(), &nn, s.data(), &dum, &one, &dum, &one, wk.data(), &lw, rw.data(), &inf);
    double e = 0; for (int i = 0; i < n; i++) e = fmax(e, fabs(Sh[(size_t)k * n + i] - s[i]) / s[0]); worst = fmax(worst, e); checked++; if (!(e < 1e-10)) wrong++; }
  printf("  ret=%lld sync=%s info != 0: %d; sampled matrices beyond index 65534: %d checked, %d with wrong singular values (worst rel. err %.3g)\n",
         (long long)r, cudaGetErrorString(ce), nz, checked, wrong, worst); }
static void dsort_nan() {
  int n = 4096; std::vector<double> x(n); for (int i = 0; i < n; i++) x[i] = i + 1; x[17] = NAN; std::swap(x[5], x[4000]);
  double *dx, *dy; magma_int_t* di; magma_dmalloc(&dx, n); magma_dmalloc(&dy, n); magma_imalloc(&di, n); magma_dsetvector(n, x.data(), 1, dx, 1, Q);
  double **dxa, **dya; magma_int_t** dia; magma_malloc((void**)&dxa, sizeof(double*)); magma_malloc((void**)&dya, sizeof(double*)); magma_malloc((void**)&dia, sizeof(magma_int_t*));
  magma_setvector(1, sizeof(double*), &dx, 1, dxa, 1, Q); magma_setvector(1, sizeof(double*), &dy, 1, dya, 1, Q); magma_setvector(1, sizeof(magma_int_t*), &di, 1, dia, 1, Q);
  magmablas_dsort_batched(MagmaDescending, n, dxa, 1, dya, 1, dia, 1, Q); std::vector<magma_int_t> idx(n); magma_getvector(n, sizeof(magma_int_t), di, 1, idx.data(), 1, Q);
  std::vector<int> seen(n, 0); int out = 0, dup = 0, miss = 0; for (auto v : idx) { if (v < 0 || v >= n) out++; else seen[v]++; } for (int v : seen) { if (v > 1) dup++; if (v == 0) miss++; }
  printf("dsort_batched descending, n=4096 with one NaN: index array: out-of-range %d, duplicated %d, missing %d (a permutation has 0/0/0)\n", out, dup, miss); }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; magma_init(); magma_queue_create(0, &Q);
  if (m == "stedx") stedx(); else if (m == "dsidi") dsidi(); else if (m == "gesvj_chunks") gesvj_chunks(); else if (m == "dsort") dsort_nan();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

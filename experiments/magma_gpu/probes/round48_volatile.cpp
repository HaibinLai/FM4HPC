// Round 48 (roxy): kernels that use warp-synchronous reductions through `volatile` shared memory without __syncwarp
// (dmdotc.cu, dgemvmdot.cu: d/s precision only; zpotf2.cu: z), checked against host dot products on sm_120.
// M253 showed the non-volatile variant failing; this round asks whether the volatile ones hold up.
// CSV: r48,routine,n,k,relerr,verdict     relerr = max |gpu - host| / (sum |v_i||w_i| * eps * 64)
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
static magma_queue_t Q; static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static void out(const char* r, int n, int k, double e) { printf("r48,%s,%d,%d,%.3g,%s\n", r, n, k, e, e <= 1 ? "OK" : "BAD"); }
int main() {
  setvbuf(stdout, NULL, _IONBF, 0); magma_init(); magma_queue_create(0, &Q); printf("r48,routine,n,k,relerr,verdict\n");
  for (int n : {1, 31, 255, 256, 257, 1000, 4097, 100000}) {
    const int K = 8; std::vector<double> V((size_t)n * K), R(n); for (auto& x : V) x = r1(); for (auto& x : R) x = r1();
    double *dV, *dR, *d1, *d2, *skp; magma_dmalloc(&dV, V.size()); magma_dmalloc(&dR, n); magma_dmalloc(&d1, (size_t)K * n + 4096); magma_dmalloc(&d2, (size_t)K * n + 4096); magma_dmalloc(&skp, 64);
    magma_dsetvector((magma_int_t)V.size(), V.data(), 1, dV, 1, Q); magma_dsetvector(n, R.data(), 1, dR, 1, Q);
    std::vector<double> ref(K), mag(K); for (int i = 0; i < K; i++) { long double s = 0, a = 0; for (int j = 0; j < n; j++) { s += (long double)V[j + (size_t)i * n] * R[j]; a += fabsl((long double)V[j + (size_t)i * n] * R[j]); } ref[i] = (double)s; mag[i] = (double)a; }
    auto err = [&](int k) { std::vector<double> g(k); magma_dgetvector(k, skp, 1, g.data(), 1, Q); double e = 0; for (int i = 0; i < k; i++) e = std::max(e, fabs(g[i] - ref[i]) / (mag[i] * 1.11e-16 * 64 + 1e-300)); return e; };
    for (int k = 1; k <= K; k++) { magma_dmdotc(n, k, dV, dR, d1, d2, skp, Q); magma_queue_sync(Q); out("dmdotc", n, k, err(k)); }
    for (int k = 1; k <= K; k++) { magma_dgemvmdot(n, k, dV, dR, d1, d2, skp, Q); magma_queue_sync(Q); out("dgemvmdot", n, k, err(k)); }
    magma_dmdotc1(n, dV, dR, d1, d2, skp, Q); magma_queue_sync(Q); out("dmdotc1", n, 1, err(1));
    magma_dmdotc2(n, dV, dR, dV + n, dR, d1, d2, skp, Q); magma_queue_sync(Q); out("dmdotc2", n, 2, err(2));
    magma_dmdotc3(n, dV, dR, dV + n, dR, dV + 2 * (size_t)n, dR, d1, d2, skp, Q); magma_queue_sync(Q); out("dmdotc3", n, 3, err(3));
    magma_dmdotc4(n, dV, dR, dV + n, dR, dV + 2 * (size_t)n, dR, dV + 3 * (size_t)n, dR, d1, d2, skp, Q); magma_queue_sync(Q); out("dmdotc4", n, 4, err(4));
    magma_free(dV); magma_free(dR); magma_free(d1); magma_free(d2); magma_free(skp);
  }
  // potf2 (zpotf2_zdotc_kernel uses volatile double shared memory): Cholesky of SPD matrices, n up to 128
  for (int n : {1, 7, 31, 32, 33, 64, 100, 128}) for (int z = 0; z < 2; z++) {
    std::vector<double> G((size_t)n * n), S((size_t)n * n); for (auto& x : G) x = r1();
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { double s = 0; for (int k = 0; k < n; k++) s += G[i + (size_t)k * n] * G[j + (size_t)k * n]; S[i + (size_t)j * n] = s + (i == j ? n : 0); }
    magma_int_t info = 0; double e = 0, an = 0; for (double x : S) an = std::max(an, fabs(x));
    if (!z) { double* dA; magma_dmalloc(&dA, S.size()); magma_dsetmatrix(n, n, S.data(), n, dA, n, Q); magma_dpotf2_gpu(MagmaLower, n, dA, n, Q, &info); std::vector<double> L(S.size()); magma_dgetmatrix(n, n, dA, n, L.data(), n, Q); magma_free(dA);
      for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { double s = 0; for (int k = 0; k <= j; k++) s += L[i + (size_t)k * n] * L[j + (size_t)k * n]; e = std::max(e, fabs(s - S[i + (size_t)j * n]) / (an * n * 1.11e-16 * 64)); } }
    else { std::vector<magmaDoubleComplex> Sz(S.size()), Lz(S.size()); for (size_t i = 0; i < S.size(); i++) Sz[i] = MAGMA_Z_MAKE(S[i], 0); magmaDoubleComplex* dA; magma_zmalloc(&dA, S.size());
      magma_zsetmatrix(n, n, Sz.data(), n, dA, n, Q); magma_zpotf2_gpu(MagmaLower, n, dA, n, Q, &info); magma_zgetmatrix(n, n, dA, n, Lz.data(), n, Q); magma_free(dA);
      for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { double s = 0; for (int k = 0; k <= j; k++) s += MAGMA_Z_REAL(Lz[i + (size_t)k * n]) * MAGMA_Z_REAL(Lz[j + (size_t)k * n]) + MAGMA_Z_IMAG(Lz[i + (size_t)k * n]) * MAGMA_Z_IMAG(Lz[j + (size_t)k * n]); e = std::max(e, fabs(s - S[i + (size_t)j * n]) / (an * n * 1.11e-16 * 64)); } }
    printf("r48,%s,%d,0,%.3g,%s info=%lld\n", z ? "zpotf2_gpu" : "dpotf2_gpu", n, e, (info == 0 && e <= 1) ? "OK" : "BAD", (long long)info);
  }
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

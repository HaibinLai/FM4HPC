// Round 40 (roxy): non-batched magmablas level-3 routines checked for the patterns found in the batched ones (round 34):
//   magmablas_dsyrk / magmablas_zherk  k = 0 (C must become beta*C), and the not-referenced triangle (sentinel 7) unchanged
//   magmablas_dtrmm                    alpha = 0 with NaN in B (B must become 0)
// CSV: r40,routine,case,observed,expected,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <complex>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
static magma_queue_t Q; static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static void rep(const char* r, const char* c, double obs, double exp_, bool bad) { printf("r40,%s,%s,%.4g,%.4g,%s\n", r, c, obs, exp_, bad ? "BAD" : "OK"); }
int main() {
  setvbuf(stdout, NULL, _IONBF, 0); magma_init(); magma_queue_create(0, &Q);
  printf("r40,routine,case,observed,expected,verdict\n");
  for (int n : {7, 31, 64, 200}) for (char up : {'L', 'U'}) {
    int k = 17; std::vector<double> A((size_t)n * k), C((size_t)n * n); for (auto& a : A) a = r1();
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) C[i + (size_t)j * n] = (up == 'L' ? i < j : i > j) ? 7.0 : r1();
    double *dA, *dC; magma_dmalloc(&dA, A.size()); magma_dmalloc(&dC, C.size()); magma_dsetmatrix(n, k, A.data(), n, dA, n, Q);
    // (1) k = 0, beta = 0.5
    magma_dsetmatrix(n, n, C.data(), n, dC, n, Q); magmablas_dsyrk(up == 'L' ? MagmaLower : MagmaUpper, MagmaNoTrans, n, 0, 1.0, dA, n, 0.5, dC, n, Q); magma_queue_sync(Q);
    std::vector<double> g(C.size()); magma_dgetmatrix(n, n, dC, n, g.data(), n, Q); double e = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if (!(up == 'L' ? i < j : i > j)) e = std::max(e, fabs(g[i + (size_t)j * n] - 0.5 * C[i + (size_t)j * n]));
    char cs[64]; snprintf(cs, sizeof cs, "n=%d %c k=0 beta=0.5 max|C-0.5C0|", n, up); rep("magmablas_dsyrk", cs, e, 0, e > 1e-12);
    // (2) k = 17: other triangle must keep the sentinel; referenced triangle vs host
    magma_dsetmatrix(n, n, C.data(), n, dC, n, Q); magmablas_dsyrk(up == 'L' ? MagmaLower : MagmaUpper, MagmaNoTrans, n, k, 1.0, dA, n, 0.5, dC, n, Q); magma_queue_sync(Q);
    magma_dgetmatrix(n, n, dC, n, g.data(), n, Q); long long ch = 0; e = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { if (up == 'L' ? i < j : i > j) { if (g[i + (size_t)j * n] != 7.0) ch++; continue; }
      double s = 0; for (int l = 0; l < k; l++) s += A[i + (size_t)l * n] * A[j + (size_t)l * n]; e = std::max(e, fabs(g[i + (size_t)j * n] - (s + 0.5 * C[i + (size_t)j * n]))); }
    snprintf(cs, sizeof cs, "n=%d %c k=17 other-triangle changed (of %d)", n, up, n * (n - 1) / 2); rep("magmablas_dsyrk", cs, (double)ch, 0, ch > 0);
    snprintf(cs, sizeof cs, "n=%d %c k=17 referenced-triangle maxerr", n, up); rep("magmablas_dsyrk", cs, e, 0, e > 1e-10);
    magma_free(dA); magma_free(dC);
  }
  for (int m : {1, 31, 100}) for (char sd : {'L', 'R'}) {   // (3) non-batched magmablas_dtrmm, alpha = 0
    int n = 9, na = sd == 'L' ? m : n; std::vector<double> A((size_t)na * na, 0), B((size_t)m * n, NAN); for (int i = 0; i < na; i++) A[i + (size_t)i * na] = 2.0;
    double *dA, *dB; magma_dmalloc(&dA, A.size()); magma_dmalloc(&dB, B.size()); magma_dsetmatrix(na, na, A.data(), na, dA, na, Q); magma_dsetmatrix(m, n, B.data(), m, dB, m, Q);
    magmablas_dtrmm(sd == 'L' ? MagmaLeft : MagmaRight, MagmaLower, MagmaNoTrans, MagmaNonUnit, m, n, 0.0, dA, na, dB, m, Q); magma_queue_sync(Q);
    std::vector<double> g(B.size()); magma_dgetmatrix(m, n, dB, m, g.data(), m, Q); long long nz = 0; for (double v : g) if (!(v == 0)) nz++;
    char cs[64]; snprintf(cs, sizeof cs, "m=%d %c alpha=0 B=NaN: nonzero (of %d)", m, sd, m * n); rep("magmablas_dtrmm", cs, (double)nz, 0, nz > 0);
    magma_free(dA); magma_free(dB);
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

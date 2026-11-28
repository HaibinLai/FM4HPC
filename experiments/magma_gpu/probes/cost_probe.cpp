// Reference costs for the GMSTRS-fix overhead (MAGMA 2.10.0, GPU 0): FP64 dgesv_gpu, and the host-side certification
// pass of gmstrs_fix_v2.patch (D2H copy of A, B, X + row-wise Dot2 over column-major A, same loop order as the patch).
#include "magma_v2.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstdlib>
static inline void two_sum(double a, double b, double& s, double& e) { s = a + b; double z = s - a; e = (a - (s - z)) + (b - z); }
int main() {
  magma_init(); magma_queue_t Q; magma_queue_create(0, &Q);
  std::vector<int> ns = {1024, 2048, 4096, 8192}; if (getenv("FIXN")) ns = {atoi(getenv("FIXN"))};
  for (int n : ns) {
    std::mt19937_64 g(1); std::uniform_real_distribution<double> s(-1, 1);
    std::vector<double> A((size_t)n * n), b(n), x(n); for (auto& a : A) a = s(g); for (int i = 0; i < n; i++) A[i + (size_t)i * n] += n;
    for (auto& v : b) v = s(g);
    double *dA, *dB; magma_int_t* ip = new magma_int_t[n]; magma_int_t info;
    magma_dmalloc(&dA, (size_t)n * n); magma_dmalloc(&dB, n);
    double best = 1e9;
    for (int r = 0; r < 4; r++) { magma_dsetmatrix(n, n, A.data(), n, dA, n, Q); magma_dsetvector(n, b.data(), 1, dB, 1, Q);
      magma_queue_sync(Q); double t0 = magma_sync_wtime(Q); magma_dgesv_gpu(n, 1, dA, n, ip, dB, n, &info); double t1 = magma_sync_wtime(Q);
      if (r) best = fmin(best, t1 - t0); }
    magma_dgetvector(n, dB, 1, x.data(), 1, Q);
    // certification pass as in the patch
    magma_dsetmatrix(n, n, A.data(), n, dA, n, Q); magma_dsetvector(n, x.data(), 1, dB, 1, Q);
    double c0 = magma_sync_wtime(Q);
    std::vector<double> Ah((size_t)n * n), xh(n), bh(n);
    magma_dgetmatrix(n, n, dA, n, Ah.data(), n, Q); magma_dgetvector(n, dB, 1, xh.data(), 1, Q); magma_dgetvector(n, dB, 1, bh.data(), 1, Q);
    double c1 = magma_sync_wtime(Q); double rmax = 0;
    for (int i = 0; i < n; i++) { double sm = bh[i], es = 0;
      for (int j = 0; j < n; j++) { double p = -Ah[i + (size_t)j * n] * xh[j]; double ep = std::fma(-Ah[i + (size_t)j * n], xh[j], -p); double e; two_sum(sm, p, sm, e); es += e + ep; }
      rmax = fmax(rmax, fabs(sm + es)); }
    double c2 = magma_sync_wtime(Q);
    printf("n=%5d  FP64 dgesv_gpu=%8.4f s (info %lld) | host certification: D2H copy=%7.4f s, Dot2 pass=%7.4f s, total=%7.4f s  (r=%g)\n",
           n, best, (long long)info, c1 - c0, c2 - c1, c2 - c0, rmax);
    magma_free(dA); magma_free(dB); delete[] ip;
  }
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

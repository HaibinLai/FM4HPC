// Round 41 (roxy): alpha = 0 with NaN in A and B.  Reference BLAS does not reference A or B when alpha = 0
// (gemm/symm/syrk/syr2k: C := beta*C; gemv/symv: y := beta*y), so the result must be finite.  Earlier rounds used finite A.
// CSV: r41,routine,case,nonfinite_or_wrong,total,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
static magma_queue_t Q; static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static double* up(const std::vector<double>& h) { double* d; magma_dmalloc(&d, h.size()); magma_dsetvector((magma_int_t)h.size(), h.data(), 1, d, 1, Q); return d; }
static std::vector<double> down(double* d, size_t k) { std::vector<double> h(k); magma_dgetvector((magma_int_t)k, d, 1, h.data(), 1, Q); return h; }
static long long bad(const std::vector<double>& g, const std::vector<double>& c0, double beta, std::vector<bool>* mask = nullptr) { long long b = 0;
  for (size_t i = 0; i < g.size(); i++) { if (mask && !(*mask)[i]) continue; double e = beta * c0[i]; if (!std::isfinite(g[i]) || fabs(g[i] - e) > 1e-12 * (1 + fabs(e))) b++; } return b; }
static void rep(const char* r, int n, long long b, long long t) { printf("r41,%s,n=%d,%lld,%lld,%s\n", r, n, b, t, b ? "BAD" : "OK"); }
static double** arr(double* p, size_t stride, int B) { std::vector<double*> h(B); for (int i = 0; i < B; i++) h[i] = p + i * stride; double** d; magma_malloc((void**)&d, B * sizeof(double*)); cudaMemcpy(d, h.data(), B * sizeof(double*), cudaMemcpyHostToDevice); return d; }
int main() {
  setvbuf(stdout, NULL, _IONBF, 0); magma_init(); magma_queue_create(0, &Q); printf("r41,routine,case,bad,total,verdict\n");
  const double be = 0.5;
  for (int n : {1, 31, 64, 200}) {
    std::vector<double> A((size_t)n * n, NAN), C((size_t)n * n), x(n, NAN), y(n); for (auto& c : C) c = r1(); for (auto& v : y) v = r1();
    { double *dA = up(A), *dC = up(C); magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, n, n, n, 0.0, dA, n, dA, n, be, dC, n, Q); magma_queue_sync(Q); rep("magmablas_dgemm", n, bad(down(dC, C.size()), C, be), C.size()); magma_free(dA); magma_free(dC); }
    { const int B = 3; std::vector<double> AB(A.size() * B, NAN), CB; for (int b = 0; b < B; b++) CB.insert(CB.end(), C.begin(), C.end()); double *dA = up(AB), *dC = up(CB); double **pa = arr(dA, A.size(), B), **pc = arr(dC, C.size(), B);
      magmablas_dgemm_batched(MagmaNoTrans, MagmaNoTrans, n, n, n, 0.0, (double const* const*)pa, n, (double const* const*)pa, n, be, pc, n, B, Q); magma_queue_sync(Q); rep("magmablas_dgemm_batched", n, bad(down(dC, CB.size()), CB, be), CB.size());
      magma_dsetvector((magma_int_t)CB.size(), CB.data(), 1, dC, 1, Q); std::vector<magma_int_t> ns(B + 1, n); magma_int_t* dn; magma_imalloc(&dn, B + 1); magma_isetvector(B + 1, ns.data(), 1, dn, 1, Q);
      magmablas_dgemm_vbatched(MagmaNoTrans, MagmaNoTrans, dn, dn, dn, 0.0, (double const* const*)pa, dn, (double const* const*)pa, dn, be, pc, dn, B, Q); magma_queue_sync(Q); rep("magmablas_dgemm_vbatched", n, bad(down(dC, CB.size()), CB, be), CB.size());
      // syrk / syr2k batched (lower triangle only)
      std::vector<bool> lo(CB.size(), false); for (int b = 0; b < B; b++) for (int j = 0; j < n; j++) for (int i = j; i < n; i++) lo[b * C.size() + i + (size_t)j * n] = true;
      magma_dsetvector((magma_int_t)CB.size(), CB.data(), 1, dC, 1, Q); magmablas_dsyrk_batched(MagmaLower, MagmaNoTrans, n, n, 0.0, (double const* const*)pa, n, be, pc, n, B, Q); magma_queue_sync(Q); rep("magmablas_dsyrk_batched", n, bad(down(dC, CB.size()), CB, be, &lo), (long long)B * n * (n + 1) / 2);
      magma_dsetvector((magma_int_t)CB.size(), CB.data(), 1, dC, 1, Q); magmablas_dsyr2k_batched(MagmaLower, MagmaNoTrans, n, n, 0.0, (double const* const*)pa, n, (double const* const*)pa, n, be, pc, n, B, Q); magma_queue_sync(Q); rep("magmablas_dsyr2k_batched", n, bad(down(dC, CB.size()), CB, be, &lo), (long long)B * n * (n + 1) / 2);
      // gemv batched
      std::vector<double> XB((size_t)n * B, NAN), YB; for (int b = 0; b < B; b++) YB.insert(YB.end(), y.begin(), y.end()); double *dx = up(XB), *dy = up(YB); double **px = arr(dx, n, B), **py = arr(dy, n, B);
      magmablas_dgemv_batched(MagmaNoTrans, n, n, 0.0, (double const* const*)pa, n, (double const* const*)px, 1, be, py, 1, B, Q); magma_queue_sync(Q); rep("magmablas_dgemv_batched", n, bad(down(dy, YB.size()), YB, be), YB.size());
      magma_free(dA); magma_free(dC); magma_free(pa); magma_free(pc); magma_free(dn); magma_free(dx); magma_free(dy); magma_free(px); magma_free(py); }
    { std::vector<bool> lo(C.size(), false); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) lo[i + (size_t)j * n] = true;
      double *dA = up(A), *dC = up(C); magmablas_dsyrk(MagmaLower, MagmaNoTrans, n, n, 0.0, dA, n, be, dC, n, Q); magma_queue_sync(Q); rep("magmablas_dsyrk", n, bad(down(dC, C.size()), C, be, &lo), (long long)n * (n + 1) / 2); magma_free(dA); magma_free(dC); }
    { double *dA = up(A), *dx = up(x), *dy = up(y); magmablas_dgemv(MagmaNoTrans, n, n, 0.0, dA, n, dx, 1, be, dy, 1, Q); magma_queue_sync(Q); rep("magmablas_dgemv", n, bad(down(dy, y.size()), y, be), y.size());
      magma_dsetvector(n, y.data(), 1, dy, 1, Q); magmablas_dsymv(MagmaLower, n, 0.0, dA, n, dx, 1, be, dy, 1, Q); magma_queue_sync(Q); rep("magmablas_dsymv", n, bad(down(dy, y.size()), y, be), y.size());
      magma_free(dA); magma_free(dx); magma_free(dy); }
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

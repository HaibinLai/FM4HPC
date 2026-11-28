// Sixth GPU round: the FP16 conversion flag (magmablas/slag2h.cu) and its caller shpotrf_gpu (src/shpotrf_gpu.cpp).  GPU 0 only.
// One mode per process: a device fault poisons the CUDA context.
//   lag2h_ok      magmablas_slag2h, all entries within FP16 range                     (control)
//   lag2h_ovf     magmablas_slag2h, one entry 1e5 > 65504                             (kernel writes *&magma_flag, a host shadow address)
//   lag2h_nan     magmablas_slag2h, one NaN entry                                     (range test is two comparisons: NaN passes)
//   lag2hb_ovf    magmablas_slag2h_batched, batch of 3, matrices 1 and 2 overflow    (only info_array[0] is copied back)
//   dshposv_big   dshposv_gpu, n = 1024 tridiag*2^40: shpotrf runs slag2h on an L with entries > 65504
//   shpotrf_oom   shpotrf_gpu with dA filling most of the memory: dhW allocation fails -> goto cleanup over the queue creation
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmablas_h.h"
#include "magma_htc.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
static magma_queue_t Q;
static void cuda_state(const char* where) {
  cudaError_t e1 = cudaDeviceSynchronize(); cudaError_t e2 = cudaGetLastError();
  printf("  [%s] cudaDeviceSynchronize: %s | cudaGetLastError: %s\n", where, cudaGetErrorString(e1), cudaGetErrorString(e2));
}
static void lag2h(float special) {
  int m = 64, n = 64; std::vector<float> A((size_t)m * n); for (size_t i = 0; i < A.size(); i++) A[i] = (float)((i % 17) * 0.25 - 2.0);
  if (special != 0) A[5 + 7 * m] = special;
  float* dA; magmaHalf* dH; magma_smalloc(&dA, (size_t)m * n); magma_malloc((void**)&dH, (size_t)m * n * sizeof(magmaHalf));
  magma_ssetmatrix(m, n, A.data(), m, dA, m, Q);
  magma_int_t info = -999; magmablas_slag2h(m, n, dA, m, dH, m, &info, Q);
  printf("magmablas_slag2h m=n=64, special entry %g: info=%lld (documented: 1 if an entry is outside the FP16 range)\n", special, (long long)info);
  cuda_state("after slag2h");
}
static void lag2hb() {
  int m = 32, n = 32, batch = 3; std::vector<float*> dA(batch); std::vector<magmaHalf*> dH(batch);
  for (int b = 0; b < batch; b++) { std::vector<float> A((size_t)m * n, 0.5f); if (b > 0) A[3] = 1e6f;
    magma_smalloc(&dA[b], (size_t)m * n); magma_malloc((void**)&dH[b], (size_t)m * n * sizeof(magmaHalf)); magma_ssetmatrix(m, n, A.data(), m, dA[b], m, Q); }
  float** dAa; magmaHalf** dHa; magma_malloc((void**)&dAa, batch * sizeof(float*)); magma_malloc((void**)&dHa, batch * sizeof(magmaHalf*));
  magma_setvector(batch, sizeof(float*), dA.data(), 1, dAa, 1, Q); magma_setvector(batch, sizeof(magmaHalf*), dH.data(), 1, dHa, 1, Q);
  std::vector<magma_int_t> info(batch, -999);
  magmablas_slag2h_batched(m, n, dAa, m, dHa, m, info.data(), batch, Q);
  printf("magmablas_slag2h_batched, batch 3, matrices 1 and 2 contain 1e6: info_array = %lld %lld %lld (expected 0 1 1)\n",
         (long long)info[0], (long long)info[1], (long long)info[2]);
  cuda_state("after slag2h_batched");
}
static void dshposv_big() {
  int n = 1024; std::vector<double> A((size_t)n * n, 0.0), b(n), xt(n, 1.0); double sc = ldexp(1.0, 40);
  for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) A[i + (size_t)j * n] = sc * (i == j ? 4.0 : (abs(i - j) == 1 ? -1.0 : 0.0));
  for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += A[i + (size_t)j * n]; b[i] = s; }
  double *dA, *dB, *dX; magma_dmalloc(&dA, (size_t)n * n); magma_dmalloc(&dB, n); magma_dmalloc(&dX, n);
  magma_dsetmatrix(n, n, A.data(), n, dA, n, Q); magma_dsetmatrix(n, 1, b.data(), n, dB, n, Q);
  magma_int_t iter = -999, info = -999; magma_int_t ret = magma_dshposv_gpu(MagmaLower, n, 1, dA, n, dB, n, dX, n, &iter, &info);
  std::vector<double> x(n, NAN); magma_dgetmatrix(n, 1, dX, n, x.data(), n, Q);
  double e = 0; int nf = 0; for (int i = 0; i < n; i++) { if (!std::isfinite(x[i])) nf++; else e = fmax(e, fabs(x[i] - 1.0)); }
  printf("dshposv_gpu n=1024 tridiag*2^40 (L entries ~ 2^20 > 65504): ret=%lld iter=%lld info=%lld  nonfinite x=%d  max|x-1| (finite)=%.3g\n",
         (long long)ret, (long long)iter, (long long)info, nf, e);
  cuda_state("after dshposv");
}
static void shpotrf_oom() {
  size_t fr, tot; cudaMemGetInfo(&fr, &tot);
  // dA (4 n^2 bytes) must fit and dhW (2 roundup(n,128) n bytes) must not: choose 4 n^2 ~ 0.8 free
  int n = (int)(sqrt(0.8 * (double)fr / 4.0)); n -= n % 128;
  printf("free %.2f GB of %.2f GB; n = %d: dA %.2f GB, dhW %.2f GB\n", fr / 1e9, tot / 1e9, n, 4.0 * n * n / 1e9, 2.0 * n * n / 1e9);
  float* dA; if (magma_smalloc(&dA, (size_t)n * n) != MAGMA_SUCCESS) { printf("dA allocation failed; choose smaller n\n"); return; }
  cudaMemset(dA, 0, (size_t)n * n * sizeof(float));
  magma_int_t info = -999;
  printf("calling magma_shpotrf_gpu ...\n"); fflush(stdout);
  magma_int_t ret = magma_shpotrf_gpu(MagmaLower, n, dA, n, &info);
  printf("returned ret=%lld info=%lld (MAGMA_ERR_DEVICE_ALLOC=%d, MAGMA_ERR_HOST_ALLOC=%d)\n", (long long)ret, (long long)info, MAGMA_ERR_DEVICE_ALLOC, MAGMA_ERR_HOST_ALLOC);
}
static void shpotrf_big() {
  printf("== shpotrf_gpu (CUDA path: cublasGemmEx COMPUTE_32F_FAST_16F) vs spotrf_gpu on SPD tridiag(-1,4,-1)*2^k, n = 2048 ==\n");
  int n = 2048;
  for (int k : {0, 8, 14, 20, 30, 40}) {
    double sc = ldexp(1.0, k); std::vector<float> A((size_t)n * n, 0.0f);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) A[i + (size_t)j * n] = (float)(sc * (i == j ? 4.0 : (abs(i - j) == 1 ? -1.0 : 0.0)));
    float* dA; magma_smalloc(&dA, (size_t)n * n);
    magma_int_t ih = -999, is = -999, inat = -999;
    magma_ssetmatrix(n, n, A.data(), n, dA, n, Q); magma_shpotrf_gpu(MagmaLower, n, dA, n, &ih);
    std::vector<float> L((size_t)n * n); magma_sgetmatrix(n, n, dA, n, L.data(), n, Q);
    long nf = 0; for (int j = 0; j < n; j++) for (int i = j; i < n; i++) if (!std::isfinite(L[i + (size_t)j * n])) nf++;
    double d = L[(n - 1) + (size_t)(n - 1) * n] / sqrt(sc);
    magma_ssetmatrix(n, n, A.data(), n, dA, n, Q); magma_shpotrf_native(MagmaLower, n, dA, n, &inat);
    magma_ssetmatrix(n, n, A.data(), n, dA, n, Q); magma_spotrf_gpu(MagmaLower, n, dA, n, &is);
    printf("k=%2d  max|a|=%.3g  shpotrf_gpu info=%lld (nonfinite L entries %ld, L(n,n)/sqrt(scale)=%.6g, exact limit sqrt(2+sqrt(3)) = 1.93185)  shpotrf_native info=%lld  spotrf_gpu info=%lld\n",
           k, 4 * sc, (long long)ih, nf, d, (long long)inat, (long long)is);
    magma_free(dA);
  }
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  std::string m = argc > 1 ? argv[1] : "lag2h_ok";
  magma_init(); magma_queue_create(0, &Q);
  if (m == "lag2h_ok") lag2h(0); else if (m == "lag2h_ovf") lag2h(1e5f); else if (m == "lag2h_nan") lag2h(NAN);
  else if (m == "lag2hb_ovf") lag2hb(); else if (m == "dshposv_big") dshposv_big(); else if (m == "shpotrf_oom") shpotrf_oom(); else if (m == "shpotrf_big") shpotrf_big();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

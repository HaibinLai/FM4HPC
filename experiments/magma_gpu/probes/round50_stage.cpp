// Round 50 (roxy): does magma_zsetmatrix_async (cublasSetMatrixAsync) from pinned host memory read the host data
// when the copy executes (CUDA semantics) or already at enqueue time?  The stream is held by a 100 ms host function,
// the copy is enqueued, then the host overwrites the source.  CSV: r50stage,bytes,lda_eq_m,device_sees(old|new)
#include "magma_v2.h"
#include <cuda_runtime.h>
#include <thread>
#include <chrono>
#include <cstdio>
#include <vector>
static void CUDART_CB slp(void*) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
int main() {
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  for (int nb : {16, 64, 90, 128, 256, 512}) for (int pad : {0, 1}) {
    int m = nb, n = nb, lda = nb + pad * 64; magmaDoubleComplex *h, *d; magma_malloc_pinned((void**)&h, (size_t)lda * n * 16); magma_zmalloc(&d, (size_t)m * n);
    for (size_t i = 0; i < (size_t)lda * n; i++) h[i] = MAGMA_Z_MAKE(1, 0);
    cudaLaunchHostFunc(magma_queue_get_cuda_stream(q), slp, nullptr);
    magma_zsetmatrix_async(m, n, h, lda, d, m, q);
    for (size_t i = 0; i < (size_t)lda * n; i++) h[i] = MAGMA_Z_MAKE(2, 0);
    magma_queue_sync(q); std::vector<magmaDoubleComplex> g((size_t)m * n); magma_zgetmatrix(m, n, d, m, g.data(), m, q);
    int old = 0, nw = 0; for (auto& x : g) (MAGMA_Z_REAL(x) == 1 ? old : nw)++;
    printf("r50stage,%d,%d,%s (old %d new %d)\n", m * n * 16, pad == 0, old && !nw ? "old" : (!old ? "new" : "mixed"), old, nw);
    magma_free_pinned(h); magma_free(d); }
  magma_queue_destroy(q); magma_finalize(); return 0; }

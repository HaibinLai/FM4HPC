// Round 51 (roxy): MAGMA routines that take a queue but do part of their work elsewhere (legacy default stream,
// one global flag), and the sync-free CSC triangular solve whose consumer wait is commented out.
// One check per process:
//   round51_stream flag <zlag2c|dlag2s|zlat2c|dlat2s|slag2h|slag2h_ok> <magma|blocking|nonblocking> <delay_ms>
//       64x64 matrix with one entry beyond the target overflow threshold; expected info = 1 (slag2h_ok: no such entry, info 0).
//       The queue's stream optionally sleeps delay_ms first (cudaLaunchHostFunc), so its kernel runs late.
//   round51_stream h2batched <k>        slag2h_batched, 4 matrices, overflow only in matrix k: expected info_array[k] = 1
//   round51_stream trtri <magma|blocking|nonblocking> [busy_ms=150]
//       an unrelated blocking stream sleeps busy_ms; then magmablas_dtrtri_diag on the queue, queue sync, check
//       A_bb * inv_bb = I right after the queue sync (snapshot 1, read on the queue) and after a device sync (2)
//   round51_stream syncfree <d|z> <bidiag|random> <m>     magma_[dz]gecscsyncfreetrsm_analysis + _solve, forward
// CSV: r51,<mode>,<args...>,observed,expected,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <algorithm>
static void CUDART_CB slp(void* ms) { std::this_thread::sleep_for(std::chrono::milliseconds((size_t)ms)); }
static magma_queue_t mkq(const std::string& k, cudaStream_t* s) {
  magma_queue_t q;
  if (k == "magma") { magma_queue_create(0, &q); *s = magma_queue_get_cuda_stream(q); return q; }
  cudaStreamCreateWithFlags(s, k == "nonblocking" ? cudaStreamNonBlocking : cudaStreamDefault);
  magma_queue_create_from_cuda(0, *s, NULL, NULL, &q); return q; }
static unsigned Sd = 7u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 3) { printf("usage: see source\n"); return 1; }
  std::string mode = argv[1]; magma_init();
  if (mode == "flag" && argc >= 5) {
    std::string r = argv[2], qk = argv[3]; int delay = atoi(argv[4]); cudaStream_t s; magma_queue_t q = mkq(qk, &s);
    const int n = 64; magma_int_t info = 0;
    if (delay > 0) cudaLaunchHostFunc(s, slp, (void*)(size_t)delay);
    if (r == "zlag2c" || r == "zlat2c") {
      std::vector<magmaDoubleComplex> h((size_t)n * n, MAGMA_Z_MAKE(0.5, 0.25)); h[5 + 7 * n] = MAGMA_Z_MAKE(1e300, 0); if (r == "zlat2c") h[7 + 5 * n] = h[5 + 7 * n];
      magmaDoubleComplex* d; magmaFloatComplex* sa; magma_zmalloc(&d, h.size()); magma_cmalloc(&sa, h.size()); cudaMemcpy(d, h.data(), h.size() * sizeof(h[0]), cudaMemcpyHostToDevice);
      if (r == "zlag2c") magmablas_zlag2c(n, n, d, n, sa, n, q, &info); else magmablas_zlat2c(MagmaLower, n, d, n, sa, n, q, &info);
    } else if (r == "dlag2s" || r == "dlat2s") {
      std::vector<double> h((size_t)n * n, 0.5); h[5 + 7 * n] = 1e300; if (r == "dlat2s") h[7 + 5 * n] = 1e300;
      double* d; float* sa; magma_dmalloc(&d, h.size()); magma_smalloc(&sa, h.size()); cudaMemcpy(d, h.data(), h.size() * sizeof(h[0]), cudaMemcpyHostToDevice);
      if (r == "dlag2s") magmablas_dlag2s(n, n, d, n, sa, n, q, &info); else magmablas_dlat2s(MagmaLower, n, d, n, sa, n, q, &info);
    } else if (r == "slag2h" || r == "slag2h_ok") {
      std::vector<float> h((size_t)n * n, 0.5f); if (r == "slag2h") h[5 + 7 * n] = 1e10f;
      float* d; magmaHalf* ha; magma_smalloc(&d, h.size()); magma_malloc((void**)&ha, h.size() * sizeof(magmaHalf)); cudaMemcpy(d, h.data(), h.size() * sizeof(h[0]), cudaMemcpyHostToDevice);
      magmablas_slag2h(n, n, d, n, ha, n, &info, q);
    } else { printf("unknown routine\n"); return 1; }
    magma_int_t at_return = info; magma_queue_sync(q); int expect = r == "slag2h_ok" ? 0 : 1;
    printf("r51,flag,%s,%s,%d,info=%lld,%d,%s\n", r.c_str(), qk.c_str(), delay, (long long)at_return, expect, at_return == expect ? "OK" : "WRONG");
  }
  else if (mode == "h2batched" && argc >= 3) {
    int k = atoi(argv[2]); const int n = 32, B = 4; cudaStream_t s; magma_queue_t q = mkq("magma", &s);
    std::vector<float*> hA(B); std::vector<magmaHalf*> hH(B);
    for (int b = 0; b < B; b++) { std::vector<float> h((size_t)n * n, 0.5f); if (b == k) h[3 + 4 * n] = 1e10f; magma_smalloc(&hA[b], h.size()); magma_malloc((void**)&hH[b], h.size() * sizeof(magmaHalf)); cudaMemcpy(hA[b], h.data(), h.size() * 4, cudaMemcpyHostToDevice); }
    float** dA; magmaHalf** dH; magma_malloc((void**)&dA, B * sizeof(float*)); magma_malloc((void**)&dH, B * sizeof(magmaHalf*));
    cudaMemcpy(dA, hA.data(), B * sizeof(float*), cudaMemcpyHostToDevice); cudaMemcpy(dH, hH.data(), B * sizeof(magmaHalf*), cudaMemcpyHostToDevice);
    std::vector<magma_int_t> info(B, 0); magmablas_slag2h_batched(n, n, dA, n, dH, n, info.data(), B, q); magma_queue_sync(q);
    char obs[64]; snprintf(obs, sizeof obs, "info=[%lld %lld %lld %lld]", (long long)info[0], (long long)info[1], (long long)info[2], (long long)info[3]);
    printf("r51,h2batched,%d,%s,info[%d]=1,%s\n", k, obs, k, info[k] == 1 ? "OK" : "WRONG");
  }
  else if (mode == "trtri" && argc >= 3) {
    std::string qk = argv[2]; cudaStream_t s; magma_queue_t q = mkq(qk, &s); cudaStream_t other; cudaStreamCreate(&other);
    const int n = 128; std::vector<double> A((size_t)n * n, 0.0); for (int j = 0; j < n; j++) { A[j + j * n] = 2 + r1(); for (int i = j + 1; i < n; i++) A[i + j * n] = 0.1 * r1(); }
    double *dA, *dinv; magma_dmalloc(&dA, A.size()); magma_dmalloc(&dinv, (size_t)n * n); cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice); cudaMemset(dinv, 0, (size_t)n * n * 8); cudaDeviceSynchronize();
    int busy = argc > 3 ? atoi(argv[3]) : 150; if (busy > 0) cudaLaunchHostFunc(other, slp, (void*)(size_t)busy);   // unrelated work on another (blocking) stream
    double t0 = now(); magmablas_dtrtri_diag(MagmaLower, MagmaNonUnit, n, dA, n, dinv, q); double t1 = now(); magma_queue_sync(q); double t2 = now();
    auto resid = [&](const std::vector<double>& X) { double e = 0; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { double sum = 0; for (int k = 0; k < n; k++) sum += A[i + k * n] * X[k + j * n]; e = std::max(e, fabs(sum - (i == j ? 1.0 : 0.0))); } return e; };
    std::vector<double> X1(A.size()), X2(A.size()); cudaMemcpyAsync(X1.data(), dinv, X1.size() * 8, cudaMemcpyDeviceToHost, s); cudaStreamSynchronize(s);
    cudaDeviceSynchronize(); cudaMemcpy(X2.data(), dinv, X2.size() * 8, cudaMemcpyDeviceToHost);
    double e1 = resid(X1), e2 = resid(X2);
    printf("r51,trtri,%s,busy %d ms: call %.1f ms queue_sync %.1f ms; |A*inv-I| after queue sync %.3g after device sync %.3g,<1e-12,%s\n", qk.c_str(), busy, t1 - t0, t2 - t1, e1, e2, (e1 < 1e-12 && e2 < 1e-12) ? "OK" : "WRONG");
  }
  else if (mode == "syncfree" && argc >= 5) {
    std::string p = argv[2], shape = argv[3]; int m = atoi(argv[4]); cudaStream_t s; magma_queue_t q = mkq("magma", &s);
    // lower-triangular CSC, diagonal first in each column
    std::vector<magma_index_t> colptr(1, 0), rowind; std::vector<double> val;
    for (int j = 0; j < m; j++) { rowind.push_back(j); val.push_back(2.0 + 0.5 * r1());
      if (shape == "bidiag") { if (j + 1 < m) { rowind.push_back(j + 1); val.push_back(-1.0); } }
      else { for (int t = 0; t < 4; t++) { int i = j + 1 + (int)((r1() + 0.5) * 40); if (i < m && std::find(rowind.begin() + colptr[j], rowind.end(), i) == rowind.end()) { rowind.push_back(i); val.push_back(0.3 * r1()); } } }
      colptr.push_back((magma_index_t)rowind.size()); }
    int nnz = (int)val.size(); std::vector<double> xt(m), b(m, 0.0); for (int i = 0; i < m; i++) xt[i] = 1 + 0.01 * i;
    for (int j = 0; j < m; j++) for (int k = colptr[j]; k < colptr[j + 1]; k++) b[rowind[k]] += val[k] * xt[j];
    magma_index_t *dcol, *drow, *dg, *dgb; magma_index_malloc(&dcol, m + 1); magma_index_malloc(&drow, nnz); magma_index_malloc(&dg, m); magma_index_malloc(&dgb, m);
    cudaMemcpy(dcol, colptr.data(), (m + 1) * sizeof(magma_index_t), cudaMemcpyHostToDevice); cudaMemcpy(drow, rowind.data(), nnz * sizeof(magma_index_t), cudaMemcpyHostToDevice);
    std::vector<double> x(m); double e = 0;
    if (p == "d") {
      double *dv, *dx, *db; magma_dmalloc(&dv, nnz); magma_dmalloc(&dx, m); magma_dmalloc(&db, m);
      cudaMemcpy(dv, val.data(), nnz * 8, cudaMemcpyHostToDevice); cudaMemcpy(db, b.data(), m * 8, cudaMemcpyHostToDevice);
      magma_dgecscsyncfreetrsm_analysis(m, nnz, dv, dcol, drow, dg, dgb, q);
      magma_dgecscsyncfreetrsm_solve(m, nnz, dv, dcol, drow, dg, dgb, dx, db, 0, 1, q); cudaDeviceSynchronize();
      cudaMemcpy(x.data(), dx, m * 8, cudaMemcpyDeviceToHost);
    } else {
      std::vector<magmaDoubleComplex> zv(nnz), zb(m), zx(m); for (int k = 0; k < nnz; k++) zv[k] = MAGMA_Z_MAKE(val[k], 0); for (int i = 0; i < m; i++) zb[i] = MAGMA_Z_MAKE(b[i], 0);
      magmaDoubleComplex *dv, *dx, *db; magma_zmalloc(&dv, nnz); magma_zmalloc(&dx, m); magma_zmalloc(&db, m);
      cudaMemcpy(dv, zv.data(), nnz * 16, cudaMemcpyHostToDevice); cudaMemcpy(db, zb.data(), m * 16, cudaMemcpyHostToDevice);
      magma_zgecscsyncfreetrsm_analysis(m, nnz, dv, dcol, drow, dg, dgb, q);
      magma_zgecscsyncfreetrsm_solve(m, nnz, dv, dcol, drow, dg, dgb, dx, db, 0, 1, q); cudaDeviceSynchronize();
      cudaMemcpy(zx.data(), dx, m * 16, cudaMemcpyDeviceToHost); for (int i = 0; i < m; i++) x[i] = MAGMA_Z_REAL(zx[i]);
    }
    int bad = 0; for (int i = 0; i < m; i++) { double d = fabs(x[i] - xt[i]); if (!(d <= 1e-10 * fabs(xt[i]))) bad++; e = std::max(e, d != d ? 1e300 : d); }
    printf("r51,syncfree,%s,%s,%d,nnz=%d max|x-x_true|=%.3g wrong_entries=%d,0,%s\n", p.c_str(), shape.c_str(), m, nnz, e, bad, bad == 0 ? "OK" : "WRONG");
  }
  else { printf("bad arguments\n"); return 1; }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_finalize(); return 0;
}

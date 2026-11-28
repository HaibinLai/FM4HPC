// Round 68 (roxy): magma_zgeqr2x{,2,3,4}_gpu with dwork at its documented size followed by a sentinel guard.
//   documented dwork: v1 "COMPLEX_16 array, dimension (N)" = 2N doubles; v2, v3, v4 "DOUBLE PRECISION array, dimension (3 N)".
//   (MAGMA's testing_zgeqr2x_gpu allocates max(5*min_mn, (BLOCK_SIZE*2+2)*min_mn) doubles.)
//   round68_geqr2x <v 1..4> <m> <n> [docsize_mult]   (docsize_mult overrides the documented size in doubles per column)
//   checks: guard doubles written; tau and the R block (ddA, upper n x n) against LAPACK zgeqrf.
// CSV: r68,v,m,n,doc_doubles,guard,first_written,tau_err,R_err,info,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
typedef magmaDoubleComplex Z;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 4) return 1; int v = atoi(argv[1]); magma_int_t m = atoi(argv[2]), n = atoi(argv[3]); double mult = argc > 4 ? atof(argv[4]) : (v == 1 ? 2 : 3);
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  magma_int_t ldda = magma_roundup(m, 32), mn = std::min(m, n), info = 0;
  std::vector<Z> A((size_t)m * n); for (auto& a : A) a = MAGMA_Z_MAKE(r1(), r1());
  Z *dA, *dtau, *dT, *ddA; magma_zmalloc(&dA, (size_t)ldda * n); magma_zmalloc(&dtau, std::max<magma_int_t>(mn, 1)); magma_zmalloc(&dT, (size_t)n * n); magma_zmalloc(&ddA, (size_t)n * n);
  magma_zsetmatrix(m, n, A.data(), m, dA, ldda, q);
  size_t doc = (size_t)(mult * n), G = 1 << 16; double* dw; magma_dmalloc(&dw, doc + G);
  { std::vector<double> s(doc + G, 777.25); magma_dsetvector((magma_int_t)(doc + G), s.data(), 1, dw, 1, q); }
  if (v == 1) magma_zgeqr2x_gpu(m, n, dA, ldda, dtau, dT, ddA, dw, &info);
  else if (v == 2) magma_zgeqr2x2_gpu(m, n, dA, ldda, dtau, dT, ddA, dw, &info);
  else if (v == 3) magma_zgeqr2x3_gpu(m, n, dA, ldda, dtau, dT, ddA, dw, &info);
  else magma_zgeqr2x4_gpu(m, n, dA, ldda, dtau, dT, ddA, dw, q, &info);
  magma_queue_sync(q);
  std::vector<double> g(G); magma_dgetvector((magma_int_t)G, dw + doc, 1, g.data(), 1, q); size_t w = 0, first = 0; for (size_t i = 0; i < G; i++) if (g[i] != 777.25) { if (!w) first = i; w++; }
  // reference
  std::vector<Z> R = A, tr(mn); magma_int_t lw = -1, ri; Z wq; lapackf77_zgeqrf(&m, &n, R.data(), &m, tr.data(), &wq, &lw, &ri); lw = (magma_int_t)MAGMA_Z_REAL(wq); { std::vector<Z> wk(lw); lapackf77_zgeqrf(&m, &n, R.data(), &m, tr.data(), wk.data(), &lw, &ri); }
  std::vector<Z> tau(mn), DD((size_t)n * n); magma_zgetvector(mn, dtau, 1, tau.data(), 1, q); magma_zgetmatrix(n, n, ddA, n, DD.data(), n, q);
  double et = 0, er = 0, ar = 0; for (int i = 0; i < mn; i++) et = std::max(et, MAGMA_Z_ABS(MAGMA_Z_SUB(tau[i], tr[i])));
  for (int j = 0; j < n; j++) for (int i = 0; i <= std::min<int>(j, (int)m - 1); i++) { er = std::max(er, MAGMA_Z_ABS(MAGMA_Z_SUB(DD[i + (size_t)j * n], R[i + (size_t)j * m]))); ar = std::max(ar, MAGMA_Z_ABS(R[i + (size_t)j * m])); }
  er = ar > 0 ? er / ar : er;
  bool ok = info == 0 && w == 0 && et < 1e-10 && er < 1e-10;
  printf("r68,%d,%lld,%lld,%zu,%zu,%zu,%.3g,%.3g,%lld,%s\n", v, (long long)m, (long long)n, doc, w, w ? first : 0, et, er, (long long)info, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

// Round 82 (roxy): magma_zhetrs_gpu (magmablas_z.h; the only in-library call, in zchesv_gpu, is commented out), never tested.
//   round82_hetrs <U|L> <n> <nrhs> <kind 0|1>
//       A Hermitian; kind 0: diagonally dominant (1x1 pivots only), kind 1: indefinite with small diagonal (2x2 pivots occur)
//       LAPACK zhetrf on the host, the factored A to the device, magma_zhetrs_gpu with the host ipiv; X against LAPACK zhetrs and |A X - B|
// CSV: r82,uplo,n,nrhs,kind,n2x2,info,err_vs_lapack,resid,cuda,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
typedef magmaDoubleComplex Z;
static unsigned Sd = 3u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 5) return 1; char ul = argv[1][0]; magma_int_t n = atoi(argv[2]), nrhs = atoi(argv[3]); int kind = atoi(argv[4]);
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  magma_int_t ld = n; std::vector<Z> A((size_t)ld * n), B((size_t)ld * nrhs);
  for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { Z v = i == j ? MAGMA_Z_MAKE(kind ? 0.1 * r1() : 4 + r1(), 0) : MAGMA_Z_MAKE(r1(), r1()); A[i + (size_t)j * ld] = v; A[j + (size_t)i * ld] = MAGMA_Z_CONJ(v); }
  for (auto& b : B) b = MAGMA_Z_MAKE(r1(), r1());
  std::vector<Z> F = A; std::vector<magma_int_t> ipiv(n); magma_int_t lw = -1, linfo; Z wq; char u = ul;
  lapackf77_zhetrf(&u, &n, F.data(), &ld, ipiv.data(), &wq, &lw, &linfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); std::vector<Z> w(std::max<magma_int_t>(lw, 1)); lapackf77_zhetrf(&u, &n, F.data(), &ld, ipiv.data(), w.data(), &lw, &linfo);
  int n2 = 0; for (int i = 0; i < n; i++) if (ipiv[i] < 0) n2++;
  std::vector<Z> XL = B; lapackf77_zhetrs(&u, &n, &nrhs, F.data(), &ld, ipiv.data(), XL.data(), &ld, &linfo);
  Z *dA, *dB; magma_zmalloc(&dA, (size_t)ld * n); magma_zmalloc(&dB, (size_t)ld * nrhs); magma_zsetmatrix(n, n, F.data(), ld, dA, ld, q); magma_zsetmatrix(n, nrhs, B.data(), ld, dB, ld, q);
  magma_int_t info = 0; magma_zhetrs_gpu(ul == 'U' ? MagmaUpper : MagmaLower, n, nrhs, dA, ld, ipiv.data(), dB, ld, &info, q); magma_queue_sync(q); cudaError_t ce = cudaGetLastError();
  std::vector<Z> X((size_t)ld * nrhs); magma_zgetmatrix(n, nrhs, dB, ld, X.data(), ld, q);
  double e = 0, a = 0, r = 0, bn = 0; int badcols = 0;
  for (int c = 0; c < nrhs; c++) { double ec = 0; for (int i = 0; i < n; i++) { ec = std::max(ec, MAGMA_Z_ABS(MAGMA_Z_SUB(X[i + (size_t)c * ld], XL[i + (size_t)c * ld]))); a = std::max(a, MAGMA_Z_ABS(XL[i + (size_t)c * ld])); }
    e = std::max(e, ec); if (!(ec < 1e-8 * (1 + a))) badcols++;
    for (int i = 0; i < n; i++) { Z s = MAGMA_Z_ZERO; for (int l = 0; l < n; l++) s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(A[i + (size_t)l * ld], X[l + (size_t)c * ld])); r = std::max(r, MAGMA_Z_ABS(MAGMA_Z_SUB(s, B[i + (size_t)c * ld]))); bn = std::max(bn, MAGMA_Z_ABS(B[i + (size_t)c * ld])); } }
  double rel = a > 0 ? e / a : e; bool ok = info == 0 && ce == cudaSuccess && rel < 1e-10;
  printf("r82,%c,%lld,%lld,%d,%d,%lld,%.3g,%.3g,%s,%s\n", ul, (long long)n, (long long)nrhs, kind, n2, (long long)info, rel, r / bn, ce == cudaSuccess ? "ok" : cudaGetErrorString(ce), ok ? "OK" : "WRONG");
  fprintf(stderr, "bad columns %d of %lld\nDONE\n", badcols, (long long)nrhs); magma_queue_destroy(q); magma_finalize(); return 0;
}

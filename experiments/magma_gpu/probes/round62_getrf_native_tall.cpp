// Round 62 (roxy): magma_zgetrf_native on tall matrices.  magma_zgetrf_recpanel_native -> magma_zgetf2_native ->
// magma_zgetf2_native_fused rejects m > ZGETF2_FUSED_MAX_M (7168 for Z) with -1 (xerbla) and the return value is dropped.
//   round62_getrf_native_tall <native|gpu> <m> <n>
//   native: magma_zgetrf_native; gpu: magma_zgetrf_gpu (hybrid) as control.
//   checks: info (from the routine), ipiv against LAPACK zgetrf, and |P A x - L U x| / (|A| |x| n) on a random x.
// CSV: r62,op,m,n,info,ipiv_mismatch,res,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
typedef magmaDoubleComplex Z;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 4) { printf("usage: see source\n"); return 1; }
  std::string op = argv[1]; magma_int_t m = atoi(argv[2]), n = atoi(argv[3]), mn = std::min(m, n), ld = magma_roundup(m, 32);
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  std::vector<Z> A((size_t)m * n); for (auto& a : A) a = MAGMA_Z_MAKE(r1(), r1());
  Z* dA; magma_zmalloc(&dA, (size_t)ld * n); magma_zsetmatrix(m, n, A.data(), m, dA, ld, q);
  std::vector<magma_int_t> ip(mn), rip(mn); magma_int_t info = 0, rinfo = 0;
  if (op == "native") magma_zgetrf_native(m, n, dA, ld, ip.data(), &info); else magma_zgetrf_gpu(m, n, dA, ld, ip.data(), &info);
  std::vector<Z> LU((size_t)m * n); magma_zgetmatrix(m, n, dA, ld, LU.data(), m, q);
  { std::vector<Z> R = A; lapackf77_zgetrf(&m, &n, R.data(), &m, rip.data(), &rinfo); }
  int mis = 0; for (magma_int_t i = 0; i < mn; i++) if (ip[i] != rip[i]) mis++;
  // residual: y = P A x (x of length n), z = L (U x); L is m x mn unit lower, U is mn x n
  std::vector<Z> x(n), y(m, MAGMA_Z_ZERO), u(mn, MAGMA_Z_ZERO), z(m, MAGMA_Z_ZERO); double am = 0, xm = 0;
  for (auto& v : x) { v = MAGMA_Z_MAKE(r1(), r1()); xm = std::max(xm, MAGMA_Z_ABS(v)); } for (auto& a : A) am = std::max(am, MAGMA_Z_ABS(a));
  for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i < m; i++) y[i] = MAGMA_Z_ADD(y[i], MAGMA_Z_MUL(A[i + j * m], x[j]));
  bool badpiv = false; for (magma_int_t i = 0; i < mn; i++) { magma_int_t p = ip[i] - 1; if (p < i || p >= m) { badpiv = true; break; } std::swap(y[i], y[p]); }
  for (magma_int_t i = 0; i < mn; i++) for (magma_int_t j = i; j < n; j++) u[i] = MAGMA_Z_ADD(u[i], MAGMA_Z_MUL(LU[i + j * m], x[j]));
  double e = 0; for (magma_int_t i = 0; i < m; i++) { z[i] = i < mn ? u[i] : MAGMA_Z_ZERO; for (magma_int_t j = 0; j < std::min(i, mn); j++) z[i] = MAGMA_Z_ADD(z[i], MAGMA_Z_MUL(LU[i + j * m], u[j])); e = std::max(e, MAGMA_Z_ABS(MAGMA_Z_SUB(y[i], z[i]))); }
  double res = badpiv ? 1e300 : e / (am * xm * n);
  bool ok = info == 0 && rinfo == 0 && res < 1e-12;
  printf("r62,%s,%lld,%lld,%lld,%d,%.3g,%s\n", op.c_str(), (long long)m, (long long)n, (long long)info, mis, res, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_free(dA); magma_queue_destroy(q); magma_finalize(); return 0;
}

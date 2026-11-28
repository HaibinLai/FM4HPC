// Round 83 (roxy): magmablas element-wise kernels on very wide matrices.  zlaset and zlacpy split their 2-D grid into
// chunks of at most 65535 blocks ("super grid"); zgeadd, zgeadd2, ztranspose(_conj), zlag2c and clag2z launch
// grid.y = ceil(n / 32) directly, which exceeds the CUDA limit of 65535 for n > 2,097,120.
//   round83_wide <op> <m> <n>     op: laset lacpy geadd geadd2 transpose transpose_conj lag2c clag2z
// CSV: r83,op,m,n,grid_y,err,cuda,verdict
#include "magma_v2.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
typedef magmaDoubleComplex Z; typedef magmaFloatComplex C;
static unsigned Sd = 9u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 4) return 1; std::string op = argv[1]; magma_int_t m = atoi(argv[2]), n = atoi(argv[3]); magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  size_t N = (size_t)m * n; std::vector<Z> A(N), B(N); for (auto& a : A) a = MAGMA_Z_MAKE(r1(), r1()); for (auto& b : B) b = MAGMA_Z_MAKE(r1(), r1());
  Z al = MAGMA_Z_MAKE(0.7, -0.3), be = MAGMA_Z_MAKE(-0.4, 0.2); double err = 0; magma_int_t info = 0;
  Z *dA, *dB; magma_zmalloc(&dA, N); magma_zmalloc(&dB, N); magma_zsetvector((magma_int_t)N, A.data(), 1, dA, 1, q); magma_zsetvector((magma_int_t)N, B.data(), 1, dB, 1, q);
  std::vector<Z> R(N), X(N);
  if (op == "laset") { magmablas_zlaset(MagmaFull, m, n, al, be, dB, m, q); for (size_t k = 0; k < N; k++) R[k] = (k % m) == (k / m) ? be : al; }
  else if (op == "lacpy") { magmablas_zlacpy(MagmaFull, m, n, dA, m, dB, m, q); R = A; }
  else if (op == "geadd") { magmablas_zgeadd(m, n, al, dA, m, dB, m, q); for (size_t k = 0; k < N; k++) R[k] = MAGMA_Z_ADD(MAGMA_Z_MUL(al, A[k]), B[k]); }
  else if (op == "geadd2") { magmablas_zgeadd2(m, n, al, dA, m, be, dB, m, q); for (size_t k = 0; k < N; k++) R[k] = MAGMA_Z_ADD(MAGMA_Z_MUL(al, A[k]), MAGMA_Z_MUL(be, B[k])); }
  else if (op == "transpose" || op == "transpose_conj") { bool cj = op == "transpose_conj"; if (cj) magmablas_ztranspose_conj(m, n, dA, m, dB, n, q); else magmablas_ztranspose(m, n, dA, m, dB, n, q);
    for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i < m; i++) R[j + (size_t)i * n] = cj ? MAGMA_Z_CONJ(A[i + (size_t)j * m]) : A[i + (size_t)j * m]; }
  else if (op == "lag2c") { C* dS; magma_cmalloc(&dS, N); magmablas_zlag2c(m, n, dA, m, dS, m, q, &info); magma_queue_sync(q); std::vector<C> S(N); magma_cgetvector((magma_int_t)N, dS, 1, S.data(), 1, q);
    for (size_t k = 0; k < N; k++) { double e = fabs(MAGMA_C_REAL(S[k]) - MAGMA_Z_REAL(A[k])) + fabs(MAGMA_C_IMAG(S[k]) - MAGMA_Z_IMAG(A[k])); if (!(e <= err)) err = e; } }
  else if (op == "clag2z") { std::vector<C> S(N); for (size_t k = 0; k < N; k++) S[k] = MAGMA_C_MAKE((float)MAGMA_Z_REAL(A[k]), (float)MAGMA_Z_IMAG(A[k])); C* dS; magma_cmalloc(&dS, N); magma_csetvector((magma_int_t)N, S.data(), 1, dS, 1, q);
    magmablas_clag2z(m, n, dS, m, dB, m, q, &info); for (size_t k = 0; k < N; k++) R[k] = MAGMA_Z_MAKE(MAGMA_C_REAL(S[k]), MAGMA_C_IMAG(S[k])); }
  magma_queue_sync(q); cudaError_t ce = cudaGetLastError();
  if (op != "lag2c") { magma_zgetvector((magma_int_t)N, dB, 1, X.data(), 1, q); for (size_t k = 0; k < N; k++) { double e = MAGMA_Z_ABS(MAGMA_Z_SUB(X[k], R[k])); if (!(e <= err)) err = e; } }
  bool ok = ce == cudaSuccess && info == 0 && err < (op == "lag2c" ? 1e-6 : 1e-14);
  printf("r83,%s,%lld,%lld,%lld,%.3g,%s,%s\n", op.c_str(), (long long)m, (long long)n, (long long)((n + 31) / 32), err, ce == cudaSuccess ? "ok" : cudaGetErrorString(ce), ok ? "OK" : "WRONG");
  magma_queue_destroy(q); magma_finalize(); return 0;
}

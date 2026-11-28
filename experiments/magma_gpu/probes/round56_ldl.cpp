// Round 56 (roxy): Hermitian / symmetric indefinite factorizations and the solvers that consume their output.
//   round56_ldl <op> <n> <nrhs> <uplo L|U> <pad> <kind>
//   op: hesv_nopiv_gpu sysv_nopiv_gpu            (drivers)
//       hetrf_nopiv_gpu sytrf_nopiv_gpu          (factor on the GPU, then magma_z{he,sy}trs_nopiv_gpu)
//       hetrf_nopiv                              (CPU interface, uploaded, then magma_zhetrs_nopiv_gpu)
//       hetrf hetrf_gpu                          (Bunch-Kaufman, ipiv "as LAPACK"; solved with LAPACK zhetrs)
//       hesv                                     (Bunch-Kaufman driver)
//       aasen0 aasen1                            (magma_zhetrf_aasen, cpu_panel 0 / 1: only the status is reported)
//   kind 0: Hermitian (symmetric for sy*) positive definite, diagonal n + r
//   kind 1: indefinite but diagonally dominant, diagonal +-(n + r) alternating (no pivoting is stable)
//   kind 2: random Hermitian (indefinite, needs pivoting; only for the Bunch-Kaufman routines)
// res = max |A x - b| / (max |A| * max |x| * n); lda = ldb = n + pad.
// CSV: r56,op,n,nrhs,uplo,pad,kind,info,res,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
typedef magmaDoubleComplex Z;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 7) { printf("usage: see source\n"); return 1; }
  std::string op = argv[1]; magma_int_t n = atoi(argv[2]), nrhs = atoi(argv[3]); char up = argv[4][0]; magma_int_t pad = atoi(argv[5]); int kind = atoi(argv[6]);
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  magma_uplo_t uplo = up == 'U' ? MagmaUpper : MagmaLower; bool sym = op.rfind("sy", 0) == 0;
  magma_int_t ld = n + pad; size_t nn = (size_t)ld * n;
  std::vector<Z> A0(nn, MAGMA_Z_ZERO);   // full matrix (both triangles), padding rows zero
  for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i <= j; i++) {
    Z v;
    if (i == j) { double d = kind == 2 ? r1() : (n + r1()) * ((kind == 1 && (j & 1)) ? -1.0 : 1.0); v = sym && kind != 2 ? MAGMA_Z_MAKE(d, r1()) : MAGMA_Z_MAKE(d, 0); }
    else v = MAGMA_Z_MAKE(r1(), r1());
    A0[i + j * ld] = v; A0[j + i * ld] = sym ? v : MAGMA_Z_CONJ(v); }
  std::vector<Z> B0((size_t)ld * nrhs, MAGMA_Z_ZERO); for (magma_int_t j = 0; j < nrhs; j++) for (magma_int_t i = 0; i < n; i++) B0[i + j * ld] = MAGMA_Z_MAKE(r1(), r1());
  std::vector<Z> A = A0, X = B0; std::vector<magma_int_t> ipiv(std::max<magma_int_t>(n, 1)); magma_int_t info = 0, LD = ld, rinfo = 0;
  Z *dA = nullptr, *dB = nullptr; magma_zmalloc(&dA, std::max<size_t>(nn, 1)); magma_zmalloc(&dB, std::max<size_t>((size_t)ld * nrhs, 1));
  auto up_ = [&]() { magma_zsetmatrix(n, n, A.data(), ld, dA, ld, q); magma_zsetmatrix(n, nrhs, X.data(), ld, dB, ld, q); };
  auto dn_ = [&]() { magma_zgetmatrix(n, nrhs, dB, ld, X.data(), ld, q); };
  bool solved = true;
  if (op == "hesv_nopiv_gpu")  { up_(); magma_zhesv_nopiv_gpu(uplo, n, nrhs, dA, ld, dB, ld, &info); dn_(); }
  else if (op == "sysv_nopiv_gpu") { up_(); magma_zsysv_nopiv_gpu(uplo, n, nrhs, dA, ld, dB, ld, &info); dn_(); }
  else if (op == "hetrf_nopiv_gpu") { up_(); magma_zhetrf_nopiv_gpu(uplo, n, dA, ld, &info); if (!info) magma_zhetrs_nopiv_gpu(uplo, n, nrhs, dA, ld, dB, ld, &info); dn_(); }
  else if (op == "sytrf_nopiv_gpu") { up_(); magma_zsytrf_nopiv_gpu(uplo, n, dA, ld, &info); if (!info) magma_zsytrs_nopiv_gpu(uplo, n, nrhs, dA, ld, dB, ld, &info); dn_(); }
  else if (op == "hetrf_nopiv") { magma_zhetrf_nopiv(uplo, n, A.data(), ld, &info); up_(); if (!info) magma_zhetrs_nopiv_gpu(uplo, n, nrhs, dA, ld, dB, ld, &info); dn_(); }
  else if (op == "hetrf") { magma_zhetrf(uplo, n, A.data(), ld, ipiv.data(), &info); if (!info) lapackf77_zhetrs(&up, &n, &nrhs, A.data(), &LD, ipiv.data(), X.data(), &LD, &rinfo); }
  else if (op == "hetrf_gpu") { up_(); magma_zhetrf_gpu(uplo, n, dA, ld, ipiv.data(), &info); magma_zgetmatrix(n, n, dA, ld, A.data(), ld, q);
    if (!info) lapackf77_zhetrs(&up, &n, &nrhs, A.data(), &LD, ipiv.data(), X.data(), &LD, &rinfo); }
  else if (op == "hesv") { magma_zhesv(uplo, n, nrhs, A.data(), ld, ipiv.data(), X.data(), ld, &info); }
  else if (op == "aasen0" || op == "aasen1") { magma_zhetrf_aasen(uplo, op == "aasen1", n, A.data(), ld, ipiv.data(), &info); solved = false; }
  double res = 0;
  if (solved && info == 0 && rinfo == 0) {
    double am = 0, xm = 0; for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i < n; i++) am = std::max(am, MAGMA_Z_ABS(A0[i + j * ld]));
    for (magma_int_t k = 0; k < nrhs; k++) for (magma_int_t i = 0; i < n; i++) xm = std::max(xm, MAGMA_Z_ABS(X[i + k * ld]));
    for (magma_int_t k = 0; k < nrhs; k++) for (magma_int_t i = 0; i < n; i++) { Z s = MAGMA_Z_ZERO;
      for (magma_int_t j = 0; j < n; j++) s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(A0[i + j * ld], X[j + k * ld]));
      double d = MAGMA_Z_ABS(MAGMA_Z_SUB(s, B0[i + k * ld])); if (!(d <= res)) res = d; }
    res /= am * xm * n;
  }
  bool ok = solved ? (info == 0 && rinfo == 0 && res < 1e-13) : info == 0;
  printf("r56,%s,%lld,%lld,%c,%lld,%d,%lld,%.3g,%s\n", op.c_str(), (long long)n, (long long)nrhs, up, (long long)pad, kind, (long long)(info ? info : rinfo), res, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_free(dA); magma_free(dB); magma_queue_destroy(q); magma_finalize(); return 0;
}

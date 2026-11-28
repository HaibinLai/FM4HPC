// Round 64 (roxy): do the magmablas BLAS honour MagmaConjTrans?  (M268: magmablas_zgeam computes ConjTrans as Trans.)
// Every op is compared with reference BLAS on complex data, for trans in {N, T, C}.
//   round64_conj <op> <m> <n> <k> <tA> <tB|side> <uplo>
//   gemm gemm_b gemm_vb : C := alpha op(A) op(B) + beta C            (tA, tB)
//   gemv gemv_b gemv_vb : y := alpha op(A) x + beta y                 (tA)
//   trsm trsm_b trsm_vb : B := alpha op(A)^-1 B  or  B op(A)^-1        (tA, side, uplo; non-unit, well conditioned)
//   trmm trmm_b         : B := alpha op(A) B  or  B op(A)
//   trsv                : x := op(A)^-1 x
//   _b: batch of 3 identical-shape problems; _vb: the same through the vbatched interface.
// CSV: r64,op,m,n,k,tA,o2,uplo,err,verdict
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
static magma_queue_t q;
static magma_trans_t T_(char c) { return c == 'N' ? MagmaNoTrans : c == 'T' ? MagmaTrans : Magma_ConjTrans; }
static double dif(const std::vector<Z>& X, const std::vector<Z>& R) { double e = 0, a = 0; for (size_t i = 0; i < X.size(); i++) { double d = MAGMA_Z_ABS(MAGMA_Z_SUB(X[i], R[i])); if (!(d <= e)) e = d; a = std::max(a, MAGMA_Z_ABS(R[i])); } return a > 0 ? e / a : e; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 8) { printf("usage: see source\n"); return 1; }
  std::string op = argv[1]; magma_int_t m = atoi(argv[2]), n = atoi(argv[3]), k = atoi(argv[4]); char tA = argv[5][0], o2 = argv[6][0], up = argv[7][0];
  magma_init(); magma_queue_create(0, &q);
  const int BC = 3; bool bat = op.size() > 2 && (op.substr(op.size() - 2) == "_b" || op.substr(op.size() - 3) == "_vb"), vb = op.size() > 3 && op.substr(op.size() - 3) == "_vb";
  std::string base = op.substr(0, op.find('_')); int nb = bat ? BC : 1;
  Z al = MAGMA_Z_MAKE(0.7, -0.3), be = MAGMA_Z_MAKE(-0.4, 0.2);
  // shapes
  magma_int_t ra, ca, rb, cb, rc, cc;   // A, B(or x), C(or y)
  bool tri = base == "trsm" || base == "trmm" || base == "trsv";
  if (base == "gemm") { ra = tA == 'N' ? m : k; ca = tA == 'N' ? k : m; rb = o2 == 'N' ? k : n; cb = o2 == 'N' ? n : k; rc = m; cc = n; }
  else if (base == "gemv") { ra = m; ca = n; rb = tA == 'N' ? n : m; cb = 1; rc = tA == 'N' ? m : n; cc = 1; }
  else if (base == "trsv") { ra = ca = n; rb = n; cb = 1; rc = cc = 0; }
  else { magma_int_t na = o2 == 'L' ? m : n; ra = ca = na; rb = m; cb = n; rc = cc = 0; }
  magma_int_t lda = std::max<magma_int_t>(ra, 1), ldb = std::max<magma_int_t>(rb, 1), ldc = std::max<magma_int_t>(rc, 1);
  std::vector<std::vector<Z>> A(nb), B(nb), C(nb), R(nb);
  for (int b = 0; b < nb; b++) {
    A[b].resize((size_t)lda * ca); B[b].resize((size_t)ldb * cb); C[b].resize((size_t)ldc * std::max<magma_int_t>(cc, 1));
    for (auto& v : A[b]) v = MAGMA_Z_MAKE(r1(), r1()); for (auto& v : B[b]) v = MAGMA_Z_MAKE(r1(), r1()); for (auto& v : C[b]) v = MAGMA_Z_MAKE(r1(), r1());
    if (tri) for (magma_int_t j = 0; j < ca; j++) for (magma_int_t i = 0; i < ra; i++) { Z& a = A[b][i + j * lda]; if (i == j) a = MAGMA_Z_MAKE(2 + r1(), r1()); else a = MAGMA_Z_MAKE(0.2 * r1(), 0.2 * r1()); }
    magma_int_t one = 1; char ul = up, sd = o2, t1 = tA, t2 = o2;
    if (base == "gemm") { R[b] = C[b]; blasf77_zgemm(&t1, &t2, &m, &n, &k, &al, A[b].data(), &lda, B[b].data(), &ldb, &be, R[b].data(), &ldc); }
    else if (base == "gemv") { R[b] = C[b]; blasf77_zgemv(&t1, &m, &n, &al, A[b].data(), &lda, B[b].data(), &one, &be, R[b].data(), &one); }
    else if (base == "trsm") { R[b] = B[b]; blasf77_ztrsm(&sd, &ul, &t1, "N", &m, &n, &al, A[b].data(), &lda, R[b].data(), &ldb); }
    else if (base == "trmm") { R[b] = B[b]; blasf77_ztrmm(&sd, &ul, &t1, "N", &m, &n, &al, A[b].data(), &lda, R[b].data(), &ldb); }
    else if (base == "trsv") { R[b] = B[b]; blasf77_ztrsv(&ul, &t1, "N", &n, A[b].data(), &lda, R[b].data(), &one); }
  }
  std::vector<Z*> hA(nb), hB(nb), hC(nb);
  for (int b = 0; b < nb; b++) { magma_zmalloc(&hA[b], A[b].size()); magma_zmalloc(&hB[b], B[b].size()); magma_zmalloc(&hC[b], C[b].size());
    magma_zsetvector((magma_int_t)A[b].size(), A[b].data(), 1, hA[b], 1, q); magma_zsetvector((magma_int_t)B[b].size(), B[b].data(), 1, hB[b], 1, q); magma_zsetvector((magma_int_t)C[b].size(), C[b].data(), 1, hC[b], 1, q); }
  Z **dA = NULL, **dB = NULL, **dC = NULL; magma_int_t *dm = NULL, *dn = NULL, *dk = NULL, *dla = NULL, *dlb = NULL, *dlc = NULL, *dinc = NULL;
  if (bat) { magma_malloc((void**)&dA, nb * sizeof(Z*)); magma_malloc((void**)&dB, nb * sizeof(Z*)); magma_malloc((void**)&dC, nb * sizeof(Z*));
    magma_setvector(nb, sizeof(Z*), hA.data(), 1, dA, 1, q); magma_setvector(nb, sizeof(Z*), hB.data(), 1, dB, 1, q); magma_setvector(nb, sizeof(Z*), hC.data(), 1, dC, 1, q); }
  if (vb) { auto mk = [&](magma_int_t v) { magma_int_t* d; magma_imalloc(&d, nb + 1); std::vector<magma_int_t> h(nb + 1, v); magma_isetvector(nb + 1, h.data(), 1, d, 1, q); return d; };
    dm = mk(m); dn = mk(n); dk = mk(k); dla = mk(lda); dlb = mk(ldb); dlc = mk(ldc); dinc = mk(1); }
  magma_trans_t ta = T_(tA), tb = T_(o2); magma_side_t side = o2 == 'R' ? MagmaRight : MagmaLeft; magma_uplo_t uplo = up == 'U' ? MagmaUpper : MagmaLower;
  if (op == "gemm") magmablas_zgemm(ta, tb, m, n, k, al, hA[0], lda, hB[0], ldb, be, hC[0], ldc, q);
  if (op == "gemm_b") magmablas_zgemm_batched(ta, tb, m, n, k, al, dA, lda, dB, ldb, be, dC, ldc, nb, q);
  if (op == "gemm_vb") magmablas_zgemm_vbatched(ta, tb, dm, dn, dk, al, dA, dla, dB, dlb, be, dC, dlc, nb, q);
  if (op == "gemv") magmablas_zgemv(ta, m, n, al, hA[0], lda, hB[0], 1, be, hC[0], 1, q);
  if (op == "gemv_b") magmablas_zgemv_batched(ta, m, n, al, dA, lda, dB, 1, be, dC, 1, nb, q);
  if (op == "gemv_vb") magmablas_zgemv_vbatched(ta, dm, dn, al, dA, dla, dB, dinc, be, dC, dinc, nb, q);
  if (op == "trsm") magmablas_ztrsm(side, uplo, ta, MagmaNonUnit, m, n, al, hA[0], lda, hB[0], ldb, q);
  if (op == "trsm_b") magmablas_ztrsm_batched(side, uplo, ta, MagmaNonUnit, m, n, al, dA, lda, dB, ldb, nb, q);
  if (op == "trsm_vb") magmablas_ztrsm_vbatched(side, uplo, ta, MagmaNonUnit, dm, dn, al, dA, dla, dB, dlb, nb, q);
  if (op == "trmm") magmablas_ztrmm(side, uplo, ta, MagmaNonUnit, m, n, al, hA[0], lda, hB[0], ldb, q);
  if (op == "trmm_b") magmablas_ztrmm_batched(side, uplo, ta, MagmaNonUnit, m, n, al, dA, lda, dB, ldb, nb, q);
  if (op == "trsv") magmablas_ztrsv(uplo, ta, MagmaNonUnit, n, hA[0], lda, hB[0], 1, q);
  magma_queue_sync(q);
  double err = 0; bool outC = base == "gemm" || base == "gemv";
  for (int b = 0; b < nb; b++) { std::vector<Z> X(outC ? C[b].size() : B[b].size()); magma_zgetvector((magma_int_t)X.size(), outC ? hC[b] : hB[b], 1, X.data(), 1, q); err = std::max(err, dif(X, R[b])); }
  bool ok = err < 1e-11;
  printf("r64,%s,%lld,%lld,%lld,%c,%c,%c,%.3g,%s\n", op.c_str(), (long long)m, (long long)n, (long long)k, tA, o2, up, err, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

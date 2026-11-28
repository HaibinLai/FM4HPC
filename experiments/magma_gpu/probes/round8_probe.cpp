// Eighth GPU round: the remaining code-reading findings of the second audit (HANDOFF §29.8).  GPU 0 only.
// One mode per process; CPU-side heap errors are made visible with MALLOC_CHECK_=3 (set by the runner).
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z;
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static std::vector<std::vector<magma_index_t>> KR, KC; static std::vector<std::vector<Z>> KV;
static magma_z_matrix csr(int n, int m, std::vector<magma_index_t> row, std::vector<magma_index_t> col, std::vector<Z> val) {
  KR.push_back(row); KC.push_back(col); KV.push_back(val); magma_z_matrix h = {Magma_CSR};
  magma_zcsrset(n, m, KR.back().data(), KC.back().data(), KV.back().data(), &h, Q); return h;
}
static magma_z_matrix vec(magma_location_t loc, int n, double v) { magma_z_matrix x = {Magma_CSR}; magma_zvinit(&x, loc, n, 1, C(v), Q); return x; }
static void show(const char* nm, magma_z_matrix d) { magma_z_matrix h = {Magma_CSR}; magma_zmtransfer(d, &h, d.memory_location, Magma_CPU, Q);
  printf("  %s =", nm); for (int i = 0; i < h.num_rows * h.num_cols && i < 8; i++) printf(" %g", MAGMA_Z_REAL(h.val[i])); printf("\n"); }

static void spmv_cpu() {
  magma_z_matrix A = csr(2, 2, {0, 2, 3}, {0, 1, 1}, {C(1), C(2), C(3)});        // [[1,2],[0,3]]
  magma_z_matrix x = vec(Magma_CPU, 2, 1.0), y = vec(Magma_CPU, 2, 7.0);
  Z* xv = x.val; Z* yv = y.val;
  magma_int_t r = magma_z_spmv(C(1), A, x, C(0), y, Q);
  printf("magma_z_spmv with A, x, y on the CPU: ret=%lld; caller's y.val pointer unchanged=%d, y.val[0..1] (expected 3 3) = %g %g\n",
         (long long)r, (int)(y.val == yv), MAGMA_Z_REAL(y.val[0]), MAGMA_Z_REAL(y.val[1]));
  printf("now freeing the caller's x (its array was already freed inside magma_z_spmv) ...\n"); fflush(stdout);
  (void)xv; magma_zmfree(&x, Q);
  printf("returned from magma_zmfree(&x)\n");
}
static void csrd() {
  magma_z_matrix A = csr(2, 2, {0, 1, 2}, {0, 0}, {C(5), C(6)});                  // row 1 = {(1,0)}: no diagonal
  magma_z_matrix B = {Magma_CSR}; magma_int_t r = magma_zmconvert(A, &B, Magma_CSR, Magma_CSRD, Q);
  printf("CSR -> CSRD, row 1 without a diagonal: ret=%lld nnz=%lld  B.col[0..1] = %d %d  (B.col[2] was written: past the nnz = 2 array)\n",
         (long long)r, (long long)B.nnz, (int)B.col[0], (int)B.col[1]);
  magma_z_matrix E = {Magma_CSR}; r = magma_zmconvert(A, &E, Magma_CSR, Magma_ELLD, Q);
  printf("CSR -> ELLD, same matrix: ret=%lld  max_nnz_row=%lld  E.col =", (long long)r, (long long)E.max_nnz_row);
  for (int i = 0; i < E.num_rows * E.max_nnz_row && i < 8; i++) printf(" %d", (int)E.col[i]); printf("\n");
  printf("freeing B and E ...\n"); fflush(stdout); magma_zmfree(&B, Q); magma_zmfree(&E, Q); printf("freed\n");
}
static void aux() {
  { magma_z_matrix A = csr(2, 2, {0, 1, 2}, {0, 1}, {C(1), C(1)}), B = csr(2, 2, {0, 1, 1}, {1}, {C(1)});   // I2 vs {(0,1)}
    real_Double_t res = -1; magma_int_t r = magma_zmdiff(A, B, &res, Q);
    printf("zmdiff(I2, {(0,1)=1}): ret=%lld res=%g (Frobenius distance sqrt(3) = 1.732)\n", (long long)r, res);
    magma_z_matrix Ai = csr(1, 1, {0, 1}, {0}, {C(1, 5)}), Bi = csr(1, 1, {0, 1}, {0}, {C(1, -5)});
    res = -1; r = magma_zmdiff(Ai, Bi, &res, Q); printf("zmdiff([1+5i], [1-5i]): ret=%lld res=%g (distance 10)\n", (long long)r, res); }
  { magma_z_matrix A = csr(2, 2, {0, 1, 2}, {0, 1}, {C(1), C(1)}), S = A;
    double nrm = 5.0; magma_int_t r = magma_zmfrobenius(A, A, S, &nrm, Q);
    printf("zmfrobenius(A, A) with *norm = 5 on entry: ret=%lld norm=%g (expected 0)\n", (long long)r, nrm);
    magma_z_matrix B3 = csr(3, 3, {0, 1, 2, 3}, {0, 1, 2}, {C(1), C(1), C(1)}); nrm = -1; r = magma_zmfrobenius(A, B3, S, &nrm, Q);
    printf("zmfrobenius(2x2, 3x3): ret=%lld norm=%g (nothing computed)\n", (long long)r, nrm); }
  { magma_z_matrix A = csr(2, 2, {0, 1, 2}, {0, 0}, {C(5), C(6)}); magma_z_matrix D = {Magma_CSR}; magma_zmtransfer(A, &D, Magma_CPU, Magma_DEV, Q);
    magma_int_t r = magma_zmdiagadd(&D, C(100), Q); show("zmdiagadd(+100) on [[5,0],[6,0]] (row 1 has no stored diagonal): values", D);
    printf("  ret=%lld (A + 100 I has (1,1) = 100)\n", (long long)r); }
  { magma_z_matrix A = csr(1, 3, {0, 3}, {0, 1, 2}, {C(1), C(2), C(3)}), T = {Magma_CSR};
    printf("zmtranspose of a 1x3 CPU CSR ...\n"); fflush(stdout);
    magma_int_t r = magma_zmtranspose(A, &T, Q);
    printf("  ret=%lld  result dims %lld x %lld (expected 3 x 1)\n", (long long)r, (long long)T.num_rows, (long long)T.num_cols); }
}
static void mtx() {
  const char* f = "bad_index.mtx"; FILE* fp = fopen(f, "w");
  fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n3 3 3\n1 1 1.0\n2 2 1.0\n9 3 1.0\n"); fclose(fp);
  magma_z_matrix A = {Magma_CSR}; printf("reading a 3x3 .mtx whose third entry has row index 9 ...\n"); fflush(stdout);
  magma_int_t r = magma_z_csr_mtx(&A, f, Q);
  printf("  ret=%lld num_rows=%lld nnz=%lld\n", (long long)r, (long long)A.num_rows, (long long)A.nnz);
}
static void applyprec() {
  int n = 8; std::vector<magma_index_t> row(n + 1), col; std::vector<Z> val;
  for (int i = 0; i < n; i++) { row[i] = (magma_index_t)col.size(); if (i) { col.push_back(i - 1); val.push_back(C(-1)); } col.push_back(i); val.push_back(C(4)); if (i + 1 < n) { col.push_back(i + 1); val.push_back(C(-1)); } }
  row[n] = (magma_index_t)col.size(); magma_z_matrix h = csr(n, n, row, col, val), dA = {Magma_CSR}; magma_zmtransfer(h, &dA, Magma_CPU, Magma_DEV, Q);
  magma_z_matrix b = vec(Magma_DEV, n, 1.0);
  for (magma_solver_type pt : {Magma_ILU, Magma_ICC}) {
    magma_z_solver_par sp; magma_z_preconditioner pp; memset(&sp, 0, sizeof sp); memset(&pp, 0, sizeof pp);
    sp.solver = Magma_PCG; pp.solver = pt; pp.trisolver = Magma_CUSOLVE; magma_zsolverinfo_init(&sp, &pp, Q);
    magma_int_t s = magma_z_precondsetup(dA, b, &sp, &pp, Q); magma_z_matrix x = vec(Magma_DEV, n, 7.0);
    magma_int_t r = magma_z_applyprecond(dA, b, &x, &pp, Q);
    printf("magma_z_applyprecond with %s (setup=%lld): ret=%lld\n", pt == Magma_ILU ? "ILU" : "ICC", (long long)s, (long long)r); show("x (initialised to 7; M^-1 b has entries ~0.36)", x);
    magma_z_matrix x2 = vec(Magma_DEV, n, 7.0); r = magma_z_applyprecond_right(MagmaConjTrans, dA, b, &x2, &pp, Q);
    printf("magma_z_applyprecond_right(MagmaConjTrans) with %s: ret=%lld\n", pt == Magma_ILU ? "ILU" : "ICC", (long long)r); show("x", x2);
  }
}
static void dense_args() {
  magma_int_t info = 0; double* d; magma_dmalloc(&d, 64); std::vector<double> hw(1000), tau(4);
  magma_dgeqrs3_gpu(4, 4, 1, d, 4, tau.data(), d, d, 2, hw.data(), 1000, &info);
  printf("dgeqrs3_gpu with lddb = 2 < m = 4 (argument 9): info=%lld\n", (long long)info);
  magma_dgeqrs_gpu(4, 4, 1, d, 4, tau.data(), d, d, 2, hw.data(), 1000, &info);
  printf("dgeqrs_gpu  with lddb = 2 < m = 4 (argument 9): info=%lld\n", (long long)info);
}
static void hegvx_leak() {
  int n = 512; std::vector<Z> A((size_t)n * n, C(0)), B((size_t)n * n, C(0)), Zm((size_t)n * n); std::vector<double> w(n), rw(7 * n);
  std::vector<magma_int_t> iw(5 * n), ifail(n);
  for (int i = 0; i < n; i++) { A[i + (size_t)i * n] = C(i + 1); B[i + (size_t)i * n] = C(-1); }   // B not positive definite
  magma_int_t m, info, lwork = -1; Z q; magma_zhegvx(1, MagmaVec, MagmaRangeAll, MagmaLower, n, A.data(), n, B.data(), n, 0, 0, 0, 0, 0, &m, w.data(), Zm.data(), n, &q, lwork, rw.data(), iw.data(), ifail.data(), &info);
  lwork = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> wk(lwork);
  size_t f0, f1, tot; cudaMemGetInfo(&f0, &tot);
  for (int k = 0; k < 20; k++) { std::vector<Z> A1 = A, B1 = B;
    magma_zhegvx(1, MagmaVec, MagmaRangeAll, MagmaLower, n, A1.data(), n, B1.data(), n, 0, 0, 0, 0, 0, &m, w.data(), Zm.data(), n, wk.data(), lwork, rw.data(), iw.data(), ifail.data(), &info); }
  cudaMemGetInfo(&f1, &tot);
  printf("zhegvx n=512, B = -I (not positive definite), 20 calls: info=%lld (n+1 = 513); device free memory dropped %.2f MB (3 n^2 complex = %.2f MB per call)\n",
         (long long)info, (f0 - f1) / 1e6, 3.0 * n * n * 16 / 1e6);
}
static void gels_cpu() {
  int m = 10, n = 10, nrhs = getenv("NRHS") ? atoi(getenv("NRHS")) : 4096; std::vector<double> A((size_t)m * n), B((size_t)m * nrhs, 1.0);
  for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) A[i + j * m] = (i == j ? 4.0 : 0.1 * ((i + 2 * j) % 5));
  magma_int_t info = -999, lwork = -1; double q; magma_dgels(MagmaNoTrans, m, n, nrhs, A.data(), m, B.data(), m, &q, lwork, &info);
  lwork = (magma_int_t)q; printf("magma_dgels workspace query (m = n = 10, nrhs = %d): lwork=%lld, nb=%lld\n", nrhs, (long long)lwork, (long long)magma_get_dgeqrf_nb(m, n));
  std::vector<double> hw(lwork); info = -999; fflush(stdout);
  magma_dgels(MagmaNoTrans, m, n, nrhs, A.data(), m, B.data(), m, hw.data(), lwork, &info);
  printf("magma_dgels returned info=%lld; x[0] of column 0 = %g\n", (long long)info, B[0]);
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : "";
  magma_init(); magma_queue_create(0, &Q);
  if (m == "spmv_cpu") spmv_cpu(); else if (m == "csrd") csrd(); else if (m == "aux") aux(); else if (m == "mtx") mtx();
  else if (m == "applyprec") applyprec(); else if (m == "dense_args") dense_args(); else if (m == "hegvx_leak") hegvx_leak(); else if (m == "gels_cpu") gels_cpu();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

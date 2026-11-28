// Thirteenth GPU round: the kernel items left after round 12 (F7, F12–F15) and two sparse items backed only by reading
// (HANDOFF §29.13).  GPU 0 only; one mode per process.
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z;
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 4321u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static magma_int_t lapack_getrf(int m, int n, std::vector<Z> A) { magma_int_t mm = m, nn = n, info; std::vector<magma_int_t> ip(std::min(m, n)); lapackf77_zgetrf(&mm, &nn, A.data(), &mm, ip.data(), &info); return info; }
static std::vector<Z> randm(int m, int n) { std::vector<Z> A((size_t)m * n); for (auto& a : A) a = C(rnd(), rnd()); return A; }

static void f7() {   // native recursive LU: the second half lacks the n1 offset
  int n = 128; std::vector<Z> A((size_t)n * n, C(0)); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = C(1); A[49 + 49 * n] = C(0);
  Z* d; magma_zmalloc(&d, (size_t)n * n); std::vector<magma_int_t> ip(n);
  magma_queue_t qs[2]; magma_event_t ev[2]; magma_queue_create(0, &qs[0]); magma_queue_create(0, &qs[1]); magma_event_create(&ev[0]); magma_event_create(&ev[1]);
  for (int recnb : {32, 64}) {
    magma_zsetmatrix(n, n, A.data(), n, d, n, Q); magma_int_t info = -999, lh = -1, ld = -1;
    magma_zgetrf_expert_gpu_work(n, n, NULL, n, NULL, &info, MagmaNative, 128, recnb, NULL, &lh, NULL, &ld, ev, qs);
    void *hw = NULL, *dw = NULL; if (lh > 0) magma_malloc_pinned(&hw, lh); if (ld > 0) magma_malloc(&dw, ld); info = -999;
    magma_zgetrf_expert_gpu_work(n, n, d, n, ip.data(), &info, MagmaNative, 128, recnb, hw, &lh, dw, &ld, ev, qs);
    magma_queue_sync(qs[0]); magma_queue_sync(qs[1]);
    printf("F7 zgetrf_expert_gpu_work(MagmaNative, nb=128, recnb=%d) 128x128 I with A(50,50)=0: info=%lld | LAPACK %lld\n", recnb, (long long)info, (long long)lapack_getrf(n, n, A));
    if (hw) magma_free_pinned(hw); if (dw) magma_free(dw);
  }
}
static void f12() {  // repeated native LU with two zero pivots: does the first one always win?
  int m = 512, n = 32; auto A = randm(m, n); for (int i = 0; i < m; i++) { A[i + (size_t)2 * m] = C(0); A[i + (size_t)19 * m] = C(0); }
  Z* d; magma_zmalloc(&d, (size_t)m * n); std::vector<magma_int_t> ip(n); std::map<long, int> hist;
  for (int r = 0; r < 300; r++) { magma_zsetmatrix(m, n, A.data(), m, d, m, Q); magma_int_t info = -999; magma_zgetrf_native(m, n, d, m, ip.data(), &info); hist[(long)info]++; }
  printf("F12 zgetrf_native 512x32, zero columns 3 and 20, 300 runs: info histogram:"); for (auto& h : hist) printf("  %ld x%d", h.first, h.second);
  printf("  | LAPACK %lld\n", (long long)lapack_getrf(m, n, A));
}
static void batched_getrf(const char* tag, int m, int n, std::vector<Z> A, bool fused) {
  Z* d; magma_zmalloc(&d, (size_t)m * n); magma_zsetmatrix(m, n, A.data(), m, d, m, Q); Z** dA; magma_malloc((void**)&dA, sizeof(Z*)); magma_setvector(1, sizeof(Z*), &d, 1, dA, 1, Q);
  magma_int_t* dp; magma_imalloc(&dp, std::max(m, n)); magma_int_t** dpa; magma_malloc((void**)&dpa, sizeof(magma_int_t*)); magma_setvector(1, sizeof(magma_int_t*), &dp, 1, dpa, 1, Q);
  magma_int_t* di; magma_imalloc(&di, 1); magma_int_t z = -999; magma_setvector(1, sizeof(magma_int_t), &z, 1, di, 1, Q);
  magma_int_t r = fused ? magma_zgetf2_fused_batched(m, n, dA, 0, 0, m, dpa, di, 1, Q) : magma_zgetrf_batched(m, n, dA, m, dpa, di, 1, Q);
  magma_int_t info; magma_getvector(1, sizeof(magma_int_t), di, 1, &info, 1, Q); std::vector<magma_int_t> ip(std::min(m, n)); magma_igetvector(std::min(m, n), dp, 1, ip.data(), 1, Q);
  int maxp = 0; for (auto p : ip) maxp = std::max(maxp, (int)p);
  printf("%s: ret=%lld info=%lld max ipiv=%d (m=%d) | LAPACK %lld\n", tag, (long long)r, (long long)info, maxp, m, (long long)lapack_getrf(m, n, A));
  magma_free(d); magma_free(dA); magma_free(dp); magma_free(dpa); magma_free(di);
}
static void f13() { batched_getrf("F13 zgetf2_fused_batched m=4 n=8 random", 4, 8, randm(4, 8), true); batched_getrf("    zgetf2_fused_batched m=8 n=8 random (control)", 8, 8, randm(8, 8), true); }
static void f14() {
  for (int m : {4, 16, 64, 512}) { int n = std::min(m, 8); auto A = randm(m, n); for (int i = 0; i < m; i++) A[i] = C(0); A[1] = C(NAN); A[m - 1] = C(5);
    char t[96]; snprintf(t, 96, "F14 zgetrf_batched %dx%d, column 1 = (0, NaN, 0, ..., 5)", m, n); batched_getrf(t, m, n, A, false); } }
static void f15() { Z* d; magma_zmalloc(&d, 16); magma_int_t info = 0;
  magmablas_zlascl(MagmaFull, 0, 0, 1.0, 2.0, 4, -1, d, 4, Q, &info); printf("F15 magmablas_zlascl n = -1: info=%lld (LAPACK zlascl: -7)\n", (long long)info);
  magmablas_zlascl(MagmaFull, 0, 0, 1.0, 2.0, 4, 4, d, 2, Q, &info); printf("    magmablas_zlascl ldda = 2 < m = 4: info=%lld (LAPACK zlascl: -9)\n", (long long)info); }
static void sparse_rhs() {
  std::vector<magma_index_t> row = {0, 1, 3}, col = {1, 0, 1}; std::vector<Z> val = {C(1), C(1), C(2)};   // [[0,1],[1,2]]: zero diagonal in row 0
  magma_z_matrix h = {Magma_CSR}, d = {Magma_CSR}; magma_zcsrset(2, 2, row.data(), col.data(), val.data(), &h, Q); magma_zmtransfer(h, &d, Magma_CPU, Magma_DEV, Q);
  magma_z_matrix b = {Magma_CSR}, sf = {Magma_CSR}; magma_zvinit(&b, Magma_CPU, 2, 1, C(1), Q);
  magma_int_t r = magma_zmscale_matrix_rhs(&d, &b, &sf, Magma_UNITDIAG, Q);
  magma_z_matrix h2 = {Magma_CSR}; magma_zmtransfer(d, &h2, d.memory_location, Magma_CPU, Q);
  printf("zmscale_matrix_rhs(UNITDIAG) on [[0,1],[1,2]], b = (1,1): ret=%lld  b = (%g, %g)  A values:", (long long)r, MAGMA_Z_REAL(b.val[0]), MAGMA_Z_REAL(b.val[1]));
  for (int i = 0; i < h2.nnz; i++) printf(" %g", MAGMA_Z_REAL(h2.val[i])); printf("\n");
  magma_z_matrix L = h; L.storage_type = Magma_CSRLIST; magma_z_matrix out = {Magma_CSR};
  r = magma_zmtransfer(L, &out, Magma_CPU, Magma_DEV, Q);
  printf("zmtransfer of a matrix tagged CSRLIST (unhandled storage) CPU->DEV: ret=%lld  out: num_rows=%lld nnz=%lld dval=%p\n", (long long)r, (long long)out.num_rows, (long long)out.nnz, (void*)out.dval);
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; magma_init(); magma_queue_create(0, &Q);
  if (m == "f7") f7(); else if (m == "f12") f12(); else if (m == "f13") f13(); else if (m == "f14") f14(); else if (m == "f15") f15(); else if (m == "sparse") sparse_rhs();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

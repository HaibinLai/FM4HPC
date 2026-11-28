// Sixteenth GPU round: the remaining candidates of the sparse control/blas per-file sweep, and zmdotc with a device
// output buffer (HANDOFF §29.16).  GPU 0 only; one mode per process; heap errors via MALLOC_CHECK_=3.
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
static Z* dev(const std::vector<Z>& v) { Z* d; magma_zmalloc(&d, std::max<size_t>(v.size(), 1)); magma_zsetvector(v.size(), v.data(), 1, d, 1, Q); return d; }
static std::vector<Z> host(const Z* d, int n) { std::vector<Z> v(n); magma_zgetvector(n, d, 1, v.data(), 1, Q); return v; }
static void pr(const char* t, const std::vector<Z>& v) { printf("  %s =", t); for (auto& z : v) printf(" (%g%+gi)", MAGMA_Z_REAL(z), MAGMA_Z_IMAG(z)); printf("\n"); }
static std::vector<std::vector<magma_index_t>> KR, KC; static std::vector<std::vector<Z>> KV;
static magma_z_matrix csr(int m, int n, std::vector<magma_index_t> r, std::vector<magma_index_t> c, std::vector<Z> v) { KR.push_back(r); KC.push_back(c); KV.push_back(v);
  magma_z_matrix h = {Magma_CSR}; magma_zcsrset(m, n, KR.back().data(), KC.back().data(), KV.back().data(), &h, Q); return h; }
static magma_z_matrix todev(magma_z_matrix h) { magma_z_matrix d = {Magma_CSR}; magma_zmtransfer(h, &d, Magma_CPU, Magma_DEV, Q); return d; }
static void showcsr(const char* t, magma_z_matrix A) { magma_z_matrix h = {Magma_CSR}; magma_zmtransfer(A, &h, A.memory_location, Magma_CPU, Q);
  printf("  %s: %lld x %lld nnz %lld:", t, (long long)h.num_rows, (long long)h.num_cols, (long long)h.nnz);
  for (int i = 0; i < h.num_rows; i++) for (int j = h.row[i]; j < h.row[i + 1]; j++) printf(" (%d,%d)=%g%+gi", i, (int)h.col[j], MAGMA_Z_REAL(h.val[j]), MAGMA_Z_IMAG(h.val[j])); printf("\n"); }

static void mdotc() { auto dv = dev({C(0, 1), C(0, 1)}); auto dr = dev({C(0, 1)}); Z *d1, *d2; magma_zmalloc(&d1, 1 << 14); magma_zmalloc(&d2, 1 << 14);
  auto ds = dev({C(-99), C(-99)}); magma_zmdotc(1, 1, dv, dr, d1, d2, ds, Q); magma_queue_sync(Q); printf("#22/27 zmdotc n=1 k=1, v=(i), r=(i): expected conj(i)*i = 1\n"); pr("skp", host(ds, 1));
  auto ds2 = dev({C(-99), C(-99)}); magma_zmdotc(1, 2, dv, dr, d1, d2, ds2, Q); magma_queue_sync(Q); printf("       k=2, v0=v1=(i): expected (1, 1)\n"); pr("skp", host(ds2, 2)); }
static void csr_sort() { magma_z_matrix A = csr(1, 2, {0, 2}, {1, 0}, {C(10), C(20)}); magma_int_t r = magma_zcsr_sort(&A, Q);
  printf("#2a zcsr_sort, row 0 = {(col 1, 10), (col 0, 20)}: ret=%lld; expected A(0,0)=20, A(0,1)=10\n", (long long)r); showcsr("A", A); }
static void csrcoo_t() { magma_z_matrix A = csr(1, 3, {0, 1}, {2}, {C(5)}), B = {Magma_CSR}, T = {Magma_CSR}; magma_zmconvert(A, &B, Magma_CSR, Magma_CSRCOO, Q);
  printf("#2b zcsrcoo_transpose of a 1x3 CSRCOO with an entry in column 2 ...\n"); fflush(stdout); magma_int_t r = magma_zcsrcoo_transpose(B, &T, Q);
  printf("  ret=%lld dims %lld x %lld (expected 3 x 1); freeing ...\n", (long long)r, (long long)T.num_rows, (long long)T.num_cols); fflush(stdout); magma_zmfree(&T, Q); printf("  freed\n"); }
static void mbdiagdom() { magma_z_matrix M = csr(2, 2, {0, 1, 2}, {0, 1}, {C(2), C(3)}), bs = {Magma_CSR}; std::vector<Z> b = {C(3)}; magma_zvset(1, 1, b.data(), &bs, Q);
  bs.num_rows = 1; double mn, mx, av; printf("#6 zmbdiagdom 2x2 with blocksizes = [3] (sum > n) ...\n"); fflush(stdout); magma_int_t r = magma_zmbdiagdom(M, bs, &mn, &mx, &av, Q);
  printf("  ret=%lld min=%g max=%g avg=%g\n", (long long)r, mn, mx, av); }
static void bitonic() { std::vector<Z> s = {C(5), C(3), C(1), C(6), C(2), C(4)}; magma_int_t r = magma_zbitonic_sort(0, 6, s.data(), 1, Q);
  printf("#7 zbitonic_sort length 6 (not a power of two): ret=%lld; result:", (long long)r); for (auto& z : s) printf(" %g", MAGMA_Z_REAL(z)); printf("\n"); }
static void vget_dev() { magma_z_matrix v = {Magma_CSR}; magma_zvinit(&v, Magma_CPU, 4, 1, C(1.5), Q); magma_int_t m, n; Z* p = NULL;
  magma_int_t r = magma_zvget_dev(v, &m, &n, &p, Q); Z h[4]; cudaError_t e = cudaMemcpy(h, p, 4 * sizeof(Z), cudaMemcpyDeviceToHost);
  cudaPointerAttributes at; cudaError_t e2 = cudaPointerGetAttributes(&at, p);
  printf("#9 zvget_dev of a CPU vector: ret=%lld m=%lld n=%lld; cudaMemcpy from the returned pointer: %s (value %g); pointer attributes: %s type=%d\n",
         (long long)r, (long long)m, (long long)n, cudaGetErrorString(e), MAGMA_Z_REAL(h[0]), cudaGetErrorString(e2), (int)at.type); cudaGetLastError(); }
static void cuspaxpy() { magma_z_matrix A = todev(csr(2, 2, {0, 1, 2}, {0, 1}, {C(1), C(2)})), B = todev(csr(2, 2, {0, 1, 2}, {1, 0}, {C(3), C(4)})), AB = {Magma_CSR};
  Z one = C(1); magma_int_t r = magma_zcuspaxpy(&one, A, &one, B, &AB, Q); cudaError_t e = cudaDeviceSynchronize();
  printf("#10 zcuspaxpy A + B, A = diag(1,2), B = antidiag(3,4): ret=%lld sync=%s; expected [[1,3],[4,2]]\n", (long long)r, cudaGetErrorString(e)); if (e == cudaSuccess && AB.dval) showcsr("AB", AB); }
static void cuspmm() { magma_z_matrix A = todev(csr(2, 2, {0, 1, 2}, {0, 1}, {C(1), C(2)})), B = todev(csr(2, 2, {0, 1, 2}, {1, 0}, {C(3), C(4)})), AB = {Magma_CSR};
  magma_int_t r = magma_zcuspmm(A, B, &AB, Q); cudaError_t e = cudaDeviceSynchronize();
  printf("#11 zcuspmm A*B: ret=%lld sync=%s; expected [[0,3],[8,0]]\n", (long long)r, cudaGetErrorString(e)); if (e == cudaSuccess && AB.dval) showcsr("AB", AB);
  magma_z_matrix hA = csr(1, 1, {0, 1}, {0}, {C(2)}), hB = csr(1, 1, {0, 1}, {0}, {C(3)}), hC = {Magma_CSR}; printf("    zcuspmm on 1x1 CPU matrices ...\n"); fflush(stdout);
  r = magma_zcuspmm(hA, hB, &hC, Q); printf("    ret=%lld\n", (long long)r); }
static void csr_sort_gpu() { magma_z_matrix A = csr(1, 2, {0, 2}, {1, 0}, {C(10), C(20)}); Z* before = A.val; magma_int_t r = magma_zcsr_sort_gpu(&A, Q);
  cudaPointerAttributes at; cudaPointerGetAttributes(&at, A.val); cudaGetLastError();
  printf("#12 zcsr_sort_gpu on a CPU CSR: ret=%lld memory_location=%d, val pointer changed=%d, new val pointer type=%d (2 = device)\n", (long long)r, (int)A.memory_location, (int)(A.val != before), (int)at.type); }
static void ge3pt() { std::vector<Z> xs = {C(1000), C(1)}; Z* dx = dev(xs); auto dy = dev({C(0)});
  magma_zge3pt(1, 1, C(1), C(0), dx + 1, dy, Q); printf("#16 zge3pt m=1, x=(1), sentinel 1000 just before x: y depends on the sentinel if it reads dx[-1]\n"); pr("y", host(dy, 1)); }
static void ellrt() { magma_z_matrix h = csr(2, 2, {0, 1, 2}, {0, 1}, {C(1), C(1)}), e = {Magma_CSR}; e.alignment = 32; e.blocksize = 64;
  magma_int_t c = magma_zmconvert(h, &e, Magma_CSR, Magma_ELLRT, Q); magma_z_matrix d = {Magma_CSR}; magma_zmtransfer(e, &d, Magma_CPU, Magma_DEV, Q);
  magma_z_matrix x = {Magma_CSR}, y = {Magma_CSR}; magma_zvinit(&x, Magma_DEV, 2, 1, C(1), Q); magma_zvinit(&y, Magma_DEV, 2, 1, C(7), Q);
  magma_int_t r = magma_z_spmv(C(1), d, x, C(0), y, Q); cudaError_t er = cudaGetLastError();
  printf("#20 ELLRT spmv with alignment 32, blocksize 64 (2048 threads per block): convert=%lld ret=%lld lastError=%s; expected y = (1,1)\n", (long long)c, (long long)r, cudaGetErrorString(er)); pr("y", host(y.dval, 2)); }
static void jaccard() { magma_z_matrix h = csr(2, 2, {0, 2, 4}, {0, 1, 0, 1}, {C(1), C(1), C(1), C(1)}), A = todev(h), hj = {Magma_CSR}, J = {Magma_CSR};
  magma_zmconvert(h, &hj, Magma_CSR, Magma_CSRCOO, Q); magma_zmtransfer(hj, &J, Magma_CPU, Magma_DEV, Q); magma_int_t r = magma_zjaccard_weights(A, &J, Q);
  printf("#24 zjaccard_weights, A = 2x2 all ones (every pair of rows identical): ret=%lld; expected all weights 1\n", (long long)r); pr("J.val", host(J.dval, 4)); }
static void lobpcg_shift() { std::vector<Z> X = {C(1), C(2), C(3), C(4), C(5), C(6)}; Z* dx = dev(X); magma_int_t r = magma_zlobpcg_shift(2, 3, 1, dx, Q);
  printf("#25 zlobpcg_shift(num_rows=2, num_vecs=3, shift=1) on row-major X = [[1,2,3],[4,5,6]]: ret=%lld; both rows should be shifted\n", (long long)r); pr("X", host(dx, 6)); }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; magma_init(); magma_queue_create(0, &Q);
  if (m == "mdotc") mdotc(); else if (m == "csr_sort") csr_sort(); else if (m == "csrcoo_t") csrcoo_t(); else if (m == "mbdiagdom") mbdiagdom(); else if (m == "bitonic") bitonic();
  else if (m == "vget_dev") vget_dev(); else if (m == "cuspaxpy") cuspaxpy(); else if (m == "cuspmm") cuspmm(); else if (m == "csr_sort_gpu") csr_sort_gpu(); else if (m == "ge3pt") ge3pt();
  else if (m == "ellrt") ellrt(); else if (m == "jaccard") jaccard(); else if (m == "lobpcg_shift") lobpcg_shift();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

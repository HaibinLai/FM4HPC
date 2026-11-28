// Twenty-third GPU round: the two round-19 tests that did not reach the code, and zheinertia (HANDOFF §29.25).  GPU 0 only.
#include "magma_v2.h"
#include "magmasparse.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z;
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static void residual_rows0() {   // A has 0 rows, b has 4: reaches the `% A.num_rows` branch (zresidual.cpp l.74)
  magma_z_matrix A = {Magma_CSR}, dA = {Magma_CSR}, b = {Magma_CSR}, x = {Magma_CSR}; std::vector<magma_index_t> r = {0};
  magma_zcsrset(0, 0, r.data(), NULL, NULL, &A, Q); magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q);
  magma_zvinit(&b, Magma_DEV, 4, 1, C(1), Q); magma_zvinit(&x, Magma_DEV, 4, 1, C(0), Q); double res[4] = {-1, -1, -1, -1};
  printf("zresidual with A 0x0 and b 4x1 ...\n"); fflush(stdout); magma_int_t ret = magma_zresidual(dA, b, x, res, Q); printf("  ret=%lld res[0]=%g\n", (long long)ret, res[0]); }
static void csrsplit2() {   // bsize 2, row 1 = {(1,1)=2, (1,0)=3}: diagonal stored before the lower entry
  std::vector<magma_index_t> r = {0, 1, 3}, c = {0, 1, 0}; std::vector<Z> v = {C(1), C(2), C(3)};
  magma_z_matrix A = {Magma_CSR}, D = {Magma_CSR}, R = {Magma_CSR}; magma_zcsrset(2, 2, r.data(), c.data(), v.data(), &A, Q);
  printf("zcsrsplit(offset 0, bsize 2), last row unsorted ...\n"); fflush(stdout); magma_int_t ret = magma_zcsrsplit(0, 2, A, &D, &R, Q);
  magma_z_matrix h = {Magma_CSR}; magma_zmtransfer(D, &h, D.memory_location, Magma_CPU, Q);
  printf("  ret=%lld D: nnz=%lld", (long long)ret, (long long)h.nnz); if (h.row) for (int i = 0; i < h.num_rows; i++) for (int j = h.row[i]; j < h.row[i + 1]; j++) printf(" (%d,%d)=%g", i, (int)h.col[j], MAGMA_Z_REAL(h.val[j]));
  printf("  (expected (0,0)=1 (1,1)=2 (1,0)=3 in some order); freeing ...\n"); fflush(stdout); magma_zmfree(&D, Q); magma_zmfree(&R, Q); printf("  freed\n"); }
static void heinertia() { int n = 4; std::vector<Z> A(n * n, C(0)); A[0] = C(1); A[5] = C(-1); A[10] = C(NAN); A[15] = C(2);
  Z* d; magma_zmalloc(&d, n * n); magma_zsetmatrix(n, n, A.data(), n, d, n, Q); std::vector<magma_int_t> ip = {1, 2, 3, 4}; magma_int_t* dip; magma_imalloc(&dip, n);
  magma_setvector(n, sizeof(magma_int_t), ip.data(), 1, dip, 1, Q); int* dn; magma_malloc((void**)&dn, 3 * sizeof(int));
  magma_int_t r = magmablas_zheinertia(MagmaLower, n, d, n, dip, dn, Q); cudaError_t ce = cudaDeviceSynchronize(); printf("  (ipiv passed as a device array; sync: %s)\n", cudaGetErrorString(ce)); int h[3] = {-1, -1, -1}; magma_getvector(3, sizeof(int), dn, 1, h, 1, Q);
  printf("magmablas_zheinertia, D = diag(1, -1, NaN, 2), no 2x2 blocks: ret=%lld  +/-/0 = %d/%d/%d\n", (long long)r, h[0], h[1], h[2]); }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; magma_init(); magma_queue_create(0, &Q);
  if (m == "residual") residual_rows0(); else if (m == "csrsplit") csrsplit2(); else if (m == "heinertia") heinertia();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

// Twenty-second GPU round: the remaining items of the independent review (HANDOFF §29.24).  GPU 0 only.
#include "magma_v2.h"
#include "magmasparse.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z;
extern "C" magma_int_t magmablas_zdiinertia(magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, int* dneig, magma_queue_t queue);
extern "C" magma_int_t magma_zmlumerge(magma_z_matrix L, magma_z_matrix U, magma_z_matrix* A, magma_queue_t queue);
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static void diinertia() { int n = 4; std::vector<Z> A(n * n, C(0)); A[0] = C(1); A[5] = C(-1); A[10] = C(NAN); A[15] = C(2);
  Z* d; magma_zmalloc(&d, n * n); magma_zsetmatrix(n, n, A.data(), n, d, n, Q); int* dn; magma_malloc((void**)&dn, 3 * sizeof(int));
  magma_int_t r = magmablas_zdiinertia(n, d, n, dn, Q); int h[3] = {-1, -1, -1}; magma_getvector(3, sizeof(int), dn, 1, h, 1, Q);
  printf("magmablas_zdiinertia (symbol present, not declared in magmablas_z.h), diag (1, -1, NaN, 2): ret=%lld  +/-/0 = %d/%d/%d\n", (long long)r, h[0], h[1], h[2]); }
static void compressor() { std::vector<magma_index_t> row = {0, 2, 3}, col = {0, 1, 1}; std::vector<Z> val = {C(1), C(0), C(2)};   // one stored zero
  magma_z_matrix A = {Magma_CSR}; magma_zcsrset(2, 2, row.data(), col.data(), val.data(), &A, Q); int own = (int)A.ownership; long nnz0 = A.nnz;
  magma_int_t r = magma_zmcsrcompressor(&A, Q);
  printf("zmcsrcompressor on a matrix from magma_zcsrset (ownership=%d) with one stored zero: ret=%lld nnz %ld -> %lld (the zero is removed if nnz drops to 2)\n", own, (long long)r, nnz0, (long long)A.nnz); }
static void lumerge(long k) {
  void (*arm)() = (void (*)())dlsym(RTLD_DEFAULT, "fi_cpu_arm"); void (*disarm)() = (void (*)())dlsym(RTLD_DEFAULT, "fi_cpu_disarm"); long (*count)() = (long (*)())dlsym(RTLD_DEFAULT, "fi_cpu_count");
  std::vector<magma_index_t> lr = {0, 1, 3}, lc = {0, 0, 1}, ur = {0, 2, 3}, uc = {0, 1, 1}; std::vector<Z> lv = {C(1), C(2), C(1)}, uv = {C(3), C(4), C(5)};
  magma_z_matrix L = {Magma_CSR}, U = {Magma_CSR}, A = {Magma_CSR}; magma_zcsrset(2, 2, lr.data(), lc.data(), lv.data(), &L, Q); magma_zcsrset(2, 2, ur.data(), uc.data(), uv.data(), &U, Q);
  if (arm) arm(); magma_int_t r = magma_zmlumerge(L, U, &A, Q); long c = count ? count() : -1; if (disarm) disarm();
  printf("zmlumerge k=%ld (host allocations inside: %ld): ret=%lld nnz=%lld; freeing the result ...\n", k, c, (long long)r, (long long)A.nnz); fflush(stdout);
  magma_zmfree(&A, Q); printf("  freed\n"); }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; long k = argc > 2 ? atol(argv[2]) : 0; magma_init(); magma_queue_create(0, &Q);
  if (m == "diinertia") diinertia(); else if (m == "compressor") compressor(); else if (m == "lumerge") lumerge(k);
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

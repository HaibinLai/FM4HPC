// Round 46 follow-up: CSR -> COO -> CSR on a 3x3 matrix, printing the arrays at each stage.
#include "magma_v2.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
static magma_queue_t Q;
static void dump(const char* t, magma_z_matrix& A) { printf("%s: storage=%d rows=%lld cols=%lld nnz=%lld\n  row:", t, (int)A.storage_type, (long long)A.num_rows, (long long)A.num_cols, (long long)A.nnz);
  int nr = A.storage_type == Magma_COO ? (int)A.nnz : (int)A.num_rows + 1; for (int i = 0; i < nr; i++) printf(" %d", (int)A.row[i]);
  printf("\n  col:"); for (int i = 0; i < (int)A.nnz; i++) printf(" %d", (int)A.col[i]); printf("\n  val:"); for (int i = 0; i < (int)A.nnz; i++) printf(" %g", MAGMA_Z_REAL(A.val[i])); printf("\n"); }
int main() {
  setvbuf(stdout, NULL, _IONBF, 0); magma_init(); magma_queue_create(0, &Q);
  // A = [1 0 2; 0 3 0; 4 0 5]
  magma_index_t r[] = {0, 2, 3, 5}, c[] = {0, 2, 1, 0, 2}; magmaDoubleComplex v[5]; double vv[] = {1, 2, 3, 4, 5}; for (int i = 0; i < 5; i++) v[i] = MAGMA_Z_MAKE(vv[i], 0);
  magma_index_t* rr = (magma_index_t*)malloc(sizeof r); magma_index_t* cc = (magma_index_t*)malloc(sizeof c); magmaDoubleComplex* vp = (magmaDoubleComplex*)malloc(sizeof v); memcpy(rr, r, sizeof r); memcpy(cc, c, sizeof c); memcpy(vp, v, sizeof v);
  magma_z_matrix A = {Magma_CSR}, B = {Magma_CSR}, C = {Magma_CSR}; magma_zcsrset(3, 3, rr, cc, vp, &A, Q); dump("CSR in", A);
  magma_int_t e = magma_zmconvert(A, &B, Magma_CSR, Magma_COO, Q); printf("CSR->COO ret %lld\n", (long long)e); dump("COO", B);
  e = magma_zmconvert(B, &C, Magma_COO, Magma_CSR, Q); printf("COO->CSR ret %lld\n", (long long)e); dump("CSR out", C);
  magma_queue_destroy(Q); magma_finalize(); return 0; }

// Round 67 (roxy): magma_zprint_csr_mtx with MagmaColMajor goes through z_transpose_csr (magma_zmio.cpp l.587), whose
// cleanup block frees the three arrays it has just returned.  Prints the matrix both ways; the harness compares the entries.
#include "magma_v2.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
int main() {
  magma_init(); magma_queue_t Q; magma_queue_create(0, &Q);
  // 3 x 4: rows {0: (0,1) (2,2)}, {1: (1,3)}, {2: (0,4) (3,5)}
  magma_index_t* row = (magma_index_t*)malloc(4 * sizeof(magma_index_t)); magma_index_t* col = (magma_index_t*)malloc(5 * sizeof(magma_index_t)); magmaDoubleComplex* val = (magmaDoubleComplex*)malloc(5 * sizeof(magmaDoubleComplex));
  magma_index_t r[] = {0, 2, 3, 5}, c[] = {0, 2, 1, 0, 3}; double v[] = {1, 2, 3, 4, 5}; for (int i = 0; i < 4; i++) row[i] = r[i]; for (int i = 0; i < 5; i++) { col[i] = c[i]; val[i] = MAGMA_Z_MAKE(v[i], 10 * v[i]); }
  printf("== RowMajor\n"); fflush(stdout); magma_zprint_csr_mtx(3, 4, 5, &val, &row, &col, MagmaRowMajor, Q); fflush(stdout);
  printf("== ColMajor\n"); fflush(stdout); magma_int_t ret = magma_zprint_csr_mtx(3, 4, 5, &val, &row, &col, MagmaColMajor, Q); fflush(stdout);
  printf("== ret %lld\n", (long long)ret); magma_queue_destroy(Q); magma_finalize(); return 0; }

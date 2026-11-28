// GPU check of the sparse preconditioner-setup findings (HANDOFF §28.7/28.8): ILU numerical zero pivot + PBiCG,
// and Jacobi with a stored zero diagonal.  A = [[0,1],[1,0]] with the zero diagonal STORED in CSR.  GPU 0.
#include "magma_v2.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstring>
#include <cmath>
static magma_queue_t Q;
static void mat(magma_z_matrix* dA, double a00, double a11) {   // 2x2 CSR with both diagonals stored
  static magma_index_t row[3] = {0, 2, 4}, col[4] = {0, 1, 0, 1};
  static magmaDoubleComplex val[4]; val[0] = MAGMA_Z_MAKE(a00, 0); val[1] = MAGMA_Z_MAKE(1, 0); val[2] = MAGMA_Z_MAKE(1, 0); val[3] = MAGMA_Z_MAKE(a11, 0);
  magma_z_matrix h = {Magma_CSR}; magma_zcsrset(2, 2, row, col, val, &h, Q); magma_zmtransfer(h, dA, Magma_CPU, Magma_DEV, Q);
}
static void vec(magma_z_matrix* d, double v0, double v1) {
  static magmaDoubleComplex v[2]; v[0] = MAGMA_Z_MAKE(v0, 0); v[1] = MAGMA_Z_MAKE(v1, 0);
  magma_z_matrix h = {Magma_CSR}; magma_zvset(2, 1, v, &h, Q); magma_zmtransfer(h, d, Magma_CPU, Magma_DEV, Q);
}
static void get(magma_z_matrix d, double* o) { magma_z_matrix h = {Magma_CSR}; magma_zmtransfer(d, &h, Magma_DEV, Magma_CPU, Q);
  for (int i = 0; i < 2; i++) o[i] = MAGMA_Z_REAL(h.val[i]); magma_zmfree(&h, Q); }
static void pbicg_ilu(const char* nm, double a00, double a11) {
  magma_z_matrix dA = {Magma_CSR}, db = {Magma_CSR}, dx = {Magma_CSR};
  mat(&dA, a00, a11); vec(&db, 1, 2); vec(&dx, 0, 0);
  magma_z_solver_par sp; magma_z_preconditioner pp; memset(&sp, 0, sizeof sp); memset(&pp, 0, sizeof pp);
  sp.solver = Magma_PBICG; sp.maxiter = 1000; sp.atol = 1e-16; sp.rtol = 1e-10; pp.solver = Magma_ILU; pp.levels = 0;
  magma_zsolverinfo_init(&sp, &pp, Q);
  fprintf(stderr, "[%s] precondsetup...\n", nm);
  magma_int_t si = magma_z_precondsetup(dA, db, &sp, &pp, Q);
  fprintf(stderr, "[%s] setup returned %lld, pbicg...\n", nm, (long long)si);
  magma_int_t ret = magma_zpbicg(dA, db, &dx, &sp, &pp, Q);
  fprintf(stderr, "[%s] pbicg returned\n", nm);
  double x[2]; get(dx, x);
  printf("%-38s precondsetup info=%lld | pbicg ret=%lld info=%lld numiter=%lld final_res=%g | x=(%g, %g)\n", nm, (long long)si,
         (long long)ret, (long long)sp.info, (long long)sp.numiter, sp.final_res, x[0], x[1]);
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  magma_init(); magma_queue_create(0, &Q);
  bool doIlu = argc < 2 || argv[1][0] != 'j';
  if (doIlu) {
  printf("== ILU(0) + PBiCG, exact solution of A x = (1,2) ==\n");
  if (argc > 1 && argv[1][0] == 'c') pbicg_ilu("A=[[2,1],[1,2]] (control)", 2, 2);   // exact x = (0, 1)
  else pbicg_ilu("A=[[0,1],[1,0]] (ILU pivot 0)", 0, 0);       // exact x = (2, 1)
  }
  printf("== Jacobi setup (magma_zjacobisetup_diagscal) ==\n");
  for (double d : {0.0, 2.0}) {
    magma_z_matrix dA = {Magma_CSR}, dd = {Magma_CSR}; mat(&dA, d, 2.0);
    fprintf(stderr, "[jacobi d=%g] setup...\n", d);
    magma_int_t info = magma_zjacobisetup_diagscal(dA, &dd, Q);
    fprintf(stderr, "[jacobi d=%g] setup returned %lld\n", d, (long long)info); double v[2]; get(dd, v);
    printf("A=[[%g,1],[1,2]] (diagonal stored): setup info=%lld  D^-1 = (%g, %g)\n", d, (long long)info, v[0], v[1]);
  }
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

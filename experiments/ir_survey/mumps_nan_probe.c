/* MUMPS (sequential) iterative refinement + error analysis on a NaN right-hand side.
 * Prediction (IR_SURVEY_zh.md): DMUMPS_SOL_OMEGA takes DXMAX from idamax over x; if that is NaN every TAU is NaN, every
 * row is skipped, OMEGA = 0 < ARRET, so refinement reports convergence and INFO(1) = 0 (outcome depends on the BLAS
 * idamax NaN behaviour). Matrix: two decoupled tridiag(-1,4,-1) blocks, n = 1000; x_true = 1.
 * ICNTL(10) = 3 (up to 3 refinement steps with stopping test), ICNTL(11) = 2 (error statistics). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "dmumps_c.h"
#define ICNTL(I) icntl[(I)-1]
#define INFOG(I) infog[(I)-1]
#define RINFOG(I) rinfog[(I)-1]

static void run(const char *name, int nanpos) {
    const int n = 1000, h = n / 2;
    int nnz = 0, *irn = malloc(sizeof(int) * 3 * n), *jcn = malloc(sizeof(int) * 3 * n);
    double *a = malloc(sizeof(double) * 3 * n), *rhs = calloc(n, sizeof(double));
    for (int i = 0; i < n; ++i) {
        irn[nnz] = i + 1; jcn[nnz] = i + 1; a[nnz++] = 4.0; rhs[i] += 4.0;
        if (i + 1 < n && i + 1 != h) {
            irn[nnz] = i + 1; jcn[nnz] = i + 2; a[nnz++] = -1.0; rhs[i] -= 1.0;
            irn[nnz] = i + 2; jcn[nnz] = i + 1; a[nnz++] = -1.0; rhs[i + 1] -= 1.0;
        }
    }
    if (nanpos >= 0) rhs[nanpos] = NAN;
    DMUMPS_STRUC_C id;
    id.comm_fortran = -987654; id.par = 1; id.sym = 0; id.job = -1;
    dmumps_c(&id);
    id.ICNTL(1) = -1; id.ICNTL(2) = -1; id.ICNTL(3) = -1; id.ICNTL(4) = 0;
    id.n = n; id.nnz = nnz; id.irn = irn; id.jcn = jcn; id.a = a; id.rhs = rhs;
    id.ICNTL(10) = 3; id.ICNTL(11) = 2;
    id.job = 6;
    dmumps_c(&id);
    int nonfin = 0; double err = 0;
    for (int i = 0; i < n; ++i) { nonfin += !isfinite(rhs[i]); if (isfinite(rhs[i])) err = fmax(err, fabs(rhs[i] - 1.0)); }
    printf("%-26s INFOG(1)=%d INFOG(15)(IR steps)=%d  omega1=RINFOG(7)=%-9.3g omega2=RINFOG(8)=%-9.3g "
           "err_bound=RINFOG(9)=%-9.3g  nonfinite(x)=%d/%d  -> %s\n",
           name, id.INFOG(1), id.INFOG(15), id.RINFOG(7), id.RINFOG(8), id.RINFOG(9), nonfin, n,
           (id.INFOG(1) == 0 && nonfin && isfinite(id.RINFOG(7)) && isfinite(id.RINFOG(8)))
               ? "SILENT: clean backward error for non-finite x" : "flagged/ok");
    id.job = -2; dmumps_c(&id);
    free(irn); free(jcn); free(a); free(rhs);
}

int main(void) {
    run("baseline (finite)", -1);
    run("b[0]=NaN", 0);
    run("b[500]=NaN", 500);
    run("b[999]=NaN", 999);
    return 0;
}

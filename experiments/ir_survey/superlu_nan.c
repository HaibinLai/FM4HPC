/* SuperLU serial dgssvx + dgsrfs: does iterative refinement report a clean backward error for a NaN solution?
 * Prediction (IR_SURVEY_zh.md, D11): berr uses SUPERLU_MAX(x,y) = (x>y)?x:y, so a NaN in row 0 is dropped as soon as
 * a later row is finite -> berr = 0, ferr tiny, info = 0, while x[0] = NaN.
 * Cases: A = I (n=2 and n=1000 tridiagonal), NaN placed in b[0] (masked) vs b[n-1] (control: NaN reaches the end). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "slu_ddefs.h"

static void run(const char *name, int n, int tri, int nanpos, double binf) {
    int nnz = tri ? 3 * n - 2 - (tri == 2 ? 2 : 0) : n;
    double *a = doubleMalloc(nnz);
    int_t *asub = intMalloc(nnz), *xa = intMalloc(n + 1);
    int k = 0;
    for (int j = 0; j < n; ++j) {
        xa[j] = k;
        if (tri && j > 0 && !(tri == 2 && j == n / 2)) { a[k] = -1.0; asub[k++] = j - 1; }
        a[k] = tri ? 4.0 : 1.0; asub[k++] = j;
        if (tri && j < n - 1 && !(tri == 2 && j == n / 2 - 1)) { a[k] = -1.0; asub[k++] = j + 1; }
    }
    xa[n] = k;
    double *b = doubleMalloc(n), *x = doubleMalloc(n), *b0 = doubleMalloc(n);
    for (int i = 0; i < n; ++i) b[i] = 1.0;
    if (nanpos >= 0) b[nanpos] = NAN;
    if (binf != 0) b[0] = binf;
    for (int i = 0; i < n; ++i) b0[i] = b[i];
    SuperMatrix A, L, U, B, X;
    dCreate_CompCol_Matrix(&A, n, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&B, n, 1, b, n, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, n, 1, x, n, SLU_DN, SLU_D, SLU_GE);
    int *perm_r = intMalloc(n), *perm_c = intMalloc(n), *etree = intMalloc(n);
    double *R = doubleMalloc(n), *C = doubleMalloc(n), ferr, berr, rpg, rcond;
    char equed[1];
    superlu_options_t opt;
    set_default_options(&opt);
    opt.ColPerm = NATURAL;
    opt.Equil = NO;
    opt.IterRefine = SLU_DOUBLE;
    SuperLUStat_t stat;
    StatInit(&stat);
    GlobalLU_t Glu;
    mem_usage_t mem;
    int info = 0;
    dgssvx(&opt, &A, perm_c, perm_r, etree, equed, R, C, &L, &U, NULL, 0, &B, &X, &rpg, &rcond, &ferr, &berr, &Glu,
           &mem, &stat, &info);
    int nnan = 0, ninf = 0;
    for (int i = 0; i < n; ++i) { nnan += isnan(x[i]); ninf += isinf(x[i]); }
    printf("%-34s n=%5d info=%d  berr=%-10.3g ferr=%-10.3g refine_steps=%d  x[0]=%g x[n-1]=%g  #NaN(x)=%d #Inf(x)=%d  -> %s\n",
           name, n, info, berr, ferr, stat.RefineSteps, x[0], x[n - 1], nnan, ninf,
           (info == 0 && (nnan || ninf) && berr <= 2.3e-16) ? "VIOLATION (clean berr, non-finite x)" : "ok/flagged");
    StatFree(&stat);
}

int main(void) {
    run("I, b[0]=NaN (masked)", 2, 0, 0, 0);
    run("I, b[n-1]=NaN (control)", 2, 0, 1, 0);
    run("tridiag, b[0]=NaN (masked)", 1000, 1, 0, 0);
    run("tridiag, b[n-1]=NaN (control)", 1000, 1, 999, 0);
    run("2 decoupled tridiag blocks, b[0]=NaN", 1000, 2, 0, 0);
    run("2 decoupled blocks, b[0]=Inf", 1000, 2, -1, INFINITY);
    run("tridiag, b[0]=Inf", 1000, 1, -1, INFINITY);
    run("tridiag, finite (baseline)", 1000, 1, -1, 0);
    return 0;
}

/* PaStiX GMRES refinement with a zero initial guess (IR_SURVEY_zh.md: z_refine_gmres.c computes normx once,
 * before the outer loop, and skips "r0 = b - A x" whenever normx == 0, so every restart uses r0 = b).
 * Documented contract (pastix_task_refine): "On entry, the initial guess x0 for the refinement step, that may be the
 * solution returned by the solve step or any other initial guess."
 * Prediction: with x0 = 0 the residual check at the top of the outer loop always sees ||b||/||b|| = 1, so GMRES runs to
 * IPARM_ITERMAX and x grows like (#cycles) * x_true, while pastix_task_refine returns PASTIX_SUCCESS.
 * Verdict uses the true residual ||b - A x||/||b|| (spm's Laplacian generator is near-singular, so the error against
 * x_true is not meaningful; <x,xt>/<xt,xt> is shown only to expose the k*x_true growth).
 * usage: pastix_gmres_x0 <laplacian spec, e.g. d:1000 or d:20:20:20> */
#include <math.h>
#include <pastix.h>
#include <spm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *rname(int r) {
    switch (r) { case PastixRefineGMRES: return "GMRES"; case PastixRefineCG: return "CG";
                 case PastixRefineSR: return "SR"; case PastixRefineBiCGSTAB: return "BiCGSTAB"; }
    return "?";
}

int main(int argc, char **argv) {
    const char *spec = argc > 1 ? argv[1] : "d:1000";
    pastix_int_t iparm[IPARM_SIZE];
    double dparm[DPARM_SIZE];
    pastixInitParam(iparm, dparm);
    iparm[IPARM_VERBOSE] = PastixVerboseNot;
    iparm[IPARM_FACTORIZATION] = PastixFactLU;
    iparm[IPARM_THREAD_NBR] = 4;
    pastix_data_t *pd = NULL;
    pastixInit(&pd, MPI_COMM_WORLD, iparm, dparm);
    spmatrix_t spm, spm2;
    char buf[64]; strncpy(buf, spec, 63); buf[63] = 0;
    if (spmReadDriver(SpmDriverLaplacian, buf, &spm) != SPM_SUCCESS) { fprintf(stderr, "spm read failed\n"); return 1; }
    if (spmCheckAndCorrect(&spm, &spm2) != 0) { spmExit(&spm); spm = spm2; }
    pastix_task_analyze(pd, &spm);
    int n = spm.nexp;
    double *xt = malloc(n * sizeof(double)), *b0 = malloc(n * sizeof(double)), *b = malloc(n * sizeof(double)),
           *x = malloc(n * sizeof(double)), *r = malloc(n * sizeof(double));
    spmGenRHS(SpmRhsRndX, 1, &spm, xt, n, b0, n);     /* before numfact, as in example/simple.c */
    spmatrix_t spmc; spmCopy(&spm, &spmc);           /* untouched copy for residuals */
    pastix_task_numfact(pd, &spm);
    double xtn = 0, xtt = 0;
    for (int i = 0; i < n; ++i) { xtn = fmax(xtn, fabs(xt[i])); xtt += xt[i] * xt[i]; }
    printf("PaStiX e2ecf99, Laplacian %s, n=%d, LU, eps_refine=%g, itermax=%ld, gmres_im=%ld\n", spec, n,
           dparm[DPARM_EPSILON_REFINEMENT], (long)iparm[IPARM_ITERMAX], (long)iparm[IPARM_GMRES_IM]);
    int methods[] = {PastixRefineGMRES, PastixRefineBiCGSTAB, PastixRefineSR};
    const char *guesses[] = {"solve result (normal use)", "x0 = 0", "x0 = 0.5*x_true"};
    for (int m = 0; m < 3; ++m) {
        for (int g = 0; g < 3; ++g) {
            iparm[IPARM_REFINEMENT] = methods[m];
            memcpy(b, b0, n * sizeof(double));
            if (g == 0) { memcpy(x, b0, n * sizeof(double)); pastix_task_solve(pd, n, 1, x, n); }
            else if (g == 1) memset(x, 0, n * sizeof(double));
            else for (int i = 0; i < n; ++i) x[i] = 0.5 * xt[i];
            int rc = pastix_task_refine(pd, n, 1, b, n, x, n);
            double err = 0, xdot = 0; int bad = 0;
            for (int i = 0; i < n; ++i) { err = fmax(err, fabs(x[i] - xt[i])); xdot += x[i] * xt[i]; bad += !isfinite(x[i]); }
            double bn = 0, rn = 0;
            memcpy(r, b0, n * sizeof(double));
            spmMatVec(SpmNoTrans, -1.0, &spmc, x, 1.0, r);
            for (int i = 0; i < n; ++i) { bn += b0[i] * b0[i]; rn += r[i] * r[i]; }
            int viol = (rc == PASTIX_SUCCESS) && (bad || !(sqrt(rn / bn) <= 1e-10));
            printf("  %-8s %-27s rc=%d (%s) nbiter=%4ld  ||b-Ax||/||b||=%-9.3g rel_err=%-10.3g  <x,xt>/<xt,xt>=%-8.4g nonfinite=%d  -> %s\n",
                   rname(methods[m]), guesses[g], rc, rc == PASTIX_SUCCESS ? "SUCCESS" : "error",
                   (long)iparm[IPARM_NBITER], sqrt(rn / bn), err / xtn, xdot / xtt, bad, viol ? "VIOLATION (SUCCESS, wrong x)" : "ok");
        }
    }
    spmExit(&spm); spmExit(&spmc);
    pastixFinalize(&pd);
    return 0;
}

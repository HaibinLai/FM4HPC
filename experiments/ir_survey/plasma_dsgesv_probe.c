/* PLASMA plasma_dsgesv (generated from compute/zcgesv.c): mixed-precision LU + refinement.
 * Documented contract (compute/zcgesv.c header): stop when RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX; "If the approach fails
 * the method switches to a double precision factorization and solve"; @retval PlasmaSuccess.
 * Predictions (IR_SURVEY_zh.md):
 *   P1 (D1/D11) NaN in b, or NaN produced by FP32 pivot growth, is accepted: retval 0, iter = 0, X contains NaN.
 *   P2 (D9)  FP32 LU singular although A is FP64-nonsingular, or FP32 conversion overflow, calls plasma_request_fail;
 *           every later task (the FP64 fallback included) is skipped, so retval != 0 and X is never written.
 * Reference: LAPACKE_dgesv in FP64.  usage: plasma_dsgesv_probe */
#include <lapacke.h>
#include <math.h>
#include <plasma.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void probe(const char *name, int n, const double *A0, const double *b0) {
    double *A = malloc(sizeof(double) * n * n), *B = malloc(sizeof(double) * n), *X = malloc(sizeof(double) * n);
    double *Ar = malloc(sizeof(double) * n * n), *xr = malloc(sizeof(double) * n);
    int *ipiv = malloc(sizeof(int) * n), *ipr = malloc(sizeof(int) * n), iter = -999;
    memcpy(A, A0, sizeof(double) * n * n); memcpy(B, b0, sizeof(double) * n);
    for (int i = 0; i < n; ++i) X[i] = -12345.0;                       /* sentinel: detects "X never written" */
    int rv = plasma_dsgesv(n, 1, A, n, ipiv, B, n, X, n, &iter);
    memcpy(Ar, A0, sizeof(double) * n * n); memcpy(xr, b0, sizeof(double) * n);
    int ri = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, 1, Ar, n, ipr, xr, n);
    int nonfin = 0, sentinel = 0; double err = 0, xn = 0;
    for (int i = 0; i < n; ++i) {
        nonfin += !isfinite(X[i]); sentinel += X[i] == -12345.0;
        err = fmax(err, fabs(X[i] - xr[i])); xn = fmax(xn, fabs(xr[i]));
    }
    int silent = rv == 0 && iter >= 0 && (nonfin || !(err <= 1e-8 * xn));
    printf("%-44s n=%3d retval=%d iter=%4d  nonfinite(X)=%3d unwritten(X)=%3d  relerr_vs_dgesv=%-9.3g dgesv_info=%d -> %s\n",
           name, n, rv, iter, nonfin, sentinel, xn > 0 ? err / xn : err, ri,
           silent ? "SILENT WRONG (success reported)" : (rv != 0 ? "error returned, no FP64 fallback" : "ok"));
    free(A); free(B); free(X); free(Ar); free(xr); free(ipiv); free(ipr);
}

static double *wilk(int n, double s) {           /* s*W(n): W_ii = 1, W_ij = -1 (i>j), W_in = 1; growth 2^(n-1) */
    double *A = calloc((size_t)n * n, sizeof(double));
    for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
        A[i + j * n] = s * ((i == j || j == n - 1) ? 1.0 : (i > j ? -1.0 : 0.0));
    return A;
}

int main(void) {
    plasma_init();
    const int n = 40;
    double *b = malloc(sizeof(double) * 300);
    /* control */
    double *R = malloc(sizeof(double) * 300 * 300);
    srand(1);
    for (int i = 0; i < 300 * 300; ++i) R[i] = (double)rand() / RAND_MAX - 0.5;
    for (int i = 0; i < 300; ++i) { R[i + i * 300] += 300.0; b[i] = 1.0; }
    probe("control: random diag-dominant", 300, R, b);
    /* P1: NaN in b */
    b[0] = NAN; probe("P1 random diag-dominant, b[0]=NaN", 300, R, b);
    b[0] = 1.0; b[150] = NAN; probe("P1 random diag-dominant, b[150]=NaN", 300, R, b);
    /* P1: FP32 pivot growth overflow; A itself fits FP32 */
    for (int k = 60; k <= 120; k += 30) {
        double *W = wilk(n, ldexp(1.0, k));
        for (int i = 0; i < n; ++i) b[i] = 0;
        for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) b[i] += W[i + j * n];   /* x_true = 1, exact */
        char nm[64]; snprintf(nm, 64, "P1 Wilkinson 2^%d*W(40), x_true=1", k);
        probe(nm, n, W, b); free(W);
    }
    /* P2: FP32-singular, FP64-nonsingular */
    double S[4] = {1.0, 1.0, 1.0, 1.0 + ldexp(1.0, -30)}, bs[2] = {2.0, 2.0 + ldexp(1.0, -30)};
    probe("P2 [[1,1],[1,1+2^-30]] (FP32 singular)", 2, S, bs);
    /* P2: b overflows FP32 */
    double I2[4] = {1, 0, 0, 1}, bo[2] = {1e39, 1.0};
    probe("P2 A=I, b=(1e39,1) (b overflows FP32)", 2, I2, bo);
    plasma_finalize();
    return 0;
}

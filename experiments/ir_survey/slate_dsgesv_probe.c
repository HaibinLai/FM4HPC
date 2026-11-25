/* SLATE LAPACK-compatible slate_dsgesv (lapack_api/lapack_gesv_mixed.cc).
 * Predictions (IR_SURVEY_zh.md): X is created from b/ldb (x and ldx ignored), the return value of gesv_mixed is dropped
 * and `*info = 0;  // todo: get a real value for info`.
 *   L1 (D7) the caller's x array is never written; the answer (if any) lands in b.
 *   L2 (D10) a singular A still returns info = 0.
 * Reference: LAPACK dgesv semantics (info > 0 for exactly singular U; x holds the solution, b unchanged). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void slate_dsgesv_(const int*, const int*, double*, const int*, int*, double*, const int*, double*, const int*, double*,
                   float*, int*, int*);
static void probe(const char *name, int n, double *A, double *b, const double *xtrue) {
    double *x = malloc(sizeof(double) * n), *work = malloc(sizeof(double) * n), *b0 = malloc(sizeof(double) * n);
    float *swork = malloc(sizeof(float) * n * (n + 1));
    int *ipiv = malloc(sizeof(int) * n), iter = -999, info = -999, one = 1;
    for (int i = 0; i < n; ++i) x[i] = -12345.0;
    memcpy(b0, b, sizeof(double) * n);
    slate_dsgesv_(&n, &one, A, &n, ipiv, b, &n, x, &n, work, swork, &iter, &info);
    int xun = 0, bchg = 0; double ex = 0, eb = 0;
    for (int i = 0; i < n; ++i) {
        xun += x[i] == -12345.0; bchg += b[i] != b0[i];
        if (xtrue) { ex = fmax(ex, fabs(x[i] - xtrue[i])); eb = fmax(eb, fabs(b[i] - xtrue[i])); }
    }
    printf("%-34s n=%3d info=%d iter=%4d  x unwritten=%3d/%d  b modified=%3d/%d", name, n, info, iter, xun, n, bchg, n);
    if (xtrue) printf("  max|x-xtrue|=%-9.3g max|b-xtrue|=%-9.3g", ex, eb);
    printf("  b[0..2]=(%.6g, %.6g, %.6g)\n", b[0], b[1], b[2]);
}
int main(void) {
    const int n = 100;
    double *A = calloc(n * n, sizeof(double)), *b = malloc(sizeof(double) * n), *xt = malloc(sizeof(double) * n);
    srand(1);
    for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) A[i + j * n] = (double)rand() / RAND_MAX - 0.5 + (i == j ? n : 0);
    for (int i = 0; i < n; ++i) xt[i] = 1.0;
    for (int i = 0; i < n; ++i) { b[i] = 0; for (int j = 0; j < n; ++j) b[i] += A[i + j * n]; }
    probe("L1 random diag-dominant, x_true=1", n, A, b, xt);
    double *Z = calloc(n * n, sizeof(double));
    for (int i = 0; i < n - 1; ++i) Z[i + i * n] = 1.0;          /* last diagonal entry 0: exactly singular */
    for (int i = 0; i < n; ++i) b[i] = 1.0;
    probe("L2 exactly singular diag(1,..,1,0)", n, Z, b, NULL);
    return 0;
}

/* Is MAGMA's NaN-unsafe acceptance inherited from reference LAPACK dsgesv (IF( RNRM.GT.XNRM*CTE ) GO TO 10)?
   Calls dsgesv_ from the system liblapack.so.3 on scaled Wilkinson matrices; compares with dgesv_. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern void dsgesv_(int*, int*, double*, int*, int*, double*, int*, double*, int*, double*, float*, int*, int*);
extern void dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
static void wilk(double* A, int n, double s) {
  for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
    double v = (i == j) ? 1.0 : (i > j ? -1.0 : 0.0);
    if (j == n - 1) v = 1.0;
    A[j * n + i] = v * s;
  }
}
int main(void) {
  int cases[][2] = {{40, 80}, {40, 100}, {30, 120}, {20, 0}};
  for (int c = 0; c < 4; c++) {
    int n = cases[c][0], e = cases[c][1], nrhs = 1, info, iter;
    double s = ldexp(1.0, e);
    double *A = malloc(sizeof(double) * n * n), *A2 = malloc(sizeof(double) * n * n);
    double *b = calloc(n, sizeof(double)), *b2 = malloc(sizeof(double) * n), *x = malloc(sizeof(double) * n);
    double *work = malloc(sizeof(double) * n); float *swork = malloc(sizeof(float) * n * (n + 1));
    int *ipiv = malloc(sizeof(int) * n);
    wilk(A, n, s);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) b[i] += A[j * n + i];
    memcpy(A2, A, sizeof(double) * n * n); memcpy(b2, b, sizeof(double) * n);
    dsgesv_(&n, &nrhs, A, &n, ipiv, b, &n, x, &n, work, swork, &iter, &info);
    int bad = 0; double err = 0;
    for (int i = 0; i < n; i++) { if (!isfinite(x[i])) bad++; else err = fmax(err, fabs(x[i] - 1)); }
    printf("W(%d)*2^%d  LAPACK dsgesv: iter=%d info=%d nonfinite=%d/%d max|x-1|=%.3g x[0]=%g\n", n, e, iter, info, bad, n, err, x[0]);
    dgesv_(&n, &nrhs, A2, &n, ipiv, b2, &n, &info);
    bad = 0; err = 0;
    for (int i = 0; i < n; i++) { if (!isfinite(b2[i])) bad++; else err = fmax(err, fabs(b2[i] - 1)); }
    printf("               LAPACK dgesv : info=%d nonfinite=%d/%d max|x-1|=%.3g\n", info, bad, n, err);
  }
  return 0;
}

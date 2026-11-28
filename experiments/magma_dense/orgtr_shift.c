/* CPU check of the reflector-shift loops in MAGMA 2.10.0 src/[sdcz]ungtr.cpp (here the d copy, dorgtr.cpp, which is
   generated from the same source).  Prediction (Lean MagmaAudit.ungtr_upper_shift_short / ungtr_lower_shift_wrong):
     upper: `for (i = 0; i < j-1; ++i)` stops one row early (LAPACK DORGTR: I = 1..J-1, i.e. i < j), so A(j-1,j)
            keeps the reflector entry of column j instead of column j+1;
     lower: `for (i = j; i < n-1; ++i)` starts on the diagonal and stops one row early (LAPACK: I = J+1..N), and the
            first-column loop `for (i = 1; i < n-1; ++i)` leaves A(n-1,0) uncleared.
   The loops below are copied verbatim from dorgtr.cpp l.140-151 and l.160-170.  The upper path then calls LAPACK
   dorgql exactly as MAGMA does; for the lower path MAGMA calls magma_dorgqr (GPU) with the T blocks of magma_dsytrd,
   which is replaced here by LAPACK dorgqr on the same reflectors (same Q in exact arithmetic).
   Control: the same code with LAPACK's loop bounds (fixed = 1) must reproduce LAPACK.
   Reference: LAPACK dorgtr on the same dsytrd output.  Build: cc orgtr_shift.c -framework Accelerate -o orgtr_shift */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
extern void dsytrd_(char*,int*,double*,int*,double*,double*,double*,double*,int*,int*);
extern void dorgtr_(char*,int*,double*,int*,double*,double*,int*,int*);
extern void dorgql_(int*,int*,int*,double*,int*,double*,double*,int*,int*);
extern void dorgqr_(int*,int*,int*,double*,int*,double*,double*,int*,int*);
#define A(i,j) (a + (i) + (size_t)(j)*lda)

static void magma_shift_and_generate(int upper, int n, double *a, int lda, double *tau, int fixed) {
  int i, j, iinfo, lwork = 64 * n; double *work = malloc(sizeof(double) * lwork);
  if (upper) {
    for (j = 0; j < n-1; ++j) {
      for (i = 0; i < (fixed ? j : j-1); ++i)
        *A(i, j) = *A(i, j + 1);
      *A(n-1, j) = 0.0;
    }
    for (i = 0; i < n-1; ++i) *A(i, n-1) = 0.0;
    *A(n-1, n-1) = 1.0;
    int i1 = n - 1; dorgql_(&i1, &i1, &i1, A(0,0), &lda, tau, work, &lwork, &iinfo);
  } else {
    for (j = n-1; j > 0; --j) {
      *A(0, j) = 0.0;
      for (i = (fixed ? j+1 : j); i < (fixed ? n : n-1); ++i)
        *A(i, j) = *A(i, j - 1);
    }
    *A(0, 0) = 1.0;
    for (i = 1; i < (fixed ? n : n-1); ++i) *A(i, 0) = 0.0;
    if (n > 1) { int i1 = n - 1; dorgqr_(&i1, &i1, &i1, A(1,1), &lda, tau, work, &lwork, &iinfo); }
  }
  free(work);
}

/* ||Q^T Q - I||_max and ||A - Q T Q^T||_max / ||A||_max, T tridiagonal from d, e */
static void quality(int n, const double *Q, const double *A0, const double *d, const double *e, double *orth, double *rec) {
  double amax = 0, o = 0, r = 0;
  for (int k = 0; k < n*n; k++) amax = fmax(amax, fabs(A0[k]));
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
    double s = 0; for (int k = 0; k < n; k++) s += Q[k + i*n] * Q[k + j*n];
    o = fmax(o, fabs(s - (i == j)));
    double t = 0;                                   /* (Q T Q^T)(i,j) */
    for (int k = 0; k < n; k++) {
      double tk = d[k] * Q[j + k*n];
      if (k > 0)   tk += e[k-1] * Q[j + (k-1)*n];
      if (k < n-1) tk += e[k]   * Q[j + (k+1)*n];
      t += Q[i + k*n] * tk;
    }
    r = fmax(r, fabs(A0[i + j*n] - t));
  }
  *orth = o; *rec = r / amax;
}

int main(void) {
  int sizes[] = {3, 4, 8, 50};
  srand(7);
  for (int up = 1; up >= 0; up--) for (int si = 0; si < 4; si++) {
    int n = sizes[si], lda = n, info, lwork = 64 * n; char uplo = up ? 'U' : 'L';
    double *A0 = malloc(sizeof(double)*n*n), *a = malloc(sizeof(double)*n*n), *q = malloc(sizeof(double)*n*n);
    double *d = malloc(sizeof(double)*n), *e = malloc(sizeof(double)*n), *tau = malloc(sizeof(double)*n), *w = malloc(sizeof(double)*lwork);
    for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { double v = rand() / (double)RAND_MAX - 0.5; A0[i+j*n] = A0[j+i*n] = v; }
    memcpy(a, A0, sizeof(double)*n*n);
    dsytrd_(&uplo, &n, a, &lda, d, e, tau, w, &lwork, &info);
    double *f = malloc(sizeof(double)*n*n); memcpy(f, a, sizeof(double)*n*n);
    memcpy(q, a, sizeof(double)*n*n);
    dorgtr_(&uplo, &n, q, &lda, tau, w, &lwork, &info);                  /* reference */
    magma_shift_and_generate(up, n, a, lda, tau, 0);                    /* MAGMA loops, verbatim */
    magma_shift_and_generate(up, n, f, lda, tau, 1);                    /* control: LAPACK loop bounds */
    double diff = 0, dfix = 0;
    for (int k = 0; k < n*n; k++) { diff = fmax(diff, fabs(a[k] - q[k])); dfix = fmax(dfix, fabs(f[k] - q[k])); }
    double o1, r1, o2, r2; quality(n, q, A0, d, e, &o1, &r1); quality(n, a, A0, d, e, &o2, &r2);
    printf("uplo=%c n=%2d  max|Q_magma-Q_lapack|=%.2e (fixed bounds: %.1e)  LAPACK: orth %.1e recon %.1e   MAGMA loops: orth %.1e recon %.1e\n",
           uplo, n, diff, dfix, o1, r1, o2, r2);
    free(f); free(A0); free(a); free(q); free(d); free(e); free(tau); free(w);
  }
  return 0;
}

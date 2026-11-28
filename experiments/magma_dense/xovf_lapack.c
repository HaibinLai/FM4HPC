/* Prediction (MagmaDense.x_overflow_accepted): A, b inside the FP32 range, FP64 solution outside it.
   The FP32 solve overflows to Inf, the residual is -Inf or NaN, and the reject-form test
   "RNRM > XNRM*CTE" (Inf > Inf, or NaN > ...) is false, so the Inf solution is accepted with ITER = 0, INFO = 0.
   LAPACK dsgesv / dsposv have the same test as MAGMA zcgesv / zcposv.  Build: cc xovf_lapack.c -framework Accelerate */
#include <stdio.h>
#include <math.h>
#include <string.h>
extern void dsgesv_(int*,int*,double*,int*,int*,double*,int*,double*,int*,double*,float*,int*,int*);
extern void dsposv_(char*,int*,int*,double*,int*,double*,int*,double*,int*,double*,float*,int*,int*);
static void run(const char *name, int n, double a, double b, int spd) {
  double A[16]={0}, B[4], X[4], W[16]; float SW[32]; int ipiv[4], iter=-99, info=-99, nrhs=1, ld=n;
  for (int i=0;i<n;i++){ A[i+i*n]=a; B[i]=b; }
  if (n==2) { A[1]=0; A[2]=0; }            /* diagonal; off-diagonals are zero (0*Inf = NaN in the residual) */
  if (spd) { char u='L'; dsposv_(&u,&n,&nrhs,A,&ld,B,&ld,X,&ld,W,SW,&iter,&info); }
  else dsgesv_(&n,&nrhs,A,&ld,ipiv,B,&ld,X,&ld,W,SW,&iter,&info);
  printf("%-8s n=%d a=%a b=%a  ITER=%d INFO=%d  x[0]=%g  exact=%g\n", name, n, a, b, iter, info, X[0], b/a);
}
int main(void) {
  double a = ldexp(1.0,-100), b = ldexp(1.0,28);        /* exact solution 2^128, just above FLT_MAX */
  run("dsgesv", 1, a, b, 0); run("dsgesv", 2, a, b, 0);
  run("dsposv", 1, a, b, 1); run("dsposv", 2, a, b, 1);
  /* control: solution inside the FP32 range */
  run("dsgesv", 2, ldexp(1.0,-90), b, 0); run("dsposv", 2, ldexp(1.0,-90), b, 1);
  /* b outside the FP32 range: dlag2s on B must flag it (ITER=-2) in LAPACK; MAGMA zcgesv has no such check */
  run("dsgesv", 2, 1.0, ldexp(1.0,200), 0);
  return 0;
}

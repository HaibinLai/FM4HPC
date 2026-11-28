/* Prediction (MagmaDense.diverge_accepted): if the FP32-preconditioned refinement diverges and the iterate
   overflows before ITERMAX, the reject-form test (Inf > Inf*cte is false) accepts it and ITER > 0 ("success").
   Matrices: A = U diag(s) V^T, s log-spaced from 1 to 1/kappa, U, V random orthogonal (Householder products),
   so kappa(A) = kappa exactly in exact arithmetic; FP64 solves these to ~kappa*u.  Build with Accelerate. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
extern void dsgesv_(int*,int*,double*,int*,int*,double*,int*,double*,int*,double*,float*,int*,int*);
extern void dgesv_(int*,int*,double*,int*,int*,double*,int*,int*);
static double rnd(unsigned *s){ *s = *s*1103515245u+12345u; return ((*s>>8)&0xffffff)/(double)0x1000000 - 0.5; }
static void house(int n, double *M, unsigned *s){ /* M <- (I - 2vv^T/v^Tv) M for a random v, applied 3 times */
  double *v = malloc(n*sizeof *v);
  for (int t=0;t<3;t++){ double vv=0; for(int i=0;i<n;i++){v[i]=rnd(s); vv+=v[i]*v[i];}
    for(int j=0;j<n;j++){ double d=0; for(int i=0;i<n;i++) d+=v[i]*M[i+j*n]; d*=2/vv; for(int i=0;i<n;i++) M[i+j*n]-=d*v[i]; } }
  free(v); }
int main(void){
  int n=64, nrhs=1;
  double kap[] = {1e6, 1e8, 1e9, 1e10, 1e11, 1e12};
  for (int kk=0; kk<6; kk++) for (unsigned seed=1; seed<=3; seed++) {
    unsigned s=seed*7919u; double kappa=kap[kk];
    double *U=calloc(n*n,8), *V=calloc(n*n,8), *A=calloc(n*n,8), *A2=calloc(n*n,8);
    for(int i=0;i<n;i++){U[i+i*n]=1; V[i+i*n]=1;} house(n,U,&s); house(n,V,&s);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++){ double a=0; for(int k=0;k<n;k++){ double sk=pow(kappa,-(double)k/(n-1)); a+=U[i+k*n]*sk*V[j+k*n]; } A[i+j*n]=a; }
    double xt[64], b[64], X[64], b2[64], W[64*1]; float *SW=malloc((n*(n+1))*sizeof(float)); int ipiv[64], iter=-99, info=-99;
    for(int i=0;i<n;i++) xt[i]=rnd(&s);
    for(int i=0;i<n;i++){ double t=0; for(int j=0;j<n;j++) t+=A[i+j*n]*xt[j]; b[i]=t; b2[i]=t; }
    for(int i=0;i<n*n;i++) A2[i]=A[i];
    dsgesv_(&n,&nrhs,A,&n,ipiv,b,&n,X,&n,W,SW,&iter,&info);
    int nonfin=0; double e=0, m=0; for(int i=0;i<n;i++){ if(!isfinite(X[i])) nonfin++; else { double d=fabs(X[i]-xt[i]); if(d>e)e=d; } if(fabs(xt[i])>m)m=fabs(xt[i]); }
    int info2; dgesv_(&n,&nrhs,A2,&n,ipiv,b2,&n,&info2); double e2=0; for(int i=0;i<n;i++){double d=fabs(b2[i]-xt[i]); if(d>e2)e2=d;}
    printf("kappa=%.0e seed=%u  dsgesv: ITER=%4d INFO=%d nonfinite=%2d  relerr(finite)=%.1e   dgesv relerr=%.1e\n",
           kappa, seed, iter, info, nonfin, e/m, e2/m);
    free(U);free(V);free(A);free(A2);free(SW);
  }
  return 0; }

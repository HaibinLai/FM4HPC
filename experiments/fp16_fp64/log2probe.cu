#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
// Same expression as cuozblasSplitCKernel: const short tau = ceil(log2(fabs(x)))
__global__ void k(const double* x, int* tau, double* lg, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ double v=log2(fabs(x[i])); lg[i]=v; short t=ceil(log2(fabs(x[i]))); tau[i]=t; } }
int main(){
  const int N=2098; double h[N]; int idx=0;
  for(int e=-1074;e<=1023;e++) h[idx++]=ldexp(1.0,e);
  double *dx,*dl; int *dt; cudaMalloc((void**)&dx,N*8); cudaMalloc((void**)&dl,N*8); cudaMalloc((void**)&dt,N*4);
  cudaMemcpy(dx,h,N*8,cudaMemcpyHostToDevice); k<<<(N+127)/128,128>>>(dx,dt,dl,N); cudaDeviceSynchronize();
  int t[N]; double l[N]; cudaMemcpy(t,dt,N*4,cudaMemcpyDeviceToHost); cudaMemcpy(l,dl,N*8,cudaMemcpyDeviceToHost);
  int bad=0; for(int i=0;i<N;i++){ int e=-1074+i; if(t[i]!=e){ bad++; printf("BAD %d\n",e); } }
  printf("exact powers of two with wrong ceil(log2): %d of %d\n",bad,N);
}

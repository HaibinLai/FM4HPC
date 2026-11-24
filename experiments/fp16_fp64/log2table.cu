// Dump GPU `short tau = ceil(log2(fabs(x)))` (the library's expression) for x = 2^e shifted by k ulps, k in [-8, 8].
// Output lines: "T e k tau". Numerical debugging aid for the model's GPU-style tau table.
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
__global__ void kern(const double* x, int* tau, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { short t = ceil(log2(fabs(x[i]))); tau[i] = t; } }
int main() {
  std::vector<double> xs; std::vector<int> es, ks;
  for (int e = -1070; e <= 1022; e++) for (int k = -8; k <= 8; k++) {
    double x = ldexp(1.0, e); for (int s = 0; s < abs(k); s++) x = nextafter(x, k > 0 ? INFINITY : 0.0);
    if (x > 0 && std::isfinite(x)) { xs.push_back(x); es.push_back(e); ks.push_back(k); } }
  int n = xs.size(); double* dx; int* dt; cudaMalloc((void**)&dx, n * 8); cudaMalloc((void**)&dt, n * 4);
  cudaMemcpy(dx, xs.data(), n * 8, cudaMemcpyHostToDevice); kern<<<(n + 255) / 256, 256>>>(dx, dt, n); cudaDeviceSynchronize();
  std::vector<int> t(n); cudaMemcpy(t.data(), dt, n * 4, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) std::printf("T %d %d %d\n", es[i], ks[i], t[i]);
  return 0;
}

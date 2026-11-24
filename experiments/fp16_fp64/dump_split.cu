// Debug harness: call the ORIGINAL library's cuozblasSplit<double,half> exactly as cuozblasRgemm does
// (A transposed then split by column with dim=K; B split by column) and dump slices + exponents.
// Usage: dump_split input.bin out_prefix   -> out_prefix.A.split, out_prefix.B.split
// File layout: int32 ns, ncols, k; int16 exp[ns*ncols] (s*ncols+col); float32 slice[ns*ncols*k] ((s*ncols+col)*k+i)
// Slices are FP16 values widened exactly to float32. Numerical debugging aid only.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuozblas.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

template <typename TYPE1, typename TYPE2> int32_t cuozblasSplit (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const TYPE1 *devInput, const int32_t ldi, TYPE1 *devOutput, const int32_t ldo, TYPE2 *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, TYPE1 *devMax);

#define CK(x) do { auto e=(x); if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e)); std::exit(2);} } while(0)

static int dump(cuozblasHandle_t* oh, const std::vector<double>& mat, int k, int cols, const std::string& path) {
  const int maxS = oh->nSplitMax;
  double *din, *dout, *dmax; half* dsplit; short* dexp;
  CK(cudaMalloc((void**)&din, mat.size()*8)); CK(cudaMalloc((void**)&dout, mat.size()*8));
  CK(cudaMalloc((void**)&dmax, size_t(cols)*8));
  CK(cudaMalloc((void**)&dsplit, size_t(k)*cols*maxS*sizeof(half)));
  CK(cudaMalloc((void**)&dexp, size_t(cols)*maxS*sizeof(short)));
  CK(cudaMemset(dsplit, 0, size_t(k)*cols*maxS*sizeof(half))); CK(cudaMemset(dexp, 0, size_t(cols)*maxS*sizeof(short)));
  CK(cudaMemcpy(din, mat.data(), mat.size()*8, cudaMemcpyHostToDevice));
  int ns = cuozblasSplit<double,half>(oh, 'c', k, cols, din, k, dout, k, dsplit, k, dexp, cols, dmax);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<half> hs(size_t(k)*cols*maxS); std::vector<short> he(size_t(cols)*maxS);
  CK(cudaMemcpy(hs.data(), dsplit, hs.size()*sizeof(half), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(he.data(), dexp, he.size()*sizeof(short), cudaMemcpyDeviceToHost));
  std::ofstream f(path, std::ios::binary);
  int32_t hdr[3] = {ns, cols, k}; f.write((char*)hdr, sizeof(hdr));
  f.write((char*)he.data(), size_t(ns)*cols*sizeof(short));
  std::vector<float> fl(size_t(ns)*cols*k);
  for (size_t i = 0; i < fl.size(); i++) fl[i] = __half2float(hs[i]);
  f.write((char*)fl.data(), fl.size()*sizeof(float));
  cudaFree(din); cudaFree(dout); cudaFree(dmax); cudaFree(dsplit); cudaFree(dexp);
  return ns;
}

int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: dump_split input.bin out_prefix\n"); return 1; }
  std::ifstream input(argv[1], std::ios::binary); uint32_t d[3] = {};
  input.read((char*)d, sizeof(d));
  const int m = d[0], n = d[1], k = d[2];
  std::vector<double> a(size_t(m)*k), b(size_t(k)*n);
  input.read((char*)a.data(), a.size()*8); input.read((char*)b.data(), b.size()*8);
  if (!input) { std::fprintf(stderr, "truncated input\n"); return 1; }
  CK(cudaSetDevice(0));
  cuozblasHandle_t oh{}; cuozblasCreate(&oh, 128ULL*1024*1024);
  oh.nSplitMax = 12; oh.reproMode = 1; oh.fastMode = 0; oh.sumMode = 2; oh.splitEpsMode = 0; oh.useBatchedGemmFlag = 0;
  // A (m x k, col-major) -> transposed k x m, as blasRomatcopy('t') does in cuozblasRgemm.
  std::vector<double> at(size_t(k)*m);
  for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) at[j + size_t(k)*i] = a[i + size_t(m)*j];
  int nsA = dump(&oh, at, k, m, std::string(argv[2]) + ".A.split");
  int nsB = dump(&oh, b, k, n, std::string(argv[2]) + ".B.split");
  std::printf("{\"splits_A\":%d,\"splits_B\":%d}\n", nsA, nsB);
  oh.useBatchedGemmFlag = 1; cuozblasDestroy(&oh);
  return 0;
}

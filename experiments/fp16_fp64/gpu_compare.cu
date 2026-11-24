// Prepared, NOT compiled/run here. Direct original OzBLAS D_H vs cuBLAS DGEMM.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuozblas.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>

#define CUDA_OK(x) do { auto e=(x); if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e)); std::exit(2); }} while(0)
#define BLAS_OK(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS){ std::fprintf(stderr,"cuBLAS %s: %d\n",#x,int(e)); std::exit(3); }} while(0)

int main(int argc,char**argv){
  if(argc!=3){std::fprintf(stderr,"usage: gpu_compare input.bin output.bin\n");return 1;}
  std::ifstream input(argv[1],std::ios::binary); uint32_t d[3]={};
  input.read(reinterpret_cast<char*>(d),sizeof(d));
  const int m=d[0],n=d[1],k=d[2];
  if(!input||m<2||n<2||m>128||n>128||k<1||k>4096){std::fprintf(stderr,"unsupported fixture dimensions\n");return 1;}
  std::vector<double>a(size_t(m)*k),b(size_t(k)*n),ref(size_t(m)*n),out(ref.size());
  input.read(reinterpret_cast<char*>(a.data()),a.size()*8);
  input.read(reinterpret_cast<char*>(b.data()),b.size()*8);
  if(!input){std::fprintf(stderr,"truncated input\n");return 1;}
  for(double v:a) if(!std::isfinite(v)) return 1;
  for(double v:b) if(!std::isfinite(v)) return 1;
  // Logical device zero respects scheduler-provided CUDA_VISIBLE_DEVICES.
  CUDA_OK(cudaSetDevice(0)); cudaDeviceProp prop{}; CUDA_OK(cudaGetDeviceProperties(&prop,0));
  if(prop.major<7){std::fprintf(stderr,"requires Volta or newer\n");return 1;}
  int runtime,driver; CUDA_OK(cudaRuntimeGetVersion(&runtime)); CUDA_OK(cudaDriverGetVersion(&driver));
  double *da=nullptr,*db=nullptr,*dr=nullptr,*dy=nullptr;
  CUDA_OK(cudaMalloc((void**)&da,a.size()*8)); CUDA_OK(cudaMalloc((void**)&db,b.size()*8));
  CUDA_OK(cudaMalloc((void**)&dr,ref.size()*8)); CUDA_OK(cudaMalloc((void**)&dy,out.size()*8));
  CUDA_OK(cudaMemcpy(da,a.data(),a.size()*8,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db,b.data(),b.size()*8,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(dr,0,ref.size()*8)); CUDA_OK(cudaMemset(dy,0,out.size()*8));
  cublasHandle_t ch; BLAS_OK(cublasCreate(&ch));
  int bv; BLAS_OK(cublasGetVersion(ch,&bv));
  BLAS_OK(cublasSetMathMode(ch,CUBLAS_DEFAULT_MATH));
  const double one=1.,zero=0.;
  BLAS_OK(cublasDgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&one,da,m,db,k,&zero,dr,m));
  CUDA_OK(cudaDeviceSynchronize());
  cuozblasHandle_t oh{}; cuozblasCreate(&oh,128ULL*1024*1024);
  CUDA_OK(cudaGetLastError());
  oh.nSplitMax=12; oh.reproMode=1; oh.fastMode=0; oh.sumMode=2;
  oh.splitEpsMode=0; oh.useBatchedGemmFlag=0;
  const int status=cuozblasRgemm<double,half,float>(&oh,'N','N',m,n,k,one,da,m,db,k,zero,dy,m);
  CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  if(status!=0||oh.nSplitA<1||oh.nSplitB<1){std::fprintf(stderr,"OzBLAS did not complete splitting\n");return 4;}
  CUDA_OK(cudaMemcpy(ref.data(),dr,ref.size()*8,cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(out.data(),dy,out.size()*8,cudaMemcpyDeviceToHost));
  std::ofstream f(argv[2],std::ios::binary);
  f.write(reinterpret_cast<char*>(d),sizeof(d));
  f.write(reinterpret_cast<char*>(ref.data()),ref.size()*8);
  f.write(reinterpret_cast<char*>(out.data()),out.size()*8);
  f.close(); if(!f){std::fprintf(stderr,"output write failed\n");return 5;}
  std::printf("{\"device\":\"%s\",\"sm\":%d,\"cuda_runtime\":%d,\"cuda_driver\":%d,\"cublas\":%d,\"splits_A\":%.0f,\"splits_B\":%.0f,\"partial_gemms\":%.0f,\"sumMode\":2,\"backend\":\"double_half_float\",\"tensor_core_execution_confirmed\":false}\n",prop.name,prop.major*10+prop.minor,runtime,driver,bv,oh.nSplitA,oh.nSplitB,oh.nSplitC);
  // Create allocated batch buffers before we selected the non-batched path.
  oh.useBatchedGemmFlag=1; cuozblasDestroy(&oh);
  BLAS_OK(cublasDestroy(ch));
  CUDA_OK(cudaFree(da));CUDA_OK(cudaFree(db));CUDA_OK(cudaFree(dr));CUDA_OK(cudaFree(dy));
  return 0;
}

#!/bin/bash
# Compile and run the GPU probes on GPU 0 only. Work dir /ssd/haibin/magma_gpu.
cd /ssd/haibin/magma_gpu
M=$PWD/magma-2.10.0
export CUDA_VISIBLE_DEVICES=0
LIBS="-L$M/lib -lmagma_sparse -lmagma -L/usr/lib/x86_64-linux-gnu -lcublas -lcusparse -lcudart -l:liblapack.so.3 -l:libblas.so.3 -fopenmp -lstdc++ -lm"
INC="-I$M/include -I$M/sparse/include -I$M/control -DADD_"
g++ -O1 -std=c++11 $INC dense_probe.cpp -o dense_probe $LIBS 2> compile_dense.log || { echo COMPILE_FAIL_DENSE; tail -20 compile_dense.log; }
g++ -O1 -std=c++11 $INC sparse_probe.cpp -o sparse_probe $LIBS 2> compile_sparse.log || { echo COMPILE_FAIL_SPARSE; tail -20 compile_sparse.log; }
{
  echo "# MAGMA 2.10.0 (tarball sha256 $(cut -c1-16 tarball.sha256)...), CUDA $(nvcc --version | tail -1), GPU 0: $(nvidia-smi -i 0 --query-gpu=name,driver_version --format=csv,noheader), $(date -u)"
  ./dense_probe rbt64
  ./dense_probe dense
} > dense_results.txt 2>&1
./sparse_probe > sparse_results.txt 2>&1
compute-sanitizer --tool memcheck ./dense_probe nrhs2 > nrhs2_memcheck.txt 2>&1
./dense_probe nrhs2 > nrhs2_plain.txt 2>&1
echo RUN_DONE

#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib OMP_NUM_THREADS=4
g++ -O2 -std=c++11 round16_probe.cpp -I$M/include -I$M/sparse/include -DADD_ -L$M/lib -lmagma_sparse -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -o round16_probe 2>&1 | grep -i error
for m in mdotc csr_sort csrcoo_t mbdiagdom bitonic vget_dev cuspaxpy cuspmm csr_sort_gpu ge3pt ellrt jaccard lobpcg_shift; do echo "######## $m"; MALLOC_CHECK_=3 timeout 120 ./round16_probe $m 2>&1 | grep -v "^%\|^$" | tail -6; echo "exit=${PIPESTATUS[0]}"; done
echo ROUND16_DONE

#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
g++ -O2 -std=c++11 round15_probe.cpp -I$M/include -I$M/sparse/include -DADD_ -L$M/lib -lmagma_sparse -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -o round15_probe 2>&1 | grep -i error
for m in csrmv_trans elltmv mdotc mdot_shfl mgecsrmv sellp_multi vread csrget compact compressor frob diagdom zsort; do echo "######## $m"; MALLOC_CHECK_=3 timeout 120 ./round15_probe $m 2>&1 | grep -v "^%\|^$" | tail -5; echo "exit=${PIPESTATUS[0]} (124 = timeout)"; done
echo ROUND15_DONE

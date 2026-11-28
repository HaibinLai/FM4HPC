#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
g++ -O2 -std=c++11 -pthread round14_probe.cpp -I$M/include -I$M/sparse/include -DADD_ -L$M/lib -lmagma_sparse -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -lcuda -ldl -o round14_probe 2>&1 | grep -i error
echo "######## sparse (MTX_ONLY)"; MTX_ONLY=1 MALLOC_CHECK_=3 timeout 300 ./round14_probe sparse 2>&1 | tail -3; echo "exit=${PIPESTATUS[0]}"
echo ROUND14B_DONE

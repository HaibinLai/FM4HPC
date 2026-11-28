#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
g++ -O2 -std=c++11 round13_probe.cpp -I$M/include -I$M/sparse/include -DADD_ -L$M/lib -lmagma_sparse -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -o round13_probe 2>&1 | grep -i error
for m in f7 f12 f13 f14 f15 sparse; do echo "######## $m"; MALLOC_CHECK_=3 timeout 900 ./round13_probe $m 2>&1 | grep -v "^$" | tail -8; echo "exit=${PIPESTATUS[0]}"; done
echo ROUND13_DONE

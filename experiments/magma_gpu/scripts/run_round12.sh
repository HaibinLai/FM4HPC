#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
g++ -O2 -std=c++11 round12_probe.cpp -I$M/include -DADD_ -L$M/lib -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -o round12_probe 2>&1 | grep -i error
for m in f1 f2 f3 f4 f5 f6 f8 f9 f10 f11; do echo "######## $m"; timeout 900 ./round12_probe $m 2>&1 | grep -v "^$" | tail -6; echo "exit=${PIPESTATUS[0]}"; done
echo ROUND12_DONE

#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
g++ -O2 -std=c++11 round20_probe.cpp -I$M/include -DADD_ -L$M/lib -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -o round20_probe 2>&1 | grep -iE "error|undefined" | head
for m in f1 f7 f3 f4 f9 f5; do echo "######## $m"; timeout 1200 ./round20_probe $m 2>&1 | grep -v "^$" | tail -10; echo "exit=${PIPESTATUS[0]}"; done
echo ROUND20_DONE

#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
echo "GPU 0 used before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"
g++ -O2 -std=c++11 round23_probe.cpp -I$M/include -I$M/sparse/include -DADD_ -L$M/lib -lmagma_sparse -lmagma -lcublas -lcusparse -lcudart -o round23_probe 2>&1 | grep -iE "error|undefined" | head
for m in heinertia; do echo "######## $m"; MALLOC_CHECK_=3 timeout 120 ./round23_probe $m 2>&1 | grep -v "^%" | tail -4; echo "exit=${PIPESTATUS[0]}"; done
echo ROUND23_DONE

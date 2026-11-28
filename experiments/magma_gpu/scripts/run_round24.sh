#!/bin/bash
# Round 24: build, then n = 4000 (control), 38000 (workspace size wraps negative), 53600 (wraps to a small positive size).
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib OMP_NUM_THREADS=32
echo "GPU 0 used before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0); host available: $(free -g | awk '/Mem/ {print $7}') GB"
g++ -O2 -std=c++11 round24_probe.cpp -I$M/include -DADD_ -L$M/lib -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -o round24_probe 2>&1 | grep -iE "error" | head
for n in 4000 38000 53600; do echo "######## n=$n"; timeout 5400 ./round24_probe $n 2>&1 | tail -4; echo "exit=${PIPESTATUS[0]}"; done
echo ROUND24_DONE

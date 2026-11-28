#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); echo "GPU 0 used before: ${used} MiB"
gcc -shared -fPIC -O1 fail_dsteqr.c -o fail_dsteqr.so -ldl
g++ -O2 -std=c++11 round21_probe.cpp -I$M/include -DADD_ -L$M/lib -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -o round21_probe 2>&1 | grep -iE "error" | head
echo "######## dsteqr calls in zstedx (no failure)"; LD_PRELOAD=$W/fail_dsteqr.so FI_DSTEQR_K=0 ./round21_probe stedx 2>&1 | grep -v "^%" | tail -40
n1=$(LD_PRELOAD=$W/fail_dsteqr.so FI_DSTEQR_K=0 ./round21_probe stedx 2>&1 | grep -c "\[dsteqr\]")
for k in 1 $(( (n1/2) + 1 )) $n1; do echo "######## forced dsteqr failure at call $k of $n1"; LD_PRELOAD=$W/fail_dsteqr.so FI_DSTEQR_K=$k timeout 300 ./round21_probe stedx 2>&1 | grep -E "forced|zstedx|DLASCL" | tail -3; done
echo "######## dsidi"; timeout 120 ./round21_probe dsidi 2>&1 | tail -6
echo "######## dsort"; timeout 120 ./round21_probe dsort 2>&1 | tail -3
echo "######## gesvj chunks"; timeout 3000 ./round21_probe gesvj_chunks 2>&1 | tail -4; echo "exit=${PIPESTATUS[0]}"
echo ROUND21_DONE

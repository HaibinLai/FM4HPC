#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
gcc -shared -fPIC -O1 fail_malloc.c -o fail_malloc.so -ldl 2>/dev/null
g++ -O2 -std=c++11 round19_probe.cpp -I$M/include -I$M/sparse/include -DADD_ -L$M/lib -lmagma_sparse -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -ldl -o round19_probe 2>&1 | grep -iE "error" | head
for m in lascl_2x2 residual residual0 csrsplit; do echo "######## $m"; MALLOC_CHECK_=3 timeout 120 ./round19_probe $m 2>&1 | grep -v "^%\|^$" | tail -5; echo "exit=${PIPESTATUS[0]}"; done
echo "######## ungbr fault injection"; LD_PRELOAD=$W/fail_malloc.so timeout 300 ./round19_probe ungbr 0 2>&1 | grep "k="
for k in 1 2 3; do LD_PRELOAD=$W/fail_malloc.so FI_BT=1 timeout 300 ./round19_probe ungbr $k 2>&1 | grep -E "k=|\(magma_z" | sed 's/.*(\(magma_z[a-z0-9_]*\).*/  site: \1/' | head -6; done
echo ROUND19_DONE

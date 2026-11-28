#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
echo "GPU 0 used before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"
gcc -shared -fPIC -O1 fail_malloc_cpu.c -o fail_malloc_cpu.so -ldl
g++ -O2 -std=c++11 round22_probe.cpp -I$M/include -I$M/sparse/include -DADD_ -L$M/lib -lmagma_sparse -lmagma -lcublas -lcusparse -lcudart -ldl -o round22_probe 2>&1 | grep -iE "error|undefined" | head
echo "######## symbols"; for s in magma_zdiagcheck_cpu magma_zdiagcheck magmablas_zdiinertia magma_zmlumerge; do echo "$s: libmagma $(nm -D $M/lib/libmagma.so | grep -cw $s), libmagma_sparse $(nm -D $M/lib/libmagma_sparse.so | grep -cw $s)"; done
echo "######## diinertia"; timeout 120 ./round22_probe diinertia 2>&1 | tail -2
echo "######## compressor"; MALLOC_CHECK_=3 timeout 120 ./round22_probe compressor 2>&1 | grep -v "^%" | tail -2
echo "######## lumerge host-allocation fault injection"; c=$(LD_PRELOAD=$W/fail_malloc_cpu.so FI_CPU_K=0 ./round22_probe lumerge 0 2>&1 | sed -n 's/.*inside: \([0-9]*\)).*/\1/p'); echo "host allocations: $c"
for k in $(seq 1 ${c:-0}); do out=$(LD_PRELOAD=$W/fail_malloc_cpu.so FI_CPU_K=$k MALLOC_CHECK_=3 timeout 120 ./round22_probe lumerge $k 2>&1); rc=$?; echo "$out" | grep -E "zmlumerge|freed|double free|corrupt|invalid" | head -3; echo "  exit=$rc"; done
echo ROUND22_DONE

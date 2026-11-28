#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
gcc -shared -fPIC -O1 fail_malloc.c -o fail_malloc.so -ldl 2>/dev/null
g++ -O2 -std=c++11 round18_probe.cpp -I$M/include -DADD_ -L$M/lib -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -ldl -o round18_probe 2>&1 | grep -iE "error" | head
echo "######## header-style call"; MALLOC_CHECK_=3 timeout 300 ./round18_probe header 2>&1 | tail -3; echo "exit=${PIPESTATUS[0]}"
line=$(LD_PRELOAD=$W/fail_malloc.so timeout 300 ./round18_probe inject 0 2>&1 | grep "k=  0"); echo "$line"
c=$(echo "$line" | sed -n 's/.*nalloc= *\([0-9]*\).*/\1/p')
for k in $(seq 1 ${c:-0}); do out=$(LD_PRELOAD=$W/fail_malloc.so FI_BT=1 MALLOC_CHECK_=3 timeout 300 ./round18_probe inject $k 2>&1); rc=$?
  site=$(echo "$out" | grep -oE "\(magma_[a-z0-9_]+" | tr -d "(" | grep -v "^magma_malloc" | head -2 | tr '\n' '<')
  l=$(echo "$out" | grep "k="); if [ -n "$l" ]; then echo "$l  [failing site: $site]"; else echo "unglq k=$k CRASH/EXIT rc=$rc [site: $site]: $(echo "$out" | tail -1 | cut -c1-100)"; fi; done
echo ROUND18_DONE

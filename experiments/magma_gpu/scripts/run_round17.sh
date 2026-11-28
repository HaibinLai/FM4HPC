#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
gcc -shared -fPIC -O1 fail_malloc.c -o fail_malloc.so -ldl 2>/dev/null
g++ -O2 -std=c++11 round17_probe.cpp -I$M/include -I$M/sparse/include -DADD_ -L$M/lib -lmagma_sparse -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -ldl -o round17_probe 2>&1 | grep -iE "error|undefined" | head
for m in syncfree2 syncfree4096 csrget_gpu orderstat rowentries trisolve; do echo "######## $m"; MALLOC_CHECK_=3 timeout 120 ./round17_probe $m 2>&1 | grep -v "^%\|^$" | tail -5; echo "exit=${PIPESTATUS[0]}"; done
for h in ungqr ungqr_gpu unmrq unmqr2_gpu unmql2_gpu gerbt_gpu; do
  line=$(LD_PRELOAD=$W/fail_malloc.so timeout 600 ./round17_probe $h 0 2>&1 | grep "k=  0"); echo "$line"
  c=$(echo "$line" | sed -n 's/.*nalloc= *\([0-9]*\).*/\1/p'); [ -z "$c" ] && continue
  for k in $(seq 1 $c); do out=$(LD_PRELOAD=$W/fail_malloc.so FI_BT=1 MALLOC_CHECK_=3 timeout 600 ./round17_probe $h $k 2>&1); rc=$?
    site=$(echo "$out" | grep -oE "\(magma_[a-z0-9_]+" | tr -d "(" | grep -v "^magma_malloc" | head -2 | tr '\n' '<')
    l=$(echo "$out" | grep "k="); if [ -n "$l" ]; then echo "$l  [failing site: $site]"; else echo "$h k=$k CRASH/EXIT rc=$rc [site: $site]: $(echo "$out" | tail -1 | cut -c1-100)"; fi; done
done
echo ROUND17_DONE

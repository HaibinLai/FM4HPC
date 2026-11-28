#!/bin/bash
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib
gcc -shared -fPIC -O1 fail_malloc.c -o fail_malloc.so -ldl 2>/dev/null
g++ -O2 -std=c++11 -pthread round14_probe.cpp -I$M/include -I$M/sparse/include -DADD_ -L$M/lib -lmagma_sparse -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -lcuda -ldl -o round14_probe 2>&1 | grep -i error
for m in lascl2 potf2 lag2c heevj sparse; do echo "######## $m"; MALLOC_CHECK_=3 timeout 1800 ./round14_probe $m 2>&1 | grep -v "^$" | tail -8; echo "exit=${PIPESTATUS[0]}"; done
for h in unmqr unmql unmlq ungqr2 trevc3 trevc3_mt; do
  line=$(LD_PRELOAD=$W/fail_malloc.so timeout 600 ./round14_probe $h 0 2>&1 | grep "k=  0"); echo "$line"
  c=$(echo "$line" | sed -n 's/.*nalloc= *\([0-9]*\).*/\1/p'); [ -z "$c" ] && continue
  for k in $(seq 1 $c); do out=$(LD_PRELOAD=$W/fail_malloc.so FI_BT=1 MALLOC_CHECK_=3 timeout 600 ./round14_probe $h $k 2>&1); rc=$?
    site=$(echo "$out" | grep -oE "\(magma_[a-z0-9_]+" | tr -d "(" | grep -v "^magma_malloc" | head -2 | tr '\n' '<')
    l=$(echo "$out" | grep "k="); if [ -n "$l" ]; then echo "$l  [failing site: $site]"; else echo "$h k=$k CRASH/EXIT rc=$rc [site: $site]: $(echo "$out" | tail -1 | cut -c1-100)"; fi; done
done
echo ROUND14_DONE

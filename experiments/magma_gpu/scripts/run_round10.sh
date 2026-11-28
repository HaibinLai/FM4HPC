#!/bin/bash
# Fault-injection sweep: for each driver, count its allocations (k = 0), then fail each one in turn (k = 1..count).
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W
export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$M/lib MAGMA_NUM_GPUS=1
gcc -shared -fPIC -O1 fail_malloc.c -o fail_malloc.so -ldl
g++ -O2 -std=c++11 round10_probe.cpp -I$M/include -DADD_ -L$M/lib -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -ldl -o round10_probe 2>&1 | grep -i error
for d in zgesv zposv zheevd zheevd_m zhegvd zhegvd_m zheevdx_2stage zheevdx_2stage_m zgeev zgeev_m zgesvd; do
  line=$(LD_PRELOAD=$W/fail_malloc.so timeout 600 ./round10_probe $d 0 2>&1 | grep "k=  0")
  echo "$line"
  c=$(echo "$line" | sed -n 's/.*nalloc= *\([0-9]*\).*/\1/p'); [ -z "$c" ] && continue
  for k in $(seq 1 $c); do
    out=$(LD_PRELOAD=$W/fail_malloc.so MALLOC_CHECK_=3 timeout 600 ./round10_probe $d $k 2>&1); rc=$?
    l=$(echo "$out" | grep "k=")
    if [ -n "$l" ]; then echo "$l"; else echo "$d k=$k CRASH/EXIT rc=$rc: $(echo "$out" | grep -v '^$' | tail -1 | cut -c1-120)"; fi
  done
done
echo "######## zgeqrf_ooc lwork = n (MALLOC_CHECK_=3)"; MALLOC_CHECK_=3 timeout 900 ./round10_probe ooc_lwork 2>&1 | tail -4; echo "exit=${PIPESTATUS[0]}"
echo "######## zgeqrf_ooc with ~20 MB free (NB = 0?)"; timeout 180 ./round10_probe ooc_nb0 2>&1 | tail -4; echo "exit=${PIPESTATUS[0]} (124 = timeout)"
echo "######## ngpu = 2 with one visible device"; MAGMA_NUM_GPUS=2 timeout 900 ./round10_probe ngpu2 2>&1 | tail -6; echo "exit=${PIPESTATUS[0]}"
echo ROUND10_DONE

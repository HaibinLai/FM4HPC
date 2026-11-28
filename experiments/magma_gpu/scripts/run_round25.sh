#!/bin/bash
# Round 25: ngpu = 2 on GPUs 0 and 1.  Waits until both have no compute process and < 500 MiB used for 3 consecutive minutes.
W=/ssd/haibin/magma_gpu; M=$W/magma-2.10.0; cd $W; ok=0
while [ $ok -lt 3 ]; do busy=0
  for g in 0 1; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g | wc -l)
    if [ "$used" -ge 500 ] || [ "$procs" -ne 0 ]; then busy=1; fi; done
  if [ $busy -eq 0 ]; then ok=$((ok+1)); else ok=0; fi; echo "$(date +%H:%M:%S) busy=$busy ok=$ok" >> $W/wait25.log; sleep 60; done
export CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$M/lib OMP_NUM_THREADS=32
echo "GPUs before: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 0,1 | tr '\n' ' ')"
g++ -O2 -std=c++11 round25_probe.cpp -I$M/include -DADD_ -L$M/lib -lmagma -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -o round25_probe 2>&1 | grep -iE "error" | head
for m in potrf_m getrf_m geqrf_m trsm_m potrf_mgpu geev_m eig; do echo "######## $m"; timeout 3600 ./round25_probe $m 2>&1 | grep -v "^$" | tail -16; echo "exit=${PIPESTATUS[0]}"; done
echo ROUND25_DONE

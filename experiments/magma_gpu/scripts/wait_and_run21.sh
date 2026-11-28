#!/bin/bash
W=/ssd/haibin/magma_gpu; ok=0
while [ $ok -lt 3 ]; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)
  if [ "$used" -lt 500 ] && [ "$procs" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi; echo "$(date +%H:%M:%S) used=${used} procs=$procs ok=$ok" >> $W/wait.log; sleep 60; done
cd $W && bash build_magma.sh > build_outer.log 2>&1 && bash run_round21.sh > round21_results.txt 2>&1

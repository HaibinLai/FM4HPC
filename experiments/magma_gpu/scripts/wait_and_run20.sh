#!/bin/bash
# Wait until GPU 0 is idle (no compute process, < 500 MiB used) for 3 consecutive checks, then build MAGMA and run round 20.
W=/ssd/haibin/magma_gpu; ok=0
while [ $ok -lt 3 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)
  if [ "$used" -lt 500 ] && [ "$procs" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
  echo "$(date +%H:%M:%S) used=${used}MiB procs=$procs ok=$ok" >> $W/wait.log; sleep 60
done
echo "GPU 0 idle at $(date)" >> $W/wait.log
cd $W && bash build_magma.sh > build_outer.log 2>&1 && bash run_round20.sh > round20_results.txt 2>&1

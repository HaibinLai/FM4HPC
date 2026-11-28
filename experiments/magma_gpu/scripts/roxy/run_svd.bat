@echo off
call D:\lean4hpc\magma\run_probe.bat round27_sweep.exe svd > D:\lean4hpc\magma\sweep\svd.csv 2> D:\lean4hpc\magma\sweep\svd.err
echo SVD_DONE >> D:\lean4hpc\magma\sweep\progress.txt

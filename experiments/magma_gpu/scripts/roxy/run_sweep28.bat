@echo off
rem Round 28: structured-input sweep, one process per group; CSV per group in D:\lean4hpc\magma\sweep28
set M=D:\lean4hpc\magma
if not exist %M%\sweep28 mkdir %M%\sweep28
for %%G in (qrcp lu chol batched heig svd) do (
  echo START %%G >> %M%\sweep28\progress.txt
  call %M%\run_probe.bat round28_sweep.exe %%G > %M%\sweep28\%%G.csv 2> %M%\sweep28\%%G.err
  echo END %%G >> %M%\sweep28\progress.txt
)
echo SWEEP28_DONE >> %M%\sweep28\progress.txt

@echo off
rem Round 27: run every operator group of round27_sweep in its own process; CSV per group in D:\lean4hpc\magma\sweep
set M=D:\lean4hpc\magma
if not exist %M%\sweep mkdir %M%\sweep
for %%G in (blas lu chol qr solve inv batched reduce heig svd geev) do (
  echo START %%G %TIME% >> %M%\sweep\progress.txt
  call %M%\run_probe.bat round27_sweep.exe %%G > %M%\sweep\%%G.csv 2> %M%\sweep\%%G.err
  echo END %%G exit=%ERRORLEVEL% %TIME% >> %M%\sweep\progress.txt
)
echo SWEEP_DONE >> %M%\sweep\progress.txt

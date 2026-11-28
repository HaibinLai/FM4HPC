@echo off
rem Run a probe with the CUDA and OpenBLAS DLLs on PATH.  Usage: run_probe.bat name args...
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set PATH=%CU%\bin\x86_64;%CU%\bin;D:\lean4hpc\openblas\bin;%PATH%
D:\lean4hpc\magma\bin\%*

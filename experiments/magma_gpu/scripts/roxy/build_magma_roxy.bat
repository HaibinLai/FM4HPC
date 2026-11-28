@echo off
rem Build MAGMA 2.10.0 on roxy (RTX 5070 Ti, sm_120): MSVC 14.44 + nvcc 13.5 from pip wheels + OpenBLAS 0.3.34 (LP64).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:/lean4hpc/magma_env/Lib/site-packages/nvidia/cu13
set PATH=D:\lean4hpc\magma_env\Scripts;D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13\bin;D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13\bin\x86_64;%PATH%
cd /d D:\lean4hpc\magma
if not exist build mkdir build
cd build
cmake -G Ninja ..\magma-2.10.0 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DUSE_FORTRAN=OFF ^
  -DFORTRAN_CONVENTION=-DADD_ -DGPU_TARGET=sm_120 -DMAGMA_ENABLE_CUDA=ON ^
  -DCMAKE_CUDA_COMPILER=%CU%/bin/nvcc.exe -DCUDAToolkit_ROOT=%CU% ^
  -DLAPACK_LIBRARIES=D:/lean4hpc/openblas/lib/libopenblas.lib > ..\configure.log 2>&1
echo configure exit %ERRORLEVEL% >> ..\configure.log
ninja -j 8 magma magma_sparse > ..\build.log 2>&1
echo BUILD_EXIT %ERRORLEVEL% >> ..\build.log

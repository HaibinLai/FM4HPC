@echo off
rem round37_2stage linked with dgerbt_kernels.cu whose transpose-vector kernel no longer skips size-1 sub-butterflies: bin\round37_2stage_rbtfix.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=-I%M%\build\include -I%M%\magma-2.10.0\include -I%M%\magma-2.10.0\control -I%M%\magma-2.10.0\magmablas -I%CU%\include
%CU%\bin\nvcc.exe -c -O3 -std=c++17 -arch=sm_120 -DADD_ -DMAGMA_NOAFFINITY -Xcompiler /MD %INC% %M%\probes\dgerbt_kernels_fixed.cu -o %M%\bin\dgerbt_kernels_fixed.obj
if errorlevel 1 ( echo NVCC_FAIL & exit /b 1 )
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /I%M%\build\include /I%M%\magma-2.10.0\include /I%CU%\include ^
   %M%\probes\round37_2stage.cpp %M%\bin\dgerbt_kernels_fixed.obj /Fo%M%\bin\ /Fe%M%\bin\round37_2stage_rbtfix.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX_BUILD_EXIT %ERRORLEVEL%

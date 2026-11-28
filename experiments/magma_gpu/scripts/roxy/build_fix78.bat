@echo off
rem round 78: round78_vbatched_blas linked with zhemm_vbatched_core.cu compiled against the fetch-fixed probes\hemmfix\hemm_template_device.cuh (round 35)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=-I%M%\build\include -I%M%\magma-2.10.0\include -I%M%\magma-2.10.0\control -I%M%\magma-2.10.0\magmablas -I%CU%\include
copy /y %M%\magma-2.10.0\magmablas\zhemm_vbatched_core.cu %M%\probes\hemmfix\zhemm_vbatched_core.cu >nul
%CU%\bin\nvcc.exe -c -O3 -std=c++17 -arch=sm_120 -DADD_ -DMAGMA_NOAFFINITY -Xcompiler /MD %INC% %M%\probes\hemmfix\zhemm_vbatched_core.cu -o %M%\bin\zhemm_vbatched_core_hemmfix.obj
if errorlevel 1 ( echo NVCC_FAIL & exit /b 1 )
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /I%M%\build\include /I%M%\magma-2.10.0\include /I%CU%\include ^
   %M%\probes\round78_vbatched_blas.cpp %M%\bin\zhemm_vbatched_core_hemmfix.obj /Fo%M%\bin\ /Fe%M%\bin\round78_vbatched_blas_hemmfix.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX78_BUILD_EXIT %ERRORLEVEL%

@echo off
rem round34_bblas linked with dsymm.cu compiled against the fetch-fixed hemm_template_device.cuh: bin\round34_bblas_hemmfix.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=-I%M%\build\include -I%M%\magma-2.10.0\include -I%M%\magma-2.10.0\control -I%M%\magma-2.10.0\magmablas -I%CU%\include
%CU%\bin\nvcc.exe -c -O3 -std=c++17 -arch=sm_120 -DADD_ -DMAGMA_NOAFFINITY -Xcompiler /MD %INC% %M%\probes\hemmfix\dsymm.cu -o %M%\bin\dsymm_hemmfix.obj
if errorlevel 1 ( echo NVCC_FAIL & exit /b 1 )
for %%f in (dsymm_batched_core dsymm_vbatched_core) do (
  %CU%\bin\nvcc.exe -c -O3 -std=c++17 -arch=sm_120 -DADD_ -DMAGMA_NOAFFINITY -Xcompiler /MD %INC% %M%\probes\hemmfix\%%f.cu -o %M%\bin\%%f_hemmfix.obj
  if errorlevel 1 ( echo NVCC_FAIL %%f & exit /b 1 )
)
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /I%M%\build\include /I%M%\magma-2.10.0\include /I%CU%\include ^
   %M%\probes\round34_bblas.cpp %M%\bin\dsymm_hemmfix.obj %M%\bin\dsymm_batched_core_hemmfix.obj %M%\bin\dsymm_vbatched_core_hemmfix.obj /Fo%M%\bin\ /Fe%M%\bin\round34_bblas_hemmfix.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX_BUILD_EXIT %ERRORLEVEL%

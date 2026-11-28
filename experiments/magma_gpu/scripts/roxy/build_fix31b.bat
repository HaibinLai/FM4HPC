@echo off
rem round31_race linked with the L1-bypassing sgetf2/dgetf2 native kernels (override the library's objects): bin\round31_race_ldcg.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=-I%M%\build\include -I%M%\magma-2.10.0\include -I%M%\magma-2.10.0\control -I%M%\magma-2.10.0\magmablas -I%CU%\include
for %%p in (s d) do (
  %CU%\bin\nvcc.exe -c -O3 -std=c++17 -arch=sm_120 -DADD_ -DMAGMA_NOAFFINITY -Xcompiler /MD %INC% %M%\probes\%%pgetf2_native_kernel_ldcg.cu -o %M%\bin\%%pgetf2_native_kernel_ldcg.obj
  if errorlevel 1 ( echo NVCC_FAIL %%p & exit /b 1 )
)
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /I%M%\build\include /I%M%\magma-2.10.0\include /I%CU%\include ^
   %M%\probes\round31_race.cpp %M%\bin\sgetf2_native_kernel_ldcg.obj %M%\bin\dgetf2_native_kernel_ldcg.obj /Fo%M%\bin\ /Fe%M%\bin\round31_race_ldcg.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
if errorlevel 1 ( echo FIX_BUILD_EXIT 1 & exit /b 1 )
rem the same kernels under the round-31 sweep: bin\round31_mixed_ldcg.exe
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /I%M%\build\include /I%M%\magma-2.10.0\include /I%CU%\include ^
   %M%\probes\round31_mixed.cpp %M%\bin\sgetf2_native_kernel_ldcg.obj %M%\bin\dgetf2_native_kernel_ldcg.obj /Fo%M%\bin\ /Fe%M%\bin\round31_mixed_ldcg.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX_BUILD_EXIT %ERRORLEVEL%

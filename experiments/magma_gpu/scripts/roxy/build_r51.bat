@echo off
rem round51: probe (bin\round51_stream.exe) and the same probe linked with the patched sync-free CSC trsm (consumer wait restored)
rem and the patched slag2h (device flag address, whole chunk of flags): bin\round51_stream_fix.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=-I%M%\build\include -I%M%\magma-2.10.0\include -I%M%\magma-2.10.0\sparse\include -I%M%\magma-2.10.0\control -I%M%\magma-2.10.0\sparse\control -I%M%\magma-2.10.0\sparse\blas -I%CU%\include
set CLF=/nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\sparse\include /I%CU%\include
set LIBS=/LIBPATH:%CU%\lib\x64 %M%\build\lib\magma_sparse.lib %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
cl %CLF% %M%\probes\round51_stream.cpp /Fo%M%\bin\round51_stream.obj /Fe%M%\bin\round51_stream.exe /link %LIBS%
if errorlevel 1 ( echo R51_BUILD_EXIT 1 & exit /b 1 )
%CU%\bin\nvcc.exe -c -O3 -std=c++17 -arch=sm_120 -DADD_ -DMAGMA_NOAFFINITY -Xcompiler /MD %INC% -I%M%\magma-2.10.0\magmablas %M%\probes\slag2h_fixed.cu -o %M%\bin\slag2h_fixed.obj
if errorlevel 1 ( echo NVCC_FAIL slag2h & exit /b 1 )
for %%p in (d z) do (
  %CU%\bin\nvcc.exe -c -O3 -std=c++17 -arch=sm_120 -DADD_ -DMAGMA_NOAFFINITY -Xcompiler /MD %INC% %M%\probes\%%pgecscsyncfreetrsm_fixed.cu -o %M%\bin\%%pgecscsyncfreetrsm_fixed.obj
  if errorlevel 1 ( echo NVCC_FAIL %%p & exit /b 1 )
)
cl %CLF% %M%\bin\round51_stream.obj %M%\bin\dgecscsyncfreetrsm_fixed.obj %M%\bin\zgecscsyncfreetrsm_fixed.obj %M%\bin\slag2h_fixed.obj /Fe%M%\bin\round51_stream_fix.exe /link %LIBS%
echo R51_BUILD_EXIT %ERRORLEVEL%

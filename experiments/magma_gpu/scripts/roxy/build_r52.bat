@echo off
rem round52: schedule-fuzzing harness + fuzzable copies of MAGMA multi-GPU sources (probes\mgpu_fuzz, made by gen_fuzz.py)
rem   bin\round52_mgpu_fuzz.exe      all fuzz_*.cpp / fuzz_*.cu copies
rem   bin\round52_mgpu_fuzz_fix.exe  the same with fixed_zgetrf_m.cpp.txt (M258 patch) instead of fuzz_zgetrf_m.cpp
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set F=%M%\probes\mgpu_fuzz
set INC=-I%M%\build\include -I%M%\magma-2.10.0\include -I%M%\magma-2.10.0\control -I%M%\magma-2.10.0\magmablas -I%F% -I%CU%\include
for %%c in (%F%\fuzz_*.cu) do (
  %CU%\bin\nvcc.exe -c -O3 -std=c++17 -arch=sm_120 -DADD_ -DMAGMA_NOAFFINITY -Xcompiler /MD %INC% %%c -o %M%\bin\%%~nc.obj
  if errorlevel 1 ( echo NVCC_FAIL %%c & exit /b 1 )
)
set CUOBJ=
for %%c in (%F%\fuzz_*.cu) do call set CUOBJ=%%CUOBJ%% %M%\bin\%%~nc.obj
set CLF=/nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /DMAGMA_HAVE_CUDA /I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\control /I%F% /I%CU%\include
set LIBS=/LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
cl %CLF% %M%\probes\round52_mgpu_fuzz.cpp %F%\fuzz_*.cpp %CUOBJ% /Fo%M%\bin\ /Fe%M%\bin\round52_mgpu_fuzz.exe /link %LIBS%
echo R52_BUILD_EXIT %ERRORLEVEL%
copy /y %F%\fixed_zgetrf_m.cpp.txt %M%\bin\fixed_zgetrf_m.cpp >nul
set FIXSRC=
for %%c in (%F%\fuzz_*.cpp) do if /i not "%%~nxc"=="fuzz_zgetrf_m.cpp" call set FIXSRC=%%FIXSRC%% %%c
cl %CLF% %M%\probes\round52_mgpu_fuzz.cpp %FIXSRC% %M%\bin\fixed_zgetrf_m.cpp %CUOBJ% /Fo%M%\bin\ /Fe%M%\bin\round52_mgpu_fuzz_fix.exe /link %LIBS%
echo R52_FIX_BUILD_EXIT %ERRORLEVEL%

@echo off
rem round52, pinned-host build with both patches: bin\round52_mgpu_fuzz_pinfix.exe = _fix sources with fixed_zgehrd_m.cpp.txt
rem (queue sync before zgehd2) instead of fuzz_zgehrd_m.cpp; objects in bin\pinfix\.  Pinned mode: R52_PIN=1.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set F=%M%\probes\mgpu_fuzz
if not exist %M%\bin\pinfix mkdir %M%\bin\pinfix
set CUOBJ=
for %%c in (%F%\fuzz_*.cu) do call set CUOBJ=%%CUOBJ%% %M%\bin\%%~nc.obj
set CLF=/nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /DMAGMA_HAVE_CUDA /I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\control /I%F% /I%CU%\include
set LIBS=/LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
copy /y %F%\fixed_zgetrf_m.cpp.txt %M%\bin\pinfix\fixed_zgetrf_m.cpp >nul
copy /y %F%\fixed_zgehrd_m.cpp.txt %M%\bin\pinfix\fixed_zgehrd_m.cpp >nul
set FIXSRC=
for %%c in (%F%\fuzz_*.cpp) do if /i not "%%~nxc"=="fuzz_zgetrf_m.cpp" if /i not "%%~nxc"=="fuzz_zgehrd_m.cpp" call set FIXSRC=%%FIXSRC%% %%c
cl %CLF% %M%\probes\round52_mgpu_fuzz.cpp %FIXSRC% %M%\bin\pinfix\fixed_zgetrf_m.cpp %M%\bin\pinfix\fixed_zgehrd_m.cpp %CUOBJ% /Fo%M%\bin\pinfix\ /Fe%M%\bin\round52_mgpu_fuzz_pinfix.exe /link %LIBS%
echo R52_PINFIX_BUILD_EXIT %ERRORLEVEL%

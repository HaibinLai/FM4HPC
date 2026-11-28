@echo off
rem round52, pinned-host build: bin\round52_mgpu_fuzz_pin.exe = the _fix sources (M258 patch), objects in bin\pin\ so a running
rem sweep's exe is not touched.  Reuses the fuzz_*.cu objects made by build_r52.bat.  The pinned mode itself is selected with R52_PIN=1.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set F=%M%\probes\mgpu_fuzz
if not exist %M%\bin\pin mkdir %M%\bin\pin
set CUOBJ=
for %%c in (%F%\fuzz_*.cu) do call set CUOBJ=%%CUOBJ%% %M%\bin\%%~nc.obj
set CLF=/nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /DMAGMA_HAVE_CUDA /I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\control /I%F% /I%CU%\include
set LIBS=/LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
copy /y %F%\fixed_zgetrf_m.cpp.txt %M%\bin\pin\fixed_zgetrf_m.cpp >nul
set FIXSRC=
for %%c in (%F%\fuzz_*.cpp) do if /i not "%%~nxc"=="fuzz_zgetrf_m.cpp" call set FIXSRC=%%FIXSRC%% %%c
cl %CLF% %M%\probes\round52_mgpu_fuzz.cpp %FIXSRC% %M%\bin\pin\fixed_zgetrf_m.cpp %CUOBJ% /Fo%M%\bin\pin\ /Fe%M%\bin\round52_mgpu_fuzz_pin.exe /link %LIBS%
echo R52_PIN_BUILD_EXIT %ERRORLEVEL%

@echo off
rem Compile one probe against the static MAGMA build.  Usage: build_probe.bat name   (probes\name.cpp -> bin\name.exe)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
if not exist %M%\bin mkdir %M%\bin
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\sparse\include /I%CU%\include ^
   %M%\probes\%1.cpp /Fo%M%\bin\%1.obj /Fe%M%\bin\%1.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib %2 cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo PROBE_BUILD_EXIT %ERRORLEVEL%

@echo off
rem round28_sweep linked with the patched zlaqps (patches 1 and 2; overrides the library's magma_zlaqps): bin\round28_fixed.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /DMAGMA_HAVE_CUDA /I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\control /I%CU%\include ^
   %M%\probes\round28_sweep.cpp %M%\probes\zlaqps_patched2.cpp /Fo%M%\bin\ /Fe%M%\bin\round28_fixed.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX_BUILD_EXIT %ERRORLEVEL%

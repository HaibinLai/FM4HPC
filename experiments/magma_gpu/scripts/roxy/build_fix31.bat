@echo off
rem round31_mixed linked with the instrumented dsgesv_gpu copy (overrides the library's magma_dsgesv_gpu): bin\round31_fixed.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /DMAGMA_HAVE_CUDA /I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\control /I%CU%\include ^
   %M%\probes\round31_mixed.cpp %M%\probes\dsgesv_patched.cpp /Fo%M%\bin\ /Fe%M%\bin\round31_fixed.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX_BUILD_EXIT %ERRORLEVEL%

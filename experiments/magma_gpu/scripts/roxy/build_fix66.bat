@echo off
rem round 66: round66_lobpcg linked with probes\zlobpcg_fixed.cpp (no break on eigenpair 0's residual alone) ahead of magma_sparse.lib
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=/I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\sparse\include /I%M%\magma-2.10.0\control /I%M%\magma-2.10.0\sparse\control /I%CU%\include
if not exist %M%\bin\fix66 mkdir %M%\bin\fix66
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY %INC% %M%\probes\round66_lobpcg.cpp %M%\probes\zlobpcg_fixed.cpp /Fo%M%\bin\fix66\ /Fe%M%\bin\round66_lobpcg_fix.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib %M%\build\lib\magma_sparse.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX66_BUILD_EXIT %ERRORLEVEL%

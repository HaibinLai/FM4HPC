@echo off
rem round 75: round75_idr linked with probes\zidr_fixed.cpp (rho = |t^H r| / (|t| |r|)) ahead of magma_sparse.lib
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=/I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\sparse\include /I%M%\magma-2.10.0\control /I%M%\magma-2.10.0\sparse\control /I%CU%\include
if not exist %M%\bin\fix75 mkdir %M%\bin\fix75
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY %INC% %M%\probes\round75_idr.cpp %M%\probes\zidr_fixed.cpp /Fo%M%\bin\fix75\ /Fe%M%\bin\round75_idr_fix.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib %M%\build\lib\magma_sparse.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX75_BUILD_EXIT %ERRORLEVEL%

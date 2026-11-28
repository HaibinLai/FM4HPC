@echo off
rem round 76: round76_precond_trans linked with probes\magma_z_precond_wrapper_fixed.cpp (ICC calls swapped in the MagmaTrans branches)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=/I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\sparse\include /I%M%\magma-2.10.0\control /I%M%\magma-2.10.0\sparse\control /I%CU%\include
if not exist %M%\bin\fix76 mkdir %M%\bin\fix76
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY %INC% %M%\probes\round76_precond_trans.cpp %M%\probes\magma_z_precond_wrapper_fixed.cpp /Fo%M%\bin\fix76\ /Fe%M%\bin\round76_precond_trans_fix.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib %M%\build\lib\magma_sparse.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX76_BUILD_EXIT %ERRORLEVEL%

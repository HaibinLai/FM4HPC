@echo off
rem round 77: round77_precond_trans2 linked with probes\magma_z_precond_wrapper_fix77.cpp (MagmaTrans Jacobi uses conj(D^{-1}))
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=/I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\sparse\include /I%M%\magma-2.10.0\control /I%M%\magma-2.10.0\sparse\control /I%CU%\include
if not exist %M%\bin\fix77 mkdir %M%\bin\fix77
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY %INC% %M%\probes\round77_precond_trans2.cpp %M%\probes\magma_z_precond_wrapper_fix77.cpp /Fo%M%\bin\fix77\ /Fe%M%\bin\round77_precond_trans2_fix.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib %M%\build\lib\magma_sparse.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX77_BUILD_EXIT %ERRORLEVEL%

@echo off
rem round 65: round65_sparse_complex linked with probes\ztfqmr_fixed.cpp and probes\zptfqmr_merge_fixed.cpp (TFQMR inner products r_tld^H x) ahead of magma_sparse.lib
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=/I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\sparse\include /I%M%\magma-2.10.0\control /I%M%\magma-2.10.0\sparse\control /I%CU%\include
if not exist %M%\bin\fix65 mkdir %M%\bin\fix65
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /DMAGMA_HAVE_CUDA %INC% %M%\probes\round65_sparse_complex.cpp %M%\probes\ztfqmr_fixed.cpp %M%\probes\zptfqmr_merge_fixed.cpp %M%\probes\ztfqmr_merge_fixed.cpp /Fo%M%\bin\fix65\ /Fe%M%\bin\round65_sparse_complex_fix.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib %M%\build\lib\magma_sparse.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX65_BUILD_EXIT %ERRORLEVEL%

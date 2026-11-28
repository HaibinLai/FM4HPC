@echo off
rem round46_formats linked with zgeellrtmv.cu whose per-row reductions use __syncwarp (overrides magma_sparse.lib): bin\round46_formats_ellrtfix.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=-I%M%\build\include -I%M%\magma-2.10.0\include -I%M%\magma-2.10.0\control -I%M%\magma-2.10.0\sparse\include -I%M%\magma-2.10.0\sparse\control -I%M%\magma-2.10.0\magmablas -I%CU%\include
%CU%\bin\nvcc.exe -c -O3 -std=c++17 -arch=sm_120 -DADD_ -DMAGMA_NOAFFINITY -Xcompiler /MD %INC% %M%\probes\zgeellrtmv_sync.cu -o %M%\bin\zgeellrtmv_sync.obj
if errorlevel 1 ( echo NVCC_FAIL & exit /b 1 )
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\sparse\include /I%CU%\include ^
   %M%\probes\round46_formats.cpp %M%\probes\magma_zmconvert_fixed.cpp %M%\probes\magma_zmtranspose_fixed.cpp %M%\bin\zgeellrtmv_sync.obj /Fo%M%\bin\ /Fe%M%\bin\round46_formats_ellrtfix.exe ^
   /I%M%\magma-2.10.0\control /I%M%\magma-2.10.0\sparse\control /DMAGMA_HAVE_CUDA /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma_sparse.lib %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX_BUILD_EXIT %ERRORLEVEL%

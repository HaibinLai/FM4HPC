@echo off
rem round 82: round82_hetrs linked with probes\zhetrs_fixed.cu (Lower forward solve through the per-column path)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
set INC=-I%M%\build\include -I%M%\magma-2.10.0\include -I%M%\magma-2.10.0\control -I%M%\magma-2.10.0\magmablas -I%CU%\include
%CU%\bin\nvcc.exe -c -O3 -std=c++17 -arch=sm_120 -DADD_ -DMAGMA_NOAFFINITY -DPRECISION_z -Xcompiler /MD %INC% %M%\probes\zhetrs_fixed.cu -o %M%\bin\zhetrs_fixed.obj
if errorlevel 1 ( echo NVCC_FAIL & exit /b 1 )
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /I%M%\build\include /I%M%\magma-2.10.0\include /I%CU%\include ^
   %M%\probes\round82_hetrs.cpp %M%\bin\zhetrs_fixed.obj /Fo%M%\bin\ /Fe%M%\bin\round82_hetrs_fix.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo FIX82_BUILD_EXIT %ERRORLEVEL%

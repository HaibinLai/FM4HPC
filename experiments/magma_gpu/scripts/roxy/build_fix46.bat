@echo off
rem round46 probes linked with magma_zmconvert.cpp whose host CSR -> COO also fills rowidx (overrides magma_sparse.lib)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
for %%p in (round46_coo round46_formats) do (
  cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /DMAGMA_HAVE_CUDA /I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\sparse\include /I%M%\magma-2.10.0\control /I%M%\magma-2.10.0\sparse\control /I%CU%\include ^
     %M%\probes\%%p.cpp %M%\probes\magma_zmconvert_coofix.cpp /Fo%M%\bin\ /Fe%M%\bin\%%p_coofix.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma_sparse.lib %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
  if errorlevel 1 ( echo FIX_BUILD_EXIT 1 & exit /b 1 )
)
echo FIX_BUILD_EXIT 0

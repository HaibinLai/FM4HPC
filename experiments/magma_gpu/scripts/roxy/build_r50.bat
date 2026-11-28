@echo off
rem round50: harness + emulated copies of [zcds]potrf3_mgpu.cpp (setdevice no-op, timing hooks); see probes\potrf_emu\emu_hooks.h
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
set M=D:\lean4hpc\magma
cl /nologo /std:c++17 /O2 /EHsc /MD /openmp /DADD_ /DMAGMA_NOAFFINITY /DMAGMA_HAVE_CUDA /I%M%\build\include /I%M%\magma-2.10.0\include /I%M%\magma-2.10.0\control /I%M%\probes\potrf_emu /I%CU%\include ^
   %M%\probes\round50_potrf_emu.cpp %M%\probes\potrf_emu\emu_zpotrf3_mgpu.cpp %M%\probes\potrf_emu\emu_dpotrf3_mgpu.cpp %M%\probes\potrf_emu\emu_cpotrf3_mgpu.cpp %M%\probes\potrf_emu\emu_spotrf3_mgpu.cpp /Fo%M%\bin\ /Fe%M%\bin\round50_potrf_emu.exe /link /LIBPATH:%CU%\lib\x64 %M%\build\lib\magma.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib D:\lean4hpc\openblas\lib\libopenblas.lib
echo R50_BUILD_EXIT %ERRORLEVEL%

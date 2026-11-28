@echo off
rem Generate MSVC import libraries for the CUDA DLLs shipped in the pip wheels (they contain no .lib files).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=D:\lean4hpc\magma_env\Lib\site-packages\nvidia\cu13
cd /d D:\lean4hpc\magma
for %%D in (cublas64_13 cublasLt64_13 cusparse64_12) do (
  dumpbin /nologo /exports %CU%\bin\x86_64\%%D.dll > %%D.exports.txt
  echo LIBRARY %%D.dll> %%D.def
  echo EXPORTS>> %%D.def
  for /f "skip=19 tokens=4" %%S in (%%D.exports.txt) do @echo %%S>> %%D.def
)
lib /nologo /machine:x64 /def:cublas64_13.def /out:%CU%\lib\x64\cublas.lib
lib /nologo /machine:x64 /def:cublasLt64_13.def /out:%CU%\lib\x64\cublasLt.lib
lib /nologo /machine:x64 /def:cusparse64_12.def /out:%CU%\lib\x64\cusparse.lib
dir /b %CU%\lib\x64
dumpbin /nologo /exports D:\lean4hpc\openblas\bin\libopenblas.dll | findstr /r /c:" dgetrf_$" /c:" zheevd_$" /c:" zgbtrf_$" /c:" cblas_dgemm$"
for %%D in (cublas64_13 cusparse64_12) do (findstr /c:"cublasZgemm_v2" /c:"cusparseCreate" %%D.def | find /c /v "")

#!/bin/bash
# SuperLU_DIST 239280c (2026-11-24), single+double, no ParMETIS, OpenMPI, OpenBLAS.  EXAMPLE/psdrive.c is patched
# (patch_psdrive.py) only to (a) inject a NaN into b[NANROW] after b = A*xtrue is generated and (b) print info,
# #NaN in the returned x and err_bounds right after psgssvx_d2.
set -e; source $(dirname $0)/env_node.sh; cd $R
[ -d superlu_dist ] || git clone -q https://github.com/xiaoyeli/superlu_dist.git
cd superlu_dist && git checkout -q 239280c && git checkout -q EXAMPLE/psdrive.c && python3 $R/scripts/patch_psdrive.py EXAMPLE/psdrive.c
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$R/env/bin/mpicc -DCMAKE_CXX_COMPILER=$R/env/bin/mpicxx \
  -Denable_single=ON -Denable_double=ON -Denable_complex16=OFF -DTPL_ENABLE_PARMETISLIB=OFF -DXSDK_ENABLE_Fortran=OFF \
  -Denable_python=OFF -Denable_tests=OFF -Denable_examples=ON -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
  -DTPL_BLAS_LIBRARIES=$R/env/lib/libopenblas.so -DTPL_ENABLE_LAPACKLIB=OFF > $R/sludist_cmake.log 2>&1
cmake --build build -j 32 --target psdrive > $R/sludist_build.log 2>&1
echo SLUDIST_OK

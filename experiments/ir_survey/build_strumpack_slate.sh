#!/bin/bash
# STRUMPACK edc9a72 (2026-11-23, 8.1.0-dev), no MPI; SLATE 878439b (2026-08-26) with LAPACK API, OpenMPI, Host only.
set -e; source $(dirname $0)/env_node.sh; cd $R
export OPENBLAS_NUM_THREADS=1
[ -d STRUMPACK ] || git clone -q https://github.com/pghysels/STRUMPACK.git
(cd STRUMPACK && git checkout -q edc9a72 && \
 cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$R/inst -DSTRUMPACK_USE_MPI=OFF \
   -DSTRUMPACK_USE_OPENMP=ON -DTPL_ENABLE_SLATE=OFF -DTPL_ENABLE_PARMETIS=OFF -DTPL_ENABLE_SCOTCH=OFF -DTPL_ENABLE_BPACK=OFF \
   -DTPL_ENABLE_ZFP=OFF -DTPL_ENABLE_COMBBLAS=OFF -DTPL_ENABLE_MAGMA=OFF -DSTRUMPACK_USE_CUDA=OFF -DSTRUMPACK_USE_HIP=OFF \
   -DTPL_METIS_PREFIX=$R/env -DBLA_VENDOR=OpenBLAS > $R/strumpack_cmake.log 2>&1 && \
 cmake --build build -j 32 --target strumpack > $R/strumpack_build.log 2>&1 && cmake --install build > /dev/null) && echo STRUMPACK_OK
[ -d slate ] || git clone -q --recursive https://github.com/icl-utk-edu/slate.git
(cd slate && git checkout -q 878439b && git submodule update -q --init --recursive && \
 cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$R/inst -Dgpu_backend=none -Dbuild_tests=no \
   -Dblas=openblas -DCMAKE_CXX_COMPILER=$R/env/bin/mpicxx -DCMAKE_C_COMPILER=$R/env/bin/mpicc > $R/slate_cmake.log 2>&1 && \
 cmake --build build -j 32 --target slate_lapack_api > $R/slate_build.log 2>&1) && echo SLATE_OK

#!/bin/bash
# PaStiX master e2ecf99 (2026-11-03), shared-memory only, scotch ordering.
set -e; source $(dirname $0)/env_node.sh; cd $R
[ -d pastix ] || git clone -q --recursive https://gitlab.inria.fr/solverstack/pastix.git
cd pastix && git checkout -q e2ecf99 && git submodule update -q --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$R/inst -DBUILD_SHARED_LIBS=ON \
  -DPASTIX_WITH_MPI=OFF -DPASTIX_WITH_FORTRAN=OFF -DPASTIX_WITH_CUDA=OFF -DPASTIX_WITH_STARPU=OFF -DPASTIX_WITH_PARSEC=OFF \
  -DPASTIX_ORDERING_SCOTCH=ON -DPASTIX_ORDERING_METIS=OFF -DBLA_VENDOR=OpenBLAS > $R/pastix_cmake.log 2>&1
cmake --build build -j 32 > $R/pastix_build.log 2>&1 && cmake --install build > /dev/null
echo PASTIX_OK

#!/bin/bash
# PLASMA master 548e0bd (2026-05-11), OpenMP tasks, OpenBLAS.
set -e; source $(dirname $0)/env_node.sh; cd $R
[ -d plasma ] || git clone -q https://github.com/icl-utk-edu/plasma.git
cd plasma && git checkout -q 548e0bd
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$R/inst -DBUILD_SHARED_LIBS=ON \
  -DBLA_VENDOR=OpenBLAS > $R/plasma_cmake.log 2>&1
cmake --build build -j 32 --target plasma plasma_core_blas > $R/plasma_build.log 2>&1
echo PLASMA_OK

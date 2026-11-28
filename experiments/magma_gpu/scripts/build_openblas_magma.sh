#!/bin/bash
# OpenBLAS 0.3.28 (C LAPACK, no Fortran) + MAGMA 2.10.0 original and GMSTRS-fix v2, for timing. /ssd/haibin/magma_gpu.
set -e
W=/ssd/haibin/magma_gpu; mkdir -p $W; cd $W
curl -sSL -o OpenBLAS-0.3.28.tar.gz https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.28/OpenBLAS-0.3.28.tar.gz
sha256sum OpenBLAS-0.3.28.tar.gz > openblas.sha256
tar xzf OpenBLAS-0.3.28.tar.gz
( cd OpenBLAS-0.3.28 && time make -j 96 NO_FORTRAN=1 C_LAPACK=1 USE_OPENMP=1 NUM_THREADS=128 DYNAMIC_ARCH=0 > ../build_openblas.log 2>&1 && make PREFIX=$W/openblas install >> ../build_openblas.log 2>&1 )
[ -f magma-2.10.0.tar.gz ] || curl -sSL -o magma-2.10.0.tar.gz https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.10.0.tar.gz
sha256sum magma-2.10.0.tar.gz > tarball.sha256
for v in orig fix; do
  rm -rf magma-$v; mkdir magma-$v; tar xzf magma-2.10.0.tar.gz -C magma-$v --strip-components=1
  cd magma-$v
  [ $v = fix ] && patch -p1 < ../gmstrs_fix_v2.patch
  cat > make.inc <<'MK'
BACKEND = cuda
CC = gcc
CXX = g++
NVCC = nvcc
FORT =
ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib
GPU_TARGET = Volta
FPIC = -fPIC
CFLAGS = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -fopenmp -std=c99
CXXFLAGS = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
NVCCFLAGS = -O3 -DNDEBUG -DADD_ -Xcompiler "$(FPIC)" -std=c++11
LDFLAGS = $(FPIC) -fopenmp
CUDADIR = /usr
OPENBLASDIR = /ssd/haibin/magma_gpu/openblas
LIB = -L$(OPENBLASDIR)/lib -lopenblas -lcublas -lcusparse -lcudart -lstdc++ -lm
INC = -I$(CUDADIR)/include
LIBDIR = -L$(OPENBLASDIR)/lib
MK
  ( time make -j 96 lib ) > ../build_magma_$v.log 2>&1
  cd ..
done
echo ALL_DONE >> build_magma_fix.log

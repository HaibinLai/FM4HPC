#!/bin/bash
# Build MAGMA 2.10.0 (dense + sparse) for V100 on eightsocket. Work dir /ssd/haibin/magma_gpu (delete when done).
set -e
W=/ssd/haibin/magma_gpu; mkdir -p $W; cd $W
[ -f magma-2.10.0.tar.gz ] || curl -sSL -o magma-2.10.0.tar.gz https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.10.0.tar.gz
sha256sum magma-2.10.0.tar.gz > tarball.sha256
[ -d magma-2.10.0 ] || tar xzf magma-2.10.0.tar.gz
cd magma-2.10.0
cat > make.inc <<'MK'
BACKEND     = cuda
CC          = gcc
CXX         = g++
NVCC        = nvcc
FORT        =
ARCH        = ar
ARCHFLAGS   = cr
RANLIB      = ranlib
GPU_TARGET  = Volta
FPIC        = -fPIC
CFLAGS      = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -fopenmp -std=c99
CXXFLAGS    = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
NVCCFLAGS   = -O3 -DNDEBUG -DADD_ -Xcompiler "$(FPIC)" -std=c++11
LDFLAGS     = $(FPIC) -fopenmp
CUDADIR     = /usr
LIB         = -L/usr/lib/x86_64-linux-gnu -l:liblapack.so.3 -l:libblas.so.3 -lcublas -lcusparse -lcudart -lstdc++ -lm -l:libgfortran.so.5
INC         = -I$(CUDADIR)/include
LIBDIR      = -L/usr/lib/x86_64-linux-gnu
MK
( time make -j 96 lib ) > ../build_lib.log 2>&1
( time make -j 96 sparse-lib ) > ../build_sparse.log 2>&1
ls -la lib/ >> ../build_lib.log
echo BUILD_DONE >> ../build_sparse.log

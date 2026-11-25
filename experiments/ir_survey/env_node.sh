# source on the node: conda toolchain under /ssd/haibin/lean4hpc-ir (deleted after the runs)
export R=/ssd/haibin/lean4hpc-ir
export PATH=$R/env/bin:$PATH
export CC=$R/env/bin/x86_64-conda-linux-gnu-gcc CXX=$R/env/bin/x86_64-conda-linux-gnu-g++ FC=$R/env/bin/x86_64-conda-linux-gnu-gfortran
export CMAKE_PREFIX_PATH=$R/env LD_LIBRARY_PATH=$R/env/lib:$R/inst/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4

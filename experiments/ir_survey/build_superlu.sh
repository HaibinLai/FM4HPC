#!/bin/bash
# build SuperLU (serial) from source with its bundled CBLAS, no cmake; then the NaN probe.  usage: build_superlu.sh SRCDIR
set -e
S=$1; cd "$S"
grep -v cmakedefine SRC/superlu_config.h.in > SRC/superlu_config.h
mkdir -p obj && cd obj
for f in ../SRC/*.c; do gcc -O2 -DAdd_ -I../SRC -c "$f" -o "s_$(basename "${f%.c}").o" 2>/dev/null || echo "skip $f"; done
for f in ../CBLAS/*.c; do gcc -O2 -DAdd_ -I../SRC -I../CBLAS -c "$f" -o "b_$(basename "${f%.c}").o" 2>/dev/null || echo "skip $f"; done
ar cr libsuperlu_all.a *.o
cd ..
gcc -O2 -DAdd_ -ISRC superlu_nan.c obj/libsuperlu_all.a -lm -o superlu_nan

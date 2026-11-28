/* LD_PRELOAD: interpose LAPACK dsteqr_; the call number FI_DSTEQR_K (1-based) reports info = FI_DSTEQR_INFO (default 3)
   after running the real routine.  FI_DSTEQR_K = 0 only logs the call sizes to stderr. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
typedef void (*fn)(const char*, const int*, double*, double*, double*, const int*, double*, int*);
static long calls = 0;
void dsteqr_(const char* compz, const int* n, double* d, double* e, double* z, const int* ldz, double* work, int* info) {
  static fn real = 0; if (!real) real = (fn)dlsym(RTLD_NEXT, "dsteqr_");
  real(compz, n, d, e, z, ldz, work, info); calls++;
  long k = getenv("FI_DSTEQR_K") ? atol(getenv("FI_DSTEQR_K")) : 0;
  fprintf(stderr, "[dsteqr] call %ld n=%d%s\n", calls, *n, (k == calls) ? "  <- forced failure" : "");
  if (k == calls) *info = getenv("FI_DSTEQR_INFO") ? atoi(getenv("FI_DSTEQR_INFO")) : 3;
}

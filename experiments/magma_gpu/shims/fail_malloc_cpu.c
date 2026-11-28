/* LD_PRELOAD: interpose libmagma's exported magma_malloc_cpu; once armed (FI_CPU_K = k > 0) the k-th call fails with
   MAGMA_ERR_HOST_ALLOC (-112).  k = 0 only counts (printed at exit). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
static long calls = 0; static int armed = 0;
void fi_cpu_arm(void) { armed = 1; calls = 0; }
void fi_cpu_disarm(void) { armed = 0; }
long fi_cpu_count(void) { return calls; }
int magma_malloc_cpu(void** p, size_t n) {
  static int (*real)(void**, size_t) = 0; if (!real) real = (int (*)(void**, size_t))dlsym(RTLD_NEXT, "magma_malloc_cpu");
  if (armed) { calls++; long k = getenv("FI_CPU_K") ? atol(getenv("FI_CPU_K")) : 0; if (k == calls) { *p = NULL; return -112; } }
  return real(p, n);
}

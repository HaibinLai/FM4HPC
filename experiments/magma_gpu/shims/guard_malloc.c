/* LD_PRELOAD shim: detect device out-of-bounds writes past the end of cudaMalloc allocations.
   Each allocation gets GUARD extra bytes filled with 0xAB; on cudaFree the guard is copied back and checked.
   Build: gcc -shared -fPIC -O1 guard_malloc.c -o guard_malloc.so -ldl -lcudart   (a write must reach the guard to be seen) */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
typedef int cudaError_t;
#define GUARD (1u << 20)
extern cudaError_t cudaMemset(void*, int, size_t);
extern cudaError_t cudaMemcpy(void*, const void*, size_t, int);
extern cudaError_t cudaDeviceSynchronize(void);
struct rec { void* p; size_t n; };
static struct rec R[65536]; static int NR = 0; static long nviol = 0, nchecked = 0;
cudaError_t cudaMalloc(void** p, size_t n) {
  static cudaError_t (*real)(void**, size_t) = 0; if (!real) real = dlsym(RTLD_NEXT, "cudaMalloc");
  cudaError_t e = real(p, n + GUARD); if (e) return e;
  cudaMemset((char*)*p + n, 0xAB, GUARD);
  if (NR < 65536) { R[NR].p = *p; R[NR].n = n; NR++; }
  return e;
}
cudaError_t cudaFree(void* p) {
  static cudaError_t (*real)(void*) = 0; if (!real) real = dlsym(RTLD_NEXT, "cudaFree");
  for (int i = 0; i < NR; i++) if (R[i].p == p && p) {
    unsigned char* h = malloc(GUARD); cudaDeviceSynchronize(); cudaMemcpy(h, (char*)p + R[i].n, GUARD, 2 /*D2H*/);
    long first = -1, last = -1, cnt = 0;
    for (size_t k = 0; k < GUARD; k++) if (h[k] != 0xAB) { if (first < 0) first = k; last = k; cnt++; }
    nchecked++;
    if (cnt) { nviol++; fprintf(stderr, "[guard] OUT-OF-BOUNDS WRITE: allocation of %zu bytes, guard bytes [%ld, %ld] modified (%ld bytes)\n", R[i].n, first, last, cnt); }
    free(h); R[i] = R[--NR]; break;
  }
  return real(p);
}
__attribute__((destructor)) static void fin(void) { fprintf(stderr, "[guard] allocations checked at free: %ld, with out-of-bounds writes: %ld\n", nchecked, nviol); }

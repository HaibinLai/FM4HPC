/* LD_PRELOAD fault injector: once armed, the k-th cudaMalloc / cudaHostAlloc / cudaMallocHost call fails with
   cudaErrorMemoryAllocation (k = 1, 2, ...); k = 0 only counts.  The probe arms it through dlsym(RTLD_DEFAULT, "fi_arm")
   right before the routine under test and reads the count with "fi_count".
   Build: gcc -shared -fPIC -O1 fail_malloc.c -o fail_malloc.so -ldl */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <execinfo.h>
#include <stdlib.h>
#include <unistd.h>
typedef int cudaError_t;
#define ERR_ALLOC 2   /* cudaErrorMemoryAllocation */
static long armed = 0, target = 0, count = 0;
void fi_arm(long k) { armed = 1; target = k; count = 0; }
void fi_disarm(void) { armed = 0; }
long fi_count(void) { return count; }
static int fail_now(void) { if (!armed) return 0; count++; if (target > 0 && count == target) {
  if (getenv("FI_BT")) { void* b[12]; int n = backtrace(b, 12); write(2, "[fi] failing allocation from:\n", 30); backtrace_symbols_fd(b + 1, n - 1, 2); }
  return 1; } return 0; }
cudaError_t cudaMalloc(void** p, size_t n) {
  static cudaError_t (*real)(void**, size_t) = 0; if (!real) real = dlsym(RTLD_NEXT, "cudaMalloc");
  if (fail_now()) { *p = 0; return ERR_ALLOC; } return real(p, n);
}
cudaError_t cudaHostAlloc(void** p, size_t n, unsigned int f) {
  static cudaError_t (*real)(void**, size_t, unsigned int) = 0; if (!real) real = dlsym(RTLD_NEXT, "cudaHostAlloc");
  if (fail_now()) { *p = 0; return ERR_ALLOC; } return real(p, n, f);
}
cudaError_t cudaMallocHost(void** p, size_t n) {
  static cudaError_t (*real)(void**, size_t) = 0; if (!real) real = dlsym(RTLD_NEXT, "cudaMallocHost");
  if (fail_now()) { *p = 0; return ERR_ALLOC; } return real(p, n);
}

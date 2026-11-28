/* LD_PRELOAD: print a backtrace on SIGSEGV (no gdb on the node). Build: gcc -shared -fPIC segv_bt.c -o segv_bt.so */
#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
static void h(int s) { void* b[64]; int n = backtrace(b, 64); fprintf(stderr, "[segv_bt] signal %d, backtrace:\n", s); backtrace_symbols_fd(b, n, 2); _exit(139); }
__attribute__((constructor)) static void init(void) { signal(SIGSEGV, h); signal(SIGBUS, h); }

// Eighteenth GPU round: magma_zunglq — header (…, dT, nb) vs definition (…, work, lwork) — and its fault injection.
#include "magma_v2.h"
#include "magma_lapack.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 97u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static double diff(const std::vector<Z>& a, const std::vector<Z>& b) { double e = 0, m = 0; for (size_t i = 0; i < a.size(); i++) { if (!std::isfinite(MAGMA_Z_REAL(a[i]))) return INFINITY;
  e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(a[i], b[i]))); m = fmax(m, MAGMA_Z_ABS(b[i])); } return e / m; }
int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : ""; long k = argc > 2 ? atol(argv[2]) : 0;
  void (*fi_arm)(long) = (void (*)(long))dlsym(RTLD_DEFAULT, "fi_arm"); void (*fi_disarm)() = (void (*)())dlsym(RTLD_DEFAULT, "fi_disarm"); long (*fi_count)() = (long (*)())dlsym(RTLD_DEFAULT, "fi_count");
  magma_init(); int n = 400; magma_int_t nn = n, inf, lw = n * 256; std::vector<Z> A((size_t)n * n), tau(n), wk(lw); for (auto& a : A) a = C(rnd(), rnd());
  lapackf77_zgelqf(&nn, &nn, A.data(), &nn, tau.data(), wk.data(), &lw, &inf);
  std::vector<Z> ref = A; lapackf77_zunglq(&nn, &nn, &nn, ref.data(), &nn, tau.data(), wk.data(), &lw, &inf);   // LAPACK Q
  if (mode == "header") {   // the way magma_z.h declares it: (…, tau, dT, nb, info)
    magma_int_t nb = magma_get_zgelqf_nb(n, n); Z* dT; magma_zmalloc(&dT, (size_t)n * nb); std::vector<Z> Q = A; magma_int_t info = -999;
    magma_zunglq(n, n, n, Q.data(), n, tau.data(), dT, nb, &info);
    printf("zunglq called as declared in magma_z.h (dT = device buffer, nb = %lld): info=%lld  ||Q - Q_lapack||/||Q|| = %.3g\n", (long long)nb, (long long)info, diff(Q, ref));
  } else {                  // the way it is defined: (…, tau, work, lwork, info), under fault injection
    std::vector<Z> Q = A; magma_int_t info = -999; if (fi_arm) fi_arm(k); magma_zunglq(n, n, n, Q.data(), n, tau.data(), wk.data(), lw, &info);
    long c = fi_count ? fi_count() : -1; if (fi_disarm) fi_disarm(); double e = diff(Q, ref);
    printf("unglq      k=%3ld nalloc=%3ld info=%-6lld diff_to_lapack=%-10.3g %s\n", k, c, (long long)info, e, (info == 0 && !(e <= 1e-10)) ? "SILENT-WRONG" : (info != 0 ? "reported" : "ok"));
  }
  magma_finalize(); return 0;
}

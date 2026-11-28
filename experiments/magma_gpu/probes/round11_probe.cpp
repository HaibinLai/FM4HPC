// Eleventh GPU round: fault injection on the helper routines that discard a sub-call's iinfo (HANDOFF §29.11).
//   LD_PRELOAD=fail_malloc.so ./round11_probe <unmtr|unghr|unmbr> <k>    (k = 0: count only).  GPU 0 only.
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
static unsigned S = 777u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static void (*fi_arm)(long) = 0; static void (*fi_disarm)() = 0; static long (*fi_count)() = 0;
static double diff(const std::vector<Z>& a, const std::vector<Z>& b) { double e = 0, m = 0; for (size_t i = 0; i < a.size(); i++) {
  if (!std::isfinite(MAGMA_Z_REAL(a[i]))) return INFINITY; e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(a[i], b[i]))); m = fmax(m, MAGMA_Z_ABS(b[i])); } return e / m; }
int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : ""; long k = argc > 2 ? atol(argv[2]) : 0;
  fi_arm = (void (*)(long))dlsym(RTLD_DEFAULT, "fi_arm"); fi_disarm = (void (*)())dlsym(RTLD_DEFAULT, "fi_disarm"); fi_count = (long (*)())dlsym(RTLD_DEFAULT, "fi_count");
  magma_init(); int n = 1000; magma_int_t nn = n, info = -999;
  std::vector<Z> A((size_t)n * n), C0((size_t)n * n), tau(n), tau2(n); for (auto& a : A) a = C(rnd(), rnd()); for (auto& c : C0) c = C(rnd(), rnd());
  std::vector<Z> ref, out, inp; long cnt = -1;
  if (mode == "unmtr" || mode == "unmbr") {
    if (mode == "unmtr") { for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) A[i + (size_t)j * n] = MAGMA_Z_CONJ(A[j + (size_t)i * n]);
      std::vector<double> d(n), e(n); magma_int_t lw = -1, inf; Z q; lapackf77_zhetrd("L", &nn, A.data(), &nn, d.data(), e.data(), tau.data(), &q, &lw, &inf);
      lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> wk(lw); lapackf77_zhetrd("L", &nn, A.data(), &nn, d.data(), e.data(), tau.data(), wk.data(), &lw, &inf); }
    else { std::vector<double> d(n), e(n); magma_int_t lw = -1, inf; Z q; lapackf77_zgebrd(&nn, &nn, A.data(), &nn, d.data(), e.data(), tau.data(), tau2.data(), &q, &lw, &inf);
      lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> wk(lw); lapackf77_zgebrd(&nn, &nn, A.data(), &nn, d.data(), e.data(), tau.data(), tau2.data(), wk.data(), &lw, &inf); }
    magma_int_t lwork = n * 256; std::vector<Z> wk(lwork);
    auto call = [&](std::vector<Z>& Cm) { if (mode == "unmtr") magma_zunmtr(MagmaLeft, MagmaLower, MagmaNoTrans, n, n, A.data(), n, tau.data(), Cm.data(), n, wk.data(), lwork, &info);
      else magma_zunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, n, n, n, A.data(), n, tau.data(), Cm.data(), n, wk.data(), lwork, &info); };
    ref = C0; call(ref); out = C0; info = -999; if (fi_arm) fi_arm(k); call(out); cnt = fi_count ? fi_count() : -1; if (fi_disarm) fi_disarm(); inp = C0;
  } else if (mode == "unghr") {
    magma_int_t nb = magma_get_zgehrd_nb(n), lw = n * nb, inf; std::vector<Z> wk(lw); Z* dT; magma_zmalloc(&dT, (size_t)nb * n);
    magma_zgehrd(n, 1, n, A.data(), n, tau.data(), wk.data(), lw, dT, &inf);
    std::vector<Z> H = A; ref = H; magma_zunghr(n, 1, n, ref.data(), n, tau.data(), dT, nb, &inf);
    out = H; info = -999; if (fi_arm) fi_arm(k); magma_zunghr(n, 1, n, out.data(), n, tau.data(), dT, nb, &info); cnt = fi_count ? fi_count() : -1; if (fi_disarm) fi_disarm(); inp = H;
  }
  double e = diff(out, ref), u = diff(out, inp);
  const char* v = (info == 0 && !(e <= 1e-12)) ? "SILENT-WRONG" : (info != 0 ? "reported" : "ok");
  printf("%-6s k=%3ld nalloc=%3ld info=%-6lld diff_to_ref=%-10.3g diff_to_input=%-10.3g %s\n", mode.c_str(), k, cnt, (long long)info, e, u, v);
  magma_finalize(); return 0;
}

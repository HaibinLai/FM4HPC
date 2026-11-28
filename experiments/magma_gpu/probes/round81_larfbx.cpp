// Round 81 (roxy): the geqr2x helpers above n = 128, and magma_zlarfbx_gpu on its own.
//   round81_larfbx dgeqr2x <v 2|3|4> <m> <n>   magma_dgeqr2x{2,3,4}_gpu (documented N >= 0), dwork 60 n doubles; R and tau against LAPACK dgeqrf
//   round81_larfbx zlarfbx <m> <k> <junk 0|1>   documented "C = H C, H = I - V T V**H" (T upper triangular, V unit lower m x k);
//       err against (I - V T V^H) c and err2 against (I - V T^H V^H) c; junk = 1 puts 1e3 into the strictly lower part of T
// CSV: r81,op,a,b,c,info,err,err2,cuda,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
typedef std::complex<double> cd;
static unsigned Sd = 5u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 5) return 1; std::string op = argv[1]; magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  long long info = 0; double err = -1, err2 = -1; bool ok = false;
  if (op == "dgeqr2x") { int v = atoi(argv[2]); magma_int_t m = atoi(argv[3]), n = atoi(argv[4]), ld = magma_roundup(m, 32), in = 0; std::vector<double> A((size_t)m * n); for (auto& a : A) a = r1();
    double *dA, *dtau, *dT, *ddA, *dw; magma_dmalloc(&dA, (size_t)ld * n); magma_dmalloc(&dtau, n); magma_dmalloc(&dT, (size_t)n * n); magma_dmalloc(&ddA, (size_t)n * n); magma_dmalloc(&dw, (size_t)60 * n);
    magma_dsetmatrix(m, n, A.data(), m, dA, ld, q);
    if (v == 2) magma_dgeqr2x2_gpu(m, n, dA, ld, dtau, dT, ddA, dw, &in); else if (v == 3) magma_dgeqr2x3_gpu(m, n, dA, ld, dtau, dT, ddA, dw, &in); else magma_dgeqr2x4_gpu(m, n, dA, ld, dtau, dT, ddA, dw, q, &in);
    magma_queue_sync(q); info = in; std::vector<double> R((size_t)n * n), tau(n); magma_dgetmatrix(n, n, ddA, n, R.data(), n, q); magma_dgetvector(n, dtau, 1, tau.data(), 1, q);
    std::vector<double> L = A, tl(n); magma_int_t lw = -1, ri; double wq; lapackf77_dgeqrf(&m, &n, L.data(), &m, tl.data(), &wq, &lw, &ri); lw = (magma_int_t)wq; std::vector<double> w(lw); lapackf77_dgeqrf(&m, &n, L.data(), &m, tl.data(), w.data(), &lw, &ri);
    double e = 0, a = 0; for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { e = std::max(e, fabs(R[i + (size_t)j * n] - L[i + (size_t)j * m])); a = std::max(a, fabs(L[i + (size_t)j * m])); }
    err = e / a; err2 = 0; for (int i = 0; i < n; i++) err2 = std::max(err2, fabs(tau[i] - tl[i])); ok = info == 0 && err < 1e-10 && err2 < 1e-10; }
  else { int m = atoi(argv[2]), k = atoi(argv[3]); bool junk = atoi(argv[4]) == 1; std::vector<cd> V((size_t)m * k, 0.0), T((size_t)k * k, 0.0), c(m);
    for (int j = 0; j < k; j++) for (int i = 0; i < m; i++) V[i + (size_t)j * m] = i < j ? cd(0) : i == j ? cd(1) : cd(r1(), r1()) * 0.2;
    for (int j = 0; j < k; j++) for (int i = 0; i < k; i++) T[i + (size_t)j * k] = i <= j ? cd(r1(), r1()) * (i == j ? 1.0 : 0.1) : (junk ? cd(1e3, 1e3) : cd(0));
    for (auto& x : c) x = cd(r1(), r1());
    std::vector<magmaDoubleComplex> hV(V.size()), hT(T.size()), hc(m); for (size_t i = 0; i < V.size(); i++) hV[i] = MAGMA_Z_MAKE(V[i].real(), V[i].imag()); for (size_t i = 0; i < T.size(); i++) hT[i] = MAGMA_Z_MAKE(T[i].real(), T[i].imag()); for (int i = 0; i < m; i++) hc[i] = MAGMA_Z_MAKE(c[i].real(), c[i].imag());
    magmaDoubleComplex *dV, *dT, *dc, *dw; magma_zmalloc(&dV, V.size()); magma_zmalloc(&dT, T.size()); magma_zmalloc(&dc, m); magma_zmalloc(&dw, 2 * k + 1);
    magma_zsetvector((magma_int_t)V.size(), hV.data(), 1, dV, 1, q); magma_zsetvector((magma_int_t)T.size(), hT.data(), 1, dT, 1, q); magma_zsetvector(m, hc.data(), 1, dc, 1, q);
    magma_zlarfbx_gpu(m, k, dV, m, dT, k, dc, dw, q); magma_queue_sync(q); magma_zgetvector(m, dc, 1, hc.data(), 1, q);
    auto apply = [&](bool herm) { std::vector<cd> w(k, 0.0), t(k, 0.0), r = c; for (int j = 0; j < k; j++) for (int i = 0; i < m; i++) w[j] += std::conj(V[i + (size_t)j * m]) * c[i];
      for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) { if (i > j) continue; cd tij = T[i + (size_t)j * k]; if (herm) t[j] += std::conj(tij) * w[i]; else t[i] += tij * w[j]; }   // T upper only
      for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) r[i] -= V[i + (size_t)j * m] * t[j]; return r; };
    std::vector<cd> RH = apply(false), RHH = apply(true); double e = 0, e2 = 0, a = 0; for (int i = 0; i < m; i++) { cd x(MAGMA_Z_REAL(hc[i]), MAGMA_Z_IMAG(hc[i])); e = std::max(e, std::abs(x - RH[i])); e2 = std::max(e2, std::abs(x - RHH[i])); a = std::max(a, std::abs(RH[i])); }
    err = e / a; err2 = e2 / a; ok = err < 1e-12; }
  cudaError_t ce = cudaGetLastError();
  printf("r81,%s,%s,%s,%s,%lld,%.3g,%.3g,%s,%s\n", op.c_str(), argv[2], argv[3], argv[4], info, err, err2, ce == cudaSuccess ? "ok" : "ERR", ok && ce == cudaSuccess ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(ce)); magma_queue_destroy(q); magma_finalize(); return 0;
}

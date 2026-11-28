// Round 39 (roxy): confirm the round 31-37 findings in complex precision (z, and c where the trigger is element-size
// dependent).  One minimal trigger per finding, printed with its control case.
// CSV: r39,finding,routine,case,observed,expected,verdict   verdict REPRODUCED (trigger wrong, control right) | NOT
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
typedef std::complex<double> cd; typedef std::complex<float> cf;
static magma_queue_t Q;
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static void rep(const char* f, const char* r, const std::string& c, double obs, double ctl, bool trig_bad, bool ctl_ok) {
  printf("r39,%s,%s,%s,%.3g,%.3g,%s\n", f, r, c.c_str(), obs, ctl, (trig_bad && ctl_ok) ? "REPRODUCED" : "NOT"); }

template <class T> static double luerr(int n, int ld, bool single) {   // ||PA - LU||_1 / (||A||_1 n eps) for [zc]getrf_native
  Sd = 11u + n; std::vector<T> A((size_t)n * n), F((size_t)n * n); for (auto& a : A) a = T(r1(), r1()); std::vector<magma_int_t> ip(n); magma_int_t info;
  void* d; magma_malloc(&d, sizeof(T) * (size_t)ld * n); magma_setmatrix(n, n, sizeof(T), A.data(), n, d, ld, Q);
  if (single) magma_cgetrf_native(n, n, (magmaFloatComplex_ptr)d, ld, ip.data(), &info); else magma_zgetrf_native(n, n, (magmaDoubleComplex_ptr)d, ld, ip.data(), &info);
  magma_getmatrix(n, n, sizeof(T), d, ld, F.data(), n, Q); magma_free(d);
  std::vector<T> PA = A; for (int i = 0; i < n; i++) { int p = (int)ip[i] - 1; if (p != i) for (int j = 0; j < n; j++) std::swap(PA[i + (size_t)j * n], PA[p + (size_t)j * n]); }
  double rn = 0, an = 0; for (int j = 0; j < n; j++) { double s = 0, sa = 0; for (int i = 0; i < n; i++) { T lu = 0; for (int k = 0; k <= std::min(i, j); k++) lu += (k == i ? T(1) : F[i + (size_t)k * n]) * F[k + (size_t)j * n];
      s += std::abs(PA[i + (size_t)j * n] - lu); sa += std::abs(A[i + (size_t)j * n]); } rn = std::max(rn, s); an = std::max(an, sa); }
  return rn / (an * n * (single ? 5.96e-8 : 1.11e-16)); }

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); magma_init(); magma_queue_create(0, &Q);
  printf("r39,finding,routine,case,observed,expected_or_control,verdict\n");
  // M242: z (16 B) and c (8 B) getrf_native with an unaligned ldda
  { double z0 = luerr<cd>(513, 513, false), z1 = luerr<cd>(513, 544, false); rep("M242", "zgetrf_native", "n=513 ldda=513 vs 544", z0, z1, z0 > 100, z1 < 1);
    double c0 = luerr<cf>(513, 513, true), c1 = luerr<cf>(513, 544, true); rep("M242", "cgetrf_native", "n=513 ldda=513 vs 544", c0, c1, c0 > 100, c1 < 1);
    double c2 = luerr<cf>(512, 516, true); rep("M242", "cgetrf_native", "n=512 ldda=516 (4128 B) vs 544", c2, c1, c2 > 100, c1 < 1); }
  // M245: zgemm_batched / zherk_batched with k = 0, beta = 0.5: C must become 0.5*C
  { int m = 40, n = 7, B = 3; std::vector<magmaDoubleComplex*> h(B); std::vector<std::vector<cd>> C0(B); for (int b = 0; b < B; b++) { C0[b].resize((size_t)m * n); for (auto& c : C0[b]) c = cd(r1(), r1()); magma_zmalloc(&h[b], (size_t)m * n); magma_zsetmatrix(m, n, (magmaDoubleComplex*)C0[b].data(), m, h[b], m, Q); }
    magmaDoubleComplex** dC; magma_malloc((void**)&dC, B * sizeof(void*)); cudaMemcpy(dC, h.data(), B * sizeof(void*), cudaMemcpyHostToDevice);
    magmablas_zgemm_batched(MagmaNoTrans, MagmaNoTrans, m, n, 0, MAGMA_Z_ONE, (magmaDoubleComplex const* const*)dC, m, (magmaDoubleComplex const* const*)dC, m, MAGMA_Z_MAKE(0.5, 0), dC, m, B, Q); magma_queue_sync(Q);
    double e = 0; for (int b = 0; b < B; b++) { std::vector<cd> g((size_t)m * n); magma_zgetmatrix(m, n, h[b], m, (magmaDoubleComplex*)g.data(), m, Q); for (size_t i = 0; i < g.size(); i++) e = std::max(e, std::abs(g[i] - 0.5 * C0[b][i])); }
    rep("M245", "zgemm_batched", "k=0 beta=0.5: max|C - 0.5 C0|", e, 0, e > 1e-3, true);
    for (int b = 0; b < B; b++) magma_zsetmatrix(m, n, (magmaDoubleComplex*)C0[b].data(), m, h[b], m, Q);
    magmablas_zherk_batched(MagmaLower, MagmaNoTrans, 7, 0, 1.0, (magmaDoubleComplex const* const*)dC, m, 0.5, dC, m, B, Q); magma_queue_sync(Q);
    e = 0; for (int b = 0; b < B; b++) { std::vector<cd> g((size_t)m * n); magma_zgetmatrix(m, n, h[b], m, (magmaDoubleComplex*)g.data(), m, Q); for (int j = 0; j < 7; j++) for (int i = j + 1; i < 7; i++) e = std::max(e, std::abs(g[i + (size_t)j * m] - 0.5 * C0[b][i + (size_t)j * m])); }
    rep("M245", "zherk_batched", "k=0 beta=0.5 (strict lower part): max|C - 0.5 C0|", e, 0, e > 1e-3, true);
    for (auto p : h) magma_free(p); magma_free(dC); }
  // M246: zherk_batched n = 31, k = 17, lower: the strictly upper triangle (sentinel 7) must not change
  { int n = 31, k = 17, B = 3; std::vector<magmaDoubleComplex*> ha(B), hc(B); std::vector<cd> A((size_t)n * k), C((size_t)n * n); for (auto& a : A) a = cd(r1(), r1());
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) C[i + (size_t)j * n] = i < j ? cd(7, 0) : cd(r1(), 0);
    for (int b = 0; b < B; b++) { magma_zmalloc(&ha[b], A.size()); magma_zmalloc(&hc[b], C.size()); magma_zsetmatrix(n, k, (magmaDoubleComplex*)A.data(), n, ha[b], n, Q); magma_zsetmatrix(n, n, (magmaDoubleComplex*)C.data(), n, hc[b], n, Q); }
    magmaDoubleComplex **dA, **dC; magma_malloc((void**)&dA, B * sizeof(void*)); magma_malloc((void**)&dC, B * sizeof(void*)); cudaMemcpy(dA, ha.data(), B * sizeof(void*), cudaMemcpyHostToDevice); cudaMemcpy(dC, hc.data(), B * sizeof(void*), cudaMemcpyHostToDevice);
    magmablas_zherk_batched(MagmaLower, MagmaNoTrans, n, k, 1.0, (magmaDoubleComplex const* const*)dA, n, 0.5, dC, n, B, Q); magma_queue_sync(Q);
    long long ch = 0; for (int b = 0; b < B; b++) { std::vector<cd> g(C.size()); magma_zgetmatrix(n, n, hc[b], n, (magmaDoubleComplex*)g.data(), n, Q); for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) if (g[i + (size_t)j * n] != cd(7, 0)) ch++; }
    rep("M246", "zherk_batched", "n=31 k=17 L: strictly-upper entries changed (of 1395)", (double)ch, 0, ch > 0, true);
    for (int b = 0; b < B; b++) { magma_free(ha[b]); magma_free(hc[b]); } magma_free(dA); magma_free(dC); }
  // M247: ztrsm_batched alpha = 0 with NaN in B: B must become 0
  { int m = 31, n = 5, B = 3; std::vector<magmaDoubleComplex*> ha(B), hb(B); std::vector<cd> A((size_t)m * m, 0), X((size_t)m * n, cd(NAN, NAN)); for (int i = 0; i < m; i++) A[i + (size_t)i * m] = 2.0;
    for (int b = 0; b < B; b++) { magma_zmalloc(&ha[b], A.size()); magma_zmalloc(&hb[b], X.size()); magma_zsetmatrix(m, m, (magmaDoubleComplex*)A.data(), m, ha[b], m, Q); magma_zsetmatrix(m, n, (magmaDoubleComplex*)X.data(), m, hb[b], m, Q); }
    magmaDoubleComplex **dA, **dB; magma_malloc((void**)&dA, B * sizeof(void*)); magma_malloc((void**)&dB, B * sizeof(void*)); cudaMemcpy(dA, ha.data(), B * sizeof(void*), cudaMemcpyHostToDevice); cudaMemcpy(dB, hb.data(), B * sizeof(void*), cudaMemcpyHostToDevice);
    magmablas_ztrsm_batched(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, m, n, MAGMA_Z_ZERO, dA, m, dB, m, B, Q); magma_queue_sync(Q);
    long long nz = 0; for (int b = 0; b < B; b++) { std::vector<cd> g(X.size()); magma_zgetmatrix(m, n, hb[b], m, (magmaDoubleComplex*)g.data(), m, Q); for (auto& v : g) if (!(v == cd(0, 0))) nz++; }
    rep("M247", "ztrsm_batched", "alpha=0, B=NaN: entries not zero (of 465)", (double)nz, 0, nz > 0, true);
    for (int b = 0; b < B; b++) { magma_free(ha[b]); magma_free(hb[b]); } magma_free(dA); magma_free(dB); }
  // M249: ztrevc3 howmany = A with the optimum workspace vs lwork = 2n: eigen-residual of right vectors of upper-triangular T
  { int n = 64; Sd = 5u; std::vector<cd> T((size_t)n * n, 0); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) T[i + (size_t)j * n] = i == j ? cd(1.0 + j, 0.3 * r1()) : cd(r1(), r1());
    auto run = [&](magma_int_t lw) { std::vector<cd> VR((size_t)n * n, 0), VL(1), work(lw + 1); std::vector<double> rw(n); magma_int_t mo = 0, info = 0;
      magma_ztrevc3(MagmaRight, MagmaAllVec, NULL, n, (magmaDoubleComplex*)T.data(), n, (magmaDoubleComplex*)VL.data(), 1, (magmaDoubleComplex*)VR.data(), n, n, &mo, (magmaDoubleComplex*)work.data(), lw, rw.data(), &info);
      double tn = 0, w = 0; for (auto& t : T) tn = std::max(tn, std::abs(t));
      for (int j = 0; j < n; j++) { double r = 0, vn = 0; for (int i = 0; i < n; i++) { cd s = 0; for (int k = 0; k < n; k++) s += T[i + (size_t)k * n] * VR[k + (size_t)j * n]; r = std::max(r, std::abs(s - T[j + (size_t)j * n] * VR[i + (size_t)j * n])); vn = std::max(vn, std::abs(VR[i + (size_t)j * n])); }
        w = std::max(w, vn > 0 ? r / (tn * vn * n * 1.11e-16) : 1e300); }
      return w; };
    double big = run(n + 2 * n * 128), small = run(2 * n); rep("M249", "ztrevc3", "n=64 howmany=A: residual with lwork=(1+2*128)n vs 2n", big, small, big > 100, small < 100); }
  // M250: zgesv_rbt n = 1, 2, 3 vs 5
  { auto run = [&](int n) { Sd = 9u + n; std::vector<cd> A((size_t)n * n), X(n), B(n, 0); for (auto& a : A) a = cd(r1(), r1()); for (int i = 0; i < n; i++) A[i + (size_t)i * n] += 2.0; for (auto& x : X) x = cd(r1(), r1());
      for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) B[i] += A[i + (size_t)j * n] * X[j]; auto Ac = A, Bc = B; magma_int_t info;
      magma_zgesv_rbt(MagmaFalse, n, 1, (magmaDoubleComplex*)Ac.data(), n, (magmaDoubleComplex*)Bc.data(), n, &info); double e = 0; for (int i = 0; i < n; i++) e = std::max(e, std::abs(Bc[i] - X[i])); return e; };
    double e5 = run(5); for (int n : {1, 2, 3}) { double e = run(n); char c[64]; snprintf(c, sizeof c, "n=%d max|x - x_true| vs n=5", n); rep("M250", "zgesv_rbt", c, e, e5, e > 1e-8, e5 < 1e-8); } }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

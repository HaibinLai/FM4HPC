// Round 53 (roxy): end-to-end eigenvector residual of magma_[dz]geev_m against magma_[dz]geev on matrices that zgebal permutes.
//   round53_geev <z|d> <n> <kind> [seed=1]
//   kind 0: random (gebal usually gives ilo = 1, ihi = n)
//   kind 1: column 0 zero below the diagonal (an isolated eigenvalue: gebal moves it to the top, ilo = 2)
//   kind 2: last k = 3 columns zero below the diagonal (ilo = 4)
//   kind 3: row n-1 zero left of the diagonal (gebal moves it to the bottom, ihi = n-1)
// JOBVL = N, JOBVR = V.  res = max_j |A v_j - w_j v_j| / (|A| |v_j|)  (complex pairs handled for d).
// CSV: r53geev,prec,n,kind,ilo,ihi,routine,info,res
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
typedef std::complex<double> C;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static bool zero_at(int i, int j, int n, int kind) { return (kind == 1 && j == 0 && i > 0) || (kind == 2 && j < 3 && i > j) || (kind == 3 && i == n - 1 && j < n - 1); }
// residual for complex eigenpairs, A given as complex
static double resid(const std::vector<C>& A, int n, const std::vector<C>& w, const std::vector<C>& V) {
  double an = 0; for (auto& a : A) an = std::max(an, std::abs(a)); double e = 0;
  for (int j = 0; j < n; j++) { double vn = 0, d = 0; for (int i = 0; i < n; i++) vn = std::max(vn, std::abs(V[i + (size_t)j * n]));
    for (int i = 0; i < n; i++) { C s = 0; for (int k = 0; k < n; k++) s += A[i + (size_t)k * n] * V[k + (size_t)j * n]; d = std::max(d, std::abs(s - w[j] * V[i + (size_t)j * n])); }
    double r = d / (an * vn); if (!(r <= e)) e = r; }
  return e; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 4) { printf("usage: round53_geev z|d n kind [seed]\n"); return 1; }
  char p = argv[1][0]; magma_int_t n = atoi(argv[2]); int kind = atoi(argv[3]); if (argc > 4) Sd = (unsigned)atoi(argv[4]);
  magma_init();
  size_t nn = (size_t)n * n; magma_int_t ilo = 0, ihi = 0, ierr, one = 1;
  if (p == 'z') {
    std::vector<magmaDoubleComplex> A0(nn);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) A0[i + (size_t)j * n] = zero_at(i, j, n, kind) ? MAGMA_Z_ZERO : MAGMA_Z_MAKE(r1(), r1());
    { std::vector<magmaDoubleComplex> B = A0; std::vector<double> sc(n); lapackf77_zgebal("B", &n, B.data(), &n, &ilo, &ihi, sc.data(), &ierr); }
    std::vector<C> Ac(nn); for (size_t i = 0; i < nn; i++) Ac[i] = C(MAGMA_Z_REAL(A0[i]), MAGMA_Z_IMAG(A0[i]));
    for (int m = 0; m < 2; m++) {
      std::vector<magmaDoubleComplex> A = A0, w(n), VR(nn), VL(1), wq(1); std::vector<double> rw(2 * n); magma_int_t info = 0, lw = -1;
      if (m == 0) magma_zgeev_m(MagmaNoVec, MagmaVec, n, A.data(), n, w.data(), VL.data(), 1, VR.data(), n, wq.data(), lw, rw.data(), &info);
      else        magma_zgeev  (MagmaNoVec, MagmaVec, n, A.data(), n, w.data(), VL.data(), 1, VR.data(), n, wq.data(), lw, rw.data(), &info);
      lw = (magma_int_t)MAGMA_Z_REAL(wq[0]); std::vector<magmaDoubleComplex> wk(std::max<magma_int_t>(lw, 1));
      if (m == 0) magma_zgeev_m(MagmaNoVec, MagmaVec, n, A.data(), n, w.data(), VL.data(), 1, VR.data(), n, wk.data(), lw, rw.data(), &info);
      else        magma_zgeev  (MagmaNoVec, MagmaVec, n, A.data(), n, w.data(), VL.data(), 1, VR.data(), n, wk.data(), lw, rw.data(), &info);
      std::vector<C> wc(n), V(nn); for (int i = 0; i < n; i++) wc[i] = C(MAGMA_Z_REAL(w[i]), MAGMA_Z_IMAG(w[i])); for (size_t i = 0; i < nn; i++) V[i] = C(MAGMA_Z_REAL(VR[i]), MAGMA_Z_IMAG(VR[i]));
      printf("r53geev,z,%lld,%d,%lld,%lld,%s,%lld,%.3g\n", (long long)n, kind, (long long)ilo, (long long)ihi, m == 0 ? "magma_zgeev_m" : "magma_zgeev", (long long)info, resid(Ac, (int)n, wc, V));
    }
  } else {
    std::vector<double> A0(nn);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) A0[i + (size_t)j * n] = zero_at(i, j, n, kind) ? 0.0 : r1();
    { std::vector<double> B = A0, sc(n); lapackf77_dgebal("B", &n, B.data(), &n, &ilo, &ihi, sc.data(), &ierr); }
    std::vector<C> Ac(nn); for (size_t i = 0; i < nn; i++) Ac[i] = A0[i];
    for (int m = 0; m < 2; m++) {
      std::vector<double> A = A0, wr(n), wi(n), VR(nn), VL(1), wq(1); magma_int_t info = 0, lw = -1;
      if (m == 0) magma_dgeev_m(MagmaNoVec, MagmaVec, n, A.data(), n, wr.data(), wi.data(), VL.data(), 1, VR.data(), n, wq.data(), lw, &info);
      else        magma_dgeev  (MagmaNoVec, MagmaVec, n, A.data(), n, wr.data(), wi.data(), VL.data(), 1, VR.data(), n, wq.data(), lw, &info);
      lw = (magma_int_t)wq[0]; std::vector<double> wk(std::max<magma_int_t>(lw, 1));
      if (m == 0) magma_dgeev_m(MagmaNoVec, MagmaVec, n, A.data(), n, wr.data(), wi.data(), VL.data(), 1, VR.data(), n, wk.data(), lw, &info);
      else        magma_dgeev  (MagmaNoVec, MagmaVec, n, A.data(), n, wr.data(), wi.data(), VL.data(), 1, VR.data(), n, wk.data(), lw, &info);
      std::vector<C> wc(n), V(nn);
      for (int j = 0; j < n; j++) { wc[j] = C(wr[j], wi[j]);
        if (wi[j] == 0) { for (int i = 0; i < n; i++) V[i + (size_t)j * n] = VR[i + (size_t)j * n]; }
        else if (j + 1 < n) { for (int i = 0; i < n; i++) { C v(VR[i + (size_t)j * n], VR[i + (size_t)(j + 1) * n]); V[i + (size_t)j * n] = v; V[i + (size_t)(j + 1) * n] = std::conj(v); }
          wc[j + 1] = C(wr[j + 1], wi[j + 1]); j++; } }
      printf("r53geev,d,%lld,%d,%lld,%lld,%s,%lld,%.3g\n", (long long)n, kind, (long long)ilo, (long long)ihi, m == 0 ? "magma_dgeev_m" : "magma_dgeev", (long long)info, resid(Ac, (int)n, wc, V));
    }
  }
  (void)one; fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_finalize(); return 0;
}

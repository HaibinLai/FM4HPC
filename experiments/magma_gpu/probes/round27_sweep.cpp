// Twenty-seventh round (roxy, RTX 5070 Ti sm_120): differential sweep of most dense MAGMA operators against reference
// LAPACK (OpenBLAS 0.3.34) on adversarial inputs: block-boundary sizes, zero columns at the first / middle / last
// position, zero matrices, extreme scaling, NaN and Inf entries.  One operator group per process:
//     round27_sweep <group>      group = lu chol qr solve inv heig geev svd reduce blas batched
// Every case prints one CSV line: group,routine,kind,m,n,info_magma,info_lapack,ratio,verdict,note
// ratio = LAPACK-style normalized residual (error / (norm * n * eps)); verdicts:
//   OK      info equal and ratio <= 30 (or not applicable)
//   INFO    info differs from LAPACK
//   WRONG   info equal (0) but ratio > 1000 on finite input, or output non-finite where LAPACK's is finite
//   WARN    30 < ratio <= 1000
//   NFDIFF  non-finite input: MAGMA and LAPACK disagree on whether the output is non-finite
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <complex>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
typedef std::complex<double> cd;
typedef magmaDoubleComplex Z;
static Z* zp(std::vector<cd>& v) { return (Z*)v.data(); }
static magma_queue_t Q;
static const double EPS = 1.1102230246251565e-16;
static const char* G = "";

// ---------------------------------------------------------------- matrices
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static cd rc() { double a = r1(); return cd(a, r1()); }
static bool finiteM(const std::vector<double>& A) { for (double a : A) if (!std::isfinite(a)) return false; return true; }
static bool finiteM(const std::vector<cd>& A) { for (auto& a : A) if (!std::isfinite(a.real()) || !std::isfinite(a.imag())) return false; return true; }
// general m x n (column major, lda = m)
static std::vector<cd> genG(const std::string& k, int m, int n, unsigned seed) {
  Sd = seed; std::vector<cd> A((size_t)m * n); for (auto& a : A) a = rc();
  auto col = [&](int j) { for (int i = 0; i < m; i++) A[i + (size_t)j * m] = 0; };
  if (m == 0 || n == 0) return A;
  if (k == "zcol0") col(0); else if (k == "zcolmid") col(n / 2); else if (k == "zcollast") col(n - 1);
  else if (k == "zero") std::fill(A.begin(), A.end(), cd(0));
  else if (k == "big") for (auto& a : A) a *= ldexp(1.0, 500); else if (k == "tiny") for (auto& a : A) a *= ldexp(1.0, -500);
  else if (k == "nan") A[m / 2 + (size_t)(n / 2) * m] = cd(NAN, 0); else if (k == "inf") A[0] = cd(INFINITY, 0);
  else if (k == "dupcol" && n > 1) for (int i = 0; i < m; i++) A[i + (size_t)(n - 1) * m] = A[i];
  return A;
}
// Hermitian n x n; "hpd*" kinds are positive definite, "indef*" have one large negative diagonal entry
static std::vector<cd> genH(const std::string& k, int n, unsigned seed) {
  Sd = seed; std::vector<cd> A((size_t)n * n);
  for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { cd v = i == j ? cd(r1(), 0) : rc(); A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = std::conj(v); }
  if (n == 0) return A;
  bool pd = k.rfind("hpd", 0) == 0 || k.rfind("indef", 0) == 0;
  if (pd) for (int i = 0; i < n; i++) A[i + (size_t)i * n] += cd(n + 1.0, 0);
  if (k == "hpd_big" || k == "herm_big") for (auto& a : A) a *= ldexp(1.0, 500);
  if (k == "hpd_tiny" || k == "herm_tiny") for (auto& a : A) a *= ldexp(1.0, -500);
  if (k == "indef0") A[0] = cd(-1e3, 0); if (k == "indefmid") A[n / 2 + (size_t)(n / 2) * n] = cd(-1e3 * n, 0);
  if (k == "indeflast") A[(size_t)n * n - 1] = cd(-1e3 * n, 0);
  if (k == "zero") std::fill(A.begin(), A.end(), cd(0));
  if (k == "ident") { std::fill(A.begin(), A.end(), cd(0)); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = 1; }
  if (k == "nan") A[n / 2 + (size_t)(n / 2) * n] = cd(NAN, 0);
  if (k == "inf") A[0] = cd(INFINITY, 0);
  return A;
}
static const std::vector<int> SIZES = {0, 1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 513};
static const std::vector<int> SIZES_S = {0, 1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257};
static const std::vector<std::string> GKINDS = {"rand", "zcol0", "zcolmid", "zcollast", "zero", "big", "tiny", "dupcol", "nan", "inf"};
static const std::vector<std::string> PKINDS = {"hpd", "hpd_big", "hpd_tiny", "indef0", "indefmid", "indeflast", "zero", "nan", "inf"};
static const std::vector<std::string> HKINDS = {"herm", "herm_big", "herm_tiny", "zero", "ident", "nan", "inf"};

// ---------------------------------------------------------------- small dense helpers (host, O(n^3))
static double nrm1(const std::vector<cd>& A, int m, int n) { double mx = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < m; i++) s += std::abs(A[i + (size_t)j * m]); mx = std::max(mx, s); } return mx; }
static std::vector<cd> mul(const std::vector<cd>& A, int m, int k, const std::vector<cd>& B, int n) { std::vector<cd> C((size_t)m * n);
  for (int j = 0; j < n; j++) for (int l = 0; l < k; l++) { cd b = B[l + (size_t)j * k]; if (b == cd(0)) continue; for (int i = 0; i < m; i++) C[i + (size_t)j * m] += A[i + (size_t)l * m] * b; } return C; }
static double diff1(const std::vector<cd>& A, const std::vector<cd>& B, int m, int n) { std::vector<cd> D(A.size()); for (size_t i = 0; i < A.size(); i++) D[i] = A[i] - B[i]; return nrm1(D, m, n); }
static double ratio(double err, double nrm, int n) { if (n == 0) return 0; double d = nrm * std::max(1, n) * EPS; if (d == 0) return err == 0 ? 0 : INFINITY; return err / d; }
static std::vector<cd> adj(const std::vector<cd>& A, int m, int n) { std::vector<cd> T((size_t)n * m); for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) T[j + (size_t)i * n] = std::conj(A[i + (size_t)j * m]); return T; }

// ---------------------------------------------------------------- reporting
static int cnt[6];
static void out(const char* rt, const std::string& kind, int m, int n, long long im, long long il, double rat, bool inFinite,
                bool magmaFinite, bool lapackFinite, const std::string& note = "") {
  const char* v = "OK";
  if (im != il) v = "INFO";
  else if (!inFinite) { if (magmaFinite != lapackFinite) v = "NFDIFF"; }
  else if (!magmaFinite && lapackFinite) v = "WRONG";
  else if (im == 0 && (std::isnan(rat) || rat > 1000)) v = "WRONG";
  else if (im == 0 && rat > 30) v = "WARN";
  static const char* names[] = {"OK", "INFO", "WRONG", "WARN", "NFDIFF"}; for (int i = 0; i < 5; i++) if (!strcmp(v, names[i])) cnt[i]++;
  printf("%s,%s,%s,%d,%d,%lld,%lld,%.3g,%s,%s\n", G, rt, kind.c_str(), m, n, im, il, rat, v, note.c_str());
}

// device helpers
static Z* up(std::vector<cd>& A, int m, int n, int ldd) { Z* d = NULL; magma_zmalloc(&d, (size_t)std::max(1, ldd) * std::max(1, n)); if (m > 0 && n > 0) magma_zsetmatrix(m, n, zp(A), std::max(1, m), d, ldd, Q); return d; }
static void down(Z* d, int ldd, std::vector<cd>& A, int m, int n) { if (m > 0 && n > 0) magma_zgetmatrix(m, n, d, ldd, zp(A), std::max(1, m), Q); }
static int ldd(int m) { return std::max(1, magma_roundup(m, 32)); }

// ---------------------------------------------------------------- LU
static double luRatio(const std::vector<cd>& A0, const std::vector<cd>& LU, const std::vector<magma_int_t>& ip, int m, int n, bool piv, std::string& note) {
  int k = std::min(m, n); if (k == 0) return 0;
  std::vector<cd> PA = A0;
  if (piv) for (int i = 0; i < k; i++) { int p = (int)ip[i] - 1; if (p < 0 || p >= m) { note = "pivot out of range"; return INFINITY; }
    if (p != i) for (int j = 0; j < n; j++) std::swap(PA[i + (size_t)j * m], PA[p + (size_t)j * m]); }
  std::vector<cd> L((size_t)m * k), U((size_t)k * n);
  for (int j = 0; j < k; j++) for (int i = 0; i < m; i++) L[i + (size_t)j * m] = i == j ? cd(1) : (i > j ? LU[i + (size_t)j * m] : cd(0));
  for (int j = 0; j < n; j++) for (int i = 0; i < k; i++) U[i + (size_t)j * k] = i <= j ? LU[i + (size_t)j * m] : cd(0);
  return ratio(diff1(PA, mul(L, m, k, U, n), m, n), nrm1(A0, m, n), std::max(m, n));
}
static void groupLU() {
  std::vector<std::pair<int, int>> shapes; for (int n : SIZES) shapes.push_back({n, n}); for (int n : {31, 64, 129}) { shapes.push_back({n + 17, n}); shapes.push_back({n, n + 17}); }
  for (auto [m, n] : shapes) for (auto& k : GKINDS) {
    auto A0 = genG(k, m, n, 1000u + m * 7 + n); bool fin = finiteM(A0); int kk = std::min(m, n);
    auto R = A0; std::vector<magma_int_t> ipr(std::max(1, kk)); magma_int_t il = 0, mm = m, nn = n, lda = std::max(1, m);
    lapackf77_zgetrf(&mm, &nn, (magmaDoubleComplex*)R.data(), &lda, ipr.data(), &il); bool lf = finiteM(R);
    for (int variant = 0; variant < 4; variant++) {
      const char* nm[] = {"zgetrf", "zgetrf_gpu", "zgetrf_native", "zgetrf_nopiv_gpu"};
      if (variant == 3 && k != "rand" && k != "big" && k != "tiny" && k != "nan" && k != "inf" && k != "zero" && k != "zcollast") continue;
      auto A = A0; std::vector<magma_int_t> ip(std::max(1, kk), 0); magma_int_t im = -999;
      if (variant == 0) magma_zgetrf(m, n, zp(A), lda, ip.data(), &im);
      else { int ld = ldd(m); Z* d = up(A, m, n, ld);
        if (variant == 1) magma_zgetrf_gpu(m, n, d, ld, ip.data(), &im); else if (variant == 2) magma_zgetrf_native(m, n, d, ld, ip.data(), &im); else magma_zgetrf_nopiv_gpu(m, n, d, ld, &im);
        down(d, ld, A, m, n); magma_free(d); }
      std::string note; long long ilr = il;
      if (variant == 3) { // no-pivot reference: info = first exactly zero diagonal in an unpivoted elimination
        auto U = A0; ilr = 0; for (int j = 0; j < kk && !ilr; j++) { if (U[j + (size_t)j * m] == cd(0)) { ilr = j + 1; break; }
          for (int i = j + 1; i < m; i++) { cd l = U[i + (size_t)j * m] / U[j + (size_t)j * m]; U[i + (size_t)j * m] = l; for (int c = j + 1; c < n; c++) U[i + (size_t)c * m] -= l * U[j + (size_t)c * m]; } } }
      double rat = fin ? luRatio(A0, A, ip, m, n, variant != 3, note) : NAN;
      out(nm[variant], k, m, n, im, ilr, rat, fin, finiteM(A), variant == 3 ? true : lf, note);
    }
  }
}

// ---------------------------------------------------------------- Cholesky
static void groupChol() {
  for (int n : SIZES) for (auto& k : PKINDS) {
    auto A0 = genH(k, n, 2000u + n); bool fin = finiteM(A0); magma_int_t il, nn = n, lda = std::max(1, n);
    auto R = A0; lapackf77_zpotrf("L", &nn, zp(R), &lda, &il); bool lf = finiteM(R);
    for (int v = 0; v < 3; v++) { const char* nm[] = {"zpotrf", "zpotrf_gpu", "zpotrf_native"};
      auto A = A0; magma_int_t im = -999;
      if (v == 0) magma_zpotrf(MagmaLower, n, zp(A), lda, &im);
      else { int ld = ldd(n); Z* d = up(A, n, n, ld); if (v == 1) magma_zpotrf_gpu(MagmaLower, n, d, ld, &im); else magma_zpotrf_native(MagmaLower, n, d, ld, &im); down(d, ld, A, n, n); magma_free(d); }
      double rat = NAN;
      if (fin && im == 0 && n > 0) { std::vector<cd> L((size_t)n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) L[i + (size_t)j * n] = A[i + (size_t)j * n];
        auto LLH = mul(L, n, n, adj(L, n, n), n); std::vector<cd> Afull = A0; rat = ratio(diff1(Afull, LLH, n, n), nrm1(A0, n, n), n); } else if (n == 0) rat = 0;
      out(nm[v], k, n, n, im, il, rat, fin, finiteM(A), lf);
    }
  }
}

// ---------------------------------------------------------------- QR family
static void groupQR() {
  std::vector<std::pair<int, int>> shapes; for (int n : SIZES) shapes.push_back({n, n}); for (int n : {31, 64, 129}) { shapes.push_back({n + 17, n}); shapes.push_back({n, n + 17}); }
  for (auto [m, n] : shapes) for (auto& k : GKINDS) {
    auto A0 = genG(k, m, n, 3000u + 3 * m + n); bool fin = finiteM(A0); int kk = std::min(m, n); magma_int_t mm = m, nn = n, lda = std::max(1, m), il;
    for (int v = 0; v < 7; v++) {
      const char* nm[] = {"zgeqrf", "zgeqrf_gpu", "zgeqrf2_gpu", "zgelqf", "zgelqf_gpu", "zgeqlf", "zgeqp3"};
      if (v == 5 && m < n) continue;
      auto A = A0; std::vector<cd> tau(std::max(1, kk)); magma_int_t im = -999; std::vector<magma_int_t> jp(std::max(1, n), 0);
      magma_int_t nb = magma_get_zgeqrf_nb(m, n); magma_int_t lw = std::max<magma_int_t>(1, std::max(m, n) * std::max<magma_int_t>(nb, 64) * 2 + 2 * nb * nb);
      std::vector<cd> work(lw); std::vector<double> rw(2 * std::max(1, n));
      if (v == 0) magma_zgeqrf(m, n, zp(A), lda, zp(tau), zp(work), lw, &im);
      else if (v == 1 || v == 2) { int ld = ldd(m); Z* d = up(A, m, n, ld); Z* dT = NULL;
        if (v == 1) { magma_zmalloc(&dT, (2 * std::min(m, n) + magma_roundup(n, 32)) * nb + 1); magma_zgeqrf_gpu(m, n, d, ld, zp(tau), dT, &im); }
        else magma_zgeqrf2_gpu(m, n, d, ld, zp(tau), &im);
        down(d, ld, A, m, n); magma_free(d); if (dT) magma_free(dT); }
      else if (v == 3) magma_zgelqf(m, n, zp(A), lda, zp(tau), zp(work), lw, &im);
      else if (v == 4) { int ld = ldd(m); Z* d = up(A, m, n, ld); magma_zgelqf_gpu(m, n, d, ld, zp(tau), zp(work), lw, &im); down(d, ld, A, m, n); magma_free(d); }
      else if (v == 5) magma_zgeqlf(m, n, zp(A), lda, zp(tau), zp(work), lw, &im);
      else magma_zgeqp3(m, n, zp(A), lda, jp.data(), zp(tau), zp(work), lw, rw.data(), &im);
      // reference info is always 0 for these; reconstruct Q from MAGMA's reflectors with LAPACK and check A0 (P) = Q R
      double rat = 0; std::string note;
      if (fin && kk > 0) {
        magma_int_t kq = kk, lwq = 64 * std::max(m, n) + 64; std::vector<cd> wq(lwq);
        if (v <= 2 || v == 6) { std::vector<cd> Qm((size_t)m * kk); for (int j = 0; j < kk; j++) for (int i = 0; i < m; i++) Qm[i + (size_t)j * m] = A[i + (size_t)j * m];
          magma_int_t mk = kk; lapackf77_zungqr(&mm, &mk, &kq, zp(Qm), &lda, zp(tau), zp(wq), &lwq, &il);
          std::vector<cd> Rm((size_t)kk * n); for (int j = 0; j < n; j++) for (int i = 0; i <= std::min(j, kk - 1); i++) Rm[i + (size_t)j * kk] = A[i + (size_t)j * m];
          std::vector<cd> AP = A0; if (v == 6) { for (int j = 0; j < n; j++) { int p = (int)jp[j] - 1; if (p < 0 || p >= n) { note = "jpvt out of range"; rat = INFINITY; break; }
              for (int i = 0; i < m; i++) AP[i + (size_t)j * m] = A0[i + (size_t)p * m]; } }
          if (rat == 0) rat = ratio(diff1(AP, mul(Qm, m, kk, Rm, n), m, n), nrm1(A0, m, n), std::max(m, n)); }
        else if (v == 3 || v == 4) { // A = L Q, Q kk x n from zunglq
          std::vector<cd> Qm((size_t)kk * n); for (int j = 0; j < n; j++) for (int i = 0; i < kk; i++) Qm[i + (size_t)j * kk] = A[i + (size_t)j * m];
          magma_int_t ldq = kk, mk = kk; lapackf77_zunglq(&mk, &nn, &kq, zp(Qm), &ldq, zp(tau), zp(wq), &lwq, &il);
          std::vector<cd> Lm((size_t)m * kk); for (int j = 0; j < kk; j++) for (int i = j; i < m; i++) Lm[i + (size_t)j * m] = A[i + (size_t)j * m];
          rat = ratio(diff1(A0, mul(Lm, m, kk, Qm, n), m, n), nrm1(A0, m, n), std::max(m, n)); }
        else { // QL, m >= n: Q = last n columns from zungql, L lower n x n in rows m-n..m-1
          std::vector<cd> Qm = A; lapackf77_zungql(&mm, &nn, &nn, zp(Qm), &lda, zp(tau), zp(wq), &lwq, &il);
          std::vector<cd> Lm((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if (i >= j) Lm[i + (size_t)j * n] = A[(m - n + i) + (size_t)j * m];
          rat = ratio(diff1(A0, mul(Qm, m, n, Lm, n), m, n), nrm1(A0, m, n), std::max(m, n)); }
      }
      bool tf = finiteM(tau); out(nm[v], k, m, n, im, 0, fin ? rat : NAN, fin, finiteM(A) && tf, true, note);
    }
  }
}

// ---------------------------------------------------------------- linear solvers
static void groupSolve() {
  for (int n : SIZES) for (auto& k : GKINDS) {
    const int nrhs = 3; auto A0 = genG(k, n, n, 4000u + n); Sd = 77u + n; std::vector<cd> B0((size_t)n * nrhs); for (auto& b : B0) b = rc();
    bool fin = finiteM(A0); magma_int_t nn = n, nr = nrhs, lda = std::max(1, n), il;
    auto R = A0, X = B0; std::vector<magma_int_t> ip(std::max(1, n)); lapackf77_zgesv(&nn, &nr, zp(R), &lda, ip.data(), zp(X), &lda, &il); bool lf = finiteM(X);
    auto check = [&](const char* nm, std::vector<cd>& A, std::vector<cd>& Xm, long long im, long long ilr, const std::vector<cd>& Aorig, bool lff) {
      double rat = 0; if (fin && im == 0 && n > 0) { auto AX = mul(Aorig, n, n, Xm, nrhs); rat = ratio(diff1(AX, B0, n, nrhs), nrm1(Aorig, n, n) * nrm1(Xm, n, nrhs), n); }
      out(nm, k, n, n, im, ilr, fin ? rat : NAN, fin, finiteM(Xm), lff); };
    { auto A = A0, Xm = B0; std::vector<magma_int_t> p(std::max(1, n)); magma_int_t im = -999; magma_zgesv(n, nrhs, zp(A), lda, p.data(), zp(Xm), lda, &im); check("zgesv", A, Xm, im, il, A0, lf); }
    { auto A = A0, Xm = B0; std::vector<magma_int_t> p(std::max(1, n)); magma_int_t im = -999; int ld = ldd(n); Z* dA = up(A, n, n, ld); Z* dB = up(Xm, n, nrhs, ld);
      magma_zgesv_gpu(n, nrhs, dA, ld, p.data(), dB, ld, &im); down(dB, ld, Xm, n, nrhs); magma_free(dA); magma_free(dB); check("zgesv_gpu", A, Xm, im, il, A0, lf); }
    // least squares (square system; LAPACK zgels as reference for info)
    { auto A = A0, Xm = B0, Ar = A0, Xr = B0; magma_int_t lw = std::max(1, n) * 64 * 4 + 4096, im = -999, ilg; std::vector<cd> w(lw);
      lapackf77_zgels("N", &nn, &nn, &nr, zp(Ar), &lda, zp(Xr), &lda, zp(w), &lw, &ilg);
      magma_zgels(MagmaNoTrans, n, n, nrhs, zp(A), lda, zp(Xm), lda, zp(w), lw, &im); check("zgels", A, Xm, im, ilg, A0, finiteM(Xr));
      auto A2 = A0, X2 = B0; int ld = ldd(n); Z* dA = up(A2, n, n, ld); Z* dB = up(X2, n, nrhs, ld); magma_int_t im2 = -999;
      magma_zgels_gpu(MagmaNoTrans, n, n, nrhs, dA, ld, dB, ld, zp(w), lw, &im2); down(dB, ld, X2, n, nrhs); magma_free(dA); magma_free(dB); check("zgels_gpu", A2, X2, im2, ilg, A0, finiteM(Xr)); }
  }
  for (int n : SIZES) for (auto& k : PKINDS) { // Hermitian solvers
    const int nrhs = 3; auto A0 = genH(k, n, 5000u + n); Sd = 99u + n; std::vector<cd> B0((size_t)n * nrhs); for (auto& b : B0) b = rc();
    bool fin = finiteM(A0); magma_int_t nn = n, nr = nrhs, lda = std::max(1, n), il, ilh;
    auto R = A0, X = B0; lapackf77_zposv("L", &nn, &nr, zp(R), &lda, zp(X), &lda, &il); bool lf = finiteM(X);
    auto R2 = A0, X2 = B0; std::vector<magma_int_t> ip(std::max(1, n)); magma_int_t lw = 64 * std::max(1, n); std::vector<cd> w(lw);
    lapackf77_zhesv("L", &nn, &nr, zp(R2), &lda, ip.data(), zp(X2), &lda, zp(w), &lw, &ilh); bool lfh = finiteM(X2);
    auto res = [&](const std::vector<cd>& Xm) { if (!fin || n == 0) return 0.0; auto AX = mul(A0, n, n, Xm, nrhs); return ratio(diff1(AX, B0, n, nrhs), nrm1(A0, n, n) * nrm1(Xm, n, nrhs), n); };
    { auto A = A0, Xm = B0; magma_int_t im = -999; magma_zposv(MagmaLower, n, nrhs, zp(A), lda, zp(Xm), lda, &im); out("zposv", k, n, n, im, il, im == 0 ? res(Xm) : 0, fin, finiteM(Xm), lf); }
    { auto Xm = B0; auto A = A0; magma_int_t im = -999; int ld = ldd(n); Z* dA = up(A, n, n, ld); Z* dB = up(Xm, n, nrhs, ld);
      magma_zposv_gpu(MagmaLower, n, nrhs, dA, ld, dB, ld, &im); down(dB, ld, Xm, n, nrhs); magma_free(dA); magma_free(dB); out("zposv_gpu", k, n, n, im, il, im == 0 ? res(Xm) : 0, fin, finiteM(Xm), lf); }
    { auto A = A0, Xm = B0; std::vector<magma_int_t> p(std::max(1, n)); magma_int_t im = -999; magma_zhesv(MagmaLower, n, nrhs, zp(A), lda, p.data(), zp(Xm), lda, &im);
      out("zhesv", k, n, n, im, ilh, im == 0 ? res(Xm) : 0, fin, finiteM(Xm), lfh); }
  }
}

// ---------------------------------------------------------------- inverses
static void groupInv() {
  auto invRatio = [](const std::vector<cd>& A0, const std::vector<cd>& Ai, int n) { if (n == 0) return 0.0; auto P = mul(A0, n, n, Ai, n); for (int i = 0; i < n; i++) P[i + (size_t)i * n] -= 1.0;
    return ratio(nrm1(P, n, n), nrm1(A0, n, n) * nrm1(Ai, n, n), n); };
  for (int n : SIZES) for (auto& k : GKINDS) { // getri_gpu after getrf_gpu
    auto A0 = genG(k, n, n, 6000u + n); bool fin = finiteM(A0); magma_int_t nn = n, lda = std::max(1, n), il, il2 = 0;
    auto R = A0; std::vector<magma_int_t> ip(std::max(1, n)); lapackf77_zgetrf(&nn, &nn, zp(R), &lda, ip.data(), &il);
    if (il == 0) { magma_int_t lw = 64 * std::max(1, n); std::vector<cd> w(lw); lapackf77_zgetri(&nn, zp(R), &lda, ip.data(), zp(w), &lw, &il2); }
    auto A = A0; int ld = ldd(n); Z* d = up(A, n, n, ld); std::vector<magma_int_t> p(std::max(1, n)); magma_int_t im = -999, im2 = 0;
    magma_zgetrf_gpu(n, n, d, ld, p.data(), &im);
    if (im == 0) { magma_int_t lw = n * magma_get_zgetri_nb(n); Z* dw; magma_zmalloc(&dw, std::max<magma_int_t>(1, lw)); magma_zgetri_gpu(n, d, ld, p.data(), dw, lw, &im2); magma_free(dw); }
    down(d, ld, A, n, n); magma_free(d);
    out("zgetri_gpu", k, n, n, im ? im : im2, il ? il : il2, (fin && im == 0 && im2 == 0) ? invRatio(A0, A, n) : 0, fin, finiteM(A), finiteM(R));
  }
  for (int n : SIZES) for (auto& k : PKINDS) { // potri after potrf (lower), full inverse reconstructed from the lower triangle
    auto A0 = genH(k, n, 7000u + n); bool fin = finiteM(A0); magma_int_t nn = n, lda = std::max(1, n), il, il2 = 0;
    auto R = A0; lapackf77_zpotrf("L", &nn, zp(R), &lda, &il); if (il == 0) lapackf77_zpotri("L", &nn, zp(R), &lda, &il2);
    for (int v = 0; v < 2; v++) { auto A = A0; magma_int_t im = -999, im2 = 0;
      if (v == 0) { magma_zpotrf(MagmaLower, n, zp(A), lda, &im); if (im == 0) magma_zpotri(MagmaLower, n, zp(A), lda, &im2); }
      else { int ld = ldd(n); Z* d = up(A, n, n, ld); magma_zpotrf_gpu(MagmaLower, n, d, ld, &im); if (im == 0) magma_zpotri_gpu(MagmaLower, n, d, ld, &im2); down(d, ld, A, n, n); magma_free(d); }
      std::vector<cd> F((size_t)n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { F[i + (size_t)j * n] = A[i + (size_t)j * n]; F[j + (size_t)i * n] = std::conj(A[i + (size_t)j * n]); }
      out(v == 0 ? "zpotri" : "zpotri_gpu", k, n, n, im ? im : im2, il ? il : il2, (fin && im == 0 && im2 == 0) ? invRatio(A0, F, n) : 0, fin, finiteM(F), finiteM(R)); }
  }
  for (int n : SIZES) for (std::string k : {"rand", "zdiag0", "zdiaglast", "nan", "big"}) { // trtri upper non-unit
    Sd = 8000u + n; std::vector<cd> A0((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) A0[i + (size_t)j * n] = i == j ? cd(2 + r1(), 0) : rc() * (1.0 / std::max(1, n));
    if (n > 0) { if (k == "zdiag0") A0[0] = 0; if (k == "zdiaglast") A0[(size_t)n * n - 1] = 0; if (k == "nan") A0[(n / 2) * (size_t)(n + 1)] = NAN; if (k == "big") for (auto& a : A0) a *= ldexp(1.0, 500); }
    bool fin = finiteM(A0); magma_int_t nn = n, lda = std::max(1, n), il; auto R = A0; lapackf77_ztrtri("U", "N", &nn, zp(R), &lda, &il);
    for (int v = 0; v < 2; v++) { auto A = A0; magma_int_t im = -999;
      if (v == 0) magma_ztrtri(MagmaUpper, MagmaNonUnit, n, zp(A), lda, &im); else { int ld = ldd(n); Z* d = up(A, n, n, ld); magma_ztrtri_gpu(MagmaUpper, MagmaNonUnit, n, d, ld, &im); down(d, ld, A, n, n); magma_free(d); }
      std::vector<cd> U((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) U[i + (size_t)j * n] = A[i + (size_t)j * n];
      out(v == 0 ? "ztrtri" : "ztrtri_gpu", k, n, n, im, il, (fin && im == 0) ? invRatio(A0, U, n) : 0, fin, finiteM(U), finiteM(R)); }
  }
}

// ---------------------------------------------------------------- Hermitian eigensolvers
static void groupHeig() {
  for (int n : SIZES_S) for (auto& k : HKINDS) for (int wantz = 0; wantz < 2; wantz++) {
    auto A0 = genH(k, n, 9000u + n); bool fin = finiteM(A0); magma_int_t nn = n, lda = std::max(1, n), il;
    std::vector<double> wref(std::max(1, n)); { auto R = A0; magma_int_t lw = 4 * std::max(1, n); std::vector<cd> w(lw); std::vector<double> rw(3 * std::max(1, n));
      lapackf77_zheev("N", "L", &nn, zp(R), &lda, wref.data(), zp(w), &lw, rw.data(), &il); }
    double anrm = fin ? nrm1(A0, n, n) : 0; magma_vec_t jz = wantz ? MagmaVec : MagmaNoVec;
    magma_int_t lw = 2 * n + n * n + n * 64 + 4096 + 400 * n, lrw = 1 + 5 * n + 2 * n * n + 4096, liw = 3 + 5 * n + 64;
    auto eval = [&](const char* nm, magma_int_t im, std::vector<double>& w, int mfound, const std::vector<cd>* Zv, int ldz, bool outFin) {
      double rat = 0; if (fin && im == 0 && n > 0) { double e = 0; for (int i = 0; i < mfound; i++) e = std::max(e, std::fabs(w[i] - wref[i])); rat = ratio(e, anrm, n);
        if (mfound != n) rat = INFINITY;
        if (Zv && rat <= 1000) { std::vector<cd> Zm((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) Zm[i + (size_t)j * n] = (*Zv)[i + (size_t)j * ldz];
          auto AZ = mul(A0, n, n, Zm, n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) AZ[i + (size_t)j * n] -= Zm[i + (size_t)j * n] * w[j];
          rat = std::max(rat, ratio(nrm1(AZ, n, n), anrm, n)); auto ZZ = mul(adj(Zm, n, n), n, n, Zm, n); for (int i = 0; i < n; i++) ZZ[i + (size_t)i * n] -= 1.0; rat = std::max(rat, ratio(nrm1(ZZ, n, n), 1.0, n)); } }
      std::string nmz = std::string(nm) + (wantz ? "(V)" : "(N)"); out(nmz.c_str(), k, n, n, im, il, fin ? rat : NAN, fin, outFin, true); };
    { auto A = A0; std::vector<double> w(std::max(1, n)); std::vector<cd> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); magma_int_t im = -999;
      magma_zheevd(jz, MagmaLower, n, zp(A), lda, w.data(), zp(wk), lw, rw.data(), lrw, iw.data(), liw, &im); eval("zheevd", im, w, n, wantz ? &A : NULL, lda, finiteM(A)); }
    { auto A = A0; std::vector<double> w(std::max(1, n)); std::vector<cd> wk(lw), wA((size_t)std::max(1, n) * n); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); magma_int_t im = -999;
      int ld = ldd(n); Z* d = up(A, n, n, ld); magma_zheevd_gpu(jz, MagmaLower, n, d, ld, w.data(), zp(wA), lda, zp(wk), lw, rw.data(), lrw, iw.data(), liw, &im); down(d, ld, A, n, n); magma_free(d);
      eval("zheevd_gpu", im, w, n, wantz ? &A : NULL, lda, finiteM(A)); }
    { auto A = A0; std::vector<double> w(std::max(1, n)); std::vector<cd> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); magma_int_t im = -999, mo = 0;
      magma_zheevdx(jz, MagmaRangeAll, MagmaLower, n, zp(A), lda, 0, 0, 1, n, &mo, w.data(), zp(wk), lw, rw.data(), lrw, iw.data(), liw, &im); eval("zheevdx", im, w, (int)mo, wantz ? &A : NULL, lda, finiteM(A)); }
    { auto A = A0; std::vector<double> w(std::max(1, n)); std::vector<cd> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); magma_int_t im = -999, mo = 0;
      magma_zheevdx_2stage(jz, MagmaRangeAll, MagmaLower, n, zp(A), lda, 0, 0, 1, n, &mo, w.data(), zp(wk), lw, rw.data(), lrw, iw.data(), liw, &im); eval("zheevdx_2stage", im, w, (int)mo, wantz ? &A : NULL, lda, finiteM(A)); }
    { auto A = A0; std::vector<double> w(std::max(1, n)); std::vector<cd> wk(lw), Zm((size_t)std::max(1, n) * n); std::vector<double> rw(lrw + 24 * n); std::vector<magma_int_t> iw(liw + 10 * n), isup(2 * std::max(1, n)); magma_int_t im = -999, mo = 0;
      magma_zheevr(jz, MagmaRangeAll, MagmaLower, n, zp(A), lda, 0, 0, 1, n, 0, &mo, w.data(), zp(Zm), lda, isup.data(), zp(wk), lw, rw.data(), lrw + 24 * n, iw.data(), liw + 10 * n, &im);
      eval("zheevr", im, w, (int)mo, wantz ? &Zm : NULL, lda, finiteM(Zm) || !wantz); }
    { auto A = A0; std::vector<double> w(std::max(1, n)); std::vector<cd> wk(lw), Zm((size_t)std::max(1, n) * n); std::vector<double> rw(7 * std::max(1, n)); std::vector<magma_int_t> iw(5 * std::max(1, n)), ifl(std::max(1, n)); magma_int_t im = -999, mo = 0;
      magma_zheevx(jz, MagmaRangeAll, MagmaLower, n, zp(A), lda, 0, 0, 1, n, 0, &mo, w.data(), zp(Zm), lda, zp(wk), lw, rw.data(), iw.data(), ifl.data(), &im);
      eval("zheevx", im, w, (int)mo, wantz ? &Zm : NULL, lda, finiteM(Zm) || !wantz); }
    { auto A = A0; std::vector<cd> B((size_t)n * n); for (int i = 0; i < n; i++) B[i + (size_t)i * n] = 1; std::vector<double> w(std::max(1, n)); std::vector<cd> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); magma_int_t im = -999;
      magma_zhegvd(1, jz, MagmaLower, n, zp(A), lda, zp(B), lda, w.data(), zp(wk), lw, rw.data(), lrw, iw.data(), liw, &im); eval("zhegvd(B=I)", im, w, n, wantz ? &A : NULL, lda, finiteM(A)); }
  }
}

// ---------------------------------------------------------------- nonsymmetric eigen, SVD, reductions
static void groupGeev() {
  for (int n : SIZES_S) for (auto& k : GKINDS) {
    auto A0 = genG(k, n, n, 10000u + n); bool fin = finiteM(A0); magma_int_t nn = n, lda = std::max(1, n), il;
    std::vector<cd> wr(std::max(1, n)), VLd(1), VRr((size_t)std::max(1, n) * n); { auto R = A0; magma_int_t lw = 4 * std::max(1, n) + 64; std::vector<cd> w(lw); std::vector<double> rw(2 * std::max(1, n)); magma_int_t one = 1;
      lapackf77_zgeev("N", "V", &nn, zp(R), &lda, zp(wr), zp(VLd), &one, zp(VRr), &lda, zp(w), &lw, rw.data(), &il); }
    auto A = A0; std::vector<cd> w(std::max(1, n)), VL(1), VR((size_t)std::max(1, n) * n); magma_int_t lw = (1 + 2 * 64) * std::max(1, n) + 4096; std::vector<cd> wk(lw); std::vector<double> rw(2 * std::max(1, n)); magma_int_t im = -999;
    magma_zgeev(MagmaNoVec, MagmaVec, n, zp(A), lda, zp(w), zp(VL), 1, zp(VR), lda, zp(wk), lw, rw.data(), &im);
    double rat = 0; if (fin && im == 0 && n > 0) { auto AV = mul(A0, n, n, VR, n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) AV[i + (size_t)j * n] -= VR[i + (size_t)j * n] * w[j];
      rat = ratio(nrm1(AV, n, n), nrm1(A0, n, n) * std::max(1.0, nrm1(VR, n, n)), n); }
    out("zgeev", k, n, n, im, il, fin ? rat : NAN, fin, finiteM(VR) && finiteM(w), finiteM(VRr) && finiteM(wr));
  }
}
static void groupSVD() {
  std::vector<std::pair<int, int>> shapes; for (int n : SIZES_S) shapes.push_back({n, n}); for (int n : {31, 64, 129}) { shapes.push_back({n + 17, n}); shapes.push_back({n, n + 17}); }
  for (auto [m, n] : shapes) for (auto& k : GKINDS) {
    if (k == "inf") continue;  // reference LAPACK (OpenBLAS 0.3.34) zgesvd/zgesdd with vectors hang on Inf input (round27_focus svdinf)
    auto A0 = genG(k, m, n, 11000u + 5 * m + n); bool fin = finiteM(A0); int kk = std::min(m, n); magma_int_t mm = m, nn = n, lda = std::max(1, m), il;
    std::vector<double> sref(std::max(1, kk)); { auto R = A0; std::vector<cd> U(1), VT(1); magma_int_t one = 1, lw = 4 * std::max(m, n) + 64; std::vector<cd> w(lw); std::vector<double> rw(5 * std::max(1, kk));
      lapackf77_zgesvd("N", "N", &mm, &nn, zp(R), &lda, sref.data(), zp(U), &one, zp(VT), &one, zp(w), &lw, rw.data(), &il); }
    double anrm = fin ? nrm1(A0, m, n) : 0;
    for (int v = 0; v < 2; v++) {
      auto A = A0; std::vector<double> s(std::max(1, kk)); std::vector<cd> U((size_t)std::max(1, m) * std::max(1, kk)), VT((size_t)std::max(1, kk) * std::max(1, n));
      magma_int_t lw = 3 * kk * kk + 4 * std::max(m, n) * 64 + 8 * kk * kk + 65536, im = -999; std::vector<cd> wk(lw); std::vector<double> rw(5 * kk * kk + 7 * kk + 64); std::vector<magma_int_t> iw(8 * std::max(1, kk));
      magma_int_t ldu = std::max(1, m), ldvt = std::max(1, kk);
      if (v == 0) magma_zgesvd(MagmaSomeVec, MagmaSomeVec, m, n, zp(A), lda, s.data(), zp(U), ldu, zp(VT), ldvt, zp(wk), lw, rw.data(), &im);
      else magma_zgesdd(MagmaSomeVec, m, n, zp(A), lda, s.data(), zp(U), ldu, zp(VT), ldvt, zp(wk), lw, rw.data(), iw.data(), &im);
      double rat = 0; if (fin && im == 0 && kk > 0) { double e = 0; for (int i = 0; i < kk; i++) e = std::max(e, std::fabs(s[i] - sref[i])); rat = ratio(e, anrm, std::max(m, n));
        std::vector<cd> US((size_t)m * kk); for (int j = 0; j < kk; j++) for (int i = 0; i < m; i++) US[i + (size_t)j * m] = U[i + (size_t)j * ldu] * s[j];
        rat = std::max(rat, ratio(diff1(A0, mul(US, m, kk, VT, n), m, n), anrm, std::max(m, n))); }
      out(v == 0 ? "zgesvd(S)" : "zgesdd(S)", k, m, n, im, il, fin ? rat : NAN, fin, finiteM(s) && finiteM(U) && finiteM(VT), true);
    }
  }
}
static void groupReduce() {
  for (int n : SIZES) for (auto& k : HKINDS) { // hetrd: d exact, |e| vs LAPACK
    auto A0 = genH(k, n, 12000u + n); bool fin = finiteM(A0); magma_int_t nn = n, lda = std::max(1, n), il, lw = 64 * std::max(1, n) + 4096;
    std::vector<double> dr(std::max(1, n)), er(std::max(1, n)); std::vector<cd> tr(std::max(1, n)), w(lw); auto R = A0; lapackf77_zhetrd("L", &nn, zp(R), &lda, dr.data(), er.data(), zp(tr), zp(w), &lw, &il);
    auto A = A0; std::vector<double> d(std::max(1, n)), e(std::max(1, n)); std::vector<cd> t(std::max(1, n)); magma_int_t im = -999; magma_zhetrd(MagmaLower, n, zp(A), lda, d.data(), e.data(), zp(t), zp(w), lw, &im);
    double err = 0; for (int i = 0; i < n; i++) err = std::max(err, std::fabs(d[i] - dr[i])); for (int i = 0; i + 1 < n; i++) err = std::max(err, std::fabs(std::fabs(e[i]) - std::fabs(er[i])));
    out("zhetrd", k, n, n, im, il, fin ? ratio(err, nrm1(A0, n, n), n) : NAN, fin, finiteM(std::vector<cd>(d.begin(), d.end())), finiteM(std::vector<cd>(dr.begin(), dr.end())));
  }
  std::vector<std::pair<int, int>> shapes; for (int n : SIZES) shapes.push_back({n, n}); for (int n : {31, 64, 129}) { shapes.push_back({n + 17, n}); shapes.push_back({n, n + 17}); }
  for (auto [m, n] : shapes) for (auto& k : GKINDS) { // gebrd: |d|, |e| vs LAPACK
    auto A0 = genG(k, m, n, 13000u + m + 3 * n); bool fin = finiteM(A0); int kk = std::min(m, n); magma_int_t mm = m, nn = n, lda = std::max(1, m), il, lw = (m + n) * 64 + 4096;
    std::vector<double> dr(std::max(1, kk)), er(std::max(1, kk)); std::vector<cd> tq(std::max(1, kk)), tp(std::max(1, kk)), w(lw); auto R = A0;
    lapackf77_zgebrd(&mm, &nn, zp(R), &lda, dr.data(), er.data(), zp(tq), zp(tp), zp(w), &lw, &il);
    auto A = A0; std::vector<double> d(std::max(1, kk)), e(std::max(1, kk)); magma_int_t im = -999; magma_zgebrd(m, n, zp(A), lda, d.data(), e.data(), zp(tq), zp(tp), zp(w), lw, &im);
    double err = 0; for (int i = 0; i < kk; i++) err = std::max(err, std::fabs(std::fabs(d[i]) - std::fabs(dr[i]))); for (int i = 0; i + 1 < kk; i++) err = std::max(err, std::fabs(std::fabs(e[i]) - std::fabs(er[i])));
    out("zgebrd", k, m, n, im, il, fin ? ratio(err, nrm1(A0, m, n), std::max(m, n)) : NAN, fin, finiteM(std::vector<cd>(d.begin(), d.end())), finiteM(std::vector<cd>(dr.begin(), dr.end())));
  }
}

// ---------------------------------------------------------------- magmablas BLAS-like kernels vs host reference
static void groupBlas() {
  const cd NaN(NAN, NAN);
  for (int n : {0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257}) for (int k : {0, 1, 17, 64}) {
    // gemm C = alpha A B + beta C, beta = 0 with NaN in C (BLAS: C must not be read), and alpha = 0
    for (int cse = 0; cse < 3; cse++) { Sd = 14000u + n + k; std::vector<cd> A((size_t)n * k), B((size_t)k * n), C((size_t)n * n); for (auto& a : A) a = rc(); for (auto& b : B) b = rc(); for (auto& c : C) c = rc();
      cd alpha = cse == 2 ? cd(0) : cd(1.5, -0.5), beta = cse == 0 ? cd(0.5, 0.25) : cd(0); if (cse >= 1) std::fill(C.begin(), C.end(), NaN);
      std::vector<cd> ref = mul(A, n, k, B, n); for (size_t i = 0; i < ref.size(); i++) ref[i] = alpha * ref[i] + (beta == cd(0) ? cd(0) : beta * C[i]);
      int ld = ldd(n), ldk = ldd(k); Z* dA = up(A, n, k, ld); Z* dB = up(B, k, n, ldk); Z* dC = up(C, n, n, ld);
      magmablas_zgemm(MagmaNoTrans, MagmaNoTrans, n, n, k, *(Z*)&alpha, dA, ld, dB, ldk, *(Z*)&beta, dC, ld, Q); std::vector<cd> R(C.size()); down(dC, ld, R, n, n);
      magma_free(dA); magma_free(dB); magma_free(dC);
      const char* nm[] = {"magmablas_zgemm", "magmablas_zgemm beta=0,C=NaN", "magmablas_zgemm alpha=0,beta=0,C=NaN"};
      out(nm[cse], "k=" + std::to_string(k), n, n, 0, 0, ratio(diff1(R, ref, n, n), std::max(1.0, nrm1(A, n, k) * nrm1(B, k, n)), std::max(1, k)), true, finiteM(R), finiteM(ref)); }
  }
  for (int n : {0, 1, 2, 31, 32, 33, 64, 65, 129, 257, 513}) {
    Sd = 15000u + n; std::vector<cd> A((size_t)n * n), x(n), y(n); for (auto& a : A) a = rc(); for (auto& v : x) v = rc();
    // gemv and hemv with beta = 0 and y = NaN
    for (int herm = 0; herm < 2; herm++) { std::vector<cd> H = A; if (herm) for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if (i < j) H[i + (size_t)j * n] = std::conj(H[j + (size_t)i * n]); else if (i == j) H[i + (size_t)i * n] = H[i + (size_t)i * n].real();
      std::fill(y.begin(), y.end(), cd(NAN, NAN)); auto ref = mul(H, n, n, x, 1); int ld = ldd(n); Z* dA = up(H, n, n, ld); Z* dx = up(x, n, 1, std::max(1, n)); Z* dy = up(y, n, 1, std::max(1, n));
      Z one = MAGMA_Z_ONE, zero = MAGMA_Z_ZERO;
      if (herm) magmablas_zhemv(MagmaLower, n, one, dA, ld, dx, 1, zero, dy, 1, Q); else magmablas_zgemv(MagmaNoTrans, n, n, one, dA, ld, dx, 1, zero, dy, 1, Q);
      std::vector<cd> R(n); down(dy, std::max(1, n), R, n, 1); magma_free(dA); magma_free(dx); magma_free(dy);
      out(herm ? "magmablas_zhemv beta=0,y=NaN" : "magmablas_zgemv beta=0,y=NaN", "rand", n, n, 0, 0, ratio(diff1(R, ref, n, 1), std::max(1.0, nrm1(H, n, n) * nrm1(x, n, 1)), n), true, finiteM(R), true); }
    // norms: lange / lanhe with a NaN entry (LAPACK 3.x propagates NaN) and on finite input
    for (int withnan = 0; withnan < 2; withnan++) { std::vector<cd> M = A; if (withnan && n > 0) M[(n / 2) * (size_t)(n + 1)] = cd(NAN, 0);
      for (int herm = 0; herm < 2; herm++) { std::vector<cd> H = M; if (herm) for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) H[i + (size_t)j * n] = std::conj(H[j + (size_t)i * n]);
        for (magma_norm_t nt : {MagmaOneNorm, MagmaInfNorm, MagmaMaxNorm, MagmaFrobeniusNorm}) {
          if (!herm && nt == MagmaFrobeniusNorm) continue; if (herm && nt == MagmaFrobeniusNorm) continue;
          char c = nt == MagmaOneNorm ? '1' : nt == MagmaInfNorm ? 'I' : nt == MagmaMaxNorm ? 'M' : 'F'; magma_int_t nn = n, lda = std::max(1, n); std::vector<double> wk(std::max(1, n));
          double ref = herm ? lapackf77_zlanhe(&c, "L", &nn, zp(H), &lda, wk.data()) : lapackf77_zlange(&c, &nn, &nn, zp(H), &lda, wk.data());
          int ld = ldd(n); Z* dA = up(H, n, n, ld); double* dw; magma_dmalloc(&dw, std::max(1, n) * 4 + 1);
          double got = herm ? magmablas_zlanhe(nt, MagmaLower, n, dA, ld, dw, std::max(1, n) * 4 + 1, Q) : magmablas_zlange(nt, n, n, dA, ld, dw, std::max(1, n) * 4 + 1, Q);
          magma_free(dA); magma_free(dw);
          std::string nm = std::string(herm ? "magmablas_zlanhe(" : "magmablas_zlange(") + c + ")";
          bool same = (std::isnan(ref) && std::isnan(got)) || std::fabs(got - ref) <= 1e-12 * std::max(1.0, std::fabs(ref));
          out(nm.c_str(), withnan ? "nan" : "rand", n, n, 0, 0, same ? 0 : (std::isnan(got) || std::isnan(ref) ? NAN : std::fabs(got - ref) / (std::max(1e-300, std::fabs(ref)) * EPS)), !withnan, !std::isnan(got), !std::isnan(ref),
              "got=" + std::to_string(got) + " ref=" + std::to_string(ref)); } } }
    // transpose, lacpy (full/lower/upper), laset, lascl, geadd
    { int ld = ldd(n); Z* dA = up(A, n, n, ld); Z* dB; magma_zmalloc(&dB, (size_t)ld * std::max(1, n));
      magmablas_ztranspose(n, n, dA, ld, dB, ld, Q); std::vector<cd> T((size_t)n * n); down(dB, ld, T, n, n); std::vector<cd> Tr((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) Tr[j + (size_t)i * n] = A[i + (size_t)j * n];
      out("magmablas_ztranspose", "rand", n, n, 0, 0, diff1(T, Tr, n, n) == 0 ? 0 : INFINITY, true, true, true);
      for (magma_uplo_t ul : {MagmaFull, MagmaLower, MagmaUpper}) { std::vector<cd> Z0((size_t)n * n, cd(7, 7)); Z* dC = up(Z0, n, n, ld); magmablas_zlacpy(ul, n, n, dA, ld, dC, ld, Q); std::vector<cd> Cc((size_t)n * n); down(dC, ld, Cc, n, n); magma_free(dC);
        std::vector<cd> ref = Z0; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if (ul == MagmaFull || (ul == MagmaLower && i >= j) || (ul == MagmaUpper && i <= j)) ref[i + (size_t)j * n] = A[i + (size_t)j * n];
        out(ul == MagmaFull ? "magmablas_zlacpy(F)" : ul == MagmaLower ? "magmablas_zlacpy(L)" : "magmablas_zlacpy(U)", "rand", n, n, 0, 0, diff1(Cc, ref, n, n) == 0 ? 0 : INFINITY, true, true, true); }
      for (magma_uplo_t ul : {MagmaFull, MagmaLower, MagmaUpper}) { std::vector<cd> Z0 = A; Z* dC = up(Z0, n, n, ld); Z off = MAGMA_Z_MAKE(2, 0), dg = MAGMA_Z_MAKE(3, 0); magmablas_zlaset(ul, n, n, off, dg, dC, ld, Q);
        std::vector<cd> Cc((size_t)n * n); down(dC, ld, Cc, n, n); magma_free(dC); std::vector<cd> ref = Z0;
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { bool in = ul == MagmaFull || (ul == MagmaLower && i >= j) || (ul == MagmaUpper && i <= j); if (in) ref[i + (size_t)j * n] = i == j ? cd(3) : cd(2); }
        out(ul == MagmaFull ? "magmablas_zlaset(F)" : ul == MagmaLower ? "magmablas_zlaset(L)" : "magmablas_zlaset(U)", "rand", n, n, 0, 0, diff1(Cc, ref, n, n) == 0 ? 0 : INFINITY, true, true, true); }
      for (auto sc : std::vector<std::pair<double, double>>{{1.0, 2.0}, {ldexp(1.0, 600), 1.0}, {1.0, ldexp(1.0, 600)}, {ldexp(1.0, -600), 1.0}}) {
        std::vector<cd> Z0 = A; Z* dC = up(Z0, n, n, ld); magma_int_t inf = -999; magmablas_zlascl(MagmaFull, 0, 0, sc.first, sc.second, n, n, dC, ld, Q, &inf);
        std::vector<cd> Cc((size_t)n * n); down(dC, ld, Cc, n, n); magma_free(dC); std::vector<cd> ref = Z0; magma_int_t nn = n, lda = std::max(1, n), zz = 0, il; double cf = sc.first, ct = sc.second;
        lapackf77_zlascl("G", &zz, &zz, &cf, &ct, &nn, &nn, zp(ref), &lda, &il);
        double e = 0, mx = 0; for (size_t i = 0; i < ref.size(); i++) { e = std::max(e, std::abs(Cc[i] - ref[i])); mx = std::max(mx, std::abs(ref[i])); }
        out("magmablas_zlascl", "cfrom=" + std::to_string(cf) + ",cto=" + std::to_string(ct), n, n, inf, il, mx > 0 ? e / (mx * EPS) : (e > 0 ? INFINITY : 0), true, finiteM(Cc), finiteM(ref)); }
      { std::vector<cd> Bv = A; for (auto& b : Bv) b = b * 0.5 + cd(1, 0); Z* dC = up(Bv, n, n, ld); Z al = MAGMA_Z_MAKE(2, 1); magmablas_zgeadd(n, n, al, dA, ld, dC, ld, Q);
        std::vector<cd> Cc((size_t)n * n); down(dC, ld, Cc, n, n); magma_free(dC); std::vector<cd> ref = Bv; for (size_t i = 0; i < ref.size(); i++) ref[i] += cd(2, 1) * A[i];
        out("magmablas_zgeadd", "rand", n, n, 0, 0, ratio(diff1(Cc, ref, n, n), std::max(1.0, nrm1(ref, n, n)), 1), true, finiteM(Cc), true); }
      magma_free(dA); magma_free(dB); }
  }
}

// ---------------------------------------------------------------- batched
static void groupBatched() {
  for (int n : {1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65}) for (std::string k : {"rand", "zcol0", "zcollast", "nan", "mixed"}) {
    const int batch = 40; std::vector<std::vector<cd>> A0(batch); std::vector<std::string> kinds(batch);
    for (int b = 0; b < batch; b++) { kinds[b] = k == "mixed" ? GKINDS[b % GKINDS.size()] : k; A0[b] = genG(kinds[b], n, n, 16000u + 97u * b + n); }
    int ld = ldd(n); size_t per = (size_t)ld * n; Z* dA; magma_zmalloc(&dA, per * batch); Z* dInv; magma_zmalloc(&dInv, per * batch); magma_int_t *dpiv, *dinfo; magma_imalloc(&dpiv, (size_t)n * batch); magma_imalloc(&dinfo, batch);
    Z **dAa, **dIa; magma_int_t** dPa; magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dIa, batch * sizeof(Z*)); magma_malloc((void**)&dPa, batch * sizeof(void*));
    std::vector<Z*> hA(batch), hI(batch); std::vector<magma_int_t*> hP(batch); for (int b = 0; b < batch; b++) { hA[b] = dA + b * per; hI[b] = dInv + b * per; hP[b] = dpiv + (size_t)b * n; }
    cudaMemcpy(dAa, hA.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dIa, hI.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dPa, hP.data(), batch * sizeof(void*), cudaMemcpyHostToDevice);
    auto load = [&](std::vector<std::vector<cd>>& M) { for (int b = 0; b < batch; b++) magma_zsetmatrix(n, n, zp(M[b]), n, hA[b], ld, Q); };
    // getrf_batched
    load(A0); magma_int_t ret = magma_zgetrf_batched(n, n, dAa, ld, dPa, dinfo, batch, Q); magma_queue_sync(Q);
    std::vector<magma_int_t> inf(batch); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, inf.data(), 1, Q);
    for (int b = 0; b < batch; b++) { std::vector<cd> LU((size_t)n * n); magma_zgetmatrix(n, n, hA[b], ld, zp(LU), n, Q); std::vector<magma_int_t> p(n); magma_getvector(n, sizeof(magma_int_t), hP[b], 1, p.data(), 1, Q);
      auto R = A0[b]; std::vector<magma_int_t> pr(n); magma_int_t nn = n, il; lapackf77_zgetrf(&nn, &nn, zp(R), &nn, pr.data(), &il); bool fin = finiteM(A0[b]); std::string note;
      double rat = fin ? luRatio(A0[b], LU, p, n, n, true, note) : NAN; if (ret) note += " ret=" + std::to_string(ret);
      out("zgetrf_batched", kinds[b] + "#" + std::to_string(b), n, n, inf[b], il, rat, fin, finiteM(LU), finiteM(R), note); }
    // getri_outofplace_batched on the factors (only meaningful where info = 0)
    magma_zgetri_outofplace_batched(n, dAa, ld, dPa, dIa, ld, dinfo, batch, Q); magma_queue_sync(Q); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, inf.data(), 1, Q);
    for (int b = 0; b < batch; b++) { if (!finiteM(A0[b])) continue; auto R = A0[b]; std::vector<magma_int_t> pr(n); magma_int_t nn = n, il, il2 = 0, lw = 64 * n; std::vector<cd> w(lw);
      lapackf77_zgetrf(&nn, &nn, zp(R), &nn, pr.data(), &il); if (il == 0) lapackf77_zgetri(&nn, zp(R), &nn, pr.data(), zp(w), &lw, &il2);
      std::vector<cd> Ai((size_t)n * n); magma_zgetmatrix(n, n, hI[b], ld, zp(Ai), n, Q); double rat = 0;
      if (inf[b] == 0 && il == 0) { auto P = mul(A0[b], n, n, Ai, n); for (int i = 0; i < n; i++) P[i + (size_t)i * n] -= 1.0; rat = ratio(nrm1(P, n, n), nrm1(A0[b], n, n) * nrm1(Ai, n, n), n); }
      out("zgetri_outofplace_batched", kinds[b] + "#" + std::to_string(b), n, n, inf[b], il ? il : il2, rat, true, finiteM(Ai), true); }
    // potrf_batched on Hermitian positive definite versions (kinds mapped to indefinite / NaN variants)
    std::vector<std::vector<cd>> H0(batch); std::vector<std::string> hk(batch);
    for (int b = 0; b < batch; b++) { hk[b] = k == "mixed" ? PKINDS[b % PKINDS.size()] : (k == "rand" ? "hpd" : k == "zcol0" ? "indef0" : k == "zcollast" ? "indeflast" : "nan"); H0[b] = genH(hk[b], n, 17000u + 31u * b + n); }
    load(H0); magma_zpotrf_batched(MagmaLower, n, dAa, ld, dinfo, batch, Q); magma_queue_sync(Q); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, inf.data(), 1, Q);
    for (int b = 0; b < batch; b++) { std::vector<cd> L((size_t)n * n); magma_zgetmatrix(n, n, hA[b], ld, zp(L), n, Q); auto R = H0[b]; magma_int_t nn = n, il; lapackf77_zpotrf("L", &nn, zp(R), &nn, &il);
      bool fin = finiteM(H0[b]); double rat = 0; if (fin && inf[b] == 0) { std::vector<cd> Lm((size_t)n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) Lm[i + (size_t)j * n] = L[i + (size_t)j * n];
        rat = ratio(diff1(H0[b], mul(Lm, n, n, adj(Lm, n, n), n), n, n), nrm1(H0[b], n, n), n); }
      out("zpotrf_batched", hk[b] + "#" + std::to_string(b), n, n, inf[b], il, fin ? rat : NAN, fin, finiteM(L), finiteM(R)); }
    // geqrf_batched
    Z* dtau; magma_zmalloc(&dtau, (size_t)n * batch); Z** dTa; magma_malloc((void**)&dTa, batch * sizeof(Z*)); std::vector<Z*> hT(batch); for (int b = 0; b < batch; b++) hT[b] = dtau + (size_t)b * n; cudaMemcpy(dTa, hT.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice);
    load(A0); magma_zgeqrf_batched(n, n, dAa, ld, dTa, dinfo, batch, Q); magma_queue_sync(Q); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, inf.data(), 1, Q);
    for (int b = 0; b < batch; b++) { bool fin = finiteM(A0[b]); std::vector<cd> QR((size_t)n * n), tau(n); magma_zgetmatrix(n, n, hA[b], ld, zp(QR), n, Q); magma_zgetvector(n, hT[b], 1, zp(tau), 1, Q); double rat = 0;
      if (fin) { std::vector<cd> Qm = QR; magma_int_t nn = n, lw = 64 * n, il; std::vector<cd> w(lw); lapackf77_zungqr(&nn, &nn, &nn, zp(Qm), &nn, zp(tau), zp(w), &lw, &il);
        std::vector<cd> Rm((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) Rm[i + (size_t)j * n] = QR[i + (size_t)j * n]; rat = ratio(diff1(A0[b], mul(Qm, n, n, Rm, n), n, n), nrm1(A0[b], n, n), n); }
      out("zgeqrf_batched", kinds[b] + "#" + std::to_string(b), n, n, inf[b], 0, fin ? rat : NAN, fin, finiteM(QR) && finiteM(tau), true); }
    magma_free(dtau); magma_free(dTa); magma_free(dA); magma_free(dInv); magma_free(dpiv); magma_free(dinfo); magma_free(dAa); magma_free(dIa); magma_free(dPa);
  }
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string g = argc > 1 ? argv[1] : ""; G = argc > 1 ? argv[1] : "";
  magma_init(); magma_queue_create(0, &Q);
  printf("group,routine,kind,m,n,info_magma,info_lapack,ratio,verdict,note\n");
  if (g == "lu") groupLU(); else if (g == "chol") groupChol(); else if (g == "qr") groupQR(); else if (g == "solve") groupSolve();
  else if (g == "inv") groupInv(); else if (g == "heig") groupHeig(); else if (g == "geev") groupGeev(); else if (g == "svd") groupSVD();
  else if (g == "reduce") groupReduce(); else if (g == "blas") groupBlas(); else if (g == "batched") groupBatched(); else printf("unknown group\n");
  fprintf(stderr, "SUMMARY %s OK=%d INFO=%d WRONG=%d WARN=%d NFDIFF=%d cuda=%s\n", G, cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cudaGetErrorString(cudaGetLastError()));
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

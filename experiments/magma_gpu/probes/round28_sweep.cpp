// Twenty-eighth round (roxy, RTX 5070 Ti sm_120): differential sweep on STRUCTURED, ill-conditioned inputs, against
// reference LAPACK (OpenBLAS 0.3.34).  Round 27 showed that random well-conditioned matrices hardly test QR with
// pivoting, SVD or eigensolvers (zgeqp3 failed only on rank-deficient input), so every input here has a prescribed
// spectrum: A = U diag(sigma) V^H with Haar unitary U, V (or Q diag(lambda) Q^H for Hermitian input).
//     round28_sweep <group>     group = qrcp svd heig lu chol batched
// CSV line per case: group,routine,kind,m,n,info_magma,info_lapack,ratio,verdict,note  (same verdicts as round 27,
// except that outputs after info > 0 are not judged, and the extra metric in `note` is explained per group)
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <complex>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
typedef std::complex<double> cd;
typedef magmaDoubleComplex Z;
static Z* zp(std::vector<cd>& v) { return (Z*)v.data(); }
static magma_queue_t Q;
static const double EPS = 1.1102230246251565e-16;
static const char* G = "";
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static cd rc() { double a = r1(); return cd(a, r1()); }
static bool finiteM(const std::vector<double>& A) { for (double a : A) if (!std::isfinite(a)) return false; return true; }
static bool finiteM(const std::vector<cd>& A) { for (auto& a : A) if (!std::isfinite(a.real()) || !std::isfinite(a.imag())) return false; return true; }
static double nrm1(const std::vector<cd>& A, int m, int n) { double mx = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < m; i++) s += std::abs(A[i + (size_t)j * m]); mx = std::max(mx, s); } return mx; }
static std::vector<cd> mul(const std::vector<cd>& A, int m, int k, const std::vector<cd>& B, int n) { std::vector<cd> C((size_t)m * n);
  for (int j = 0; j < n; j++) for (int l = 0; l < k; l++) { cd b = B[l + (size_t)j * k]; if (b == cd(0)) continue; for (int i = 0; i < m; i++) C[i + (size_t)j * m] += A[i + (size_t)l * m] * b; } return C; }
static std::vector<cd> adj(const std::vector<cd>& A, int m, int n) { std::vector<cd> T((size_t)n * m); for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) T[j + (size_t)i * n] = std::conj(A[i + (size_t)j * m]); return T; }
static double diff1(const std::vector<cd>& A, const std::vector<cd>& B, int m, int n) { std::vector<cd> D(A.size()); for (size_t i = 0; i < A.size(); i++) D[i] = A[i] - B[i]; return nrm1(D, m, n); }
static double ratio(double err, double nrm, int n) { if (n == 0) return 0; double d = nrm * std::max(1, n) * EPS; if (d == 0) return err == 0 ? 0 : INFINITY; return err / d; }
static int cnt[5];
static void out(const char* rt, const std::string& kind, int m, int n, long long im, long long il, double rat, bool magmaFinite, const std::string& note = "") {
  const char* v = "OK";
  if (im != il) v = "INFO"; else if (im == 0 && !magmaFinite) v = "WRONG"; else if (im == 0 && (std::isnan(rat) || rat > 1000)) v = "WRONG"; else if (im == 0 && rat > 30) v = "WARN";
  static const char* names[] = {"OK", "INFO", "WRONG", "WARN"}; for (int i = 0; i < 4; i++) if (!strcmp(v, names[i])) cnt[i]++;
  printf("%s,%s,%s,%d,%d,%lld,%lld,%.3g,%s,%s\n", G, rt, kind.c_str(), m, n, im, il, rat, v, note.c_str()); }
static int ldd(int m) { return std::max(1, (int)magma_roundup(m, 32)); }
static Z* up(std::vector<cd>& A, int m, int n, int ld) { Z* d = NULL; magma_zmalloc(&d, (size_t)std::max(1, ld) * std::max(1, n)); if (m > 0 && n > 0) magma_zsetmatrix(m, n, zp(A), std::max(1, m), d, ld, Q); return d; }
static void down(Z* d, int ld, std::vector<cd>& A, int m, int n) { if (m > 0 && n > 0) magma_zgetmatrix(m, n, d, ld, zp(A), std::max(1, m), Q); }

// Haar-like unitary m x k (k <= m): Q factor of a random complex Gaussian-ish matrix
static std::vector<cd> haar(int m, int k) { std::vector<cd> X((size_t)m * k); for (auto& x : X) x = rc(); if (m == 0 || k == 0) return X;
  magma_int_t mm = m, kk = k, lw = 64 * std::max(m, 1), il; std::vector<cd> tau(k), w(lw);
  lapackf77_zgeqrf(&mm, &kk, zp(X), &mm, zp(tau), zp(w), &lw, &il); lapackf77_zungqr(&mm, &kk, &kk, zp(X), &mm, zp(tau), zp(w), &lw, &il); return X; }
// singular value profiles (descending), length k
static std::vector<double> profile(const std::string& k, int n) { std::vector<double> s(n);
  for (int i = 0; i < n; i++) {
    if (k == "lowrank_half") s[i] = i < n / 2 ? 1.0 : 0.0; else if (k == "rank1") s[i] = i == 0 ? 1.0 : 0.0;
    else if (k == "cond1e12") s[i] = n > 1 ? pow(1e-12, (double)i / (n - 1)) : 1.0; else if (k == "cond1e8_one_small") s[i] = i == n - 1 ? 1e-8 : 1.0;
    else if (k == "cluster") s[i] = 1.0 + 1e-13 * i; else if (k == "gap_at_half") s[i] = i < n / 2 ? 1.0 : 1e-10; else s[i] = 1.0; }
  return s; }
static std::vector<cd> gen(const std::string& k, int m, int n, unsigned seed, std::vector<double>* sig = NULL) {
  Sd = seed; int kk = std::min(m, n); std::vector<cd> A((size_t)m * n); if (kk == 0) return A;
  if (k == "kahan") { double c = 0.3, sn = sqrt(1 - c * c); for (int j = 0; j < n; j++) for (int i = 0; i <= std::min(j, m - 1); i++) A[i + (size_t)j * m] = pow(sn, i) * (i == j ? 1.0 : -c); return A; }
  if (k == "graded") { for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) A[i + (size_t)j * m] = rc() * pow(10.0, -200.0 * i / std::max(1, m - 1) + 100.0); return A; }
  auto s = profile(k, kk); if (sig) *sig = s; auto U = haar(m, kk), V = haar(n, kk);
  for (int l = 0; l < kk; l++) for (int j = 0; j < n; j++) { cd v = s[l] * std::conj(V[j + (size_t)l * n]); if (v == cd(0)) continue; for (int i = 0; i < m; i++) A[i + (size_t)j * m] += U[i + (size_t)l * m] * v; }
  return A; }
static std::vector<cd> genH(const std::string& k, int n, unsigned seed, bool pd) {
  Sd = seed; std::vector<cd> A((size_t)n * n); if (n == 0) return A; auto Qm = haar(n, n); std::vector<double> lam(n);
  for (int i = 0; i < n; i++) {
    if (k == "cluster") lam[i] = 1.0 + 1e-12 * i; else if (k == "cond1e12") lam[i] = n > 1 ? pow(1e-12, (double)i / (n - 1)) : 1.0;
    else if (k == "lowrank_half") lam[i] = i < n / 2 ? 1.0 : 0.0; else if (k == "twoclusters") lam[i] = i < n / 2 ? -1.0 + 1e-14 * i : 1.0 + 1e-14 * i;
    else if (k == "near_singular_pd") lam[i] = i == n - 1 ? 1e-15 : 1.0; else lam[i] = 1.0 + i;
    if (!pd && k == "cond1e12" && (i % 2)) lam[i] = -lam[i]; }
  for (int l = 0; l < n; l++) for (int j = 0; j < n; j++) { cd v = lam[l] * std::conj(Qm[j + (size_t)l * n]); for (int i = 0; i < n; i++) A[i + (size_t)j * n] += Qm[i + (size_t)l * n] * v; }
  for (int j = 0; j < n; j++) { A[j + (size_t)j * n] = A[j + (size_t)j * n].real(); for (int i = j + 1; i < n; i++) A[j + (size_t)i * n] = std::conj(A[i + (size_t)j * n]); }
  return A; }
static const std::vector<int> NS = {1, 2, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 200, 255, 256, 257};
static const std::vector<std::string> GK = {"lowrank_half", "rank1", "cond1e12", "cond1e8_one_small", "gap_at_half", "kahan", "graded"};
static const std::vector<std::string> HK = {"cluster", "cond1e12", "lowrank_half", "twoclusters", "spread"};

// ---- QR with column pivoting (CPU and GPU interfaces) and plain QR: A P = Q R, rank gap vs LAPACK
static void groupQRCP() {
  for (int n : NS) for (auto& k : GK) for (int shape = 0; shape < 2; shape++) {
    int m = shape ? n + 17 : n; auto A0 = gen(k, m, n, 100u + 7 * m + n); int kk = std::min(m, n); magma_int_t mm = m, nn = n, lda = m, il;
    // LAPACK reference for the rank gap: |R(r,r)| where r = numerical rank boundary of the profile
    auto Ar = A0; std::vector<magma_int_t> jr(n, 0); std::vector<cd> tr(kk); magma_int_t lw = (n + 1) * 64 + 4096; std::vector<cd> w(lw); std::vector<double> rw(2 * n);
    lapackf77_zgeqp3(&mm, &nn, zp(Ar), &lda, jr.data(), zp(tr), zp(w), &lw, rw.data(), &il);
    for (int v = 0; v < 3; v++) {
      const char* nm[] = {"zgeqp3", "zgeqp3_gpu", "zgeqrf"};
      auto A = A0; std::vector<magma_int_t> jp(n, 0); std::vector<cd> tau(kk); magma_int_t im = -999;
      if (v == 0) magma_zgeqp3(m, n, zp(A), lda, jp.data(), zp(tau), zp(w), lw, rw.data(), &im);
      else if (v == 1) { int ld = ldd(m); Z* d = up(A, m, n, ld); magma_int_t lwg = (n + 1) * 64 + 2 * n * 64 + 4096; Z* dw; magma_zmalloc(&dw, lwg); std::vector<double> rwg(2 * n);
        // tau must be a DEVICE array (MAGMA's own testing_zgeqp3_gpu.cpp uses dtau), although magma_z.h declares
        // `magmaDoubleComplex *tau` and the documentation does not say so; a host tau gives an illegal memory access
        Z* dtau; magma_zmalloc(&dtau, std::max(1, kk));
        magma_zgeqp3_gpu(m, n, d, ld, jp.data(), (Z*)dtau, dw, lwg, rwg.data(), &im); if (kk > 0) magma_zgetvector(kk, dtau, 1, zp(tau), 1, Q);
        down(d, ld, A, m, n); magma_free(d); magma_free(dw); magma_free(dtau); }
      else { magma_zgeqrf(m, n, zp(A), lda, zp(tau), zp(w), lw, &im); for (int j = 0; j < n; j++) jp[j] = j + 1; }
      std::string note; double rat = 0;
      std::vector<int> seen(n, 0); bool perm = true; for (int j = 0; j < n; j++) { int p = (int)jp[j] - 1; if (p < 0 || p >= n || seen[p]++) perm = false; }
      if (!perm) { note = "jpvt not a permutation"; rat = INFINITY; }
      else if (kk > 0) { std::vector<cd> Qm((size_t)m * kk); for (int j = 0; j < kk; j++) for (int i = 0; i < m; i++) Qm[i + (size_t)j * m] = A[i + (size_t)j * m];
        magma_int_t mk = kk, lwq = 64 * m + 64; std::vector<cd> wq(lwq); lapackf77_zungqr(&mm, &mk, &mk, zp(Qm), &lda, zp(tau), zp(wq), &lwq, &il);
        std::vector<cd> Rm((size_t)kk * n); for (int j = 0; j < n; j++) for (int i = 0; i <= std::min(j, kk - 1); i++) Rm[i + (size_t)j * kk] = A[i + (size_t)j * m];
        std::vector<cd> AP((size_t)m * n); for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) AP[i + (size_t)j * m] = A0[i + (size_t)(jp[j] - 1) * m];
        rat = ratio(diff1(AP, mul(Qm, m, kk, Rm, n), m, n), nrm1(A0, m, n), std::max(m, n));
        if (v < 2) { // rank-revealing quality: max over i of |R_magma(i,i)| / |R_lapack(i,i)| beyond the rank boundary, scaled
          int r = (k == "lowrank_half") ? kk / 2 : (k == "rank1") ? 1 : (k == "gap_at_half") ? kk / 2 : -1;
          if (r >= 0 && r < kk) { double rm = std::abs(A[r + (size_t)r * m]), rl = std::abs(Ar[r + (size_t)r * m]), r0 = std::abs(A[0]);
            char b[160]; snprintf(b, sizeof b, "|R(r,r)|/|R(0,0)| magma=%.2e lapack=%.2e", rm / r0, rl / std::abs(Ar[0])); note = b;
            double tol = (k == "gap_at_half") ? 1e-6 : 1e-10; if (rm / r0 > tol && rl / std::abs(Ar[0]) <= tol) { rat = std::max(rat, 1e6); note += " RANK-NOT-REVEALED"; } } } }
      out(nm[v], k + (shape ? "/tall" : ""), m, n, im, 0, rat, finiteM(A), note);
    }
  }
}

// ---- SVD: singular values vs the prescribed profile and LAPACK; reconstruction and orthogonality
static void groupSVD() {
  for (int n : NS) for (auto& k : GK) for (int shape = 0; shape < 2; shape++) {
    if (k == "graded" && n > 129) continue;
    int m = shape ? n + 17 : n; int kk = std::min(m, n); auto A0 = gen(k, m, n, 200u + 3 * m + n); magma_int_t mm = m, nn = n, lda = m, il;
    std::vector<double> sref(kk); { auto R = A0; std::vector<cd> U(1), VT(1); magma_int_t one = 1, lw = 8 * std::max(m, n) + 64; std::vector<cd> w(lw); std::vector<double> rw(5 * kk);
      lapackf77_zgesvd("N", "N", &mm, &nn, zp(R), &lda, sref.data(), zp(U), &one, zp(VT), &one, zp(w), &lw, rw.data(), &il); }
    double smax = sref[0];
    for (int v = 0; v < 2; v++) {
      auto A = A0; std::vector<double> s(kk); std::vector<cd> U((size_t)m * kk), VT((size_t)kk * n); magma_int_t lw = 3 * kk * kk + 8 * std::max(m, n) * 64 + 8 * kk * kk + 65536, im = -999;
      std::vector<cd> wk(lw); std::vector<double> rw(5 * kk * kk + 7 * kk + 64); std::vector<magma_int_t> iw(8 * kk);
      if (v == 0) magma_zgesvd(MagmaSomeVec, MagmaSomeVec, m, n, zp(A), lda, s.data(), zp(U), m, zp(VT), kk, zp(wk), lw, rw.data(), &im);
      else magma_zgesdd(MagmaSomeVec, m, n, zp(A), lda, s.data(), zp(U), m, zp(VT), kk, zp(wk), lw, rw.data(), iw.data(), &im);
      double e = 0; for (int i = 0; i < kk; i++) e = std::max(e, std::fabs(s[i] - sref[i])); double rat = ratio(e, smax, std::max(m, n));
      std::vector<cd> US((size_t)m * kk); for (int j = 0; j < kk; j++) for (int i = 0; i < m; i++) US[i + (size_t)j * m] = U[i + (size_t)j * m] * s[j];
      rat = std::max(rat, ratio(diff1(A0, mul(US, m, kk, VT, n), m, n), nrm1(A0, m, n), std::max(m, n)));
      auto UU = mul(adj(U, m, kk), kk, m, U, kk); for (int i = 0; i < kk; i++) UU[i + (size_t)i * kk] -= 1.0; double ou = ratio(nrm1(UU, kk, kk), 1.0, std::max(m, n));
      auto VV = mul(VT, kk, n, adj(VT, kk, n), kk); for (int i = 0; i < kk; i++) VV[i + (size_t)i * kk] -= 1.0; double ov = ratio(nrm1(VV, kk, kk), 1.0, std::max(m, n));
      char b[96]; snprintf(b, sizeof b, "orthU=%.2g orthV=%.2g", ou, ov); rat = std::max(rat, std::max(ou, ov));
      out(v == 0 ? "zgesvd(S)" : "zgesdd(S)", k + (shape ? "/tall" : ""), m, n, im, il, rat, finiteM(s) && finiteM(U) && finiteM(VT), b);
    }
  }
}

// ---- Hermitian eigensolvers on clustered / ill-conditioned spectra: eigenvalues, residual, orthogonality
static void groupHeig() {
  for (int n : NS) for (auto& k : HK) {
    auto A0 = genH(k, n, 300u + n, false); magma_int_t nn = n, lda = n, il; std::vector<double> wref(n);
    { auto R = A0; magma_int_t lw = 4 * n; std::vector<cd> w(lw); std::vector<double> rw(3 * n); lapackf77_zheev("N", "L", &nn, zp(R), &lda, wref.data(), zp(w), &lw, rw.data(), &il); }
    double anrm = nrm1(A0, n, n); magma_int_t th = 16; (void)th;
    magma_int_t lw = 2 * n + n * n + n * 512 + 200000, lrw = 1 + 5 * n + 2 * n * n + 4096, liw = 3 + 5 * n + 64;
    auto eval = [&](const char* nm, magma_int_t im, std::vector<double>& w, int mfound, std::vector<cd>& Zm, int ldz) {
      double rat = 0; std::string note; if (im == 0) { double e = 0; for (int i = 0; i < std::min(mfound, n); i++) e = std::max(e, std::fabs(w[i] - wref[i])); rat = ratio(e, anrm, n); if (mfound != n) { rat = INFINITY; note = "mout=" + std::to_string(mfound); }
        if (rat <= 1000) { std::vector<cd> Zc((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) Zc[i + (size_t)j * n] = Zm[i + (size_t)j * ldz];
          auto AZ = mul(A0, n, n, Zc, n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) AZ[i + (size_t)j * n] -= Zc[i + (size_t)j * n] * w[j];
          double res = ratio(nrm1(AZ, n, n), anrm, n); auto ZZ = mul(adj(Zc, n, n), n, n, Zc, n); for (int i = 0; i < n; i++) ZZ[i + (size_t)i * n] -= 1.0; double orth = ratio(nrm1(ZZ, n, n), 1.0, n);
          char b[96]; snprintf(b, sizeof b, "res=%.2g orth=%.2g", res, orth); note = b; rat = std::max(rat, std::max(res, orth)); } }
      out(nm, k, n, n, im, il, rat, true, note); };
    { auto A = A0; std::vector<double> w(n); std::vector<cd> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); magma_int_t im = -999;
      magma_zheevd(MagmaVec, MagmaLower, n, zp(A), lda, w.data(), zp(wk), lw, rw.data(), lrw, iw.data(), liw, &im); eval("zheevd(V)", im, w, n, A, lda); }
    { auto A = A0; std::vector<double> w(n); std::vector<cd> wk(lw), wA((size_t)n * n); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); magma_int_t im = -999; int ld = ldd(n); Z* d = up(A, n, n, ld);
      magma_zheevd_gpu(MagmaVec, MagmaLower, n, d, ld, w.data(), zp(wA), lda, zp(wk), lw, rw.data(), lrw, iw.data(), liw, &im); down(d, ld, A, n, n); magma_free(d); eval("zheevd_gpu(V)", im, w, n, A, lda); }
    { auto A = A0; std::vector<double> w(n); std::vector<cd> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); magma_int_t im = -999, mo = 0;
      magma_zheevdx(MagmaVec, MagmaRangeAll, MagmaLower, n, zp(A), lda, 0, 0, 1, n, &mo, w.data(), zp(wk), lw, rw.data(), lrw, iw.data(), liw, &im); eval("zheevdx(V)", im, w, n == 1 ? 1 : (int)mo, A, lda); }
    { auto A = A0; std::vector<double> w(n); std::vector<cd> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); magma_int_t im = -999, mo = 0;
      magma_zheevdx_2stage(MagmaVec, MagmaRangeAll, MagmaLower, n, zp(A), lda, 0, 0, 1, n, &mo, w.data(), zp(wk), lw, rw.data(), lrw, iw.data(), liw, &im); eval("zheevdx_2stage(V)", im, w, n == 1 ? 1 : (int)mo, A, lda); }
    { auto A = A0; std::vector<double> w(n); std::vector<cd> wk(lw), Zm((size_t)n * n); std::vector<double> rw(lrw + 24 * n); std::vector<magma_int_t> iw(liw + 10 * n), isup(2 * n); magma_int_t im = -999, mo = 0;
      magma_zheevr(MagmaVec, MagmaRangeAll, MagmaLower, n, zp(A), lda, 0, 0, 1, n, 0, &mo, w.data(), zp(Zm), lda, isup.data(), zp(wk), lw, rw.data(), lrw + 24 * n, iw.data(), liw + 10 * n, &im); eval("zheevr(V)", im, w, (int)mo, Zm, lda); }
    { auto A = A0; std::vector<double> w(n); std::vector<cd> wk(lw), Zm((size_t)n * n); std::vector<double> rw(7 * n); std::vector<magma_int_t> iw(5 * n), ifl(n); magma_int_t im = -999, mo = 0;
      magma_zheevx(MagmaVec, MagmaRangeAll, MagmaLower, n, zp(A), lda, 0, 0, 1, n, 0, &mo, w.data(), zp(Zm), lda, zp(wk), lw, rw.data(), iw.data(), ifl.data(), &im); eval("zheevx(V)", im, w, (int)mo, Zm, lda); }
  }
}

// ---- LU / Cholesky on graded and ill-conditioned inputs: backward error of the solve
static void groupLU() {
  for (int n : NS) for (auto& k : GK) {
    auto A0 = gen(k, n, n, 400u + n); const int nrhs = 2; Sd = 41u + n; std::vector<cd> B0((size_t)n * nrhs); for (auto& b : B0) b = rc();
    magma_int_t nn = n, nr = nrhs, lda = n, il; auto R = A0, X = B0; std::vector<magma_int_t> ip(n); lapackf77_zgesv(&nn, &nr, zp(R), &lda, ip.data(), zp(X), &lda, &il);
    for (int v = 0; v < 3; v++) { const char* nm[] = {"zgesv", "zgesv_gpu", "zgetrf_native+getrs"};
      auto A = A0, Xm = B0; std::vector<magma_int_t> p(n); magma_int_t im = -999; int ld = ldd(n);
      if (v == 0) magma_zgesv(n, nrhs, zp(A), lda, p.data(), zp(Xm), lda, &im);
      else { Z* dA = up(A, n, n, ld); Z* dB = up(Xm, n, nrhs, ld);
        if (v == 1) magma_zgesv_gpu(n, nrhs, dA, ld, p.data(), dB, ld, &im); else { magma_zgetrf_native(n, n, dA, ld, p.data(), &im); if (im == 0) magma_zgetrs_gpu(MagmaNoTrans, n, nrhs, dA, ld, p.data(), dB, ld, &im); }
        down(dB, ld, Xm, n, nrhs); magma_free(dA); magma_free(dB); }
      double rat = 0; if (im == 0) { auto AX = mul(A0, n, n, Xm, nrhs); rat = ratio(diff1(AX, B0, n, nrhs), nrm1(A0, n, n) * nrm1(Xm, n, nrhs), n); }
      out(nm[v], k, n, n, im, il, rat, finiteM(Xm)); }
  }
}
static void groupChol() {
  for (int n : NS) for (std::string k : {"cond1e12", "near_singular_pd", "cluster", "spread"}) {
    auto A0 = genH(k, n, 500u + n, true); const int nrhs = 2; Sd = 51u + n; std::vector<cd> B0((size_t)n * nrhs); for (auto& b : B0) b = rc();
    magma_int_t nn = n, nr = nrhs, lda = n, il; auto R = A0, X = B0; lapackf77_zposv("L", &nn, &nr, zp(R), &lda, zp(X), &lda, &il);
    for (int v = 0; v < 3; v++) { const char* nm[] = {"zposv", "zposv_gpu", "zpotrf_native+potrs"};
      auto A = A0, Xm = B0; magma_int_t im = -999; int ld = ldd(n);
      if (v == 0) magma_zposv(MagmaLower, n, nrhs, zp(A), lda, zp(Xm), lda, &im);
      else { Z* dA = up(A, n, n, ld); Z* dB = up(Xm, n, nrhs, ld);
        if (v == 1) magma_zposv_gpu(MagmaLower, n, nrhs, dA, ld, dB, ld, &im); else { magma_zpotrf_native(MagmaLower, n, dA, ld, &im); if (im == 0) magma_zpotrs_gpu(MagmaLower, n, nrhs, dA, ld, dB, ld, &im); }
        down(dB, ld, Xm, n, nrhs); magma_free(dA); magma_free(dB); }
      double rat = 0; if (im == 0) { auto AX = mul(A0, n, n, Xm, nrhs); rat = ratio(diff1(AX, B0, n, nrhs), nrm1(A0, n, n) * nrm1(Xm, n, nrhs), n); }
      out(nm[v], k, n, n, im, il, rat, finiteM(Xm)); }
  }
}

// ---- batched solvers on structured inputs (small n, 64 matrices of mixed kinds)
static void groupBatched() {
  const int batch = 64;
  for (int n : {1, 2, 3, 4, 7, 8, 15, 16, 17, 24, 31, 32, 33, 48, 64}) {
    std::vector<std::vector<cd>> A0(batch), H0(batch), B0(batch); std::vector<std::string> kinds(batch), hk(batch); const int nrhs = 2;
    for (int b = 0; b < batch; b++) { kinds[b] = GK[b % GK.size()]; A0[b] = gen(kinds[b], n, n, 600u + 31u * b + n); hk[b] = std::vector<std::string>{"cond1e12", "near_singular_pd", "cluster", "spread"}[b % 4];
      H0[b] = genH(hk[b], n, 700u + 17u * b + n, true); Sd = 800u + b + n; B0[b].resize((size_t)n * nrhs); for (auto& x : B0[b]) x = rc(); }
    int ld = ldd(n); size_t per = (size_t)ld * n, perb = (size_t)ld * nrhs; Z *dA, *dB; magma_int_t *dpiv, *dinfo; magma_zmalloc(&dA, per * batch); magma_zmalloc(&dB, perb * batch); magma_imalloc(&dpiv, (size_t)n * batch); magma_imalloc(&dinfo, batch);
    Z **dAa, **dBa; magma_int_t** dPa; magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dBa, batch * sizeof(Z*)); magma_malloc((void**)&dPa, batch * sizeof(void*));
    std::vector<Z*> hA(batch), hB(batch); std::vector<magma_int_t*> hP(batch); for (int b = 0; b < batch; b++) { hA[b] = dA + b * per; hB[b] = dB + b * perb; hP[b] = dpiv + (size_t)b * n; }
    cudaMemcpy(dAa, hA.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dBa, hB.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dPa, hP.data(), batch * sizeof(void*), cudaMemcpyHostToDevice);
    for (int v = 0; v < 3; v++) { // gesv_batched, gesv_nopiv-free getrf_batched+getrs_batched, posv_batched
      auto& M = v == 2 ? H0 : A0; for (int b = 0; b < batch; b++) { magma_zsetmatrix(n, n, zp(M[b]), n, hA[b], ld, Q); magma_zsetmatrix(n, nrhs, zp(B0[b]), n, hB[b], ld, Q); }
      magma_int_t ret = 0;
      if (v == 0) ret = magma_zgesv_batched(n, nrhs, dAa, ld, dPa, dBa, ld, dinfo, batch, Q);
      else if (v == 1) { ret = magma_zgetrf_batched(n, n, dAa, ld, dPa, dinfo, batch, Q); magma_zgetrs_batched(MagmaNoTrans, n, nrhs, dAa, ld, dPa, dBa, ld, batch, Q); }
      else ret = magma_zposv_batched(MagmaLower, n, nrhs, dAa, ld, dBa, ld, dinfo, batch, Q);
      magma_queue_sync(Q); std::vector<magma_int_t> inf(batch); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, inf.data(), 1, Q);
      const char* nm[] = {"zgesv_batched", "zgetrf_batched+getrs_batched", "zposv_batched"};
      for (int b = 0; b < batch; b++) { std::vector<cd> Xm((size_t)n * nrhs); magma_zgetmatrix(n, nrhs, hB[b], ld, zp(Xm), n, Q);
        auto R = M[b], X = B0[b]; magma_int_t nn = n, nr = nrhs, il; std::vector<magma_int_t> ip(n);
        if (v == 2) lapackf77_zposv("L", &nn, &nr, zp(R), &nn, zp(X), &nn, &il); else lapackf77_zgesv(&nn, &nr, zp(R), &nn, ip.data(), zp(X), &nn, &il);
        double rat = 0; if (inf[b] == 0) { auto AX = mul(M[b], n, n, Xm, nrhs); rat = ratio(diff1(AX, B0[b], n, nrhs), nrm1(M[b], n, n) * nrm1(Xm, n, nrhs), n); }
        out(nm[v], (v == 2 ? hk[b] : kinds[b]) + "#" + std::to_string(b), n, n, inf[b], il, rat, finiteM(Xm), ret ? "ret=" + std::to_string(ret) : ""); }
    }
    magma_free(dA); magma_free(dB); magma_free(dpiv); magma_free(dinfo); magma_free(dAa); magma_free(dBa); magma_free(dPa);
  }
}

// zheevr: MAGMA vs LAPACK zheevr orthogonality on the cond1e12 spectrum (same matrices as groupHeig), and zheevdx_2stage
static void heevrcmp() {
  for (int n : {31, 63, 127, 200, 255}) for (std::string k : {"cond1e12", "lowrank_half"}) {
    auto A0 = genH(k, n, 300u + n, false); magma_int_t nn = n, lda = n, lw = 2 * n + n * n + n * 512 + 200000, lrw = 24 * n + 2 * n * n + 4096, liw = 10 * n + 64;
    auto orth = [&](std::vector<cd>& Zm) { auto ZZ = mul(adj(Zm, n, n), n, n, Zm, n); for (int i = 0; i < n; i++) ZZ[i + (size_t)i * n] -= 1.0; return ratio(nrm1(ZZ, n, n), 1.0, n); };
    std::vector<double> w(n); std::vector<cd> wk(lw), Zl((size_t)n * n), Zm((size_t)n * n); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw), isup(2 * n); magma_int_t mo, il, im; double vl = 0, vu = 0, ab = 0; magma_int_t one = 1;
    auto A = A0; lapackf77_zheevr("V", "A", "L", &nn, zp(A), &lda, &vl, &vu, &one, &nn, &ab, &mo, w.data(), zp(Zl), &lda, isup.data(), zp(wk), &lw, rw.data(), &lrw, iw.data(), &liw, &il);
    A = A0; magma_zheevr(MagmaVec, MagmaRangeAll, MagmaLower, n, zp(A), lda, 0, 0, 1, n, 0, &mo, w.data(), zp(Zm), lda, isup.data(), zp(wk), lw, rw.data(), lrw, iw.data(), liw, &im);
    printf("heevr %-12s n=%3d  orth LAPACK zheevr=%.3g  MAGMA zheevr=%.3g\n", k.c_str(), n, orth(Zl), orth(Zm));
  }
}

// one call per process: qp3small <routine 0|1|2> <m> <n> <zero|one|rand>; reports the CUDA status after the call
static void qp3small(int v, int m, int n, const std::string& k) {
  std::vector<cd> A((size_t)m * n); Sd = 9u; for (auto& a : A) a = k == "zero" ? cd(0) : k == "one" ? cd(1) : rc();
  int kk = std::min(m, n); std::vector<magma_int_t> jp(n, 0); std::vector<cd> tau(std::max(1, kk)); magma_int_t im = -999, lw = (n + 1) * 64 + 4096; std::vector<cd> w(lw); std::vector<double> rw(2 * n + 2);
  if (v == 0) magma_zgeqp3(m, n, zp(A), m, jp.data(), zp(tau), zp(w), lw, rw.data(), &im);
  else if (v == 1) { int ld = ldd(m); Z* d = up(A, m, n, ld); magma_int_t lwg = (n + 1) * 64 + 2 * n * 64 + 4096; Z* dw; magma_zmalloc(&dw, lwg);
    magma_zgeqp3_gpu(m, n, d, ld, jp.data(), zp(tau), dw, lwg, rw.data(), &im); down(d, ld, A, m, n); magma_free(d); magma_free(dw); }
  else if (v == 3) { int ld = ldd(m); Z* d = up(A, m, n, ld); magma_int_t lwg = (n + 1) * 64 + 2 * n * 64 + 4096; Z *dw, *dtau; magma_zmalloc(&dw, lwg); magma_zmalloc(&dtau, std::max(1, kk));
    magma_zgeqp3_gpu(m, n, d, ld, jp.data(), dtau, dw, lwg, rw.data(), &im); down(d, ld, A, m, n); magma_free(d); magma_free(dw); magma_free(dtau); }
  else magma_zgeqrf(m, n, zp(A), m, zp(tau), zp(w), lw, &im);
  cudaError_t es = cudaDeviceSynchronize(), el = cudaGetLastError();
  const char* nm[] = {"zgeqp3", "zgeqp3_gpu", "zgeqrf", "zgeqp3_gpu(dtau)"};
  printf("%-16s m=%d n=%d %-4s info=%lld jpvt[0]=%lld  sync=%s last=%s\n", nm[v], m, n, k.c_str(), (long long)im, (long long)jp[0], cudaGetErrorString(es), cudaGetErrorString(el));
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string g = argc > 1 ? argv[1] : ""; G = argc > 1 ? argv[1] : "";
  magma_init(); magma_queue_create(0, &Q);
  printf("group,routine,kind,m,n,info_magma,info_lapack,ratio,verdict,note\n");
  if (g == "qrcp") groupQRCP(); else if (g == "svd") groupSVD(); else if (g == "heig") groupHeig(); else if (g == "lu") groupLU();
  else if (g == "chol") groupChol(); else if (g == "batched") groupBatched(); else if (g == "heevrcmp") heevrcmp(); else if (g == "qp3small" && argc > 5) qp3small(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argv[5]); else printf("unknown group\n");
  fprintf(stderr, "SUMMARY %s OK=%d INFO=%d WRONG=%d WARN=%d cuda=%s\n", G, cnt[0], cnt[1], cnt[2], cnt[3], cudaGetErrorString(cudaGetLastError()));
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

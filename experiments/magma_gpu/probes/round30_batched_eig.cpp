// Thirtieth round (roxy): batched Jacobi eigensolver and SVD on structured inputs, against LAPACK zheev / zgesvd.
//     round30_batched_eig heevj | gesvj
// heevj: magma_zheevj_batched_small_sm (jobz = V, lower), n = 1..32;  gesvj: magma_zgesvj_batched (jobu = jobv = S),
// square and tall shapes.  Each batch holds 56 matrices: 8 spectrum kinds x 7 seeds.
// CSV: routine,kind,m,n,batch_index,info,ratio_values,ratio_residual,ratio_orth,verdict
//   ratios are normalized by n*eps (values relative to the largest |lambda| / sigma; residual relative to ||A||)
//   verdict: OK (all <= 100), WARN (<= 1e4), WRONG (> 1e4 or non-finite on finite input, with info 0), INFO (info != 0)
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
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static cd rc() { double a = r1(); return cd(a, r1()); }
static bool fin(const std::vector<cd>& v) { for (auto& a : v) if (!std::isfinite(a.real()) || !std::isfinite(a.imag())) return false; return true; }
static bool fin(const std::vector<double>& v) { for (auto a : v) if (!std::isfinite(a)) return false; return true; }
static std::vector<cd> mul(const std::vector<cd>& A, int m, int k, const std::vector<cd>& B, int n) { std::vector<cd> C((size_t)m * n);
  for (int j = 0; j < n; j++) for (int l = 0; l < k; l++) { cd b = B[l + (size_t)j * k]; for (int i = 0; i < m; i++) C[i + (size_t)j * m] += A[i + (size_t)l * m] * b; } return C; }
static std::vector<cd> adj(const std::vector<cd>& A, int m, int n) { std::vector<cd> T((size_t)n * m); for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) T[j + (size_t)i * n] = std::conj(A[i + (size_t)j * m]); return T; }
static double maxabs(const std::vector<cd>& A) { double m = 0; for (auto& a : A) m = std::max(m, std::abs(a)); return m; }
static std::vector<cd> haar(int m, int k) { std::vector<cd> X((size_t)m * k); for (auto& x : X) x = rc(); magma_int_t mm = m, kk = k, lw = 64 * m + 64, il; std::vector<cd> tau(k), w(lw);
  lapackf77_zgeqrf(&mm, &kk, zp(X), &mm, zp(tau), zp(w), &lw, &il); lapackf77_zungqr(&mm, &kk, &kk, zp(X), &mm, zp(tau), zp(w), &lw, &il); return X; }
static const std::vector<std::string> KINDS = {"spread", "cluster1e-12", "cond1e12", "lowrank_half", "rank1", "zero", "ident", "nan"};
static std::vector<double> spec(const std::string& k, int n) { std::vector<double> s(n);
  for (int i = 0; i < n; i++) { s[i] = 1.0 + i;
    if (k == "cluster1e-12") s[i] = 1.0 + 1e-12 * i; if (k == "cond1e12") s[i] = n > 1 ? pow(1e-12, (double)i / (n - 1)) : 1.0;
    if (k == "lowrank_half") s[i] = i < n / 2 ? 1.0 : 0.0; if (k == "rank1") s[i] = i == 0 ? 1.0 : 0.0; if (k == "zero") s[i] = 0; if (k == "ident") s[i] = 1.0; }
  return s; }
static std::vector<cd> herm(const std::string& k, int n, unsigned seed) { Sd = seed; auto Qm = haar(n, n); auto lam = spec(k, n); if (k == "spread" || k == "cond1e12") for (int i = 1; i < n; i += 2) lam[i] = -lam[i];
  std::vector<cd> A((size_t)n * n); for (int l = 0; l < n; l++) for (int j = 0; j < n; j++) { cd v = lam[l] * std::conj(Qm[j + (size_t)l * n]); for (int i = 0; i < n; i++) A[i + (size_t)j * n] += Qm[i + (size_t)l * n] * v; }
  for (int j = 0; j < n; j++) { A[j + (size_t)j * n] = A[j + (size_t)j * n].real(); for (int i = j + 1; i < n; i++) A[j + (size_t)i * n] = std::conj(A[i + (size_t)j * n]); }
  if (k == "nan") A[(n / 2) * (size_t)(n + 1)] = NAN; return A; }
static std::vector<cd> gen(const std::string& k, int m, int n, unsigned seed) { Sd = seed; int kk = std::min(m, n); auto U = haar(m, kk), V = haar(n, kk); auto s = spec(k, kk);
  std::vector<cd> A((size_t)m * n); for (int l = 0; l < kk; l++) for (int j = 0; j < n; j++) { cd v = s[l] * std::conj(V[j + (size_t)l * n]); for (int i = 0; i < m; i++) A[i + (size_t)j * m] += U[i + (size_t)l * m] * v; }
  if (k == "nan") A[m / 2 + (size_t)(n / 2) * m] = NAN; return A; }
static void out(const char* rt, const std::string& k, int m, int n, int b, long long info, double rv, double rr, double ro, bool inFin, bool outFin) {
  const char* v = "OK"; double mx = std::max(rv, std::max(rr, ro));
  if (info != 0) v = "INFO"; else if (!inFin) v = outFin ? "NAN-HIDDEN" : "OK"; else if (!outFin || !(mx <= 1e4)) v = "WRONG"; else if (mx > 100) v = "WARN";
  printf("%s,%s,%d,%d,%d,%lld,%.3g,%.3g,%.3g,%s\n", rt, k.c_str(), m, n, b, info, rv, rr, ro, v); }

static void heevj() {
  for (int n : {1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 24, 31, 32}) {
    const int batch = (int)KINDS.size() * 7; std::vector<std::vector<cd>> A0(batch); for (int b = 0; b < batch; b++) A0[b] = herm(KINDS[b % KINDS.size()], n, 1000u + 13u * b + n);
    Z* dA; double* dW; magma_int_t* dinfo; magma_zmalloc(&dA, (size_t)n * n * batch); magma_dmalloc(&dW, (size_t)n * batch); magma_imalloc(&dinfo, batch);
    std::vector<Z*> hA(batch); std::vector<double*> hW(batch); for (int b = 0; b < batch; b++) { hA[b] = dA + (size_t)b * n * n; hW[b] = dW + (size_t)b * n; magma_zsetmatrix(n, n, zp(A0[b]), n, hA[b], n, Q); }
    Z** dAa; double** dWa; magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dWa, batch * sizeof(double*));
    cudaMemcpy(dAa, hA.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dWa, hW.data(), batch * sizeof(double*), cudaMemcpyHostToDevice);
    magma_int_t ret = magma_zheevj_batched_small_sm(MagmaVec, MagmaLower, n, dAa, n, dWa, dinfo, batch, Q); magma_queue_sync(Q);
    cudaError_t ce = cudaGetLastError(); if (ret || ce != cudaSuccess) fprintf(stderr, "heevj n=%d ret=%lld cuda=%s\n", n, (long long)ret, cudaGetErrorString(ce));
    std::vector<magma_int_t> inf(batch); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, inf.data(), 1, Q);
    for (int b = 0; b < batch; b++) { const std::string& k = KINDS[b % KINDS.size()]; std::vector<cd> Zm((size_t)n * n); std::vector<double> w(n); magma_zgetmatrix(n, n, hA[b], n, zp(Zm), n, Q); magma_dgetvector(n, hW[b], 1, w.data(), 1, Q);
      bool inFin = fin(A0[b]); double rv = 0, rr = 0, ro = 0;
      if (inFin) { auto R = A0[b]; std::vector<double> wr(n); magma_int_t nn = n, lw = 4 * n, il; std::vector<cd> wk(lw); std::vector<double> rw(3 * n); lapackf77_zheev("N", "L", &nn, zp(R), &nn, wr.data(), zp(wk), &lw, rw.data(), &il);
        std::vector<double> ws = w; std::sort(ws.begin(), ws.end()); double scale = std::max(1e-300, std::max(std::fabs(wr[0]), std::fabs(wr[n - 1]))), e = 0; for (int i = 0; i < n; i++) e = std::max(e, std::fabs(ws[i] - wr[i]));
        rv = e / (scale * n * EPS); if (scale == 1e-300) rv = e == 0 ? 0 : INFINITY;
        auto AZ = mul(A0[b], n, n, Zm, n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) AZ[i + (size_t)j * n] -= Zm[i + (size_t)j * n] * w[j];
        double an = std::max(maxabs(A0[b]), 1e-300); rr = maxabs(AZ) / (an * n * EPS); if (an == 1e-300) rr = maxabs(AZ) == 0 ? 0 : INFINITY;
        auto ZZ = mul(adj(Zm, n, n), n, n, Zm, n); for (int i = 0; i < n; i++) ZZ[i + (size_t)i * n] -= 1.0; ro = maxabs(ZZ) / (n * EPS); }
      out("zheevj_batched_small_sm", k, n, n, b, inf[b], rv, rr, ro, inFin, fin(w) && fin(Zm)); }
    magma_free(dA); magma_free(dW); magma_free(dinfo); magma_free(dAa); magma_free(dWa);
  }
}

static void gesvj() {
  std::vector<std::pair<int, int>> shapes; for (int n : {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 48, 64}) { shapes.push_back({n, n}); shapes.push_back({2 * n + 3, n}); }
  for (auto [m, n] : shapes) {
    const int batch = (int)KINDS.size() * 7; std::vector<std::vector<cd>> A0(batch); for (int b = 0; b < batch; b++) A0[b] = gen(KINDS[b % KINDS.size()], m, n, 2000u + 11u * b + 3 * m + n);
    Z *dA, *dU, *dV; double* dS; magma_int_t* dinfo; magma_zmalloc(&dA, (size_t)m * n * batch); magma_zmalloc(&dU, (size_t)m * n * batch); magma_zmalloc(&dV, (size_t)n * n * batch); magma_dmalloc(&dS, (size_t)n * batch); magma_imalloc(&dinfo, batch);
    std::vector<Z*> hA(batch), hU(batch), hV(batch); std::vector<double*> hS(batch);
    for (int b = 0; b < batch; b++) { hA[b] = dA + (size_t)b * m * n; hU[b] = dU + (size_t)b * m * n; hV[b] = dV + (size_t)b * n * n; hS[b] = dS + (size_t)b * n; magma_zsetmatrix(m, n, zp(A0[b]), m, hA[b], m, Q); }
    Z **dAa, **dUa, **dVa; double** dSa; magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dUa, batch * sizeof(Z*)); magma_malloc((void**)&dVa, batch * sizeof(Z*)); magma_malloc((void**)&dSa, batch * sizeof(double*));
    cudaMemcpy(dAa, hA.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dUa, hU.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice);
    cudaMemcpy(dVa, hV.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dSa, hS.data(), batch * sizeof(double*), cudaMemcpyHostToDevice);
    magma_int_t ret = magma_zgesvj_batched(MagmaSomeVec, MagmaSomeVec, m, n, dAa, m, dSa, dUa, m, dVa, n, dinfo, batch, Q); magma_queue_sync(Q);
    cudaError_t ce = cudaGetLastError(); if (ret || ce != cudaSuccess) fprintf(stderr, "gesvj m=%d n=%d ret=%lld cuda=%s\n", m, n, (long long)ret, cudaGetErrorString(ce));
    std::vector<magma_int_t> inf(batch); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, inf.data(), 1, Q);
    for (int b = 0; b < batch; b++) { const std::string& k = KINDS[b % KINDS.size()]; std::vector<cd> U((size_t)m * n), V((size_t)n * n); std::vector<double> s(n);
      magma_zgetmatrix(m, n, hU[b], m, zp(U), m, Q); magma_zgetmatrix(n, n, hV[b], n, zp(V), n, Q); magma_dgetvector(n, hS[b], 1, s.data(), 1, Q);
      bool inFin = fin(A0[b]); double rv = 0, rr = 0, ro = 0;
      if (inFin) { auto R = A0[b]; std::vector<double> sr(n); magma_int_t mm = m, nn = n, one = 1, lw = 8 * m + 64, il; std::vector<cd> wk(lw), d1(1); std::vector<double> rw(5 * n);
        lapackf77_zgesvd("N", "N", &mm, &nn, zp(R), &mm, sr.data(), zp(d1), &one, zp(d1), &one, zp(wk), &lw, rw.data(), &il);
        std::vector<double> ss = s; std::sort(ss.rbegin(), ss.rend()); double scale = std::max(sr[0], 1e-300), e = 0; for (int i = 0; i < n; i++) e = std::max(e, std::fabs(ss[i] - sr[i]));
        rv = e / (scale * std::max(m, n) * EPS); if (sr[0] == 0) rv = e == 0 ? 0 : INFINITY;
        std::vector<cd> US((size_t)m * n); for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) US[i + (size_t)j * m] = U[i + (size_t)j * m] * s[j];
        auto R2 = mul(US, m, n, adj(V, n, n), n); for (size_t t = 0; t < R2.size(); t++) R2[t] -= A0[b][t]; double an = std::max(maxabs(A0[b]), 1e-300);
        rr = maxabs(R2) / (an * std::max(m, n) * EPS); if (maxabs(A0[b]) == 0) rr = maxabs(R2) == 0 ? 0 : INFINITY;
        auto VV = mul(adj(V, n, n), n, n, V, n); for (int i = 0; i < n; i++) VV[i + (size_t)i * n] -= 1.0; ro = maxabs(VV) / (n * EPS);
        // U columns with nonzero sigma must be orthonormal
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { if (s[i] <= sr[0] * 1e-10 || s[j] <= sr[0] * 1e-10) continue; cd d = 0; for (int t = 0; t < m; t++) d += std::conj(U[t + (size_t)i * m]) * U[t + (size_t)j * m];
            ro = std::max(ro, std::abs(d - (i == j ? 1.0 : 0.0)) / (m * EPS)); } }
      out("zgesvj_batched", k, m, n, b, inf[b], rv, rr, ro, inFin, fin(s) && fin(U) && fin(V)); }
    magma_free(dA); magma_free(dU); magma_free(dV); magma_free(dS); magma_free(dinfo); magma_free(dAa); magma_free(dUa); magma_free(dVa); magma_free(dSa);
  }
}

// print W and Z of heevj for tiny matrices, batch of 3: [[2,1],[1,3]], I, diag(1,2)
static void heevjdbg() {
  const int n = 2, batch = 12; std::vector<std::vector<cd>> A0 = {{2, 1, 1, 3}, {1, 0, 0, 1}, {1, 0, 0, 2}, {2, cd(1, -1), cd(1, 1), 3}, {1, cd(0, -1e-16), cd(0, 1e-16), 1},
    {0, 1, 1, 0}, {1, 2, 2, -1}, {-2, 0, 0, 1}, {-3, 1, 1, -2}, {3, 1, 1, 2}, {5, 1, 1, 1}, {2, 1, 1, 2}};
  Z* dA; double* dW; magma_int_t* dinfo; magma_zmalloc(&dA, n * n * batch); magma_dmalloc(&dW, n * batch); magma_imalloc(&dinfo, batch);
  std::vector<Z*> hA(batch); std::vector<double*> hW(batch); for (int b = 0; b < batch; b++) { hA[b] = dA + b * n * n; hW[b] = dW + b * n; magma_zsetmatrix(n, n, zp(A0[b]), n, hA[b], n, Q); }
  Z** dAa; double** dWa; magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dWa, batch * sizeof(double*));
  cudaMemcpy(dAa, hA.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dWa, hW.data(), batch * sizeof(double*), cudaMemcpyHostToDevice);
  for (int pass = 0; pass < 2; pass++) {
    for (int b = 0; b < batch; b++) magma_zsetmatrix(n, n, zp(A0[b]), n, hA[b], n, Q);
    magma_int_t ret = magma_zheevj_batched_small_sm(pass ? MagmaNoVec : MagmaVec, MagmaLower, n, dAa, n, dWa, dinfo, batch, Q); magma_queue_sync(Q);
    std::vector<magma_int_t> inf(batch); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, inf.data(), 1, Q);
    for (int b = 0; b < batch; b++) { std::vector<cd> Zm(n * n); std::vector<double> w(n); magma_zgetmatrix(n, n, hA[b], n, zp(Zm), n, Q); magma_dgetvector(n, hW[b], 1, w.data(), 1, Q);
      double res = 0; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { cd az = 0; for (int l = 0; l < n; l++) az += A0[b][i + l * n] * Zm[l + j * n]; res = std::max(res, std::abs(az - w[j] * Zm[i + j * n])); }
      if (pass == 0) printf("  [A%d: diag %g %g, max|A z_j - w_j z_j| = %.3g]\n", b, A0[b][0].real(), A0[b][3].real(), res);
      printf("jobz=%c A%d ret=%lld info=%lld  W = [%.17g, %.17g]  Z = [[%.6g%+.6gi, %.6g%+.6gi], [%.6g%+.6gi, %.6g%+.6gi]] (columns)\n", pass ? 'N' : 'V', b, (long long)ret, (long long)inf[b], w[0], w[1],
             Zm[0].real(), Zm[0].imag(), Zm[1].real(), Zm[1].imag(), Zm[2].real(), Zm[2].imag(), Zm[3].real(), Zm[3].imag()); } }
}

// identical matrices in every batch slot: do results depend on batchCount / slot?  heevjbatch <n>
static void heevjbatch(int n) {
  Sd = 4242u; auto Aone = herm("spread", n, 31337u);
  for (int batch : {1, 2, 3, 4, 7, 8, 16, 31, 32, 56, 64}) {
    Z* dA; double* dW; magma_int_t* dinfo; magma_zmalloc(&dA, (size_t)n * n * batch); magma_dmalloc(&dW, (size_t)n * batch); magma_imalloc(&dinfo, batch);
    std::vector<Z*> hA(batch); std::vector<double*> hW(batch); for (int b = 0; b < batch; b++) { hA[b] = dA + (size_t)b * n * n; hW[b] = dW + (size_t)b * n; magma_zsetmatrix(n, n, zp(Aone), n, hA[b], n, Q); }
    Z** dAa; double** dWa; magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dWa, batch * sizeof(double*));
    cudaMemcpy(dAa, hA.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dWa, hW.data(), batch * sizeof(double*), cudaMemcpyHostToDevice);
    magma_zheevj_batched_small_sm(MagmaVec, MagmaLower, n, dAa, n, dWa, dinfo, batch, Q); magma_queue_sync(Q);
    int bad = 0; double worst = 0; int firstbad = -1;
    for (int b = 0; b < batch; b++) { std::vector<cd> Zm((size_t)n * n); std::vector<double> w(n); magma_zgetmatrix(n, n, hA[b], n, zp(Zm), n, Q); magma_dgetvector(n, hW[b], 1, w.data(), 1, Q);
      auto AZ = mul(Aone, n, n, Zm, n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) AZ[i + (size_t)j * n] -= Zm[i + (size_t)j * n] * w[j];
      double r = maxabs(AZ) / (maxabs(Aone) * n * EPS); worst = std::max(worst, r); if (!(r < 100)) { bad++; if (firstbad < 0) firstbad = b; } }
    printf("heevj n=%d batch=%2d: %d of %d slots wrong (first %d), worst residual ratio %.3g\n", n, batch, bad, batch, firstbad, worst);
    magma_free(dA); magma_free(dW); magma_free(dinfo); magma_free(dAa); magma_free(dWa);
  }
}

// root-cause check without touching the library: the expert interface with sort_flag = 0 (no swaps) vs 1
static void heevjsort() {
  printf("sizeof(long) on this platform = %d bytes\n", (int)sizeof(long));
  for (int n : {2, 5, 16, 32}) for (int sortf = 0; sortf < 2; sortf++) {
    const int batch = 16; int bad = 0; double worst = 0;
    std::vector<std::vector<cd>> A0(batch); for (int b = 0; b < batch; b++) A0[b] = herm("spread", n, 31337u + b);
    Z* dA; double* dW; magma_int_t* dinfo; magma_zmalloc(&dA, (size_t)n * n * batch); magma_dmalloc(&dW, (size_t)n * batch); magma_imalloc(&dinfo, batch);
    std::vector<Z*> hA(batch); std::vector<double*> hW(batch); for (int b = 0; b < batch; b++) { hA[b] = dA + (size_t)b * n * n; hW[b] = dW + (size_t)b * n; magma_zsetmatrix(n, n, zp(A0[b]), n, hA[b], n, Q); }
    Z** dAa; double** dWa; magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dWa, batch * sizeof(double*));
    cudaMemcpy(dAa, hA.data(), batch * sizeof(Z*), cudaMemcpyHostToDevice); cudaMemcpy(dWa, hW.data(), batch * sizeof(double*), cudaMemcpyHostToDevice);
    magma_zheevj_batched_expert_small_sm(MagmaVec, MagmaLower, n, dAa, n, dWa, dinfo, NULL, NULL, 30, sortf, 100, magma_get_zheevj_batched_small_nthreads(n), batch, Q); magma_queue_sync(Q);
    for (int b = 0; b < batch; b++) { std::vector<cd> Zm((size_t)n * n); std::vector<double> w(n); magma_zgetmatrix(n, n, hA[b], n, zp(Zm), n, Q); magma_dgetvector(n, hW[b], 1, w.data(), 1, Q);
      auto AZ = mul(A0[b], n, n, Zm, n); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) AZ[i + (size_t)j * n] -= Zm[i + (size_t)j * n] * w[j];
      double r = maxabs(AZ) / (maxabs(A0[b]) * n * EPS); worst = std::max(worst, r); if (!(r < 100)) bad++; }
    printf("heevj expert n=%2d sort_flag=%d: %2d of %d matrices wrong, worst residual ratio %.3g\n", n, sortf, bad, batch, worst);
    magma_free(dA); magma_free(dW); magma_free(dinfo); magma_free(dAa); magma_free(dWa);
  }
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string g = argc > 1 ? argv[1] : ""; magma_init(); magma_queue_create(0, &Q);
  printf("routine,kind,m,n,batch_index,info,ratio_values,ratio_residual,ratio_orth,verdict\n");
  if (g == "heevj") heevj(); else if (g == "heevjdbg") heevjdbg(); else if (g == "heevjsort") heevjsort(); else if (g == "heevjbatch" && argc > 2) heevjbatch(atoi(argv[2])); else if (g == "gesvj") gesvj(); else printf("unknown\n");
  fprintf(stderr, "DONE %s cuda=%s\n", g.c_str(), cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

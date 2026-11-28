// Round 34 (roxy): differential test of the batched and vbatched BLAS in magmablas (d precision; z/c/s are generated
// from the same templates).  Reference: naive host loops with the reference-BLAS contract
//   * beta = 0: C / y is not read (NaN in C must disappear);  alpha = 0: A, B are not read;  k = 0: C := beta*C
//   * syrk / syr2k: only the uplo triangle of C is referenced and changed; the other triangle stays bit-identical
//   * trsm / trmm: alpha = 0 sets B to zero; only the uplo triangle of A is read (the other triangle holds NaN)
// Cases: transposes, sizes 0/1/odd/>64, ld = rows + {0..3}, batchCount 3 (batched) or 5 mixed sizes incl. zeros (vbatched).
// CSV: bblas,routine,case,status,maxerr,detail   status OK | WRONG (finite, err > 1) | NAN (non-finite where ref finite)
//      | TOUCHED (element outside the referenced triangle / beyond m,n changed)
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
static magma_queue_t Q;
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static const double EPS = 1.1102230246251565e-16;

struct M { int r, c, ld; std::vector<double> h; double* d = nullptr; };
static M mk(int r, int c, int pad, double fill = NAN, bool rnd = true) {
  M m; m.r = r; m.c = c; m.ld = std::max(1, r) + pad; m.h.assign((size_t)m.ld * std::max(1, c), fill);   // padding and unused entries = fill
  if (rnd) for (int j = 0; j < c; j++) for (int i = 0; i < r; i++) m.h[i + (size_t)j * m.ld] = r1();
  magma_dmalloc(&m.d, m.h.size()); magma_dsetvector((magma_int_t)m.h.size(), m.h.data(), 1, m.d, 1, Q); return m; }
static std::vector<double> back(const M& m) { std::vector<double> v(m.h.size()); magma_dgetvector((magma_int_t)v.size(), m.d, 1, v.data(), 1, Q); return v; }
static double** parr(std::vector<M>& v) { std::vector<double*> h; for (auto& m : v) h.push_back(m.d); double** d; magma_malloc((void**)&d, h.size() * sizeof(double*)); cudaMemcpy(d, h.data(), h.size() * sizeof(double*), cudaMemcpyHostToDevice); return d; }
static magma_int_t* iarr(const std::vector<int>& v) { std::vector<magma_int_t> h(v.begin(), v.end()); h.push_back(0); magma_int_t* d; magma_imalloc(&d, h.size()); magma_isetvector((magma_int_t)h.size(), h.data(), 1, d, 1, Q); return d; }
static void freeAll(std::vector<M>& v) { for (auto& m : v) magma_free(m.d); }

// compare result (got) with reference (ref) and bound (scale for the rounding error); mask(i,j) = element is written by the op
struct Acc { double err = 0; int nan = 0, touched = 0, wrong = 0; std::string where; };
static void cmp(Acc& a, const M& out, const std::vector<double>& got, const std::vector<double>& ref, const std::vector<double>& bound, int k, std::function<bool(int, int)> written) {
  for (int j = 0; j < std::max(1, out.c); j++) for (int i = 0; i < out.ld; i++) {
    size_t p = i + (size_t)j * out.ld; if (p >= got.size()) continue; bool w = i < out.r && j < out.c && written(i, j);
    if (!w) { if (!(got[p] == out.h[p] || (std::isnan(got[p]) && std::isnan(out.h[p])))) { if (!a.touched++) a.where = "touched(" + std::to_string(i) + ";" + std::to_string(j) + ")"; } continue; }
    if (!std::isfinite(got[p])) { if (std::isfinite(ref[p])) { if (!a.nan++ && a.where.empty()) a.where = "nan(" + std::to_string(i) + ";" + std::to_string(j) + ")"; } continue; }
    double e = fabs(got[p] - ref[p]) / (8.0 * (k + 2) * EPS * bound[p] + 1e-300); if (e > a.err) a.err = e;
    if (e > 1 && !a.wrong++ && a.where.empty()) a.where = "wrong(" + std::to_string(i) + ";" + std::to_string(j) + ")";
  } }
static void report(const char* rt, const std::string& cs, const Acc& a) {
  const char* s = a.touched ? "TOUCHED" : a.nan ? "NAN" : a.wrong ? "WRONG" : "OK";
  printf("bblas,%s,%s,%s,%.3g,%s touched=%d nan=%d wrong=%d\n", rt, cs.c_str(), s, a.err, a.where.c_str(), a.touched, a.nan, a.wrong); }
static double el(const M& m, int i, int j) { return m.h[i + (size_t)j * m.ld]; }

// ---------------------------------------------------------------- gemm
static void refGemm(const M& A, const M& B, const M& C, char ta, char tb, int m, int n, int k, double al, double be, std::vector<double>& ref, std::vector<double>& bd) {
  ref = C.h; bd.assign(C.h.size(), 0);
  for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { double s = 0, b = 0;
    if (al != 0) for (int l = 0; l < k; l++) { double x = ta == 'N' ? el(A, i, l) : el(A, l, i), y = tb == 'N' ? el(B, l, j) : el(B, j, l); s += x * y; b += fabs(x * y); }
    size_t p = i + (size_t)j * C.ld; double c = C.h[p]; ref[p] = al * s + (be == 0 ? 0 : be * c); bd[p] = fabs(al) * b + (be == 0 ? 0 : fabs(be * c)) + 1e-300; } }
static void gemm(bool vb) {
  struct S { int m, n, k; }; std::vector<std::vector<S>> sets = vb ? std::vector<std::vector<S>>{{{31, 33, 17}, {0, 5, 5}, {7, 0, 3}, {12, 9, 0}, {65, 40, 129}}, {{1, 1, 1}, {33, 1, 64}, {1, 65, 2}, {0, 0, 0}, {40, 40, 40}}}
      : std::vector<std::vector<S>>{{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, {{31, 33, 17}, {31, 33, 17}, {31, 33, 17}}, {{65, 7, 0}, {65, 7, 0}, {65, 7, 0}}, {{40, 40, 129}, {40, 40, 129}, {40, 40, 129}}, {{5, 3, 1}, {5, 3, 1}, {5, 3, 1}}};
  for (size_t si = 0; si < sets.size(); si++) for (char ta : {'N', 'T'}) for (char tb : {'N', 'T'}) for (double al : {1.0, 0.0, -0.5}) for (double be : {0.0, 1.0, 0.5}) {
    auto& ss = sets[si]; int B = (int)ss.size(); Sd = 17u + (unsigned)si; std::vector<M> A, Bm, C; std::vector<int> vm, vn, vk, la, lb, lc;
    for (int b = 0; b < B; b++) { auto s = ss[b]; int pad = vb ? b % 4 : 1;
      A.push_back(ta == 'N' ? mk(s.m, s.k, pad) : mk(s.k, s.m, pad)); Bm.push_back(tb == 'N' ? mk(s.k, s.n, pad) : mk(s.n, s.k, pad));
      C.push_back(mk(s.m, s.n, pad)); if (be == 0) for (auto& x : C.back().h) x = NAN;   // beta = 0: C holds NaN, must be overwritten
      magma_dsetvector((magma_int_t)C.back().h.size(), C.back().h.data(), 1, C.back().d, 1, Q);
      vm.push_back(s.m); vn.push_back(s.n); vk.push_back(s.k); la.push_back(A.back().ld); lb.push_back(Bm.back().ld); lc.push_back(C.back().ld); }
    double **dA = parr(A), **dB = parr(Bm), **dC = parr(C); magma_trans_t tA = ta == 'N' ? MagmaNoTrans : MagmaTrans, tB = tb == 'N' ? MagmaNoTrans : MagmaTrans;
    if (vb) { magma_int_t *m = iarr(vm), *n = iarr(vn), *k = iarr(vk), *a = iarr(la), *bb = iarr(lb), *c = iarr(lc);
      magmablas_dgemm_vbatched(tA, tB, m, n, k, al, (double const* const*)dA, a, (double const* const*)dB, bb, be, dC, c, B, Q); magma_queue_sync(Q);
      magma_free(m); magma_free(n); magma_free(k); magma_free(a); magma_free(bb); magma_free(c); }
    else { magmablas_dgemm_batched(tA, tB, ss[0].m, ss[0].n, ss[0].k, al, (double const* const*)dA, la[0], (double const* const*)dB, lb[0], be, dC, lc[0], B, Q); magma_queue_sync(Q); }
    Acc acc; for (int b = 0; b < B; b++) { std::vector<double> ref, bd; refGemm(A[b], Bm[b], C[b], ta, tb, vm[b], vn[b], vk[b], al, be, ref, bd); cmp(acc, C[b], back(C[b]), ref, bd, vk[b], [](int, int) { return true; }); }
    char cs[128]; snprintf(cs, sizeof cs, "set%zu %c%c al=%g be=%g", si, ta, tb, al, be); report(vb ? "dgemm_vbatched" : "dgemm_batched", cs, acc);
    freeAll(A); freeAll(Bm); freeAll(C); magma_free(dA); magma_free(dB); magma_free(dC); }
}

// ---------------------------------------------------------------- gemv
static void gemv(bool vb) {
  struct S { int m, n, inc; }; std::vector<std::vector<S>> sets = vb ? std::vector<std::vector<S>>{{{31, 33, 1}, {0, 5, 1}, {7, 0, 2}, {65, 129, 1}, {1, 1, 3}}}
      : std::vector<std::vector<S>>{{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, {{31, 33, 2}, {31, 33, 2}, {31, 33, 2}}, {{0, 7, 1}, {0, 7, 1}, {0, 7, 1}}, {{65, 129, 1}, {65, 129, 1}, {65, 129, 1}}, {{7, 0, 1}, {7, 0, 1}, {7, 0, 1}}};
  for (size_t si = 0; si < sets.size(); si++) for (char t : {'N', 'T'}) for (double al : {1.0, 0.0, -0.5}) for (double be : {0.0, 1.0, 0.5}) {
    auto& ss = sets[si]; int B = (int)ss.size(); Sd = 29u + (unsigned)si; std::vector<M> A, X, Y; std::vector<int> vm, vn, la, ix, iy;
    for (int b = 0; b < B; b++) { auto s = ss[b]; int lx = t == 'N' ? s.n : s.m, ly = t == 'N' ? s.m : s.n;
      A.push_back(mk(s.m, s.n, (vb ? b % 3 : 1))); X.push_back(mk(std::max(1, 1 + (lx - 1) * s.inc), 1, 0)); Y.push_back(mk(std::max(1, 1 + (ly - 1) * s.inc), 1, 0));
      if (be == 0) { for (auto& x : Y.back().h) x = NAN; magma_dsetvector((magma_int_t)Y.back().h.size(), Y.back().h.data(), 1, Y.back().d, 1, Q); }
      vm.push_back(s.m); vn.push_back(s.n); la.push_back(A.back().ld); ix.push_back(s.inc); iy.push_back(s.inc); }
    double **dA = parr(A), **dX = parr(X), **dY = parr(Y); magma_trans_t tr = t == 'N' ? MagmaNoTrans : MagmaTrans;
    if (vb) { magma_int_t *m = iarr(vm), *n = iarr(vn), *a = iarr(la), *x = iarr(ix), *y = iarr(iy);
      magmablas_dgemv_vbatched(tr, m, n, al, dA, a, dX, x, be, dY, y, B, Q); magma_queue_sync(Q); magma_free(m); magma_free(n); magma_free(a); magma_free(x); magma_free(y); }
    else { magmablas_dgemv_batched(tr, ss[0].m, ss[0].n, al, (double const* const*)dA, la[0], (double const* const*)dX, ix[0], be, dY, iy[0], B, Q); magma_queue_sync(Q); }
    Acc acc; for (int b = 0; b < B; b++) { auto s = ss[b]; int ly = t == 'N' ? s.m : s.n, lx = t == 'N' ? s.n : s.m; std::vector<double> ref = Y[b].h, bd(ref.size(), 1e-300); std::vector<bool> wr(ref.size(), false);
      // reference BLAS: quick return when m = 0 or n = 0 (y is not scaled by beta, even for trans = T)
      for (int i = 0; i < ((s.m == 0 || s.n == 0) ? 0 : ly); i++) { double sm = 0, bb = 0; if (al != 0) for (int l = 0; l < lx; l++) { double a = t == 'N' ? el(A[b], i, l) : el(A[b], l, i); double x = X[b].h[(size_t)l * s.inc]; sm += a * x; bb += fabs(a * x); }
        size_t p = (size_t)i * s.inc; ref[p] = al * sm + (be == 0 ? 0 : be * Y[b].h[p]); bd[p] = fabs(al) * bb + (be == 0 ? 0 : fabs(be * Y[b].h[p])) + 1e-300; wr[p] = true; }
      M yv = Y[b]; yv.r = (int)ref.size(); yv.c = 1; yv.ld = (int)ref.size(); cmp(acc, yv, back(Y[b]), ref, bd, lx, [&](int i, int) { return (bool)wr[i]; }); }
    char cs[128]; snprintf(cs, sizeof cs, "set%zu %c al=%g be=%g", si, t, al, be); report(vb ? "dgemv_vbatched" : "dgemv_batched", cs, acc);
    freeAll(A); freeAll(X); freeAll(Y); magma_free(dA); magma_free(dX); magma_free(dY); }
}

// ---------------------------------------------------------------- syrk / syr2k (C := alpha op(A) op(A)^T + beta C, uplo triangle only)
static void syrk(bool vb, bool two) {
  struct S { int n, k; }; std::vector<std::vector<S>> sets = vb ? std::vector<std::vector<S>>{{{31, 17}, {0, 5}, {7, 0}, {65, 129}, {1, 1}}}
      : std::vector<std::vector<S>>{{{1, 1}, {1, 1}, {1, 1}}, {{31, 17}, {31, 17}, {31, 17}}, {{40, 0}, {40, 0}, {40, 0}}, {{65, 129}, {65, 129}, {65, 129}}};
  if (two && vb) return;   // syr2k_vbatched exists but is checked through the batched form only
  for (size_t si = 0; si < sets.size(); si++) for (char up : {'L', 'U'}) for (char t : {'N', 'T'}) for (double al : {1.0, 0.0}) for (double be : {0.0, 0.5}) {
    auto& ss = sets[si]; int B = (int)ss.size(); Sd = 41u + (unsigned)si; std::vector<M> A, Bm, C; std::vector<int> vn, vk, la, lc;
    for (int b = 0; b < B; b++) { auto s = ss[b]; A.push_back(t == 'N' ? mk(s.n, s.k, (vb ? b % 3 : 1)) : mk(s.k, s.n, (vb ? b % 3 : 1))); Bm.push_back(t == 'N' ? mk(s.n, s.k, (vb ? b % 3 : 1)) : mk(s.k, s.n, (vb ? b % 3 : 1)));
      C.push_back(mk(s.n, s.n, (vb ? b % 3 : 1))); if (be == 0) { for (int j = 0; j < s.n; j++) for (int i = 0; i < s.n; i++) if (up == 'L' ? i >= j : i <= j) C.back().h[i + (size_t)j * C.back().ld] = NAN;
        magma_dsetvector((magma_int_t)C.back().h.size(), C.back().h.data(), 1, C.back().d, 1, Q); }
      vn.push_back(s.n); vk.push_back(s.k); la.push_back(A.back().ld); lc.push_back(C.back().ld); }
    double **dA = parr(A), **dB = parr(Bm), **dC = parr(C); magma_uplo_t ul = up == 'L' ? MagmaLower : MagmaUpper; magma_trans_t tr = t == 'N' ? MagmaNoTrans : MagmaTrans;
    if (vb) { magma_int_t *n = iarr(vn), *k = iarr(vk), *a = iarr(la), *c = iarr(lc); magmablas_dsyrk_vbatched(ul, tr, n, k, al, (double const* const*)dA, a, be, dC, c, B, Q); magma_queue_sync(Q);
      magma_free(n); magma_free(k); magma_free(a); magma_free(c); }
    else if (!two) { magmablas_dsyrk_batched(ul, tr, ss[0].n, ss[0].k, al, (double const* const*)dA, la[0], be, dC, lc[0], B, Q); magma_queue_sync(Q); }
    else { magmablas_dsyr2k_batched(ul, tr, ss[0].n, ss[0].k, al, (double const* const*)dA, la[0], (double const* const*)dB, la[0], be, dC, lc[0], B, Q); magma_queue_sync(Q); }
    Acc acc; for (int b = 0; b < B; b++) { auto s = ss[b]; std::vector<double> ref = C[b].h, bd(ref.size(), 1e-300);
      auto op = [&](const M& X, int i, int l) { return t == 'N' ? el(X, i, l) : el(X, l, i); };
      for (int j = 0; j < s.n; j++) for (int i = 0; i < s.n; i++) { if (!(up == 'L' ? i >= j : i <= j)) continue; double sm = 0, bb = 0;
        if (al != 0) for (int l = 0; l < s.k; l++) { double v = two ? op(A[b], i, l) * op(Bm[b], j, l) + op(Bm[b], i, l) * op(A[b], j, l) : op(A[b], i, l) * op(A[b], j, l); sm += v;
            bb += two ? fabs(op(A[b], i, l) * op(Bm[b], j, l)) + fabs(op(Bm[b], i, l) * op(A[b], j, l)) : fabs(v); }
        size_t p = i + (size_t)j * C[b].ld; ref[p] = al * sm + (be == 0 ? 0 : be * C[b].h[p]); bd[p] = fabs(al) * bb + (be == 0 ? 0 : fabs(be * C[b].h[p])) + 1e-300; }
      cmp(acc, C[b], back(C[b]), ref, bd, s.k, [&](int i, int j) { return up == 'L' ? i >= j : i <= j; }); }
    char cs[128]; snprintf(cs, sizeof cs, "set%zu %c%c al=%g be=%g", si, up, t, al, be); report(vb ? "dsyrk_vbatched" : two ? "dsyr2k_batched" : "dsyrk_batched", cs, acc);
    freeAll(A); freeAll(Bm); freeAll(C); magma_free(dA); magma_free(dB); magma_free(dC); }
}

// ---------------------------------------------------------------- trsm / trmm (B := alpha op(A)^{-1} B or alpha op(A) B, side L/R)
static void trxm(bool vb, bool solve) {
  struct S { int m, n; }; std::vector<std::vector<S>> sets = vb ? std::vector<std::vector<S>>{{{31, 17}, {0, 5}, {7, 0}, {65, 40}, {1, 1}}}
      : std::vector<std::vector<S>>{{{1, 1}, {1, 1}, {1, 1}}, {{31, 17}, {31, 17}, {31, 17}}, {{65, 40}, {65, 40}, {65, 40}}, {{33, 129}, {33, 129}, {33, 129}}};
  for (size_t si = 0; si < sets.size(); si++) for (char sd : {'L', 'R'}) for (char up : {'L', 'U'}) for (char t : {'N', 'T'}) for (char dg : {'N', 'U'}) for (double al : {1.0, 0.0, -2.0}) {
    auto& ss = sets[si]; int B = (int)ss.size(); Sd = 53u + (unsigned)si; std::vector<M> A, X; std::vector<int> vm, vn, la, lb;
    for (int b = 0; b < B; b++) { auto s = ss[b]; int na = sd == 'L' ? s.m : s.n; M a = mk(na, na, (vb ? b % 3 : 1));
      for (int j = 0; j < na; j++) for (int i = 0; i < na; i++) { double& v = a.h[i + (size_t)j * a.ld]; bool in = up == 'L' ? i >= j : i <= j; v = !in ? NAN : i == j ? (dg == 'U' ? NAN : 2.0 + 0.5 * r1()) : 0.3 * r1() / std::max(1, na); }
      magma_dsetvector((magma_int_t)a.h.size(), a.h.data(), 1, a.d, 1, Q); A.push_back(a); X.push_back(mk(s.m, s.n, (vb ? b % 3 : 1)));
      if (al == 0) { for (auto& x : X.back().h) x = NAN; magma_dsetvector((magma_int_t)X.back().h.size(), X.back().h.data(), 1, X.back().d, 1, Q); }   // alpha = 0: B is not read
      vm.push_back(s.m); vn.push_back(s.n); la.push_back(A.back().ld); lb.push_back(X.back().ld); }
    double **dA = parr(A), **dX = parr(X); magma_side_t si_ = sd == 'L' ? MagmaLeft : MagmaRight; magma_uplo_t ul = up == 'L' ? MagmaLower : MagmaUpper;
    magma_trans_t tr = t == 'N' ? MagmaNoTrans : MagmaTrans; magma_diag_t di = dg == 'N' ? MagmaNonUnit : MagmaUnit;
    if (vb) { magma_int_t *m = iarr(vm), *n = iarr(vn), *a = iarr(la), *bb = iarr(lb);
      if (solve) magmablas_dtrsm_vbatched(si_, ul, tr, di, m, n, al, dA, a, dX, bb, B, Q); else magmablas_dtrmm_vbatched(si_, ul, tr, di, m, n, al, dA, a, dX, bb, B, Q);
      magma_queue_sync(Q); magma_free(m); magma_free(n); magma_free(a); magma_free(bb); }
    else { if (solve) magmablas_dtrsm_batched(si_, ul, tr, di, ss[0].m, ss[0].n, al, dA, la[0], dX, lb[0], B, Q); else magmablas_dtrmm_batched(si_, ul, tr, di, ss[0].m, ss[0].n, al, dA, la[0], dX, lb[0], B, Q); magma_queue_sync(Q); }
    Acc acc;
    for (int b = 0; b < B; b++) { auto s = ss[b]; int na = sd == 'L' ? s.m : s.n; std::vector<double> T((size_t)na * na, 0);   // dense op(A) with the implicit unit diagonal
      for (int j = 0; j < na; j++) for (int i = 0; i < na; i++) { bool in = up == 'L' ? i >= j : i <= j; double v = !in ? 0 : (i == j && dg == 'U') ? 1.0 : el(A[b], i, j); if (t == 'N') T[i + (size_t)j * na] = v; else T[j + (size_t)i * na] = v; }
      std::vector<double> ref = X[b].h, bd(ref.size(), 1e-300); std::vector<double> Bh((size_t)s.m * s.n); for (int j = 0; j < s.n; j++) for (int i = 0; i < s.m; i++) Bh[i + (size_t)j * s.m] = al == 0 ? 0 : al * el(X[b], i, j);
      std::vector<double> R = Bh;
      if (solve && al != 0) {   // solve T * R = Bh (left) or R * T = Bh (right) by substitution on the dense T
        if (sd == 'L') { for (int j = 0; j < s.n; j++) { bool lowerT = (up == 'L') == (t == 'N');
            if (lowerT) for (int i = 0; i < s.m; i++) { double v = Bh[i + (size_t)j * s.m]; for (int l = 0; l < i; l++) v -= T[i + (size_t)l * na] * R[l + (size_t)j * s.m]; R[i + (size_t)j * s.m] = v / T[i + (size_t)i * na]; }
            else for (int i = s.m - 1; i >= 0; i--) { double v = Bh[i + (size_t)j * s.m]; for (int l = i + 1; l < s.m; l++) v -= T[i + (size_t)l * na] * R[l + (size_t)j * s.m]; R[i + (size_t)j * s.m] = v / T[i + (size_t)i * na]; } } }
        else { for (int i = 0; i < s.m; i++) { bool lowerT = (up == 'L') == (t == 'N');   // x T = b: columns
            if (!lowerT) for (int j = 0; j < s.n; j++) { double v = Bh[i + (size_t)j * s.m]; for (int l = 0; l < j; l++) v -= R[i + (size_t)l * s.m] * T[l + (size_t)j * na]; R[i + (size_t)j * s.m] = v / T[j + (size_t)j * na]; }
            else for (int j = s.n - 1; j >= 0; j--) { double v = Bh[i + (size_t)j * s.m]; for (int l = j + 1; l < s.n; l++) v -= R[i + (size_t)l * s.m] * T[l + (size_t)j * na]; R[i + (size_t)j * s.m] = v / T[j + (size_t)j * na]; } } } }
      else if (!solve && al != 0) { for (int j = 0; j < s.n; j++) for (int i = 0; i < s.m; i++) { double v = 0; if (sd == 'L') for (int l = 0; l < s.m; l++) v += T[i + (size_t)l * na] * Bh[l + (size_t)j * s.m]; else for (int l = 0; l < s.n; l++) v += Bh[i + (size_t)l * s.m] * T[l + (size_t)j * na]; R[i + (size_t)j * s.m] = v; } }
      double mx = 0; for (double v : R) mx = std::max(mx, fabs(v));
      for (int j = 0; j < s.n; j++) for (int i = 0; i < s.m; i++) { size_t p = i + (size_t)j * X[b].ld; ref[p] = R[i + (size_t)j * s.m]; bd[p] = 64.0 * (mx + 1e-300); }
      cmp(acc, X[b], back(X[b]), ref, bd, na, [](int, int) { return true; }); }
    char cs[128]; snprintf(cs, sizeof cs, "set%zu %c%c%c%c al=%g", si, sd, up, t, dg, al); report(solve ? (vb ? "dtrsm_vbatched" : "dtrsm_batched") : (vb ? "dtrmm_vbatched" : "dtrmm_batched"), cs, acc);
    freeAll(A); freeAll(X); magma_free(dA); magma_free(dX); }
}

// ---------------------------------------------------------------- round 35: symm / symv (batched, vbatched), trsv_batched, potrs_batched
// The triangle of A that is not referenced holds NaN (must not be read); beta = 0 puts NaN into C / y.
static void symAfill(M& a, char up) { int n = a.r; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if (!(up == 'L' ? i >= j : i <= j)) a.h[i + (size_t)j * a.ld] = NAN;
  magma_dsetvector((magma_int_t)a.h.size(), a.h.data(), 1, a.d, 1, Q); }
static double symel(const M& a, char up, int i, int j) { bool in = up == 'L' ? i >= j : i <= j; return in ? el(a, i, j) : el(a, j, i); }
static void symm(bool vb) {
  struct S { int m, n; }; std::vector<std::vector<S>> sets = vb ? std::vector<std::vector<S>>{{{31, 17}, {0, 5}, {7, 0}, {65, 40}, {1, 1}}}
      : std::vector<std::vector<S>>{{{1, 1}, {1, 1}, {1, 1}}, {{31, 17}, {31, 17}, {31, 17}}, {{65, 40}, {65, 40}, {65, 40}}, {{33, 129}, {33, 129}, {33, 129}}};
  for (size_t si = 0; si < sets.size(); si++) for (char sd : {'L', 'R'}) for (char up : {'L', 'U'}) for (double al : {1.0, 0.0}) for (double be : {0.0, 0.5}) {
    auto& ss = sets[si]; int B = (int)ss.size(); Sd = 71u + (unsigned)si; std::vector<M> A, Bm, C; std::vector<int> vm, vn, la, lb, lc;
    for (int b = 0; b < B; b++) { auto s = ss[b]; int na = sd == 'L' ? s.m : s.n; int pad = vb ? b % 3 : 1; M a = mk(na, na, pad); symAfill(a, up); A.push_back(a);
      Bm.push_back(mk(s.m, s.n, pad)); C.push_back(mk(s.m, s.n, pad)); if (be == 0) { for (auto& x : C.back().h) x = NAN; magma_dsetvector((magma_int_t)C.back().h.size(), C.back().h.data(), 1, C.back().d, 1, Q); }
      vm.push_back(s.m); vn.push_back(s.n); la.push_back(A.back().ld); lb.push_back(Bm.back().ld); lc.push_back(C.back().ld); }
    double **dA = parr(A), **dB = parr(Bm), **dC = parr(C); magma_side_t sid = sd == 'L' ? MagmaLeft : MagmaRight; magma_uplo_t ul = up == 'L' ? MagmaLower : MagmaUpper;
    if (vb) { magma_int_t *m = iarr(vm), *n = iarr(vn), *a = iarr(la), *bb = iarr(lb), *c = iarr(lc); magmablas_dsymm_vbatched(sid, ul, m, n, al, dA, a, dB, bb, be, dC, c, B, Q); magma_queue_sync(Q);
      magma_free(m); magma_free(n); magma_free(a); magma_free(bb); magma_free(c); }
    else { magmablas_dsymm_batched(sid, ul, ss[0].m, ss[0].n, al, dA, la[0], dB, lb[0], be, dC, lc[0], B, Q); magma_queue_sync(Q); }
    Acc acc; for (int b = 0; b < B; b++) { auto s = ss[b]; std::vector<double> ref = C[b].h, bd(ref.size(), 1e-300); int kk = sd == 'L' ? s.m : s.n;
      for (int j = 0; j < s.n; j++) for (int i = 0; i < s.m; i++) { double sm = 0, bb = 0;
        if (al != 0) for (int l = 0; l < kk; l++) { double v = sd == 'L' ? symel(A[b], up, i, l) * el(Bm[b], l, j) : el(Bm[b], i, l) * symel(A[b], up, l, j); sm += v; bb += fabs(v); }
        size_t p = i + (size_t)j * C[b].ld; ref[p] = al * sm + (be == 0 ? 0 : be * C[b].h[p]); bd[p] = fabs(al) * bb + (be == 0 ? 0 : fabs(be * C[b].h[p])) + 1e-300; }
      cmp(acc, C[b], back(C[b]), ref, bd, kk, [](int, int) { return true; }); }
    char cs[128]; snprintf(cs, sizeof cs, "set%zu %c%c al=%g be=%g", si, sd, up, al, be); report(vb ? "dsymm_vbatched" : "dsymm_batched", cs, acc);
    freeAll(A); freeAll(Bm); freeAll(C); magma_free(dA); magma_free(dB); magma_free(dC); }
}
static void symv(bool vb) {
  struct S { int n, inc; }; std::vector<std::vector<S>> sets = vb ? std::vector<std::vector<S>>{{{31, 1}, {0, 1}, {65, 2}, {1, 1}, {129, 1}}}
      : std::vector<std::vector<S>>{{{1, 1}, {1, 1}, {1, 1}}, {{31, 2}, {31, 2}, {31, 2}}, {{65, 1}, {65, 1}, {65, 1}}, {{129, 1}, {129, 1}, {129, 1}}};
  for (size_t si = 0; si < sets.size(); si++) for (char up : {'L', 'U'}) for (double al : {1.0, 0.0, -0.5}) for (double be : {0.0, 1.0, 0.5}) {
    auto& ss = sets[si]; int B = (int)ss.size(); Sd = 83u + (unsigned)si; std::vector<M> A, X, Y; std::vector<int> vn, la, ix, iy;
    for (int b = 0; b < B; b++) { auto s = ss[b]; M a = mk(s.n, s.n, vb ? b % 3 : 1); symAfill(a, up); A.push_back(a);
      X.push_back(mk(std::max(1, 1 + (s.n - 1) * s.inc), 1, 0)); Y.push_back(mk(std::max(1, 1 + (s.n - 1) * s.inc), 1, 0));
      if (be == 0) { for (auto& x : Y.back().h) x = NAN; magma_dsetvector((magma_int_t)Y.back().h.size(), Y.back().h.data(), 1, Y.back().d, 1, Q); }
      vn.push_back(s.n); la.push_back(A.back().ld); ix.push_back(s.inc); iy.push_back(s.inc); }
    double **dA = parr(A), **dX = parr(X), **dY = parr(Y); magma_uplo_t ul = up == 'L' ? MagmaLower : MagmaUpper;
    if (vb) { magma_int_t *n = iarr(vn), *a = iarr(la), *x = iarr(ix), *y = iarr(iy); magmablas_dsymv_vbatched(ul, n, al, dA, a, dX, x, be, dY, y, B, Q); magma_queue_sync(Q);
      magma_free(n); magma_free(a); magma_free(x); magma_free(y); }
    else { magmablas_dsymv_batched(ul, ss[0].n, al, dA, la[0], dX, ix[0], be, dY, iy[0], B, Q); magma_queue_sync(Q); }
    Acc acc; for (int b = 0; b < B; b++) { auto s = ss[b]; std::vector<double> ref = Y[b].h, bd(ref.size(), 1e-300); std::vector<bool> wr(ref.size(), false);
      for (int i = 0; i < s.n; i++) { double sm = 0, bb = 0; if (al != 0) for (int l = 0; l < s.n; l++) { double v = symel(A[b], up, i, l) * X[b].h[(size_t)l * s.inc]; sm += v; bb += fabs(v); }
        size_t p = (size_t)i * s.inc; ref[p] = al * sm + (be == 0 ? 0 : be * Y[b].h[p]); bd[p] = fabs(al) * bb + (be == 0 ? 0 : fabs(be * Y[b].h[p])) + 1e-300; wr[p] = true; }
      M yv = Y[b]; yv.r = (int)ref.size(); yv.c = 1; yv.ld = (int)ref.size(); cmp(acc, yv, back(Y[b]), ref, bd, s.n, [&](int i, int) { return (bool)wr[i]; }); }
    char cs[128]; snprintf(cs, sizeof cs, "set%zu %c al=%g be=%g", si, up, al, be); report(vb ? "dsymv_vbatched" : "dsymv_batched", cs, acc);
    freeAll(A); freeAll(X); freeAll(Y); magma_free(dA); magma_free(dX); magma_free(dY); }
}
static void trsvPotrs() {
  for (int n : {1, 5, 31, 33, 64, 100, 257}) for (char up : {'L', 'U'}) for (char t : {'N', 'T'}) for (char dg : {'N', 'U'}) for (int inc : {1, 2}) {
    const int B = 3; Sd = 91u + n; std::vector<M> A, X;
    for (int b = 0; b < B; b++) { M a = mk(n, n, 1); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { double& v = a.h[i + (size_t)j * a.ld]; bool in = up == 'L' ? i >= j : i <= j;
          v = !in ? NAN : i == j ? (dg == 'U' ? NAN : 2.0 + 0.5 * r1()) : 0.3 * r1() / n; }
      magma_dsetvector((magma_int_t)a.h.size(), a.h.data(), 1, a.d, 1, Q); A.push_back(a); X.push_back(mk(1 + (n - 1) * inc, 1, 0)); }
    double **dA = parr(A), **dX = parr(X); magmablas_dtrsv_batched(up == 'L' ? MagmaLower : MagmaUpper, t == 'N' ? MagmaNoTrans : MagmaTrans, dg == 'N' ? MagmaNonUnit : MagmaUnit, n, dA, A[0].ld, dX, inc, B, Q); magma_queue_sync(Q);
    Acc acc; for (int b = 0; b < B; b++) { std::vector<double> T((size_t)n * n, 0);
      for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { bool in = up == 'L' ? i >= j : i <= j; double v = !in ? 0 : (i == j && dg == 'U') ? 1.0 : el(A[b], i, j); if (t == 'N') T[i + (size_t)j * n] = v; else T[j + (size_t)i * n] = v; }
      bool lowerT = (up == 'L') == (t == 'N'); std::vector<double> x(n), rhs(n); for (int i = 0; i < n; i++) rhs[i] = X[b].h[(size_t)i * inc];
      if (lowerT) for (int i = 0; i < n; i++) { double v = rhs[i]; for (int l = 0; l < i; l++) v -= T[i + (size_t)l * n] * x[l]; x[i] = v / T[i + (size_t)i * n]; }
      else for (int i = n - 1; i >= 0; i--) { double v = rhs[i]; for (int l = i + 1; l < n; l++) v -= T[i + (size_t)l * n] * x[l]; x[i] = v / T[i + (size_t)i * n]; }
      double mx = 0; for (double v : x) mx = std::max(mx, fabs(v)); std::vector<double> ref = X[b].h, bd(ref.size(), 1e-300); std::vector<bool> wr(ref.size(), false);
      for (int i = 0; i < n; i++) { ref[(size_t)i * inc] = x[i]; bd[(size_t)i * inc] = 64 * mx; wr[(size_t)i * inc] = true; }
      M xv = X[b]; xv.r = (int)ref.size(); xv.c = 1; xv.ld = (int)ref.size(); cmp(acc, xv, back(X[b]), ref, bd, n, [&](int i, int) { return (bool)wr[i]; }); }
    char cs[128]; snprintf(cs, sizeof cs, "n=%d %c%c%c inc=%d", n, up, t, dg, inc); report("dtrsv_batched", cs, acc);
    freeAll(A); freeAll(X); magma_free(dA); magma_free(dX); }
  // potrs_batched: A holds the Cholesky factor in the lower triangle, the strictly upper triangle holds the sentinel 7.0
  for (int n : {1, 5, 31, 33, 64, 100, 257, 513}) for (int nrhs : {1, 3}) {
    const int B = 3; Sd = 97u + n; std::vector<M> L, X;
    for (int b = 0; b < B; b++) { M a = mk(n, n, 1); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) a.h[i + (size_t)j * a.ld] = i < j ? 7.0 : i == j ? 1.5 + 0.5 * r1() : 0.3 * r1() / n;
      magma_dsetvector((magma_int_t)a.h.size(), a.h.data(), 1, a.d, 1, Q); L.push_back(a); X.push_back(mk(n, nrhs, 1)); }
    double **dL = parr(L), **dX = parr(X); magma_dpotrs_batched(MagmaLower, n, nrhs, dL, L[0].ld, dX, X[0].ld, B, Q); magma_queue_sync(Q);
    Acc acc; long long changedA = 0;
    for (int b = 0; b < B; b++) { auto g = back(L[b]); for (size_t q = 0; q < g.size(); q++) if (!(g[q] == L[b].h[q] || (std::isnan(g[q]) && std::isnan(L[b].h[q])))) changedA++;
      std::vector<double> ref = X[b].h, bd(ref.size(), 1e-300);
      for (int c = 0; c < nrhs; c++) { std::vector<double> y(n), x(n); for (int i = 0; i < n; i++) { double v = el(X[b], i, c); for (int l = 0; l < i; l++) v -= el(L[b], i, l) * y[l]; y[i] = v / el(L[b], i, i); }
        for (int i = n - 1; i >= 0; i--) { double v = y[i]; for (int l = i + 1; l < n; l++) v -= el(L[b], l, i) * x[l]; x[i] = v / el(L[b], i, i); }
        double mx = 0; for (double v : x) mx = std::max(mx, fabs(v)); for (int i = 0; i < n; i++) { ref[i + (size_t)c * X[b].ld] = x[i]; bd[i + (size_t)c * X[b].ld] = 64 * mx; } }
      cmp(acc, X[b], back(X[b]), ref, bd, n, [](int, int) { return true; }); }
    if (changedA) { acc.touched += (int)changedA; if (acc.where.empty()) acc.where = "A changed"; }
    char cs[128]; snprintf(cs, sizeof cs, "n=%d nrhs=%d", n, nrhs); report("dpotrs_batched", cs, acc);
    freeAll(L); freeAll(X); magma_free(dL); magma_free(dX); }
}

// symm debug: one 4x3 side-L lower case, batched vs non-batched magmablas_dsymm vs host
static void symmdbg() {
  int m = 4, n = 3; for (int nanfill = 0; nanfill < 2; nanfill++) { Sd = 5u; M a = mk(m, m, 0), b = mk(m, n, 0), c = mk(m, n, 0), c2 = mk(m, n, 0);
    for (int j = 0; j < m; j++) for (int i = 0; i < j; i++) a.h[i + (size_t)j * a.ld] = nanfill ? NAN : 100.0;   // strictly upper: not referenced
    magma_dsetvector((magma_int_t)a.h.size(), a.h.data(), 1, a.d, 1, Q); c2.h = c.h; magma_dsetvector((magma_int_t)c2.h.size(), c2.h.data(), 1, c2.d, 1, Q);
    std::vector<M> A{a}, Bv{b}, C{c}; double **dA = parr(A), **dB = parr(Bv), **dC = parr(C);
    magmablas_dsymm_batched(MagmaLeft, MagmaLower, m, n, 1.0, dA, a.ld, dB, b.ld, 0.5, dC, c.ld, 1, Q);
    magmablas_dsymm(MagmaLeft, MagmaLower, m, n, 1.0, a.d, a.ld, b.d, b.ld, 0.5, c2.d, c2.ld, Q); magma_queue_sync(Q);
    auto g = back(c), g2 = back(c2);
    std::vector<double> ob = c.h; { magma_int_t mm = m, nn = n, la = a.ld, lb = b.ld, lc = c.ld; double al = 1.0, be = 0.5; blasf77_dsymm("L", "L", &mm, &nn, &al, a.h.data(), &la, b.h.data(), &lb, &be, ob.data(), &lc); }
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { double r = 0; for (int l = 0; l < m; l++) r += symel(a, 'L', i, l) * el(b, l, j); r += 0.5 * el(c, i, j);
      double rfull = 0; for (int l = 0; l < m; l++) rfull += el(a, i, l) * el(b, l, j); rfull += 0.5 * el(c, i, j);
      printf("symmdbg upper=%s (%d,%d) batched=%.6f nonbatched=%.6f ref=%.6f ref_if_full_A_read=%.6f openblas=%.6f\n", nanfill ? "NaN" : "100", i, j, g[i + (size_t)j * c.ld], g2[i + (size_t)j * c2.ld], r, rfull, ob[i + (size_t)j * c.ld]); }
    magma_free(a.d); magma_free(b.d); magma_free(c.d); magma_free(c2.d); magma_free(dA); magma_free(dB); magma_free(dC); }
}

// symm structure probe: A lower-stored with chosen pattern, B(i,j) = 10*i + j, C = 0, alpha 1, beta 0 -> print C
static void symmstruct() {
  int m = 4, n = 2; const char* nm[] = {"A=I", "A=diag(1,2,3,4)", "A(2,0)=A(0,2)=1 only", "A(3,1)=1 only"};
  for (int pat = 0; pat < 4; pat++) for (int nb = 0; nb < 2; nb++) {
    M a = mk(m, m, 0, 0.0, false), b = mk(m, n, 0, 0.0, false), c = mk(m, n, 0, 5.0, false); cudaGetLastError();
    for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) { double v = 0; if (pat == 0 && i == j) v = 1; if (pat == 1 && i == j) v = i + 1; if (pat == 2 && i == 2 && j == 0) v = 1; if (pat == 3 && i == 3 && j == 1) v = 1; if (i < j) v = -999; a.h[i + (size_t)j * m] = v; }
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) b.h[i + (size_t)j * m] = 10 * i + j;
    magma_dsetvector(m * m, a.h.data(), 1, a.d, 1, Q); magma_dsetvector(m * n, b.h.data(), 1, b.d, 1, Q);
    if (nb) { std::vector<M> A{a}, Bv{b}, C{c}; double **dA = parr(A), **dB = parr(Bv), **dC = parr(C); magmablas_dsymm_batched(MagmaLeft, MagmaLower, m, n, 1.0, dA, m, dB, m, 0.0, dC, m, 1, Q); magma_free(dA); magma_free(dB); magma_free(dC); }
    else magmablas_dsymm(MagmaLeft, MagmaLower, m, n, 1.0, a.d, m, b.d, m, 0.0, c.d, m, Q);
    cudaError_t le = cudaGetLastError(); magma_queue_sync(Q); auto g = back(c); printf("symmstruct %-22s %-10s err=%s C(init 5) =", nm[pat], nb ? "batched" : "nonbatched", cudaGetErrorString(le));
    for (int i = 0; i < m; i++) { printf(" ["); for (int j = 0; j < n; j++) printf(" %g", g[i + (size_t)j * m]); printf(" ]"); } printf("\n");
    magma_free(a.d); magma_free(b.d); magma_free(c.d); }
}

// symm with A = I (lower stored, upper = NaN or 0): C must equal B.  Sizes across the tile sizes; non-batched and batched.
static void symmeye() {
  for (int up = 0; up < 2; up++) for (int sd = 0; sd < 2; sd++) for (int m : {1, 2, 4, 8, 16, 31, 32, 33, 64, 65, 128}) for (int nb = 0; nb < 2; nb++) for (int upfill = 0; upfill < 2; upfill++) {
    int n = 7, na = sd ? n : m; Sd = 3u + m; M a = mk(na, na, 0, 0.0, false), b = mk(m, n, 0), c = mk(m, n, 0, 0.0, false);
    for (int j = 0; j < na; j++) for (int i = 0; i < na; i++) { bool in = up ? i <= j : i >= j; a.h[i + (size_t)j * na] = i == j ? 1.0 : in ? 0.0 : (upfill ? NAN : 0.0); }
    magma_dsetvector(na * na, a.h.data(), 1, a.d, 1, Q);
    magma_side_t S = sd ? MagmaRight : MagmaLeft; magma_uplo_t U = up ? MagmaUpper : MagmaLower;
    if (nb) { std::vector<M> A{a}, Bv{b}, C{c}; double **dA = parr(A), **dB = parr(Bv), **dC = parr(C); magmablas_dsymm_batched(S, U, m, n, 1.0, dA, na, dB, m, 0.0, dC, m, 1, Q); magma_queue_sync(Q); magma_free(dA); magma_free(dB); magma_free(dC); }
    else { magmablas_dsymm(S, U, m, n, 1.0, a.d, na, b.d, m, 0.0, c.d, m, Q); magma_queue_sync(Q); }
    auto g = back(c); double e = 0; int nan = 0; for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { double d = g[i + (size_t)j * m] - b.h[i + (size_t)j * m]; if (std::isnan(d)) nan++; else e = std::max(e, fabs(d)); }
    printf("symmeye,%s,%s,%s,m=%d,nonref=%s,maxdiff=%g,nan=%d,%s\n", nb ? "batched" : "nonbatched", sd ? "R" : "L", up ? "U" : "L", m, upfill ? "NaN" : "0", e, nan, (e == 0 && nan == 0) ? "OK" : "BAD");
    magma_free(a.d); magma_free(b.d); magma_free(c.d); }
}

// symm/hemm in all precisions vs cuBLAS: A = I (lower), B random, beta = 0 -> C must equal B
static void symmprec() {
  for (int m : {4, 33, 64}) { int n = 5;
    std::vector<double> Bh((size_t)m * n); Sd = 9u; for (auto& x : Bh) x = r1();
    auto chk = [&](const char* nm, std::vector<double> g) { double e = 0; for (size_t i = 0; i < Bh.size(); i++) e = std::max(e, fabs(g[i] - Bh[i])); printf("symmprec,%s,m=%d,maxdiff=%g,%s\n", nm, m, e, e < 1e-5 ? "OK" : "BAD"); };
    { std::vector<double> Ah((size_t)m * m, 0); for (int i = 0; i < m; i++) Ah[i + (size_t)i * m] = 1; double *a, *b, *c; magma_dmalloc(&a, m * m); magma_dmalloc(&b, m * n); magma_dmalloc(&c, m * n);
      magma_dsetmatrix(m, m, Ah.data(), m, a, m, Q); magma_dsetmatrix(m, n, Bh.data(), m, b, m, Q); std::vector<double> g((size_t)m * n);
      magmablas_dsymm(MagmaLeft, MagmaLower, m, n, 1.0, a, m, b, m, 0.0, c, m, Q); magma_dgetmatrix(m, n, c, m, g.data(), m, Q); chk("magmablas_dsymm", g);
      magma_dsymm(MagmaLeft, MagmaLower, m, n, 1.0, a, m, b, m, 0.0, c, m, Q); magma_dgetmatrix(m, n, c, m, g.data(), m, Q); chk("magma_dsymm(cublas)", g);
      magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, m, n, m, 1.0, a, m, b, m, 0.0, c, m, Q); magma_dgetmatrix(m, n, c, m, g.data(), m, Q); chk("magmablas_dgemm", g);
      magma_free(a); magma_free(b); magma_free(c); }
    { std::vector<float> Ah((size_t)m * m, 0), Bf(Bh.begin(), Bh.end()), g((size_t)m * n); for (int i = 0; i < m; i++) Ah[i + (size_t)i * m] = 1; float *a, *b, *c; magma_smalloc(&a, m * m); magma_smalloc(&b, m * n); magma_smalloc(&c, m * n);
      magma_ssetmatrix(m, m, Ah.data(), m, a, m, Q); magma_ssetmatrix(m, n, Bf.data(), m, b, m, Q);
      magmablas_ssymm(MagmaLeft, MagmaLower, m, n, 1.0f, a, m, b, m, 0.0f, c, m, Q); magma_sgetmatrix(m, n, c, m, g.data(), m, Q); chk("magmablas_ssymm", std::vector<double>(g.begin(), g.end()));
      magma_free(a); magma_free(b); magma_free(c); }
    { std::vector<magmaDoubleComplex> Ah((size_t)m * m, MAGMA_Z_ZERO), Bz((size_t)m * n), g((size_t)m * n); for (int i = 0; i < m; i++) Ah[i + (size_t)i * m] = MAGMA_Z_ONE; for (size_t i = 0; i < Bz.size(); i++) Bz[i] = MAGMA_Z_MAKE(Bh[i], 0);
      magmaDoubleComplex *a, *b, *c; magma_zmalloc(&a, m * m); magma_zmalloc(&b, m * n); magma_zmalloc(&c, m * n); magma_zsetmatrix(m, m, Ah.data(), m, a, m, Q); magma_zsetmatrix(m, n, Bz.data(), m, b, m, Q);
      magmablas_zhemm(MagmaLeft, MagmaLower, m, n, MAGMA_Z_ONE, a, m, b, m, MAGMA_Z_ZERO, c, m, Q); magma_zgetmatrix(m, n, c, m, g.data(), m, Q);
      std::vector<double> gr(g.size()); for (size_t i = 0; i < g.size(); i++) gr[i] = MAGMA_Z_REAL(g[i]); chk("magmablas_zhemm", gr);
      magmablas_zsymm(MagmaLeft, MagmaLower, m, n, MAGMA_Z_ONE, a, m, b, m, MAGMA_Z_ZERO, c, m, Q); magma_zgetmatrix(m, n, c, m, g.data(), m, Q);
      for (size_t i = 0; i < g.size(); i++) gr[i] = MAGMA_Z_REAL(g[i]); chk("magmablas_zsymm", gr);
      magma_free(a); magma_free(b); magma_free(c); }
  }
}

// ---------------------------------------------------------------- contracts of callers
// potrf_batched / potrf_vbatched (lower): "the strictly upper triangular part of A is not referenced" -> must be unchanged;
// magmablas_dtrsm / magmablas_dtrmm (non-batched) with alpha = 0 and NaN in B -> B must become 0.
static void contract(const char* only = nullptr) {
  std::vector<int> ns = {5, 16, 31, 33, 64, 100, 257, 513};
  if (only) { ns.clear(); for (const char* p = only; *p; ) { ns.push_back(atoi(p)); while (*p && *p != ':') p++; if (*p) p++; } }
  for (int vb = 0; vb < 2; vb++) for (int n : ns) {
    const int B = 3; Sd = 61u + n; std::vector<M> A; std::vector<int> vn, la;
    for (int b = 0; b < B; b++) { M a = mk(n, n, 1); std::vector<double> G(a.h); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { double s = 0; for (int k = 0; k < n; k++) s += G[i + (size_t)k * a.ld] * G[j + (size_t)k * a.ld]; a.h[i + (size_t)j * a.ld] = i >= j ? s + (i == j ? n : 0) : 7.0; }
      magma_dsetvector((magma_int_t)a.h.size(), a.h.data(), 1, a.d, 1, Q); A.push_back(a); vn.push_back(n); la.push_back(a.ld); }
    double** dA = parr(A); magma_int_t* dinfo; magma_imalloc(&dinfo, B + 1);
    if (getenv("R34_INFO")) { std::vector<magma_int_t> pre(B + 1, (magma_int_t)atoi(getenv("R34_INFO"))); magma_isetvector(B + 1, pre.data(), 1, dinfo, 1, Q); }   // prefill the output info
    if (vb) { magma_int_t *dn = iarr(vn), *dl = iarr(la); magma_dpotrf_vbatched(MagmaLower, dn, dA, dl, dinfo, B, Q); magma_queue_sync(Q); magma_free(dn); magma_free(dl); }
    else { magma_dpotrf_batched(MagmaLower, n, dA, la[0], dinfo, B, Q); magma_queue_sync(Q); }
    std::vector<magma_int_t> info(B); magma_igetvector(B, dinfo, 1, info.data(), 1, Q);
    long long changed = 0, inf = 0; double ferr = 0;
    for (int b = 0; b < B; b++) { auto g = back(A[b]); if (info[b]) inf = info[b];
      for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) if (g[i + (size_t)j * A[b].ld] != 7.0) changed++;
      for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { double l = 0; for (int k = 0; k <= j; k++) l += g[i + (size_t)k * A[b].ld] * g[j + (size_t)k * A[b].ld]; ferr = std::max(ferr, fabs(l - A[b].h[i + (size_t)j * A[b].ld]) / (n * n * EPS * 4 * n)); } }
    printf("contract,%s,n=%d,%s,info=%lld upper_changed=%lld of %lld; factor_err=%.3g\n", vb ? "dpotrf_vbatched" : "dpotrf_batched", n, changed ? "TOUCHED" : (ferr > 1 ? "WRONG" : "OK"), inf, changed, (long long)B * n * (n - 1) / 2, ferr);
    freeAll(A); magma_free(dA); magma_free(dinfo); }
  if (only) return;
  for (int tm = 0; tm < 2; tm++) for (int n : {1, 31, 100}) {   // non-batched trsm / trmm, alpha = 0, B full of NaN
    M a = mk(n, n, 0); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) a.h[i + (size_t)j * a.ld] = i == j ? 2.0 : (i > j ? 0.01 : NAN);
    magma_dsetvector((magma_int_t)a.h.size(), a.h.data(), 1, a.d, 1, Q); M x = mk(n, 7, 0, NAN, false);
    if (tm == 0) magmablas_dtrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, n, 7, 0.0, a.d, a.ld, x.d, x.ld, Q);
    else magma_dtrmm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, n, 7, 0.0, a.d, a.ld, x.d, x.ld, Q);
    magma_queue_sync(Q); auto g = back(x); long long nn = 0; for (double v : g) if (!(v == 0)) nn++;
    printf("contract,%s,n=%d,%s,alpha=0 B=NaN: %lld of %lld entries not zero\n", tm == 0 ? "magmablas_dtrsm" : "magma_dtrmm(cublas)", n, nn ? "NAN" : "OK", nn, (long long)g.size());
    magma_free(a.d); magma_free(x.d); }
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string g = argc > 1 ? argv[1] : "all";
  magma_init(); magma_queue_create(0, &Q);
  printf("bblas,routine,case,status,maxerr,detail\n");
  if (g == "symm") { symm(false); symm(true); }
  if (g == "symv") { symv(false); symv(true); }
  if (g == "trsv") trsvPotrs();
  if (g == "symmdbg") symmdbg();
  if (g == "symmstruct") symmstruct();
  if (g == "symmeye") symmeye();
  if (g == "symmprec") symmprec();
  if (g == "contract") contract();
  if (g == "potrf") contract(argc > 2 ? argv[2] : "300:384:400:448:480:500:511:512:513:520:600:1024");
  if (g == "all" || g == "gemm") { gemm(false); gemm(true); }
  if (g == "all" || g == "gemv") { gemv(false); gemv(true); }
  if (g == "all" || g == "syrk") { syrk(false, false); syrk(true, false); syrk(false, true); }
  if (g == "all" || g == "trsm") { trxm(false, true); trxm(true, true); }
  if (g == "all" || g == "trmm") { trxm(false, false); trxm(true, false); }
  fprintf(stderr, "DONE %s cuda=%s\n", g.c_str(), cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

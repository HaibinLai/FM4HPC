// Thirty-first round (roxy, RTX 5070 Ti sm_120): differential sweep of the mixed-precision iterative-refinement
// solvers on matrices with prescribed condition numbers, scalings outside the FP16 / FP32 ranges, and a NaN in b.
//     round31_mixed <group>     group = ge (general) | po (SPD) | z (complex zcgesv/zcposv) | ls (dsgeqrsv)
// Every result with info = 0 must have a double-precision backward error: when IR converges its own test bounds the
// residual by sqrt(n)*eps*||A||*||x||, and when it falls back the double factorization gives LAPACK quality.
// CSV: solver,kind,cond,scale,n,nrhs,iter,info,berr_ratio,ferr_magma,ferr_lapack,verdict
//   berr_ratio = max_j ||b_j - A x_j||_inf / ((||A||_inf ||x_j||_inf + ||b_j||_inf) * n * eps)
//   verdict  OK (info 0, berr_ratio <= 100) | WRONG (info 0, berr_ratio > 100 or x non-finite on finite input)
//            INFO (info != 0) | NAN-ACCEPTED (b has NaN, info 0, iter >= 0)
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
typedef std::complex<double> cd;
static magma_queue_t Q;
static const double EPS = 1.1102230246251565e-16;
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static std::vector<double> orth(int n) { std::vector<double> X((size_t)n * n); for (auto& x : X) x = r1(); magma_int_t nn = n, lw = 64 * n, il; std::vector<double> tau(n), w(lw);
  lapackf77_dgeqrf(&nn, &nn, X.data(), &nn, tau.data(), w.data(), &lw, &il); lapackf77_dorgqr(&nn, &nn, &nn, X.data(), &nn, tau.data(), w.data(), &lw, &il); return X; }
// A = U diag(s) V^T with s geometric from 1 to 1/cond (spd: V = U)
static std::vector<double> gen(int n, double cond, bool spd, unsigned seed) { Sd = seed; auto U = orth(n), V = spd ? U : orth(n); std::vector<double> A((size_t)n * n);
  for (int l = 0; l < n; l++) { double s = n > 1 ? pow(1.0 / cond, (double)l / (n - 1)) : 1.0; for (int j = 0; j < n; j++) { double v = s * V[j + (size_t)l * n]; for (int i = 0; i < n; i++) A[i + (size_t)j * n] += U[i + (size_t)l * n] * v; } }
  if (spd) for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) { double a = 0.5 * (A[i + (size_t)j * n] + A[j + (size_t)i * n]); A[i + (size_t)j * n] = a; A[j + (size_t)i * n] = a; }
  return A; }
static double ninf(const std::vector<double>& A, int m, int n) { double mx = 0; for (int i = 0; i < m; i++) { double s = 0; for (int j = 0; j < n; j++) s += fabs(A[i + (size_t)j * m]); mx = std::max(mx, s); } return mx; }
static bool fin(const std::vector<double>& v) { for (double a : v) if (!std::isfinite(a)) return false; return true; }
static double berr(const std::vector<double>& A, const std::vector<double>& X, const std::vector<double>& B, int n, int nrhs) { double an = ninf(A, n, n), worst = 0;
  for (int c = 0; c < nrhs; c++) { double rn = 0, xn = 0, bn = 0; for (int i = 0; i < n; i++) { double s = B[i + (size_t)c * n]; for (int j = 0; j < n; j++) s -= A[i + (size_t)j * n] * X[j + (size_t)c * n];
      rn = std::max(rn, fabs(s)); xn = std::max(xn, fabs(X[i + (size_t)c * n])); bn = std::max(bn, fabs(B[i + (size_t)c * n])); }
    double d = (an * xn + bn) * n * EPS; worst = std::max(worst, d > 0 ? rn / d : (rn == 0 ? 0 : INFINITY)); }
  return worst; }
static double ferr(const std::vector<double>& X, const std::vector<double>& Xt) { double e = 0, m = 0; for (size_t i = 0; i < X.size(); i++) { e = std::max(e, fabs(X[i] - Xt[i])); m = std::max(m, fabs(Xt[i])); } return m > 0 ? e / m : e; }
static void out(const char* s, const char* kind, double cond, double scale, int n, int nrhs, long long iter, long long info, double br, double fm, double fl, bool finIn, bool finX) {
  const char* v = "OK"; if (!finIn) v = (info == 0 && iter >= 0) ? "NAN-ACCEPTED" : (info == 0 ? "NAN-FALLBACK-OK" : "INFO");
  else if (info != 0) v = "INFO"; else if (!finX || !(br <= 100)) v = "WRONG";
  printf("%s,%s,%.0e,%.3g,%d,%d,%lld,%lld,%.3g,%.3g,%.3g,%s\n", s, kind, cond, scale, n, nrhs, iter, info, br, fm, fl, v); }

typedef magma_int_t (*GeFn)(magma_trans_t, magma_int_t, magma_int_t, magmaDouble_ptr, magma_int_t, magma_int_t*, magmaInt_ptr, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magmaFloat_ptr, magma_int_t*, magma_int_t*);
struct Case { const char* kind; std::vector<double> A, B, Xt; double cond, scale; };
static std::vector<Case> cases(int n, int nrhs, bool spd) {
  std::vector<Case> C;
  for (double cond : {1e1, 1e4, 1e8, 1e12, 1e15}) for (double sc : {1.0, ldexp(1.0, -30), ldexp(1.0, 20), ldexp(1.0, 100)}) {
    if (cond > 1e4 && sc != 1.0) continue;
    Case c{"cond", gen(n, cond, spd, 100u + n + (unsigned)log10(cond)), {}, {}, cond, sc}; for (auto& a : c.A) a *= sc; Sd = 7u + n;
    c.Xt.resize((size_t)n * nrhs); for (auto& x : c.Xt) x = r1(); c.B.assign((size_t)n * nrhs, 0);
    for (int k = 0; k < nrhs; k++) for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) c.B[i + (size_t)k * n] += c.A[i + (size_t)j * n] * c.Xt[j + (size_t)k * n];
    C.push_back(c); }
  { Case c = C[0]; c.kind = "nan_in_b"; c.B[0] = NAN; C.push_back(c); }
  return C; }

static void runGe(const char* nm, GeFn f, int n, int nrhs, const Case& c, int dxmode = -1) {
  int ld = (int)magma_roundup(n, 32); magmaDouble_ptr dA, dB, dX, dwd; magmaFloat_ptr dws; magma_int_t* dipiv; std::vector<magma_int_t> ip(n);
  magma_dmalloc(&dA, (size_t)ld * n); magma_dmalloc(&dB, (size_t)ld * nrhs); magma_dmalloc(&dX, (size_t)ld * nrhs); magma_dmalloc(&dwd, (size_t)n * nrhs + 64);
  magma_smalloc(&dws, (size_t)2 * n * (n + nrhs) + 4096); magma_imalloc(&dipiv, n);
  magma_dsetmatrix(n, n, c.A.data(), n, dA, ld, Q); magma_dsetmatrix(n, nrhs, c.B.data(), n, dB, ld, Q);
  magma_int_t iter = -999, info = -999; real_Double_t ft = 0;
  if (dxmode < 0) f(MagmaNoTrans, n, nrhs, dA, ld, ip.data(), dipiv, dB, ld, dX, ld, dwd, dws, &iter, &info);
  else { magma_refinement_t fa[] = {Magma_PREC_SS, Magma_PREC_SHT, Magma_PREC_HS, Magma_PREC_XHS_H, Magma_PREC_XHS_HTC, Magma_PREC_XSH_STC};
    magma_refinement_t so[] = {Magma_REFINE_IRSTRS, Magma_REFINE_IRGMSTRS, Magma_REFINE_IRGMDTRS};
    magma_dxgesv_gmres_gpu(MagmaNoTrans, n, nrhs, dA, ld, ip.data(), dipiv, dB, ld, dX, ld, dwd, dws, fa[dxmode / 3], so[dxmode % 3], &iter, &info, &ft); }
  std::vector<double> X((size_t)n * nrhs); magma_dgetmatrix(n, nrhs, dX, ld, X.data(), n, Q);
  auto R = c.A; auto Xl = c.B; magma_int_t nn = n, nr = nrhs, il; std::vector<magma_int_t> ipl(n); lapackf77_dgesv(&nn, &nr, R.data(), &nn, ipl.data(), Xl.data(), &nn, &il);
  bool fi = fin(c.B); std::string name = nm; if (dxmode >= 0) { const char* fn[] = {"SS", "SHT", "HS", "XHS_H", "XHS_HTC", "XSH_STC"}; const char* sn[] = {"IRSTRS", "IRGMSTRS", "IRGMDTRS"}; name += std::string("(") + fn[dxmode / 3] + "," + sn[dxmode % 3] + ")"; }
  out(name.c_str(), c.kind, c.cond, c.scale, n, nrhs, iter, info, fi ? berr(c.A, X, c.B, n, nrhs) : NAN, ferr(X, c.Xt), ferr(Xl, c.Xt), fi, fin(X));
  magma_free(dA); magma_free(dB); magma_free(dX); magma_free(dwd); magma_free(dws); magma_free(dipiv);
}

// Bisection filters (round 31 follow-up): R31_N=<n> keeps one size; R31_RUN=<letters> keeps solvers s=dsgesv i=dsgesv_iteref
// h=dhgesv_iteref x=dxgesv_gmres (default "sihx").
static bool keepN(int n) { const char* e = getenv("R31_N"); return !e || atoi(e) == n; }
static bool keepS(char k) { const char* e = getenv("R31_RUN"); return !e || strchr(e, k); }
static void groupGe() {
  for (int n : {64, 200, 513}) if (keepN(n)) for (int nrhs : {1, 2}) for (auto& c : cases(n, nrhs, false)) {
    if (keepS('s')) runGe("dsgesv_gpu", magma_dsgesv_gpu, n, nrhs, c);
    if (keepS('i')) runGe("dsgesv_iteref_gpu", magma_dsgesv_iteref_gpu, n, nrhs, c);
    if (keepS('h')) runGe("dhgesv_iteref_gpu", magma_dhgesv_iteref_gpu, n, nrhs, c);
    if (keepS('x')) for (int m = 0; m < 18; m++) runGe("dxgesv_gmres_gpu", NULL, n, nrhs, c, m); }
}
static void groupPo() {
  for (int n : {64, 200, 513}) for (int nrhs : {1, 2}) for (auto& c : cases(n, nrhs, true)) for (int v = 0; v < 3; v++) {
    int ld = (int)magma_roundup(n, 32); magmaDouble_ptr dA, dB, dX, dwd; magmaFloat_ptr dws; magma_dmalloc(&dA, (size_t)ld * n); magma_dmalloc(&dB, (size_t)ld * nrhs); magma_dmalloc(&dX, (size_t)ld * nrhs);
    magma_dmalloc(&dwd, (size_t)n * nrhs + 64); magma_smalloc(&dws, (size_t)n * (n + nrhs) + 4096);
    magma_dsetmatrix(n, n, c.A.data(), n, dA, ld, Q); magma_dsetmatrix(n, nrhs, c.B.data(), n, dB, ld, Q); magma_int_t iter = -999, info = -999;
    const char* nm[] = {"dsposv_gpu", "dshposv_gpu", "dshposv_native"};
    if (v == 0) magma_dsposv_gpu(MagmaLower, n, nrhs, dA, ld, dB, ld, dX, ld, dwd, dws, &iter, &info);
    else if (v == 1) magma_dshposv_gpu(MagmaLower, n, nrhs, dA, ld, dB, ld, dX, ld, &iter, &info);
    else magma_dshposv_native(MagmaLower, n, nrhs, dA, ld, dB, ld, dX, ld, &iter, &info);
    std::vector<double> X((size_t)n * nrhs); magma_dgetmatrix(n, nrhs, dX, ld, X.data(), n, Q);
    auto R = c.A; auto Xl = c.B; magma_int_t nn = n, nr = nrhs, il; lapackf77_dposv("L", &nn, &nr, R.data(), &nn, Xl.data(), &nn, &il);
    bool fi = fin(c.B); out(nm[v], c.kind, c.cond, c.scale, n, nrhs, iter, info, fi ? berr(c.A, X, c.B, n, nrhs) : NAN, ferr(X, c.Xt), ferr(Xl, c.Xt), fi, fin(X));
    magma_free(dA); magma_free(dB); magma_free(dX); magma_free(dwd); magma_free(dws); }
}
static void groupLs() {
  for (int n : {64, 200, 513}) for (int nrhs : {1, 2}) for (auto& c : cases(n, nrhs, false)) {
    int ld = (int)magma_roundup(n, 32); magmaDouble_ptr dA, dB, dX; magma_dmalloc(&dA, (size_t)ld * n); magma_dmalloc(&dB, (size_t)ld * nrhs); magma_dmalloc(&dX, (size_t)ld * nrhs);
    magma_dsetmatrix(n, n, c.A.data(), n, dA, ld, Q); magma_dsetmatrix(n, nrhs, c.B.data(), n, dB, ld, Q); magma_int_t iter = -999, info = -999;
    magma_dsgeqrsv_gpu(n, n, nrhs, dA, ld, dB, ld, dX, ld, &iter, &info);
    std::vector<double> X((size_t)n * nrhs); magma_dgetmatrix(n, nrhs, dX, ld, X.data(), n, Q);
    auto R = c.A; auto Xl = c.B; magma_int_t nn = n, nr = nrhs, il; std::vector<magma_int_t> ipl(n); lapackf77_dgesv(&nn, &nr, R.data(), &nn, ipl.data(), Xl.data(), &nn, &il);
    bool fi = fin(c.B); out("dsgeqrsv_gpu", c.kind, c.cond, c.scale, n, nrhs, iter, info, fi ? berr(c.A, X, c.B, n, nrhs) : NAN, ferr(X, c.Xt), ferr(Xl, c.Xt), fi, fin(X));
    magma_free(dA); magma_free(dB); magma_free(dX); }
}
// complex zcgesv / zcposv: real-valued test matrices embedded in complex arithmetic plus a complex shift of x
static void groupZ() {
  for (int n : {64, 200, 513}) for (int nrhs : {1, 2}) for (bool spd : {false, true}) for (auto& c : cases(n, nrhs, spd)) {
    int ld = (int)magma_roundup(n, 32); std::vector<cd> A(c.A.begin(), c.A.end()), Xt((size_t)n * nrhs), B((size_t)n * nrhs, 0);
    for (size_t i = 0; i < Xt.size(); i++) Xt[i] = cd(c.Xt[i], 0.5 * c.Xt[(i * 7) % Xt.size()]);
    for (int k = 0; k < nrhs; k++) for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) B[i + (size_t)k * n] += A[i + (size_t)j * n] * Xt[j + (size_t)k * n];
    if (std::string(c.kind) == "nan_in_b") B[0] = cd(NAN, 0);
    magmaDoubleComplex_ptr dA, dB, dX, dwd; magmaFloatComplex_ptr dws; magma_int_t* dipiv; std::vector<magma_int_t> ip(n);
    magma_zmalloc(&dA, (size_t)ld * n); magma_zmalloc(&dB, (size_t)ld * nrhs); magma_zmalloc(&dX, (size_t)ld * nrhs); magma_zmalloc(&dwd, (size_t)n * nrhs + 64); magma_cmalloc(&dws, (size_t)n * (n + nrhs) + 4096); magma_imalloc(&dipiv, n);
    magma_zsetmatrix(n, n, (magmaDoubleComplex*)A.data(), n, dA, ld, Q); magma_zsetmatrix(n, nrhs, (magmaDoubleComplex*)B.data(), n, dB, ld, Q); magma_int_t iter = -999, info = -999;
    if (spd) magma_zcposv_gpu(MagmaLower, n, nrhs, dA, ld, dB, ld, dX, ld, dwd, dws, &iter, &info); else magma_zcgesv_gpu(MagmaNoTrans, n, nrhs, dA, ld, ip.data(), dipiv, dB, ld, dX, ld, dwd, dws, &iter, &info);
    std::vector<cd> X((size_t)n * nrhs); magma_zgetmatrix(n, nrhs, dX, ld, (magmaDoubleComplex*)X.data(), n, Q);
    double an = ninf(c.A, n, n), worst = 0; bool fx = true, fb = true; for (auto& b : B) if (!std::isfinite(b.real())) fb = false;
    for (int k = 0; k < nrhs; k++) { double rn = 0, xn = 0, bn = 0; for (int i = 0; i < n; i++) { cd s = B[i + (size_t)k * n]; for (int j = 0; j < n; j++) s -= A[i + (size_t)j * n] * X[j + (size_t)k * n];
        rn = std::max(rn, std::abs(s)); xn = std::max(xn, std::abs(X[i + (size_t)k * n])); bn = std::max(bn, std::abs(B[i + (size_t)k * n])); if (!std::isfinite(X[i + (size_t)k * n].real())) fx = false; }
      worst = std::max(worst, rn / ((an * xn + bn) * n * EPS)); }
    out(spd ? "zcposv_gpu" : "zcgesv_gpu", c.kind, c.cond, c.scale, n, nrhs, iter, info, fb ? worst : NAN, 0, 0, fb, fx);
    magma_free(dA); magma_free(dB); magma_free(dX); magma_free(dwd); magma_free(dws); magma_free(dipiv); }
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string g = argc > 1 ? argv[1] : ""; magma_init(); magma_queue_create(0, &Q);
  printf("solver,kind,cond,scale,n,nrhs,iter,info,berr_ratio,ferr_magma,ferr_lapack,verdict\n");
  if (g == "ge") groupGe(); else if (g == "po") groupPo(); else if (g == "ls") groupLs(); else if (g == "z") groupZ(); else printf("unknown\n");
  fprintf(stderr, "DONE %s cuda=%s\n", g.c_str(), cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

// Round 80 (roxy): the recent sparse findings re-checked in other precisions (they were found with z).
//   round80_prec dshift <n>            M279 in d: magma_d_spmv_shift (CSR, ELL, ELLPACKT), alpha = 0.7: err against alpha (A - lambda I) x + beta y, err2 against alpha A x - lambda x + beta y
//   round80_prec dslice <n> <ns> <sl>  M280 in d: magma_dmslice, info / start / end
//   round80_prec dicc <g>              M284 in d: ICC on the 2-D Laplacian, bilinear identity u^T M^{-1} v = (M^{-T} u)^T v (left(T)(right(T) u)); also PBiCG + ICC
//   round80_prec cidr <rot0|rot90> <s> M282 in c: magma_cidr on e^{i theta} tridiag(-1, 2.1, -1), n = 200, rtol 1e-5
//   round80_prec cjac <g>              M285 in c: Jacobi on the complex stencil (diagonal 4.05 + 0.8i), Hermitian identity; LSQR + Jacobi and LSQR without preconditioner, rtol 1e-5
// CSV: r80,op,a,b,c,info,x1,x2,verdict
#include "magma_v2.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
typedef std::complex<float> cf;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static magma_queue_t Q;
template <class T> static T* cp(const std::vector<T>& v) { T* p = (T*)malloc(std::max<size_t>(v.size(), 1) * sizeof(T)); memcpy(p, v.data(), v.size() * sizeof(T)); return p; }
// ---- d helpers
static magma_d_matrix dcsr(int n, const std::vector<std::vector<std::pair<int, double>>>& R) { std::vector<magma_index_t> row(1, 0), col; std::vector<double> val;
  for (int i = 0; i < n; i++) { for (auto& p : R[i]) { col.push_back(p.first); val.push_back(p.second); } row.push_back((magma_index_t)col.size()); }
  magma_d_matrix A = {Magma_CSR}; magma_dcsrset(n, n, cp(row), cp(col), cp(val), &A, Q); return A; }
static magma_d_matrix ddev(const std::vector<double>& h) { magma_d_matrix x = {Magma_CSR}, dx = {Magma_CSR}; std::vector<double> t = h; magma_dvset((magma_int_t)h.size(), 1, t.data(), &x, Q); magma_dmtransfer(x, &dx, Magma_CPU, Magma_DEV, Q); return dx; }
static std::vector<double> dhost(magma_d_matrix dx) { magma_d_matrix x = {Magma_CSR}; magma_dmtransfer(dx, &x, Magma_DEV, Magma_CPU, Q); return std::vector<double>(x.val, x.val + x.num_rows); }
// ---- c helpers
static magma_c_matrix ccsr(int n, const std::vector<std::vector<std::pair<int, cf>>>& R) { std::vector<magma_index_t> row(1, 0), col; std::vector<magmaFloatComplex> val;
  for (int i = 0; i < n; i++) { for (auto& p : R[i]) { col.push_back(p.first); val.push_back(MAGMA_C_MAKE(p.second.real(), p.second.imag())); } row.push_back((magma_index_t)col.size()); }
  magma_c_matrix A = {Magma_CSR}; magma_ccsrset(n, n, cp(row), cp(col), cp(val), &A, Q); return A; }
static magma_c_matrix cdev(const std::vector<cf>& h) { std::vector<magmaFloatComplex> t(h.size()); for (size_t i = 0; i < h.size(); i++) t[i] = MAGMA_C_MAKE(h[i].real(), h[i].imag()); magma_c_matrix x = {Magma_CSR}, dx = {Magma_CSR};
  magma_cvset((magma_int_t)h.size(), 1, t.data(), &x, Q); magma_cmtransfer(x, &dx, Magma_CPU, Magma_DEV, Q); return dx; }
static std::vector<cf> chost(magma_c_matrix dx) { magma_c_matrix x = {Magma_CSR}; magma_cmtransfer(dx, &x, Magma_DEV, Magma_CPU, Q); std::vector<cf> h(x.num_rows); for (size_t i = 0; i < h.size(); i++) h[i] = cf(MAGMA_C_REAL(x.val[i]), MAGMA_C_IMAG(x.val[i])); return h; }
static std::vector<std::vector<std::pair<int, double>>> lap(int g) { int n = g * g; std::vector<std::vector<std::pair<int, double>>> R(n);
  for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) { int r = x + y * g; if (y) R[r].push_back({r - g, -1}); if (x) R[r].push_back({r - 1, -1}); R[r].push_back({r, 4.05}); if (x + 1 < g) R[r].push_back({r + 1, -1}); if (y + 1 < g) R[r].push_back({r + g, -1}); } return R; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
  if (argc < 3) return 1; std::string op = argv[1]; magma_init(); magma_queue_create(0, &Q);
  long long info = 0; double x1 = -1, x2 = -1; bool ok = false; std::string tag = std::string(argv[2]) + "," + (argc > 3 ? argv[3] : "-") + "," + (argc > 4 ? argv[4] : "-");
  if (op == "dshift") { int n = atoi(argv[2]); std::vector<std::vector<std::pair<int, double>>> R(n); std::vector<double> D((size_t)n * n, 0);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i == j || (r1() + 0.5) < 0.05) { double v = r1() + (i == j ? 3 : 0); R[i].push_back({j, v}); D[i + (size_t)j * n] = v; }
    magma_d_matrix A = dcsr(n, R); double worst = 0, worst2 = 0;
    for (magma_storage_t st : {Magma_CSR, Magma_ELL, Magma_ELLPACKT}) { magma_d_matrix B = {Magma_CSR}, dA = {Magma_CSR}; if (st == Magma_CSR) B = A; else magma_dmconvert(A, &B, Magma_CSR, st, Q); magma_dmtransfer(B, &dA, Magma_CPU, Magma_DEV, Q);
      std::vector<double> x(n), y(n); for (auto& v : x) v = r1(); for (auto& v : y) v = r1(); double al = 0.7, la = 0.4, be = -0.4; magma_d_matrix dx = ddev(x), dy = ddev(y);
      magma_int_t i1 = magma_d_spmv_shift(al, dA, la, dx, be, 0, n, NULL, dy, Q); if (i1) info = i1; magma_queue_sync(Q); std::vector<double> X = dhost(dy); double e = 0, e2 = 0, a = 0;
      for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += D[i + (size_t)j * n] * x[j]; double r1v = al * (s - la * x[i]) + be * y[i], r2v = al * s - la * x[i] + be * y[i]; e = std::max(e, fabs(X[i] - r1v)); e2 = std::max(e2, fabs(X[i] - r2v)); a = std::max(a, fabs(r1v)); }
      worst = std::max(worst, e / a); worst2 = std::max(worst2, e2 / a); }
    x1 = worst; x2 = worst2; ok = info == 0 && x1 < 1e-13; }
  else if (op == "dslice") { int n = atoi(argv[2]), ns = atoi(argv[3]), sl = atoi(argv[4]); std::vector<std::vector<std::pair<int, double>>> R(n); for (int i = 0; i < n; i++) { if (i) R[i].push_back({i - 1, -1}); R[i].push_back({i, 2}); if (i + 1 < n) R[i].push_back({i + 1, -1}); }
    magma_d_matrix A = dcsr(n, R), B = {Magma_CSR}, AL = {Magma_CSR}, AN = {Magma_CSR}; std::vector<magma_index_t> ci(n + 64); std::vector<double> cv(n + 64); magma_int_t st = -1, en = -1;
    info = magma_dmslice(ns, sl, A, &B, &AL, &AN, ci.data(), cv.data(), &st, &en, Q); x1 = (double)st; x2 = (double)en; int s = (n + ns - 1) / ns; int ls = std::min(sl * s, n), le = std::min((sl + 1) * s, n);
    ok = info == 0 && st == ls && en == le; }
  else if (op == "dicc") { int g = atoi(argv[2]), n = g * g; auto R = lap(g); magma_d_matrix A = dcsr(n, R), dA = {Magma_CSR}; magma_dmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q);
    std::vector<double> xt(n), bh(n, 0); for (auto& v : xt) v = r1(); for (int i = 0; i < n; i++) for (auto& p : R[i]) bh[i] += p.second * xt[p.first]; magma_d_matrix db = ddev(bh);
    magma_d_solver_par sp; memset(&sp, 0, sizeof(sp)); sp.solver = Magma_PBICG; sp.rtol = 1e-10; sp.maxiter = 2 * n; magma_d_preconditioner pp; memset(&pp, 0, sizeof(pp)); pp.solver = Magma_ICC;
    info = magma_d_precondsetup(dA, db, &sp, &pp, Q); std::vector<double> u(n), v(n); for (auto& a : u) a = r1(); for (auto& a : v) a = r1();
    magma_d_matrix du = ddev(u), dv = ddev(v), t1 = {Magma_CSR}, y = {Magma_CSR}, t2 = {Magma_CSR}, yt = {Magma_CSR}; for (auto* m : {&t1, &y, &t2, &yt}) magma_dvinit(m, Magma_DEV, n, 1, 0.0, Q);
    magma_d_applyprecond_left(MagmaNoTrans, dA, dv, &t1, &pp, Q); magma_d_applyprecond_right(MagmaNoTrans, dA, t1, &y, &pp, Q); magma_d_applyprecond_right(MagmaTrans, dA, du, &t2, &pp, Q); magma_d_applyprecond_left(MagmaTrans, dA, t2, &yt, &pp, Q);
    magma_queue_sync(Q); std::vector<double> Y = dhost(y), YT = dhost(yt); double l = 0, r = 0; for (int i = 0; i < n; i++) { l += u[i] * Y[i]; r += YT[i] * v[i]; } x1 = fabs(l - r) / fabs(l);
    magma_d_matrix dx = {Magma_CSR}; magma_dvinit(&dx, Magma_DEV, n, 1, 0.0, Q); magma_int_t si = magma_dpbicg(dA, db, &dx, &sp, &pp, Q); std::vector<double> X = dhost(dx); double nr = 0, nb = 0;
    for (int i = 0; i < n; i++) { double t = 0; for (auto& p : R[i]) t += p.second * X[p.first]; nr += (bh[i] - t) * (bh[i] - t); nb += bh[i] * bh[i]; } x2 = sqrt(nr / nb); fprintf(stderr, "dpbicg+icc info %lld iter %lld res %.3g\n", (long long)si, (long long)sp.numiter, x2);
    ok = info == 0 && x1 < 1e-12; }
  else if (op == "cidr") { std::string mat = argv[2]; int s = atoi(argv[3]), n = 200; double th = mat == "rot90" ? 3.14159265358979323846 / 2 : 0; cf e((float)cos(th), (float)sin(th));
    std::vector<std::vector<std::pair<int, cf>>> R(n); for (int i = 0; i < n; i++) { if (i) R[i].push_back({i - 1, -e}); R[i].push_back({i, 2.1f * e}); if (i + 1 < n) R[i].push_back({i + 1, -e}); }
    magma_c_matrix A = ccsr(n, R), dA = {Magma_CSR}; magma_cmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q); std::vector<cf> xt(n), bh(n, 0.f); for (auto& v : xt) v = cf((float)r1(), (float)r1()); for (int i = 0; i < n; i++) for (auto& p : R[i]) bh[i] += p.second * xt[p.first];
    magma_c_matrix db = cdev(bh), dx = {Magma_CSR}; magma_cvinit(&dx, Magma_DEV, n, 1, MAGMA_C_ZERO, Q); magma_c_solver_par sp; memset(&sp, 0, sizeof(sp)); sp.rtol = 1e-5; sp.maxiter = 2 * n; sp.restart = s;
    info = magma_cidr(dA, db, &dx, &sp, Q); std::vector<cf> X = chost(dx); double nr = 0, nb = 0; for (int i = 0; i < n; i++) { cf t = 0; for (auto& p : R[i]) t += p.second * X[p.first]; nr += std::norm(bh[i] - t); nb += std::norm(bh[i]); }
    x1 = (double)sp.numiter; x2 = sqrt(nr / nb); ok = info == 0 && x2 < 1e-4; }
  else if (op == "cjac") { int g = atoi(argv[2]), n = g * g; std::vector<std::vector<std::pair<int, cf>>> R(n); cf dg(4.05f, 0.8f), lo(-1, 0.3f), upv(-1, -0.2f);
    for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) { int r = x + y * g; if (y) R[r].push_back({r - g, lo}); if (x) R[r].push_back({r - 1, lo}); R[r].push_back({r, dg}); if (x + 1 < g) R[r].push_back({r + 1, upv}); if (y + 1 < g) R[r].push_back({r + g, upv}); }
    magma_c_matrix A = ccsr(n, R), dA = {Magma_CSR}; magma_cmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q); std::vector<cf> xt(n), bh(n, 0.f); for (auto& v : xt) v = cf((float)r1(), (float)r1()); for (int i = 0; i < n; i++) for (auto& p : R[i]) bh[i] += p.second * xt[p.first];
    magma_c_matrix db = cdev(bh); magma_c_solver_par sp; memset(&sp, 0, sizeof(sp)); sp.solver = Magma_LSQR; sp.rtol = 1e-5; sp.maxiter = 2 * n; magma_c_preconditioner pp; memset(&pp, 0, sizeof(pp)); pp.solver = Magma_JACOBI;
    info = magma_c_precondsetup(dA, db, &sp, &pp, Q); std::vector<cf> u(n), v(n); for (auto& a : u) a = cf((float)r1(), (float)r1()); for (auto& a : v) a = cf((float)r1(), (float)r1());
    magma_c_matrix du = cdev(u), dv = cdev(v), t1 = {Magma_CSR}, y = {Magma_CSR}, t2 = {Magma_CSR}, yt = {Magma_CSR}; for (auto* m : {&t1, &y, &t2, &yt}) magma_cvinit(m, Magma_DEV, n, 1, MAGMA_C_ZERO, Q);
    magma_c_applyprecond_left(MagmaNoTrans, dA, dv, &t1, &pp, Q); magma_c_applyprecond_right(MagmaNoTrans, dA, t1, &y, &pp, Q); magma_c_applyprecond_right(MagmaTrans, dA, du, &t2, &pp, Q); magma_c_applyprecond_left(MagmaTrans, dA, t2, &yt, &pp, Q);
    magma_queue_sync(Q); std::vector<cf> Y = chost(y), YT = chost(yt); std::complex<double> l = 0, r = 0; for (int i = 0; i < n; i++) { l += std::complex<double>(std::conj(u[i]) * Y[i]); r += std::complex<double>(std::conj(YT[i]) * v[i]); } x1 = std::abs(l - r) / std::abs(l);
    for (int pc = 0; pc < 2; pc++) { magma_c_matrix dx = {Magma_CSR}; magma_cvinit(&dx, Magma_DEV, n, 1, MAGMA_C_ZERO, Q); magma_c_solver_par s2 = sp; magma_c_preconditioner p2 = pp; if (!pc) p2.solver = Magma_NONE;
      magma_int_t si = magma_clsqr(dA, db, &dx, &s2, &p2, Q); std::vector<cf> X = chost(dx); double nr = 0, nb = 0; for (int i = 0; i < n; i++) { cf t = 0; for (auto& p : R[i]) t += p.second * X[p.first]; nr += std::norm(bh[i] - t); nb += std::norm(bh[i]); }
      fprintf(stderr, "clsqr %s: info %lld iter %lld res %.3g\n", pc ? "jacobi" : "none", (long long)si, (long long)s2.numiter, sqrt(nr / nb)); if (pc) x2 = sqrt(nr / nb); }
    ok = info == 0 && x1 < 1e-5; }
  printf("r80,%s,%s,%lld,%.3g,%.3g,%s\n", op.c_str(), tag.c_str(), info, x1, x2, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

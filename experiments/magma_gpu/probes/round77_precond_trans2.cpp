// Round 77 (roxy): copy of round76_precond_trans with a complex matrix option, the Hermitian identity, Jacobi, and LSQR.
// Round 76 (roxy): the transposed preconditioner applications used by PBiCG / PQMR (the shadow system needs M^{-T}).
// A is the real 2-D Laplacian on a g x g grid plus 0.05 I (stored as complex), so for a real M, M^{-T} = M^{-H}.
//   round76_precond_trans ident <ilu|icc> <g> <setup solver pbicg|pqmr>
//       y  = right(N, left(N, v))  = M^{-1} v     (the order PBiCG uses for r)
//       yt = left(T, right(T, u))  = should be M^{-T} u   (the order PBiCG uses for rt)
//       checks the bilinear identity u^T y = yt^T v (holds for every invertible M iff yt = M^{-T} u); also prints the swapped
//       candidate right(T, left(T, u)) against the identity
//   R77_CPX=1: diagonal 4.05 + 0.8 i, off-diagonals -1 + 0.3 i (lower) and -1 - 0.2 i (upper): complex, non-Hermitian
//   ident also prints the Hermitian identity u^H y = yt^H v (holds iff yt = M^{-H} u); prec jacobi allowed
//   round76_precond_trans solve <pbicg|pqmr|bicg|qmr> <ilu|icc|none> <g>   b = A x_true, x0 = 0, rtol 1e-10, maxiter 2n
// CSV: r76i,prec,g,setup,info,rel_ident,rel_ident_swapped,verdict   /   r76s,solver,prec,g,info,iter,true_res,verdict
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
typedef std::complex<double> cd;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static magma_queue_t Q;
static cd C(magmaDoubleComplex v) { return cd(MAGMA_Z_REAL(v), MAGMA_Z_IMAG(v)); }
static magmaDoubleComplex Z(cd v) { return MAGMA_Z_MAKE(v.real(), v.imag()); }
static magma_z_matrix dvec(const std::vector<cd>& h) { std::vector<magmaDoubleComplex> t(h.size()); for (size_t i = 0; i < h.size(); i++) t[i] = Z(h[i]); magma_z_matrix x = {Magma_CSR}, dx = {Magma_CSR};
  magma_zvset((magma_int_t)h.size(), 1, t.data(), &x, Q); magma_zmtransfer(x, &dx, Magma_CPU, Magma_DEV, Q); return dx; }
static std::vector<cd> hvec(magma_z_matrix dx) { magma_z_matrix x = {Magma_CSR}; magma_zmtransfer(dx, &x, Magma_DEV, Magma_CPU, Q); std::vector<cd> h(x.num_rows); for (size_t i = 0; i < h.size(); i++) h[i] = C(x.val[i]); return h; }
static cd dotu(const std::vector<cd>& a, const std::vector<cd>& b) { cd s = 0; for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i]; return s; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
  if (argc < 4) return 1; std::string op = argv[1]; magma_init(); magma_queue_create(0, &Q);
  bool ident = op == "ident"; std::string prec = ident ? argv[2] : argv[3]; int g = atoi(ident ? argv[3] : argv[4]); int n = g * g;
  bool cpx = getenv("R77_CPX") != NULL; cd dg = cpx ? cd(4.05, 0.8) : cd(4.05, 0), lo = cpx ? cd(-1, 0.3) : cd(-1, 0), upv = cpx ? cd(-1, -0.2) : cd(-1, 0);
  std::vector<std::vector<std::pair<int, cd>>> R(n);
  for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) { int r = x + y * g; if (y) R[r].push_back({r - g, lo}); if (x) R[r].push_back({r - 1, lo}); R[r].push_back({r, dg}); if (x + 1 < g) R[r].push_back({r + 1, upv}); if (y + 1 < g) R[r].push_back({r + g, upv}); }
  std::vector<magma_index_t> row(1, 0), col; std::vector<magmaDoubleComplex> val; for (int i = 0; i < n; i++) { for (auto& p : R[i]) { col.push_back(p.first); val.push_back(Z(p.second)); } row.push_back((magma_index_t)col.size()); }
  magma_index_t* rp = (magma_index_t*)malloc(row.size() * sizeof(magma_index_t)); magma_index_t* cp = (magma_index_t*)malloc(col.size() * sizeof(magma_index_t)); magmaDoubleComplex* vp = (magmaDoubleComplex*)malloc(val.size() * sizeof(magmaDoubleComplex));
  memcpy(rp, row.data(), row.size() * sizeof(magma_index_t)); memcpy(cp, col.data(), col.size() * sizeof(magma_index_t)); memcpy(vp, val.data(), val.size() * sizeof(magmaDoubleComplex));
  magma_z_matrix A = {Magma_CSR}, dA = {Magma_CSR}; magma_zcsrset(n, n, rp, cp, vp, &A, Q); magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q);
  std::vector<cd> xt(n), bh(n, 0.0); for (auto& v : xt) v = cd(r1(), r1()); for (int i = 0; i < n; i++) for (auto& p : R[i]) bh[i] += p.second * xt[p.first];
  magma_z_matrix db = dvec(bh);
  magma_z_solver_par sp; memset(&sp, 0, sizeof(sp)); sp.rtol = 1e-10; sp.maxiter = 2 * n; sp.verbose = 0;
  magma_z_preconditioner pp; memset(&pp, 0, sizeof(pp)); pp.solver = prec == "ilu" ? Magma_ILU : prec == "icc" ? Magma_ICC : prec == "jacobi" ? Magma_JACOBI : Magma_NONE;
  if (ident) {
    std::string su = argv[4]; sp.solver = su == "pqmr" ? Magma_PQMR : su == "lsqr" ? Magma_LSQR : Magma_PBICG;
    magma_int_t info = magma_z_precondsetup(dA, db, &sp, &pp, Q);
    std::vector<cd> u(n), v(n); for (auto& a : u) a = cd(r1(), r1()); for (auto& a : v) a = cd(r1(), r1());
    magma_z_matrix du = dvec(u), dv = dvec(v), t1 = {Magma_CSR}, y = {Magma_CSR}, t2 = {Magma_CSR}, yt = {Magma_CSR}, t3 = {Magma_CSR}, ys = {Magma_CSR};
    for (auto* m : {&t1, &y, &t2, &yt, &t3, &ys}) magma_zvinit(m, Magma_DEV, n, 1, MAGMA_Z_ZERO, Q);
    magma_int_t i1 = magma_z_applyprecond_left(MagmaNoTrans, dA, dv, &t1, &pp, Q); magma_int_t i2 = magma_z_applyprecond_right(MagmaNoTrans, dA, t1, &y, &pp, Q);
    magma_int_t i3 = magma_z_applyprecond_right(MagmaTrans, dA, du, &t2, &pp, Q); magma_int_t i4 = magma_z_applyprecond_left(MagmaTrans, dA, t2, &yt, &pp, Q);
    magma_int_t i5 = magma_z_applyprecond_left(MagmaTrans, dA, du, &t3, &pp, Q); magma_int_t i6 = magma_z_applyprecond_right(MagmaTrans, dA, t3, &ys, &pp, Q);
    magma_queue_sync(Q); if (!info) info = i1 ? i1 : i2 ? i2 : i3 ? i3 : i4 ? i4 : i5 ? i5 : i6;
    std::vector<cd> Y = hvec(y), YT = hvec(yt), YS = hvec(ys); cd l = dotu(u, Y), r = dotu(YT, v), rs = dotu(YS, v);
    double e = std::abs(l - r) / std::abs(l), es = std::abs(l - rs) / std::abs(l);
    std::vector<cd> uc = u, Yc = YT; for (auto& a : uc) a = std::conj(a); for (auto& a : Yc) a = std::conj(a); cd lh = dotu(uc, Y), rh = dotu(Yc, v); double eh = std::abs(lh - rh) / std::abs(lh);
    printf("r77i,%s,%s,%d,%s,%lld,%.3g,%.3g,%.3g,%s\n", cpx ? "cpx" : "real", prec.c_str(), g, su.c_str(), (long long)info, e, es, eh, info == 0 && (cpx ? eh : e) < 1e-12 ? "OK" : "WRONG");
  } else {
    std::string so = argv[2]; sp.solver = so == "pbicg" ? Magma_PBICG : so == "pqmr" ? Magma_PQMR : so == "lsqr" ? Magma_LSQR : so == "bicg" ? Magma_BICG : Magma_QMR;
    magma_int_t info = 0; if (so[0] == 'p' || so == "lsqr") info = magma_z_precondsetup(dA, db, &sp, &pp, Q);
    magma_z_matrix dx = {Magma_CSR}; magma_zvinit(&dx, Magma_DEV, n, 1, MAGMA_Z_ZERO, Q); sp.maxiter = 2 * n; sp.rtol = 1e-10;
    magma_int_t si = so == "pbicg" ? magma_zpbicg(dA, db, &dx, &sp, &pp, Q) : so == "pqmr" ? magma_zpqmr(dA, db, &dx, &sp, &pp, Q) : so == "lsqr" ? magma_zlsqr(dA, db, &dx, &sp, &pp, Q) : so == "bicg" ? magma_zbicg(dA, db, &dx, &sp, Q) : magma_zqmr(dA, db, &dx, &sp, Q);
    if (!info) info = si; magma_queue_sync(Q); std::vector<cd> X = hvec(dx);
    double nb = 0, nr = 0; for (int i = 0; i < n; i++) { cd t = 0; for (auto& p : R[i]) t += p.second * X[p.first]; nr += std::norm(bh[i] - t); nb += std::norm(bh[i]); }
    double tr = sqrt(nr / nb); printf("r77s,%s,%s,%s,%d,%lld,%lld,%.3g,%.3g,%s\n", cpx ? "cpx" : "real", so.c_str(), prec.c_str(), g, (long long)info, (long long)sp.numiter, sp.final_res / sqrt(nb), tr, info == 0 && tr < 1e-8 ? "OK" : "WRONG");
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

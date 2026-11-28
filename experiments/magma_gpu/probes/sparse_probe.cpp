// GPU confirmation of the MAGMA v2.10.0 sparse predictions (HANDOFF §28.5/28.7, Lean MagmaSparse.lean).
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <complex>
typedef std::complex<double> cd;
static magma_queue_t Q;

struct Sys { int n; std::vector<cd> A; std::vector<cd> b; };   // dense column-major

static void to_csr(const Sys& s, magma_z_matrix* dA) {
  int n = s.n; std::vector<magma_index_t> row(n + 1), col; std::vector<magmaDoubleComplex> val;
  for (int i = 0; i < n; i++) { row[i] = (magma_index_t)col.size();
    for (int j = 0; j < n; j++) { cd a = s.A[i + j * n]; if (a != cd(0, 0)) { col.push_back(j); val.push_back(MAGMA_Z_MAKE(a.real(), a.imag())); } } }
  row[n] = (magma_index_t)col.size();
  magma_z_matrix hA = {Magma_CSR};
  magma_zcsrset(n, n, row.data(), col.data(), val.data(), &hA, Q);
  magma_zmtransfer(hA, dA, Magma_CPU, Magma_DEV, Q);
}
static void to_dev(const std::vector<cd>& v, magma_z_matrix* d) {
  std::vector<magmaDoubleComplex> t(v.size()); for (size_t i = 0; i < v.size(); i++) t[i] = MAGMA_Z_MAKE(v[i].real(), v[i].imag());
  magma_z_matrix h = {Magma_CSR}; magma_zvset((magma_int_t)v.size(), 1, t.data(), &h, Q);
  magma_zmtransfer(h, d, Magma_CPU, Magma_DEV, Q);
}
static std::vector<cd> from_dev(magma_z_matrix d) {
  magma_z_matrix h = {Magma_CSR}; magma_zmtransfer(d, &h, Magma_DEV, Magma_CPU, Q);
  std::vector<cd> v(h.num_rows); for (int i = 0; i < h.num_rows; i++) v[i] = cd(MAGMA_Z_REAL(h.val[i]), MAGMA_Z_IMAG(h.val[i]));
  magma_zmfree(&h, Q); return v;
}
static std::vector<cd> ref_solve(const Sys& s) {   // LAPACK zgesv with pivoting
  int n = s.n; std::vector<magmaDoubleComplex> A(n * n), x(n); std::vector<magma_int_t> ip(n); magma_int_t one = 1, info;
  for (int i = 0; i < n * n; i++) A[i] = MAGMA_Z_MAKE(s.A[i].real(), s.A[i].imag());
  for (int i = 0; i < n; i++) x[i] = MAGMA_Z_MAKE(s.b[i].real(), s.b[i].imag());
  magma_int_t nn = n; lapackf77_zgesv(&nn, &one, A.data(), &nn, ip.data(), x.data(), &nn, &info);
  std::vector<cd> r(n); for (int i = 0; i < n; i++) r[i] = cd(MAGMA_Z_REAL(x[i]), MAGMA_Z_IMAG(x[i])); return r;
}
static double nrm(const std::vector<cd>& v) { double s = 0; for (auto z : v) s += std::norm(z); return sqrt(s); }
static double rel(const std::vector<cd>& x, const std::vector<cd>& y) { std::vector<cd> d(x.size()); for (size_t i = 0; i < x.size(); i++) d[i] = x[i] - y[i]; return nrm(d) / nrm(y); }
static double true_res(const Sys& s, const std::vector<cd>& x) { int n = s.n; std::vector<cd> r(s.b); for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) r[i] -= s.A[i + j * n] * x[j]; return nrm(r); }

static void init_par(magma_z_solver_par* sp, magma_z_preconditioner* pp, magma_solver_type st, int maxiter) {
  memset(sp, 0, sizeof *sp); memset(pp, 0, sizeof *pp);
  sp->solver = st; sp->maxiter = maxiter; sp->atol = 1e-16; sp->rtol = 1e-10; sp->verbose = 0; sp->restart = 30;
  pp->solver = Magma_NONE;
  magma_zsolverinfo_init(sp, pp, Q);
}

enum Kind { CGMERGE, BICG, CGRES, PCGMERGE };
static void run(const char* name, Kind k, const Sys& s, const std::vector<cd>& x0, int maxiter = 1000) {
  magma_z_matrix dA = {Magma_CSR}, db = {Magma_CSR}, dx = {Magma_CSR};
  to_csr(s, &dA); to_dev(s.b, &db); to_dev(x0, &dx);
  magma_z_solver_par sp; magma_z_preconditioner pp; init_par(&sp, &pp, Magma_CG, maxiter);
  magma_int_t ret = 0;
  if (k == CGMERGE) ret = magma_zcg_merge(dA, db, &dx, &sp, Q);
  if (k == BICG)    ret = magma_zbicg(dA, db, &dx, &sp, Q);
  if (k == CGRES)   ret = magma_zcg_res(dA, db, &dx, &sp, Q);
  if (k == PCGMERGE) ret = magma_zpcg_merge(dA, db, &dx, &sp, &pp, Q);
  auto x = from_dev(dx);
  bool fin = true; for (auto z : x) if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) fin = false;
  double nb = nrm(s.b), tr = fin ? true_res(s, x) : NAN;
  double re = NAN; if (fin && std::isfinite(nb)) re = rel(x, ref_solve(s));
  printf("%-44s ret=%-5lld info=%-5lld numiter=%-5lld final_res/||b||=%-10.3g true_res/||b||=%-10.3g relerr=%-9.2e iter_res/init_res=%.2e  x[0]=(%g,%g)\n",
         name, (long long)ret, (long long)sp.info, (long long)sp.numiter, sp.final_res / nb, tr / nb, re,
         sp.iter_res / sp.init_res, x[0].real(), x[0].imag());
  magma_zmfree(&dA, Q); magma_zmfree(&db, Q); magma_zmfree(&dx, Q);
}

static Sys diag2(double a1, double a2, cd b1, cd b2) { Sys s{2, {a1, 0, 0, a2}, {b1, b2}}; return s; }

static Sys hpd(int n, double cond, bool complexData, unsigned seed) {   // A = Q diag Q^H by modified Gram-Schmidt
  srand(seed); auto r = []() { return rand() / (double)RAND_MAX - 0.5; };
  std::vector<cd> M(n * n); for (auto& z : M) z = complexData ? cd(r(), r()) : cd(r(), 0);
  for (int j = 0; j < n; j++) { for (int k = 0; k < j; k++) { cd d = 0; for (int i = 0; i < n; i++) d += std::conj(M[i + k * n]) * M[i + j * n];
      for (int i = 0; i < n; i++) M[i + j * n] -= d * M[i + k * n]; }
    double nr = 0; for (int i = 0; i < n; i++) nr += std::norm(M[i + j * n]); nr = sqrt(nr); for (int i = 0; i < n; i++) M[i + j * n] /= nr; }
  Sys s; s.n = n; s.A.assign(n * n, 0); s.b.resize(n);
  for (int k = 0; k < n; k++) { double lam = 1 + (cond - 1) * k / (n - 1.0);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) s.A[i + j * n] += M[i + k * n] * lam * std::conj(M[j + k * n]); }
  for (auto& z : s.b) z = complexData ? cd(r(), r()) : cd(r(), 0);
  return s;
}

int main() {
  magma_init(); magma_queue_create(0, &Q);
  printf("== S1/S2 CG_MERGE (default solver), complex 2x2 ==\n");
  run("cg_merge A=diag(2,3) b=(2,i)", CGMERGE, diag2(2, 3, cd(2, 0), cd(0, 1)), {0, 0});
  run("cg_merge A=diag(1,1.5) b=(1,0.5i)", CGMERGE, diag2(1, 1.5, cd(1, 0), cd(0, 0.5)), {0, 0});
  printf("== S3/S4 CG_MERGE on HPD n=64, cond 10 ==\n");
  for (unsigned sd : {1u, 2u}) {
    Sys c = hpd(64, 10, true, sd); run(sd == 1 ? "cg_merge complex HPD seed1" : "cg_merge complex HPD seed2", CGMERGE, c, std::vector<cd>(64, 0));
    Sys r = hpd(64, 10, false, sd); run(sd == 1 ? "cg_merge real SPD in complex seed1 (control)" : "cg_merge real SPD in complex seed2 (control)", CGMERGE, r, std::vector<cd>(64, 0));
  }
  printf("== S5 BiCG, b with NaN ==\n");
  run("bicg A=I b=(NaN,1) x0=0", BICG, diag2(1, 1, cd(NAN, 0), cd(1, 0)), {0, 0});
  run("bicg A=I b=(1,1) x0=0 (control)", BICG, diag2(1, 1, cd(1, 0), cd(1, 0)), {0, 0});
  printf("== S6 CG_RES, exact initial guess ==\n");
  run("cg_res A=diag(2,3) b=(2,3) x0=(1,1) exact", CGRES, diag2(2, 3, cd(2, 0), cd(3, 0)), {1, 1});
  run("cg_res A=diag(2,3) b=(2,3) x0=0 (control)", CGRES, diag2(2, 3, cd(2, 0), cd(3, 0)), {0, 0});
  printf("== S7 solver wrapper with an unhandled solver class ==\n");
  {
    Sys s = diag2(2, 3, cd(2, 0), cd(3, 0));
    magma_z_matrix dA = {Magma_CSR}, db = {Magma_CSR}, dx = {Magma_CSR};
    to_csr(s, &dA); to_dev(s.b, &db); to_dev({7, 7}, &dx);
    magma_zopts opts; memset(&opts, 0, sizeof opts);
    init_par(&opts.solver_par, &opts.precond_par, Magma_ITERREF, 1000);
    magma_int_t ret = magma_z_solver(dA, db, &dx, &opts, Q);
    auto x = from_dev(dx);
    printf("%-44s ret=%lld  x=(%g, %g) (x0 = (7, 7), exact = (1, 1))\n", "magma_z_solver solver=ITERREF", (long long)ret, x[0].real(), x[1].real());
  }
  printf("== S8 PCG_MERGE, large x0, SLOW->SUCCESS upgrade relative to init_res ==\n");
  {
    int n = 1000; Sys s; s.n = n; s.A.assign(n * n, 0); s.b.assign(n, cd(1, 0));
    for (int i = 0; i < n; i++) s.A[i + i * n] = 1 + 9.0 * i / (n - 1);
    srand(7); std::vector<cd> x0(n); for (auto& z : x0) z = cd(1e6 * (rand() / (double)RAND_MAX - 0.5), 0);
    for (int mi : {10, 20, 30, 40, 50, 60}) { char nm[64]; snprintf(nm, 64, "pcg_merge diag(1..10) n=1000 x0~1e6 maxiter=%d", mi); run(nm, PCGMERGE, s, x0, mi); }
  }
  magma_queue_destroy(Q); magma_finalize();
  return 0;
}

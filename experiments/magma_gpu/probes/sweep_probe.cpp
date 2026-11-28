// GPU sweep: every linear solver class of magma_z_solver (sparse wrapper) on six inputs.  GPU 0 only.
// Usage: ./sweep_probe <solver-index>   (one process per solver so a crash or hang affects one row only)
#include "magma_v2.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
typedef std::complex<double> cd;
static magma_queue_t Q;
struct Sol { const char* nm; magma_solver_type t; };
static Sol SOLVERS[] = {
  {"BICG", Magma_BICG}, {"PBICG", Magma_PBICG}, {"BICGSTAB", Magma_BICGSTAB}, {"BICGSTABMERGE", Magma_BICGSTABMERGE},
  {"PBICGSTAB", Magma_PBICGSTAB}, {"PBICGSTABMERGE", Magma_PBICGSTABMERGE}, {"CG", Magma_CG}, {"CGMERGE", Magma_CGMERGE},
  {"PCG", Magma_PCG}, {"PCGMERGE", Magma_PCGMERGE}, {"CGS", Magma_CGS}, {"CGSMERGE", Magma_CGSMERGE}, {"PCGS", Magma_PCGS},
  {"PCGSMERGE", Magma_PCGSMERGE}, {"QMR", Magma_QMR}, {"QMRMERGE", Magma_QMRMERGE}, {"PQMR", Magma_PQMR},
  {"PQMRMERGE", Magma_PQMRMERGE}, {"TFQMR", Magma_TFQMR}, {"TFQMRMERGE", Magma_TFQMRMERGE}, {"PTFQMR", Magma_PTFQMR},
  {"PTFQMRMERGE", Magma_PTFQMRMERGE}, {"GMRES", Magma_GMRES}, {"PGMRES", Magma_PGMRES}, {"IDR", Magma_IDR},
  {"IDRMERGE", Magma_IDRMERGE}, {"PIDR", Magma_PIDR}, {"PIDRMERGE", Magma_PIDRMERGE}, {"LSQR", Magma_LSQR},
  {"JACOBI", Magma_JACOBI}, {"BOMBARD", Magma_BOMBARD}, {"BOMBARDMERGE", Magma_BOMBARDMERGE} };
static const int NS = sizeof(SOLVERS) / sizeof(SOLVERS[0]);

struct In { const char* nm; int n; std::vector<cd> A, b, x0, xt; };
static unsigned Sd; static double r() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static void mv(const In& I, const std::vector<cd>& x, std::vector<cd>& y) { int n = I.n; y.assign(n, 0); for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) y[i] += I.A[i * n + j] * x[j]; }
static In realSPD(const char* nm) { In I{nm, 64}; int n = 64; I.A.assign(n * n, 0); for (int i = 0; i < n; i++) { I.A[i * n + i] = 4; if (i) I.A[i * n + i - 1] = -1; if (i + 1 < n) I.A[i * n + i + 1] = -1; }
  I.xt.resize(n); for (auto& v : I.xt) v = r(); mv(I, I.xt, I.b); I.x0.assign(n, 0); return I; }
static In complexHPD() { In I{"complex HPD", 64}; int n = 64; Sd = 5u;
  std::vector<cd> M(n * n); for (auto& z : M) z = cd(r(), r());
  for (int j = 0; j < n; j++) { for (int k = 0; k < j; k++) { cd d = 0; for (int i = 0; i < n; i++) d += std::conj(M[i * n + k]) * M[i * n + j]; for (int i = 0; i < n; i++) M[i * n + j] -= d * M[i * n + k]; }
    double nr = 0; for (int i = 0; i < n; i++) nr += std::norm(M[i * n + j]); nr = sqrt(nr); for (int i = 0; i < n; i++) M[i * n + j] /= nr; }
  I.A.assign(n * n, 0); for (int k = 0; k < n; k++) { double lam = 1 + 9.0 * k / (n - 1); for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) I.A[i * n + j] += M[i * n + k] * lam * std::conj(M[j * n + k]); }
  I.xt.resize(n); for (auto& v : I.xt) v = cd(r(), r()); mv(I, I.xt, I.b); I.x0.assign(n, 0); return I; }
static In complexNonHerm() { In I{"complex non-Hermitian", 64}; int n = 64; Sd = 9u; I.A.resize(n * n);
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) I.A[i * n + j] = (i == j ? cd(4, 1) : cd(0.3 * r(), 0.3 * r()) / 8.0);
  I.xt.resize(n); for (auto& v : I.xt) v = cd(r(), r()); mv(I, I.xt, I.b); I.x0.assign(n, 0); return I; }

static void run(const Sol& s, In I) {
  int n = I.n; std::vector<magma_index_t> row(n + 1), col; std::vector<magmaDoubleComplex> val;
  for (int i = 0; i < n; i++) { row[i] = (magma_index_t)col.size(); for (int j = 0; j < n; j++) if (I.A[i * n + j] != cd(0, 0)) { col.push_back(j); val.push_back(MAGMA_Z_MAKE(I.A[i * n + j].real(), I.A[i * n + j].imag())); } }
  row[n] = (magma_index_t)col.size();
  magma_z_matrix hA = {Magma_CSR}, dA = {Magma_CSR}, hb = {Magma_CSR}, db = {Magma_CSR}, hx = {Magma_CSR}, dx = {Magma_CSR};
  magma_zcsrset(n, n, row.data(), col.data(), val.data(), &hA, Q); magma_zmtransfer(hA, &dA, Magma_CPU, Magma_DEV, Q);
  std::vector<magmaDoubleComplex> bb(n), xx(n); for (int i = 0; i < n; i++) { bb[i] = MAGMA_Z_MAKE(I.b[i].real(), I.b[i].imag()); xx[i] = MAGMA_Z_MAKE(I.x0[i].real(), I.x0[i].imag()); }
  magma_zvset(n, 1, bb.data(), &hb, Q); magma_zmtransfer(hb, &db, Magma_CPU, Magma_DEV, Q);
  magma_zvset(n, 1, xx.data(), &hx, Q); magma_zmtransfer(hx, &dx, Magma_CPU, Magma_DEV, Q);
  magma_zopts o; memset(&o, 0, sizeof o);
  o.solver_par.solver = s.t; o.solver_par.maxiter = 1000; o.solver_par.restart = getenv("SWEEP_RESTART") ? atoi(getenv("SWEEP_RESTART")) : 30; o.solver_par.atol = 1e-16; o.solver_par.rtol = 1e-10;
  o.precond_par.solver = Magma_NONE; o.precond_par.maxiter = 3;
  magma_zsolverinfo_init(&o.solver_par, &o.precond_par, Q);
  magma_int_t ret = magma_z_solver(dA, db, &dx, &o, Q);
  magma_z_matrix ox = {Magma_CSR}; magma_zmtransfer(dx, &ox, Magma_DEV, Magma_CPU, Q);
  std::vector<cd> x(n); bool fin = true; for (int i = 0; i < n; i++) { x[i] = cd(MAGMA_Z_REAL(ox.val[i]), MAGMA_Z_IMAG(ox.val[i])); if (!std::isfinite(x[i].real()) || !std::isfinite(x[i].imag())) fin = false; }
  std::vector<cd> Ax; mv(I, x, Ax); double rr = 0, nb = 0; for (int i = 0; i < n; i++) { rr += std::norm(I.b[i] - Ax[i]); nb += std::norm(I.b[i]); }
  double rel = sqrt(rr / nb);
  magma_int_t info = o.solver_par.info;
  const char* verdict = "ok";
  bool good = fin && rel <= 1e-6;
  if (info == 0 && !good) verdict = "SILENT-FAIL"; else if (info != 0 && good) verdict = "false-failure"; else if (info != 0) verdict = "reported-failure";
  printf("%-15s | %-22s | ret=%-5lld info=%-5lld numiter=%-5lld | finite=%d true_rel_res=%-10.3g | %s\n", s.nm, I.nm, (long long)ret, (long long)info,
         (long long)o.solver_par.numiter, (int)fin, rel, verdict);
  fflush(stdout);
  magma_zmfree(&dA, Q); magma_zmfree(&db, Q); magma_zmfree(&dx, Q); magma_zmfree(&ox, Q);
}

int main(int argc, char** argv) {
  int k = argc > 1 ? atoi(argv[1]) : -1;
  if (k < 0 || k >= NS) { printf("%d\n", NS); return 0; }
  magma_init(); magma_queue_create(0, &Q);
  Sd = 1u; In a = realSPD("real SPD (control)"); run(SOLVERS[k], a);
  run(SOLVERS[k], complexHPD()); run(SOLVERS[k], complexNonHerm());
  Sd = 2u; In d = realSPD("tiny b=1e-20, x0 random"); for (auto& v : d.b) v *= 1e-20; for (auto& v : d.xt) v *= 1e-20; for (auto& v : d.x0) v = r(); run(SOLVERS[k], d);
  Sd = 3u; In e = realSPD("b[0]=NaN"); e.b[0] = cd(NAN, 0); run(SOLVERS[k], e);
  { In f{"exact x0 (residual 0)", 64}; int n = 64; f.A.assign(n * n, 0); for (int i = 0; i < n; i++) f.A[i * n + i] = 2 + (i % 3);
    f.xt.resize(n); for (int i = 0; i < n; i++) f.xt[i] = 1 + (i % 5); mv(f, f.xt, f.b); f.x0 = f.xt; run(SOLVERS[k], f); }
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

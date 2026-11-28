// Round 45 (roxy): the round-29 sparse solver harness on edge-case systems (n = 1, identity, diagonal, 2x2, b = 0,
// an empty row, one dense row).  Same verdicts as round 29; for the empty-row (singular) system FAIL is the honest answer.
// Derived from round29_sparse.cpp; only the problem list differs.
// (round 29 header follows)
// Twenty-ninth round (roxy): differential sweep of the MAGMA sparse iterative solvers.  Each run is checked against the
// TRUE residual ||b - A x|| / ||b|| computed on the host, and against what the solver reports (info, final_res).
//     round29_sparse <solver-name>      one solver per process; loops over matrices x preconditioners x formats
// Options are passed through magma_zparse_opts (a synthetic argv), so every default is the library's own.
// CSV: solver,precond,format,matrix,n,ret,info,numiter,final_res,true_abs,true_rel,verdict
//   LIE        info == 0 (success) but true_rel > 1e-6, or x non-finite
//   GAP        info == 0, true_rel <= 1e-6, but the reported final_res understates the true residual by > 1e3
//   FALSEFAIL  info != 0 but true_rel <= 1e-8
//   FAIL       info != 0 and true_rel > 1e-8 (an honest failure)
//   OK         info == 0 and true_rel <= 1e-6
#include "magma_v2.h"
#include "magmasparse.h"
#include <cuda_runtime.h>
#include <complex>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
typedef std::complex<double> cd;
static magma_queue_t Q;
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }

struct Csr { int n; std::vector<int> row, col; std::vector<cd> val; };
static Csr grid(int g, double ax, double ay, double conv, cd shift, bool neumann) {
  Csr A; A.n = g * g; A.row.push_back(0);
  for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) { int i = y * g + x; cd d = 0; std::vector<std::pair<int, cd>> e;
    auto nb = [&](int xx, int yy, double w, double c) { if (xx < 0 || yy < 0 || xx >= g || yy >= g) { if (!neumann) d += w; return; } e.push_back({yy * g + xx, cd(-w - c, 0)}); d += w; };
    nb(x - 1, y, ax, conv); nb(x + 1, y, ax, 0); nb(x, y - 1, ay, 0); nb(x, y + 1, ay, 0); d += -shift; if (conv != 0) d += conv;
    e.push_back({i, d}); std::sort(e.begin(), e.end(), [](auto& a, auto& b) { return a.first < b.first; });
    for (auto& p : e) { A.col.push_back(p.first); A.val.push_back(p.second); } A.row.push_back((int)A.col.size()); }
  return A; }
static std::vector<cd> spmv(const Csr& A, const std::vector<cd>& x) { std::vector<cd> y(A.n); for (int i = 0; i < A.n; i++) for (int k = A.row[i]; k < A.row[i + 1]; k++) y[i] += A.val[k] * x[A.col[k]]; return y; }
static double nrm(const std::vector<cd>& v) { double s = 0; for (auto& a : v) s += std::norm(a); return sqrt(s); }

struct Prob { std::string nm; Csr A; std::vector<cd> b, x0; };
static Csr fromDense(int n, const std::vector<double>& D) { Csr A; A.n = n; A.row.push_back(0);
  for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) if (D[i + (size_t)j * n] != 0) { A.col.push_back(j); A.val.push_back(D[i + (size_t)j * n]); } A.row.push_back((int)A.col.size()); } return A; }
static std::vector<Prob> problems() {
  std::vector<Prob> P;
  auto mk = [&](std::string nm, Csr A, bool zero_rhs) { Prob p{nm, A}; Sd = 31u + (unsigned)nm.size(); std::vector<cd> xt(A.n); for (auto& v : xt) v = cd(r1(), 0); p.b = spmv(A, xt); if (zero_rhs) for (auto& v : p.b) v = 0; p.x0.assign(A.n, 0); P.push_back(p); };
  { std::vector<double> D(1, 3.0); mk("n1", fromDense(1, D), false); }
  { int n = 8; std::vector<double> D((size_t)n * n, 0); for (int i = 0; i < n; i++) D[i + (size_t)i * n] = 1; mk("identity8", fromDense(n, D), false); }
  { int n = 64; std::vector<double> D((size_t)n * n, 0); for (int i = 0; i < n; i++) D[i + (size_t)i * n] = 1 + i; mk("diag64", fromDense(n, D), false); }
  { std::vector<double> D = {4, -1, -1, 4}; mk("tridiag2", fromDense(2, D), false); }
  { mk("zero_rhs", grid(16, 1, 1, 0, 0, false), true); }
  { Csr A = grid(16, 1, 1, 0, 0, false); Csr B; B.n = A.n; B.row.push_back(0); for (int i = 0; i < A.n; i++) { if (i != 37) for (int k = A.row[i]; k < A.row[i + 1]; k++) { B.col.push_back(A.col[k]); B.val.push_back(A.val[k]); } B.row.push_back((int)B.col.size()); }
    Prob p{"emptyrow", B}; Sd = 5u; std::vector<cd> xt(B.n); for (auto& v : xt) v = cd(r1(), 0); p.b = spmv(B, xt); p.x0.assign(B.n, 0); P.push_back(p); }   // consistent b, but A singular
  { int n = 64; std::vector<double> D((size_t)n * n, 0); for (int i = 0; i < n; i++) D[i + (size_t)i * n] = n + 1.0; for (int j = 0; j < n; j++) if (j != 5) D[5 + (size_t)j * n] = 0.5; mk("denserow", fromDense(n, D), false); }
  return P; }

static void run(const std::string& solver, const std::string& prec, const std::string& fmt, const Prob& p) {
  int n = p.A.n;
  std::vector<std::string> args = {"probe", "--solver", solver, "--precond", prec, "--format", fmt, "--maxiter", "1000", "--rtol", "1e-10", "--restart", "30", "--blocksize", "8", "--alignment", "8"};
  std::vector<char*> av; for (auto& a : args) av.push_back((char*)a.c_str());
  magma_zopts o; memset(&o, 0, sizeof o); int i = 1; magma_int_t pe = magma_zparse_opts((int)av.size(), av.data(), &o, &i, Q);
  if (fmt == "ELLRT") { o.output_format = Magma_ELLRT; }   // round 46: magma_zparse_opts does not map --format ELLRT (falls back to CSR)
  magma_z_matrix hA = {Magma_CSR}, hB = {Magma_CSR}, dB = {Magma_CSR}, db = {Magma_CSR}, dx = {Magma_CSR}, hx = {Magma_CSR}, hb = {Magma_CSR};
  std::vector<magma_index_t> row(p.A.row.begin(), p.A.row.end()), col(p.A.col.begin(), p.A.col.end());
  std::vector<magmaDoubleComplex> val(p.A.val.size()); for (size_t k = 0; k < val.size(); k++) val[k] = MAGMA_Z_MAKE(p.A.val[k].real(), p.A.val[k].imag());
  magma_zcsrset(n, n, row.data(), col.data(), val.data(), &hA, Q);
  std::vector<magmaDoubleComplex> bb(n), xx(n); for (int k = 0; k < n; k++) { bb[k] = MAGMA_Z_MAKE(p.b[k].real(), p.b[k].imag()); xx[k] = MAGMA_Z_MAKE(p.x0[k].real(), p.x0[k].imag()); }
  magma_zvset(n, 1, bb.data(), &hb, Q); magma_zmtransfer(hb, &db, Magma_CPU, Magma_DEV, Q);
  magma_zvset(n, 1, xx.data(), &hx, Q); magma_zmtransfer(hx, &dx, Magma_CPU, Magma_DEV, Q);
  magma_zsolverinfo_init(&o.solver_par, &o.precond_par, Q);
  magma_int_t ps = 0; if (o.precond_par.solver != Magma_NONE) ps = magma_z_precondsetup(hA, db, &o.solver_par, &o.precond_par, Q);
  hB.blocksize = o.blocksize; hB.alignment = o.alignment; magma_int_t ce = magma_zmconvert(hA, &hB, Magma_CSR, o.output_format, Q); magma_zmtransfer(hB, &dB, Magma_CPU, Magma_DEV, Q);
  if (getenv("R29_DEBUG")) { fprintf(stderr, "solver=%d prec=%d rtol=%g atol=%g maxiter=%lld restart=%lld scaling=%d fmt=%d row0:", (int)o.solver_par.solver, (int)o.precond_par.solver,
      o.solver_par.rtol, o.solver_par.atol, (long long)o.solver_par.maxiter, (long long)o.solver_par.restart, (int)o.scaling, (int)o.output_format);
    for (int k = p.A.row[0]; k < p.A.row[1]; k++) fprintf(stderr, " (%d,%g)", p.A.col[k], p.A.val[k].real()); fprintf(stderr, "\n"); }
  magma_int_t ret = magma_z_solver(dB, db, &dx, &o, Q);
  magma_z_matrix ox = {Magma_CSR}; magma_zmtransfer(dx, &ox, Magma_DEV, Magma_CPU, Q);
  std::vector<cd> x(n); bool fin = true; for (int k = 0; k < n; k++) { x[k] = cd(MAGMA_Z_REAL(ox.val[k]), MAGMA_Z_IMAG(ox.val[k])); if (!std::isfinite(x[k].real()) || !std::isfinite(x[k].imag())) fin = false; }
  std::vector<cd> Ax = spmv(p.A, x), rr(n); for (int k = 0; k < n; k++) rr[k] = p.b[k] - Ax[k]; double ta = nrm(rr), tr = ta / nrm(p.b);
  magma_int_t info = o.solver_par.info; double fr = o.solver_par.final_res;
  const char* v;
  if (info == 0) v = (!fin || !(tr <= 1e-6)) ? "LIE" : (ta > 1e3 * std::max(fr, 1e-300) && tr > 1e-8 ? "GAP" : "OK");
  else v = (fin && tr <= 1e-8) ? "FALSEFAIL" : "FAIL";
  printf("%s,%s,%s,%s,%d,%lld,%lld,%lld,%.3g,%.3g,%.3g,%s,parse=%lld setup=%lld convert=%lld\n", solver.c_str(), prec.c_str(), fmt.c_str(), p.nm.c_str(), n, (long long)ret, (long long)info,
         (long long)o.solver_par.numiter, fr, ta, tr, v, (long long)pe, (long long)ps, (long long)ce);
  magma_zsolverinfo_free(&o.solver_par, &o.precond_par, Q);
  magma_zmfree(&dB, Q); magma_zmfree(&hB, Q); magma_zmfree(&db, Q); magma_zmfree(&dx, Q); magma_zmfree(&ox, Q); magma_zmfree(&hb, Q); magma_zmfree(&hx, Q);
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 2) { printf("usage: round29_sparse <solver>\n"); return 0; }
  std::string s = argv[1]; magma_init(); magma_queue_create(0, &Q);
  if (s == "sizes") {   // CG / PCG / IDR / BICGSTAB on Poisson grids and 1-D tridiagonal systems of growing size
    std::vector<Prob> P;
    auto add = [&](std::string nm, Csr A) { Prob p{nm, A}; Sd = 5u; std::vector<cd> xt(A.n); for (auto& v : xt) v = cd(r1(), r1()); p.b = spmv(A, xt); p.x0.assign(A.n, 0); P.push_back(p); };
    for (int g : {4, 8, 16, 24, 32}) add("poisson2d_g" + std::to_string(g), grid(g, 1, 1, 0, 0, false));
    for (int n : {64, 256, 1024}) { Csr A; A.n = n; A.row.push_back(0); for (int i = 0; i < n; i++) { if (i) { A.col.push_back(i - 1); A.val.push_back(-1); } A.col.push_back(i); A.val.push_back(4);
        if (i + 1 < n) { A.col.push_back(i + 1); A.val.push_back(-1); } A.row.push_back((int)A.col.size()); } add("tridiag_n" + std::to_string(n), A); }
    for (auto sv : {"CG", "PCG", "IDR", "BICGSTAB"}) for (auto& p : P) run(sv, "NONE", "CSR", p);
    magma_queue_destroy(Q); magma_finalize(); return 0; }
  bool pre = s[0] == 'P' || s == "GMRES" || s == "BICGSTAB" || s == "CG" || s == "IDR";   // solvers that accept a preconditioner argument
  std::vector<std::string> precs = {"NONE"}; if (pre) { precs.push_back("JACOBI"); precs.push_back("ILU"); }
  auto P = problems();
  const char* only = getenv("R29_ONLY");
  // round 45 isolation filters: R45_PROB, R45_FMT, R45_PREC (an illegal memory access poisons the whole CUDA context,
  // so the triage runs each case in its own process)
  const char* fp = getenv("R45_PROB"); const char* ff = getenv("R45_FMT"); const char* fc = getenv("R45_PREC");
  for (auto& fmt : (getenv("R45_FMT") && std::string(getenv("R45_FMT")) == "ELLRT") ? std::vector<std::string>{"ELLRT"} : std::vector<std::string>{"CSR", "ELL", "SELLP"}) for (auto& pc : precs) for (auto& p : P) { if (only && p.nm != only) continue; if (only && fmt != "CSR") continue;
    if (fp && p.nm != fp) continue; if (ff && fmt != ff) continue; if (fc && pc != fc) continue; run(s, pc, fmt, p); }
  fprintf(stderr, "DONE %s cuda=%s\n", s.c_str(), cudaGetErrorString(cudaDeviceSynchronize()));
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

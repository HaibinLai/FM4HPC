// STRUMPACK SparseSolverMixedPrecision<float,double,int> (float factorization, double refinement) and SparseSolver<double>.
// Predictions (IR_SURVEY_zh.md):
//   S1 (D10) SparseSolverMixedPrecision::solve always returns SUCCESS ("// TODO check convergence").
//   S2 (D3)  refinement starts from x = 0 and loops `while (res_norm > atol ...)` with atol = 1e-10 absolute, so for
//            (A,b) scaled by 1e-12 the loop never runs and x = 0 is returned with SUCCESS.
//   S3 (D1)  NaN in b: the loop test is false for NaN, so it stops and returns SUCCESS.
// A: 1D tridiag(-1, 4, -1), n = 2000 (well conditioned); x_exact = 1.
#include <cmath>
#include <cstdio>
#include <vector>
#include "StrumpackSparseSolver.hpp"
#include "StrumpackSparseSolverMixedPrecision.hpp"
#include "sparse/CSRMatrix.hpp"
using namespace strumpack;

static CSRMatrix<double, int> tri(int n, double s) {
  std::vector<int> ptr(n + 1), ind; std::vector<double> val;
  for (int i = 0; i < n; ++i) {
    ptr[i] = ind.size();
    if (i > 0) { ind.push_back(i - 1); val.push_back(-s); }
    ind.push_back(i); val.push_back(4 * s);
    if (i < n - 1) { ind.push_back(i + 1); val.push_back(-s); }
  }
  ptr[n] = ind.size();
  return CSRMatrix<double, int>(n, ptr.data(), ind.data(), val.data());
}

template <typename S> static void run(const char* solver, const char* name, double s, int nanpos, KrylovSolver ks) {
  const int n = 2000;
  auto A = tri(n, s);
  DenseMatrix<double> xe(n, 1), b(n, 1), x(n, 1);
  for (int i = 0; i < n; ++i) xe(i, 0) = 1.0;
  A.spmv(xe.data(), b.data());
  if (nanpos >= 0) b(nanpos, 0) = NAN;
  S sp(false);
  sp.options().set_Krylov_solver(ks);
  sp.options().set_verbose(false);
  sp.set_matrix(A);
  sp.reorder();
  sp.factor();
  auto rc = sp.solve(b, x);
  int nonfin = 0, zero = 0; double err = 0;
  for (int i = 0; i < n; ++i) { nonfin += !std::isfinite(x(i, 0)); zero += x(i, 0) == 0.0; err = std::fmax(err, std::fabs(x(i, 0) - 1.0)); }
  bool silent = rc == ReturnCode::SUCCESS && (nonfin || zero == n || !(err <= 1e-5));  // 1e-5: default rel_tol is 1e-6
  std::printf("%-22s %-30s scale=%-7.0e rc=%s  max|x-1|=%-9.3g #zero(x)=%4d #nonfinite(x)=%4d -> %s\n", solver, name, s,
              rc == ReturnCode::SUCCESS ? "SUCCESS" : "other", err, zero, nonfin,
              silent ? (zero == n ? "SILENT WRONG: x = 0, SUCCESS" : "SILENT WRONG (SUCCESS)") : "ok (within default rel_tol 1e-6)");
}

int main() {
  using MP = SparseSolverMixedPrecision<float, double, int>;
  using SD = SparseSolver<double, int>;
  std::printf("STRUMPACK edc9a72, tridiag(-1,4,-1) n=2000, x_exact=1, default options except Krylov solver\n");
  for (double s : {1.0, 1e-6, 1e-12, 1e-20}) run<MP>("Mixed<float,double>", "REFINE", s, -1, KrylovSolver::REFINE);
  for (double s : {1.0, 1e-12}) run<MP>("Mixed<float,double>", "PREC_GMRES", s, -1, KrylovSolver::PREC_GMRES);
  run<MP>("Mixed<float,double>", "REFINE, b[0]=NaN", 1.0, 0, KrylovSolver::REFINE);
  for (double s : {1.0, 1e-12, 1e-20}) run<SD>("SparseSolver<double>", "AUTO (default)", s, -1, KrylovSolver::AUTO);
  run<SD>("SparseSolver<double>", "AUTO, b[0]=NaN", 1.0, 0, KrylovSolver::AUTO);
  return 0;
}

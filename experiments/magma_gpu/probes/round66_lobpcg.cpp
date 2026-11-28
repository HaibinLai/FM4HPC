// Round 66 (roxy): sparse LOBPCG (magma_zlobpcg) on Hermitian matrices, against dense LAPACK zheevd.
//   round66_lobpcg <g> <k> <theta_scale> <precond NONE|JACOBI>
//   A: g*g 2-D magnetic Laplacian, off-diagonals -exp(+-i theta_e), theta_e = theta_scale * (0.37 (lo%7) + 0.11 (hi%5)),
//      diagonal 4 (theta_scale 0 gives the real 2-D Laplacian, a control).  k smallest eigenpairs requested (--ev k).
//   The call follows the library's own path: magma_zparse_opts, magma_zeigensolverinfo_init, magma_zlobpcg.
//   ew  = max_j |lambda_j - lambda_ref_j| / max|lambda_ref|;  res = max_j |A v_j - lambda_j v_j| / (|A| |v_j|)
// CSV: r66,g,n,k,theta,precond,info,iters,ew,res,init_res,final_res,verdict  (R66_DBG: per-vector residuals on stderr)
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
typedef std::complex<double> cd;
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 5) { printf("usage: see source\n"); return 1; }
  int g = atoi(argv[1]), k = atoi(argv[2]); double ts = atof(argv[3]); std::string pc = argv[4]; int n = g * g;
  magma_init(); magma_queue_t Q; magma_queue_create(0, &Q);
  // CSR
  std::vector<int> row(1, 0), col; std::vector<cd> val; std::vector<cd> D((size_t)n * n, 0.0);
  for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) { int i = y * g + x; std::vector<std::pair<int, cd>> e;
    auto nb = [&](int xx, int yy) { if (xx < 0 || yy < 0 || xx >= g || yy >= g) return; int j = yy * g + xx; int lo = std::min(i, j), hi = std::max(i, j);
      double t = ts * (0.37 * (lo % 7) + 0.11 * (hi % 5)); e.push_back({j, -std::polar(1.0, i < j ? t : -t)}); };
    nb(x - 1, y); nb(x + 1, y); nb(x, y - 1); nb(x, y + 1); e.push_back({i, cd(4.0, 0)});
    std::sort(e.begin(), e.end(), [](auto& a, auto& b) { return a.first < b.first; });
    for (auto& p : e) { col.push_back(p.first); val.push_back(p.second); D[i + (size_t)p.first * n] = p.second; } row.push_back((int)col.size()); }
  // reference eigenvalues
  std::vector<double> wr(n); { std::vector<magmaDoubleComplex> Dz((size_t)n * n); for (size_t i = 0; i < Dz.size(); i++) Dz[i] = MAGMA_Z_MAKE(D[i].real(), D[i].imag());
    magma_int_t N = n, lw = -1, lrw = -1, liw = -1, iq, info; magmaDoubleComplex wq; double rq;
    lapackf77_zheevd("N", "L", &N, Dz.data(), &N, wr.data(), &wq, &lw, &rq, &lrw, &iq, &liw, &info);
    lw = (magma_int_t)MAGMA_Z_REAL(wq); lrw = (magma_int_t)rq; liw = iq; std::vector<magmaDoubleComplex> w(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw);
    lapackf77_zheevd("N", "L", &N, Dz.data(), &N, wr.data(), w.data(), &lw, rw.data(), &lrw, iw.data(), &liw, &info); }
  // MAGMA sparse matrix on the device
  std::vector<magma_index_t> hrow(row.begin(), row.end()), hcol(col.begin(), col.end()); std::vector<magmaDoubleComplex> hval(val.size());
  for (size_t i = 0; i < val.size(); i++) hval[i] = MAGMA_Z_MAKE(val[i].real(), val[i].imag());
  magma_z_matrix hA = {Magma_CSR}, dA = {Magma_CSR};
  magma_zcsrset(n, n, hrow.data(), hcol.data(), hval.data(), &hA, Q); magma_zmtransfer(hA, &dA, Magma_CPU, Magma_DEV, Q);
  std::string kk = std::to_string(k);
  std::vector<std::string> args = {"probe", "--solver", "LOBPCG", "--ev", kk, "--precond", pc, "--maxiter", "1000", "--rtol", "1e-8"};
  std::vector<char*> av; for (auto& a : args) av.push_back((char*)a.c_str());
  magma_zopts o; memset(&o, 0, sizeof o); int ai = 1; magma_zparse_opts((int)av.size(), av.data(), &o, &ai, Q);
  o.solver_par.ev_length = n; magma_zeigensolverinfo_init(&o.solver_par, Q);
  o.precond_par.solver = pc == "JACOBI" ? Magma_JACOBI : Magma_NONE;
  magma_z_matrix db = {Magma_CSR}; magma_zvinit(&db, Magma_DEV, n, 1, MAGMA_Z_ONE, Q);
  if (pc != "NONE") magma_z_precondsetup(dA, db, &o.solver_par, &o.precond_par, Q);
  magma_int_t info = magma_zlobpcg(dA, &o.solver_par, &o.precond_par, Q);
  // results
  std::vector<magmaDoubleComplex> V((size_t)n * k); magma_zgetmatrix(n, k, o.solver_par.eigenvectors, n, V.data(), n, Q);
  double am = 0; for (auto& v : val) am = std::max(am, std::abs(v)); double lm = 0; for (int j = 0; j < k; j++) lm = std::max(lm, fabs(wr[j]));
  std::vector<double> lam(o.solver_par.eigenvalues, o.solver_par.eigenvalues + k), lsort = lam; std::sort(lsort.begin(), lsort.end());
  double ew = 0; for (int j = 0; j < k; j++) ew = std::max(ew, fabs(lsort[j] - wr[j]) / lm);
  double res = 0; for (int j = 0; j < k; j++) { double vn = 0, r = 0; std::vector<cd> v(n); for (int i = 0; i < n; i++) { v[i] = cd(MAGMA_Z_REAL(V[i + (size_t)j * n]), MAGMA_Z_IMAG(V[i + (size_t)j * n])); vn = std::max(vn, std::abs(v[i])); }
    for (int i = 0; i < n; i++) { cd s = 0; for (int p = row[i]; p < row[i + 1]; p++) s += val[p] * v[col[p]]; r = std::max(r, std::abs(s - lam[j] * v[i])); } res = std::max(res, r / (am * vn)); if (getenv("R66_DBG")) fprintf(stderr, "vec %d lambda %.12g ref %.12g res %.3g\n", j, lam[j], wr[j], r / (am * vn)); }
  bool ok = info == 0 && ew < 1e-6 && res < 1e-6;
  printf("r66,%d,%d,%d,%g,%s,%lld,%lld,%.3g,%.3g,%.3g,%.3g,%s\n", g, n, k, ts, pc.c_str(), (long long)info, (long long)o.solver_par.numiter, ew, res, o.solver_par.init_res, o.solver_par.final_res, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

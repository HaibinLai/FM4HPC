// Seventh GPU round: findings of the second sparse audit (control / blas / precond) and the dense QR / eig / tridiagonal
// audit (HANDOFF §29.7).  GPU 0 only.  One mode per process (several modes may fault or hang).
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z; typedef std::complex<double> cd;
static magma_queue_t Q;
static Z C(double re, double im = 0) { return MAGMA_Z_MAKE(re, im); }
static void cuda_state(const char* w) { cudaError_t a = cudaDeviceSynchronize(), b = cudaGetLastError();
  printf("  [%s] cudaDeviceSynchronize: %s | cudaGetLastError: %s\n", w, cudaGetErrorString(a), cudaGetErrorString(b)); }

// dense row-major n x m (complex) -> host CSR
static magma_z_matrix csr_host(int n, int m, const std::vector<cd>& A) {
  static std::vector<std::vector<magma_index_t>> keepR, keepC; static std::vector<std::vector<Z>> keepV;
  std::vector<magma_index_t> row(n + 1), col; std::vector<Z> val;
  for (int i = 0; i < n; i++) { row[i] = (magma_index_t)col.size(); for (int j = 0; j < m; j++) if (A[(size_t)i * m + j] != cd(0, 0)) {
    col.push_back(j); val.push_back(C(A[(size_t)i * m + j].real(), A[(size_t)i * m + j].imag())); } }
  row[n] = (magma_index_t)col.size(); keepR.push_back(row); keepC.push_back(col); keepV.push_back(val);
  magma_z_matrix h = {Magma_CSR}; magma_zcsrset(n, m, keepR.back().data(), keepC.back().data(), keepV.back().data(), &h, Q); return h;
}
static magma_z_matrix to_dev(magma_z_matrix h) { magma_z_matrix d = {Magma_CSR}; magma_zmtransfer(h, &d, Magma_CPU, Magma_DEV, Q); return d; }
static magma_z_matrix dvec(int n, const std::vector<cd>& v) { std::vector<Z> b(n); for (int i = 0; i < n; i++) b[i] = C(v[i].real(), v[i].imag());
  magma_z_matrix h = {Magma_CSR}, d = {Magma_CSR}; magma_zvset(n, 1, b.data(), &h, Q); magma_zmtransfer(h, &d, Magma_CPU, Magma_DEV, Q); return d; }
static std::vector<cd> hvec(magma_z_matrix d) { magma_z_matrix h = {Magma_CSR}; magma_zmtransfer(d, &h, Magma_DEV, Magma_CPU, Q);
  std::vector<cd> v(h.num_rows * h.num_cols); for (size_t i = 0; i < v.size(); i++) v[i] = cd(MAGMA_Z_REAL(h.val[i]), MAGMA_Z_IMAG(h.val[i])); return v; }
static void pv(const char* nm, const std::vector<cd>& v, int k) { printf("  %s =", nm); for (int i = 0; i < k && i < (int)v.size(); i++)
  printf(" (%g%+gi)", v[i].real(), v[i].imag()); printf("\n"); }
static double trueres(int n, const std::vector<cd>& A, const std::vector<cd>& b, const std::vector<cd>& x) {
  double r = 0, nb = 0; bool fin = true; for (int i = 0; i < n; i++) { cd s = b[i]; for (int j = 0; j < n; j++) s -= A[(size_t)i * n + j] * x[j];
    r += std::norm(s); nb += std::norm(b[i]); if (!std::isfinite(x[i].real())) fin = false; } return fin ? sqrt(r / nb) : INFINITY; }

// precond + Krylov solve through the public wrapper
static void solve(const char* tag, int n, const std::vector<cd>& A, const std::vector<cd>& b, magma_solver_type sol,
                  magma_solver_type pre, magma_solver_type tri, int levels) {
  magma_z_matrix dA = to_dev(csr_host(n, n, A)), db = dvec(n, b), dx = dvec(n, std::vector<cd>(n, 0));
  magma_zopts o; memset(&o, 0, sizeof o);
  o.solver_par.solver = sol; o.solver_par.maxiter = 1000; o.solver_par.atol = 1e-16; o.solver_par.rtol = 1e-10; o.solver_par.restart = 30;
  o.precond_par.solver = pre; o.precond_par.trisolver = tri; o.precond_par.levels = levels; o.precond_par.maxiter = 3; o.precond_par.sweeps = 5;
  magma_zsolverinfo_init(&o.solver_par, &o.precond_par, Q);
  magma_int_t ps = magma_z_precondsetup(dA, db, &o.solver_par, &o.precond_par, Q);
  magma_int_t r = magma_z_solver(dA, db, &dx, &o, Q);
  std::vector<cd> x = hvec(dx);
  printf("%-40s precondsetup=%lld solver ret=%lld info=%lld numiter=%lld  true_rel_res=%.3g\n", tag, (long long)ps, (long long)r,
         (long long)o.solver_par.info, (long long)o.solver_par.numiter, trueres(n, A, b, x));
  if (tri == Magma_SYNCFREESOLVE && o.precond_par.L.drow) {      // CSC expected: drow = column pointers (n+1, nondecreasing, last = nnz)
    int nn = o.precond_par.L.num_rows; std::vector<magma_index_t> r(nn + 1), c(o.precond_par.L.nnz);
    magma_index_getvector(nn + 1, o.precond_par.L.drow, 1, r.data(), 1, Q); magma_index_getvector(o.precond_par.L.nnz, o.precond_par.L.dcol, 1, c.data(), 1, Q);
    int mono = 1; for (int i = 0; i < nn; i++) if (r[i] > r[i + 1]) mono = 0;
    printf("  L after setup: nnz=%lld  drow[0..7] =", (long long)o.precond_par.L.nnz); for (int i = 0; i < 8; i++) printf(" %d", (int)r[i]);
    printf("  drow[n]=%d nondecreasing=%d | dcol[0..7] =", (int)r[nn], mono); for (int i = 0; i < 8; i++) printf(" %d", (int)c[i]); printf("\n");
  }
  magma_zprecondfree(&o.precond_par, Q);
}
static std::vector<cd> tridiag(int n, cd lo, cd d, cd up) { std::vector<cd> A((size_t)n * n, 0);
  for (int i = 0; i < n; i++) { A[(size_t)i * n + i] = d; if (i) A[(size_t)i * n + i - 1] = lo; if (i + 1 < n) A[(size_t)i * n + i + 1] = up; } return A; }
static std::vector<cd> rhs(int n, const std::vector<cd>& A) { std::vector<cd> b(n, 0); for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) b[i] += A[(size_t)i * n + j] * cd(1 + j % 3, 0); return b; }

static void syncfree() {
  int n = 64; auto A = tridiag(n, -1, 4, -1); auto b = rhs(n, A);
  solve("ILU0 + PBICGSTAB, trisolver CUSOLVE", n, A, b, Magma_PBICGSTAB, Magma_ILU, Magma_CUSOLVE, 0);
  solve("ILU0 + PBICGSTAB, trisolver SYNCFREESOLVE", n, A, b, Magma_PBICGSTAB, Magma_ILU, Magma_SYNCFREESOLVE, 0);
  cuda_state("after syncfree");
}
static void icc_complex() {
  int n = 64;
  { auto A = tridiag(n, -1, 4, -1); auto b = rhs(n, A); solve("ICC + PCG, real tridiag(-1,4,-1)", n, A, b, Magma_PCG, Magma_ICC, Magma_CUSOLVE, 0); }
  { auto A = tridiag(n, cd(0, -1), 4, cd(0, 1)); auto b = rhs(n, A); solve("ICC + PCG, Hermitian tridiag(-i,4,i)", n, A, b, Magma_PCG, Magma_ICC, Magma_CUSOLVE, 0); }
  { auto A = tridiag(n, cd(0, -1), 4, cd(0, 1)); auto b = rhs(n, A); solve("CG (no precond), Hermitian tridiag(-i,4,i)", n, A, b, Magma_CG, Magma_NONE, Magma_CUSOLVE, 0); }
}
static void symbilu() {
  int n = 200; std::vector<cd> A((size_t)n * n, 0); for (int i = 0; i < n; i++) { A[(size_t)i * n + i] = 2.0 * n; A[i] = 1; A[(size_t)i * n] = 1; }
  A[0] = 2.0 * n; auto b = rhs(n, A);
  solve("ILU(0) + PBICGSTAB, arrowhead n=200", n, A, b, Magma_PBICGSTAB, Magma_ILU, Magma_CUSOLVE, 0);
  solve("ILU(1) + PBICGSTAB, arrowhead n=200", n, A, b, Magma_PBICGSTAB, Magma_ILU, Magma_CUSOLVE, 1);
}
static void spmv_cases() {
  // (a) dimension mismatch: A 4x4, x 3x1
  { auto A = tridiag(4, -1, 4, -1); magma_z_matrix dA = to_dev(csr_host(4, 4, A)), dxx = dvec(3, {1, 1, 1}), dy = dvec(4, {7, 7, 7, 7});
    magma_int_t r = magma_z_spmv(C(1), dA, dxx, C(0), dy, Q); printf("spmv A 4x4, x 3x1: ret=%lld\n", (long long)r); pv("y (initialised to 7)", hvec(dy), 4); }
  std::vector<cd> A2 = {1, 2, 0, 3};    // [[1,2],[0,3]], x = (1,1): A x = (3,3), A^T x = (1,5)
  // (b) CSR -> DENSE (CPU), then SpMV on the device
  { magma_z_matrix h = csr_host(2, 2, A2), hd = {Magma_CSR}, dd = {Magma_CSR}; magma_int_t c = magma_zmconvert(h, &hd, Magma_CSR, Magma_DENSE, Q);
    magma_zmtransfer(hd, &dd, Magma_CPU, Magma_DEV, Q); magma_z_matrix dxx = dvec(2, {1, 1}), dy = dvec(2, {7, 7});
    magma_int_t r = magma_z_spmv(C(1), dd, dxx, C(0), dy, Q); printf("spmv DENSE [[1,2],[0,3]] x=(1,1): convert=%lld ret=%lld (expected y = 3 3)\n", (long long)c, (long long)r); pv("y", hvec(dy), 2); }
  // (c) CSR -> CSC on the device, then SpMV
  { magma_z_matrix d = to_dev(csr_host(2, 2, A2)), dc = {Magma_CSR}; magma_int_t c = magma_zmconvert(d, &dc, Magma_CSR, Magma_CSC, Q);
    magma_z_matrix dxx = dvec(2, {1, 1}), dy = dvec(2, {7, 7});
    magma_int_t r = magma_z_spmv(C(1), dc, dxx, C(0), dy, Q); printf("spmv CSC  [[1,2],[0,3]] x=(1,1): convert=%lld ret=%lld (expected y = 3 3)\n", (long long)c, (long long)r); pv("y", hvec(dy), 2); }
  cuda_state("after CSC spmv");
  // (d) SELL-P, alignment 1, tall 300 x 1 matrix of ones
  { int n = 300; std::vector<cd> A(n, 1.0); magma_z_matrix h = csr_host(n, 1, A), hs = {Magma_CSR}, ds = {Magma_CSR};
    hs.blocksize = 32; hs.alignment = 1; magma_int_t c = magma_zmconvert(h, &hs, Magma_CSR, Magma_SELLP, Q);
    magma_zmtransfer(hs, &ds, Magma_CPU, Magma_DEV, Q); magma_z_matrix dxx = dvec(1, {1}), dy = dvec(n, std::vector<cd>(n, 7));
    magma_int_t r = magma_z_spmv(C(1), ds, dxx, C(0), dy, Q); auto y = hvec(dy); int bad = 0, first = -1;
    for (int i = 0; i < n; i++) if (y[i] != cd(1, 0)) { bad++; if (first < 0) first = i; }
    printf("spmv SELLP (blocksize 32, alignment 1) 300x1 ones, x=(1): convert=%lld ret=%lld  rows with y != 1: %d (first %d, value %g)\n",
           (long long)c, (long long)r, bad, first, first >= 0 ? y[first].real() : 0.0); }
}
static void mscale() {
  std::vector<cd> A = {-4, 1, 1, -4}; magma_z_matrix d = to_dev(csr_host(2, 2, A));
  magma_int_t r = magma_zmscale(&d, Magma_UNITDIAG, Q); magma_z_matrix h = {Magma_CSR}; magma_zmtransfer(d, &h, d.memory_location, Magma_CPU, Q);
  printf("zmscale UNITDIAG on [[-4,1],[1,-4]]: ret=%lld  values:", (long long)r); for (int i = 0; i < h.nnz; i++) printf(" %g", MAGMA_Z_REAL(h.val[i])); printf("\n");
  std::vector<cd> B = {cd(0, 2), 1, 1, cd(0, 2)}; magma_z_matrix e = to_dev(csr_host(2, 2, B));
  r = magma_zmscale(&e, Magma_UNITDIAG, Q); magma_z_matrix g = {Magma_CSR}; magma_zmtransfer(e, &g, e.memory_location, Magma_CPU, Q);
  printf("zmscale UNITDIAG on [[2i,1],[1,2i]]: ret=%lld  values:", (long long)r); for (int i = 0; i < g.nnz; i++) printf(" (%g%+gi)", MAGMA_Z_REAL(g.val[i]), MAGMA_Z_IMAG(g.val[i])); printf("\n");
}
// ---------------- dense ----------------
static unsigned S = 7u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static void sygvdx() {
  int n = 50; std::vector<double> A((size_t)n * n), B((size_t)n * n, 0), w(n), Al;
  for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { double v = rnd(); A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = v; }
  for (int i = 0; i < n; i++) { A[i + (size_t)i * n] += i; B[i + (size_t)i * n] = 1; }
  Al = A; { magma_int_t nn = n, lw = -1, info; double q; lapackf77_dsyev("N", "L", &nn, Al.data(), &nn, w.data(), &q, &lw, &info); lw = (magma_int_t)q;
    std::vector<double> wk(lw); lapackf77_dsyev("N", "L", &nn, Al.data(), &nn, w.data(), wk.data(), &lw, &info); }
  double lmax = w[n - 1], lmin = w[0];
  for (int mode = 0; mode < 2; mode++) {
    std::vector<double> A1 = A, B1 = B, ww(n, NAN); magma_int_t m = -1, info = -1, lwork = -1, liwork = -1; double q; magma_int_t iq;
    magma_range_t rg = mode == 0 ? MagmaRangeI : MagmaRangeV; double vl = lmax - 1e-6, vu = lmax + 1e-6;
    magma_dsygvdx(1, MagmaVec, rg, MagmaLower, n, A1.data(), n, B1.data(), n, vl, vu, n, n, &m, ww.data(), &q, lwork, &iq, liwork, &info);
    lwork = (magma_int_t)q; liwork = iq; std::vector<double> wk(lwork); std::vector<magma_int_t> iw(liwork);
    magma_dsygvdx(1, MagmaVec, rg, MagmaLower, n, A1.data(), n, B1.data(), n, vl, vu, n, n, &m, ww.data(), wk.data(), lwork, iw.data(), liwork, &info);
    printf("dsygvdx n=50 B=I, %s: info=%lld m=%lld w[0]=%.6g (requested: m=1, w[0]=lambda_max=%.6g; lambda_min=%.6g)\n",
           mode == 0 ? "RangeI il=iu=50" : "RangeV (lmax-1e-6, lmax+1e-6]", (long long)info, (long long)m, ww[0], lmax, lmin);
  }
}
static void geqp3(bool swap) {
  int m = 64, n = 64; std::vector<double> A((size_t)m * n); for (auto& a : A) a = rnd();
  double *dA, *dwork, *dtau; magma_dmalloc(&dA, (size_t)m * n); magma_dmalloc(&dtau, n); std::vector<magma_int_t> jp(n, 0);
  if (swap) jp[1] = 1; else jp[0] = 1;
  magma_int_t nb = magma_get_dgeqp3_nb(m, n), ldoc = (n + 1) * nb + 2 * n, info = -999;
  // the documented LWORK first (the wrapper rejects it: it also needs df and dlsticcs)
  magma_dmalloc(&dwork, 4 * ldoc + 1024); magma_dsetmatrix(m, n, A.data(), m, dA, m, Q);
  { std::vector<magma_int_t> j2(n, 0); magma_dgeqp3_gpu(m, n, dA, m, j2.data(), dtau, dwork, ldoc, &info);
    printf("dgeqp3_gpu with the documented LWORK = (N+1)*NB + 2*N = %lld: info=%lld\n", (long long)ldoc, (long long)info); }
  magma_int_t lwork = 4 * ldoc + 1024; magma_dsetmatrix(m, n, A.data(), m, dA, m, Q); std::vector<double> t0(n, 12345.0); magma_dsetvector(n, t0.data(), 1, dtau, 1, Q);
  printf("calling dgeqp3_gpu with jpvt[%d] = 1, LWORK = %lld ...\n", swap ? 1 : 0, (long long)lwork); fflush(stdout);
  info = -999; magma_dgeqp3_gpu(m, n, dA, m, jp.data(), dtau, dwork, lwork, &info);
  std::vector<double> F((size_t)m * n), tau(n); magma_dgetmatrix(m, n, dA, m, F.data(), m, Q); magma_dgetvector(n, dtau, 1, tau.data(), 1, Q);
  int same = 0; for (int i = 0; i < m; i++) if (F[i] == A[i]) same++;
  double c0 = 0; for (int i = 0; i < m; i++) c0 += A[i] * A[i];
  // control: no fixed columns
  std::vector<magma_int_t> jf(n, 0); magma_dsetmatrix(m, n, A.data(), m, dA, m, Q); magma_int_t inf2; magma_dgeqp3_gpu(m, n, dA, m, jf.data(), dtau, dwork, lwork, &inf2);
  std::vector<double> G((size_t)m * n); magma_dgetmatrix(m, n, dA, m, G.data(), m, Q); double cj = 0; for (int i = 0; i < m; i++) cj += A[i + (size_t)(jf[0] - 1) * m] * A[i + (size_t)(jf[0] - 1) * m];
  printf("info=%lld  jpvt[0]=%lld  output column 0: %d of %d entries identical to the input column; |R(0,0)|=%.6g vs ||A(:,0)||=%.6g; tau[0]=%g\n",
         (long long)info, (long long)jp[0], same, m, fabs(F[0]), sqrt(c0), tau[0]);
  printf("control (no fixed column): info=%lld  |R(0,0)|=%.6g vs ||A(:,jpvt[0])||=%.6g\n", (long long)inf2, fabs(G[0]), sqrt(cj));
}
static void gels() {
  int n = 256, nrhs = 1; std::vector<double> A((size_t)n * n), b(n);
  for (auto& a : A) a = rnd(); for (int i = 0; i < n; i++) A[i + (size_t)i * n] += 4; for (int i = 0; i < n; i++) A[i] = 0;   // column 0 = 0
  for (auto& v : b) v = rnd();
  { std::vector<double> Al = A, bl = b; magma_int_t nn = n, one = 1, lw = -1, info; double q;
    lapackf77_dgels("N", &nn, &nn, &one, Al.data(), &nn, bl.data(), &nn, &q, &lw, &info); lw = (magma_int_t)q; std::vector<double> wk(lw);
    lapackf77_dgels("N", &nn, &nn, &one, Al.data(), &nn, bl.data(), &nn, wk.data(), &lw, &info); printf("LAPACK dgels, column 0 = 0: info=%lld\n", (long long)info); }
  double *dA, *dB; magma_dmalloc(&dA, (size_t)n * n); magma_dmalloc(&dB, n); magma_dsetmatrix(n, n, A.data(), n, dA, n, Q); magma_dsetmatrix(n, 1, b.data(), n, dB, n, Q);
  magma_int_t nb = magma_get_dgeqrf_nb(n, n), lwork = (n - n + nb) * (nrhs + nb) + nrhs * nb; std::vector<double> hw(lwork); magma_int_t info = -999;
  magma_dgels_gpu(MagmaNoTrans, n, n, nrhs, dA, n, dB, n, hw.data(), lwork, &info);
  std::vector<double> x(n); magma_dgetmatrix(n, 1, dB, n, x.data(), n, Q);
  int fin = 1; double xm = 0, rr = 0, bn = 0; for (int i = 0; i < n; i++) { if (!std::isfinite(x[i])) fin = 0; xm = fmax(xm, fabs(x[i])); }
  for (int i = 0; i < n; i++) { double s = b[i]; for (int j = 0; j < n; j++) s -= A[i + (size_t)j * n] * x[j]; rr += s * s; bn += b[i] * b[i]; }
  // least-squares optimality: A^T r should be ~0
  double atr = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < n; i++) { double r = b[i]; for (int k = 0; k < n; k++) r -= A[i + (size_t)k * n] * x[k]; s += A[i + (size_t)j * n] * r; } atr = fmax(atr, fabs(s)); }
  printf("magma_dgels_gpu (nb=%lld), column 0 = 0: info=%lld  x finite=%d  max|x|=%.3g  ||b-Ax||/||b||=%.3g  max|A^T r|=%.3g\n", (long long)nb,
         (long long)info, fin, xm, sqrt(rr / bn), atr);
}
static void stedx() {
  int n = 200; std::vector<double> d(n), e(n - 1, 0.5); for (int i = 0; i < n; i++) d[i] = 1 + (i % 7) / 7.0; d[3] = NAN; e[9] = 0;
  std::vector<double> Z((size_t)n * n); magma_int_t lrwork = 1 + 4 * n + 2 * n * n, liwork = 3 + 5 * n; std::vector<double> rw(lrwork); std::vector<magma_int_t> iw(liwork);
  double* dw; magma_dmalloc(&dw, 3 * n * n / 2 + 3 * n); magma_int_t info = -999;
  { std::vector<double> d1 = d, e1 = e, Z1((size_t)10 * 10), w1(2 * 10); magma_int_t m = 10, info1; lapackf77_dsteqr("I", &m, d1.data(), e1.data(), Z1.data(), &m, w1.data(), &info1);
    printf("LAPACK dsteqr on the first block (rows 0..9, d[3]=NaN): info=%lld\n", (long long)info1); }
  magma_dstedx(MagmaRangeAll, n, 0, 0, 0, 0, d.data(), e.data(), Z.data(), n, rw.data(), lrwork, iw.data(), liwork, dw, &info);
  int nf = 0; for (double v : d) if (!std::isfinite(v)) nf++;
  printf("magma_dstedx n=200, d[3]=NaN, split at e[9]=0: info=%lld  non-finite eigenvalues=%d\n", (long long)info, nf);
}
static void lag2s() {
  int m = 8; std::vector<double> A(m * m, 0.5); A[9] = NAN; double* dA; float* dS; magma_dmalloc(&dA, m * m); magma_smalloc(&dS, m * m);
  magma_dsetmatrix(m, m, A.data(), m, dA, m, Q); magma_int_t info = -999; magmablas_dlag2s(m, m, dA, m, dS, m, Q, &info);
  printf("magmablas_dlag2s with one NaN: info=%lld (documented: 1 if an entry is outside the FP32 range; Inf is flagged)\n", (long long)info);
  A[9] = INFINITY; magma_dsetmatrix(m, m, A.data(), m, dA, m, Q); info = -999; magmablas_dlag2s(m, m, dA, m, dS, m, Q, &info);
  printf("magmablas_dlag2s with one Inf: info=%lld\n", (long long)info);
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : "";
  magma_init(); magma_queue_create(0, &Q);
  if (m == "syncfree") syncfree(); else if (m == "icc") icc_complex(); else if (m == "symbilu") symbilu(); else if (m == "spmv") spmv_cases();
  else if (m == "mscale") mscale(); else if (m == "sygvdx") sygvdx(); else if (m == "geqp3") geqp3(false); else if (m == "geqp3_swap") geqp3(true);
  else if (m == "gels") gels(); else if (m == "stedx") stedx(); else if (m == "lag2s") lag2s();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

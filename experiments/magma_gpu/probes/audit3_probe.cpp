// Third GPU round: audit items 10 (gesvj_batched), 11 (getrf_m info), 15 (hgetrf singular panel) and a parameter scan for
// the FGMRES sticky-success sequence (MagmaSparse.fgmres_sticky_success).  GPU 0 only.
// Modes: gesvj, getrfm (run with MAGMA_NGR_NB set), hgetrf, fgmres.
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
static magma_queue_t Q;
typedef magmaDoubleComplex Z;
static Z R(double x) { return MAGMA_Z_MAKE(x, 0); }
static unsigned S = 31337u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }

static void gesvj() {
  printf("== gesvj_batched: one NaN entry, batch of 2 (matrix 0 has the NaN, matrix 1 is clean) ==\n");
  for (int n : {16, 64, 256}) {
    int m = n, batch = 2; std::vector<std::vector<Z>> A(batch, std::vector<Z>((size_t)m * n));
    for (auto& M : A) for (auto& a : M) a = R(rnd()); A[0][3 + 5 * (size_t)m] = R(NAN);
    std::vector<Z*> dA(batch), dU(batch), dV(batch); std::vector<double*> dS(batch);
    for (int b = 0; b < batch; b++) { magma_zmalloc(&dA[b], (size_t)m * n); magma_zmalloc(&dU[b], (size_t)m * n); magma_zmalloc(&dV[b], (size_t)n * n);
      magma_dmalloc(&dS[b], n); magma_zsetmatrix(m, n, A[b].data(), m, dA[b], m, Q); }
    Z **dAa, **dUa, **dVa; double** dSa; magma_int_t* dinfo;
    magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dUa, batch * sizeof(Z*)); magma_malloc((void**)&dVa, batch * sizeof(Z*));
    magma_malloc((void**)&dSa, batch * sizeof(double*)); magma_imalloc(&dinfo, batch);
    magma_setvector(batch, sizeof(Z*), dA.data(), 1, dAa, 1, Q); magma_setvector(batch, sizeof(Z*), dU.data(), 1, dUa, 1, Q);
    magma_setvector(batch, sizeof(Z*), dV.data(), 1, dVa, 1, Q); magma_setvector(batch, sizeof(double*), dS.data(), 1, dSa, 1, Q);
    magma_int_t ret = magma_zgesvj_batched(MagmaVec, MagmaVec, m, n, dAa, m, dSa, dUa, m, dVa, n, dinfo, batch, Q);
    std::vector<magma_int_t> info(batch); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, info.data(), 1, Q);
    for (int b = 0; b < batch; b++) { std::vector<double> s(n); magma_dgetvector(n, dS[b], 1, s.data(), 1, Q); int nf = 0; for (double v : s) if (!std::isfinite(v)) nf++;
      printf("n=%3d matrix %d (%s): ret=%lld info=%lld  nonfinite singular values=%d  s[0]=%g\n", n, b, b == 0 ? "NaN" : "clean", (long long)ret,
             (long long)info[b], nf, s[0]); }
    for (int b = 0; b < batch; b++) { magma_free(dA[b]); magma_free(dU[b]); magma_free(dV[b]); magma_free(dS[b]); }
    magma_free(dAa); magma_free(dUa); magma_free(dVa); magma_free(dSa); magma_free(dinfo);
  }
}

static void getrfm() {
  const char* e = getenv("MAGMA_NGR_NB");
  printf("== getrf_m (out-of-core LU, CPU interface), MAGMA_NGR_NB=%s: exactly zero column(s) ==\n", e ? e : "(unset)");
  int n = 1024;
  struct Case { std::vector<int> zeroCols; };
  for (auto zc : std::vector<std::vector<int>>{{100}, {300}, {600}, {900}, {300, 900}}) {
    std::vector<double> A((size_t)n * n); S = 11u; for (auto& a : A) a = rnd(); for (int i = 0; i < n; i++) A[i + (size_t)i * n] += 8;
    for (int c : zc) for (int i = 0; i < n; i++) A[i + (size_t)c * n] = 0;
    std::vector<double> Al = A; std::vector<magma_int_t> ip(n); magma_int_t nn = n, linfo; lapackf77_dgetrf(&nn, &nn, Al.data(), &nn, ip.data(), &linfo);
    magma_int_t info = -999; magma_dgetrf_m(1, n, n, A.data(), n, ip.data(), &info);
    printf("zero column(s) at");
    for (int c : zc) printf(" %d", c);
    printf(": LAPACK dgetrf info=%lld | magma_dgetrf_m info=%lld\n", (long long)linfo, (long long)info);
  }
}

static void hgetrf() {
  printf("== hgetrf_gpu with an exactly singular first panel (column 0 = 0), n = 1024 aligned ==\n");
  int n = 1024; std::vector<float> A((size_t)n * n); S = 21u; for (auto& a : A) a = (float)rnd(); for (int i = 0; i < n; i++) A[i + (size_t)i * n] += 4.0f;
  for (int i = 0; i < n; i++) A[i] = 0.0f;                    // column 0
  float* dA; magma_smalloc(&dA, (size_t)n * n); magma_ssetmatrix(n, n, A.data(), n, dA, n, Q);
  std::vector<magma_int_t> ip(n); magma_int_t info = -999; magma_hgetrf_gpu(n, n, dA, n, ip.data(), &info);
  std::vector<float> F((size_t)n * n); magma_sgetmatrix(n, n, dA, n, F.data(), n, Q);
  // how close is the output to A and to A^T (outside the first 128 rows/cols, which the panel may have touched)?
  double dA0 = 0, dAT = 0; long eqA = 0, eqAT = 0, cnt = 0;
  for (int j = 256; j < n; j++) for (int i = 256; i < n; i++) { cnt++; float f = F[i + (size_t)j * n];
    if (f == A[i + (size_t)j * n]) eqA++; if (f == A[j + (size_t)i * n]) eqAT++; }
  printf("info=%lld | in the trailing block (rows/cols >= 256): %ld of %ld entries equal A, %ld equal A^T\n", (long long)info, eqA, cnt, eqAT);
  magma_free(dA);
}

static void fgmres() {
  printf("== sparse FGMRES scan for 'info = 0 with a non-finite x or a large true residual' ==\n");
  long total = 0, bad = 0, nonfin = 0;
  for (int n : {4, 8, 16, 32}) for (double nb : {1e-8, 1e-10, 1e-12, 1e-14}) for (double rtol : {1e-20, 1e-30}) for (int restart : {2, 3, 5, 10})
  for (int kind = 0; kind < 3; kind++) for (int seed = 0; seed < 3; seed++) {
    S = 1000u * n + 17u * seed + kind; std::vector<double> A((size_t)n * n);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) A[i * n + j] = (kind == 0 ? 0.3 * rnd() : (kind == 1 ? rnd() : 0.0)) + (i == j ? (kind == 2 ? (i % 2 ? 1.0 : 2.0) : 1.0) : 0.0);
    std::vector<magma_index_t> row(n + 1), col; std::vector<Z> val;
    for (int i = 0; i < n; i++) { row[i] = (magma_index_t)col.size(); for (int j = 0; j < n; j++) if (A[i * n + j] != 0) { col.push_back(j); val.push_back(R(A[i * n + j])); } }
    row[n] = (magma_index_t)col.size();
    magma_z_matrix hA = {Magma_CSR}, dA = {Magma_CSR}, hb = {Magma_CSR}, db = {Magma_CSR}, hx = {Magma_CSR}, dx = {Magma_CSR};
    magma_zcsrset(n, n, row.data(), col.data(), val.data(), &hA, Q); magma_zmtransfer(hA, &dA, Magma_CPU, Magma_DEV, Q);
    std::vector<Z> b(n), x0(n, R(0)); double s2 = 0; for (auto& v : b) { double t = rnd(); v = R(t); s2 += t * t; }
    for (auto& v : b) v = R(MAGMA_Z_REAL(v) * nb / sqrt(s2));
    magma_zvset(n, 1, b.data(), &hb, Q); magma_zmtransfer(hb, &db, Magma_CPU, Magma_DEV, Q);
    magma_zvset(n, 1, x0.data(), &hx, Q); magma_zmtransfer(hx, &dx, Magma_CPU, Magma_DEV, Q);
    magma_z_solver_par sp; magma_z_preconditioner pp; memset(&sp, 0, sizeof sp); memset(&pp, 0, sizeof pp);
    sp.solver = Magma_GMRES; sp.maxiter = 200; sp.restart = restart; sp.atol = 1e-16; sp.rtol = rtol; pp.solver = Magma_NONE;
    magma_zsolverinfo_init(&sp, &pp, Q); sp.restart = restart;
    magma_zfgmres(dA, db, &dx, &sp, &pp, Q);
    magma_z_matrix ox = {Magma_CSR}; magma_zmtransfer(dx, &ox, Magma_DEV, Magma_CPU, Q);
    bool fin = true; double rr = 0; for (int i = 0; i < n; i++) { double r = MAGMA_Z_REAL(b[i]); for (int j = 0; j < n; j++) r -= A[i * n + j] * MAGMA_Z_REAL(ox.val[j]);
      rr += r * r; if (!std::isfinite(MAGMA_Z_REAL(ox.val[i]))) fin = false; }
    double rel = sqrt(rr) / nb; total++;
    if (sp.info == 0 && (!fin || rel > 1e-6)) { bad++; if (!fin) nonfin++;
      if (bad <= 12) printf("  info=0 n=%d ||b||=%g rtol=%g restart=%d kind=%d seed=%d: numiter=%lld finite(x)=%d true_res/||b||=%g final_res=%g\n",
                            n, nb, rtol, restart, kind, seed, (long long)sp.numiter, (int)fin, rel, sp.final_res); }
    magma_zmfree(&dA, Q); magma_zmfree(&db, Q); magma_zmfree(&dx, Q); magma_zmfree(&ox, Q);
  }
  printf("scanned %ld solves: %ld report info = 0 with a non-finite x or true residual > 1e-6 ||b|| (%ld non-finite)\n", total, bad, nonfin);
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  std::string m = argc > 1 ? argv[1] : "gesvj";
  magma_init(); magma_queue_create(0, &Q);
  if (m == "gesvj") gesvj(); else if (m == "getrfm") getrfm(); else if (m == "hgetrf") hgetrf(); else if (m == "fgmres") fgmres();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

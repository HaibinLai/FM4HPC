// Fifth GPU round: bpcg multi-RHS through the wrapper, zgesv_rbt_batched (info, leak), bombard residual fields,
// single-precision re-checks (c/s), and the ILU zero-pivot crash (run with LD_PRELOAD=segv_bt.so).  GPU 0 only.
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
static void segv_h(int sg) { void* b[64]; int n = backtrace(b, 64); fprintf(stderr, "[segv] signal %d, backtrace:\n", sg); backtrace_symbols_fd(b, n, 2); _exit(139); }
static magma_queue_t Q;
typedef magmaDoubleComplex Z;
static Z R(double x) { return MAGMA_Z_MAKE(x, 0); }
static unsigned S = 4711u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
extern "C" magma_int_t magma_cgbtrs_batched_strided(magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t,
    magmaFloatComplex*, magma_int_t, magma_int_t, magma_int_t*, magma_int_t, magmaFloatComplex*, magma_int_t, magma_int_t,
    magma_int_t*, magma_int_t, magma_queue_t);

static void csr_z(int n, const std::vector<double>& A, magma_z_matrix* dA) {
  std::vector<magma_index_t> row(n + 1), col; std::vector<Z> val;
  for (int i = 0; i < n; i++) { row[i] = (magma_index_t)col.size(); for (int j = 0; j < n; j++) if (A[i * n + j] != 0) { col.push_back(j); val.push_back(R(A[i * n + j])); } }
  row[n] = (magma_index_t)col.size(); magma_z_matrix h = {Magma_CSR};
  magma_zcsrset(n, n, row.data(), col.data(), val.data(), &h, Q); magma_zmtransfer(h, dA, Magma_CPU, Magma_DEV, Q);
}
static std::vector<double> tridiag(int n) { std::vector<double> A(n * n, 0); for (int i = 0; i < n; i++) { A[i * n + i] = 4; if (i) A[i * n + i - 1] = -1; if (i + 1 < n) A[i * n + i + 1] = -1; } return A; }

static void bpcg() {
  printf("== R1: magma_z_solver with a 2-column right-hand side and solver = CG (routes to zbpcg) ==\n");
  int n = 64; auto A = tridiag(n); magma_z_matrix dA = {Magma_CSR}; csr_z(n, A, &dA);
  std::vector<double> xt(2 * n); for (auto& v : xt) v = rnd();
  std::vector<Z> b(2 * n), x0(2 * n, R(0));
  for (int c = 0; c < 2; c++) for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += A[i * n + j] * xt[c * n + j]; b[c * n + i] = R(s); }
  magma_z_matrix hb = {Magma_CSR}, db = {Magma_CSR}, hx = {Magma_CSR}, dx = {Magma_CSR};
  magma_zvset(n, 2, b.data(), &hb, Q); magma_zmtransfer(hb, &db, Magma_CPU, Magma_DEV, Q);
  magma_zvset(n, 2, x0.data(), &hx, Q); magma_zmtransfer(hx, &dx, Magma_CPU, Magma_DEV, Q);
  magma_zopts o; memset(&o, 0, sizeof o); o.solver_par.solver = Magma_CG; o.solver_par.maxiter = 1000; o.solver_par.atol = 1e-16; o.solver_par.rtol = 1e-10;
  o.precond_par.solver = Magma_NONE; magma_zsolverinfo_init(&o.solver_par, &o.precond_par, Q);
  magma_int_t ret = magma_z_solver(dA, db, &dx, &o, Q);
  magma_z_matrix ox = {Magma_CSR}; magma_zmtransfer(dx, &ox, Magma_DEV, Magma_CPU, Q);
  for (int c = 0; c < 2; c++) { double e = 0, m = 0; for (int i = 0; i < n; i++) { e = fmax(e, fabs(MAGMA_Z_REAL(ox.val[c * n + i]) - xt[c * n + i])); m = fmax(m, fabs(xt[c * n + i])); }
    printf("column %d: relerr=%.3g\n", c, e / m); }
  printf("ret=%lld info=%lld numiter=%lld (b.num_rows=%lld b.num_cols=%lld -> zbpcg num_vecs = b.num_rows/A.num_rows = %lld)\n", (long long)ret,
         (long long)o.solver_par.info, (long long)o.solver_par.numiter, (long long)db.num_rows, (long long)db.num_cols, (long long)(db.num_rows / n));
}

static void rbtb() {
  printf("== R2: zgesv_rbt_batched ==\n");
  int n = 64, batch = 4; std::vector<Z*> dA(batch), dB(batch); std::vector<std::vector<double>> xt(batch, std::vector<double>(n));
  for (int b = 0; b < batch; b++) { std::vector<Z> A(n * n), B(n); for (auto& a : A) a = R(rnd()); for (int i = 0; i < n; i++) A[i + i * n] = R(MAGMA_Z_REAL(A[i + i * n]) + 8);
    if (b == 2) for (int i = 0; i < n; i++) A[i + 7 * n] = R(0);        // matrix 2: exactly singular (zero column)
    for (int i = 0; i < n; i++) xt[b][i] = rnd();
    for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += MAGMA_Z_REAL(A[i + j * n]) * xt[b][j]; B[i] = R(s); }
    magma_zmalloc(&dA[b], n * n); magma_zmalloc(&dB[b], n); magma_zsetmatrix(n, n, A.data(), n, dA[b], n, Q); magma_zsetvector(n, B.data(), 1, dB[b], 1, Q); }
  Z **dAa, **dBa; magma_int_t* dinfo; magma_malloc((void**)&dAa, batch * sizeof(Z*)); magma_malloc((void**)&dBa, batch * sizeof(Z*)); magma_imalloc(&dinfo, batch);
  magma_setvector(batch, sizeof(Z*), dA.data(), 1, dAa, 1, Q); magma_setvector(batch, sizeof(Z*), dB.data(), 1, dBa, 1, Q);
  magma_int_t ret = magma_zgesv_rbt_batched(n, 1, dAa, n, dBa, n, dinfo, batch, Q);
  std::vector<magma_int_t> info(batch); magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, info.data(), 1, Q);
  for (int b = 0; b < batch; b++) { std::vector<Z> x(n); magma_zgetvector(n, dB[b], 1, x.data(), 1, Q); int nf = 0; double e = 0, m = 0;
    for (int i = 0; i < n; i++) { if (!std::isfinite(MAGMA_Z_REAL(x[i]))) nf++; else e = fmax(e, fabs(MAGMA_Z_REAL(x[i]) - xt[b][i])); m = fmax(m, fabs(xt[b][i])); }
    printf("matrix %d%s: return=%lld info_array=%lld nonfinite=%d relerr(finite part)=%.2e\n", b, b == 2 ? " (singular)" : "", (long long)ret, (long long)info[b], nf, e / m); }
  // leak: repeated calls, device free memory and host RSS
  int n2 = 4096; std::vector<Z> A2((size_t)n2 * n2), B2(n2); for (auto& a : A2) a = R(rnd()); for (int i = 0; i < n2; i++) { A2[i + (size_t)i * n2] = R(MAGMA_Z_REAL(A2[i + (size_t)i * n2]) + n2); B2[i] = R(1); }
  Z *dA2, *dB2; magma_zmalloc(&dA2, (size_t)n2 * n2); magma_zmalloc(&dB2, n2); Z **dA2a, **dB2a; magma_int_t* di2;
  magma_malloc((void**)&dA2a, sizeof(Z*)); magma_malloc((void**)&dB2a, sizeof(Z*)); magma_imalloc(&di2, 1);
  magma_setvector(1, sizeof(Z*), &dA2, 1, dA2a, 1, Q); magma_setvector(1, sizeof(Z*), &dB2, 1, dB2a, 1, Q);
  auto rss = []() { long a = 0, r = 0; FILE* f = fopen("/proc/self/statm", "r"); if (f) { if (fscanf(f, "%ld %ld", &a, &r) != 2) r = 0; fclose(f); } return r * 4096L; };
  size_t f0, t0, f1, t1; magma_zsetmatrix(n2, n2, A2.data(), n2, dA2, n2, Q); magma_zgesv_rbt_batched(n2, 1, dA2a, n2, dB2a, n2, di2, 1, Q);  // warm-up
  magma_queue_sync(Q); magma_mem_info(&f0, &t0); long r0 = rss();
  int calls = 200;
  for (int k = 0; k < calls; k++) { magma_zsetvector(n2, B2.data(), 1, dB2, 1, Q); magma_zgesv_rbt_batched(n2, 1, dA2a, n2, dB2a, n2, di2, 1, Q); }
  magma_queue_sync(Q); magma_mem_info(&f1, &t1); long r1 = rss();
  printf("leak: %d calls with n=%d: device free memory dropped by %.2f MB (dv = 2n complex = %.3f MB per call); host RSS grew by %.2f MB (hu+hv = 4n complex = %.3f MB per call)\n",
         calls, n2, ((double)f0 - (double)f1) / 1e6, 2.0 * n2 * 16 / 1e6, (r1 - r0) / 1e6, 4.0 * n2 * 16 / 1e6);
}

static void bombard() {
  printf("== R3: bombard / bombard_merge, reported iter_res vs true residual ==\n");
  int n = 64; std::vector<double> A(n * n); for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) A[i * n + j] = (i == j ? 3.0 : 0.0) + 0.4 * rnd();
  for (magma_solver_type st : {Magma_BOMBARD, Magma_BOMBARDMERGE}) {
    magma_z_matrix dA = {Magma_CSR}; csr_z(n, A, &dA); std::vector<Z> b(n, R(1)), x0(n, R(0));
    magma_z_matrix hb = {Magma_CSR}, db = {Magma_CSR}, hx = {Magma_CSR}, dx = {Magma_CSR};
    magma_zvset(n, 1, b.data(), &hb, Q); magma_zmtransfer(hb, &db, Magma_CPU, Magma_DEV, Q); magma_zvset(n, 1, x0.data(), &hx, Q); magma_zmtransfer(hx, &dx, Magma_CPU, Magma_DEV, Q);
    for (int mi : {3, 6, 10, 1000}) {
      magma_zopts o; memset(&o, 0, sizeof o); o.solver_par.solver = st; o.solver_par.maxiter = mi; o.solver_par.atol = 1e-16; o.solver_par.rtol = 1e-10;
      o.precond_par.solver = Magma_NONE; magma_zsolverinfo_init(&o.solver_par, &o.precond_par, Q);
      magma_zvset(n, 1, x0.data(), &hx, Q); magma_zmfree(&dx, Q); magma_zmtransfer(hx, &dx, Magma_CPU, Magma_DEV, Q);
      magma_z_solver(dA, db, &dx, &o, Q);
      magma_z_matrix ox = {Magma_CSR}; magma_zmtransfer(dx, &ox, Magma_DEV, Magma_CPU, Q);
      double rr = 0; for (int i = 0; i < n; i++) { double r = 1; for (int j = 0; j < n; j++) r -= A[i * n + j] * MAGMA_Z_REAL(ox.val[j]); rr += r * r; }
      printf("%s maxiter=%4d: info=%lld numiter=%lld iter_res=%.3e final_res=%.3e true ||b-Ax||=%.3e\n", st == Magma_BOMBARD ? "BOMBARD     " : "BOMBARDMERGE",
             mi, (long long)o.solver_par.info, (long long)o.solver_par.numiter, o.solver_par.iter_res, o.solver_par.final_res, sqrt(rr));
      magma_zmfree(&ox, Q);
    }
  }
}

static void single() {
  printf("== R4: single-precision variants ==\n");
  { // c: gbtrs_batched_strided alias
    int n = 8, kl = 1, ku = 1, ldab = 4, batch = 2; std::vector<magmaFloatComplex> AB(ldab * n * batch, MAGMA_C_MAKE(0, 0)), B(n * batch);
    for (int b = 0; b < batch; b++) for (int j = 0; j < n; j++) for (int i = std::max(0, j - 1); i <= std::min(n - 1, j + 1); i++)
      AB[b * ldab * n + (kl + ku + i - j) + j * ldab] = MAGMA_C_MAKE(i == j ? 4.0f : -1.0f, 0);
    for (auto& v : B) v = MAGMA_C_MAKE(1, 0);
    magmaFloatComplex *dAB, *dB; magma_int_t *dp, *di; magma_cmalloc(&dAB, ldab * n * batch); magma_cmalloc(&dB, n * batch); magma_imalloc(&dp, n * batch); magma_imalloc(&di, batch);
    magma_csetvector(ldab * n * batch, AB.data(), 1, dAB, 1, Q); magma_csetvector(n * batch, B.data(), 1, dB, 1, Q);
    magma_cgbtrf_batched_strided(n, n, kl, ku, dAB, ldab, ldab * n, dp, n, di, batch, Q);
    magma_int_t r = magma_cgbtrs_batched_strided(MagmaNoTrans, n, kl, ku, 1, dAB, ldab, ldab * n, dp, n, dB, n, n, di, batch, Q);
    std::vector<magmaFloatComplex> x(n * batch); magma_cgetvector(n * batch, dB, 1, x.data(), 1, Q);
    printf("c gbtrs_batched_strided: ret=%lld x[0]=%g (a correct solve gives about 0.366)\n", (long long)r, MAGMA_C_REAL(x[0]));
  }
  { // c: trtri_gpu zero at 300
    int n = 512; std::vector<magmaFloatComplex> A(n * n, MAGMA_C_MAKE(0, 0)); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) A[i + j * n] = MAGMA_C_MAKE(i == j ? 2.0f : 0.01f * (float)rnd(), 0);
    A[300 + 300 * n] = MAGMA_C_MAKE(0, 0); magmaFloatComplex* dA; magma_cmalloc(&dA, n * n); magma_csetmatrix(n, n, A.data(), n, dA, n, Q);
    magma_int_t info = -999; magma_ctrtri_gpu(MagmaUpper, MagmaNonUnit, n, dA, n, &info); printf("c trtri_gpu n=512 A(300,300)=0: info=%lld (LAPACK 301)\n", (long long)info);
  }
  { // s: potf2_gpu A(0,0) = -1
    int n = 8; std::vector<float> A(n * n, 0); for (int i = 0; i < n; i++) A[i + i * n] = 4; A[0] = -1; float* dA; magma_smalloc(&dA, n * n);
    magma_ssetmatrix(n, n, A.data(), n, dA, n, Q); magma_int_t info = -999; magma_spotf2_gpu(MagmaLower, n, dA, n, Q, &info);
    printf("s potf2_gpu A(0,0)=-1: info=%lld (LAPACK 1)\n", (long long)info);
  }
  { // c sparse cg_merge on diag(2,3), b=(2,i)
    magma_index_t row[3] = {0, 1, 2}, col[2] = {0, 1}; magmaFloatComplex val[2] = {MAGMA_C_MAKE(2, 0), MAGMA_C_MAKE(3, 0)};
    magma_c_matrix hA = {Magma_CSR}, dA = {Magma_CSR}, hb = {Magma_CSR}, db = {Magma_CSR}, hx = {Magma_CSR}, dx = {Magma_CSR};
    magma_ccsrset(2, 2, row, col, val, &hA, Q); magma_cmtransfer(hA, &dA, Magma_CPU, Magma_DEV, Q);
    magmaFloatComplex bv[2] = {MAGMA_C_MAKE(2, 0), MAGMA_C_MAKE(0, 1)}, xv[2] = {MAGMA_C_MAKE(0, 0), MAGMA_C_MAKE(0, 0)};
    magma_cvset(2, 1, bv, &hb, Q); magma_cmtransfer(hb, &db, Magma_CPU, Magma_DEV, Q); magma_cvset(2, 1, xv, &hx, Q); magma_cmtransfer(hx, &dx, Magma_CPU, Magma_DEV, Q);
    magma_c_solver_par sp; magma_c_preconditioner pp; memset(&sp, 0, sizeof sp); memset(&pp, 0, sizeof pp); sp.maxiter = 1000; sp.atol = 1e-8; sp.rtol = 1e-5; pp.solver = Magma_NONE;
    magma_csolverinfo_init(&sp, &pp, Q); magma_ccg_merge(dA, db, &dx, &sp, Q);
    magma_c_matrix ox = {Magma_CSR}; magma_cmtransfer(dx, &ox, Magma_DEV, Magma_CPU, Q);
    printf("c cg_merge A=diag(2,3) b=(2,i): info=%lld numiter=%lld x=(%g%+gi, %g%+gi) (exact (1, 0.333i))\n", (long long)sp.info, (long long)sp.numiter,
           MAGMA_C_REAL(ox.val[0]), MAGMA_C_IMAG(ox.val[0]), MAGMA_C_REAL(ox.val[1]), MAGMA_C_IMAG(ox.val[1]));
  }
}

static void ilu() {
  printf("== R5: ILU(0) setup on [[0,1],[1,0]] with the zero diagonal stored (expect a crash; backtrace from segv_bt.so) ==\n");
  magma_index_t row[3] = {0, 2, 4}, col[4] = {0, 1, 0, 1}; Z val[4] = {R(0), R(1), R(1), R(0)};
  magma_z_matrix h = {Magma_CSR}, dA = {Magma_CSR}, hb = {Magma_CSR}, db = {Magma_CSR}; magma_zcsrset(2, 2, row, col, val, &h, Q); magma_zmtransfer(h, &dA, Magma_CPU, Magma_DEV, Q);
  Z bv[2] = {R(1), R(2)}; magma_zvset(2, 1, bv, &hb, Q); magma_zmtransfer(hb, &db, Magma_CPU, Magma_DEV, Q);
  magma_z_solver_par sp; magma_z_preconditioner pp; memset(&sp, 0, sizeof sp); memset(&pp, 0, sizeof pp); sp.solver = getenv("ILU_OUTER_PCG") ? Magma_PCG : Magma_PBICG; pp.solver = Magma_ILU;
  magma_zsolverinfo_init(&sp, &pp, Q);
  static char altstack[1 << 20]; stack_t ss; ss.ss_sp = altstack; ss.ss_size = sizeof altstack; ss.ss_flags = 0; sigaltstack(&ss, NULL);
  struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = segv_h; sa.sa_flags = SA_ONSTACK; sigaction(SIGSEGV, &sa, NULL); sigaction(SIGBUS, &sa, NULL);
  magma_int_t si = magma_z_precondsetup(dA, db, &sp, &pp, Q); printf("outer solver %s: setup returned %lld; precond->L.memory_location=%d (Magma_CPU=%d, Magma_DEV=%d), L.nnz=%lld\n", getenv("ILU_OUTER_PCG") ? "PCG" : "PBICG", (long long)si, (int)pp.L.memory_location, (int)Magma_CPU, (int)Magma_DEV, (long long)pp.L.nnz);
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : "bpcg";
  magma_init(); magma_queue_create(0, &Q);
  if (m == "bpcg") bpcg(); else if (m == "rbtb") rbtb(); else if (m == "bombard") bombard(); else if (m == "single") single(); else if (m == "ilu") ilu();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

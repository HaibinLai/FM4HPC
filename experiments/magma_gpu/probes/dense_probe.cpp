// GPU confirmation of the MAGMA v2.10.0 dense predictions (HANDOFF §28.4/28.5, Lean MagmaDense.lean).
// Modes:  dense   – dsgesv/dsposv solution and B overflow, zchesv saddle matrices, dshposv tiny/scaled, dgesv_rbt n=1..3
//         rbt64   – dgesv_rbt on the constructed n=64 zero-pivot matrix (must be the FIRST rand() use in the process)
//         nrhs2   – dshposv with nrhs = 2 on a hard SPD matrix (run under compute-sanitizer --tool memcheck)
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

static magma_queue_t Q;

static int nonfinite(const std::vector<double>& x) { int c = 0; for (double v : x) if (!std::isfinite(v)) c++; return c; }
static double relerr(const std::vector<double>& x, const std::vector<double>& xt) {
  double e = 0, m = 0; for (size_t i = 0; i < x.size(); i++) { e = fmax(e, fabs(x[i] - xt[i])); m = fmax(m, fabs(xt[i])); }
  return nonfinite(x) ? INFINITY : e / m;
}

// ---- dsgesv / dsposv (zcgesv / zcposv in d precision) ----
static void run_ds(const char* name, bool spd, int n, const std::vector<double>& A, const std::vector<double>& b,
                   const std::vector<double>& xt) {
  magmaDouble_ptr dA, dB, dX, dwd; magmaFloat_ptr dws; magmaInt_ptr dipiv;
  magma_dmalloc(&dA, n * n); magma_dmalloc(&dB, n); magma_dmalloc(&dX, n); magma_dmalloc(&dwd, n);
  magma_smalloc(&dws, n * (n + 1)); magma_imalloc(&dipiv, n);
  std::vector<magma_int_t> ipiv(n);
  magma_dsetmatrix(n, n, A.data(), n, dA, n, Q); magma_dsetmatrix(n, 1, b.data(), n, dB, n, Q);
  magma_int_t iter = -999, info = -999;
  if (spd) magma_dsposv_gpu(MagmaLower, n, 1, dA, n, dB, n, dX, n, dwd, dws, &iter, &info);
  else     magma_dsgesv_gpu(MagmaNoTrans, n, 1, dA, n, ipiv.data(), dipiv, dB, n, dX, n, dwd, dws, &iter, &info);
  std::vector<double> x(n); magma_dgetmatrix(n, 1, dX, n, x.data(), n, Q);
  printf("%-34s n=%5d  iter=%4lld info=%lld  x[0]=%-12g nonfinite=%5d  relerr=%.2e\n", name, n, (long long)iter,
         (long long)info, x[0], nonfinite(x), relerr(x, xt));
  magma_free(dA); magma_free(dB); magma_free(dX); magma_free(dwd); magma_free(dws); magma_free(dipiv);
}

static void ds_cases() {
  printf("== dsgesv_gpu / dsposv_gpu: solution overflow (A, b in FP32 range) and B overflow ==\n");
  for (int n : {1, 2, 1024}) for (double ea : {-100.0, -90.0}) {
    std::vector<double> A(n * n, 0.0), b(n), xt(n);
    for (int i = 0; i < n; i++) { A[i + i * n] = ldexp(1.0, (int)ea); b[i] = ldexp(1.0, 28); xt[i] = ldexp(1.0, 28 - (int)ea); }
    char nm[64];
    snprintf(nm, 64, "dsgesv a=2^%d b=2^28", (int)ea); run_ds(nm, false, n, A, b, xt);
    snprintf(nm, 64, "dsposv a=2^%d b=2^28", (int)ea); run_ds(nm, true, n, A, b, xt);
  }
  for (int n : {2, 1024}) {
    std::vector<double> A(n * n, 0.0), b(n), xt(n);
    for (int i = 0; i < n; i++) { A[i + i * n] = 1.0; b[i] = ldexp(1.0, 200); xt[i] = b[i]; }
    run_ds("dsgesv a=1 b=2^200", false, n, A, b, xt);
    run_ds("dsposv a=1 b=2^200", true, n, A, b, xt);
  }
}

// ---- zchesv (complex Hermitian; real symmetric data) ----
static void zchesv_cases() {
  printf("== zchesv_gpu on A = [[d, c^T], [c, M]] (inputs identical to zchesv_port.py) ==\n");
  std::ifstream f("zchesv_inputs.txt"); std::string line;
  while (std::getline(f, line)) {
    int n, seed; double d; { std::istringstream h(line); h >> n >> d >> seed; }
    auto rd = [&](int cnt) { std::getline(f, line); std::istringstream s(line); std::vector<double> v(cnt); std::string t;
                             for (int i = 0; i < cnt; i++) { s >> t; v[i] = strtod(t.c_str(), nullptr); } return v; };
    auto A = rd(n * n), b = rd(n), xt = rd(n);
    std::vector<magmaDoubleComplex> zA(n * n), zb(n), zx(n);
    for (int i = 0; i < n * n; i++) zA[i] = MAGMA_Z_MAKE(A[i], 0.0);
    for (int i = 0; i < n; i++) zb[i] = MAGMA_Z_MAKE(b[i], 0.0);
    magmaDoubleComplex_ptr dA, dB, dX, dwd; magmaFloatComplex_ptr dws;
    magma_zmalloc(&dA, n * n); magma_zmalloc(&dB, n); magma_zmalloc(&dX, n); magma_zmalloc(&dwd, n);
    magma_cmalloc(&dws, n * (n + 1));
    magma_zsetmatrix(n, n, zA.data(), n, dA, n, Q); magma_zsetmatrix(n, 1, zb.data(), n, dB, n, Q);
    magma_int_t iter = -999, info = -999;
    magma_zchesv_gpu(MagmaLower, n, 1, dA, n, dB, n, dX, n, dwd, dws, &iter, &info);
    magma_zgetmatrix(n, 1, dX, n, zx.data(), n, Q);
    std::vector<double> x(n); bool isB = true;
    for (int i = 0; i < n; i++) { x[i] = MAGMA_Z_REAL(zx[i]); if (x[i] != b[i] || MAGMA_Z_IMAG(zx[i]) != 0) isB = false; }
    printf("zchesv d=2^%-4d seed=%d  iter=%4lld info=%lld  X==B:%d  nonfinite=%d  relerr=%.2e\n", (int)log2(d), seed,
           (long long)iter, (long long)info, (int)isB, nonfinite(x), relerr(x, xt));
    magma_free(dA); magma_free(dB); magma_free(dX); magma_free(dwd); magma_free(dws);
  }
}

// ---- dshposv (FP16 Cholesky + FGMRES): tiny b, scaling, return value ----
static void spd_matrix(int n, double scale, std::vector<double>& A) {   // SPD, diagonally dominant, deterministic
  A.assign(n * n, 0.0);
  for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) A[i + j * n] = scale * (i == j ? 4.0 : (abs(i - j) == 1 ? -1.0 : 0.0));
}
static void run_dshposv(const char* name, int n, int nrhs, const std::vector<double>& A, const std::vector<double>& b,
                        const std::vector<double>& xt) {
  magmaDouble_ptr dA, dB, dX; magma_dmalloc(&dA, n * n); magma_dmalloc(&dB, n * nrhs); magma_dmalloc(&dX, n * nrhs);
  magma_dsetmatrix(n, n, A.data(), n, dA, n, Q); magma_dsetmatrix(n, nrhs, b.data(), n, dB, n, Q);
  magma_int_t iter = -999, info = -999;
  magma_int_t ret = magma_dshposv_gpu(MagmaLower, n, nrhs, dA, n, dB, n, dX, n, &iter, &info);
  std::vector<double> x(n * nrhs); magma_dgetmatrix(n, nrhs, dX, n, x.data(), n, Q);
  // backward error of column 0 in long double: ||b - A x||inf / (||A||inf ||x||inf)
  long double rn = 0, an = 0, xn = 0;
  for (int i = 0; i < n; i++) { long double r = b[i], s = 0; for (int j = 0; j < n; j++) { r -= (long double)A[i + j * n] * x[j]; s += fabsl(A[i + j * n]); }
    rn = fmaxl(rn, fabsl(r)); an = fmaxl(an, s); xn = fmaxl(xn, fabsl((long double)x[i])); }
  double be = (double)(rn / (an * xn)), thr = sqrt((double)n) * ldexp(1.0, -53);
  std::vector<double> x0(x.begin(), x.begin() + n), xt0(xt.begin(), xt.begin() + n);
  printf("%-30s n=%4d nrhs=%d  ret=%lld iter=%4lld info=%lld  x0max=%-10.3g relerr(col0)=%.2e  bwd/thr=%.2e", name, n, nrhs,
         (long long)ret, (long long)iter, (long long)info, xn == 0 ? 0.0 : (double)xn, relerr(x0, xt0), be / thr);
  if (nrhs > 1) { std::vector<double> x1(x.begin() + n, x.end()), xt1(xt.begin() + n, xt.end());
    double m1 = 0; for (double v : x1) m1 = fmax(m1, fabs(v));
    printf("  col1: max|x|=%.3g relerr=%.2e", m1, relerr(x1, xt1)); }
  printf("\n");
  magma_free(dA); magma_free(dB); magma_free(dX);
}
static void dshposv_cases() {
  printf("== dshposv_gpu (public wrapper: use_gmres=1, preprocess=1) ==\n");
  int n = 256; std::vector<double> A, b(n), xt(n);
  for (int k : {0, -40, 40}) {
    spd_matrix(n, ldexp(1.0, k), A); for (int i = 0; i < n; i++) { xt[i] = 1.0; }
    for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += A[i + j * n] * xt[j]; b[i] = s; }
    char nm[64]; snprintf(nm, 64, "dshposv tridiag*2^%d", k); run_dshposv(nm, n, 1, A, b, xt);
  }
  spd_matrix(n, 1.0, A);
  for (double nb : {1e-15, 1e-17}) {   // scale x_true so that ||b||_2 = nb
    for (int i = 0; i < n; i++) xt[i] = 1.0;
    double s2 = 0; for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += A[i + j * n]; b[i] = s; s2 += s * s; }
    double f = nb / sqrt(s2); for (int i = 0; i < n; i++) { b[i] *= f; xt[i] *= f; }
    char nm[64]; snprintf(nm, 64, "dshposv ||b||_2=%.0e", nb); run_dshposv(nm, n, 1, A, b, xt);
  }
}

// ---- dgesv_rbt (CPU interface) ----
static void rbt_small() {
  printf("== dgesv_rbt(refine=true), diagonally dominant random A, n = 1..6 ==\n");
  unsigned s = 12345u; auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffffff) / (double)0x1000000 - 0.5; };
  for (int n = 1; n <= 6; n++) {
    std::vector<double> A(n * n), xt(n), b(n);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) A[i + j * n] = rnd() + (i == j ? n : 0);
    for (int i = 0; i < n; i++) xt[i] = rnd() + 1.0;
    for (int i = 0; i < n; i++) { double t = 0; for (int j = 0; j < n; j++) t += A[i + j * n] * xt[j]; b[i] = t; }
    std::vector<double> Ac = A, x = b; magma_int_t info = -999;
    magma_dgesv_rbt(MagmaTrue, n, 1, Ac.data(), n, x.data(), n, &info);
    printf("dgesv_rbt n=%d  info=%lld  relerr=%.2e\n", n, (long long)info, relerr(x, xt));
  }
}
static void rbt64() {
  std::ifstream f("rbt_zero_pivot_n64.txt"); std::string line; std::getline(f, line); int n = atoi(line.c_str());
  auto rd = [&](int cnt) { std::getline(f, line); std::istringstream s(line); std::vector<double> v(cnt); std::string t;
                           for (int i = 0; i < cnt; i++) { s >> t; v[i] = strtod(t.c_str(), nullptr); } return v; };
  auto A = rd(n * n), b = rd(n);
  // reference: LAPACK dgesv with pivoting on a copy
  std::vector<double> Ar = A, xr = b; std::vector<magma_int_t> ip(n); magma_int_t one = 1, ir = 0;
  lapackf77_dgesv(&n, &one, Ar.data(), &n, ip.data(), xr.data(), &n, &ir);
  std::vector<double> Ac = A, x = b; magma_int_t info = -999;
  magma_dgesv_rbt(MagmaTrue, n, 1, Ac.data(), n, x.data(), n, &info);
  printf("dgesv_rbt n=64 zero-pivot instance (first rand() use)  info=%lld nonfinite=%d  relerr vs LAPACK dgesv=%.2e  (dgesv info=%lld)\n",
         (long long)info, nonfinite(x), relerr(x, xr), (long long)ir);
}

static void nrhs_nan(int nrhs, bool withNaN) {   // well-scaled SPD; a NaN in column 0 keeps FGMRES running to restrt
  int n = 512; std::vector<double> A, b(n * nrhs), xt(n * nrhs); spd_matrix(n, 1.0, A);
  for (int c = 0; c < nrhs; c++) for (int i = 0; i < n; i++) xt[i + c * n] = (c == 0 ? 1.0 : -1.0);
  for (int c = 0; c < nrhs; c++) for (int i = 0; i < n; i++) { double t = 0; for (int j = 0; j < n; j++) t += A[i + j * n] * xt[j + c * n]; b[i + c * n] = t; }
  if (withNaN) b[0] = NAN;
  char nm[64]; snprintf(nm, 64, "dshposv nrhs=%d b[0]=%s", nrhs, withNaN ? "NaN" : "finite");
  run_dshposv(nm, n, nrhs, A, b, xt);
}
static void nrhs2() {
  printf("== dshposv_gpu nrhs=2, ill-conditioned SPD (for compute-sanitizer) ==\n");
  int n = 512, nrhs = 2; std::vector<double> A(n * n, 0.0), b(n * nrhs), xt(n * nrhs);
  // SPD with eigenvalues spread over 1..1e12 via a diagonal matrix plus weak coupling
  for (int i = 0; i < n; i++) A[i + i * n] = pow(1e12, (double)i / (n - 1));
  for (int i = 0; i + 1 < n; i++) { A[i + (i + 1) * n] = A[(i + 1) + i * n] = 0.1; }
  for (int c = 0; c < nrhs; c++) for (int i = 0; i < n; i++) xt[i + c * n] = (c == 0 ? 1.0 : -1.0);
  for (int c = 0; c < nrhs; c++) for (int i = 0; i < n; i++) { double t = 0; for (int j = 0; j < n; j++) t += A[i + j * n] * xt[j + c * n]; b[i + c * n] = t; }
  run_dshposv("dshposv nrhs=2 hard SPD", n, nrhs, A, b, xt);
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "dense";
  magma_init(); magma_queue_create(0, &Q);
  if (mode == "rbt64") rbt64();                 // nothing else may call rand() before this
  else if (mode == "nrhs2") nrhs2();
  else if (mode == "nan2") nrhs_nan(2, true);
  else if (mode == "nan1") nrhs_nan(1, true);
  else if (mode == "clean2") nrhs_nan(2, false);
  else { ds_cases(); zchesv_cases(); dshposv_cases(); rbt_small(); }
  magma_queue_destroy(Q); magma_finalize();
  return 0;
}

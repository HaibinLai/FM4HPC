// Tenth GPU round (HANDOFF §29.10).  GPU 0 only; one call per process.
//   ./round10_probe <driver> <k>   with LD_PRELOAD=fail_malloc.so: the k-th device/pinned allocation inside the driver
//                                  fails (k = 0: count only).  Prints: driver k nalloc info err verdict.
//   ./round10_probe ooc_lwork | ooc_nb0      zgeqrf_ooc with lwork = n / with almost no free memory
//   ./round10_probe ngpu2                    *_m drivers with ngpu = 2 while only GPU 0 is visible
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
typedef magmaDoubleComplex Z;
extern "C" void zhegv_(const magma_int_t*, const char*, const char*, const magma_int_t*, Z*, const magma_int_t*, Z*, const magma_int_t*,
                       double*, Z*, const magma_int_t*, double*, magma_int_t*);
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 4242u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static void (*fi_arm)(long) = 0; static void (*fi_disarm)() = 0; static long (*fi_count)() = 0;
static void arm(long k) { if (fi_arm) fi_arm(k); } static long disarm() { long c = fi_count ? fi_count() : -1; if (fi_disarm) fi_disarm(); return c; }

static std::vector<Z> herm(int n) { std::vector<Z> A((size_t)n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) {
  Z v = i == j ? C(rnd()) : C(rnd(), rnd()); A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = MAGMA_Z_CONJ(v); } return A; }
static std::vector<Z> hpd(int n) { auto B = herm(n); for (int i = 0; i < n; i++) B[i + (size_t)i * n] = MAGMA_Z_ADD(B[i + (size_t)i * n], C(n)); return B; }
static double relerr(const std::vector<double>& w, const std::vector<double>& r) { double e = 0, m = 0; for (size_t i = 0; i < r.size(); i++) {
  if (!std::isfinite(w[i])) return INFINITY; e = fmax(e, fabs(w[i] - r[i])); m = fmax(m, fabs(r[i])); } return e / m; }
// complex eigenvalue sets: max over computed of distance to nearest reference
static double seterr(const std::vector<Z>& w, const std::vector<Z>& r) { double e = 0, m = 0; for (auto& a : w) { if (!std::isfinite(MAGMA_Z_REAL(a)) || !std::isfinite(MAGMA_Z_IMAG(a))) return INFINITY;
  double d = 1e300; for (auto& b : r) d = fmin(d, MAGMA_Z_ABS(MAGMA_Z_SUB(a, b))); e = fmax(e, d); } for (auto& b : r) m = fmax(m, MAGMA_Z_ABS(b)); return e / m; }

typedef void (*heev_t)(int, Z*, double*, Z*, magma_int_t, double*, magma_int_t, magma_int_t*, magma_int_t, magma_int_t*);
static void ref_heevd(int n, std::vector<Z> A, std::vector<double>& w) { w.resize(n); magma_int_t nn = n, lw = -1, lrw = -1, liw = -1, info; Z q; double rq; magma_int_t iq;
  lapackf77_zheevd("N", "L", &nn, A.data(), &nn, w.data(), &q, &lw, &rq, &lrw, &iq, &liw, &info); lw = (magma_int_t)MAGMA_Z_REAL(q); lrw = (magma_int_t)rq; liw = iq;
  std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); lapackf77_zheevd("N", "L", &nn, A.data(), &nn, w.data(), wk.data(), &lw, rw.data(), &lrw, iw.data(), &liw, &info); }

static int run_driver(const std::string& d, long k, magma_int_t* info_out, double* err_out) {
  magma_int_t info = -999; double err = NAN;
  if (d == "zheevd" || d == "zheevd_m" || d == "zheevdx_2stage" || d == "zheevdx_2stage_m" || d == "zhegvd" || d == "zhegvd_m") {
    bool gen = d.find("hegvd") != std::string::npos; int n = d.find("2stage") != std::string::npos ? 1000 : 600;
    auto A0 = herm(n); auto B0 = hpd(n); std::vector<double> wr;
    if (gen) { // reference: eigenvalues of L^-1 A L^-H via LAPACK zhegv
      std::vector<Z> A = A0, B = B0; wr.resize(n); magma_int_t nn = n, one = 1, lw = -1, inf; Z q; std::vector<double> rw(3 * n);
      zhegv_(&one, "N", "L", &nn, A.data(), &nn, B.data(), &nn, wr.data(), &q, &lw, rw.data(), &inf); lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> wk(lw);
      zhegv_(&one, "N", "L", &nn, A.data(), &nn, B.data(), &nn, wr.data(), wk.data(), &lw, rw.data(), &inf);
    } else ref_heevd(n, A0, wr);
    std::vector<Z> A = A0, B = B0; std::vector<double> w(n, NAN); magma_int_t lw = -1, lrw = -1, liw = -1, m = 0; Z q; double rq; magma_int_t iq;
    auto call = [&](Z* wk, magma_int_t lw_, double* rw, magma_int_t lrw_, magma_int_t* iw, magma_int_t liw_) {
      if (d == "zheevd") magma_zheevd(MagmaVec, MagmaLower, n, A.data(), n, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info);
      else if (d == "zheevd_m") magma_zheevd_m(1, MagmaVec, MagmaLower, n, A.data(), n, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info);
      else if (d == "zheevdx_2stage") magma_zheevdx_2stage(MagmaVec, MagmaRangeAll, MagmaLower, n, A.data(), n, 0, 0, 1, n, &m, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info);
      else if (d == "zheevdx_2stage_m") magma_zheevdx_2stage_m(1, MagmaVec, MagmaRangeAll, MagmaLower, n, A.data(), n, 0, 0, 1, n, &m, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info);
      else if (d == "zhegvd") magma_zhegvd(1, MagmaVec, MagmaLower, n, A.data(), n, B.data(), n, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info);
      else magma_zhegvd_m(1, 1, MagmaVec, MagmaLower, n, A.data(), n, B.data(), n, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info); };
    call(&q, lw, &rq, lrw, &iq, liw); lw = (magma_int_t)MAGMA_Z_REAL(q); lrw = (magma_int_t)rq; liw = iq;
    std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); A = A0; B = B0; info = -999;
    arm(k); call(wk.data(), lw, rw.data(), lrw, iw.data(), liw); long c = disarm();
    err = relerr(w, wr); *info_out = info; *err_out = err; return (int)c;
  }
  if (d == "zgeev" || d == "zgeev_m") {
    int n = 600; std::vector<Z> A0((size_t)n * n); for (auto& a : A0) a = C(rnd(), rnd());
    std::vector<Z> wr(n), A = A0; { magma_int_t nn = n, lw = -1, inf, one = 1; Z q; std::vector<double> rw(2 * n); Z dummy;
      lapackf77_zgeev("N", "N", &nn, A.data(), &nn, wr.data(), &dummy, &one, &dummy, &one, &q, &lw, rw.data(), &inf); lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> wk(lw);
      lapackf77_zgeev("N", "N", &nn, A.data(), &nn, wr.data(), &dummy, &one, &dummy, &one, wk.data(), &lw, rw.data(), &inf); }
    A = A0; std::vector<Z> w(n, C(NAN)), VR((size_t)n * n), VL(1); std::vector<double> rw(2 * n); Z q; magma_int_t lw = -1;
    auto call = [&](Z* wk, magma_int_t lw_) { if (d == "zgeev") magma_zgeev(MagmaNoVec, MagmaVec, n, A.data(), n, w.data(), VL.data(), 1, VR.data(), n, wk, lw_, rw.data(), &info);
      else magma_zgeev_m(MagmaNoVec, MagmaVec, n, A.data(), n, w.data(), VL.data(), 1, VR.data(), n, wk, lw_, rw.data(), &info); };
    call(&q, lw); lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> wk(lw); A = A0; info = -999;
    arm(k); call(wk.data(), lw); long c = disarm(); err = seterr(w, wr); *info_out = info; *err_out = err; return (int)c;
  }
  if (d == "zgesvd") {
    int n = 600; std::vector<Z> A0((size_t)n * n); for (auto& a : A0) a = C(rnd(), rnd()); std::vector<double> sr(n), s(n, NAN);
    { std::vector<Z> A = A0; magma_int_t nn = n, lw = -1, inf, one = 1; Z q, dummy; std::vector<double> rw(5 * n);
      lapackf77_zgesvd("N", "N", &nn, &nn, A.data(), &nn, sr.data(), &dummy, &one, &dummy, &one, &q, &lw, rw.data(), &inf); lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> wk(lw);
      lapackf77_zgesvd("N", "N", &nn, &nn, A.data(), &nn, sr.data(), &dummy, &one, &dummy, &one, wk.data(), &lw, rw.data(), &inf); }
    std::vector<Z> A = A0, U((size_t)n * n), VT((size_t)n * n); std::vector<double> rw(5 * n); Z q; magma_int_t lw = -1;
    magma_zgesvd(MagmaAllVec, MagmaAllVec, n, n, A.data(), n, s.data(), U.data(), n, VT.data(), n, &q, lw, rw.data(), &info);
    lw = (magma_int_t)MAGMA_Z_REAL(q); std::vector<Z> wk(lw); A = A0; info = -999;
    arm(k); magma_zgesvd(MagmaAllVec, MagmaAllVec, n, n, A.data(), n, s.data(), U.data(), n, VT.data(), n, wk.data(), lw, rw.data(), &info); long c = disarm();
    err = relerr(s, sr); *info_out = info; *err_out = err; return (int)c;
  }
  if (d == "zgesv" || d == "zposv") {
    int n = 1000; std::vector<Z> A0 = d == "zposv" ? hpd(n) : std::vector<Z>((size_t)n * n); if (d == "zgesv") { for (auto& a : A0) a = C(rnd(), rnd()); for (int i = 0; i < n; i++) A0[i + (size_t)i * n] = MAGMA_Z_ADD(A0[i + (size_t)i * n], C(n)); }
    std::vector<Z> xt(n), b(n, C(0)); for (auto& v : xt) v = C(rnd(), rnd()); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) b[i] = MAGMA_Z_ADD(b[i], MAGMA_Z_MUL(A0[i + (size_t)j * n], xt[j]));
    std::vector<Z> A = A0, x = b; std::vector<magma_int_t> ip(n); arm(k);
    if (d == "zgesv") magma_zgesv(n, 1, A.data(), n, ip.data(), x.data(), n, &info); else magma_zposv(MagmaLower, n, 1, A.data(), n, x.data(), n, &info);
    long c = disarm(); double e = 0, m = 0; for (int i = 0; i < n; i++) { if (!std::isfinite(MAGMA_Z_REAL(x[i]))) { e = INFINITY; break; } e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(x[i], xt[i]))); m = fmax(m, MAGMA_Z_ABS(xt[i])); }
    *info_out = info; *err_out = e / m; return (int)c;
  }
  return -1;
}
static size_t hog(size_t leave) { size_t fr, tot; cudaMemGetInfo(&fr, &tot); size_t take = fr > leave ? fr - leave : 0; void* p = 0;
  if (take) cudaMalloc(&p, take); cudaMemGetInfo(&fr, &tot); return fr; }
static void ooc(bool nb0) {
  int m = 20000, n = 4096; std::vector<Z> A((size_t)m * n); for (auto& a : A) a = C(rnd(), rnd()); std::vector<Z> tau(n);
  size_t fr = hog(nb0 ? (size_t)20e6 : (size_t)300e6); magma_int_t nb = magma_get_zgeqrf_nb(m, n);
  long NB = (long)(0.8 * (fr / sizeof(Z)) / m); NB = NB / nb * nb;
  magma_int_t lwork = nb0 ? n * nb : n; std::vector<Z> wk(lwork); magma_int_t info = -999;
  printf("zgeqrf_ooc m=%d n=%d: free %.1f MB, nb=%lld, NB=%ld (out of core: NB < n), lwork=%lld (documented >= n*nb = %lld) ...\n", m, n, fr / 1e6, (long long)nb, NB,
         (long long)lwork, (long long)(n * nb)); fflush(stdout);
  magma_zgeqrf_ooc(m, n, A.data(), m, tau.data(), wk.data(), lwork, &info);
  printf("  returned info=%lld\n", (long long)info);
}
static void ngpu2() {
  int nv = 0; cudaGetDeviceCount(&nv); printf("visible devices: %d; magma_num_gpus() = %lld\n", nv, (long long)magma_num_gpus());
  { int n = 2000; auto A0 = hpd(n); std::vector<Z> A = A0; magma_int_t info = -999; magma_zpotrf_m(2, MagmaLower, n, A.data(), n, &info);
    std::vector<Z> L = A0; magma_int_t nn = n, i2; lapackf77_zpotrf("L", &nn, L.data(), &nn, &i2); double e = 0; for (int j = 0; j < n; j++) for (int i = j; i < n; i++) e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(A[i + (size_t)j * n], L[i + (size_t)j * n])));
    printf("zpotrf_m(ngpu=2) n=2000: info=%lld  max|L - L_lapack|=%.3g\n", (long long)info, e); }
  { int n = 600; auto A0 = herm(n); std::vector<double> wr; ref_heevd(n, A0, wr); std::vector<Z> A = A0; std::vector<double> w(n, NAN);
    magma_int_t lw = -1, lrw = -1, liw = -1, info; Z q; double rq; magma_int_t iq;
    magma_zheevd_m(2, MagmaVec, MagmaLower, n, A.data(), n, w.data(), &q, lw, &rq, lrw, &iq, liw, &info); lw = (magma_int_t)MAGMA_Z_REAL(q); lrw = (magma_int_t)rq; liw = iq;
    std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); A = A0; info = -999;
    magma_zheevd_m(2, MagmaVec, MagmaLower, n, A.data(), n, w.data(), wk.data(), lw, rw.data(), lrw, iw.data(), liw, &info);
    printf("zheevd_m(ngpu=2) n=600: info=%lld  eigenvalue relerr=%.3g\n", (long long)info, relerr(w, wr)); }
  cudaError_t e = cudaGetLastError(); printf("cudaGetLastError after the calls: %s\n", cudaGetErrorString(e));
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; long k = argc > 2 ? atol(argv[2]) : 0;
  fi_arm = (void (*)(long))dlsym(RTLD_DEFAULT, "fi_arm"); fi_disarm = (void (*)())dlsym(RTLD_DEFAULT, "fi_disarm"); fi_count = (long (*)())dlsym(RTLD_DEFAULT, "fi_count");
  magma_init();
  if (m == "ooc_lwork") ooc(false); else if (m == "ooc_nb0") ooc(true); else if (m == "ngpu2") ngpu2();
  else { magma_int_t info; double err; int c = run_driver(m, k, &info, &err);
    const char* v = (info == 0 && !(err <= 1e-8)) ? "SILENT-WRONG" : (info != 0 ? "reported" : "ok");
    printf("%-18s k=%3ld nalloc=%3d info=%-6lld err=%-10.3g %s\n", m.c_str(), k, c, (long long)info, err, v); }
  magma_finalize(); return 0;
}

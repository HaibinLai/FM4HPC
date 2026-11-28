// Round 52 (roxy): schedule fuzzing of multi-GPU routines with ngpu virtual devices on GPU 0 (mgpu_fuzz/fuzz_hooks.h).
//   round52_mgpu_fuzz <getrf|potrfU|potrfL|trsm|hegst|geqrf> <ngpu> <n> <target> [ms=100] [seed=1]
//   getrf  magma_zgetrf_mgpu (+ zgetrf2_mgpu), 1-D block-column cyclic d_lA; check P*A*x = L*(U*x)
//   potrfU/L  magma_zpotrf_mgpu (+ zpotrf3_mgpu): column / row cyclic; check against LAPACK zpotrf
//   trsm   magma_ztrsm_m (CPU interface), Left Lower NoTrans NonUnit, B n x n; check against blas ztrsm
//   hegst  magma_zhegst_m (CPU interface), itype 1, Lower; check against LAPACK zhegst
//   geqrf  magma_zgeqrf2_mgpu, column cyclic; check R and tau against LAPACK zgeqrf
//   getrfm / getrfmooc  magma_zgetrf_m (CPU interface; ooc sets MAGMA_NGR_NB=64 to force the out-of-core path); P*A*x = L*(U*x)
//   potrfm magma_zpotrf_m (CPU interface), Lower; check against LAPACK zpotrf
//   hetrd  magma_zhetrd_mgpu (+ zlatrd_mgpu, zher2k_mgpu, zhemv_mgpu), nqueue 2, Lower; d and |e| against LAPACK zhetrd
//   unmqr  magma_zunmqr_m, Left NoTrans, k = n/2 reflectors from LAPACK zgeqrf; against LAPACK zunmqr
//   geqrfm magma_zgeqrf_m (CPU interface; geqrf2_mgpu fuzzed); R and tau against LAPACK zgeqrf
//   ungqr  magma_zungqr_m (ngpu from magma_num_gpus, overridden), T from magma_zgeqrf_gpu; Q against LAPACK zungqr
//   heevd2s magma_zheevdx_2stage_m (he2hb_mgpu, hemm/her2k_mgpu, bulge_back_m, applyQ_v2_m, stedx_m, dlaex*_m, unmqr_m), V, All, Lower;
//          eigenvalues against LAPACK zheevd and max |A v - w v| / (|A| n)
//   gehrd  magma_zgehrd_m (+ zlahr2_m, zlahru_m; ngpu from magma_num_gpus, overridden), ilo 1, ihi n; H and tau against LAPACK zgehrd
// R52_PIN (environment): pinned host arrays for the CPU-interface routines, see H() / unpin().
// target = -1: no delay, prints the number of points; otherwise the queue of point `target` sleeps ms.
// Check: P*A*x against L*(U*x) for a random x (O(n^2)); rel = max|diff| / (max|A| * sum|x|).
// CSV: r52,routine,ngpu,n,target,points,delayed_at,info,rel,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include "fuzz_hooks.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
FuzzState g_fuzz;
extern "C" { double r52_t0 = 0; }   // debug timer origin (R52_DBG)
extern "C" decltype(magma_zgetrf_mgpu) fuzz_zgetrf_mgpu;
extern "C" decltype(magma_zpotrf_mgpu) fuzz_zpotrf_mgpu;
extern "C" decltype(magma_ztrsm_m) fuzz_ztrsm_m;
extern "C" decltype(magma_zhegst_m) fuzz_zhegst_m;
extern "C" decltype(magma_zgeqrf2_mgpu) fuzz_zgeqrf2_mgpu;
extern "C" decltype(magma_zgetrf_m) fuzz_zgetrf_m;
extern "C" decltype(magma_zpotrf_m) fuzz_zpotrf_m;
extern "C" decltype(magma_zhetrd_mgpu) fuzz_zhetrd_mgpu;
extern "C" decltype(magma_zunmqr_m) fuzz_zunmqr_m;
extern "C" decltype(magma_zgeqrf_m) fuzz_zgeqrf_m;
extern "C" decltype(magma_zungqr_m) fuzz_zungqr_m;
extern "C" decltype(magma_zheevdx_2stage_m) fuzz_zheevdx_2stage_m;
extern "C" decltype(magma_zgehrd_m) fuzz_zgehrd_m;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
typedef magmaDoubleComplex Z;
// R52_PIN set: host arrays handed to the CPU-interface routines are pinned copies (as in MAGMA's testers), copied back right
// after the call returns and before any device synchronisation, so asynchronous copies the routine left unfinished show up.
static const bool PIN = getenv("R52_PIN") != nullptr;
struct PinRec { void* pin; void* host; size_t bytes; }; static std::vector<PinRec> g_pins;
template <class T> static T* H(std::vector<T>& v) { if (!PIN || v.empty()) return v.data(); void* p = nullptr; size_t b = v.size() * sizeof(T);
  if (magma_malloc_pinned(&p, b) != MAGMA_SUCCESS) { fprintf(stderr, "pinned alloc failed\n"); exit(3); } memcpy(p, v.data(), b); g_pins.push_back({p, v.data(), b}); return (T*)p; }
static void unpin() { for (auto& r : g_pins) memcpy(r.host, r.pin, r.bytes); cudaDeviceSynchronize(); for (auto& r : g_pins) magma_free_pinned(r.pin); g_pins.clear(); }
static Z zm(Z a, Z b) { return MAGMA_Z_MUL(a, b); }
static const char* where() { return g_fuzz.where ? g_fuzz.where + std::string(g_fuzz.where).find_last_of("/\\") + 1 : "-"; }
static void report(const char* r, int g, int n, long long info, double rel, double tol) {
  printf("r52,%s%s,%d,%d,%lld,%lld,%s:%d,%lld,%.3g,%s\n", r, PIN ? "/pin" : "", g, n, g_fuzz.target, g_fuzz.count, where(), g_fuzz.line, info, rel, (info == 0 && rel < tol) ? "OK" : "WRONG"); }
static double lu_resid(const std::vector<Z>& A, const std::vector<Z>& LU, const std::vector<magma_int_t>& ipiv, int n) {
  std::vector<Z> x(n), y(n, MAGMA_Z_ZERO), u(n, MAGMA_Z_ZERO), z(n, MAGMA_Z_ZERO); double amax = 0, xs = 0;
  for (auto& v : x) { v = MAGMA_Z_MAKE(r1(), r1()); xs += MAGMA_Z_ABS(v); } for (auto& a : A) amax = std::max(amax, MAGMA_Z_ABS(a));
  for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) y[i] = MAGMA_Z_ADD(y[i], zm(A[i + (size_t)j * n], x[j]));
  for (int i = 0; i < n; i++) { int p = (int)ipiv[i] - 1; if (p < i || p >= n) return INFINITY; std::swap(y[i], y[p]); }
  for (int i = 0; i < n; i++) for (int j = i; j < n; j++) u[i] = MAGMA_Z_ADD(u[i], zm(LU[i + (size_t)j * n], x[j]));
  for (int i = 0; i < n; i++) { z[i] = u[i]; for (int j = 0; j < i; j++) z[i] = MAGMA_Z_ADD(z[i], zm(LU[i + (size_t)j * n], u[j])); }
  double e = 0; for (int i = 0; i < n; i++) { double d = MAGMA_Z_ABS(MAGMA_Z_SUB(y[i], z[i])); if (!(d <= e)) e = d; } return e / (amax * xs); }
static std::vector<Z> herm(int n, double shift) { std::vector<Z> A((size_t)n * n);
  for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { Z v = i == j ? MAGMA_Z_MAKE(r1() + shift, 0) : MAGMA_Z_MAKE(r1(), r1()); A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = MAGMA_Z_CONJ(v); } return A; }
static double maxdiff(const std::vector<Z>& X, const std::vector<Z>& R, int n, int part) {   // part: 0 all, 1 lower, 2 upper
  double e = 0, a = 0; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { if ((part == 1 && i < j) || (part == 2 && i > j)) continue;
    double d = MAGMA_Z_ABS(MAGMA_Z_SUB(X[i + (size_t)j * n], R[i + (size_t)j * n])); if (!(d <= e)) e = d; a = std::max(a, MAGMA_Z_ABS(R[i + (size_t)j * n])); } return e / a; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 5) { printf("usage: see source\n"); return 1; }
  std::string r = argv[1]; int g = atoi(argv[2]), n = atoi(argv[3]); g_fuzz.target = atoll(argv[4]); if (argc > 5) g_fuzz.ms = atoi(argv[5]); if (argc > 6) Sd = (unsigned)atoi(argv[6]);
  const unsigned seed0 = Sd;
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  if (r == "getrf") {
    int nb = (int)magma_get_zgetrf_nb(n, n); std::vector<Z> A((size_t)n * n); for (auto& a : A) a = MAGMA_Z_MAKE(r1(), r1());
    int ldda = magma_roundup(n, 32); std::vector<Z*> dA(g); std::vector<int> nl(g);
    auto dev = [&](int J) { return (J / nb) % g; }; auto loc = [&](int J) { return (J / (nb * g)) * nb + J % nb; };
    for (int d = 0; d < g; d++) { nl[d] = ((n / nb) / g) * nb; if (d < (n / nb) % g) nl[d] += nb; else if (d == (n / nb) % g) nl[d] += n % nb;
      magma_zmalloc(&dA[d], (size_t)ldda * std::max(nl[d], 1)); std::vector<Z> L((size_t)ldda * std::max(nl[d], 1), MAGMA_Z_ZERO);
      for (int J = 0; J < n; J++) if (dev(J) == d) for (int i = 0; i < n; i++) L[i + (size_t)loc(J) * ldda] = A[i + (size_t)J * n];
      magma_zsetvector((magma_int_t)L.size(), L.data(), 1, dA[d], 1, q); }
    magma_queue_sync(q); std::vector<magma_int_t> ipiv(n); magma_int_t info = 0;
    fuzz_zgetrf_mgpu(g, n, n, dA.data(), ldda, ipiv.data(), &info); cudaDeviceSynchronize();
    std::vector<Z> LU((size_t)n * n);
    for (int d = 0; d < g; d++) { std::vector<Z> L((size_t)ldda * std::max(nl[d], 1)); magma_zgetvector((magma_int_t)L.size(), dA[d], 1, L.data(), 1, q);
      for (int J = 0; J < n; J++) if (dev(J) == d) for (int i = 0; i < n; i++) LU[i + (size_t)J * n] = L[i + (size_t)loc(J) * ldda]; magma_free(dA[d]); }
    std::vector<Z> x(n), y(n, MAGMA_Z_ZERO), u(n, MAGMA_Z_ZERO), z(n, MAGMA_Z_ZERO); double amax = 0, xs = 0;
    for (auto& v : x) { v = MAGMA_Z_MAKE(r1(), r1()); xs += MAGMA_Z_ABS(v); } for (auto& a : A) amax = std::max(amax, MAGMA_Z_ABS(a));
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) y[i] = MAGMA_Z_ADD(y[i], zm(A[i + (size_t)j * n], x[j]));
    bool okp = true; for (int i = 0; i < n; i++) { int p = (int)ipiv[i] - 1; if (p < i || p >= n) { okp = false; break; } std::swap(y[i], y[p]); }
    for (int i = 0; i < n; i++) for (int j = i; j < n; j++) u[i] = MAGMA_Z_ADD(u[i], zm(LU[i + (size_t)j * n], x[j]));
    for (int i = 0; i < n; i++) { z[i] = u[i]; for (int j = 0; j < i; j++) z[i] = MAGMA_Z_ADD(z[i], zm(LU[i + (size_t)j * n], u[j])); }
    double e = 0; for (int i = 0; i < n; i++) { double d = MAGMA_Z_ABS(MAGMA_Z_SUB(y[i], z[i])); if (!(d <= e)) e = d; }
    double rel = okp ? e / (amax * xs) : INFINITY;
    report("getrf", g, n, (long long)info, rel, 1e-11);
  }
  else if (r == "potrfU" || r == "potrfL") {
    bool up = r == "potrfU"; int nb = (int)magma_get_zpotrf_nb(n); std::vector<Z> A = herm(n, n), R = A; magma_int_t rinfo; char uc = up ? 'U' : 'L'; magma_int_t nn = n;
    lapackf77_zpotrf(&uc, &nn, R.data(), &nn, &rinfo);
    std::vector<int> nl(g); for (int d = 0; d < g; d++) { nl[d] = (n / (nb * g)) * nb; if (d < (n / nb) % g) nl[d] += nb; else if (d == (n / nb) % g) nl[d] += n % nb; }
    int ldda = up ? magma_roundup(n, 32) : magma_roundup((n / (nb * g) + 1) * nb, 32);
    auto dev = [&](int x) { return (x / nb) % g; }; auto loc = [&](int x) { return (x / (nb * g)) * nb + x % nb; };
    std::vector<Z*> dA(g); std::vector<size_t> cnt(g);
    for (int d = 0; d < g; d++) { cnt[d] = up ? (size_t)ldda * std::max(nl[d], 1) : (size_t)ldda * n; magma_zmalloc(&dA[d], cnt[d]); std::vector<Z> L(cnt[d], MAGMA_Z_ZERO);
      if (up) { for (int J = 0; J < n; J++) if (dev(J) == d) for (int i = 0; i < n; i++) L[i + (size_t)loc(J) * ldda] = A[i + (size_t)J * n]; }
      else { for (int I = 0; I < n; I++) if (dev(I) == d) for (int j = 0; j < n; j++) L[loc(I) + (size_t)j * ldda] = A[I + (size_t)j * n]; }
      magma_zsetvector((magma_int_t)cnt[d], L.data(), 1, dA[d], 1, q); }
    magma_queue_sync(q); magma_int_t info = 0; fuzz_zpotrf_mgpu(g, up ? MagmaUpper : MagmaLower, n, dA.data(), ldda, &info); cudaDeviceSynchronize();
    std::vector<Z> X((size_t)n * n, MAGMA_Z_ZERO);
    for (int d = 0; d < g; d++) { std::vector<Z> L(cnt[d]); magma_zgetvector((magma_int_t)cnt[d], dA[d], 1, L.data(), 1, q);
      for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { if (up ? dev(j) != d : dev(i) != d) continue; X[i + (size_t)j * n] = up ? L[i + (size_t)loc(j) * ldda] : L[loc(i) + (size_t)j * ldda]; } magma_free(dA[d]); }
    report(r.c_str(), g, n, info + rinfo, maxdiff(X, R, n, up ? 2 : 1), 1e-10);
  }
  else if (r == "trsm") {
    std::vector<Z> A((size_t)n * n, MAGMA_Z_ZERO), B((size_t)n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) A[i + (size_t)j * n] = i == j ? MAGMA_Z_MAKE(2 + r1(), r1()) : MAGMA_Z_MAKE(0.1 * r1(), 0.1 * r1());
    for (auto& b : B) b = MAGMA_Z_MAKE(r1(), r1()); std::vector<Z> R = B; Z one = MAGMA_Z_ONE; magma_int_t nn = n;
    blasf77_ztrsm("L", "L", "N", "N", &nn, &nn, &one, A.data(), &nn, R.data(), &nn);
    fuzz_ztrsm_m(g, MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, n, n, one, H(A), n, H(B), n); unpin(); cudaDeviceSynchronize();
    report("trsm", g, n, 0, maxdiff(B, R, n, 0), 1e-10);
  }
  else if (r == "hegst") {
    std::vector<Z> A = herm(n, 0), Bm = herm(n, n); magma_int_t nn = n, rinfo, one = 1; lapackf77_zpotrf("L", &nn, Bm.data(), &nn, &rinfo);
    std::vector<Z> R = A; lapackf77_zhegst(&one, "L", &nn, R.data(), &nn, Bm.data(), &nn, &rinfo);
    magma_int_t info = 0; fuzz_zhegst_m(g, 1, MagmaLower, n, H(A), n, H(Bm), n, &info); unpin(); cudaDeviceSynchronize();
    report("hegst", g, n, info + rinfo, maxdiff(A, R, n, 1), 1e-10);
  }
  else if (r == "geqrf") {
    int nb = (int)magma_get_zgeqrf_nb(n, n); std::vector<Z> A((size_t)n * n); for (auto& a : A) a = MAGMA_Z_MAKE(r1(), r1());
    std::vector<Z> R = A, tr(n), tau(n); magma_int_t nn = n, lw = -1, rinfo; Z wq; lapackf77_zgeqrf(&nn, &nn, R.data(), &nn, tr.data(), &wq, &lw, &rinfo);
    lw = (magma_int_t)MAGMA_Z_REAL(wq); std::vector<Z> w(lw); lapackf77_zgeqrf(&nn, &nn, R.data(), &nn, tr.data(), w.data(), &lw, &rinfo);
    int ldda = magma_roundup(n, 32); std::vector<Z*> dA(g); std::vector<int> nl(g);
    auto dev = [&](int J) { return (J / nb) % g; }; auto loc = [&](int J) { return (J / (nb * g)) * nb + J % nb; };
    for (int d = 0; d < g; d++) { nl[d] = ((n / nb) / g) * nb; if (d < (n / nb) % g) nl[d] += nb; else if (d == (n / nb) % g) nl[d] += n % nb;
      magma_zmalloc(&dA[d], (size_t)ldda * std::max(nl[d], 1)); std::vector<Z> L((size_t)ldda * std::max(nl[d], 1), MAGMA_Z_ZERO);
      for (int J = 0; J < n; J++) if (dev(J) == d) for (int i = 0; i < n; i++) L[i + (size_t)loc(J) * ldda] = A[i + (size_t)J * n];
      magma_zsetvector((magma_int_t)L.size(), L.data(), 1, dA[d], 1, q); }
    magma_queue_sync(q); magma_int_t info = 0; fuzz_zgeqrf2_mgpu(g, n, n, dA.data(), ldda, tau.data(), &info); cudaDeviceSynchronize();
    std::vector<Z> X((size_t)n * n);
    for (int d = 0; d < g; d++) { std::vector<Z> L((size_t)ldda * std::max(nl[d], 1)); magma_zgetvector((magma_int_t)L.size(), dA[d], 1, L.data(), 1, q);
      for (int J = 0; J < n; J++) if (dev(J) == d) for (int i = 0; i < n; i++) X[i + (size_t)J * n] = L[i + (size_t)loc(J) * ldda]; magma_free(dA[d]); }
    double et = 0; for (int i = 0; i < n; i++) et = std::max(et, MAGMA_Z_ABS(MAGMA_Z_SUB(tau[i], tr[i])));
    report("geqrf", g, n, info + rinfo, std::max(maxdiff(X, R, n, 2), et), 1e-10);
  }
  else if (r == "getrfm" || r == "getrfmooc") {
    if (r == "getrfmooc") _putenv_s("MAGMA_NGR_NB", "64");
    std::vector<Z> A((size_t)n * n); for (auto& a : A) a = MAGMA_Z_MAKE(r1(), r1()); std::vector<Z> LU = A; std::vector<magma_int_t> ipiv(n); magma_int_t info = 0;
    fuzz_zgetrf_m(g, n, n, H(LU), n, H(ipiv), &info); unpin(); cudaDeviceSynchronize();
    report(r.c_str(), g, n, info, lu_resid(A, LU, ipiv, n), 1e-11);
  }
  else if (r == "potrfm") {
    std::vector<Z> A = herm(n, n), R = A; magma_int_t nn = n, rinfo; lapackf77_zpotrf("L", &nn, R.data(), &nn, &rinfo);
    magma_int_t info = 0; fuzz_zpotrf_m(g, MagmaLower, n, H(A), n, &info); unpin(); cudaDeviceSynchronize();
    report("potrfm", g, n, info + rinfo, maxdiff(A, R, n, 1), 1e-10);
  }
  else if (r == "hetrd") {
    std::vector<Z> A = herm(n, 0), R = A; magma_int_t nn = n, lw = -1, rinfo = 0; std::vector<double> d(n), e(n), dr(n), er(n); std::vector<Z> tau(n), tr(n); Z wq;
    char cache[256]; snprintf(cache, sizeof cache, "D:\\lean4hpc\\magma\\sweep52\\hetrd_ref_%d_%u.bin", n, seed0);   // the LAPACK reference depends only on (n, seed)
    FILE* fc = fopen(cache, "rb");
    if (fc && fread(dr.data(), 8, n, fc) == (size_t)n && fread(er.data(), 8, n, fc) == (size_t)n) { fclose(fc); }
    else { if (fc) fclose(fc);
      lapackf77_zhetrd("L", &nn, R.data(), &nn, dr.data(), er.data(), tr.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); { std::vector<Z> w(lw); lapackf77_zhetrd("L", &nn, R.data(), &nn, dr.data(), er.data(), tr.data(), w.data(), &lw, &rinfo); }
      if (rinfo == 0 && (fc = fopen(cache, "wb"))) { fwrite(dr.data(), 8, n, fc); fwrite(er.data(), 8, n, fc); fclose(fc); } }
    magma_int_t info = 0, ml = -1; fuzz_zhetrd_mgpu(g, 2, MagmaLower, n, A.data(), n, d.data(), e.data(), tau.data(), &wq, ml, &info);
    ml = (magma_int_t)MAGMA_Z_REAL(wq); std::vector<Z> w(std::max<magma_int_t>(ml, 1)); g_fuzz.count = 0;
    fuzz_zhetrd_mgpu(g, 2, MagmaLower, n, H(A), n, H(d), H(e), H(tau), H(w), ml, &info); unpin(); cudaDeviceSynchronize();
    double em = 0, am = 0; for (int i = 0; i < n; i++) { em = std::max(em, fabs(d[i] - dr[i])); am = std::max(am, fabs(dr[i])); if (i + 1 < n) { em = std::max(em, fabs(fabs(e[i]) - fabs(er[i]))); am = std::max(am, fabs(er[i])); } }
    report("hetrd", g, n, info + rinfo, em / am, 1e-10);
  }
  else if (r == "unmqr" || r == "geqrfm" || r == "ungqr") {
    std::vector<Z> A((size_t)n * n); for (auto& a : A) a = MAGMA_Z_MAKE(r1(), r1()); magma_int_t nn = n, lw = -1, rinfo; Z wq; std::vector<Z> tau(n);
    if (r == "geqrfm") {
      std::vector<Z> R = A, tr(n); lapackf77_zgeqrf(&nn, &nn, R.data(), &nn, tr.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); { std::vector<Z> w(lw); lapackf77_zgeqrf(&nn, &nn, R.data(), &nn, tr.data(), w.data(), &lw, &rinfo); }
      std::vector<Z> w(std::max<magma_int_t>(lw, (magma_int_t)n * 64)); magma_int_t info = 0;
      fuzz_zgeqrf_m(g, n, n, H(A), n, H(tau), H(w), (magma_int_t)w.size(), &info); unpin(); cudaDeviceSynchronize();
      double et = 0; for (int i = 0; i < n; i++) et = std::max(et, MAGMA_Z_ABS(MAGMA_Z_SUB(tau[i], tr[i])));
      report("geqrfm", g, n, info + rinfo, std::max(maxdiff(A, R, n, 2), et), 1e-10);
    } else if (r == "unmqr") {
      int k = n / 2; magma_int_t kk = k; lapackf77_zgeqrf(&nn, &kk, A.data(), &nn, tau.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); { std::vector<Z> w(lw); lapackf77_zgeqrf(&nn, &kk, A.data(), &nn, tau.data(), w.data(), &lw, &rinfo); }
      std::vector<Z> C((size_t)n * n); for (auto& c : C) c = MAGMA_Z_MAKE(r1(), r1()); std::vector<Z> R = C; lw = -1;
      lapackf77_zunmqr("L", "N", &nn, &nn, &kk, A.data(), &nn, tau.data(), R.data(), &nn, &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); { std::vector<Z> w(lw); lapackf77_zunmqr("L", "N", &nn, &nn, &kk, A.data(), &nn, tau.data(), R.data(), &nn, w.data(), &lw, &rinfo); }
      magma_int_t info = 0, ml = -1; fuzz_zunmqr_m(g, MagmaLeft, MagmaNoTrans, n, n, k, A.data(), n, tau.data(), C.data(), n, &wq, ml, &info);
      ml = std::max<magma_int_t>((magma_int_t)MAGMA_Z_REAL(wq), 1); std::vector<Z> w(ml); g_fuzz.count = 0;
      fuzz_zunmqr_m(g, MagmaLeft, MagmaNoTrans, n, n, k, H(A), n, H(tau), H(C), n, H(w), ml, &info); unpin(); cudaDeviceSynchronize();
      report("unmqr", g, n, info + rinfo, maxdiff(C, R, n, 0), 1e-10);
    } else {
      int nb = (int)magma_get_zgeqrf_nb(n, n), ldda = magma_roundup(n, 32); Z *dA, *dT; magma_zmalloc(&dA, (size_t)ldda * n); magma_zmalloc(&dT, (size_t)(2 * n + magma_roundup(n, 32)) * nb);
      magma_zsetmatrix(n, n, A.data(), n, dA, ldda, q); magma_int_t info = 0; magma_zgeqrf_gpu(n, n, dA, ldda, tau.data(), dT, &info);
      magma_zgetmatrix(n, n, dA, ldda, A.data(), n, q); std::vector<Z> T((size_t)nb * n); magma_zgetmatrix(nb, n, dT, nb, T.data(), nb, q); magma_free(dA); magma_free(dT);
      std::vector<Z> R = A; lw = -1; lapackf77_zungqr(&nn, &nn, &nn, R.data(), &nn, tau.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); { std::vector<Z> w(lw); lapackf77_zungqr(&nn, &nn, &nn, R.data(), &nn, tau.data(), w.data(), &lw, &rinfo); }
      g_fuzz.ngpu = g; fuzz_zungqr_m(n, n, n, H(A), n, H(tau), H(T), nb, &info); unpin(); cudaDeviceSynchronize();
      report("ungqr", g, n, info + rinfo, maxdiff(A, R, n, 0), 1e-10);
    }
  }
  else if (r == "heevd2s") {
    std::vector<Z> A = herm(n, 0), R = A, A0 = A; magma_int_t nn = n, rinfo; std::vector<double> w(n), wr(n);
    { magma_int_t lw = -1, lrw = -1, liw = -1, iq; Z wq; double rq; lapackf77_zheevd("N", "L", &nn, R.data(), &nn, wr.data(), &wq, &lw, &rq, &lrw, &iq, &liw, &rinfo);
      lw = (magma_int_t)MAGMA_Z_REAL(wq); lrw = (magma_int_t)rq; liw = iq; std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw);
      lapackf77_zheevd("N", "L", &nn, R.data(), &nn, wr.data(), wk.data(), &lw, rw.data(), &lrw, iw.data(), &liw, &rinfo); }
    magma_int_t mout = 0, info = 0, lw = -1, lrw = -1, liw = -1, iq; Z wq; double rq;
    fuzz_zheevdx_2stage_m(g, MagmaVec, MagmaRangeAll, MagmaLower, n, A.data(), n, 0, 0, 0, 0, &mout, w.data(), &wq, lw, &rq, lrw, &iq, liw, &info);
    lw = (magma_int_t)MAGMA_Z_REAL(wq); lrw = (magma_int_t)rq; liw = iq; std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); g_fuzz.count = 0;
    fuzz_zheevdx_2stage_m(g, MagmaVec, MagmaRangeAll, MagmaLower, n, H(A), n, 0, 0, 0, 0, &mout, H(w), H(wk), lw, H(rw), lrw, H(iw), liw, &info); unpin(); cudaDeviceSynchronize();
    double ew = 0, am = 0; for (int i = 0; i < n; i++) { ew = std::max(ew, fabs(w[i] - wr[i])); am = std::max(am, fabs(wr[i])); } ew /= am;
    std::vector<Z> AV((size_t)n * n); Z one = MAGMA_Z_ONE, zero = MAGMA_Z_ZERO; blasf77_zgemm("N", "N", &nn, &nn, &nn, &one, A0.data(), &nn, A.data(), &nn, &zero, AV.data(), &nn);
    double ev = 0; for (int j = 0; j < (int)mout; j++) for (int i = 0; i < n; i++) ev = std::max(ev, MAGMA_Z_ABS(MAGMA_Z_SUB(AV[i + (size_t)j * n], MAGMA_Z_MAKE(w[j] * MAGMA_Z_REAL(A[i + (size_t)j * n]), w[j] * MAGMA_Z_IMAG(A[i + (size_t)j * n])))));
    ev /= am * n; if (mout != n) ev = INFINITY;
    report("heevd2s", g, n, info + rinfo, std::max(ew, ev), 1e-12);
  }
  else if (r == "gehrd") {
    std::vector<Z> A((size_t)n * n); for (auto& a : A) a = MAGMA_Z_MAKE(r1(), r1()); std::vector<Z> R = A, tau(n), tr(n); magma_int_t nn = n, one = 1, lw = -1, rinfo; Z wq;
    lapackf77_zgehrd(&nn, &one, &nn, R.data(), &nn, tr.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); { std::vector<Z> w(lw); lapackf77_zgehrd(&nn, &one, &nn, R.data(), &nn, tr.data(), w.data(), &lw, &rinfo); }
    int nb = (int)magma_get_zgehrd_nb(n); std::vector<Z> T((size_t)nb * n); magma_int_t info = 0, ml = -1;
    g_fuzz.ngpu = g; fuzz_zgehrd_m(n, 1, n, A.data(), n, tau.data(), &wq, ml, T.data(), &info); ml = std::max<magma_int_t>((magma_int_t)MAGMA_Z_REAL(wq), 1); std::vector<Z> w(ml); g_fuzz.count = 0;
    r52_t0 = magma_wtime(); fuzz_zgehrd_m(n, 1, n, H(A), n, H(tau), H(w), ml, H(T), &info); if (getenv("R52_DBG")) fprintf(stderr, "returned at %.1f ms\n", 1e3*(magma_wtime()-r52_t0)); unpin(); cudaDeviceSynchronize();
    double e = 0, a = 0; int ei = -1, ej = -1, nbad = 0, i0 = n, i1 = -1, j0 = n, j1 = -1;
    for (int j = 0; j < n; j++) for (int i = 0; i <= std::min(n - 1, j + 1); i++) { double d = MAGMA_Z_ABS(MAGMA_Z_SUB(A[i + (size_t)j * n], R[i + (size_t)j * n]));
      if (d > e) { e = d; ei = i; ej = j; } if (d > 1e-8) { nbad++; i0 = std::min(i0, i); i1 = std::max(i1, i); j0 = std::min(j0, j); j1 = std::max(j1, j); } a = std::max(a, MAGMA_Z_ABS(R[i + (size_t)j * n])); }
    if (getenv("R52_DBG")) fprintf(stderr, "gehrd max err at (%d,%d); %d entries > 1e-8 in rows %d..%d cols %d..%d\n", ei, ej, nbad, i0, i1, j0, j1);
    double et = 0; for (int i = 0; i + 1 < n; i++) et = std::max(et, MAGMA_Z_ABS(MAGMA_Z_SUB(tau[i], tr[i])));
    report("gehrd", g, n, info + rinfo, std::max(e / a, et), 1e-10);
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

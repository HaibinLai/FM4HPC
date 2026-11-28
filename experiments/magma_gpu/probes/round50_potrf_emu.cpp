// Round 50 (roxy): magma_[zd]potrf3_mgpu with ngpu virtual devices on one GPU (see potrf_emu/emu_hooks.h).
// Static model (scripts/potrf_mgpu_hb.py): the stream-2 trsm that reads the received diagonal block dlP(d,0,0,buf)
// at iteration k is unordered with the stream-1 copy that overwrites the same buffer at iteration k+ngpu.
//   round50_potrf_emu <z|c|d|s> <U|L> ngpu n nb h kind ms fix [seed] [dev]     kind 0 none, 1/2 every op, 3/4 one op on device dev
// CSV: r50,prec,uplo,ngpu,n,nb,h,kind,ms,fix,info,relerr,delays,fixwaits,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include "emu_hooks.h"
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <type_traits>
EmuState g_emu;
extern "C" magma_int_t emu_zpotrf3_mgpu(magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr*, magma_int_t,
  magmaDoubleComplex_ptr*, magma_int_t, magmaDoubleComplex*, magma_int_t, magma_int_t, magma_queue_t[][3], magma_event_t[][5], magma_int_t*);
extern "C" magma_int_t emu_cpotrf3_mgpu(magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magmaFloatComplex_ptr*, magma_int_t,
  magmaFloatComplex_ptr*, magma_int_t, magmaFloatComplex*, magma_int_t, magma_int_t, magma_queue_t[][3], magma_event_t[][5], magma_int_t*);
extern "C" magma_int_t emu_spotrf3_mgpu(magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magmaFloat_ptr*, magma_int_t,
  magmaFloat_ptr*, magma_int_t, float*, magma_int_t, magma_int_t, magma_queue_t[][3], magma_event_t[][5], magma_int_t*);
extern "C" magma_int_t emu_dpotrf3_mgpu(magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magmaDouble_ptr*, magma_int_t,
  magmaDouble_ptr*, magma_int_t, double*, magma_int_t, magma_int_t, magma_queue_t[][3], magma_event_t[][5], magma_int_t*);
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
typedef std::complex<double> zc; typedef std::complex<float> cc;
static void mk(double& x) { x = r1(); } static void mk(zc& x) { x = zc(r1(), r1()); } static void mk(float& x) { x = (float)r1(); } static void mk(cc& x) { x = cc((float)r1(), (float)r1()); }
static double cj(double x) { return x; } static zc cj(zc x) { return std::conj(x); } static float cj(float x) { return x; } static cc cj(cc x) { return std::conj(x); }
static void ref(char u, int n, double* a, magma_int_t* info) { magma_int_t nn = n; lapackf77_dpotrf(&u, &nn, a, &nn, info); }
static void ref(char u, int n, zc* a, magma_int_t* info) { magma_int_t nn = n; lapackf77_zpotrf(&u, &nn, (magmaDoubleComplex*)a, &nn, info); }
static void ref(char u, int n, float* a, magma_int_t* info) { magma_int_t nn = n; lapackf77_spotrf(&u, &nn, a, &nn, info); }
static void ref(char u, int n, cc* a, magma_int_t* info) { magma_int_t nn = n; lapackf77_cpotrf(&u, &nn, (magmaFloatComplex*)a, &nn, info); }
static magma_int_t emu(int g, magma_uplo_t up, int n, int nb, float** dA, int ldda, float** dP, int lddp, float* w, int lda, int h, magma_queue_t q[][3], magma_event_t e[][5], magma_int_t* info) {
  return emu_spotrf3_mgpu(g, up, n, n, 0, 0, nb, dA, ldda, dP, lddp, w, lda, h, q, e, info); }
static magma_int_t emu(int g, magma_uplo_t up, int n, int nb, cc** dA, int ldda, cc** dP, int lddp, cc* w, int lda, int h, magma_queue_t q[][3], magma_event_t e[][5], magma_int_t* info) {
  return emu_cpotrf3_mgpu(g, up, n, n, 0, 0, nb, (magmaFloatComplex_ptr*)dA, ldda, (magmaFloatComplex_ptr*)dP, lddp, (magmaFloatComplex*)w, lda, h, q, e, info); }
static magma_int_t emu(int g, magma_uplo_t up, int n, int nb, double** dA, int ldda, double** dP, int lddp, double* w, int lda, int h, magma_queue_t q[][3], magma_event_t e[][5], magma_int_t* info) {
  return emu_dpotrf3_mgpu(g, up, n, n, 0, 0, nb, dA, ldda, dP, lddp, w, lda, h, q, e, info); }
static magma_int_t emu(int g, magma_uplo_t up, int n, int nb, zc** dA, int ldda, zc** dP, int lddp, zc* w, int lda, int h, magma_queue_t q[][3], magma_event_t e[][5], magma_int_t* info) {
  return emu_zpotrf3_mgpu(g, up, n, n, 0, 0, nb, (magmaDoubleComplex_ptr*)dA, ldda, (magmaDoubleComplex_ptr*)dP, lddp, (magmaDoubleComplex*)w, lda, h, q, e, info); }

template <class T> static int run(char prec, char uc, int g, int n, int nb, int h, int kind, int ms, int fix) {
  bool upper = uc == 'U'; magma_uplo_t up = upper ? MagmaUpper : MagmaLower;
  std::vector<T> G((size_t)n * n), A((size_t)n * n); for (auto& x : G) mk(x);
  for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { T s = 0; for (int k = 0; k < n; k += 7) s += G[i + (size_t)k * n] * cj(G[j + (size_t)k * n]); A[i + (size_t)j * n] = s + (i == j ? T(n) : T(0)); }
  std::vector<T> R = A; magma_int_t rinfo; ref(uc, n, R.data(), &rinfo);
  int nl[8]; for (int d = 0; d < g; d++) { nl[d] = (n / (nb * g)) * nb; if (d < (n / nb) % g) nl[d] += nb; else if (d == (n / nb) % g) nl[d] += n % nb; }
  int ldda = upper ? magma_roundup(n, 32) : magma_roundup((n / (nb * g) + 1) * nb, 32);
  auto dev = [&](int x) { return (x / nb) % g; }; auto loc = [&](int x) { return (x / (nb * g)) * nb + x % nb; };
  magma_queue_t q[8][3]; magma_event_t ev[8][5]; T* dA[8]; T* dP[8]; int lddp = magma_roundup(n, nb);
  g_emu.ngpu = g; g_emu.kind = kind; g_emu.ms = ms; g_emu.fix = fix;
  for (int d = 0; d < g; d++) {
    size_t cnt = upper ? (size_t)ldda * std::max(nl[d], 1) : (size_t)ldda * n;
    magma_malloc((void**)&dA[d], cnt * sizeof(T)); magma_malloc((void**)&dP[d], (size_t)g * nb * lddp * sizeof(T));
    for (int s = 0; s < 3; s++) magma_queue_create(0, &q[d][s]); for (int s = 0; s < 5; s++) magma_event_create(&ev[d][s]);
    g_emu.abase[d] = (char*)dA[d]; g_emu.abytes[d] = cnt * sizeof(T); g_emu.pbase[d] = (char*)dP[d]; g_emu.pbytes[d] = (size_t)g * nb * lddp * sizeof(T);
    for (int s = 0; s < 3; s++) g_emu.q[d][s] = q[d][s]; cudaEventCreateWithFlags(&g_emu.ev[d], cudaEventDisableTiming); g_emu.rec[d] = 0;
    std::vector<T> L(cnt, T(0));
    if (upper) { for (int J = 0; J < n; J++) if (dev(J) == d) for (int i = 0; i < n; i++) L[i + (size_t)loc(J) * ldda] = A[i + (size_t)J * n]; }
    else { for (int I = 0; I < n; I++) if (dev(I) == d) for (int j = 0; j < n; j++) L[loc(I) + (size_t)j * ldda] = A[I + (size_t)j * n]; }
    magma_setvector((magma_int_t)cnt, sizeof(T), L.data(), 1, dA[d], 1, q[d][0]); magma_queue_sync(q[d][0]); }
  cudaEventCreate(&g_emu.t0); cudaEventCreate(&g_emu.tA); for (auto& x : g_emu.tB) cudaEventCreate(&x);
  T* w; magma_malloc_pinned((void**)&w, (size_t)n * nb * h * sizeof(T));
  g_emu.hbase = (char*)w; g_emu.esz = sizeof(T); g_emu.hld = upper ? n : nb * h;
  magma_int_t info = 0; emu(g, up, n, nb, dA, ldda, dP, lddp, w, upper ? n : nb * h, h, q, ev, &info);
  for (int d = 0; d < g; d++) for (int s = 0; s < 3; s++) magma_queue_sync(q[d][s]);
  if (g_emu.tset) { float ms0, ms1; cudaEventElapsedTime(&ms0, g_emu.t0, g_emu.tA); fprintf(stderr, "delayed op on vdev %d ran %.3f ms (start..end)\n", g_emu.tdev, ms0);
    for (int b = 0; b < g_emu.nB; b++) { cudaEventElapsedTime(&ms1, g_emu.t0, g_emu.tB[b]); fprintf(stderr, g_emu.kind == 6 ? "  later D2H into the same host rows %d (vdev %d) finished at %+.3f ms relative to its start\n" : "  later dlP copy %d on vdev %d finished at %+.3f ms relative to its start\n", b, g_emu.tdev, ms1); } }
  double e = 0, an = 0; int eb_i = -1, eb_j = -1;
  for (int d = 0; d < g; d++) { size_t cnt = g_emu.abytes[d] / sizeof(T); std::vector<T> L(cnt); magma_getvector((magma_int_t)cnt, sizeof(T), dA[d], 1, L.data(), 1, q[d][0]);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { if (upper ? i > j : i < j) continue; int owner = dev(upper ? j : i); if (owner != d) continue;
      T x = upper ? L[i + (size_t)loc(j) * ldda] : L[loc(i) + (size_t)j * ldda]; T r = R[i + (size_t)j * n]; double dd = std::abs(x - r); if (!(dd <= e)) { e = dd; eb_i = i / nb; eb_j = j / nb; } an = std::max(an, (double)std::abs(r)); } }
  if (e / an > (std::is_same<T, float>::value || std::is_same<T, cc>::value ? 1e-5 : 1e-10)) fprintf(stderr, "max error at block (%d,%d)\n", eb_i, eb_j);
  double rel = e / an; const double tol = std::is_same<T, float>::value || std::is_same<T, cc>::value ? 1e-5 : 1e-10; const char* v = (info == 0 && rinfo == 0 && rel < tol) ? "OK" : "WRONG";
  printf("r50,%c,%c,%d,%d,%d,%d,%d,%d,%d,%lld,%.3g,%lld,%lld,%s\n", prec, uc, g, n, nb, h, kind, ms, fix, (long long)info, rel, g_emu.n_delay, g_emu.n_fixwait, v);
  for (int d = 0; d < g; d++) { magma_free(dA[d]); magma_free(dP[d]); for (int s = 0; s < 3; s++) magma_queue_destroy(q[d][s]); for (int s = 0; s < 5; s++) magma_event_destroy(ev[d][s]); cudaEventDestroy(g_emu.ev[d]); }
  magma_free_pinned(w); return 0;
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 10) { printf("usage: round50_potrf_emu <z|d> <U|L> ngpu n nb h kind ms fix [seed]\n"); return 1; }
  char p = argv[1][0], u = argv[2][0]; int g = atoi(argv[3]), n = atoi(argv[4]), nb = atoi(argv[5]), h = atoi(argv[6]), kind = atoi(argv[7]), ms = atoi(argv[8]), fix = atoi(argv[9]);
  if (argc > 10) Sd = (unsigned)atoi(argv[10]); if (argc > 11) g_emu.dev = atoi(argv[11]); if (g < 1 || g > 8) return 1;
  magma_init(); int r = p == 'z' ? run<zc>(p, u, g, n, nb, h, kind, ms, fix) : p == 'c' ? run<cc>(p, u, g, n, nb, h, kind, ms, fix) : p == 's' ? run<float>(p, u, g, n, nb, h, kind, ms, fix) : run<double>(p, u, g, n, nb, h, kind, ms, fix);
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_finalize(); return r;
}

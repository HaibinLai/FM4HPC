// Fourteenth GPU round: the items that were still backed only by reading (HANDOFF §29.14).  GPU 0 only; one mode per
// process.  Helper modes take k and are run under LD_PRELOAD=fail_malloc.so (k = 0: count only).
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
typedef magmaDoubleComplex Z;
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 2468u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static std::vector<Z> randm(int m, int n) { std::vector<Z> A((size_t)m * n); for (auto& a : A) a = C(rnd(), rnd()); return A; }
static double diff(const std::vector<Z>& a, const std::vector<Z>& b) { double e = 0, m = 0; for (size_t i = 0; i < a.size(); i++) {
  if (!std::isfinite(MAGMA_Z_REAL(a[i])) || !std::isfinite(MAGMA_Z_IMAG(a[i]))) return INFINITY; e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(a[i], b[i]))); m = fmax(m, MAGMA_Z_ABS(b[i])); } return m > 0 ? e / m : e; }

// ---- zlascl2: D read before the ind < m guard; D placed at the end of a mapped page, the next page unmapped ----
static void lascl2() {
  CUdevice dev; cuDeviceGet(&dev, 0); CUmemAllocationProp p; memset(&p, 0, sizeof p); p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = 0; size_t g = 0; cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  CUdeviceptr base; cuMemAddressReserve(&base, 2 * g, 0, 0, 0); CUmemGenericAllocationHandle h; cuMemCreate(&h, g, &p, 0); cuMemMap(base, g, 0, h, 0);
  CUmemAccessDesc ad; ad.location = p.location; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; cuMemSetAccess(base, g, &ad, 1);
  for (int m : {128, 100}) {
    double* dD = (double*)(base + g - m * sizeof(double)); std::vector<double> D(m, 2.0); cudaMemcpy(dD, D.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    Z* dA; magma_zmalloc(&dA, (size_t)m * 4); magma_int_t info = -999;
    magmablas_zlascl2(MagmaFull, m, 4, dD, dA, m, Q, &info); cudaError_t e = cudaDeviceSynchronize();
    printf("zlascl2 m=%d (D ends exactly at an unmapped page): info=%lld  cudaDeviceSynchronize: %s\n", m, (long long)info, cudaGetErrorString(e));
    if (e != cudaSuccess) break;
  }
}
// ---- zpotf2_gpu: warp reduction without __syncwarp, repeated runs ----
static void potf2_repeat() {
  int n = 128; std::vector<Z> A = randm(n, n); std::vector<Z> H((size_t)n * n, C(0));
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) H[i + (size_t)j * n] = MAGMA_Z_ADD(H[i + (size_t)j * n], MAGMA_Z_MUL(A[i + (size_t)k * n], MAGMA_Z_CONJ(A[j + (size_t)k * n])));
  for (int i = 0; i < n; i++) H[i + (size_t)i * n] = MAGMA_Z_ADD(H[i + (size_t)i * n], C(n));
  std::vector<Z> L = H; { magma_int_t nn = n, info; lapackf77_zpotrf("L", &nn, L.data(), &nn, &info); }
  Z* d; magma_zmalloc(&d, (size_t)n * n); std::vector<Z> F((size_t)n * n), F0; int differ = 0, bad = 0; std::map<long, int> infos;
  for (int r = 0; r < 1000; r++) { magma_zsetmatrix(n, n, H.data(), n, d, n, Q); magma_int_t info = -999; magma_zpotf2_gpu(MagmaLower, n, d, n, Q, &info); infos[(long)info]++;
    magma_zgetmatrix(n, n, d, n, F.data(), n, Q); for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) F[i + (size_t)j * n] = L[i + (size_t)j * n];
    if (r == 0) F0 = F; else if (diff(F, F0) != 0) differ++; if (diff(F, L) > 1e-10) bad++; }
  printf("zpotf2_gpu n=128 HPD, 1000 runs: info histogram:"); for (auto& x : infos) printf(" %ld x%d", x.first, x.second);
  printf("; runs differing bitwise from run 0: %d; runs with relative error > 1e-10 against LAPACK: %d\n", differ, bad);
}
// ---- zlag2c: one global __device__ flag shared by concurrent calls on different queues ----
static void lag2c_race() {
  int n = 512; std::vector<Z> Abig((size_t)n * n, C(1.0)); Abig[7] = C(1e39); std::vector<Z> Aok((size_t)n * n, C(1.0));
  Z *dBig, *dOk; magmaFloatComplex *sBig, *sOk; magma_zmalloc(&dBig, (size_t)n * n); magma_zmalloc(&dOk, (size_t)n * n); magma_cmalloc(&sBig, (size_t)n * n); magma_cmalloc(&sOk, (size_t)n * n);
  magma_zsetmatrix(n, n, Abig.data(), n, dBig, n, Q); magma_zsetmatrix(n, n, Aok.data(), n, dOk, n, Q);
  int wrongOk = 0, wrongBig = 0, R = 2000;
  std::thread t1([&] { magma_queue_t q; magma_queue_create(0, &q); for (int r = 0; r < R; r++) { magma_int_t i = -9; magmablas_zlag2c(n, n, dBig, n, sBig, n, q, &i); if (i != 1) wrongBig++; } magma_queue_destroy(q); });
  std::thread t2([&] { magma_queue_t q; magma_queue_create(0, &q); for (int r = 0; r < R; r++) { magma_int_t i = -9; magmablas_zlag2c(n, n, dOk, n, sOk, n, q, &i); if (i != 0) wrongOk++; } magma_queue_destroy(q); });
  t1.join(); t2.join();
  printf("zlag2c from two host threads, two queues, %d calls each: overflowing matrix reported 0 in %d calls; clean matrix reported 1 in %d calls\n", R, wrongBig, wrongOk);
}
// ---- zheevj: max_sweeps not checked; noconv race ----
static void heevj() {
  int n = 16, b = 64; std::vector<std::vector<Z>> M(b); for (auto& A : M) { A.resize((size_t)n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) {
      Z v = i == j ? C(rnd() * 4) : C(rnd(), rnd()); A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = MAGMA_Z_CONJ(v); } }
  std::vector<std::vector<double>> Wref(b, std::vector<double>(n)); for (int k = 0; k < b; k++) { std::vector<Z> A = M[k]; magma_int_t nn = n, lw = 4 * n, info; std::vector<Z> wk(lw); std::vector<double> rw(3 * n);
    lapackf77_zheev("N", "L", &nn, A.data(), &nn, Wref[k].data(), wk.data(), &lw, rw.data(), &info); }
  std::vector<Z*> d(b); std::vector<double*> dW(b); for (int k = 0; k < b; k++) { magma_zmalloc(&d[k], (size_t)n * n); magma_dmalloc(&dW[k], n); }
  Z** dA; double** dWa; magma_malloc((void**)&dA, b * sizeof(Z*)); magma_malloc((void**)&dWa, b * sizeof(double*)); magma_setvector(b, sizeof(Z*), d.data(), 1, dA, 1, Q); magma_setvector(b, sizeof(double*), dW.data(), 1, dWa, 1, Q);
  magma_int_t* dinfo; magma_imalloc(&dinfo, b); int* dsw; magma_malloc((void**)&dsw, b * sizeof(int));
  auto run = [&](int maxsw, int& nonzero, int& wrong) { for (int k = 0; k < b; k++) magma_zsetmatrix(n, n, M[k].data(), n, d[k], n, Q);
    magma_zheevj_batched_expert_small_sm(MagmaNoVec, MagmaLower, n, dA, n, dWa, dinfo, NULL, dsw, 30.0, 1, maxsw, 4, b, Q);
    std::vector<magma_int_t> inf(b); magma_getvector(b, sizeof(magma_int_t), dinfo, 1, inf.data(), 1, Q); nonzero = 0; wrong = 0;
    for (int k = 0; k < b; k++) { if (inf[k] != 0) nonzero++; std::vector<double> w(n); magma_dgetvector(n, dW[k], 1, w.data(), 1, Q); std::sort(w.begin(), w.end());
      double e = 0, mx = 0; for (int i = 0; i < n; i++) { e = fmax(e, fabs(w[i] - Wref[k][i])); mx = fmax(mx, fabs(Wref[k][i])); } if (!(e <= 1e-10 * mx)) wrong++; } };
  int nz, wr; run(0, nz, wr); printf("zheevj expert, 64 Hermitian 16x16, max_sweeps = 0: matrices with info != 0: %d; with wrong eigenvalues: %d\n", nz, wr);
  run(1, nz, wr); printf("                                     max_sweeps = 1: info != 0: %d; wrong eigenvalues: %d\n", nz, wr);
  int tnz = 0, twr = 0; for (int r = 0; r < 200; r++) { run(100, nz, wr); tnz += nz; twr += wr; }
  printf("                                     max_sweeps = 100, 200 repetitions (race check): info != 0 in %d, wrong eigenvalues in %d of %d matrix solves\n", tnz, twr, 200 * b);
}
// ---- sparse: wide CPU transpose; truncated .mtx ----
static void sparse_items() {
  int w = 1000; std::vector<magma_index_t> row = {0, (magma_index_t)w}, col(w); std::vector<Z> val(w); for (int j = 0; j < w; j++) { col[j] = j; val[j] = C(j + 1); }
  magma_z_matrix A = {Magma_CSR}, T = {Magma_CSR}; magma_zcsrset(1, w, row.data(), col.data(), val.data(), &A, Q);
  magma_int_t r = 0;
  if (!getenv("MTX_ONLY")) { printf("zmtranspose of a 1x%d CPU CSR ...\n", w); fflush(stdout); r = magma_zmtranspose(A, &T, Q);
  printf("  ret=%lld dims %lld x %lld; freeing the result ...\n", (long long)r, (long long)T.num_rows, (long long)T.num_cols); fflush(stdout); magma_zmfree(&T, Q); printf("  freed\n"); }
  const char* f = "truncated.mtx"; FILE* fp = fopen(f, "w"); fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n3 3 3\n1 1 1.0\n2 2 1.0\n"); fclose(fp);
  magma_z_matrix B = {Magma_CSR}; r = magma_z_csr_mtx(&B, f, Q);
  printf("magma_z_csr_mtx of a file announcing 3 entries but containing 2: ret=%lld nnz=%lld", (long long)r, (long long)B.nnz);
  if (r == 0 && B.val) { printf("  entries:"); for (int i = 0; i < B.nnz; i++) printf(" (%d,%g)", (int)B.col[i], MAGMA_Z_REAL(B.val[i])); } printf("\n");
}
// ---- helpers under fault injection ----
static void (*fi_arm)(long) = 0; static void (*fi_disarm)() = 0; static long (*fi_count)() = 0;
static void helper(const std::string& h, long k) {
  int n = 400; magma_int_t nn = n, info = -999, lw = n * 256; std::vector<Z> A = randm(n, n), tau(n), C0 = randm(n, n), wk(lw), ref, out; long cnt = -1;
  auto arm = [&]() { if (fi_arm) fi_arm(k); }; auto disarm = [&]() { cnt = fi_count ? fi_count() : -1; if (fi_disarm) fi_disarm(); };
  if (h == "unmqr" || h == "unmql" || h == "unmlq" || h == "ungqr2") {
    std::vector<Z> w2(lw); magma_int_t inf;
    if (h == "unmql") lapackf77_zgeqlf(&nn, &nn, A.data(), &nn, tau.data(), w2.data(), &lw, &inf);
    else if (h == "unmlq") lapackf77_zgelqf(&nn, &nn, A.data(), &nn, tau.data(), w2.data(), &lw, &inf);
    else lapackf77_zgeqrf(&nn, &nn, A.data(), &nn, tau.data(), w2.data(), &lw, &inf);
    auto call = [&](std::vector<Z>& X) {
      if (h == "unmqr") magma_zunmqr(MagmaLeft, MagmaNoTrans, n, n, n, A.data(), n, tau.data(), X.data(), n, wk.data(), lw, &info);
      else if (h == "unmql") magma_zunmql(MagmaLeft, MagmaNoTrans, n, n, n, A.data(), n, tau.data(), X.data(), n, wk.data(), lw, &info);
      else if (h == "unmlq") magma_zunmlq(MagmaLeft, MagmaNoTrans, n, n, n, A.data(), n, tau.data(), X.data(), n, wk.data(), lw, &info);
      else magma_zungqr2(n, n, n, X.data(), n, tau.data(), &info); };
    std::vector<Z> start = h == "ungqr2" ? A : C0; ref = start; call(ref); out = start; info = -999; arm(); call(out); disarm();
  } else if (h == "trevc3" || h == "trevc3_mt") {
    std::vector<Z> T((size_t)n * n, C(0)); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) T[i + (size_t)j * n] = i == j ? C(j + 1.0, 0.5 * rnd()) : C(rnd(), rnd());
    std::vector<double> rw(n); magma_int_t m; Z dummy; auto call = [&](std::vector<Z>& VR) {
      if (h == "trevc3") magma_ztrevc3(MagmaRight, MagmaAllVec, NULL, n, T.data(), n, &dummy, 1, VR.data(), n, n, &m, wk.data(), lw, rw.data(), &info);
      else magma_ztrevc3_mt(MagmaRight, MagmaAllVec, NULL, n, T.data(), n, &dummy, 1, VR.data(), n, n, &m, wk.data(), lw, rw.data(), &info); };
    ref.assign((size_t)n * n, C(0)); call(ref); out.assign((size_t)n * n, C(0)); info = -999; arm(); call(out); disarm();
  }
  double e = diff(out, ref); const char* v = (info == 0 && !(e <= 1e-12)) ? "SILENT-WRONG" : (info != 0 ? "reported" : "ok");
  printf("%-10s k=%3ld nalloc=%3ld info=%-6lld diff_to_unfailed=%-10.3g %s\n", h.c_str(), k, cnt, (long long)info, e, v);
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; long k = argc > 2 ? atol(argv[2]) : 0;
  fi_arm = (void (*)(long))dlsym(RTLD_DEFAULT, "fi_arm"); fi_disarm = (void (*)())dlsym(RTLD_DEFAULT, "fi_disarm"); fi_count = (long (*)())dlsym(RTLD_DEFAULT, "fi_count");
  magma_init(); magma_queue_create(0, &Q);
  if (m == "lascl2") lascl2(); else if (m == "potf2") potf2_repeat(); else if (m == "lag2c") lag2c_race(); else if (m == "heevj") heevj(); else if (m == "sparse") sparse_items();
  else helper(m, k);
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

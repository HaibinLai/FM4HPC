// Seventeenth GPU round: the four sparse ids that were not reachable through the public headers (called through
// their exported symbols), the sync-free solve with correct CSC arrays, and fault injection on the remaining public
// dense helpers that allocate (HANDOFF §29.17).  GPU 0 only.
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z;
extern "C" magma_int_t magma_zcsrget_gpu(magma_z_matrix, magma_int_t*, magma_int_t*, magmaIndex_ptr*, magmaIndex_ptr*, magmaDoubleComplex_ptr*, magma_queue_t);
extern "C" magma_int_t magma_zorderstatistics(magmaDoubleComplex*, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex*, magma_queue_t);
extern "C" magma_int_t magma_zgecscsyncfreetrsm_analysis(magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magmaIndex_ptr, magmaIndex_ptr, magmaIndex_ptr, magmaIndex_ptr, magma_queue_t);
extern "C" magma_int_t magma_zgecscsyncfreetrsm_solve(magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magmaIndex_ptr, magmaIndex_ptr, magmaIndex_ptr, magmaIndex_ptr,
                                                     magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magma_queue_t);
magma_int_t magma_ztrisolve_analysis(magma_z_matrix M, magma_solve_info_t* solve_info, bool upper_triangular, bool unit_diagonal, bool transpose, magma_queue_t queue);
magma_int_t magma_ztrisolve(magma_z_matrix M, magma_solve_info_t solve_info, bool upper_triangular, bool unit_diagonal, bool transpose, magma_z_matrix b, magma_z_matrix x, magma_queue_t queue);
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 1357u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
template <class T> static T* devv(const std::vector<T>& v) { T* d; cudaMalloc((void**)&d, std::max<size_t>(v.size(), 1) * sizeof(T)); cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }
template <class T> static std::vector<T> hostv(const T* d, int n) { std::vector<T> v(n); cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }

static void syncfree(int n) {   // lower bidiagonal L = I + subdiag(1), CSC: column j has rows j, j+1; b = 1 ⇒ x alternates 1, 0, 1, 0
  std::vector<magma_index_t> cp(n + 1), ri; std::vector<Z> v; for (int j = 0; j < n; j++) { cp[j] = ri.size(); ri.push_back(j); v.push_back(C(1)); if (j + 1 < n) { ri.push_back(j + 1); v.push_back(C(1)); } }
  cp[n] = ri.size(); int nnz = ri.size(); auto dv = devv(v); auto dc = devv(cp); auto dr = devv(ri); magma_index_t *g, *gb; magma_index_malloc(&g, n); magma_index_malloc(&gb, n);
  auto db = devv(std::vector<Z>(n, C(1))); auto dx = devv(std::vector<Z>(n, C(0)));
  magma_int_t a = magma_zgecscsyncfreetrsm_analysis(n, nnz, dv, dc, dr, g, gb, Q);
  magma_int_t s = magma_zgecscsyncfreetrsm_solve(n, nnz, dv, dc, dr, g, gb, dx, db, 0, 1, Q); cudaError_t e = cudaDeviceSynchronize();
  auto x = hostv(dx, n); int wrong = 0; for (int i = 0; i < n; i++) if (MAGMA_Z_REAL(x[i]) != (i % 2 == 0 ? 1.0 : 0.0)) wrong++;
  printf("#17 sync-free forward solve with correct CSC arrays, lower bidiagonal n=%d, b=1 (exact x = 1,0,1,0,...): analysis=%lld solve=%lld sync=%s; wrong entries: %d; x[0..3] = %g %g %g %g\n",
         n, (long long)a, (long long)s, cudaGetErrorString(e), wrong, MAGMA_Z_REAL(x[0]), n > 1 ? MAGMA_Z_REAL(x[1]) : 0., n > 2 ? MAGMA_Z_REAL(x[2]) : 0., n > 3 ? MAGMA_Z_REAL(x[3]) : 0.); }
static void csrget_gpu() { std::vector<magma_index_t> r = {0, 1, 2}, c = {0, 1}; std::vector<Z> v = {C(3), C(4)}; magma_z_matrix A = {Magma_CSR}; magma_zcsrset(2, 2, r.data(), c.data(), v.data(), &A, Q);
  magma_int_t m, n; magmaIndex_ptr dr, dc; Z* dv; magma_int_t ret = magma_zcsrget_gpu(A, &m, &n, &dr, &dc, &dv, Q); Z h[2]; cudaError_t e = cudaMemcpy(h, dv, 2 * sizeof(Z), cudaMemcpyDeviceToHost);
  printf("#5 zcsrget_gpu of a CPU CSR [[3,0],[0,4]]: ret=%lld m=%lld n=%lld; cudaMemcpy from the returned val: %s (%g, %g)\n", (long long)ret, (long long)m, (long long)n, cudaGetErrorString(e), MAGMA_Z_REAL(h[0]), MAGMA_Z_REAL(h[1])); cudaGetLastError(); }
static void orderstat() { for (int order = 0; order < 2; order++) { std::vector<Z> v = order ? std::vector<Z>{C(NAN), C(1)} : std::vector<Z>{C(1), C(NAN)}; Z el = C(-99);
    magma_int_t r = magma_zorderstatistics(v.data(), 2, 1, 0, &el, Q); printf("#7 zorderstatistics(%s, k=1, r=0): ret=%lld element=%g (MAGMA_ERR_NAN = %d)\n", order ? "{NaN, 1}" : "{1, NaN}", (long long)r, MAGMA_Z_REAL(el), MAGMA_ERR_NAN); } }
static void rowentries() { std::vector<Z> v = {C(0, 1), C(1)}; magma_z_matrix A = {Magma_CSR}; magma_zvset(1, 2, v.data(), &A, Q); A.storage_type = Magma_DENSE; A.num_rows = 1; A.num_cols = 2;
  magma_int_t r = magma_zrowentries(&A, Q); printf("#3 zrowentries on the DENSE row (i, 1): ret=%lld max_nnz_row=%lld (expected 2)\n", (long long)r, (long long)A.max_nnz_row); }
static void trisolve() { std::vector<magma_index_t> r = {0, 0, 2}, c = {0, 1}; std::vector<Z> v = {C(1), C(1)};   // L = [[0,0],[1,1]]: row 0 has no stored entry (zero diagonal)
  magma_z_matrix h = {Magma_CSR}, L = {Magma_CSR}; magma_zcsrset(2, 2, r.data(), c.data(), v.data(), &h, Q); magma_zmtransfer(h, &L, Magma_CPU, Magma_DEV, Q);
  magma_z_matrix b = {Magma_CSR}, x = {Magma_CSR}; magma_zvinit(&b, Magma_DEV, 2, 1, C(1), Q); magma_zvinit(&x, Magma_DEV, 2, 1, C(0), Q);
  magma_solve_info_t si; memset(&si, 0, sizeof si); magma_int_t a = magma_ztrisolve_analysis(L, &si, false, false, false, Q);
  magma_int_t s = magma_ztrisolve(L, si, false, false, false, b, x, Q); auto xv = hostv(x.dval, 2);
  printf("#14 ztrisolve, L = [[0,0],[1,1]] (zero diagonal), b = (1,1): analysis=%lld solve=%lld x = (%g, %g)\n", (long long)a, (long long)s, MAGMA_Z_REAL(xv[0]), MAGMA_Z_REAL(xv[1])); }
// ---- fault injection on the remaining public helpers that allocate ----
static void (*fi_arm)(long) = 0; static void (*fi_disarm)() = 0; static long (*fi_count)() = 0;
static double diff(const std::vector<Z>& a, const std::vector<Z>& b) { double e = 0, m = 0; for (size_t i = 0; i < a.size(); i++) { if (!std::isfinite(MAGMA_Z_REAL(a[i]))) return INFINITY;
  e = fmax(e, MAGMA_Z_ABS(MAGMA_Z_SUB(a[i], b[i]))); m = fmax(m, MAGMA_Z_ABS(b[i])); } return m > 0 ? e / m : e; }
static void helper(const std::string& h, long k) {
  int n = 400; magma_int_t nn = n, info = -999, nb = magma_get_zgeqrf_nb(n, n), lw = n * 256; std::vector<Z> A((size_t)n * n), tau(n), C0((size_t)n * n), wk(lw), ref, out; long cnt = -1;
  for (auto& a : A) a = C(rnd(), rnd()); for (auto& c : C0) c = C(rnd(), rnd());
  auto arm = [&]() { if (fi_arm) fi_arm(k); }; auto disarm = [&]() { cnt = fi_count ? fi_count() : -1; if (fi_disarm) fi_disarm(); };
  if (h == "ungqr" || h == "ungqr_gpu") {
    Z *dA, *dT; magma_zmalloc(&dA, (size_t)n * n); magma_zmalloc(&dT, (size_t)(2 * n + magma_roundup(n, 32)) * nb); magma_zsetmatrix(n, n, A.data(), n, dA, n, Q);
    magma_int_t inf; magma_zgeqrf_gpu(n, n, dA, n, tau.data(), dT, &inf); std::vector<Z> F((size_t)n * n); magma_zgetmatrix(n, n, dA, n, F.data(), n, Q);
    if (h == "ungqr") { ref = F; magma_zungqr(n, n, n, ref.data(), n, tau.data(), dT, nb, &inf); out = F; info = -999; arm(); magma_zungqr(n, n, n, out.data(), n, tau.data(), dT, nb, &info); disarm(); }
    else { magma_zsetmatrix(n, n, F.data(), n, dA, n, Q); magma_zungqr_gpu(n, n, n, dA, n, tau.data(), dT, nb, &inf); ref.resize((size_t)n * n); magma_zgetmatrix(n, n, dA, n, ref.data(), n, Q);
      magma_zsetmatrix(n, n, F.data(), n, dA, n, Q); info = -999; arm(); magma_zungqr_gpu(n, n, n, dA, n, tau.data(), dT, nb, &info); disarm(); out.resize((size_t)n * n); magma_zgetmatrix(n, n, dA, n, out.data(), n, Q); }
  } else if (h == "unmrq") {
    magma_int_t inf; lapackf77_zgerqf(&nn, &nn, A.data(), &nn, tau.data(), wk.data(), &lw, &inf);
    ref = C0; magma_zunmrq(MagmaLeft, MagmaNoTrans, n, n, n, A.data(), n, tau.data(), ref.data(), n, wk.data(), lw, &inf);
    out = C0; info = -999; arm(); magma_zunmrq(MagmaLeft, MagmaNoTrans, n, n, n, A.data(), n, tau.data(), out.data(), n, wk.data(), lw, &info); disarm();
  } else if (h == "unmqr2_gpu" || h == "unmql2_gpu") {
    magma_int_t inf; if (h == "unmqr2_gpu") lapackf77_zgeqrf(&nn, &nn, A.data(), &nn, tau.data(), wk.data(), &lw, &inf); else lapackf77_zgeqlf(&nn, &nn, A.data(), &nn, tau.data(), wk.data(), &lw, &inf);
    Z *dA, *dC; magma_zmalloc(&dA, (size_t)n * n); magma_zmalloc(&dC, (size_t)n * n); magma_zsetmatrix(n, n, A.data(), n, dA, n, Q);
    auto call = [&](magma_int_t* inf2) { if (h == "unmqr2_gpu") magma_zunmqr2_gpu(MagmaLeft, MagmaNoTrans, n, n, n, dA, n, tau.data(), dC, n, A.data(), n, inf2);
      else magma_zunmql2_gpu(MagmaLeft, MagmaNoTrans, n, n, n, dA, n, tau.data(), dC, n, A.data(), n, inf2); };
    magma_zsetmatrix(n, n, C0.data(), n, dC, n, Q); call(&inf); ref.resize((size_t)n * n); magma_zgetmatrix(n, n, dC, n, ref.data(), n, Q);
    magma_zsetmatrix(n, n, C0.data(), n, dC, n, Q); info = -999; arm(); call(&info); disarm(); out.resize((size_t)n * n); magma_zgetmatrix(n, n, dC, n, out.data(), n, Q);
  } else if (h == "gerbt_gpu") {
    Z *dA, *dB; magma_zmalloc(&dA, (size_t)n * n); magma_zmalloc(&dB, n); std::vector<Z> U(2 * n), V(2 * n), B(n); for (auto& u : U) u = C(1 + 0.1 * rnd()); for (auto& x : V) x = C(1 + 0.1 * rnd()); for (auto& x : B) x = C(rnd());
    magma_int_t inf; magma_zsetmatrix(n, n, A.data(), n, dA, n, Q); magma_zsetmatrix(n, 1, B.data(), n, dB, n, Q); magma_zgerbt_gpu(MagmaFalse, n, 1, dA, n, dB, n, U.data(), V.data(), &inf);
    ref.resize((size_t)n * n); magma_zgetmatrix(n, n, dA, n, ref.data(), n, Q);
    magma_zsetmatrix(n, n, A.data(), n, dA, n, Q); magma_zsetmatrix(n, 1, B.data(), n, dB, n, Q); info = -999; arm(); magma_zgerbt_gpu(MagmaFalse, n, 1, dA, n, dB, n, U.data(), V.data(), &info); disarm();
    out.resize((size_t)n * n); magma_zgetmatrix(n, n, dA, n, out.data(), n, Q);
  }
  double e = diff(out, ref); const char* v = (info == 0 && !(e <= 1e-12)) ? "SILENT-WRONG" : (info != 0 ? "reported" : "ok");
  printf("%-11s k=%3ld nalloc=%3ld info=%-6lld diff_to_unfailed=%-10.3g %s\n", h.c_str(), k, cnt, (long long)info, e, v);
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; long k = argc > 2 ? atol(argv[2]) : 0;
  fi_arm = (void (*)(long))dlsym(RTLD_DEFAULT, "fi_arm"); fi_disarm = (void (*)())dlsym(RTLD_DEFAULT, "fi_disarm"); fi_count = (long (*)())dlsym(RTLD_DEFAULT, "fi_count");
  magma_init(); magma_queue_create(0, &Q);
  if (m == "syncfree2") syncfree(2); else if (m == "syncfree4096") syncfree(4096); else if (m == "csrget_gpu") csrget_gpu(); else if (m == "orderstat") orderstat();
  else if (m == "rowentries") rowentries(); else if (m == "trisolve") trisolve(); else helper(m, k);
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

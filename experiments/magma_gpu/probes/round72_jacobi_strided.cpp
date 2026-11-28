// Round 72 (roxy): strided interfaces of the batched Jacobi solvers against their pointer-array forms and LAPACK.
//   round72_jacobi_strided gesvj <m> <n> <batch>   magma_zgesvj_batched (arrays) and magma_zgesvj_expert_batched_strided
//                                                   (jobu = jobv = Vec, strides padded by 5): S against LAPACK zgesvd, |A V - U S|
//   round72_jacobi_strided heevj <n> <uplo> <batch> magma_zheevj_batched_small_sm and _strided_small_sm (jobz Vec): W against
//                                                   LAPACK zheevd, |A V - V W|
// CSV: r72,op,a,b,batch,info_arr,info_str,err_arr,err_str,diff_arr_str,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
typedef magmaDoubleComplex Z;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static magma_queue_t q;
template <class T> static T** ptrs(T* base, size_t stride, int bc) { std::vector<T*> h(bc); for (int b = 0; b < bc; b++) h[b] = base + b * stride; T** d; magma_malloc((void**)&d, bc * sizeof(T*)); magma_setvector(bc, sizeof(T*), h.data(), 1, d, 1, q); return d; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 5) return 1; std::string op = argv[1]; magma_init(); magma_queue_create(0, &q);
  long long ia = 0, is = 0; double ea = 0, es = 0, dd = 0; int bc;
  if (op == "gesvj") {
    magma_int_t m = atoi(argv[2]), n = atoi(argv[3]); bc = atoi(argv[4]); magma_int_t mn = std::min(m, n), ld = m;
    size_t sA = (size_t)ld * n + 5, sU = (size_t)ld * mn + 5, sV = (size_t)n * n + 5, sS = mn + 5;
    std::vector<Z> A(sA * bc); for (auto& a : A) a = MAGMA_Z_MAKE(r1(), r1());
    std::vector<double> Sref((size_t)mn * bc);
    for (int b = 0; b < bc; b++) { std::vector<Z> W(A.begin() + b * sA, A.begin() + b * sA + (size_t)ld * n); magma_int_t lw = -1, info; Z wq; std::vector<double> rw(5 * mn + 5); Z du;
      lapackf77_zgesvd("N", "N", &m, &n, W.data(), &ld, &Sref[(size_t)b * mn], &du, &ld, &du, &n, &wq, &lw, rw.data(), &info); lw = (magma_int_t)MAGMA_Z_REAL(wq); std::vector<Z> wk(lw);
      lapackf77_zgesvd("N", "N", &m, &n, W.data(), &ld, &Sref[(size_t)b * mn], &du, &ld, &du, &n, wk.data(), &lw, rw.data(), &info); }
    double res[2] = {0, 0}; std::vector<double> Sout[2];
    for (int v = 0; v < 2; v++) {
      Z *dA, *dU, *dV; double* dS; magma_int_t* dinfo; magma_zmalloc(&dA, sA * bc); magma_zmalloc(&dU, sU * bc); magma_zmalloc(&dV, sV * bc); magma_dmalloc(&dS, sS * bc); magma_imalloc(&dinfo, bc);
      magma_zsetvector((magma_int_t)(sA * bc), A.data(), 1, dA, 1, q);
      if (v == 0) { Z** pA = ptrs(dA, sA, bc); Z** pU = ptrs(dU, sU, bc); Z** pV = ptrs(dV, sV, bc); double** pS = ptrs(dS, sS, bc);
        magma_zgesvj_batched(MagmaVec, MagmaVec, m, n, pA, ld, pS, pU, ld, pV, n, dinfo, bc, q); }
      else { int64_t lw = -1; magma_zgesvj_expert_batched_strided(MagmaVec, MagmaVec, m, n, NULL, ld, sA, NULL, sS, NULL, ld, sU, NULL, n, sV, NULL, NULL, &lw, bc, q);
        void* w = NULL; if (lw > 0) magma_malloc(&w, lw);
        magma_zgesvj_expert_batched_strided(MagmaVec, MagmaVec, m, n, dA, ld, sA, dS, sS, dU, ld, sU, dV, n, sV, dinfo, w, &lw, bc, q); }
      magma_queue_sync(q); std::vector<magma_int_t> hi(bc); magma_igetvector(bc, dinfo, 1, hi.data(), 1, q); for (auto x : hi) if (x) (v ? is : ia) = x;
      std::vector<double> S(sS * bc); magma_dgetvector((magma_int_t)S.size(), dS, 1, S.data(), 1, q); std::vector<Z> U(sU * bc), V(sV * bc); magma_zgetvector((magma_int_t)U.size(), dU, 1, U.data(), 1, q); magma_zgetvector((magma_int_t)V.size(), dV, 1, V.data(), 1, q);
      double e = 0; Sout[v].assign((size_t)mn * bc, 0);
      for (int b = 0; b < bc; b++) { std::vector<double> s(S.begin() + b * sS, S.begin() + b * sS + mn); for (int j = 0; j < mn; j++) Sout[v][(size_t)b * mn + j] = s[j]; std::vector<double> ss = s; std::sort(ss.rbegin(), ss.rend());
        for (int j = 0; j < mn; j++) e = std::max(e, fabs(ss[j] - Sref[(size_t)b * mn + j]) / Sref[(size_t)b * mn]);
        for (int j = 0; j < mn; j++) for (int i = 0; i < m; i++) { Z t = MAGMA_Z_ZERO; for (int l = 0; l < n; l++) t = MAGMA_Z_ADD(t, MAGMA_Z_MUL(A[b * sA + i + (size_t)l * ld], V[b * sV + l + (size_t)j * n]));
          Z us = MAGMA_Z_MUL(U[b * sU + i + (size_t)j * ld], MAGMA_Z_MAKE(s[j], 0)); res[v] = std::max(res[v], MAGMA_Z_ABS(MAGMA_Z_SUB(t, us)) / Sref[(size_t)b * mn]); } }
      (v ? es : ea) = std::max(e, res[v]);
      magma_free(dA); magma_free(dU); magma_free(dV); magma_free(dS); magma_free(dinfo); }
    for (size_t i = 0; i < Sout[0].size(); i++) dd = std::max(dd, fabs(Sout[0][i] - Sout[1][i]));
    printf("r72,gesvj,%lld,%lld,%d,%lld,%lld,%.3g,%.3g,%.3g,%s\n", (long long)m, (long long)n, bc, ia, is, ea, es, dd, ia == 0 && is == 0 && ea < 1e-12 && es < 1e-12 ? "OK" : "WRONG");
  } else {
    magma_int_t n = atoi(argv[2]); char ul = argv[3][0]; bc = atoi(argv[4]); magma_int_t ld = n; size_t sA = (size_t)ld * n + 5, sW = n + 5;
    std::vector<Z> A(sA * bc, MAGMA_Z_ZERO); for (int b = 0; b < bc; b++) for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { Z v = i == j ? MAGMA_Z_MAKE(r1(), 0) : MAGMA_Z_MAKE(r1(), r1()); A[b * sA + i + (size_t)j * ld] = v; A[b * sA + j + (size_t)i * ld] = MAGMA_Z_CONJ(v); }
    std::vector<double> Wref((size_t)n * bc);
    for (int b = 0; b < bc; b++) { std::vector<Z> H(A.begin() + b * sA, A.begin() + b * sA + (size_t)ld * n); magma_int_t lw = -1, lrw = -1, liw = -1, iq, info; Z wq; double rq; char u = ul;
      lapackf77_zheevd("N", &u, &n, H.data(), &ld, &Wref[(size_t)b * n], &wq, &lw, &rq, &lrw, &iq, &liw, &info); lw = (magma_int_t)MAGMA_Z_REAL(wq); lrw = (magma_int_t)rq; liw = iq;
      std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); lapackf77_zheevd("N", &u, &n, H.data(), &ld, &Wref[(size_t)b * n], wk.data(), &lw, rw.data(), &lrw, iw.data(), &liw, &info); }
    std::vector<double> Wout[2];
    for (int v = 0; v < 2; v++) {
      Z* dA; double* dW; magma_int_t* dinfo; magma_zmalloc(&dA, sA * bc); magma_dmalloc(&dW, sW * bc); magma_imalloc(&dinfo, bc); magma_zsetvector((magma_int_t)(sA * bc), A.data(), 1, dA, 1, q);
      magma_uplo_t u = ul == 'U' ? MagmaUpper : MagmaLower;
      if (v == 0) { Z** pA = ptrs(dA, sA, bc); double** pW = ptrs(dW, sW, bc); magma_zheevj_batched_small_sm(MagmaVec, u, n, pA, ld, pW, dinfo, bc, q); }
      else magma_zheevj_batched_strided_small_sm(MagmaVec, u, n, dA, ld, sA, dW, sW, dinfo, bc, q);
      magma_queue_sync(q); std::vector<magma_int_t> hi(bc); magma_igetvector(bc, dinfo, 1, hi.data(), 1, q); for (auto x : hi) if (x) (v ? is : ia) = x;
      std::vector<double> W(sW * bc); magma_dgetvector((magma_int_t)W.size(), dW, 1, W.data(), 1, q); std::vector<Z> V(sA * bc); magma_zgetvector((magma_int_t)V.size(), dA, 1, V.data(), 1, q);
      double e = 0; Wout[v].assign((size_t)n * bc, 0);
      for (int b = 0; b < bc; b++) { std::vector<double> w(W.begin() + b * sW, W.begin() + b * sW + n); std::vector<double> ws = w; std::sort(ws.begin(), ws.end()); double am = 0; for (int j = 0; j < n; j++) am = std::max(am, fabs(Wref[(size_t)b * n + j]));
        for (int j = 0; j < n; j++) { Wout[v][(size_t)b * n + j] = w[j]; e = std::max(e, fabs(ws[j] - Wref[(size_t)b * n + j]) / am); }
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { Z t = MAGMA_Z_ZERO; for (int l = 0; l < n; l++) t = MAGMA_Z_ADD(t, MAGMA_Z_MUL(A[b * sA + i + (size_t)l * ld], V[b * sA + l + (size_t)j * ld]));
          e = std::max(e, MAGMA_Z_ABS(MAGMA_Z_SUB(t, MAGMA_Z_MUL(V[b * sA + i + (size_t)j * ld], MAGMA_Z_MAKE(w[j], 0)))) / am); } }
      (v ? es : ea) = e; magma_free(dA); magma_free(dW); magma_free(dinfo); }
    for (size_t i = 0; i < Wout[0].size(); i++) dd = std::max(dd, fabs(Wout[0][i] - Wout[1][i]));
    printf("r72,heevj,%lld,%c,%d,%lld,%lld,%.3g,%.3g,%.3g,%s\n", (long long)n, ul, bc, ia, is, ea, es, dd, ia == 0 && is == 0 && ea < 1e-11 && es < 1e-11 ? "OK" : "WRONG");
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

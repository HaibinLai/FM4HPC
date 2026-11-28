// Round 57 (roxy): producer -> consumer pairs, with the producer's real output and the consumer's documented contract.
//   round57_trd_batched trd  <hetrd_gpu|hetrd2_gpu> <nh> <other> <uplo L|U> <side L|R> <trans N|C> <pad>
//     magma_zhetrd_gpu / magma_zhetrd2_gpu on an nh x nh Hermitian matrix; then magma_zunmtr_gpu with dA and the host copy
//     wA they return ("as returned by ZHETRD_GPU").  C is nh x other (Left) or other x nh (Right).
//     checks: d, e against LAPACK zhetrd (|e|), wA against dA on the referenced triangle, and C against LAPACK zunmtr
//     applied to wA with the routine's tau.
//   round57_trd_batched bqr <m> <n> <k> <side L|R> <trans N|C> <pad> <batch>
//     magma_zgeqrf_batched on nq x k matrices (nq = m Left, n Right), then magma_zunmqr_batched (workspace by query);
//     C against LAPACK zunmqr on each matrix's downloaded reflectors.
// CSV: r57,kind,op,dims...,info,errs...,verdict
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
static std::vector<Z> rnd(size_t k) { std::vector<Z> v(k); for (auto& x : v) x = MAGMA_Z_MAKE(r1(), r1()); return v; }
static double rel(const Z* X, int ldx, const Z* R, int ldr, int rows, int cols) { double e = 0, a = 0;
  for (int j = 0; j < cols; j++) for (int i = 0; i < rows; i++) { double d = MAGMA_Z_ABS(MAGMA_Z_SUB(X[i + (size_t)j * ldx], R[i + (size_t)j * ldr])); if (!(d <= e)) e = d; a = std::max(a, MAGMA_Z_ABS(R[i + (size_t)j * ldr])); }
  return a > 0 ? e / a : e; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 3) { printf("usage: see source\n"); return 1; }
  std::string kind = argv[1]; magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  if (kind == "trd") {
    std::string op = argv[2]; magma_int_t nh = atoi(argv[3]), ot = atoi(argv[4]); char up = argv[5][0], sd = argv[6][0], tr = argv[7][0]; magma_int_t pad = atoi(argv[8]);
    magma_uplo_t uplo = up == 'U' ? MagmaUpper : MagmaLower; magma_side_t side = sd == 'R' ? MagmaRight : MagmaLeft; magma_trans_t trans = tr == 'C' ? Magma_ConjTrans : MagmaNoTrans;
    magma_int_t ldda = nh + pad, lda = nh + pad, info = 0, rinfo = 0, LD = nh;
    std::vector<Z> A0((size_t)nh * nh); for (magma_int_t j = 0; j < nh; j++) for (magma_int_t i = 0; i <= j; i++) { Z v = i == j ? MAGMA_Z_MAKE(r1(), 0) : MAGMA_Z_MAKE(r1(), r1()); A0[i + j * nh] = v; A0[j + i * nh] = MAGMA_Z_CONJ(v); }
    // LAPACK reference d, e
    std::vector<double> dr(nh), er(std::max<magma_int_t>(nh, 1)), d(nh), e(std::max<magma_int_t>(nh, 1)); std::vector<Z> tr_(nh), tau(nh);
    { std::vector<Z> R = A0; magma_int_t lw = -1; Z wq; lapackf77_zhetrd(&up, &nh, R.data(), &LD, dr.data(), er.data(), tr_.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); std::vector<Z> w(lw); lapackf77_zhetrd(&up, &nh, R.data(), &LD, dr.data(), er.data(), tr_.data(), w.data(), &lw, &rinfo); }
    Z* dA; magma_zmalloc(&dA, (size_t)ldda * nh); magma_zsetmatrix(nh, nh, A0.data(), nh, dA, ldda, q);
    std::vector<Z> wA((size_t)lda * nh, MAGMA_Z_ZERO); magma_int_t nb = magma_get_zhetrd_nb(nh), lw = std::max<magma_int_t>(nh * nb, 1); std::vector<Z> work(lw);
    if (op == "hetrd_gpu") magma_zhetrd_gpu(uplo, nh, dA, ldda, d.data(), e.data(), tau.data(), wA.data(), lda, work.data(), lw, &info);
    else { magma_int_t ldw = ldda * magma_ceildiv(nh, 64) + 2 * ldda * nb; Z* dw; magma_zmalloc(&dw, ldw);
      magma_zhetrd2_gpu(uplo, nh, dA, ldda, d.data(), e.data(), tau.data(), wA.data(), lda, work.data(), lw, dw, ldw, &info); magma_free(dw); }
    double ed = 0, am = 0; for (magma_int_t i = 0; i < nh; i++) { ed = std::max(ed, fabs(d[i] - dr[i])); am = std::max(am, fabs(dr[i])); if (i + 1 < nh) { ed = std::max(ed, fabs(fabs(e[i]) - fabs(er[i]))); am = std::max(am, fabs(er[i])); } }
    ed = am > 0 ? ed / am : ed;
    // wA against dA on the referenced triangle (diagonal included)
    std::vector<Z> hA((size_t)ldda * nh); magma_zgetmatrix(nh, nh, dA, ldda, hA.data(), ldda, q); double ew = 0;
    for (magma_int_t j = 0; j < nh; j++) for (magma_int_t i = 0; i < nh; i++) { if ((up == 'L' && i < j) || (up == 'U' && i > j)) continue;
      ew = std::max(ew, MAGMA_Z_ABS(MAGMA_Z_SUB(wA[i + j * lda], hA[i + j * ldda]))); }
    // unmtr_gpu
    magma_int_t m = sd == 'L' ? nh : ot, n = sd == 'L' ? ot : nh, lddc = m + pad, LDA = lda, LDC = m; std::vector<Z> C = rnd((size_t)m * n), Rc = C;
    { magma_int_t lw2 = -1; Z wq; char t = tr; lapackf77_zunmtr(&sd, &up, &t, &m, &n, wA.data(), &LDA, tau.data(), Rc.data(), &LDC, &wq, &lw2, &rinfo); lw2 = (magma_int_t)MAGMA_Z_REAL(wq); std::vector<Z> w(lw2);
      lapackf77_zunmtr(&sd, &up, &t, &m, &n, wA.data(), &LDA, tau.data(), Rc.data(), &LDC, w.data(), &lw2, &rinfo); }
    Z* dC; magma_zmalloc(&dC, (size_t)lddc * std::max<magma_int_t>(n, 1)); magma_zsetmatrix(m, n, C.data(), m, dC, lddc, q); magma_int_t info2 = 0;
    magma_zunmtr_gpu(side, uplo, trans, m, n, dA, ldda, tau.data(), dC, lddc, wA.data(), lda, &info2);
    std::vector<Z> X((size_t)m * n); magma_zgetmatrix(m, n, dC, lddc, X.data(), m, q); double ec = rel(X.data(), m, Rc.data(), m, m, n);
    bool ok = info == 0 && info2 == 0 && rinfo == 0 && ed < 1e-10 && ew == 0 && ec < 1e-10;
    printf("r57,trd,%s,%lld,%lld,%c,%c,%c,%lld,%lld/%lld,%.3g,%.3g,%.3g,%s\n", op.c_str(), (long long)nh, (long long)ot, up, sd, tr, (long long)pad, (long long)info, (long long)info2, ed, ew, ec, ok ? "OK" : "WRONG");
    magma_free(dA); magma_free(dC);
  } else {
    magma_int_t m = atoi(argv[2]), n = atoi(argv[3]), k = atoi(argv[4]); char sd = argv[5][0], tr = argv[6][0]; magma_int_t pad = atoi(argv[7]), bc = atoi(argv[8]);
    magma_side_t side = sd == 'R' ? MagmaRight : MagmaLeft; magma_trans_t trans = tr == 'C' ? Magma_ConjTrans : MagmaNoTrans;
    magma_int_t nq = sd == 'L' ? m : n, ldda = nq + pad, lddc = m + pad, mn = std::min(nq, k);
    std::vector<Z*> hA(bc), hT(bc), hC(bc); std::vector<std::vector<Z>> A(bc), C(bc);
    for (int b = 0; b < bc; b++) { A[b] = rnd((size_t)nq * k); C[b] = rnd((size_t)m * n); magma_zmalloc(&hA[b], (size_t)ldda * k); magma_zmalloc(&hT[b], std::max<magma_int_t>(mn, 1)); magma_zmalloc(&hC[b], (size_t)lddc * n);
      magma_zsetmatrix(nq, k, A[b].data(), nq, hA[b], ldda, q); magma_zsetmatrix(m, n, C[b].data(), m, hC[b], lddc, q); }
    Z **dA_array, **dT_array, **dC_array; magma_int_t* dinfo; magma_malloc((void**)&dA_array, bc * sizeof(Z*)); magma_malloc((void**)&dT_array, bc * sizeof(Z*)); magma_malloc((void**)&dC_array, bc * sizeof(Z*)); magma_imalloc(&dinfo, bc);
    magma_setvector(bc, sizeof(Z*), hA.data(), 1, dA_array, 1, q); magma_setvector(bc, sizeof(Z*), hT.data(), 1, dT_array, 1, q); magma_setvector(bc, sizeof(Z*), hC.data(), 1, dC_array, 1, q);
    magma_int_t info = magma_zgeqrf_batched(nq, k, dA_array, ldda, dT_array, dinfo, bc, q); magma_queue_sync(q);
    std::vector<magma_int_t> hinfo(bc); magma_igetvector(bc, dinfo, 1, hinfo.data(), 1, q); for (auto x : hinfo) if (x) info = x;
    int64_t lwd = -1; magma_zunmqr_batched(side, trans, m, n, mn, dA_array, ldda, dT_array, dC_array, lddc, NULL, &lwd, dinfo, bc, q);
    void* dw = NULL; if (lwd > 0) magma_malloc(&dw, lwd);
    magma_int_t info2 = magma_zunmqr_batched(side, trans, m, n, mn, dA_array, ldda, dT_array, dC_array, lddc, dw, &lwd, dinfo, bc, q); magma_queue_sync(q);
    magma_igetvector(bc, dinfo, 1, hinfo.data(), 1, q); for (auto x : hinfo) if (x) info2 = x;
    double ec = 0; magma_int_t rinfo = 0;
    for (int b = 0; b < bc; b++) {
      std::vector<Z> V((size_t)ldda * k), tau(std::max<magma_int_t>(mn, 1)), X((size_t)m * n), R = C[b]; magma_zgetmatrix(nq, k, hA[b], ldda, V.data(), ldda, q); magma_zgetvector(mn, hT[b], 1, tau.data(), 1, q);
      magma_zgetmatrix(m, n, hC[b], lddc, X.data(), m, q);
      magma_int_t LDA = ldda, LDC = m, lw = -1; Z wq; lapackf77_zunmqr(&sd, &tr, &m, &n, &mn, V.data(), &LDA, tau.data(), R.data(), &LDC, &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq);
      std::vector<Z> w(lw); lapackf77_zunmqr(&sd, &tr, &m, &n, &mn, V.data(), &LDA, tau.data(), R.data(), &LDC, w.data(), &lw, &rinfo);
      ec = std::max(ec, rel(X.data(), m, R.data(), m, m, n)); magma_free(hA[b]); magma_free(hT[b]); magma_free(hC[b]); }
    bool ok = info == 0 && info2 == 0 && rinfo == 0 && ec < 1e-10;
    printf("r57,bqr,unmqr_batched,%lld,%lld,%lld,%c,%c,%lld,%lld,%lld/%lld,%lld,%.3g,%s\n", (long long)m, (long long)n, (long long)k, sd, tr, (long long)pad, (long long)bc, (long long)info, (long long)info2, (long long)lwd, ec, ok ? "OK" : "WRONG");
    if (dw) magma_free(dw); magma_free(dA_array); magma_free(dT_array); magma_free(dC_array); magma_free(dinfo);
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

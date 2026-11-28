// Round 58 (roxy): CPU-interface producer -> consumer pairs on MAGMA's own producer output.
// Round 36 fed the consumers reflectors made by LAPACK; here the producer is MAGMA, and the consumer (MAGMA) is compared
// with the LAPACK consumer applied to the same producer output, so only the consumer and the hand-over are tested.
//   round58_cpu_pairs <prod> <m> <n> <pad> <cons> <a1> <a2> <a3> <other>   (unused a* given as x or -)
//   prod: brd (magma_zgebrd)  lqf (magma_zgelqf)  lqfgpu (magma_zgelqf_gpu, downloaded)  qlf (magma_zgeqlf)
//         trd (magma_zhetrd, n = m)  qrf (magma_zgeqrf)
//   cons: unmbr <vect Q|P> <side> <trans>   ungbr <vect Q|P>
//         unmlq | unmql | unmqr <side> <trans> -      unmtr <side> <trans> <uplo>      ungqr2
//   C is nq x other (Left) or other x nq (Right).  lda = ldc = rows + pad.
//   prod_err: producer against the LAPACK producer on the same input (QR-like: reflectors and tau; brd/trd: |d|, |e|).
// CSV: r58,prod,cons,m,n,pad,a1,a2,a3,other,info,prod_err,cons_err,verdict
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
template <class F> static magma_int_t qry(F f) { magma_int_t lw = -1; Z wq = MAGMA_Z_ZERO; f(&wq, lw); return std::max<magma_int_t>((magma_int_t)MAGMA_Z_REAL(wq), 1); }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 10) { printf("usage: see source\n"); return 1; }
  std::string prod = argv[1], cons = argv[5]; magma_int_t m = atoi(argv[2]), n = atoi(argv[3]), pad = atoi(argv[4]), ot = atoi(argv[9]);
  char a1 = argv[6][0], a2 = argv[7][0], a3 = argv[8][0];
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  if (prod == "trd") n = m;
  magma_int_t lda = m + pad, mn = std::min(m, n), info = 0, rinfo = 0;
  std::vector<Z> A0((size_t)lda * n, MAGMA_Z_ZERO);
  if (prod == "trd") { for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i <= j; i++) { Z v = i == j ? MAGMA_Z_MAKE(r1(), 0) : MAGMA_Z_MAKE(r1(), r1()); A0[i + j * lda] = v; A0[j + i * lda] = MAGMA_Z_CONJ(v); } }
  else for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i < m; i++) A0[i + j * lda] = MAGMA_Z_MAKE(r1(), r1());
  std::vector<Z> A = A0, R = A0, tau(std::max<magma_int_t>(mn, 1)), taup(std::max<magma_int_t>(mn, 1)), rtau(tau.size()), rtaup(tau.size());
  std::vector<double> d(std::max<magma_int_t>(mn, 1)), e(std::max<magma_int_t>(mn, 1)), dr(d.size()), er(d.size());
  double pe = 0; char up = a3;
  // ---- producer (MAGMA) and the LAPACK producer
  if (prod == "brd") {
    magma_int_t lw = qry([&](Z* w, magma_int_t l) { magma_zgebrd(m, n, A.data(), lda, d.data(), e.data(), tau.data(), taup.data(), w, l, &info); }); std::vector<Z> w(lw);
    magma_zgebrd(m, n, A.data(), lda, d.data(), e.data(), tau.data(), taup.data(), w.data(), lw, &info);
    magma_int_t l2 = qry([&](Z* w2, magma_int_t l) { lapackf77_zgebrd(&m, &n, R.data(), &lda, dr.data(), er.data(), rtau.data(), rtaup.data(), w2, &l, &rinfo); }); std::vector<Z> w2(l2);
    lapackf77_zgebrd(&m, &n, R.data(), &lda, dr.data(), er.data(), rtau.data(), rtaup.data(), w2.data(), &l2, &rinfo);
    double am = 0; for (magma_int_t i = 0; i < mn; i++) { pe = std::max(pe, fabs(fabs(d[i]) - fabs(dr[i]))); am = std::max(am, fabs(dr[i])); if (i + 1 < mn) pe = std::max(pe, fabs(fabs(e[i]) - fabs(er[i]))); } pe /= am;
  } else if (prod == "trd") {
    magma_int_t lw = qry([&](Z* w, magma_int_t l) { magma_zhetrd(up == 'U' ? MagmaUpper : MagmaLower, n, A.data(), lda, d.data(), e.data(), tau.data(), w, l, &info); }); std::vector<Z> w(lw);
    magma_zhetrd(up == 'U' ? MagmaUpper : MagmaLower, n, A.data(), lda, d.data(), e.data(), tau.data(), w.data(), lw, &info);
    magma_int_t l2 = qry([&](Z* w2, magma_int_t l) { lapackf77_zhetrd(&up, &n, R.data(), &lda, dr.data(), er.data(), rtau.data(), w2, &l, &rinfo); }); std::vector<Z> w2(l2);
    lapackf77_zhetrd(&up, &n, R.data(), &lda, dr.data(), er.data(), rtau.data(), w2.data(), &l2, &rinfo);
    double am = 0; for (magma_int_t i = 0; i < n; i++) { pe = std::max(pe, fabs(d[i] - dr[i])); am = std::max(am, fabs(dr[i])); if (i + 1 < n) pe = std::max(pe, fabs(fabs(e[i]) - fabs(er[i]))); } pe /= am;
  } else {
    if (prod == "qrf") { magma_int_t lw = qry([&](Z* w, magma_int_t l) { magma_zgeqrf(m, n, A.data(), lda, tau.data(), w, l, &info); }); std::vector<Z> w(lw); magma_zgeqrf(m, n, A.data(), lda, tau.data(), w.data(), lw, &info);
      magma_int_t l2 = qry([&](Z* w2, magma_int_t l) { lapackf77_zgeqrf(&m, &n, R.data(), &lda, rtau.data(), w2, &l, &rinfo); }); std::vector<Z> w2(l2); lapackf77_zgeqrf(&m, &n, R.data(), &lda, rtau.data(), w2.data(), &l2, &rinfo); }
    if (prod == "lqf") { magma_int_t lw = qry([&](Z* w, magma_int_t l) { magma_zgelqf(m, n, A.data(), lda, tau.data(), w, l, &info); }); std::vector<Z> w(lw); magma_zgelqf(m, n, A.data(), lda, tau.data(), w.data(), lw, &info); }
    if (prod == "lqfgpu") { Z* dA; magma_zmalloc(&dA, (size_t)lda * n); magma_zsetmatrix(m, n, A.data(), lda, dA, lda, q);
      magma_int_t lw = qry([&](Z* w, magma_int_t l) { magma_zgelqf_gpu(m, n, dA, lda, tau.data(), w, l, &info); }); std::vector<Z> w(lw); magma_zgelqf_gpu(m, n, dA, lda, tau.data(), w.data(), lw, &info);
      magma_zgetmatrix(m, n, dA, lda, A.data(), lda, q); magma_free(dA); }
    if (prod == "lqf" || prod == "lqfgpu") { magma_int_t l2 = qry([&](Z* w2, magma_int_t l) { lapackf77_zgelqf(&m, &n, R.data(), &lda, rtau.data(), w2, &l, &rinfo); }); std::vector<Z> w2(l2); lapackf77_zgelqf(&m, &n, R.data(), &lda, rtau.data(), w2.data(), &l2, &rinfo); }
    if (prod == "qlf") { magma_int_t lw = qry([&](Z* w, magma_int_t l) { magma_zgeqlf(m, n, A.data(), lda, tau.data(), w, l, &info); }); lw = std::max<magma_int_t>(lw, 2 * 128 * 128 + n * 128); std::vector<Z> w(lw);
      magma_zgeqlf(m, n, A.data(), lda, tau.data(), w.data(), lw, &info);
      magma_int_t l2 = qry([&](Z* w2, magma_int_t l) { lapackf77_zgeqlf(&m, &n, R.data(), &lda, rtau.data(), w2, &l, &rinfo); }); std::vector<Z> w2(l2); lapackf77_zgeqlf(&m, &n, R.data(), &lda, rtau.data(), w2.data(), &l2, &rinfo); }
    pe = rel(A.data(), lda, R.data(), lda, m, n); for (magma_int_t i = 0; i < mn; i++) pe = std::max(pe, MAGMA_Z_ABS(MAGMA_Z_SUB(tau[i], rtau[i])));
  }
  // ---- consumer: MAGMA against LAPACK on the producer output A, tau (taup)
  double ce = INFINITY; magma_int_t cinfo = 0, crinfo = 0;
  if (cons == "ungbr" || cons == "ungqr2") {
    magma_int_t qm, qn, qk; char vect = a1; Z* t = tau.data();
    if (cons == "ungqr2") { qm = m; qn = mn; qk = mn; }
    else if (vect == 'Q') { qm = m; qk = n; qn = std::min(m, n); }          // Q: m x min(m,n)
    else { qn = n; qk = m; qm = std::min(m, n); t = taup.data(); }        // P**H: min(m,n) x n
    std::vector<Z> X = A, Y = A;
    if (cons == "ungqr2") { magma_zungqr2(qm, qn, qk, X.data(), lda, t, &cinfo); magma_int_t l2 = qry([&](Z* w, magma_int_t l) { lapackf77_zungqr(&qm, &qn, &qk, Y.data(), &lda, t, w, &l, &crinfo); }); std::vector<Z> w(l2); lapackf77_zungqr(&qm, &qn, &qk, Y.data(), &lda, t, w.data(), &l2, &crinfo); }
    else { magma_vect_t mv = vect == 'Q' ? MagmaQ : MagmaP;
      magma_int_t lw = qry([&](Z* w, magma_int_t l) { magma_zungbr(mv, qm, qn, qk, X.data(), lda, t, w, l, &cinfo); }); std::vector<Z> w(lw); magma_zungbr(mv, qm, qn, qk, X.data(), lda, t, w.data(), lw, &cinfo);
      magma_int_t l2 = qry([&](Z* w2, magma_int_t l) { lapackf77_zungbr(&vect, &qm, &qn, &qk, Y.data(), &lda, t, w2, &l, &crinfo); }); std::vector<Z> w2(l2); lapackf77_zungbr(&vect, &qm, &qn, &qk, Y.data(), &lda, t, w2.data(), &l2, &crinfo); }
    ce = rel(X.data(), lda, Y.data(), lda, qm, qn);
  } else {
    char sd, tr, vect = 'Q'; magma_int_t nq, k; Z* Ap = A.data(); Z* t = tau.data();
    if (cons == "unmbr") { vect = a1; sd = a2; tr = a3; if (vect == 'Q') { nq = m; k = n; } else { nq = n; k = m; t = taup.data(); } }
    else { sd = a1; tr = a2;
      if (cons == "unmqr") { nq = m; k = mn; }
      else if (cons == "unmlq") { nq = n; k = mn; }
      else if (cons == "unmql") { nq = m; k = mn; Ap = A.data() + (size_t)(n - mn) * lda; }
      else { nq = n; k = n - 1; } }   // unmtr
    magma_int_t cm = sd == 'L' ? nq : ot, cn = sd == 'L' ? ot : nq, ldc = cm + pad;
    std::vector<Z> C0((size_t)ldc * cn, MAGMA_Z_ZERO); for (magma_int_t j = 0; j < cn; j++) for (magma_int_t i = 0; i < cm; i++) C0[i + j * ldc] = MAGMA_Z_MAKE(r1(), r1());
    std::vector<Z> X = C0, Y = C0; magma_side_t side = sd == 'R' ? MagmaRight : MagmaLeft; magma_trans_t trans = tr == 'C' ? Magma_ConjTrans : MagmaNoTrans;
    std::vector<Z> Ac = A;   // consumers may scribble on A (restored on exit by LAPACK convention); use a copy for MAGMA
    Z* Apc = Ac.data() + (Ap - A.data());
    auto mcall = [&](Z* w, magma_int_t l) {
      if (cons == "unmbr") magma_zunmbr(vect == 'Q' ? MagmaQ : MagmaP, side, trans, cm, cn, k, Apc, lda, t, X.data(), ldc, w, l, &cinfo);
      if (cons == "unmqr") magma_zunmqr(side, trans, cm, cn, k, Apc, lda, t, X.data(), ldc, w, l, &cinfo);
      if (cons == "unmlq") magma_zunmlq(side, trans, cm, cn, k, Apc, lda, t, X.data(), ldc, w, l, &cinfo);
      if (cons == "unmql") magma_zunmql(side, trans, cm, cn, k, Apc, lda, t, X.data(), ldc, w, l, &cinfo);
      if (cons == "unmtr") magma_zunmtr(side, up == 'U' ? MagmaUpper : MagmaLower, trans, cm, cn, Apc, lda, t, X.data(), ldc, w, l, &cinfo); };
    auto lcall = [&](Z* w, magma_int_t l) {
      if (cons == "unmbr") lapackf77_zunmbr(&vect, &sd, &tr, &cm, &cn, &k, Ap, &lda, t, Y.data(), &ldc, w, &l, &crinfo);
      if (cons == "unmqr") lapackf77_zunmqr(&sd, &tr, &cm, &cn, &k, Ap, &lda, t, Y.data(), &ldc, w, &l, &crinfo);
      if (cons == "unmlq") lapackf77_zunmlq(&sd, &tr, &cm, &cn, &k, Ap, &lda, t, Y.data(), &ldc, w, &l, &crinfo);
      if (cons == "unmql") lapackf77_zunmql(&sd, &tr, &cm, &cn, &k, Ap, &lda, t, Y.data(), &ldc, w, &l, &crinfo);
      if (cons == "unmtr") lapackf77_zunmtr(&sd, &up, &tr, &cm, &cn, Ap, &lda, t, Y.data(), &ldc, w, &l, &crinfo); };
    magma_int_t lw = qry(mcall); std::vector<Z> w(lw); mcall(w.data(), lw);
    magma_int_t l2 = qry(lcall); std::vector<Z> w2(l2); lcall(w2.data(), l2);
    ce = rel(X.data(), ldc, Y.data(), ldc, cm, cn);
  }
  bool ok = info == 0 && rinfo == 0 && cinfo == 0 && crinfo == 0 && pe < 1e-10 && ce < 1e-10;
  printf("r58,%s,%s,%lld,%lld,%lld,%c,%c,%c,%lld,%lld/%lld,%.3g,%.3g,%s\n", prod.c_str(), cons.c_str(), (long long)m, (long long)n, (long long)pad, a1, a2, a3, (long long)ot,
         (long long)(info ? info : rinfo), (long long)(cinfo ? cinfo : crinfo), pe, ce, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

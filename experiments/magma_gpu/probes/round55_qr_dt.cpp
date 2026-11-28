// Round 55 (roxy): consumers of the dT produced by magma_zgeqrf_gpu / magma_zgeqrf3_gpu, against LAPACK on the same
// reflectors (MAGMA's own, downloaded), so only the T hand-over and the blocking are tested.
//   round55_qr_dt <op> <m> <n> <k|nrhs> <pad> [side L|R] [trans N|C]
//   ungqr_gpu / ungqr : geqrf_gpu of A (m x n), then Q (m x n) from the first k reflectors; against LAPACK zungqr(m, n, k)
//   unmqr_gpu / unmqr2s_gpu (magma_zunmqr_2stage_gpu, public via magma_v2.h -> magma_zbulge.h): geqrf_gpu of A (nq x k, nq = m for L, n for R), C (m x n); against LAPACK zunmqr
//   geqrs_gpu / geqrs3_gpu : geqrf_gpu / geqrf3_gpu of A (m x n, m >= n), nrhs right-hand sides; X against LAPACK zgels
//   gels_gpu / gels3_gpu   : the drivers; X against LAPACK zgels
// ldda = lddc = lddb = rows + pad.  rel = max |X - X_lapack| / max |X_lapack|.
// unmqr_gpu: dT is allocated at geqrf_gpu's documented size (2 min(nq,k) + roundup(k,32)) nb followed by a sentinel guard;
//   guard = elements of the guard written, guard_end = offset one past the last one written.
// CSV: r55,op,m,n,k,pad,side,trans,nb,info,rel,guard,guard_end,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
typedef magmaDoubleComplex Z;
static size_t g_guard = 0, g_guard_end = 0;   // unmqr_gpu: elements written past the documented end of dT
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static std::vector<Z> rnd(size_t k) { std::vector<Z> v(k); for (auto& x : v) x = MAGMA_Z_MAKE(r1(), r1()); return v; }
static double rel(const std::vector<Z>& X, int ldx, const std::vector<Z>& R, int ldr, int rows, int cols) { double e = 0, a = 0;
  for (int j = 0; j < cols; j++) for (int i = 0; i < rows; i++) { double d = MAGMA_Z_ABS(MAGMA_Z_SUB(X[i + (size_t)j * ldx], R[i + (size_t)j * ldr])); if (!(d <= e)) e = d; a = std::max(a, MAGMA_Z_ABS(R[i + (size_t)j * ldr])); }
  return e / a; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 6) { printf("usage: see source\n"); return 1; }
  std::string op = argv[1]; magma_int_t m = atoi(argv[2]), n = atoi(argv[3]), k = atoi(argv[4]), pad = atoi(argv[5]);
  char sd = argc > 6 ? argv[6][0] : 'L', tr = argc > 7 ? argv[7][0] : 'N';
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  magma_side_t side = sd == 'R' ? MagmaRight : MagmaLeft; magma_trans_t trans = tr == 'C' ? Magma_ConjTrans : MagmaNoTrans;
  magma_int_t info = 0, rinfo = 0, nb = 0; double e = INFINITY;
  auto query = [&](auto f) { magma_int_t lw = -1; Z wq; f(&wq, lw); return std::max<magma_int_t>((magma_int_t)MAGMA_Z_REAL(wq), 1); };
  if (op == "ungqr_gpu" || op == "ungqr") {
    nb = magma_get_zgeqrf_nb(m, n); magma_int_t ldda = m + pad, mn = std::min(m, n);
    std::vector<Z> A = rnd((size_t)m * n), tau(mn); Z *dA, *dT; magma_zmalloc(&dA, (size_t)ldda * n); magma_zmalloc(&dT, (size_t)(2 * mn + magma_roundup(n, 32)) * nb);
    magma_zsetmatrix(m, n, A.data(), m, dA, ldda, q); magma_zgeqrf_gpu(m, n, dA, ldda, tau.data(), dT, &info);
    std::vector<Z> V((size_t)ldda * n); magma_zgetmatrix(m, n, dA, ldda, V.data(), ldda, q);      // MAGMA's reflectors
    std::vector<Z> R = V; magma_int_t LD = ldda;
    magma_int_t lw = query([&](Z* w, magma_int_t l) { lapackf77_zungqr(&m, &n, &k, R.data(), &LD, tau.data(), w, &l, &rinfo); });
    { std::vector<Z> w(lw); lapackf77_zungqr(&m, &n, &k, R.data(), &LD, tau.data(), w.data(), &lw, &rinfo); }
    std::vector<Z> Q((size_t)ldda * n);
    if (op == "ungqr_gpu") { magma_zungqr_gpu(m, n, k, dA, ldda, tau.data(), dT, nb, &info); magma_zgetmatrix(m, n, dA, ldda, Q.data(), ldda, q); }
    else { Q = V; magma_zungqr(m, n, k, Q.data(), ldda, tau.data(), dT, nb, &info); }
    e = rel(Q, ldda, R, ldda, m, n); magma_free(dA); magma_free(dT);
  }
  else if (op == "unmqr_gpu" || op == "unmqr2s_gpu") {
    magma_int_t nq = side == MagmaLeft ? m : n; nb = magma_get_zgeqrf_nb(nq, k); magma_int_t ldda = nq + pad, lddc = m + pad, mn = std::min(nq, k);
    std::vector<Z> A = rnd((size_t)nq * k), tau(mn), C = rnd((size_t)m * n); Z *dA, *dT, *dC;
    size_t tdoc = (size_t)(2 * mn + magma_roundup(k, 32)) * nb, tg = (size_t)(std::max(m, n) + 64) * nb * 2;   // documented size of geqrf_gpu's dT, guard after it
    magma_zmalloc(&dA, (size_t)ldda * k); magma_zmalloc(&dT, tdoc + tg); magma_zmalloc(&dC, (size_t)lddc * n);
    { std::vector<Z> g(tg, MAGMA_Z_MAKE(777.0, -777.0)); magma_zsetvector((magma_int_t)tg, g.data(), 1, dT + tdoc, 1, q); }
    magma_zsetmatrix(nq, k, A.data(), nq, dA, ldda, q); magma_zgeqrf_gpu(nq, k, dA, ldda, tau.data(), dT, &info);
    std::vector<Z> V((size_t)ldda * k); magma_zgetmatrix(nq, k, dA, ldda, V.data(), ldda, q);
    std::vector<Z> R = C; magma_int_t LDA = ldda, LDC = m; char t = tr == 'C' ? 'C' : 'N';
    magma_int_t lw = query([&](Z* w, magma_int_t l) { lapackf77_zunmqr(&sd, &t, &m, &n, &mn, V.data(), &LDA, tau.data(), R.data(), &LDC, w, &l, &rinfo); });
    { std::vector<Z> w(lw); lapackf77_zunmqr(&sd, &t, &m, &n, &mn, V.data(), &LDA, tau.data(), R.data(), &LDC, w.data(), &lw, &rinfo); }
    magma_zsetmatrix(m, n, C.data(), m, dC, lddc, q);
    if (op == "unmqr_gpu") {
      magma_int_t hl = query([&](Z* w, magma_int_t l) { magma_zunmqr_gpu(side, trans, m, n, mn, dA, ldda, tau.data(), dC, lddc, w, l, dT, nb, &info); }); std::vector<Z> hw(hl);
      magma_zunmqr_gpu(side, trans, m, n, mn, dA, ldda, tau.data(), dC, lddc, hw.data(), hl, dT, nb, &info);
    } else {
      // R55_EXPLICIT_V: store the unit lower trapezoid explicitly (zeros above, ones on the diagonal), the form
      // he2hb leaves for its caller zheevdx_2stage; the documentation asks only for zgeqrf's output.
      if (getenv("R55_EXPLICIT_V")) magmablas_zlaset(MagmaUpper, nq, mn, MAGMA_Z_ZERO, MAGMA_Z_ONE, dA, ldda, q);
      // R55_FULLT: overwrite the T of every block (dT + i*nb, ld nb) with LAPACK zlarft of the same reflectors, i.e. the
      // form he2hb provides; zgeqrf_gpu leaves the last block's T (factored on the CPU) unwritten.
      if (getenv("R55_FULLT")) for (magma_int_t i = 0; i < mn; i += nb) { magma_int_t ib = std::min(nb, mn - i), rows = nq - i; std::vector<Z> Tb((size_t)nb * ib, MAGMA_Z_ZERO);
        lapackf77_zlarft("F", "C", &rows, &ib, &V[i + (size_t)i * ldda], &LDA, &tau[i], Tb.data(), &nb); magma_zsetmatrix(ib, ib, Tb.data(), nb, dT + (size_t)i * nb, nb, q); }
      magma_zunmqr_2stage_gpu(side, trans, m, n, mn, dA, ldda, dC, lddc, dT, nb, &info);
    }
    std::vector<Z> X((size_t)m * n); magma_zgetmatrix(m, n, dC, lddc, X.data(), m, q);
    e = rel(X, m, R, m, m, n);
    { std::vector<Z> g(tg); magma_zgetvector((magma_int_t)tg, dT + tdoc, 1, g.data(), 1, q); size_t w = 0, last = 0;
      for (size_t i = 0; i < tg; i++) if (MAGMA_Z_REAL(g[i]) != 777.0 || MAGMA_Z_IMAG(g[i]) != -777.0) { w++; last = i + 1; }
      g_guard = w; g_guard_end = last; }
    magma_free(dA); magma_free(dT); magma_free(dC);
  }
  else {   // geqrs_gpu geqrs3_gpu gels_gpu gels3_gpu: k = nrhs
    magma_int_t nrhs = k; nb = magma_get_zgeqrf_nb(m, n); magma_int_t ldda = m + pad, lddb = m + pad, mn = std::min(m, n);
    std::vector<Z> A = rnd((size_t)m * n), B = rnd((size_t)m * nrhs), tau(mn); Z *dA, *dT, *dB;
    magma_zmalloc(&dA, (size_t)ldda * n); magma_zmalloc(&dT, (size_t)(2 * mn + magma_roundup(n, 32)) * nb); magma_zmalloc(&dB, (size_t)lddb * nrhs);
    magma_zsetmatrix(m, n, A.data(), m, dA, ldda, q); magma_zsetmatrix(m, nrhs, B.data(), m, dB, lddb, q);
    std::vector<Z> RA = A, RB = B; magma_int_t LDA = m, LDB = m;
    magma_int_t lw = query([&](Z* w, magma_int_t l) { lapackf77_zgels("N", &m, &n, &nrhs, RA.data(), &LDA, RB.data(), &LDB, w, &l, &rinfo); });
    { std::vector<Z> w(lw); lapackf77_zgels("N", &m, &n, &nrhs, RA.data(), &LDA, RB.data(), &LDB, w.data(), &lw, &rinfo); }
    magma_int_t hl = std::max<magma_int_t>(1, (m - n + nb) * (nrhs + nb) + (m + n + 64) * (nrhs + 2 * nb)); std::vector<Z> hw(hl);
    if (op == "geqrs_gpu")  { magma_zgeqrf_gpu (m, n, dA, ldda, tau.data(), dT, &info); if (!info) magma_zgeqrs_gpu (m, n, nrhs, dA, ldda, tau.data(), dT, dB, lddb, hw.data(), hl, &info); }
    if (op == "geqrs3_gpu") { magma_zgeqrf3_gpu(m, n, dA, ldda, tau.data(), dT, &info); if (!info) magma_zgeqrs3_gpu(m, n, nrhs, dA, ldda, tau.data(), dT, dB, lddb, hw.data(), hl, &info); }
    if (op == "gels_gpu")   magma_zgels_gpu (MagmaNoTrans, m, n, nrhs, dA, ldda, dB, lddb, hw.data(), hl, &info);
    if (op == "gels3_gpu")  magma_zgels3_gpu(MagmaNoTrans, m, n, nrhs, dA, ldda, dB, lddb, hw.data(), hl, &info);
    std::vector<Z> X((size_t)n * nrhs); magma_zgetmatrix(n, nrhs, dB, lddb, X.data(), n, q);
    e = rel(X, n, RB, m, n, nrhs); magma_free(dA); magma_free(dT); magma_free(dB);
  }
  bool ok = info == 0 && rinfo == 0 && e < 1e-10;
  ok = ok && g_guard == 0;
  printf("r55,%s,%lld,%lld,%lld,%lld,%c,%c,%lld,%lld,%.3g,%zu,%zu,%s\n", op.c_str(), (long long)m, (long long)n, (long long)k, (long long)pad, sd, tr, (long long)nb, (long long)(info ? info : rinfo), e, g_guard, g_guard_end, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

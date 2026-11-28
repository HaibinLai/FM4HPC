// Round 59 (roxy): device workspaces allocated at exactly their documented size, followed by a sentinel guard.
// After the call: guard = number of guard elements written (must be 0), plus a cheap correctness check where one exists.
//   round59_guard <op> <m> <n> <p> [side L|R]
//   geqrf_gpu m n            dT  (2 min(m,n) + roundup(n,32)) nb
//   geqrf3_gpu m n           dT  same
//   geqrs_gpu m n nrhs       dT  2 min(m,n) nb + roundup(n,32) max(nb,nrhs)   (geqrs_gpu's own doc); X against LAPACK zgels
//   geqrs3_gpu m n nrhs      dT  same, with geqrf3_gpu
//   he2hb n nb               dT  n*nb (lower)
//   hetrd2 n                 dwork  ldda*ceil(n/64) + 2*ldda*nb (lower); d against LAPACK zhetrd
//   getri n                  dwork  n*nb, nb = magma_get_zgetri_nb(n); |A*inv - I|
//   stedx n                  dwork  3n^2/2 + 3n doubles (range all); eigenvalues against LAPACK dsterf
//   geqp3 m n                dwork  (n+1)*nb, nb = magma_get_zgeqp3_nb(m,n)
//   gegqr m n ikind          dwork  n^2 (1, 4), 3n^2 + min(m,n) + 2 (2), none (3); host work as documented; |Q^H Q - I|
//   trsm m n                 dX  lddx*n,  d_dinvA  ceil(dim/128)*128*128 (Left lower notrans nonunit, flag 1); against blas ztrsm
//   hemv n                   dwork  ldda*ceil(n/64) (lower); against blas zhemv
//   larfb m n k side         dwork  ldwork*k, ldwork = n (Left) or m (Right); against LAPACK zlarfb
// CSV: r59,op,m,n,p,side,info,guard,err,verdict
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
static magma_queue_t q;
static const size_t G = 1 << 16;   // guard elements
struct Guard {   // a device buffer of `doc` elements of T followed by G sentinel elements
  void* p = nullptr; size_t doc = 0, esz = 0;
  template <class T> T* make(size_t d) { doc = d; esz = sizeof(T); magma_malloc(&p, (d + G) * esz); std::vector<unsigned char> s(G * esz, 0xA5);
    magma_setvector((magma_int_t)(G * esz), 1, s.data(), 1, (char*)p + d * esz, 1, q); return (T*)p; }
  size_t written() { std::vector<unsigned char> s(G * esz); magma_getvector((magma_int_t)(G * esz), 1, (char*)p + doc * esz, 1, s.data(), 1, q);
    size_t w = 0; for (size_t i = 0; i < G; i++) for (size_t b = 0; b < esz; b++) if (s[i * esz + b] != 0xA5) { w++; break; } return w; }
  ~Guard() { if (p) magma_free(p); } };
static double relm(const std::vector<Z>& X, int ldx, const std::vector<Z>& R, int ldr, int rows, int cols) { double e = 0, a = 0;
  for (int j = 0; j < cols; j++) for (int i = 0; i < rows; i++) { double d = MAGMA_Z_ABS(MAGMA_Z_SUB(X[i + (size_t)j * ldx], R[i + (size_t)j * ldr])); if (!(d <= e)) e = d; a = std::max(a, MAGMA_Z_ABS(R[i + (size_t)j * ldr])); }
  return a > 0 ? e / a : e; }
static std::vector<Z> herm(int n) { std::vector<Z> A((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { Z v = i == j ? MAGMA_Z_MAKE(r1(), 0) : MAGMA_Z_MAKE(r1(), r1()); A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = MAGMA_Z_CONJ(v); } return A; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 5) { printf("usage: see source\n"); return 1; }
  std::string op = argv[1]; magma_int_t m = atoi(argv[2]), n = atoi(argv[3]), p = atoi(argv[4]); char sd = argc > 5 ? argv[5][0] : 'L';
  magma_init(); magma_queue_create(0, &q);
  magma_int_t info = 0, rinfo = 0; size_t guard = 0; double err = -1;
  magma_int_t mn = std::min(m, n), ldda = magma_roundup(std::max<magma_int_t>(m, 1), 32);
  if (op == "geqrf_gpu" || op == "geqrf3_gpu" || op == "geqrs_gpu" || op == "geqrs3_gpu") {
    magma_int_t nb = magma_get_zgeqrf_nb(m, n), nrhs = std::max<magma_int_t>(p, 1); bool solve = op.find("geqrs") == 0, three = op.find('3') != std::string::npos;
    size_t tdoc = solve ? (size_t)2 * mn * nb + (size_t)magma_roundup(n, 32) * std::max(nb, nrhs) : (size_t)(2 * mn + magma_roundup(n, 32)) * nb;
    std::vector<Z> A = rnd((size_t)m * n), B = rnd((size_t)m * nrhs), tau(std::max<magma_int_t>(mn, 1)); Z *dA, *dB; magma_zmalloc(&dA, (size_t)ldda * n); magma_zmalloc(&dB, (size_t)ldda * nrhs);
    Guard g; Z* dT = g.make<Z>(tdoc); magma_zsetmatrix(m, n, A.data(), m, dA, ldda, q); magma_zsetmatrix(m, nrhs, B.data(), m, dB, ldda, q);
    if (three) magma_zgeqrf3_gpu(m, n, dA, ldda, tau.data(), dT, &info); else magma_zgeqrf_gpu(m, n, dA, ldda, tau.data(), dT, &info);
    if (solve && !info) { magma_int_t lw = -1; Z wq; if (three) magma_zgeqrs3_gpu(m, n, nrhs, dA, ldda, tau.data(), dT, dB, ldda, &wq, lw, &info); else magma_zgeqrs_gpu(m, n, nrhs, dA, ldda, tau.data(), dT, dB, ldda, &wq, lw, &info);
      lw = std::max<magma_int_t>((magma_int_t)MAGMA_Z_REAL(wq), (m - n + nb) * (nrhs + nb) + nrhs * nb); std::vector<Z> hw(lw);
      if (three) magma_zgeqrs3_gpu(m, n, nrhs, dA, ldda, tau.data(), dT, dB, ldda, hw.data(), lw, &info); else magma_zgeqrs_gpu(m, n, nrhs, dA, ldda, tau.data(), dT, dB, ldda, hw.data(), lw, &info);
      std::vector<Z> X((size_t)n * nrhs); magma_zgetmatrix(n, nrhs, dB, ldda, X.data(), n, q);
      std::vector<Z> RA = A, RB = B; magma_int_t LD = m, lw2 = -1; lapackf77_zgels("N", &m, &n, &nrhs, RA.data(), &LD, RB.data(), &LD, &wq, &lw2, &rinfo); lw2 = (magma_int_t)MAGMA_Z_REAL(wq); std::vector<Z> w2(lw2);
      lapackf77_zgels("N", &m, &n, &nrhs, RA.data(), &LD, RB.data(), &LD, w2.data(), &lw2, &rinfo); err = relm(X, n, RB, m, n, nrhs); }
    guard = g.written(); magma_free(dA); magma_free(dB);
  } else if (op == "he2hb") {
    magma_int_t nb = std::max<magma_int_t>(p, 1); std::vector<Z> A = herm(n), tau(std::max<magma_int_t>(n, 1)); Guard g; Z* dT = g.make<Z>((size_t)n * nb);
    magma_int_t lw = -1; Z wq; magma_zhetrd_he2hb(MagmaLower, n, nb, A.data(), n, tau.data(), &wq, lw, dT, &info); lw = std::max<magma_int_t>((magma_int_t)MAGMA_Z_REAL(wq), 1); std::vector<Z> w(lw);
    magma_zhetrd_he2hb(MagmaLower, n, nb, A.data(), n, tau.data(), w.data(), lw, dT, &info); guard = g.written();
  } else if (op == "hetrd2") {
    magma_int_t nb = magma_get_zhetrd_nb(n), ld = magma_roundup(n, 32); std::vector<Z> A0 = herm(n), wA((size_t)n * n), tau(n); std::vector<double> d(n), e(n), dr(n), er(n);
    Z* dA; magma_zmalloc(&dA, (size_t)ld * n); magma_zsetmatrix(n, n, A0.data(), n, dA, ld, q); size_t ldw = (size_t)ld * magma_ceildiv(n, 64) + 2 * (size_t)ld * nb;
    Guard g; Z* dw = g.make<Z>(ldw); magma_int_t lw = n * nb; std::vector<Z> w(lw);
    magma_zhetrd2_gpu(MagmaLower, n, dA, ld, d.data(), e.data(), tau.data(), wA.data(), n, w.data(), lw, dw, (magma_int_t)ldw, &info); guard = g.written();
    std::vector<Z> R = A0, tr(n); magma_int_t LD = n, l2 = -1; Z wq; lapackf77_zhetrd("L", &n, R.data(), &LD, dr.data(), er.data(), tr.data(), &wq, &l2, &rinfo); l2 = (magma_int_t)MAGMA_Z_REAL(wq); std::vector<Z> w2(l2);
    lapackf77_zhetrd("L", &n, R.data(), &LD, dr.data(), er.data(), tr.data(), w2.data(), &l2, &rinfo); err = 0; double am = 0; for (int i = 0; i < n; i++) { err = std::max(err, fabs(d[i] - dr[i])); am = std::max(am, fabs(dr[i])); } err /= am; magma_free(dA);
  } else if (op == "getri") {
    magma_int_t nb = magma_get_zgetri_nb(n), ld = magma_roundup(n, 32); std::vector<Z> A = rnd((size_t)n * n); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = MAGMA_Z_ADD(A[i + (size_t)i * n], MAGMA_Z_MAKE(n, 0));
    Z* dA; magma_zmalloc(&dA, (size_t)ld * n); magma_zsetmatrix(n, n, A.data(), n, dA, ld, q); std::vector<magma_int_t> ipiv(n); magma_zgetrf_gpu(n, n, dA, ld, ipiv.data(), &info);
    Guard g; Z* dw = g.make<Z>((size_t)n * nb); magma_zgetri_gpu(n, dA, ld, ipiv.data(), dw, n * nb, &info); guard = g.written();
    std::vector<Z> Inv((size_t)n * n); magma_zgetmatrix(n, n, dA, ld, Inv.data(), n, q); err = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { Z s = MAGMA_Z_ZERO; for (int k = 0; k < n; k++) s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(A[i + (size_t)k * n], Inv[k + (size_t)j * n])); if (i == j) s = MAGMA_Z_SUB(s, MAGMA_Z_ONE); err = std::max(err, MAGMA_Z_ABS(s)); }
    magma_free(dA);
  } else if (op == "stedx") {
    std::vector<double> d(n), e(std::max<magma_int_t>(n, 1)), dr, er; for (auto& x : d) x = r1(); for (auto& x : e) x = r1(); dr = d; er = e;
    std::vector<Z> Zm((size_t)n * n); magma_int_t lrw = 1 + 4 * n + 2 * n * n, liw = 3 + 5 * n; std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw);
    Guard g; double* dw = g.make<double>((size_t)3 * n * n / 2 + 3 * n);
    magma_zstedx(MagmaRangeAll, n, 0, 0, 0, 0, d.data(), e.data(), Zm.data(), n, rw.data(), lrw, iw.data(), liw, dw, &info); guard = g.written();
    lapackf77_dsterf(&n, dr.data(), er.data(), &rinfo); err = 0; double am = 0; for (int i = 0; i < n; i++) { err = std::max(err, fabs(d[i] - dr[i])); am = std::max(am, fabs(dr[i])); } err /= am;
  } else if (op == "geqp3") {
    magma_int_t nb = magma_get_zgeqp3_nb(m, n); std::vector<Z> A = rnd((size_t)m * n), tau(std::max<magma_int_t>(mn, 1)); std::vector<magma_int_t> jpvt(n, 0); std::vector<double> rw(2 * n);
    Z* dA; magma_zmalloc(&dA, (size_t)ldda * n); magma_zsetmatrix(m, n, A.data(), m, dA, ldda, q); Guard g; Z* dw = g.make<Z>((size_t)(n + 1) * nb);
    magma_zgeqp3_gpu(m, n, dA, ldda, jpvt.data(), tau.data(), dw, (n + 1) * nb, rw.data(), &info); guard = g.written(); magma_free(dA);
  } else if (op == "gegqr") {
    magma_int_t ik = p; std::vector<Z> A = rnd((size_t)m * n); Z* dA; magma_zmalloc(&dA, (size_t)ldda * n); magma_zsetmatrix(m, n, A.data(), m, dA, ldda, q);
    size_t dws = ik == 2 ? (size_t)3 * n * n + mn + 2 : ik == 3 ? 0 : (size_t)n * n; std::vector<Z> hw(ik == 1 ? (size_t)5 * n * n + 7 * n + 64 : (size_t)3 * n * n);
    Guard g; Z* dw = g.make<Z>(dws); magma_zgegqr_gpu(ik, m, n, dA, ldda, dw, hw.data(), &info); guard = g.written();
    std::vector<Z> Q((size_t)m * n); magma_zgetmatrix(m, n, dA, ldda, Q.data(), m, q); err = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { Z s = MAGMA_Z_ZERO; for (int k = 0; k < m; k++) s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(MAGMA_Z_CONJ(Q[k + (size_t)i * m]), Q[k + (size_t)j * m])); if (i == j) s = MAGMA_Z_SUB(s, MAGMA_Z_ONE); err = std::max(err, MAGMA_Z_ABS(s)); }
    magma_free(dA);
  } else if (op == "trsm") {
    magma_side_t side = sd == 'R' ? MagmaRight : MagmaLeft; magma_int_t na = sd == 'R' ? n : m, lda = magma_roundup(na, 32);
    std::vector<Z> A((size_t)na * na, MAGMA_Z_ZERO), B = rnd((size_t)m * n); for (int j = 0; j < na; j++) for (int i = j; i < na; i++) A[i + (size_t)j * na] = i == j ? MAGMA_Z_MAKE(2 + r1(), r1()) : MAGMA_Z_MAKE(0.1 * r1(), 0.1 * r1());
    Z *dA, *dB; magma_zmalloc(&dA, (size_t)lda * na); magma_zmalloc(&dB, (size_t)ldda * n); magma_zsetmatrix(na, na, A.data(), na, dA, lda, q); magma_zsetmatrix(m, n, B.data(), m, dB, ldda, q);
    Guard gx, gi; Z* dX = gx.make<Z>((size_t)ldda * n); size_t il = (size_t)magma_ceildiv(na, 128) * 128 * 128; Z* dI = gi.make<Z>(il);
    Z one = MAGMA_Z_ONE; magmablas_ztrsm_work(side, MagmaLower, MagmaNoTrans, MagmaNonUnit, m, n, one, dA, lda, dB, ldda, dX, ldda, 1, dI, (magma_int_t)il, q); magma_queue_sync(q);
    guard = gx.written() + gi.written(); std::vector<Z> X((size_t)m * n), R = B; magma_zgetmatrix(m, n, dB, ldda, X.data(), m, q);
    magma_int_t M = m, N = n, LA = na; char s = sd; blasf77_ztrsm(&s, "L", "N", "N", &M, &N, &one, A.data(), &LA, R.data(), &M); err = relm(X, m, R, m, m, n); magma_free(dA); magma_free(dB);
  } else if (op == "hemv") {
    magma_int_t ld = magma_roundup(n, 32); std::vector<Z> A = herm(n), x = rnd(n), y = rnd(n); Z *dA, *dx, *dy; magma_zmalloc(&dA, (size_t)ld * n); magma_zmalloc(&dx, n); magma_zmalloc(&dy, n);
    magma_zsetmatrix(n, n, A.data(), n, dA, ld, q); magma_zsetvector(n, x.data(), 1, dx, 1, q); magma_zsetvector(n, y.data(), 1, dy, 1, q);
    size_t lw = (size_t)ld * magma_ceildiv(n, 64); Guard g; Z* dw = g.make<Z>(lw); Z al = MAGMA_Z_MAKE(0.7, 0.2), be = MAGMA_Z_MAKE(0.3, -0.1);
    magmablas_zhemv_work(MagmaLower, n, al, dA, ld, dx, 1, be, dy, 1, dw, (magma_int_t)lw, q); magma_queue_sync(q); guard = g.written();
    std::vector<Z> Y(n), R = y; magma_zgetvector(n, dy, 1, Y.data(), 1, q); magma_int_t N = n, one = 1; blasf77_zhemv("L", &N, &al, A.data(), &N, x.data(), &one, &be, R.data(), &one); err = relm(Y, n, R, n, n, 1);
    magma_free(dA); magma_free(dx); magma_free(dy);
  } else if (op == "larfb") {
    magma_int_t k = p; magma_side_t side = sd == 'R' ? MagmaRight : MagmaLeft; magma_int_t nv = sd == 'R' ? n : m, ldwork = sd == 'R' ? m : n;
    std::vector<Z> V = rnd((size_t)nv * k), tau(k), C = rnd((size_t)m * n); magma_int_t NV = nv, K = k; { magma_int_t lw = -1; Z wq; lapackf77_zgeqrf(&NV, &K, V.data(), &NV, tau.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); std::vector<Z> w(lw); lapackf77_zgeqrf(&NV, &K, V.data(), &NV, tau.data(), w.data(), &lw, &rinfo); }
    std::vector<Z> T((size_t)k * k, MAGMA_Z_ZERO); lapackf77_zlarft("F", "C", &NV, &K, V.data(), &NV, tau.data(), T.data(), &K);
    std::vector<Z> Ve = V; for (int j = 0; j < k; j++) for (int i = 0; i <= j && i < nv; i++) Ve[i + (size_t)j * nv] = i == j ? MAGMA_Z_ONE : MAGMA_Z_ZERO;   // explicit V for magma_zlarfb_gpu
    Z *dV, *dT, *dC; magma_zmalloc(&dV, (size_t)nv * k); magma_zmalloc(&dT, (size_t)k * k); magma_zmalloc(&dC, (size_t)m * n);
    magma_zsetmatrix(nv, k, Ve.data(), nv, dV, nv, q); magma_zsetmatrix(k, k, T.data(), k, dT, k, q); magma_zsetmatrix(m, n, C.data(), m, dC, m, q);
    Guard g; Z* dw = g.make<Z>((size_t)ldwork * k); magma_zlarfb_gpu(side, MagmaNoTrans, MagmaForward, MagmaColumnwise, m, n, k, dV, nv, dT, k, dC, m, dw, ldwork, q); magma_queue_sync(q); guard = g.written();
    std::vector<Z> X((size_t)m * n), R = C; magma_zgetmatrix(m, n, dC, m, X.data(), m, q); magma_int_t M = m, N = n; std::vector<Z> w((size_t)std::max(m, n) * k); magma_int_t lw = std::max(m, n);
    char s = sd; lapackf77_zlarfb(&s, "N", "F", "C", &M, &N, &K, V.data(), &NV, T.data(), &K, R.data(), &M, w.data(), &lw); err = relm(X, m, R, m, m, n);
    magma_free(dV); magma_free(dT); magma_free(dC);
  }
  bool ok = info == 0 && rinfo == 0 && guard == 0 && (err < 0 || err < 1e-10);
  printf("r59,%s,%lld,%lld,%lld,%c,%lld,%zu,%.3g,%s\n", op.c_str(), (long long)m, (long long)n, (long long)p, sd, (long long)(info ? info : rinfo), guard, err, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

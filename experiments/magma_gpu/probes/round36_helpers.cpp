// Round 36 (roxy): direct differential test of the class-(c) helper routines (INVENTORY.md) against LAPACK on identical
// inputs, d precision (z/c/s are generated from the same sources).  The Householder data come from host LAPACK
// factorizations, so MAGMA and LAPACK apply / generate exactly the same Q.
//   orm     dormqr, dormlq, dormql, dormrq (host), dormqr2_gpu, dormql2_gpu, dormtr, dormtr_gpu, dormbr (Q and P)
//   org     dorgqr2, dorgbr (Q and P)
//   larfb   dlarfb_gpu  (direct F/B x storev C/R x side x trans)
//   trevc   dtrevc3     (side R/L/B x howmany A/B/S, quasi-triangular T with 2x2 blocks)
// CSV: helpers,routine,case,info,relerr,verdict   relerr = max|magma - lapack| / max(1, max|lapack|); BAD if > 1e-10 or info != lapack info
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
static magma_queue_t Q;
static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
typedef std::vector<double> V;
static V rnd(size_t n) { V v(n); for (auto& x : v) x = r1(); return v; }
static double relerr(const V& a, const V& b, int rows, int cols, int ld) { double e = 0, m = 1; for (int j = 0; j < cols; j++) for (int i = 0; i < rows; i++) { double x = a[i + (size_t)j * ld], y = b[i + (size_t)j * ld];
    m = std::max(m, fabs(y)); double d = fabs(x - y); if (!(d <= e)) e = d; } return e / m; }
static void out(const char* r, const std::string& cs, long long info, long long linfo, double e) {
  printf("helpers,%s,%s,%lld,%.3g,%s\n", r, cs.c_str(), info, e, (info != linfo || !(e <= 1e-10)) ? "BAD" : "OK"); }
static char cs_[160];
#define CASE(...) (snprintf(cs_, sizeof cs_, __VA_ARGS__), std::string(cs_))

// factor a (rows x cols) matrix with LAPACK; kind: 0 QR, 1 LQ, 2 QL, 3 RQ.  Returns A (with V) and tau.
static void factor(int kind, int rows, int cols, V& A, V& tau, int lda) {
  magma_int_t m = rows, n = cols, ld = lda, info, lw = -1; double q; tau.assign(std::max(1, std::min(rows, cols)), 0);
  auto call = [&](double* w, magma_int_t l) { if (kind == 0) lapackf77_dgeqrf(&m, &n, A.data(), &ld, tau.data(), w, &l, &info); else if (kind == 1) lapackf77_dgelqf(&m, &n, A.data(), &ld, tau.data(), w, &l, &info);
    else if (kind == 2) lapackf77_dgeqlf(&m, &n, A.data(), &ld, tau.data(), w, &l, &info); else lapackf77_dgerqf(&m, &n, A.data(), &ld, tau.data(), w, &l, &info); };
  call(&q, lw); lw = (magma_int_t)q + 1; V w(lw); call(w.data(), lw); }

static void orm() {
  struct S { int m, n, k; }; std::vector<S> ss = {{1, 1, 1}, {5, 3, 2}, {64, 40, 30}, {200, 130, 100}, {129, 65, 0}, {300, 257, 257}, {600, 70, 520}};
  const char* fam[] = {"dormqr", "dormlq", "dormql", "dormrq"};
  for (auto s : ss) for (char sd : {'L', 'R'}) for (char tr : {'N', 'T'}) for (int kind = 0; kind < 4; kind++) {
    int nq = sd == 'L' ? s.m : s.n, k = std::min(s.k, nq); bool rowV = kind == 1 || kind == 3; int ar = rowV ? std::max(1, k) : nq, ac = rowV ? nq : std::max(1, k), lda = std::max(1, ar);
    Sd = 11u + s.m + 7 * kind; V A = rnd((size_t)lda * ac), tau; if (k > 0) factor(kind, rowV ? k : nq, rowV ? nq : k, A, tau, lda); else tau.assign(1, 0);
    V C = rnd((size_t)s.m * s.n), Cl = C, Cm = C; magma_int_t m = s.m, n = s.n, kk = k, ld = lda, ldc = std::max(1, s.m), info = -999, linfo = -999; char sdc[2] = {sd, 0}, trc[2] = {tr, 0};
    { magma_int_t lw = -1; double q; auto lap = [&](double* w, magma_int_t l) { if (kind == 0) lapackf77_dormqr(sdc, trc, &m, &n, &kk, A.data(), &ld, tau.data(), Cl.data(), &ldc, w, &l, &linfo);
        else if (kind == 1) lapackf77_dormlq(sdc, trc, &m, &n, &kk, A.data(), &ld, tau.data(), Cl.data(), &ldc, w, &l, &linfo); else if (kind == 2) lapackf77_dormql(sdc, trc, &m, &n, &kk, A.data(), &ld, tau.data(), Cl.data(), &ldc, w, &l, &linfo);
        else lapackf77_dormrq(sdc, trc, &m, &n, &kk, A.data(), &ld, tau.data(), Cl.data(), &ldc, w, &l, &linfo); }; lap(&q, lw); lw = (magma_int_t)q + 1; V w(lw); lap(w.data(), lw); }
    { magma_side_t S_ = sd == 'L' ? MagmaLeft : MagmaRight; magma_trans_t T_ = tr == 'N' ? MagmaNoTrans : MagmaTrans; V Ac = A; double q; magma_int_t lw = -1;
      auto mg = [&](double* w, magma_int_t l) { if (kind == 0) magma_dormqr(S_, T_, m, n, kk, Ac.data(), ld, tau.data(), Cm.data(), ldc, w, l, &info); else if (kind == 1) magma_dormlq(S_, T_, m, n, kk, Ac.data(), ld, tau.data(), Cm.data(), ldc, w, l, &info);
        else if (kind == 2) magma_dormql(S_, T_, m, n, kk, Ac.data(), ld, tau.data(), Cm.data(), ldc, w, l, &info); else magma_dormrq(S_, T_, m, n, kk, Ac.data(), ld, tau.data(), Cm.data(), ldc, w, l, &info); };
      mg(&q, lw); lw = std::max((magma_int_t)q, (magma_int_t)1) + 1; V w(lw); mg(w.data(), lw);
      out(fam[kind], CASE("m=%d n=%d k=%d %c%c", s.m, s.n, k, sd, tr), info, linfo, relerr(Cm, Cl, s.m, s.n, ldc));
      if (kind == 0 || kind == 2) {   // the _gpu variants: dA (V on the device), wA (host copy of V)
        Cm = C; magmaDouble_ptr dA, dC; magma_dmalloc(&dA, (size_t)lda * ac + 1); magma_dmalloc(&dC, (size_t)ldc * s.n + 1); magma_dsetmatrix(lda, ac, A.data(), lda, dA, lda, Q); magma_dsetmatrix(s.m, s.n, C.data(), ldc, dC, ldc, Q);
        if (kind == 0) magma_dormqr2_gpu(S_, T_, m, n, kk, dA, ld, tau.data(), dC, ldc, A.data(), ld, &info); else magma_dormql2_gpu(S_, T_, m, n, kk, dA, ld, tau.data(), dC, ldc, A.data(), ld, &info);
        magma_dgetmatrix(s.m, s.n, dC, ldc, Cm.data(), ldc, Q); out(kind == 0 ? "dormqr2_gpu" : "dormql2_gpu", CASE("m=%d n=%d k=%d %c%c", s.m, s.n, k, sd, tr), info, linfo, relerr(Cm, Cl, s.m, s.n, ldc));
        magma_free(dA); magma_free(dC); } }
  }
}

static void ormtrbr() {
  for (int nq : {1, 5, 64, 130, 257}) for (char up : {'L', 'U'}) for (char sd : {'L', 'R'}) for (char tr : {'N', 'T'}) {   // dormtr / dormtr_gpu after LAPACK dsytrd
    int m = sd == 'L' ? nq : 37, n = sd == 'L' ? 29 : nq; Sd = 23u + nq; V A = rnd((size_t)nq * nq); for (int j = 0; j < nq; j++) for (int i = 0; i < nq; i++) A[i + (size_t)j * nq] = A[j + (size_t)i * nq];
    magma_int_t N = nq, info, lw = -1, linfo, minfo; double q; V d(nq), e(nq), tau(std::max(1, nq)); char upc[2] = {up, 0}, sdc[2] = {sd, 0}, trc[2] = {tr, 0};
    lapackf77_dsytrd(upc, &N, A.data(), &N, d.data(), e.data(), tau.data(), &q, &lw, &info); lw = (magma_int_t)q + 1; { V w(lw); lapackf77_dsytrd(upc, &N, A.data(), &N, d.data(), e.data(), tau.data(), w.data(), &lw, &info); }
    V C = rnd((size_t)m * n), Cl = C, Cm = C; magma_int_t M = m, Nn = n, ldc = m; lw = -1;
    lapackf77_dormtr(sdc, upc, trc, &M, &Nn, A.data(), &N, tau.data(), Cl.data(), &ldc, &q, &lw, &linfo); lw = (magma_int_t)q + 1; { V w(lw); lapackf77_dormtr(sdc, upc, trc, &M, &Nn, A.data(), &N, tau.data(), Cl.data(), &ldc, w.data(), &lw, &linfo); }
    magma_side_t S_ = sd == 'L' ? MagmaLeft : MagmaRight; magma_uplo_t U_ = up == 'L' ? MagmaLower : MagmaUpper; magma_trans_t T_ = tr == 'N' ? MagmaNoTrans : MagmaTrans;
    { V Ac = A; lw = -1; magma_dormtr(S_, U_, T_, M, Nn, Ac.data(), N, tau.data(), Cm.data(), ldc, &q, lw, &minfo); lw = std::max((magma_int_t)q, (magma_int_t)1) + 1; V w(lw);
      magma_dormtr(S_, U_, T_, M, Nn, Ac.data(), N, tau.data(), Cm.data(), ldc, w.data(), lw, &minfo); out("dormtr", CASE("nq=%d %c%c%c", nq, up, sd, tr), minfo, linfo, relerr(Cm, Cl, m, n, ldc)); }
    { Cm = C; magmaDouble_ptr dA, dC; magma_dmalloc(&dA, (size_t)nq * nq); magma_dmalloc(&dC, (size_t)m * n); magma_dsetmatrix(nq, nq, A.data(), nq, dA, nq, Q); magma_dsetmatrix(m, n, C.data(), m, dC, m, Q);
      magma_dormtr_gpu(S_, U_, T_, M, Nn, dA, N, tau.data(), dC, ldc, A.data(), N, &minfo); magma_dgetmatrix(m, n, dC, m, Cm.data(), m, Q);
      out("dormtr_gpu", CASE("nq=%d %c%c%c", nq, up, sd, tr), minfo, linfo, relerr(Cm, Cl, m, n, ldc)); magma_free(dA); magma_free(dC); }
  }
  // dormbr after LAPACK dgebrd of an (am x an) matrix; vect Q uses tauq, P uses taup
  struct S { int am, an; }; for (auto s : std::vector<S>{{5, 3}, {3, 5}, {64, 40}, {40, 64}, {200, 130}, {130, 200}, {257, 257}}) for (char vect : {'Q', 'P'}) for (char sd : {'L', 'R'}) for (char tr : {'N', 'T'}) {
    Sd = 31u + s.am; V A = rnd((size_t)s.am * s.an); int mn = std::min(s.am, s.an); V d(mn), e(std::max(1, mn)), tq(mn), tp(mn); magma_int_t am = s.am, an = s.an, info, lw = -1, linfo, minfo; double q;
    lapackf77_dgebrd(&am, &an, A.data(), &am, d.data(), e.data(), tq.data(), tp.data(), &q, &lw, &info); lw = (magma_int_t)q + 1; { V w(lw); lapackf77_dgebrd(&am, &an, A.data(), &am, d.data(), e.data(), tq.data(), tp.data(), w.data(), &lw, &info); }
    // vect = Q: Q is am x am, k = an;  vect = P: P is an x an, k = am
    int nq = vect == 'Q' ? s.am : s.an, k = vect == 'Q' ? s.an : s.am; int m = sd == 'L' ? nq : 23, n = sd == 'L' ? 19 : nq;
    V C = rnd((size_t)m * n), Cl = C, Cm = C; magma_int_t M = m, N = n, K = k, lda = s.am, ldc = m; char vc[2] = {vect, 0}, sdc[2] = {sd, 0}, trc[2] = {tr, 0}; double* tau = vect == 'Q' ? tq.data() : tp.data(); lw = -1;
    lapackf77_dormbr(vc, sdc, trc, &M, &N, &K, A.data(), &lda, tau, Cl.data(), &ldc, &q, &lw, &linfo); lw = (magma_int_t)q + 1; { V w(lw); lapackf77_dormbr(vc, sdc, trc, &M, &N, &K, A.data(), &lda, tau, Cl.data(), &ldc, w.data(), &lw, &linfo); }
    V Ac = A; lw = -1; magma_vect_t vv = vect == 'Q' ? MagmaQ : MagmaP; magma_side_t S_ = sd == 'L' ? MagmaLeft : MagmaRight; magma_trans_t T_ = tr == 'N' ? MagmaNoTrans : MagmaTrans;
    magma_dormbr(vv, S_, T_, M, N, K, Ac.data(), lda, tau, Cm.data(), ldc, &q, lw, &minfo); lw = std::max((magma_int_t)q, (magma_int_t)1) + 1; { V w(lw); magma_dormbr(vv, S_, T_, M, N, K, Ac.data(), lda, tau, Cm.data(), ldc, w.data(), lw, &minfo); }
    out("dormbr", CASE("A=%dx%d %c%c%c", s.am, s.an, vect, sd, tr), minfo, linfo, relerr(Cm, Cl, m, n, ldc));
  }
}

static void org() {
  struct S { int m, n, k; }; for (auto s : std::vector<S>{{1, 1, 1}, {5, 3, 2}, {64, 64, 64}, {200, 130, 100}, {300, 257, 257}, {129, 65, 0}, {600, 600, 300}}) {   // dorgqr2
    Sd = 41u + s.m; V A = rnd((size_t)s.m * s.n), tau; if (s.k > 0) { V F = rnd((size_t)s.m * s.k); factor(0, s.m, s.k, F, tau, s.m); for (size_t i = 0; i < F.size(); i++) A[i] = F[i]; } else tau.assign(1, 0);
    V Al = A, Am = A; magma_int_t m = s.m, n = s.n, k = s.k, linfo, minfo, lw = -1; double q;
    lapackf77_dorgqr(&m, &n, &k, Al.data(), &m, tau.data(), &q, &lw, &linfo); lw = (magma_int_t)q + 1; { V w(lw); lapackf77_dorgqr(&m, &n, &k, Al.data(), &m, tau.data(), w.data(), &lw, &linfo); }
    magma_dorgqr2(m, n, k, Am.data(), m, tau.data(), &minfo); out("dorgqr2", CASE("m=%d n=%d k=%d", s.m, s.n, s.k), minfo, linfo, relerr(Am, Al, s.m, s.n, s.m));
  }
  struct B { int am, an; }; for (auto s : std::vector<B>{{5, 3}, {3, 5}, {64, 40}, {40, 64}, {200, 130}, {130, 200}, {257, 257}}) for (char vect : {'Q', 'P'}) {   // dorgbr after dgebrd
    Sd = 47u + s.am; V A = rnd((size_t)s.am * s.an); int mn = std::min(s.am, s.an); V d(mn), e(std::max(1, mn)), tq(mn), tp(mn); magma_int_t am = s.am, an = s.an, info, lw = -1, linfo, minfo; double q;
    lapackf77_dgebrd(&am, &an, A.data(), &am, d.data(), e.data(), tq.data(), tp.data(), &q, &lw, &info); lw = (magma_int_t)q + 1; { V w(lw); lapackf77_dgebrd(&am, &an, A.data(), &am, d.data(), e.data(), tq.data(), tp.data(), w.data(), &lw, &info); }
    // generate the full square Q (am x am, k = an) or P^T (an x an, k = am) as in LAPACK's examples: m = n = order, k as above
    int ord = vect == 'Q' ? s.am : s.an, k = vect == 'Q' ? s.an : s.am; int ld = std::max(s.am, ord); V G((size_t)ld * ord, 0);
    for (int j = 0; j < std::min(ord, s.an); j++) for (int i = 0; i < std::min(ord, s.am); i++) G[i + (size_t)j * ld] = A[i + (size_t)j * s.am];
    V Gl = G, Gm = G; magma_int_t M = ord, N = ord, K = k, LD = ld; char vc[2] = {vect, 0}; double* tau = vect == 'Q' ? tq.data() : tp.data(); lw = -1;
    lapackf77_dorgbr(vc, &M, &N, &K, Gl.data(), &LD, tau, &q, &lw, &linfo); lw = (magma_int_t)q + 1; { V w(lw); lapackf77_dorgbr(vc, &M, &N, &K, Gl.data(), &LD, tau, w.data(), &lw, &linfo); }
    lw = -1; magma_dorgbr(vect == 'Q' ? MagmaQ : MagmaP, M, N, K, Gm.data(), LD, tau, &q, lw, &minfo); lw = std::max((magma_int_t)q, (magma_int_t)1) + 1; { V w(lw); magma_dorgbr(vect == 'Q' ? MagmaQ : MagmaP, M, N, K, Gm.data(), LD, tau, w.data(), lw, &minfo); }
    out("dorgbr", CASE("A=%dx%d %c", s.am, s.an, vect), minfo, linfo, relerr(Gm, Gl, ord, ord, ld));
  }
}

static void larfb() {
  struct S { int m, n, k; }; for (auto s : std::vector<S>{{5, 3, 2}, {64, 40, 32}, {200, 130, 64}, {129, 65, 1}, {300, 257, 128}}) for (char dir : {'F', 'B'}) for (char sv : {'C', 'R'}) for (char sd : {'L', 'R'}) for (char tr : {'N', 'T'}) {
    int nq = sd == 'L' ? s.m : s.n, k = std::min(s.k, nq); Sd = 53u + s.m; int vr = sv == 'C' ? nq : k, vc = sv == 'C' ? k : nq; V Vm = rnd((size_t)vr * vc), T((size_t)k * k, 0), tau(k);
    // unit-diagonal Householder structure as produced by the matching factorization (QR / LQ / QL / RQ)
    int kind = dir == 'F' ? (sv == 'C' ? 0 : 1) : (sv == 'C' ? 2 : 3); factor(kind, vr, vc, Vm, tau, vr);
    magma_int_t N = nq, K = k, ldv = vr, ldt = k; char dc[2] = {dir, 0}, svc[2] = {sv, 0}; lapackf77_dlarft(dc, svc, &N, &K, Vm.data(), &ldv, tau.data(), T.data(), &ldt);
    V C = rnd((size_t)s.m * s.n), Cl = C, Cm = C; magma_int_t M = s.m, Nn = s.n, ldc = s.m, ldw = std::max(s.m, s.n); char sdc[2] = {sd, 0}, trc[2] = {tr, 0}; V w((size_t)ldw * k + 1);
    lapackf77_dlarfb(sdc, trc, dc, svc, &M, &Nn, &K, Vm.data(), &ldv, T.data(), &ldt, Cl.data(), &ldc, w.data(), &ldw);
    V Vx = Vm;   // MAGMA larfb_gpu documents explicit storage of the unit triangle (LAPACK's dlarfb ignores it)
    for (int j = 0; j < vc; j++) for (int i = 0; i < vr; i++) { int r = sv == 'C' ? i : j, c = sv == 'C' ? j : i;   // r: index along nq, c: reflector index
      int dpos = dir == 'F' ? c : nq - k + c; double& x = Vx[i + (size_t)j * vr]; if (r == dpos) x = 1.0; else if (dir == 'F' ? r < dpos : r > dpos) x = 0.0; }
    magmaDouble_ptr dV, dT, dC, dW; magma_dmalloc(&dV, Vm.size()); magma_dmalloc(&dT, T.size()); magma_dmalloc(&dC, C.size()); magma_dmalloc(&dW, (size_t)ldw * k + 1);
    magma_dsetvector((magma_int_t)Vx.size(), Vx.data(), 1, dV, 1, Q); magma_dsetvector((magma_int_t)T.size(), T.data(), 1, dT, 1, Q); magma_dsetvector((magma_int_t)C.size(), C.data(), 1, dC, 1, Q);
    magma_int_t info = magma_dlarfb_gpu(sd == 'L' ? MagmaLeft : MagmaRight, tr == 'N' ? MagmaNoTrans : MagmaTrans, dir == 'F' ? MagmaForward : MagmaBackward, sv == 'C' ? MagmaColumnwise : MagmaRowwise,
                                        M, Nn, K, dV, ldv, dT, ldt, dC, ldc, dW, ldw, Q);
    magma_queue_sync(Q); magma_dgetvector((magma_int_t)C.size(), dC, 1, Cm.data(), 1, Q);
    out("dlarfb_gpu", CASE("m=%d n=%d k=%d %c%c%c%c", s.m, s.n, k, dir, sv, sd, tr), info, 0, relerr(Cm, Cl, s.m, s.n, s.m));
    magma_free(dV); magma_free(dT); magma_free(dC); magma_free(dW);
  }
}

static void trevc() {
  for (int n : {1, 2, 5, 33, 64, 130}) for (char side : {'R', 'L', 'B'}) for (char how : {'A', 'B', 'S'}) {
    Sd = 61u + n; V T((size_t)n * n, 0); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) T[i + (size_t)j * n] = i == j ? 1.0 + j + 0.3 * r1() : r1();
    for (int j = 0; j + 1 < n; j += 3) { double a = T[j + (size_t)j * n]; T[j + 1 + (size_t)(j + 1) * n] = a; T[j + (size_t)(j + 1) * n] = 0.7; T[j + 1 + (size_t)j * n] = -0.4; if (j + 2 < n) T[j + 2 + (size_t)(j + 1) * n] = 0; }   // 2x2 blocks, standard form
    std::vector<magma_int_t> sel(n, 0); for (int j = 0; j < n; j++) sel[j] = (j % 2 == 0); for (int j = 0; j + 1 < n; j += 3) sel[j + 1] = sel[j];   // select both halves of each complex pair
    V Qm((size_t)n * n); { V X = rnd((size_t)n * n), tau; factor(0, n, n, X, tau, n); magma_int_t N = n, info, lw = -1; double q; lapackf77_dorgqr(&N, &N, &N, X.data(), &N, tau.data(), &q, &lw, &info); lw = (magma_int_t)q + 1; V w(lw); lapackf77_dorgqr(&N, &N, &N, X.data(), &N, tau.data(), w.data(), &lw, &info); Qm = X; }
    V VLl = how == 'B' ? Qm : V((size_t)n * n, 0), VRl = VLl, VLm = VLl, VRm = VLl; magma_int_t N = n, mm = n, ml = 0, mo = 0, linfo, minfo, lw = -1; double q; char sc[2] = {side, 0}, hc[2] = {how, 0};
    std::vector<magma_int_t> sl = sel, sm = sel;
    lapackf77_dtrevc3(sc, hc, sl.data(), &N, T.data(), &N, VLl.data(), &N, VRl.data(), &N, &mm, &ml, &q, &lw, &linfo); lw = (magma_int_t)q + 1; { V w(lw); lapackf77_dtrevc3(sc, hc, sl.data(), &N, T.data(), &N, VLl.data(), &N, VRl.data(), &N, &mm, &ml, w.data(), &lw, &linfo); }
    magma_side_t S_ = side == 'R' ? MagmaRight : side == 'L' ? MagmaLeft : MagmaBothSides; magma_vec_t H_ = how == 'A' ? MagmaAllVec : how == 'B' ? MagmaBacktransVec : MagmaSomeVec;
    lw = -1; magma_dtrevc3(S_, H_, sm.data(), N, T.data(), N, VLm.data(), N, VRm.data(), N, mm, &mo, &q, lw, &minfo); lw = std::max((magma_int_t)q, (magma_int_t)(3 * n + 2 * n * 128)) + 1;
    if (getenv("R36_NB")) { int nbv = atoi(getenv("R36_NB")); lw = nbv == 0 ? std::max(1, 3 * n) : n + 2 * n * nbv; }   // exact workspace: 0 = unblocked version 1
    { V w(lw); if (getenv("R36_MT")) magma_dtrevc3_mt(S_, H_, sm.data(), N, T.data(), N, VLm.data(), N, VRm.data(), N, mm, &mo, w.data(), lw, &minfo); else magma_dtrevc3(S_, H_, sm.data(), N, T.data(), N, VLm.data(), N, VRm.data(), N, mm, &mo, w.data(), lw, &minfo); }
    auto resid = [&](const V& X, bool left) {   // max over eigenpairs of ||T v - lambda v|| / (||T|| ||v|| n eps); columns follow select in S mode
      double tn = 0; for (double t : T) tn = std::max(tn, fabs(t)); double w = 0; int col = 0;
      for (int j = 0; j < n; j++) { bool pair = j + 1 < n && T[j + 1 + (size_t)j * n] != 0; if (how == 'S' && !sel[j]) { if (pair) j++; continue; }
        if (!pair) { double l = T[j + (size_t)j * n], r = 0, vn = 0; for (int i = 0; i < n; i++) { double sm = 0; for (int q2 = 0; q2 < n; q2++) sm += (left ? T[q2 + (size_t)i * n] : T[i + (size_t)q2 * n]) * X[q2 + (size_t)col * n];
              r = std::max(r, fabs(sm - l * X[i + (size_t)col * n])); vn = std::max(vn, fabs(X[i + (size_t)col * n])); } w = std::max(w, vn > 0 ? r / (tn * vn * n * 1.1e-16) : 1e300); col++; }
        else { double a2 = T[j + (size_t)j * n], b = sqrt(fabs(T[j + (size_t)(j + 1) * n])) * sqrt(fabs(T[j + 1 + (size_t)j * n])), r = 0, vn = 0;
          for (int i = 0; i < n; i++) { double tx = 0, ty = 0; for (int q2 = 0; q2 < n; q2++) { double t = left ? T[q2 + (size_t)i * n] : T[i + (size_t)q2 * n]; tx += t * X[q2 + (size_t)col * n]; ty += t * X[q2 + (size_t)(col + 1) * n]; }
            double x = X[i + (size_t)col * n], y = X[i + (size_t)(col + 1) * n];
            double ex = left ? a2 * x + b * y : a2 * x - b * y, ey = left ? -b * x + a2 * y : b * x + a2 * y; r = std::max(r, std::max(fabs(tx - ex), fabs(ty - ey))); vn = std::max(vn, fabs(x) + fabs(y)); }
          w = std::max(w, vn > 0 ? r / (tn * vn * n * 1.1e-16) : 1e300); col += 2; j++; } }
      return w; };
    if (how != 'B') { double rm = 0, rl = 0; if (side != 'L') { rm = std::max(rm, resid(VRm, false)); rl = std::max(rl, resid(VRl, false)); } if (side != 'R') { rm = std::max(rm, resid(VLm, true)); rl = std::max(rl, resid(VLl, true)); }
      printf("helpers,dtrevc3_resid,n=%d %c%c,%lld,%.3g,%s,lapack_resid=%.3g\n", n, side, how, (long long)minfo, rm, rm <= 100 ? "OK" : "BAD", rl); }
    int cols = how == 'S' ? (int)ml : n; double e = 0; if (side != 'L') e = std::max(e, relerr(VRm, VRl, n, cols, n)); if (side != 'R') e = std::max(e, relerr(VLm, VLl, n, cols, n));
    out("dtrevc3", CASE("n=%d %c%c m=%lld/%lld", n, side, how, (long long)mo, (long long)ml), minfo, linfo, mo == ml ? e : 1e300);
  }
}

// gehrd / gehrd2 (panel helpers lahr2, lahru): A0 = Q H Q^T with Q from LAPACK dorghr applied to MAGMA's reflectors
static void gehrd() {
  for (int v = 0; v < 2; v++) for (int n : {1, 5, 33, 64, 130, 257, 513}) for (int part = 0; part < 2; part++) {
    int ilo = part && n > 4 ? 3 : 1, ihi = part && n > 4 ? n - 2 : n; Sd = 71u + n; V A0 = rnd((size_t)n * n);
    for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) if (j < ilo - 1 || i > ihi - 1) A0[i + (size_t)j * n] = 0;   // balanced form outside ilo..ihi
    V A = A0, tau(std::max(1, n)); magma_int_t N = n, info = -999, nb = magma_get_dgehrd_nb(n), lw = n * nb; V w(lw + 1);
    if (v == 0) { magmaDouble_ptr dT; magma_dmalloc(&dT, (size_t)nb * n + 1); magma_dgehrd(N, ilo, ihi, A.data(), N, tau.data(), w.data(), lw, dT, &info); magma_free(dT); }
    else magma_dgehrd2(N, ilo, ihi, A.data(), N, tau.data(), w.data(), lw, &info);
    V H((size_t)n * n, 0); for (int j = 0; j < n; j++) for (int i = 0; i <= std::min(n - 1, j + 1); i++) H[i + (size_t)j * n] = A[i + (size_t)j * n];
    V Qm = A; magma_int_t lo = ilo, hi = ihi, li, l2 = -1; double q; lapackf77_dorghr(&N, &lo, &hi, Qm.data(), &N, tau.data(), &q, &l2, &li); l2 = (magma_int_t)q + 1; { V w2(l2); lapackf77_dorghr(&N, &lo, &hi, Qm.data(), &N, tau.data(), w2.data(), &l2, &li); }
    double r = 0, o = 0, an = 0; for (double x : A0) an = std::max(an, fabs(x));
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { double sm = 0, oo = 0; for (int k = 0; k < n; k++) { double qh = 0; for (int l = 0; l < n; l++) qh += Qm[i + (size_t)l * n] * H[l + (size_t)k * n]; sm += qh * Qm[j + (size_t)k * n]; oo += Qm[k + (size_t)i * n] * Qm[k + (size_t)j * n]; }
        r = std::max(r, fabs(sm - A0[i + (size_t)j * n])); o = std::max(o, fabs(oo - (i == j ? 1.0 : 0.0))); }
    double e = std::max(r / (an * n * 1.1e-16 * 10), o / (n * 1.1e-16 * 10));
    printf("helpers,%s,n=%d ilo=%d ihi=%d,%lld,%.3g,%s\n", v ? "dgehrd2" : "dgehrd", n, ilo, ihi, (long long)info, e, (info == 0 && e <= 100) ? "OK" : "BAD");
  }
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string g = argc > 1 ? argv[1] : "all";
  magma_init(); magma_queue_create(0, &Q);
  printf("helpers,routine,case,info,relerr,verdict\n");
  if (g == "all" || g == "orm") orm();
  if (g == "all" || g == "ormtrbr") ormtrbr();
  if (g == "all" || g == "org") org();
  if (g == "all" || g == "larfb") larfb();
  if (g == "all" || g == "trevc") trevc();
  if (g == "all" || g == "gehrd") gehrd();
  fprintf(stderr, "DONE %s cuda=%s\n", g.c_str(), cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

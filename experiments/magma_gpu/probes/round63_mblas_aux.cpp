// Round 63 (roxy): magmablas auxiliary kernels never covered before, against CPU references built from their own docs.
//   round63_mblas_aux <op> <m> <n> <k> <o1> <o2> <pad>
//   geadd2 m n                    B := alpha A + beta B
//   geam m n tA tB                C := alpha op(A) + beta op(B)       (o1, o2 in N|T|C)
//   gemm_reduce m n k             doc: C := alpha A^T B + beta C (A k x m, B k x n); err_T and err_H reported
//   gemv_conj m n incx            y := alpha A conj(x) + beta y
//   laset_band m n k uplo         diagonal := diag, k-1 off-diagonals := offdiag (others untouched)
//   laswpx m n                    row interchanges k1..k2 = 1..min(m,n) with a random ipiv, column-major (ldx = 1, ldy = ld)
//   laswp2 m n                    the same on A stored row-wise (dAT)
//   swapdblk n nb                 swap the n/nb diagonal nb x nb blocks of A and B (inca = incb = 1)
//   symmetrize n uplo             copy the uplo triangle to the other (Hermitian: conj), diagonal real
//   trmv n uplo trans diag incx   x := op(A) x
//   transpose_conj m n            AT := A^H (out of place)
//   transpose_inplace n           A := A^T;   transpose_conj_inplace n: A := A^H
//   symv n uplo                   complex symmetric y := alpha A x + beta y (magmablas_zsymv_work, lwork ldda*ceil(n/64))
//   larfg n                       Householder: [beta; 0] = H [alpha; x], against LAPACK zlarfg
//   lacpy_conj n                  A2[i*lda2] := conj(A1[i*lda1])
// CSV: r63,op,m,n,k,o1,o2,pad,err,err2,verdict
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
static Z rz() { return MAGMA_Z_MAKE(r1(), r1()); }
static Z mul(Z a, Z b) { return MAGMA_Z_MUL(a, b); }
static Z add(Z a, Z b) { return MAGMA_Z_ADD(a, b); }
static Z cj(Z a) { return MAGMA_Z_CONJ(a); }
static double dif(const std::vector<Z>& X, const std::vector<Z>& R) { double e = 0, a = 0; for (size_t i = 0; i < X.size(); i++) { double d = MAGMA_Z_ABS(MAGMA_Z_SUB(X[i], R[i])); if (!(d <= e)) e = d; a = std::max(a, MAGMA_Z_ABS(R[i])); } return a > 0 ? e / a : e; }
static magma_queue_t q;
static Z* up(const std::vector<Z>& h) { Z* d; magma_zmalloc(&d, std::max<size_t>(h.size(), 1)); magma_zsetvector((magma_int_t)h.size(), h.data(), 1, d, 1, q); return d; }
static std::vector<Z> dn(Z* d, size_t n) { std::vector<Z> h(n); magma_zgetvector((magma_int_t)n, d, 1, h.data(), 1, q); magma_free(d); return h; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 8) { printf("usage: see source\n"); return 1; }
  std::string op = argv[1]; int m = atoi(argv[2]), n = atoi(argv[3]), k = atoi(argv[4]); char o1 = argv[5][0], o2 = argv[6][0]; int pad = atoi(argv[7]);
  magma_init(); magma_queue_create(0, &q);
  Z al = MAGMA_Z_MAKE(0.7, -0.3), be = MAGMA_Z_MAKE(-0.4, 0.2); double err = -1, err2 = -1;
  auto at = [](std::vector<Z>& M, int ld, int i, int j) -> Z& { return M[i + (size_t)j * ld]; };
  if (op == "geadd2") { int ld = m + pad; std::vector<Z> A((size_t)ld * n), B((size_t)ld * n); for (auto& x : A) x = rz(); for (auto& x : B) x = rz(); std::vector<Z> R = B;
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) at(R, ld, i, j) = add(mul(al, at(A, ld, i, j)), mul(be, at(B, ld, i, j)));
    Z* dA = up(A); Z* dB = up(B); magmablas_zgeadd2(m, n, al, dA, ld, be, dB, ld, q); magma_free(dA); err = dif(dn(dB, B.size()), R); }
  else if (op == "geam") { int ra = o1 == 'N' ? m : n, ca = o1 == 'N' ? n : m, rb = o2 == 'N' ? m : n, cb = o2 == 'N' ? n : m, la = ra + pad, lb = rb + pad, lc = m + pad;
    std::vector<Z> A((size_t)la * ca), B((size_t)lb * cb), C((size_t)lc * n, MAGMA_Z_ZERO), R; for (auto& x : A) x = rz(); for (auto& x : B) x = rz(); R = C;
    auto op_ = [&](std::vector<Z>& M, int ld, char t, int i, int j) { Z v = t == 'N' ? at(M, ld, i, j) : at(M, ld, j, i); return t == 'C' ? cj(v) : v; };
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) at(R, lc, i, j) = add(mul(al, op_(A, la, o1, i, j)), mul(be, op_(B, lb, o2, i, j)));
    magma_trans_t ta = o1 == 'N' ? MagmaNoTrans : o1 == 'T' ? MagmaTrans : Magma_ConjTrans, tb = o2 == 'N' ? MagmaNoTrans : o2 == 'T' ? MagmaTrans : Magma_ConjTrans;
    Z* dA = up(A); Z* dB = up(B); Z* dC = up(C); magmablas_zgeam(ta, tb, m, n, al, dA, la, be, dB, lb, dC, lc, q); magma_free(dA); magma_free(dB);
    std::vector<Z> X = dn(dC, C.size()); for (int j = 0; j < n; j++) for (int i = m; i < lc; i++) at(X, lc, i, j) = MAGMA_Z_ZERO; err = dif(X, R); }
  else if (op == "gemm_reduce") { int la = k + pad, lc = m + pad; std::vector<Z> A((size_t)la * m), B((size_t)la * n), C((size_t)lc * n); for (auto& x : A) x = rz(); for (auto& x : B) x = rz(); for (auto& x : C) x = rz();
    std::vector<Z> RT = C, RH = C; for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { Z st = MAGMA_Z_ZERO, sh = MAGMA_Z_ZERO; for (int l = 0; l < k; l++) { st = add(st, mul(at(A, la, l, i), at(B, la, l, j))); sh = add(sh, mul(cj(at(A, la, l, i)), at(B, la, l, j))); }
      at(RT, lc, i, j) = add(mul(al, st), mul(be, at(C, lc, i, j))); at(RH, lc, i, j) = add(mul(al, sh), mul(be, at(C, lc, i, j))); }
    Z* dA = up(A); Z* dB = up(B); Z* dC = up(C); magmablas_zgemm_reduce(m, n, k, al, dA, la, dB, la, be, dC, lc, q); magma_free(dA); magma_free(dB); std::vector<Z> X = dn(dC, C.size()); err = dif(X, RT); err2 = dif(X, RH); }
  else if (op == "gemv_conj") { int ld = m + pad, ix = std::max(k, 1); std::vector<Z> A((size_t)ld * n), x((size_t)n * ix), y(m); for (auto& v : A) v = rz(); for (auto& v : x) v = rz(); for (auto& v : y) v = rz();
    std::vector<Z> R = y; for (int i = 0; i < m; i++) { Z s = MAGMA_Z_ZERO; for (int j = 0; j < n; j++) s = add(s, mul(at(A, ld, i, j), cj(x[(size_t)j * ix]))); R[i] = add(mul(al, s), mul(be, y[i])); }
    Z* dA = up(A); Z* dx = up(x); Z* dy = up(y); magmablas_zgemv_conj(m, n, al, dA, ld, dx, ix, be, dy, 1, q); magma_free(dA); magma_free(dx); err = dif(dn(dy, m), R); }
  else if (op == "laset_band") { int ld = m + pad; std::vector<Z> A((size_t)ld * n); for (auto& v : A) v = rz(); std::vector<Z> R = A; Z od = MAGMA_Z_MAKE(3, 1), dg = MAGMA_Z_MAKE(5, -2);
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { int d = o1 == 'U' ? j - i : i - j; if (d == 0) at(R, ld, i, j) = dg; else if (d > 0 && d < k) at(R, ld, i, j) = od; }
    Z* dA = up(A); magmablas_zlaset_band(o1 == 'U' ? MagmaUpper : MagmaLower, m, n, k, od, dg, dA, ld, q); err = dif(dn(dA, A.size()), R); }
  else if (op == "laswpx" || op == "laswp2") { int ld = m + pad, mn = std::min(m, n); std::vector<Z> A((size_t)ld * n); for (auto& v : A) v = rz(); std::vector<magma_int_t> ipiv(mn);
    for (int i = 0; i < mn; i++) ipiv[i] = i + 1 + (magma_int_t)((r1() + 0.5) * (m - i)) % (m - i);
    std::vector<Z> R = A; magma_int_t N = n, LD = ld, k1 = 1, k2 = mn, inc = 1; lapackf77_zlaswp(&N, R.data(), &LD, &k1, &k2, ipiv.data(), &inc);
    if (op == "laswpx") { Z* dA = up(A); magmablas_zlaswpx(n, dA, 1, ld, 1, mn, ipiv.data(), 1, q); err = dif(dn(dA, A.size()), R); }
    else { int ldt = n + pad; std::vector<Z> AT((size_t)ldt * m, MAGMA_Z_ZERO); for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) at(AT, ldt, j, i) = at(A, ld, i, j);
      magma_int_t* dip; magma_imalloc(&dip, mn); magma_isetvector(mn, ipiv.data(), 1, dip, 1, q); Z* dAT = up(AT); magmablas_zlaswp2(n, dAT, ldt, 1, mn, dip, 1, q); magma_free(dip);
      std::vector<Z> XT = dn(dAT, AT.size()), X = A; for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) at(X, ld, i, j) = at(XT, ldt, j, i); err = dif(X, R); } }
  else if (op == "swapdblk") { int nb = k, ld = n + pad; std::vector<Z> A((size_t)ld * n), B((size_t)ld * n); for (auto& v : A) v = rz(); for (auto& v : B) v = rz(); std::vector<Z> RA = A, RB = B;
    for (int b = 0; b < n / nb; b++) for (int j = 0; j < nb; j++) for (int i = 0; i < nb; i++) std::swap(at(RA, ld, b * nb + i, b * nb + j), at(RB, ld, b * nb + i, b * nb + j));
    Z* dA = up(A); Z* dB = up(B); magmablas_zswapdblk(n, nb, dA, ld, 1, dB, ld, 1, q); err = std::max(dif(dn(dA, A.size()), RA), dif(dn(dB, B.size()), RB)); }
  else if (op == "symmetrize") { int ld = n + pad; std::vector<Z> A((size_t)ld * n); for (auto& v : A) v = rz(); std::vector<Z> RH = A, RS = A;
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { bool src = o1 == 'L' ? i > j : i < j; if (i == j) { at(RH, ld, i, i) = MAGMA_Z_MAKE(MAGMA_Z_REAL(at(A, ld, i, i)), 0); at(RS, ld, i, i) = at(RH, ld, i, i); }
      else if (src) { at(RH, ld, j, i) = cj(at(A, ld, i, j)); at(RS, ld, j, i) = at(A, ld, i, j); } }
    Z* dA = up(A); magmablas_zsymmetrize(o1 == 'U' ? MagmaUpper : MagmaLower, n, dA, ld, q); std::vector<Z> X = dn(dA, A.size()); err = dif(X, RH); err2 = dif(X, RS); }
  else if (op == "trmv") { int ld = n + pad, ix = std::max(k, 1); std::vector<Z> A((size_t)ld * n), x((size_t)n * ix); for (auto& v : A) v = rz(); for (auto& v : x) v = rz(); std::vector<Z> R = x;
    char U = o1, T = o2, D = argv[7][1] ? argv[7][1] : 'N'; magma_int_t N = n, LD = ld, IX = ix; blasf77_ztrmv(&U, &T, "N", &N, A.data(), &LD, R.data(), &IX);
    Z* dA = up(A); Z* dx = up(x); magmablas_ztrmv(U == 'U' ? MagmaUpper : MagmaLower, T == 'N' ? MagmaNoTrans : T == 'T' ? MagmaTrans : Magma_ConjTrans, MagmaNonUnit, n, dA, ld, dx, ix, q); magma_free(dA); err = dif(dn(dx, x.size()), R); (void)D; }
  else if (op == "transpose_conj") { int la = m + pad, lt = n + pad; std::vector<Z> A((size_t)la * n), T((size_t)lt * m, MAGMA_Z_ZERO); for (auto& v : A) v = rz(); std::vector<Z> R = T;
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) at(R, lt, j, i) = cj(at(A, la, i, j)); Z* dA = up(A); Z* dT = up(T); magmablas_ztranspose_conj(m, n, dA, la, dT, lt, q); magma_free(dA); err = dif(dn(dT, T.size()), R); }
  else if (op == "transpose_inplace" || op == "transpose_conj_inplace") { int ld = n + pad; bool c = op == "transpose_conj_inplace"; std::vector<Z> A((size_t)ld * n); for (auto& v : A) v = rz(); std::vector<Z> R = A;
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { Z v = at(A, ld, j, i); at(R, ld, i, j) = c ? cj(v) : v; }
    Z* dA = up(A); if (c) magmablas_ztranspose_conj_inplace(n, dA, ld, q); else magmablas_ztranspose_inplace(n, dA, ld, q); err = dif(dn(dA, A.size()), R); }
  else if (op == "symv") { int ld = magma_roundup(n, 32); std::vector<Z> A((size_t)ld * n, MAGMA_Z_ZERO), x(n), y(n); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { Z v = rz(); at(A, ld, i, j) = v; at(A, ld, j, i) = v; }
    for (auto& v : x) v = rz(); for (auto& v : y) v = rz(); std::vector<Z> R = y; for (int i = 0; i < n; i++) { Z s = MAGMA_Z_ZERO; for (int j = 0; j < n; j++) s = add(s, mul(at(A, ld, i, j), x[j])); R[i] = add(mul(al, s), mul(be, y[i])); }
    std::vector<Z> Ad = A; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if ((o1 == 'L' && i < j) || (o1 == 'U' && i > j)) at(Ad, ld, i, j) = MAGMA_Z_MAKE(99, 99);   // the other triangle must not be read
    Z* dA = up(Ad); Z* dx = up(x); Z* dy = up(y); magma_int_t lw = ld * magma_ceildiv(n, 64); Z* dw; magma_zmalloc(&dw, lw);
    magmablas_zsymv_work(o1 == 'U' ? MagmaUpper : MagmaLower, n, al, dA, ld, dx, 1, be, dy, 1, dw, lw, q); magma_free(dw); magma_free(dA); magma_free(dx); err = dif(dn(dy, n), R); }
  else if (op == "larfg") { std::vector<Z> v(n); for (auto& a : v) a = rz(); std::vector<Z> R = v; Z tau; magma_int_t N = n, one = 1; lapackf77_zlarfg(&N, &R[0], &R[1], &one, &tau);
    Z* dv = up(v); Z* dt; magma_zmalloc(&dt, 1); magmablas_zlarfg(n, dv, dv + 1, 1, dt, q); std::vector<Z> X = dn(dv, n); std::vector<Z> T = dn(dt, 1); err = dif(X, R); err2 = MAGMA_Z_ABS(MAGMA_Z_SUB(T[0], tau)); }
  else if (op == "lacpy_conj") { int l1 = std::max(k, 1), l2 = std::max(pad, 1); std::vector<Z> A1((size_t)n * l1), A2((size_t)n * l2, MAGMA_Z_ZERO); for (auto& v : A1) v = rz(); std::vector<Z> R = A2;
    for (int i = 0; i < n; i++) R[(size_t)i * l2] = cj(A1[(size_t)i * l1]); Z* d1 = up(A1); Z* d2 = up(A2); magmablas_zlacpy_conj(n, d1, l1, d2, l2, q); magma_free(d1); err = dif(dn(d2, A2.size()), R); }
  magma_queue_sync(q);
  bool ok = op == "gemm_reduce" ? (err < 1e-12) : op == "symmetrize" ? (err < 1e-14) : op == "larfg" ? (err < 1e-12 && err2 < 1e-12) : (err >= 0 && err < 1e-12);
  printf("r63,%s,%d,%d,%d,%c,%c,%s,%.3g,%.3g,%s\n", op.c_str(), m, n, k, o1, o2, argv[7], err, err2, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

// Round 71 (roxy): user-facing batched routines never covered before, 3 problems per batch, against CPU references.
//   round71_batched_misc <op> <m> <n> <k> <o1> <o2>
//   gemm_s  m n k tA tB   magmablas_zgemm_batched_strided (stride = ld*cols + 7 to catch stride misuse)
//   gemv_s  m n - tA -    magmablas_zgemv_batched_strided (incx = incy = 1, stride = len + 3)
//   geadd   m n           magmablas_zgeadd_batched: B := alpha A + B
//   lacpy   m n - uplo    magmablas_zlacpy_batched (U / L / F; the rest of B untouched)
//   laset   m n - uplo    magmablas_zlaset_batched (offdiag on the uplo part, diag on the diagonal, the rest untouched)
//   transc  m n           magmablas_ztranspose_conj_batched: AT := A^H
//   hemv    - n - uplo    magmablas_zhemv_batched (other triangle poisoned)
//   gbsv / gbsv_s  - n kl ku   magma_zgbsv_batched / _strided, nrhs 2, against LAPACK zgbsv (band storage ldab = 2kl+ku+1)
// CSV: r71,op,m,n,k,o1,o2,info,err,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
typedef std::complex<double> cd; typedef magmaDoubleComplex Z;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static magma_queue_t q; static const int BC = 3;
static Z zz(cd v) { return MAGMA_Z_MAKE(v.real(), v.imag()); } static cd cc(Z v) { return cd(MAGMA_Z_REAL(v), MAGMA_Z_IMAG(v)); }
static Z* up(const std::vector<cd>& h) { std::vector<Z> t(h.size()); for (size_t i = 0; i < h.size(); i++) t[i] = zz(h[i]); Z* d; magma_zmalloc(&d, std::max<size_t>(t.size(), 1)); magma_zsetvector((magma_int_t)t.size(), t.data(), 1, d, 1, q); return d; }
static std::vector<cd> dn(Z* d, size_t n) { std::vector<Z> t(n); magma_zgetvector((magma_int_t)n, d, 1, t.data(), 1, q); std::vector<cd> h(n); for (size_t i = 0; i < n; i++) h[i] = cc(t[i]); return h; }
static double dif(const std::vector<cd>& X, const std::vector<cd>& R) { double e = 0, a = 0; for (size_t i = 0; i < X.size(); i++) { double d = std::abs(X[i] - R[i]); if (!(d <= e)) e = d; a = std::max(a, std::abs(R[i])); } return a > 0 ? e / a : e; }
static std::vector<cd> rnd(size_t k) { std::vector<cd> v(k); for (auto& x : v) x = cd(r1(), r1()); return v; }
static Z** ptrs(Z* base, size_t stride) { std::vector<Z*> h(BC); for (int b = 0; b < BC; b++) h[b] = base + b * stride; Z** d; magma_malloc((void**)&d, BC * sizeof(Z*)); magma_setvector(BC, sizeof(Z*), h.data(), 1, d, 1, q); return d; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 7) return 1; std::string op = argv[1]; int m = atoi(argv[2]), n = atoi(argv[3]), k = atoi(argv[4]); char o1 = argv[5][0], o2 = argv[6][0];
  magma_init(); magma_queue_create(0, &q);
  cd al(0.7, -0.3), be(-0.4, 0.2); double err = -1; long long info = 0;
  auto T = [](char c) { return c == 'N' ? MagmaNoTrans : c == 'T' ? MagmaTrans : Magma_ConjTrans; };
  auto opA = [](const std::vector<cd>& M, int ld, char t, int i, int j, size_t off) { cd v = t == 'N' ? M[off + i + (size_t)j * ld] : M[off + j + (size_t)i * ld]; return t == 'C' ? std::conj(v) : v; };
  if (op == "gemm_s") { int ra = o1 == 'N' ? m : k, ca = o1 == 'N' ? k : m, rb = o2 == 'N' ? k : n, cb = o2 == 'N' ? n : k;
    size_t sA = (size_t)ra * ca + 7, sB = (size_t)rb * cb + 7, sC = (size_t)m * n + 7; std::vector<cd> A = rnd(sA * BC), B = rnd(sB * BC), C = rnd(sC * BC), R = C;
    for (int b = 0; b < BC; b++) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { cd s = 0; for (int l = 0; l < k; l++) s += opA(A, ra, o1, i, l, b * sA) * opA(B, rb, o2, l, j, b * sB); R[b * sC + i + (size_t)j * m] = al * s + be * C[b * sC + i + (size_t)j * m]; }
    Z* dA = up(A); Z* dB = up(B); Z* dC = up(C); magmablas_zgemm_batched_strided(T(o1), T(o2), m, n, k, zz(al), dA, ra, sA, dB, rb, sB, zz(be), dC, m, sC, BC, q); magma_queue_sync(q); err = dif(dn(dC, C.size()), R); }
  else if (op == "gemv_s") { int lx = o1 == 'N' ? n : m, ly = o1 == 'N' ? m : n; size_t sA = (size_t)m * n + 7, sx = lx + 3, sy = ly + 3; std::vector<cd> A = rnd(sA * BC), x = rnd(sx * BC), y = rnd(sy * BC), R = y;
    for (int b = 0; b < BC; b++) for (int i = 0; i < ly; i++) { cd s = 0; for (int j = 0; j < lx; j++) s += opA(A, m, o1, i, j, b * sA) * x[b * sx + j]; R[b * sy + i] = al * s + be * y[b * sy + i]; }
    Z* dA = up(A); Z* dx = up(x); Z* dy = up(y); magmablas_zgemv_batched_strided(T(o1), m, n, zz(al), dA, m, sA, dx, 1, sx, zz(be), dy, 1, sy, BC, q); magma_queue_sync(q); err = dif(dn(dy, y.size()), R); }
  else if (op == "geadd" || op == "lacpy" || op == "laset" || op == "transc") { int ld = m + 2, ldt = n + 2; size_t s = (size_t)ld * n, st = (size_t)ldt * m;
    std::vector<cd> A = rnd(s * BC), B = rnd((op == "transc" ? st : s) * BC), R = B; cd od(3, 1), dg(5, -2);
    for (int b = 0; b < BC; b++) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { size_t ia = b * s + i + (size_t)j * ld;
      bool in = o1 == 'F' || (o1 == 'U' && i <= j) || (o1 == 'L' && i >= j);
      if (op == "geadd") R[ia] = al * A[ia] + B[ia];
      else if (op == "lacpy") { if (in) R[ia] = A[ia]; }
      else if (op == "laset") { if (i == j) R[ia] = dg; else if ((o1 == 'F') || (o1 == 'U' && i < j) || (o1 == 'L' && i > j)) R[ia] = od; }
      else R[b * st + j + (size_t)i * ldt] = std::conj(A[ia]); }
    Z* dA = up(A); Z* dB = up(B); Z** pA = ptrs(dA, s); Z** pB = ptrs(dB, op == "transc" ? st : s); magma_uplo_t u = o1 == 'U' ? MagmaUpper : o1 == 'L' ? MagmaLower : MagmaFull;
    if (op == "geadd") magmablas_zgeadd_batched(m, n, zz(al), (Z const* const*)pA, ld, pB, ld, BC, q);
    else if (op == "lacpy") magmablas_zlacpy_batched(u, m, n, (Z const* const*)pA, ld, pB, ld, BC, q);
    else if (op == "laset") magmablas_zlaset_batched(u, m, n, zz(od), zz(dg), pB, ld, BC, q);
    else magmablas_ztranspose_conj_batched(m, n, pA, ld, pB, ldt, BC, q);
    magma_queue_sync(q); err = dif(dn(dB, B.size()), R); }
  else if (op == "hemv") { int ld = n + 1; size_t s = (size_t)ld * n; std::vector<cd> A(s * BC), x = rnd((size_t)n * BC), y = rnd((size_t)n * BC), R = y;
    for (int b = 0; b < BC; b++) for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { cd v(r1(), r1()); if (i == j) v = cd(v.real(), 0); A[b * s + i + (size_t)j * ld] = v; A[b * s + j + (size_t)i * ld] = std::conj(v); }
    for (int b = 0; b < BC; b++) for (int i = 0; i < n; i++) { cd t = 0; for (int j = 0; j < n; j++) t += A[b * s + i + (size_t)j * ld] * x[b * n + j]; R[b * n + i] = al * t + be * y[b * n + i]; }
    std::vector<cd> Ad = A; for (int b = 0; b < BC; b++) for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if ((o1 == 'L' && i < j) || (o1 == 'U' && i > j)) Ad[b * s + i + (size_t)j * ld] = cd(1e6, 1e6);
    Z* dA = up(Ad); Z* dx = up(x); Z* dy = up(y); Z** pA = ptrs(dA, s); Z** px = ptrs(dx, n); Z** py = ptrs(dy, n);
    magmablas_zhemv_batched(o1 == 'U' ? MagmaUpper : MagmaLower, n, zz(al), pA, ld, px, 1, zz(be), py, 1, BC, q); magma_queue_sync(q); err = dif(dn(dy, y.size()), R); }
  else if (op == "gbsv" || op == "gbsv_s") { int kl = k, ku = atoi(argv[5]), N = n, nrhs = 2, ldab = 2 * kl + ku + 1; size_t sA = (size_t)ldab * N, sB = (size_t)N * nrhs;
    std::vector<cd> AB(sA * BC, 0.0), B = rnd(sB * BC);
    for (int b = 0; b < BC; b++) for (int j = 0; j < N; j++) for (int i = std::max(0, j - ku); i <= std::min(N - 1, j + kl); i++) AB[b * sA + kl + ku + i - j + (size_t)j * ldab] = i == j ? cd(1 + r1(), r1()) : cd(r1(), r1());   // not diagonally dominant: pivoting matters
    std::vector<cd> R = B; for (int b = 0; b < BC; b++) { std::vector<Z> ab(sA), bb(sB); for (size_t i = 0; i < sA; i++) ab[i] = zz(AB[b * sA + i]); for (size_t i = 0; i < sB; i++) bb[i] = zz(B[b * sB + i]);
      std::vector<magma_int_t> ip(N); magma_int_t NN = N, KL = kl, KU = ku, NR = nrhs, LD = ldab, rinfo; lapackf77_zgbsv(&NN, &KL, &KU, &NR, ab.data(), &LD, ip.data(), bb.data(), &NN, &rinfo); for (size_t i = 0; i < sB; i++) R[b * sB + i] = cc(bb[i]); }
    Z* dA = up(AB); Z* dB = up(B); magma_int_t *dip, *dinfo; magma_imalloc(&dip, (size_t)N * BC); magma_imalloc(&dinfo, BC);
    if (op == "gbsv") { Z** pA = ptrs(dA, sA); Z** pB = ptrs(dB, sB); std::vector<magma_int_t*> hp(BC); for (int b = 0; b < BC; b++) hp[b] = dip + b * N; magma_int_t** pp; magma_malloc((void**)&pp, BC * sizeof(magma_int_t*)); magma_setvector(BC, sizeof(magma_int_t*), hp.data(), 1, pp, 1, q);
      info = magma_zgbsv_batched(N, kl, ku, nrhs, pA, ldab, pp, pB, N, dinfo, BC, q); }
    else info = magma_zgbsv_batched_strided(N, kl, ku, nrhs, dA, ldab, sA, dip, N, dB, N, sB, dinfo, BC, q);
    magma_queue_sync(q); std::vector<magma_int_t> hi(BC); magma_igetvector(BC, dinfo, 1, hi.data(), 1, q); for (auto v : hi) if (v) info = v; err = dif(dn(dB, B.size()), R); }
  bool ok = info == 0 && err >= 0 && err < 1e-11;
  printf("r71,%s,%d,%d,%d,%c,%c,%lld,%.3g,%s\n", op.c_str(), m, n, k, o1, o2, info, err, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

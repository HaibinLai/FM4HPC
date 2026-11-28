// Twelfth GPU round: status logic inside the magmablas kernels (HANDOFF §29.12).  GPU 0 only; one mode per process.
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z;
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static unsigned S = 1234u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static std::vector<Z> ident(int m, int n) { std::vector<Z> A((size_t)m * n, C(0)); for (int i = 0; i < std::min(m, n); i++) A[i + (size_t)i * m] = C(1); return A; }
static magma_int_t lapack_getrf(int m, int n, std::vector<Z> A) { magma_int_t mm = m, nn = n, info; std::vector<magma_int_t> ip(std::min(m, n)); lapackf77_zgetrf(&mm, &nn, A.data(), &mm, ip.data(), &info); return info; }
static magma_int_t lapack_potrf(int n, std::vector<Z> A) { magma_int_t nn = n, info; lapackf77_zpotrf("L", &nn, A.data(), &nn, &info); return info; }
struct Batch { int b; std::vector<Z*> d; Z** dA; magma_int_t* dinfo; std::vector<magma_int_t*> dip; magma_int_t** dipa; };
static Batch upload(const std::vector<std::vector<Z>>& Ms, int ld, int cols, int npiv) {
  Batch B; B.b = (int)Ms.size(); B.d.resize(B.b); B.dip.resize(B.b);
  for (int i = 0; i < B.b; i++) { magma_zmalloc(&B.d[i], (size_t)ld * cols); magma_zsetmatrix(ld, cols, Ms[i].data(), ld, B.d[i], ld, Q); magma_imalloc(&B.dip[i], std::max(npiv, 1)); }
  magma_malloc((void**)&B.dA, B.b * sizeof(Z*)); magma_setvector(B.b, sizeof(Z*), B.d.data(), 1, B.dA, 1, Q);
  magma_malloc((void**)&B.dipa, B.b * sizeof(magma_int_t*)); magma_setvector(B.b, sizeof(magma_int_t*), B.dip.data(), 1, B.dipa, 1, Q);
  magma_imalloc(&B.dinfo, B.b); std::vector<magma_int_t> m1(B.b, -999); magma_setvector(B.b, sizeof(magma_int_t), m1.data(), 1, B.dinfo, 1, Q); return B; }
static std::vector<magma_int_t> infos(Batch& B) { std::vector<magma_int_t> v(B.b); magma_getvector(B.b, sizeof(magma_int_t), B.dinfo, 1, v.data(), 1, Q); return v; }

static void f1() {   // getrf_vbatched: zero pivot in the second panel
  int n = 64, b = 2; std::vector<std::vector<Z>> M = {ident(n, n), ident(n, n)}; M[0][39 + 39 * n] = C(0);
  Batch B = upload(M, n, n, n); std::vector<magma_int_t> mv(b + 1, n), lv(b + 1, n); magma_int_t *dm, *dn, *dl; magma_imalloc(&dm, b + 1); magma_imalloc(&dn, b + 1); magma_imalloc(&dl, b + 1);
  magma_setvector(b + 1, sizeof(magma_int_t), mv.data(), 1, dm, 1, Q); magma_setvector(b + 1, sizeof(magma_int_t), mv.data(), 1, dn, 1, Q); magma_setvector(b + 1, sizeof(magma_int_t), lv.data(), 1, dl, 1, Q);
  magma_zgetrf_vbatched(dm, dn, B.dA, dl, B.dipa, B.dinfo, b, Q); auto v = infos(B);
  printf("F1 zgetrf_vbatched 64x64, matrix 0 = I with A(40,40)=0: info = %lld %lld | LAPACK %lld %lld\n", (long long)v[0], (long long)v[1], (long long)lapack_getrf(n, n, M[0]), (long long)lapack_getrf(n, n, M[1]));
  M[0] = ident(n, n); M[0][9 + 9 * n] = C(0); Batch B2 = upload(M, n, n, n); magma_zgetrf_vbatched(dm, dn, B2.dA, dl, B2.dipa, B2.dinfo, b, Q); v = infos(B2);
  printf("   control, A(10,10)=0 (first panel): info = %lld %lld | LAPACK %lld\n", (long long)v[0], (long long)v[1], (long long)lapack_getrf(n, n, M[0]));
}
// band storage AB(kl+ku+i-j, j) = A(i,j), lddab = 2kl+ku+1
static std::vector<Z> toband(const std::vector<Z>& A, int m, int n, int kl, int ku, int ld) { std::vector<Z> AB((size_t)ld * n, C(0));
  for (int j = 0; j < n; j++) for (int i = std::max(0, j - ku); i < std::min(m, j + kl + 1); i++) AB[(kl + ku + i - j) + (size_t)j * ld] = A[i + (size_t)j * m]; return AB; }
static magma_int_t lapack_gbtrf(int m, int n, int kl, int ku, std::vector<Z> AB, int ld) { magma_int_t mm = m, nn = n, l = kl, u = ku, lda = ld, info; std::vector<magma_int_t> ip(std::min(m, n));
  lapackf77_zgbtrf(&mm, &nn, &l, &u, AB.data(), &lda, ip.data(), &info); return info; }
static void gb(const char* tag, int m, int n, int kl, int ku, std::vector<Z> A, int junkRow = -1, int junkCol = -1) {
  int ld = 2 * kl + ku + 1; auto AB = toband(A, m, n, kl, ku, ld); if (junkRow >= 0) AB[(kl + ku + junkRow - junkCol) + (size_t)junkCol * ld] = C(1);
  std::vector<std::vector<Z>> M = {AB, toband(ident(m, n), m, n, kl, ku, ld)}; Batch B = upload(M, ld, n, std::min(m, n));
  magma_int_t r = magma_zgbtrf_batched(m, n, kl, ku, B.dA, ld, B.dipa, B.dinfo, 2, Q); auto v = infos(B);
  std::vector<magma_int_t> ip(std::min(m, n)); magma_igetvector(std::min(m, n), B.dip[0], 1, ip.data(), 1, Q); int maxp = 0; for (auto p : ip) maxp = std::max(maxp, (int)p);
  printf("%s: ret=%lld info = %lld %lld | LAPACK zgbtrf %lld  (max ipiv of matrix 0 = %d, m = %d)\n", tag, (long long)r, (long long)v[0], (long long)v[1], (long long)lapack_gbtrf(m, n, kl, ku, AB, ld), maxp, m);
}
static void f2() { int n = 200; auto A = ident(n, n); A[4 + 4 * n] = C(0); gb("F2 zgbtrf_batched n=200 kl=ku=64 (generic fallback), A(5,5)=0", n, n, 64, 64, A);
  A = ident(n, n); A[199 + 199 * n] = C(0); gb("   same, only A(200,200)=0", n, n, 64, 64, A); }
static void f8() { { int m = 6, n = 4; auto A = ident(m, n); A[3 + 3 * m] = C(0); A[4 + 3 * m] = C(1); gb("F8a zgbtrf_batched m=6 n=4 kl=2 ku=1, A(4,4)=0, A(5,4)=1", m, n, 2, 1, A); }
  { int m = 4, n = 6; auto A = ident(m, n); A[3 + 3 * m] = C(0); gb("F8b zgbtrf_batched m=4 n=6 kl=2 ku=1, A(4,4)=0, junk 1.0 in the slot of row 5", m, n, 2, 1, A, 4, 3); } }
static void f3() { int n = 8; auto A = ident(n, n); A[2 + 2 * n] = C(0); Z* d; magma_zmalloc(&d, n * n); magma_zsetmatrix(n, n, A.data(), n, d, n, Q);
  std::vector<magma_int_t> ip(n); magma_int_t info = -999; magma_zgetf2_gpu(n, n, d, n, ip.data(), Q, &info); std::vector<Z> F(n * n); magma_zgetmatrix(n, n, d, n, F.data(), n, Q);
  int nf = 0; for (auto& x : F) if (!std::isfinite(MAGMA_Z_REAL(x)) || !std::isfinite(MAGMA_Z_IMAG(x))) nf++;
  printf("F3 zgetf2_gpu 8x8 I with A(3,3)=0: info=%lld (LAPACK %lld), non-finite entries in the factor: %d\n", (long long)info, (long long)lapack_getrf(n, n, A), nf); }
static void f4() { int n = 8; for (double v : {0.0, (double)NAN}) { auto A = ident(n, n); A[2 + 2 * n] = C(v); Z* d; magma_zmalloc(&d, n * n);
    magma_zsetmatrix(n, n, A.data(), n, d, n, Q); magma_int_t i1 = -999; magma_zpotf2_gpu(MagmaLower, n, d, n, Q, &i1);
    magma_zsetmatrix(n, n, A.data(), n, d, n, Q); magma_int_t i2 = -999; magma_zpotrf_native(MagmaUpper, n, d, n, &i2);
    printf("F4 8x8 I with A(3,3)=%g: zpotf2_gpu info=%lld, zpotrf_native(Upper) info=%lld | LAPACK %lld\n", v, (long long)i1, (long long)i2, (long long)lapack_potrf(n, A)); magma_free(d); } }
static void f5() { for (int n : {12, 16}) { std::vector<Z> A((size_t)n * n, C(0)); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = C(4); for (int i = 1; i < n; i++) { A[i + (size_t)(i - 1) * n] = C(-1); A[(i - 1) + (size_t)i * n] = C(-1); }
    auto A0 = A; A0[9 + 9 * n] = C(NAN); std::vector<std::vector<Z>> M = {A0, A}; Batch B = upload(M, n, n, 0);
    magma_zpotrf_batched(MagmaLower, n, B.dA, n, B.dinfo, 2, Q); auto v = infos(B);
    printf("F5 zpotrf_batched n=%d, matrix 0 A(10,10)=NaN: info = %lld %lld | LAPACK %lld\n", n, (long long)v[0], (long long)v[1], (long long)lapack_potrf(n, A0)); } }
static void f6() { int m = 16384, n = 64; std::vector<Z> A((size_t)m * n); for (auto& a : A) a = C(rnd(), rnd()); for (int i = 0; i < m; i++) A[i + (size_t)2 * m] = C(0);
  Z* d; magma_zmalloc(&d, (size_t)m * n); magma_zsetmatrix(m, n, A.data(), m, d, m, Q); std::vector<magma_int_t> ip(n); magma_int_t info = -999;
  magma_zgetrf_native(m, n, d, m, ip.data(), &info);
  printf("F6 zgetrf_native m=16384 n=64, column 3 all zero: info=%lld | LAPACK %lld\n", (long long)info, (long long)lapack_getrf(m, n, A));
  { int m2 = 4096; std::vector<Z> A2(A.begin(), A.begin() + (size_t)m2 * n); for (int j = 0; j < n; j++) for (int i = 0; i < m2; i++) A2[i + (size_t)j * m2] = A[i + (size_t)j * m];
    magma_zsetmatrix(m2, n, A2.data(), m2, d, m2, Q); info = -999; magma_zgetrf_native(m2, n, d, m2, ip.data(), &info); printf("   control m=4096: info=%lld\n", (long long)info); } }
static void f9() { int n = 8; std::vector<Z> A((size_t)n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { Z v = i == j ? C(rnd() + 2) : C(rnd(), rnd()); A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = MAGMA_Z_CONJ(v); }
  auto A0 = A; A0[1 + 4 * n] = C(NAN); A0[4 + 1 * n] = C(NAN); std::vector<std::vector<Z>> M = {A0, A}; Batch B = upload(M, n, n, 0);
  std::vector<double*> dW(2); for (auto& w : dW) magma_dmalloc(&w, n); double** dWa; magma_malloc((void**)&dWa, 2 * sizeof(double*)); magma_setvector(2, sizeof(double*), dW.data(), 1, dWa, 1, Q);
  magma_int_t r = magma_zheevj_batched_small_sm(MagmaVec, MagmaLower, n, B.dA, n, dWa, B.dinfo, 2, Q); auto v = infos(B);
  std::vector<double> w0(n); magma_dgetvector(n, dW[0], 1, w0.data(), 1, Q); int nf = 0; for (double x : w0) if (!std::isfinite(x)) nf++;
  printf("F9 zheevj_batched_small_sm n=8, matrix 0 has a NaN pair: ret=%lld info = %lld %lld; non-finite eigenvalues of matrix 0: %d\n", (long long)r, (long long)v[0], (long long)v[1], nf); }
static void f10() { int n = 2048, b = 2; std::vector<std::vector<Z>> M(b); for (auto& A : M) { A.assign((size_t)n * n, C(0)); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = C(2); }
  M[0][0] = C(1e-12); Batch B = upload(M, n, n, 0); std::vector<magma_int_t> mv(b + 1, n); magma_int_t *dm, *dn, *dl; magma_imalloc(&dm, b + 1); magma_imalloc(&dn, b + 1); magma_imalloc(&dl, b + 1);
  magma_setvector(b + 1, sizeof(magma_int_t), mv.data(), 1, dm, 1, Q); magma_setvector(b + 1, sizeof(magma_int_t), mv.data(), 1, dn, 1, Q); magma_setvector(b + 1, sizeof(magma_int_t), mv.data(), 1, dl, 1, Q);
  std::vector<double> tol(b, 1e-8); double* dtol; magma_dmalloc(&dtol, b); magma_dsetvector(b, tol.data(), 1, dtol, 1, Q);
  magma_zgetrf_nopiv_expert_vbatched(dm, dn, B.dA, dl, dtol, 1e-8, B.dinfo, b, Q); auto v = infos(B);
  printf("F10 zgetrf_nopiv_expert_vbatched 2048x2048, matrix 0 A(1,1)=1e-12 < tol 1e-8: info = %lld %lld (documented: number of replaced pivots, expected 1 0)\n", (long long)v[0], (long long)v[1]); }
static void f11() { int n = 8; std::vector<std::vector<Z>> M(2); for (auto& A : M) { A.resize(n * n); for (auto& a : A) a = C(rnd(), rnd()); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = MAGMA_Z_ADD(A[i + (size_t)i * n], C(4)); }
  for (int i = 0; i < n; i++) M[0][i + 2 * n] = C(0);
  Batch B = upload(M, n, n, n); std::vector<std::vector<Z>> R(2, std::vector<Z>(n, C(1))); std::vector<Z*> dB(2); for (int i = 0; i < 2; i++) { magma_zmalloc(&dB[i], n); magma_zsetvector(n, R[i].data(), 1, dB[i], 1, Q); }
  Z** dBa; magma_malloc((void**)&dBa, 2 * sizeof(Z*)); magma_setvector(2, sizeof(Z*), dB.data(), 1, dBa, 1, Q);
  magma_zgesv_batched(n, 1, B.dA, n, B.dipa, dBa, n, B.dinfo, 2, Q); auto v = infos(B); std::vector<Z> x(n); magma_zgetvector(n, dB[0], 1, x.data(), 1, Q);
  int nf = 0; for (auto& e : x) if (!std::isfinite(MAGMA_Z_REAL(e)) || !std::isfinite(MAGMA_Z_IMAG(e))) nf++;
  printf("F11 zgesv_batched n=8 nrhs=1, matrix 0 column 3 = 0: info = %lld %lld (LAPACK %lld); B of matrix 0: %d of %d entries non-finite (LAPACK leaves B untouched)\n",
         (long long)v[0], (long long)v[1], (long long)lapack_getrf(n, n, M[0]), nf, n); }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; magma_init(); magma_queue_create(0, &Q);
  if (m == "f1") f1(); else if (m == "f2") f2(); else if (m == "f3") f3(); else if (m == "f4") f4(); else if (m == "f5") f5(); else if (m == "f6") f6();
  else if (m == "f8") f8(); else if (m == "f9") f9(); else if (m == "f10") f10(); else if (m == "f11") f11();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

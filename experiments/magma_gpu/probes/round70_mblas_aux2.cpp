// Round 70 (roxy): the remaining untested magmablas auxiliaries, against CPU references written from their docs.
//   round70_mblas_aux2 herk_small <n> <k> <uplo> <trans N|C> <beta0 0|1> <nblocks>   C := alpha op(A) op(A)^H + beta C, n <= 32;
//                      the other triangle must stay untouched
//   round70_mblas_aux2 symtiles <m> <ntile> <mstride> <nstride> <uplo>   ZSYMMETRIZE_TILES (Hermitian copy, real diagonal, per tile)
//   round70_mblas_aux2 laswp_sym <n> <k2>                 symmetric pivoting of the lower triangle, k1 = 1, ipiv random >= k;
//                      err against swapping rows+columns of the full Hermitian matrix (err) and of the complex symmetric one (err2)
//   round70_mblas_aux2 hemv|symv <n> <uplo> <incx>         y := alpha A x + beta y (Hermitian / complex symmetric), the other triangle poisoned
// CSV: r70,op,a,b,c,d,e,err,err2,verdict
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
static magma_queue_t q;
static Z zz(cd v) { return MAGMA_Z_MAKE(v.real(), v.imag()); } static cd cc(Z v) { return cd(MAGMA_Z_REAL(v), MAGMA_Z_IMAG(v)); }
static Z* up(const std::vector<cd>& h) { std::vector<Z> t(h.size()); for (size_t i = 0; i < h.size(); i++) t[i] = zz(h[i]); Z* d; magma_zmalloc(&d, std::max<size_t>(t.size(), 1)); magma_zsetvector((magma_int_t)t.size(), t.data(), 1, d, 1, q); return d; }
static std::vector<cd> dn(Z* d, size_t n) { std::vector<Z> t(n); magma_zgetvector((magma_int_t)n, d, 1, t.data(), 1, q); magma_free(d); std::vector<cd> h(n); for (size_t i = 0; i < n; i++) h[i] = cc(t[i]); return h; }
static double dif(const std::vector<cd>& X, const std::vector<cd>& R) { double e = 0, a = 0; for (size_t i = 0; i < X.size(); i++) { double d = std::abs(X[i] - R[i]); if (!(d <= e)) e = d; a = std::max(a, std::abs(R[i])); } return a > 0 ? e / a : e; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 3) return 1; std::string op = argv[1]; magma_init(); magma_queue_create(0, &q);
  double err = -1, err2 = -1; std::string tag;
  if (op == "herk_small") {
    int n = atoi(argv[2]), k = atoi(argv[3]); char ul = argv[4][0], tr = argv[5][0]; bool b0 = atoi(argv[6]) == 1; int nb = atoi(argv[7]);
    int ra = tr == 'N' ? n : k, ca = tr == 'N' ? k : n, lda = ra, ldc = n; std::vector<cd> A((size_t)lda * ca), C((size_t)ldc * n);
    for (auto& v : A) v = cd(r1(), r1()); for (auto& v : C) v = cd(r1(), r1()); for (int i = 0; i < n; i++) C[i + (size_t)i * n] = cd(C[i + (size_t)i * n].real(), 0);
    double al = 0.7, be = b0 ? 0.0 : -0.4; std::vector<cd> R = C;
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { if ((ul == 'L' && i < j) || (ul == 'U' && i > j)) continue; cd s = 0;
      for (int l = 0; l < k; l++) { cd a = tr == 'N' ? A[i + (size_t)l * lda] : std::conj(A[l + (size_t)i * lda]); cd b = tr == 'N' ? std::conj(A[j + (size_t)l * lda]) : A[l + (size_t)j * lda]; s += a * b; }
      R[i + (size_t)j * n] = al * s + (b0 ? cd(0) : be * C[i + (size_t)j * n]); if (i == j) R[i + (size_t)j * n] = cd(R[i + (size_t)j * n].real(), 0); }
    Z* dA = up(A); Z* dC = up(C); magmablas_zherk_small_reduce(ul == 'U' ? MagmaUpper : MagmaLower, tr == 'N' ? MagmaNoTrans : Magma_ConjTrans, n, k, al, dA, lda, be, dC, ldc, nb, q); magma_free(dA);
    err = dif(dn(dC, C.size()), R); tag = std::string(argv[2]) + "," + argv[3] + "," + argv[4] + argv[5] + "," + argv[6] + "," + argv[7];
  } else if (op == "symtiles") {
    int m = atoi(argv[2]), nt = atoi(argv[3]), ms = atoi(argv[4]), ns = atoi(argv[5]); char ul = argv[6][0]; int rows = m + ms * (nt - 1), cols = m + ns * (nt - 1), ld = rows;
    std::vector<cd> A((size_t)ld * cols); for (auto& v : A) v = cd(r1(), r1()); std::vector<cd> R = A;
    for (int t = 0; t < nt; t++) { int r0 = t * ms, c0 = t * ns; for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) { cd& d = R[(r0 + i) + (size_t)(c0 + j) * ld];
        bool src = ul == 'L' ? i > j : i < j; if (i == j) d = cd(A[(r0 + i) + (size_t)(c0 + j) * ld].real(), 0); else if (!src) d = std::conj(A[(r0 + j) + (size_t)(c0 + i) * ld]); } }
    Z* dA = up(A); magmablas_zsymmetrize_tiles(ul == 'U' ? MagmaUpper : MagmaLower, m, dA, ld, nt, ms, ns, q); err = dif(dn(dA, A.size()), R);
    tag = std::string(argv[2]) + "," + argv[3] + "," + argv[4] + "," + argv[5] + "," + argv[6];
  } else if (op == "laswp_sym") {
    int n = atoi(argv[2]), k2 = atoi(argv[3]); std::vector<cd> H((size_t)n * n), S((size_t)n * n);
    for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { cd v(r1(), r1()); if (i == j) { H[i + (size_t)i * n] = cd(v.real(), 0); S[i + (size_t)i * n] = v; } else { H[i + (size_t)j * n] = v; H[j + (size_t)i * n] = std::conj(v); S[i + (size_t)j * n] = v; S[j + (size_t)i * n] = v; } }
    std::vector<magma_int_t> ipiv(k2); for (int i = 0; i < k2; i++) ipiv[i] = i + 1 + (magma_int_t)((r1() + 0.5) * (n - i)) % (n - i);
    auto apply = [&](std::vector<cd> M) { for (int kk = 0; kk < k2; kk++) { int p = (int)ipiv[kk] - 1; if (p == kk) continue; for (int j = 0; j < n; j++) std::swap(M[kk + (size_t)j * n], M[p + (size_t)j * n]); for (int i = 0; i < n; i++) std::swap(M[i + (size_t)kk * n], M[i + (size_t)p * n]); } return M; };
    std::vector<cd> RH = apply(H), RS = apply(S);
    // the device gets the lower triangle of H (upper part poisoned with the value it should never read); compare the lower triangle only
    std::vector<cd> L = H; for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) L[i + (size_t)j * n] = cd(99, 99);
    Z* dA = up(L); magmablas_zlaswp_sym(n, dA, n, 1, k2, ipiv.data(), 1, q); std::vector<cd> X = dn(dA, L.size());
    std::vector<cd> xl, rh, rs; for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { xl.push_back(X[i + (size_t)j * n]); rh.push_back(RH[i + (size_t)j * n]); rs.push_back(RS[i + (size_t)j * n]); }
    err = dif(xl, rh); err2 = dif(xl, rs); tag = std::string(argv[2]) + "," + argv[3] + ",-,-,-";
    // RS is built from S, not H: also compare against "H's lower triangle moved without conjugation" by rebuilding S from H's lower triangle
    std::vector<cd> S2((size_t)n * n); for (int j = 0; j < n; j++) for (int i = j; i < n; i++) { S2[i + (size_t)j * n] = H[i + (size_t)j * n]; S2[j + (size_t)i * n] = H[i + (size_t)j * n]; }
    std::vector<cd> RS2 = apply(S2), rs2; for (int j = 0; j < n; j++) for (int i = j; i < n; i++) rs2.push_back(RS2[i + (size_t)j * n]); err2 = dif(xl, rs2);
  } else {   // hemv / symv
    bool he = op == "hemv"; int n = atoi(argv[2]); char ul = argv[3][0]; int ix = atoi(argv[4]); int ld = n;
    std::vector<cd> A((size_t)n * n), x((size_t)n * ix), y((size_t)n * ix); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { cd v(r1(), r1()); if (i == j) { A[i + (size_t)i * n] = he ? cd(v.real(), 0) : v; } else { A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = he ? std::conj(v) : v; } }
    for (auto& v : x) v = cd(r1(), r1()); for (auto& v : y) v = cd(r1(), r1()); cd al(0.7, -0.3), be(-0.4, 0.2); std::vector<cd> R = y;
    for (int i = 0; i < n; i++) { cd s = 0; for (int j = 0; j < n; j++) s += A[i + (size_t)j * n] * x[(size_t)j * ix]; R[(size_t)i * ix] = al * s + be * y[(size_t)i * ix]; }
    std::vector<cd> Ad = A; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if ((ul == 'L' && i < j) || (ul == 'U' && i > j)) Ad[i + (size_t)j * n] = cd(1e6, 1e6);
    Z* dA = up(Ad); Z* dx = up(x); Z* dy = up(y); magma_uplo_t u = ul == 'U' ? MagmaUpper : MagmaLower;
    if (he) magmablas_zhemv(u, n, zz(al), dA, ld, dx, ix, zz(be), dy, ix, q); else magmablas_zsymv(u, n, zz(al), dA, ld, dx, ix, zz(be), dy, ix, q);
    magma_free(dA); magma_free(dx); err = dif(dn(dy, y.size()), R); tag = std::string(argv[2]) + "," + argv[3] + "," + argv[4] + ",-,-";
  }
  magma_queue_sync(q);
  bool ok = op == "laswp_sym" ? (err < 1e-14 || err2 < 1e-14) : (err < 1e-12);
  printf("r70,%s,%s,%.3g,%.3g,%s\n", op.c_str(), tag.c_str(), err, err2, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

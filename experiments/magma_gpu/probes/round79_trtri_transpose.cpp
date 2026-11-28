// Round 79 (roxy): magmablas_ztrtri_diag_batched / _vbatched (the diagonal-block inverses that the batched trsm uses) and
// magmablas_zgetmatrix_transpose / zsetmatrix_transpose (single GPU), never tested before.
//   round79_trtri_transpose trtri <b|v> <U|L> <N|U> <nmax> <batch>
//       A: triangular, diagonal 2 + random, off-diagonal 0.3 * random; the other triangle is NaN, and the diagonal too when diag = Unit
//       dinvA (NB = 128 rows, roundup(n, 128) columns) preset to a sentinel, resetozero = 1; v: sizes random in [0, nmax] (0, 1, nmax included)
//       check: for each 128 x 128 diagonal block (the last one partial, size r): inv(r x r) * A_kk = I, the rest of dinvA = 0
//   round79_trtri_transpose getT <m> <n> <nb> <pinned 0|1>   dAT (n x m, ldda n+2) -> hA (m x n, lda m+1); dwork 2*lddw*nb
//       prints how many entries of hA were still the sentinel right after the call (before any sync) and the error after a sync
//   round79_trtri_transpose setT <m> <n> <nb>                hA (m x n) -> dAT (n x m)
// CSV: r79t,b|v,uplo,diag,nmax,batch,max_err,nonzero_outside,verdict  /  r79g,op,m,n,nb,pinned,stale_before_sync,err,verdict
#include "magma_v2.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
typedef std::complex<double> cd; typedef magmaDoubleComplex Z;
static unsigned Sd = 11u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static magma_queue_t q;
static Z zz(cd v) { return MAGMA_Z_MAKE(v.real(), v.imag()); } static cd cc(Z v) { return cd(MAGMA_Z_REAL(v), MAGMA_Z_IMAG(v)); }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 5) return 1; std::string op = argv[1]; magma_init(); magma_queue_create(0, &q);
  if (op == "trtri") {
    bool vb = argv[2][0] == 'v'; char ul = argv[3][0], dg = argv[4][0]; int nmax = atoi(argv[5]), B = atoi(argv[6]); const int NB = 128;
    std::vector<int> ns(B, nmax); if (vb) { for (int b = 0; b < B; b++) ns[b] = (int)((r1() + 0.5) * (nmax + 1)) % (nmax + 1); ns[0] = 0; if (B > 1) ns[1] = 1; ns[B - 1] = nmax; }
    std::vector<std::vector<cd>> A(B); std::vector<int> ld(B); std::vector<Z*> pa(B), pi(B); std::vector<size_t> isz(B);
    const double NaN = std::nan("");
    for (int b = 0; b < B; b++) { int n = ns[b]; ld[b] = std::max(n, 1) + 2; A[b].assign((size_t)ld[b] * std::max(n, 1), getenv("R79_PADZERO") ? cd(0) : cd(NaN, NaN));
      for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { bool st = ul == 'L' ? i >= j : i <= j; if (!st) continue; if (i == j) A[b][i + (size_t)j * ld[b]] = dg == 'U' ? cd(NaN, NaN) : cd(2 + r1(), r1()); else A[b][i + (size_t)j * ld[b]] = 0.3 * cd(r1(), r1()); }
      std::vector<Z> t(A[b].size()); for (size_t i = 0; i < t.size(); i++) t[i] = zz(A[b][i]); magma_zmalloc(&pa[b], t.size()); magma_zsetvector((magma_int_t)t.size(), t.data(), 1, pa[b], 1, q);
      isz[b] = (size_t)NB * std::max(magma_roundup(std::max(n, 1), NB), (magma_int_t)NB); std::vector<Z> s(isz[b], MAGMA_Z_MAKE(-9, 9)); magma_zmalloc(&pi[b], isz[b]); magma_zsetvector((magma_int_t)s.size(), s.data(), 1, pi[b], 1, q); }
    Z** dpa; Z** dpi; magma_malloc((void**)&dpa, B * sizeof(Z*)); magma_malloc((void**)&dpi, B * sizeof(Z*)); magma_setvector(B, sizeof(Z*), pa.data(), 1, dpa, 1, q); magma_setvector(B, sizeof(Z*), pi.data(), 1, dpi, 1, q);
    magma_uplo_t U = ul == 'U' ? MagmaUpper : MagmaLower; magma_diag_t D = dg == 'U' ? MagmaUnit : MagmaNonUnit;
    if (!vb) magmablas_ztrtri_diag_batched(U, D, nmax, (const Z* const*)dpa, ld[0], dpi, 1, B, q);
    else { std::vector<magma_int_t> hn(B + 1), hl(B + 1); for (int b = 0; b < B; b++) { hn[b] = ns[b]; hl[b] = ld[b]; } magma_int_t *dn, *dl; magma_imalloc(&dn, B + 1); magma_imalloc(&dl, B + 1);
      magma_isetvector(B + 1, hn.data(), 1, dn, 1, q); magma_isetvector(B + 1, hl.data(), 1, dl, 1, q); magmablas_ztrtri_diag_vbatched(U, D, nmax, dn, (const Z* const*)dpa, dl, dpi, 1, B, q); }
    magma_queue_sync(q);
    double emax = 0; long long outside = 0; int wrongm = 0;
    for (int b = 0; b < B; b++) { int n = ns[b]; std::vector<Z> t(isz[b]); magma_zgetvector((magma_int_t)t.size(), pi[b], 1, t.data(), 1, q); double e = 0; std::vector<char> used(isz[b], 0);
      for (int k = 0; k * NB < n; k++) { int r = std::min(NB, n - k * NB), o = k * NB;
        auto inv = [&](int i, int j) { size_t p = (size_t)i + (size_t)(o + j) * NB; used[p] = 1; return cc(t[p]); };
        auto a = [&](int i, int j) { bool st = ul == 'L' ? i >= j : i <= j; if (!st) return cd(0); if (i == j && dg == 'U') return cd(1); return A[b][(o + i) + (size_t)(o + j) * ld[b]]; };
        for (int j = 0; j < r; j++) for (int i = 0; i < r; i++) { cd s = 0; for (int l = 0; l < r; l++) s += inv(i, l) * a(l, j); double d = std::abs(s - (i == j ? cd(1) : cd(0))); if (!(d <= e)) e = d; } }
      size_t doc = n > 0 ? (size_t)NB * magma_roundup(n, NB) : 0; for (size_t p = 0; p < doc; p++) if (!used[p] && cc(t[p]) != cd(0)) { outside++; if (getenv("R79_DBG") && outside <= atoi(getenv("R79_DBG"))) fprintf(stderr, "outside mat %d n %d at (%zu,%zu): %g %g\n", b, n, p % NB, p / NB, cc(t[p]).real(), cc(t[p]).imag()); }   // documented size (NB, ceil(n/NB)*NB)
      if (!(e < 1e-12)) { wrongm++; if (wrongm <= 3) fprintf(stderr, "mat %d n %d: err %.3g\n", b, n, e); } if (!(e <= emax)) emax = e; }
    printf("r79t,%c,%c,%c,%d,%d,%.3g,%lld,%s\n", vb ? 'v' : 'b', ul, dg, nmax, B, emax, outside, wrongm == 0 && outside == 0 ? "OK" : "WRONG");
  } else {
    int m = atoi(argv[2]), n = atoi(argv[3]), nb = atoi(argv[4]); bool pin = op == "getT" && argc > 5 && atoi(argv[5]) == 1;
    int ldda = n + 2, lda = m + 1, lddw = m + 1; std::vector<cd> A((size_t)lda * n, 0.0); for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) A[i + (size_t)j * lda] = cd(r1(), r1());
    Z *dAT, *dw; magma_zmalloc(&dAT, (size_t)ldda * m); magma_zmalloc(&dw, (size_t)2 * lddw * nb + 1); magma_queue_t qs[2]; magma_queue_create(0, &qs[0]); magma_queue_create(0, &qs[1]);
    long long stale = -1; double err = 0;
    if (op == "getT") { std::vector<Z> T((size_t)ldda * m, MAGMA_Z_ZERO); for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) T[j + (size_t)i * ldda] = zz(A[i + (size_t)j * lda]);
      magma_zsetvector((magma_int_t)T.size(), T.data(), 1, dAT, 1, q); magma_queue_sync(q);
      Z* hA; if (pin) magma_zmalloc_pinned(&hA, (size_t)lda * n); else hA = (Z*)malloc((size_t)lda * n * sizeof(Z)); for (size_t i = 0; i < (size_t)lda * n; i++) hA[i] = MAGMA_Z_MAKE(-5, 5);
      magmablas_zgetmatrix_transpose(m, n, nb, dAT, ldda, hA, lda, dw, lddw, qs);
      stale = 0; for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) if (cc(hA[i + (size_t)j * lda]) == cd(-5, 5)) stale++;
      magma_queue_sync(qs[0]); magma_queue_sync(qs[1]);
      for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) err = std::max(err, std::abs(cc(hA[i + (size_t)j * lda]) - A[i + (size_t)j * lda]));
      for (int j = 0; j < n; j++) for (int i = m; i < lda; i++) if (cc(hA[i + (size_t)j * lda]) != cd(-5, 5)) err = INFINITY;   // padding row must stay
      if (pin) magma_free_pinned(hA); else free(hA); }
    else { std::vector<Z> h(A.size()); for (size_t i = 0; i < A.size(); i++) h[i] = zz(A[i]); std::vector<Z> T0((size_t)ldda * m, MAGMA_Z_MAKE(-5, 5)); magma_zsetvector((magma_int_t)T0.size(), T0.data(), 1, dAT, 1, q); magma_queue_sync(q);
      magmablas_zsetmatrix_transpose(m, n, nb, h.data(), lda, dAT, ldda, dw, lddw, qs); magma_queue_sync(qs[0]); magma_queue_sync(qs[1]);
      std::vector<Z> T((size_t)ldda * m); magma_zgetvector((magma_int_t)T.size(), dAT, 1, T.data(), 1, q);
      for (int i = 0; i < m; i++) { for (int j = 0; j < n; j++) err = std::max(err, std::abs(cc(T[j + (size_t)i * ldda]) - A[i + (size_t)j * lda])); for (int j = n; j < ldda; j++) if (cc(T[j + (size_t)i * ldda]) != cd(-5, 5)) err = INFINITY; } }
    printf("r79g,%s,%d,%d,%d,%d,%lld,%.3g,%s\n", op.c_str(), m, n, nb, pin ? 1 : 0, stale, err, err == 0 ? "OK" : "WRONG");
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

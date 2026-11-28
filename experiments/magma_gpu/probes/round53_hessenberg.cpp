// Round 53 (roxy): Hessenberg reduction with ilo > 1.  magma_zgehrd_m + magma_zunghr_m against magma_zgehrd + magma_zunghr and LAPACK.
//   round53_hessenberg <n> <ilo> <ihi> [seed=1]
// A is random, made upper triangular in rows and columns 1:ilo-1 and ihi+1:n (the LAPACK precondition, as left by zgebal).
// For each pair (m = multi-GPU interface, s = single-GPU interface):
//   H    max |H - H_lapack| / max |H_lapack| over i <= j+1, tau: max |tau - tau_lapack|
//   Q    the routine's unghr against LAPACK zunghr applied to the same reflectors (isolates the T hand-over)
//   res  max |Q H Q^H - A| / max |A| and max |Q^H Q - I|  (the documented contract A = Q H Q^H)
//   pair 2: magma_zgehrd_m + fixed_zunghr_m (proposed fix, below)
// CSV: r53,n,ilo,ihi,nb,pair,info,H,tau,Q,res,orth
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
typedef magmaDoubleComplex Z;
// Proposed fix for magma_zunghr_m: magma_zgehrd_m rounds ilo down to a block boundary (ilo_r) and stores the T of the block
// starting at global column i at T + i*nb, so hand ungqr_m the reflectors from ilo_r on (those in ilo_r .. ilo-2 have tau = 0)
// and T + ilo_r*nb.  When gehrd_m took the unblocked path (nb >= nh) nothing is rounded and T is unused.
static magma_int_t fixed_zunghr_m(magma_int_t n, magma_int_t ilo, magma_int_t ihi, Z* A, magma_int_t lda, Z* tau, Z* T, magma_int_t nb, magma_int_t* info) {
  #define A(i,j) (A + (i) + (j)*lda)
  *info = 0; magma_int_t nh = ihi - ilo, iinfo;
  for (magma_int_t j = ihi-1; j >= ilo; --j) { for (magma_int_t i = 0; i < j; ++i) *A(i,j) = MAGMA_Z_ZERO; for (magma_int_t i = j+1; i < ihi; ++i) *A(i,j) = *A(i,j-1); for (magma_int_t i = ihi; i < n; ++i) *A(i,j) = MAGMA_Z_ZERO; }
  for (magma_int_t j = 0; j < ilo; ++j) { for (magma_int_t i = 0; i < n; ++i) *A(i,j) = MAGMA_Z_ZERO; *A(j,j) = MAGMA_Z_ONE; }
  for (magma_int_t j = ihi; j < n; ++j) { for (magma_int_t i = 0; i < n; ++i) *A(i,j) = MAGMA_Z_ZERO; *A(j,j) = MAGMA_Z_ONE; }
  magma_int_t ilo0 = ilo - 1, ilo_r = (nb > 1 && nb < ihi - ilo0) ? (ilo0 / nb) * nb : ilo0, k = ihi - 1 - ilo_r;
  if (nh > 0) magma_zungqr_m(k, k, k, A(ilo_r+1, ilo_r+1), lda, tau + ilo_r, T + ilo_r*nb, nb, &iinfo);
  #undef A
  return *info; }
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static double amax(const std::vector<Z>& X) { double a = 0; for (auto& x : X) a = std::max(a, MAGMA_Z_ABS(x)); return a; }
static double dmax(const std::vector<Z>& X, const std::vector<Z>& Y) { double e = 0; for (size_t i = 0; i < X.size(); i++) { double d = MAGMA_Z_ABS(MAGMA_Z_SUB(X[i], Y[i])); if (!(d <= e)) e = d; } return e; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 4) { printf("usage: round53_hessenberg n ilo ihi [seed]\n"); return 1; }
  magma_int_t n = atoi(argv[1]), ilo = atoi(argv[2]), ihi = atoi(argv[3]); if (argc > 4) Sd = (unsigned)atoi(argv[4]);
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  magma_int_t nb = magma_get_zgehrd_nb(n), rinfo; size_t nn2 = (size_t)n * n;
  std::vector<Z> A0(nn2);
  for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i < n; i++) {
    bool zero = i > j && (j < ilo - 1 || i >= ihi);   // upper triangular in rows/cols 1:ilo-1 and ihi+1:n
    A0[i + j * n] = zero ? MAGMA_Z_ZERO : MAGMA_Z_MAKE(r1(), r1()); }
  // LAPACK reference
  std::vector<Z> R = A0, tr(n, MAGMA_Z_ZERO); Z wq; magma_int_t lw = -1;
  lapackf77_zgehrd(&n, &ilo, &ihi, R.data(), &n, tr.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq);
  { std::vector<Z> w(lw); lapackf77_zgehrd(&n, &ilo, &ihi, R.data(), &n, tr.data(), w.data(), &lw, &rinfo); }
  double an = amax(A0), hn = 0; for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i <= std::min(n - 1, j + 1); i++) hn = std::max(hn, MAGMA_Z_ABS(R[i + j * n]));
  for (int pair = 0; pair < 3; pair++) {
    std::vector<Z> A = A0, tau(n, MAGMA_Z_ZERO); magma_int_t info = 0, info2 = 0;
    std::vector<Z> Q;
    if (pair != 1) {
      std::vector<Z> T((size_t)nb * n); magma_int_t ml = -1; magma_zgehrd_m(n, ilo, ihi, A.data(), n, tau.data(), &wq, ml, T.data(), &info);
      ml = std::max<magma_int_t>((magma_int_t)MAGMA_Z_REAL(wq), 1); std::vector<Z> w(ml);
      magma_zgehrd_m(n, ilo, ihi, A.data(), n, tau.data(), w.data(), ml, T.data(), &info);
      Q = A; if (pair == 0) magma_zunghr_m(n, ilo, ihi, Q.data(), n, tau.data(), T.data(), nb, &info2); else fixed_zunghr_m(n, ilo, ihi, Q.data(), n, tau.data(), T.data(), nb, &info2);
    } else {
      Z* dT; magma_zmalloc(&dT, (size_t)nb * n); std::vector<Z> w((size_t)n * nb + 1);
      magma_zgehrd(n, ilo, ihi, A.data(), n, tau.data(), w.data(), (magma_int_t)w.size(), dT, &info);
      Q = A; magma_zunghr(n, ilo, ihi, Q.data(), n, tau.data(), dT, nb, &info2); magma_free(dT);
    }
    // H and tau against LAPACK
    double eh = 0; for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i <= std::min(n - 1, j + 1); i++) eh = std::max(eh, MAGMA_Z_ABS(MAGMA_Z_SUB(A[i + j * n], R[i + j * n])));
    double et = 0; for (magma_int_t i = 0; i + 1 < n; i++) et = std::max(et, MAGMA_Z_ABS(MAGMA_Z_SUB(tau[i], tr[i])));
    // LAPACK zunghr on the routine's own reflectors
    std::vector<Z> QR = A; lw = -1; lapackf77_zunghr(&n, &ilo, &ihi, QR.data(), &n, tau.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq);
    { std::vector<Z> w(lw); lapackf77_zunghr(&n, &ilo, &ihi, QR.data(), &n, tau.data(), w.data(), &lw, &rinfo); }
    double eq = dmax(Q, QR);
    // contract: Q H Q^H = A0, Q^H Q = I
    std::vector<Z> H = A; for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = j + 2; i < n; i++) H[i + j * n] = MAGMA_Z_ZERO;
    std::vector<Z> QH(nn2), QHQ(nn2), QtQ(nn2), I(nn2, MAGMA_Z_ZERO); Z one = MAGMA_Z_ONE, zero = MAGMA_Z_ZERO;
    blasf77_zgemm("N", "N", &n, &n, &n, &one, Q.data(), &n, H.data(), &n, &zero, QH.data(), &n);
    blasf77_zgemm("N", "C", &n, &n, &n, &one, QH.data(), &n, Q.data(), &n, &zero, QHQ.data(), &n);
    blasf77_zgemm("C", "N", &n, &n, &n, &one, Q.data(), &n, Q.data(), &n, &zero, QtQ.data(), &n);
    for (magma_int_t i = 0; i < n; i++) I[i + i * n] = MAGMA_Z_ONE;
    printf("r53,%lld,%lld,%lld,%lld,%s,%lld/%lld,%.3g,%.3g,%.3g,%.3g,%.3g\n", (long long)n, (long long)ilo, (long long)ihi, (long long)nb, pair == 0 ? "gehrd_m+unghr_m" : pair == 1 ? "gehrd+unghr" : "gehrd_m+fixed_unghr_m",
           (long long)info, (long long)info2, eh / hn, et, eq, dmax(QHQ, A0) / an, dmax(QtQ, I));
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

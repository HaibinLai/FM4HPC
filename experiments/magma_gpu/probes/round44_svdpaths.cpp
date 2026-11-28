// Round 44 (roxy): every lwork-selected code path of magma_dgesvd and magma_dgesdd.  Like LAPACK they choose among
// several algorithms by the size of lwork (dgesvd.cpp has dozens of `if (lwork >= ...)` switches); earlier rounds used
// only the queried optimum, which exercises one path per job combination.  M249 (trevc3) was a bug that only the
// "large workspace" path had.  For each shape x job x lwork level: singular values vs LAPACK dgesdd,
// ||A - U S V^T|| / (||A|| max(m,n) eps), ||U^T U - I|| / (m eps), ||V V^T - I|| / (n eps).
// lwork levels: the smallest value the routine accepts (found by bisection on info == lwork-argument error), the queried
// optimum, and 6 geometric steps in between.
// CSV: r44,routine,shape,job,lwork,info,sv_err,resid,orthU,orthV,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
static magma_queue_t Q; static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
typedef std::vector<double> V; static const double EPS = 1.1102230246251565e-16;
static magma_vec_t jv(char c) { return c == 'N' ? MagmaNoVec : c == 'O' ? MagmaOverwriteVec : c == 'S' ? MagmaSomeVec : MagmaAllVec; }

// run one SVD; returns info and fills s, U (m x ucols), VT (vrows x n) as far as the job provides them
struct Out { magma_int_t info; V s, U, VT; int ucols, vrows; };
static Out svd(bool dd, char ju, char jvt, int m, int n, const V& A0, magma_int_t lwork) {
  int mn = std::min(m, n); Out o; o.s.assign(mn, 0); V A = A0; std::vector<magma_int_t> iw(8 * mn + 8);
  int ldu = m, ucolsA = m, ldvt = n;
  o.U.assign((size_t)m * m, 0); o.VT.assign((size_t)n * n, 0); V work(std::max<magma_int_t>(lwork, 1));
  if (dd) magma_dgesdd(jv(ju), m, n, A.data(), m, o.s.data(), o.U.data(), ldu, o.VT.data(), ldvt, work.data(), lwork, iw.data(), &o.info);
  else magma_dgesvd(jv(ju), jv(jvt), m, n, A.data(), m, o.s.data(), o.U.data(), ldu, o.VT.data(), ldvt, work.data(), lwork, &o.info);
  // where the vectors are: 'O' overwrites A (U when m >= n for gesdd O; for gesvd jobu=O -> U in A, jobvt=O -> VT in A)
  char eu = dd ? ju : ju, ev = dd ? ju : jvt;
  o.ucols = eu == 'A' ? m : (eu == 'S' || eu == 'O') ? mn : 0; o.vrows = ev == 'A' ? n : (ev == 'S' || ev == 'O') ? mn : 0;
  if (dd && ju == 'O') { if (m >= n) { o.U = V(A.begin(), A.begin() + (size_t)m * mn); o.ucols = mn; /* VT in VT */ o.vrows = n; } else { o.ucols = m; V vt((size_t)n * n, 0); for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) vt[i + (size_t)j * n] = A[i + (size_t)j * m]; o.VT = vt; o.vrows = mn; } }
  if (!dd && ju == 'O') { o.U = V(A.begin(), A.begin() + (size_t)m * mn); o.ucols = mn; }
  if (!dd && jvt == 'O') { V vt((size_t)n * n, 0); for (int j = 0; j < n; j++) for (int i = 0; i < mn; i++) vt[i + (size_t)j * n] = A[i + (size_t)j * m]; o.VT = vt; o.vrows = mn; }
  if (dd && ju == 'S') { o.ucols = mn; o.vrows = mn; } if (dd && ju == 'A') { o.ucols = m; o.vrows = n; } if (dd && ju == 'N') { o.ucols = 0; o.vrows = 0; }
  return o; }

static void check(bool dd, char ju, char jvt, int m, int n) {
  int mn = std::min(m, n); Sd = 7u + m + 3 * n; V A0((size_t)m * n); for (auto& a : A0) a = r1();
  V sref(mn); { V A = A0, U(1), VT(1); std::vector<magma_int_t> iw(8 * mn + 8); magma_int_t M = m, N = n, one = 1, info, lw = -1; double q;
    lapackf77_dgesdd("N", &M, &N, A.data(), &M, sref.data(), U.data(), &one, VT.data(), &one, &q, &lw, iw.data(), &info); lw = (magma_int_t)q + 1; V w(lw);
    lapackf77_dgesdd("N", &M, &N, A.data(), &M, sref.data(), U.data(), &one, VT.data(), &one, w.data(), &lw, iw.data(), &info); }
  // optimum from the query
  magma_int_t opt; { V A = A0, s(mn), U((size_t)m * m), VT((size_t)n * n); std::vector<magma_int_t> iw(8 * mn + 8); double q = 0; magma_int_t info;
    if (dd) magma_dgesdd(jv(ju), m, n, A.data(), m, s.data(), U.data(), m, VT.data(), n, &q, -1, iw.data(), &info); else magma_dgesvd(jv(ju), jv(jvt), m, n, A.data(), m, s.data(), U.data(), m, VT.data(), n, &q, -1, &info); opt = (magma_int_t)q; }
  // smallest accepted lwork: bisection on "info == -(lwork argument)" (gesvd: -13, gesdd: -12)
  magma_int_t argno = dd ? 12 : 13, lo = 1, hi = opt; auto rejected = [&](magma_int_t lw) { Out o = svd(dd, ju, jvt, m, n, A0, lw); return o.info == -argno; };
  if (!rejected(1)) hi = 1; else { while (lo + 1 < hi) { magma_int_t mid = (lo + hi) / 2; if (rejected(mid)) lo = mid; else hi = mid; } }
  magma_int_t minacc = hi; std::vector<magma_int_t> lws = {minacc}; for (int s = 1; s <= 6; s++) lws.push_back((magma_int_t)(minacc * pow((double)opt / minacc, s / 7.0))); lws.push_back(opt); lws.push_back(opt + 64);
  std::sort(lws.begin(), lws.end()); lws.erase(std::unique(lws.begin(), lws.end()), lws.end());
  double an = 0; for (double a : A0) an = std::max(an, fabs(a));
  for (magma_int_t lw : lws) {
    Out o = svd(dd, ju, jvt, m, n, A0, lw); double se = 0, res = 0, ou = 0, ov = 0;
    for (int i = 0; i < mn; i++) se = std::max(se, fabs(o.s[i] - sref[i]) / (sref[0] * mn * EPS * 10));
    if (o.ucols > 0) for (int j = 0; j < o.ucols; j++) for (int jj = 0; jj <= j; jj++) { double s = 0; for (int i = 0; i < m; i++) s += o.U[i + (size_t)j * m] * o.U[i + (size_t)jj * m]; ou = std::max(ou, fabs(s - (j == jj)) / (m * EPS * 10)); }
    if (o.vrows > 0) for (int i = 0; i < o.vrows; i++) for (int ii = 0; ii <= i; ii++) { double s = 0; for (int j = 0; j < n; j++) s += o.VT[i + (size_t)j * n] * o.VT[ii + (size_t)j * n]; ov = std::max(ov, fabs(s - (i == ii)) / (n * EPS * 10)); }
    if (o.ucols >= mn && o.vrows >= mn) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { double s = 0; for (int k = 0; k < mn; k++) s += o.U[i + (size_t)k * m] * o.s[k] * o.VT[k + (size_t)j * n]; res = std::max(res, fabs(s - A0[i + (size_t)j * m]) / (an * std::max(m, n) * EPS * 10)); }
    bool ok = o.info == 0 && se <= 100 && res <= 100 && ou <= 100 && ov <= 100;
    printf("r44,%s,%dx%d,%c%c,%lld,%lld,%.3g,%.3g,%.3g,%.3g,%s\n", dd ? "dgesdd" : "dgesvd", m, n, ju, dd ? '-' : jvt, (long long)lw, (long long)o.info, se, res, ou, ov, ok ? "OK" : "BAD");
  }
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); magma_init(); magma_queue_create(0, &Q); printf("r44,routine,shape,job,lwork,info,sv_err,resid,orthU,orthV,verdict\n");
  struct Sh { int m, n; }; std::vector<Sh> sh = {{300, 100}, {150, 100}, {100, 100}, {100, 150}, {100, 300}};
  std::string g = argc > 1 ? argv[1] : "all";
  for (auto s : sh) {
    if (g == "all" || g == "svd") for (char ju : {'N', 'O', 'S', 'A'}) for (char jvt : {'N', 'O', 'S', 'A'}) { if (ju == 'O' && jvt == 'O') continue; check(false, ju, jvt, s.m, s.n); }
    if (g == "all" || g == "sdd") for (char jz : {'N', 'O', 'S', 'A'}) check(true, jz, jz, s.m, s.n);
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

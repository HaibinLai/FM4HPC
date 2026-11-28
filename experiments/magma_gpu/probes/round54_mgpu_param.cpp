// Round 54 (roxy, library build, ngpu = 1): CPU-interface multi-GPU drivers under parameters round 38 did not use:
// lda > n (padding rows must stay untouched), z precision, Upper, itype 2/3, jobz N, range V, geev_m left vectors, m != n.
//   round54_mgpu_param <routine> <n> <pad> <uplo L|U> <itype 1..3> <jobz N|V> <range A|I|V> [m]
//   routine: heevd heevdx heevdx2s hegvd hegvdx hegvdx2s   (eigenvalues against LAPACK zheevd / zhegvd, vectors by residual)
//            geev (jobz = left vectors, range ignored; right vectors always) getrf potrf geqrf gehrd (against LAPACK, same lda)
// lda = n + pad (and ldb, ldvl, ldvr); pad rows hold a sentinel that must survive.
// CSV: r54,routine,n,pad,uplo,itype,jobz,range,m,info,mout/mref,ew,res,padbad,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
typedef magmaDoubleComplex Z; typedef std::complex<double> C;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static C c(Z z) { return C(MAGMA_Z_REAL(z), MAGMA_Z_IMAG(z)); }
static Z zz(C x) { return MAGMA_Z_MAKE(x.real(), x.imag()); }
static const Z SENT = MAGMA_Z_MAKE(12345.0, -6789.0);
static int ld;   // lda = ldb = n + pad
static std::vector<Z> padded(const std::vector<C>& M, int rows, int cols) { std::vector<Z> P((size_t)ld * cols, SENT);
  for (int j = 0; j < cols; j++) for (int i = 0; i < rows; i++) P[i + (size_t)j * ld] = zz(M[i + (size_t)j * rows]); return P; }
static int padbad(const std::vector<Z>& P, int rows, int cols) { int b = 0; for (int j = 0; j < cols; j++) for (int i = rows; i < ld; i++)
  if (MAGMA_Z_REAL(P[i + (size_t)j * ld]) != 12345.0 || MAGMA_Z_IMAG(P[i + (size_t)j * ld]) != -6789.0) b++; return b; }
static std::vector<C> herm(int n, double shift) { std::vector<C> A((size_t)n * n);
  for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { C v = i == j ? C(r1() + shift, 0) : C(r1(), r1()); A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = std::conj(v); } return A; }
static std::vector<C> mul(const std::vector<C>& A, const std::vector<C>& B, int n, int k, int m) { std::vector<C> R((size_t)n * m, 0.0);   // (n x k)(k x m)
  for (int j = 0; j < m; j++) for (int l = 0; l < k; l++) { C b = B[l + (size_t)j * k]; if (b == 0.0) continue; for (int i = 0; i < n; i++) R[i + (size_t)j * n] += A[i + (size_t)l * n] * b; } return R; }
static double nrm(const std::vector<C>& A) { double a = 0; for (auto& x : A) a = std::max(a, std::abs(x)); return a; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 8) { printf("usage: see source\n"); return 1; }
  std::string r = argv[1]; int n = atoi(argv[2]), pad = atoi(argv[3]); char up = argv[4][0]; int itype = atoi(argv[5]); char jz = argv[6][0], rg = argv[7][0];
  int m = argc > 8 ? atoi(argv[8]) : n; ld = std::max(n, m) + pad;
  magma_init();
  magma_uplo_t uplo = up == 'U' ? MagmaUpper : MagmaLower; magma_vec_t jobz = jz == 'V' ? MagmaVec : MagmaNoVec;
  magma_range_t range = rg == 'I' ? MagmaRangeI : rg == 'V' ? MagmaRangeV : MagmaRangeAll;
  magma_int_t nn = n, info = 0, mout = -1, mref = 0, rinfo = 0, LD = ld; double ew = 0, res = 0; int pb = 0;
  bool gen = r.rfind("hegv", 0) == 0;
  if (r.rfind("heev", 0) == 0 || gen) {
    std::vector<C> A0 = herm(n, 0), B0 = gen ? herm(n, n) : std::vector<C>();
    // reference eigenvalues (all), LAPACK
    std::vector<double> wr(n);
    { std::vector<Z> A = padded(A0, n, n), B = gen ? padded(B0, n, n) : std::vector<Z>(); magma_int_t lw = -1, lrw = -1, liw = -1, iq, it = itype; Z wq; double rq;
      if (gen) lapackf77_zhegvd(&it, "N", &up, &nn, A.data(), &LD, B.data(), &LD, wr.data(), &wq, &lw, &rq, &lrw, &iq, &liw, &rinfo);
      else     lapackf77_zheevd("N", &up, &nn, A.data(), &LD, wr.data(), &wq, &lw, &rq, &lrw, &iq, &liw, &rinfo);
      lw = (magma_int_t)MAGMA_Z_REAL(wq); lrw = (magma_int_t)rq; liw = iq; std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw);
      if (gen) lapackf77_zhegvd(&it, "N", &up, &nn, A.data(), &LD, B.data(), &LD, wr.data(), wk.data(), &lw, rw.data(), &lrw, iw.data(), &liw, &rinfo);
      else     lapackf77_zheevd("N", &up, &nn, A.data(), &LD, wr.data(), wk.data(), &lw, rw.data(), &lrw, iw.data(), &liw, &rinfo); }
    // selection: I -> il..iu = n/4+1 .. n/2 ; V -> the same eigenvalues by midpoints
    magma_int_t il = n / 4 + 1, iu = std::max<magma_int_t>(il, n / 2); double vl = 0, vu = 0; int s0 = 0, s1 = n;   // [s0, s1)
    if (range != MagmaRangeAll) { s0 = (int)il - 1; s1 = (int)iu;
      vl = s0 > 0 ? 0.5 * (wr[s0 - 1] + wr[s0]) : wr[0] - 1; vu = s1 < n ? 0.5 * (wr[s1 - 1] + wr[s1]) : wr[n - 1] + 1; }
    mref = s1 - s0;
    std::vector<Z> A = padded(A0, n, n), B = gen ? padded(B0, n, n) : std::vector<Z>(); std::vector<double> w(n);
    magma_int_t lw = -1, lrw = -1, liw = -1, iq = 0; Z wq; double rq = 0;
    auto call = [&](Z* wk, magma_int_t lw_, double* rw, magma_int_t lrw_, magma_int_t* iw, magma_int_t liw_) {
      if (r == "heevd")    { magma_zheevd_m(1, jobz, uplo, n, A.data(), ld, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info); mout = n; }
      if (r == "heevdx")   magma_zheevdx_m(1, jobz, range, uplo, n, A.data(), ld, vl, vu, il, iu, &mout, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info);
      if (r == "heevdx2s") magma_zheevdx_2stage_m(1, jobz, range, uplo, n, A.data(), ld, vl, vu, il, iu, &mout, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info);
      if (r == "hegvd")    { magma_zhegvd_m(1, itype, jobz, uplo, n, A.data(), ld, B.data(), ld, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info); mout = n; }
      if (r == "hegvdx")   magma_zhegvdx_m(1, itype, jobz, range, uplo, n, A.data(), ld, B.data(), ld, vl, vu, il, iu, &mout, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info);
      if (r == "hegvdx2s") magma_zhegvdx_2stage_m(1, itype, jobz, range, uplo, n, A.data(), ld, B.data(), ld, vl, vu, il, iu, &mout, w.data(), wk, lw_, rw, lrw_, iw, liw_, &info); };
    call(&wq, lw, &rq, lrw, &iq, liw);
    lw = std::max<magma_int_t>((magma_int_t)MAGMA_Z_REAL(wq), 1); lrw = std::max<magma_int_t>((magma_int_t)rq, 1); liw = std::max<magma_int_t>(iq, 1);
    std::vector<Z> wk(lw); std::vector<double> rw(lrw); std::vector<magma_int_t> iw(liw); mout = -1;
    call(wk.data(), lw, rw.data(), lrw, iw.data(), liw);
    double am = 0; for (auto x : wr) am = std::max(am, fabs(x));
    if (mout == mref) for (int j = 0; j < mout; j++) ew = std::max(ew, fabs(w[j] - wr[s0 + j]) / am); else ew = INFINITY;
    pb = padbad(A, n, n) + (gen ? padbad(B, n, n) : 0);
    if (jobz == MagmaVec && mout == mref && mout > 0) {
      std::vector<C> V((size_t)n * mout); for (int j = 0; j < mout; j++) for (int i = 0; i < n; i++) V[i + (size_t)j * n] = c(A[i + (size_t)j * ld]);
      std::vector<C> L, Rt;   // residual L - Rt with Rt = lambda * (B v | v)
      if (!gen) { L = mul(A0, V, n, n, mout); Rt = V; }
      else if (itype == 1) { L = mul(A0, V, n, n, mout); Rt = mul(B0, V, n, n, mout); }
      else if (itype == 2) { L = mul(A0, mul(B0, V, n, n, mout), n, n, mout); Rt = V; }
      else { L = mul(B0, mul(A0, V, n, n, mout), n, n, mout); Rt = V; }
      double sc = nrm(A0) * (gen ? nrm(B0) : 1.0) * n, vn = nrm(V);
      for (int j = 0; j < mout; j++) { double rj = 0; for (int i = 0; i < n; i++) rj = std::max(rj, std::abs(L[i + (size_t)j * n] - w[j] * Rt[i + (size_t)j * n]) / (sc * vn));
        res = std::max(res, rj); if (getenv("R54_DBG")) { double cn = 0; for (int i = 0; i < n; i++) cn = std::max(cn, std::abs(V[i + (size_t)j * n])); fprintf(stderr, "col %d w %.6g ref %.6g res %.3g |v| %.3g\n", j, w[j], wr[s0 + j], rj, cn); } }
    }
  }
  else if (r == "geev") {
    std::vector<C> A0((size_t)n * n); for (auto& a : A0) a = C(r1(), r1());
    std::vector<Z> A = padded(A0, n, n), VL((size_t)ld * n, SENT), VR((size_t)ld * n, SENT), w(n), wq(1); std::vector<double> rw(2 * n); magma_int_t lw = -1;
    magma_vec_t jl = jobz;   // jobz field = left vectors
    magma_zgeev_m(jl, MagmaVec, n, A.data(), ld, w.data(), VL.data(), ld, VR.data(), ld, wq.data(), lw, rw.data(), &info);
    lw = (magma_int_t)MAGMA_Z_REAL(wq[0]); std::vector<Z> wk(std::max<magma_int_t>(lw, 1));
    magma_zgeev_m(jl, MagmaVec, n, A.data(), ld, w.data(), VL.data(), ld, VR.data(), ld, wk.data(), lw, rw.data(), &info);
    mout = mref = n; double an = nrm(A0);
    for (int j = 0; j < n; j++) { double vn = 0, d = 0, vnl = 0, dl = 0; for (int i = 0; i < n; i++) { vn = std::max(vn, std::abs(c(VR[i + (size_t)j * ld]))); vnl = std::max(vnl, std::abs(c(VL[i + (size_t)j * ld]))); }
      for (int i = 0; i < n; i++) { C s = 0, sl = 0; for (int k = 0; k < n; k++) { s += A0[i + (size_t)k * n] * c(VR[k + (size_t)j * ld]); sl += std::conj(c(VL[k + (size_t)j * ld])) * A0[k + (size_t)i * n]; }
        d = std::max(d, std::abs(s - c(w[j]) * c(VR[i + (size_t)j * ld]))); dl = std::max(dl, std::abs(sl - c(w[j]) * std::conj(c(VL[i + (size_t)j * ld])))); }
      res = std::max(res, d / (an * vn)); if (jl == MagmaVec) res = std::max(res, dl / (an * vnl)); }
    pb = padbad(A, n, n) + padbad(VR, n, n) + (jl == MagmaVec ? padbad(VL, n, n) : 0);
  }
  else {   // getrf potrf geqrf gehrd: against LAPACK with the same lda
    std::vector<C> A0 = r == "potrf" ? herm(n, n) : std::vector<C>((size_t)m * n); if (r != "potrf") for (auto& a : A0) a = C(r1(), r1());
    int rows = r == "potrf" || r == "gehrd" ? n : m; magma_int_t mm = rows;
    std::vector<Z> A = padded(A0, rows, n), R = A; std::vector<magma_int_t> ip(std::min(rows, n)), ipr(std::min(rows, n)); std::vector<Z> tau(n), tr(n);
    int part = 0;   // 0 all, 1 lower, 2 upper, 3 hessenberg
    if (r == "getrf") { lapackf77_zgetrf(&mm, &nn, R.data(), &LD, ipr.data(), &rinfo); magma_zgetrf_m(1, rows, n, A.data(), ld, ip.data(), &info);
      for (size_t i = 0; i < ip.size(); i++) if (ip[i] != ipr[i]) ew = INFINITY; }
    if (r == "potrf") { lapackf77_zpotrf(&up, &nn, R.data(), &LD, &rinfo); magma_zpotrf_m(1, uplo, n, A.data(), ld, &info); part = up == 'U' ? 2 : 1; }
    if (r == "geqrf") { magma_int_t lw = -1; Z wq; lapackf77_zgeqrf(&mm, &nn, R.data(), &LD, tr.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); { std::vector<Z> wk(lw); lapackf77_zgeqrf(&mm, &nn, R.data(), &LD, tr.data(), wk.data(), &lw, &rinfo); }
      std::vector<Z> wk(std::max<magma_int_t>(lw, (magma_int_t)n * 64)); magma_zgeqrf_m(1, rows, n, A.data(), ld, tau.data(), wk.data(), (magma_int_t)wk.size(), &info); part = 2;
      for (int i = 0; i < std::min(rows, n); i++) ew = std::max(ew, std::abs(c(tau[i]) - c(tr[i]))); }
    if (r == "gehrd") { magma_int_t lw = -1, one = 1; Z wq; lapackf77_zgehrd(&nn, &one, &nn, R.data(), &LD, tr.data(), &wq, &lw, &rinfo); lw = (magma_int_t)MAGMA_Z_REAL(wq); { std::vector<Z> wk(lw); lapackf77_zgehrd(&nn, &one, &nn, R.data(), &LD, tr.data(), wk.data(), &lw, &rinfo); }
      int nb = (int)magma_get_zgehrd_nb(n); std::vector<Z> T((size_t)nb * n); magma_int_t ml = -1; magma_zgehrd_m(n, 1, n, A.data(), ld, tau.data(), &wq, ml, T.data(), &info);
      ml = std::max<magma_int_t>((magma_int_t)MAGMA_Z_REAL(wq), 1); std::vector<Z> wk(ml); magma_zgehrd_m(n, 1, n, A.data(), ld, tau.data(), wk.data(), ml, T.data(), &info); part = 3;
      for (int i = 0; i + 1 < n; i++) ew = std::max(ew, std::abs(c(tau[i]) - c(tr[i]))); }
    double e = 0, a = 0; for (int j = 0; j < n; j++) for (int i = 0; i < rows; i++) {
      if ((part == 1 && i < j) || (part == 2 && i > j) || (part == 3 && i > j + 1)) continue;
      e = std::max(e, std::abs(c(A[i + (size_t)j * ld]) - c(R[i + (size_t)j * ld]))); a = std::max(a, std::abs(c(R[i + (size_t)j * ld]))); }
    res = e / a; pb = padbad(A, rows, n); mout = mref = 0;
  }
  bool ok = info == 0 && rinfo == 0 && mout == mref && ew < 1e-10 && res < 1e-10 && pb == 0;
  printf("r54,%s,%d,%d,%c,%d,%c,%c,%d,%lld,%lld/%lld,%.3g,%.3g,%d,%s\n", r.c_str(), n, pad, up, itype, jz, rg, m, (long long)(info ? info : rinfo), (long long)mout, (long long)mref, ew, res, pb, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_finalize(); return 0;
}

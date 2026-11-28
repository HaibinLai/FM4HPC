// Round 78 (roxy): the variable-size batched BLAS (magmablas_z*_vbatched), never tested before, against CPU references
// written from the BLAS definitions.  Each batch has sizes drawn from [0, maxsz] (a 0 and maxsz itself always occur), ld = rows + 3,
// complex data; the ld padding and (for Hermitian/symmetric/triangular outputs) the triangle that must not be written hold sentinels.
//   round78_vbatched_blas gemm  <tA N|T|C> <tB N|T|C> <maxsz> <batch>
//   round78_vbatched_blas gemv  <t N|T|C>  <inc 1|2>  <maxsz> <batch>
//   round78_vbatched_blas hemm  <side L|R> <uplo U|L> <maxsz> <batch>
//   round78_vbatched_blas hemv  <uplo U|L> <inc 1|2>  <maxsz> <batch>
//   round78_vbatched_blas herk|syrk|her2k|syr2k <uplo U|L> <trans N|C (her) / N|T (syr)> <maxsz> <batch>
//   round78_vbatched_blas trmm|trsm <side L|R><uplo U|L> <trans N|T|C><diag N|U> <maxsz> <batch>   (e.g. "LU" "CN")
// CSV: r78,op,o1,o2,maxsz,batch,max_rel_err,wrong_mats,padding_sentinels_touched,other_triangle_written,verdict
//      (verdict OK_M246: herk/syrk/her2k/syr2k wrote the other triangle, the known M246; everything else correct)
#include "magma_v2.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
typedef std::complex<double> cd; typedef magmaDoubleComplex Z;
static unsigned Sd = 7u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static magma_queue_t q;
static const cd SENT(-777, 555);
static Z zz(cd v) { return MAGMA_Z_MAKE(v.real(), v.imag()); } static cd cc(Z v) { return cd(MAGMA_Z_REAL(v), MAGMA_Z_IMAG(v)); }
struct Mat { int r, c, ld; std::vector<cd> h; Z* d = NULL; cd& at(int i, int j) { return h[i + (size_t)j * ld]; } };
static Mat mk(int r, int c, bool fill = true) { Mat M; M.r = r; M.c = c; M.ld = r + 3; M.h.assign((size_t)M.ld * std::max(c, 1), SENT); if (fill) for (int j = 0; j < c; j++) for (int i = 0; i < r; i++) M.at(i, j) = cd(r1(), r1()); return M; }
static void up(Mat& M) { std::vector<Z> t(M.h.size()); for (size_t i = 0; i < t.size(); i++) t[i] = zz(M.h[i]); magma_zmalloc(&M.d, t.size()); magma_zsetvector((magma_int_t)t.size(), t.data(), 1, M.d, 1, q); }
static std::vector<cd> dn(Mat& M) { std::vector<Z> t(M.h.size()); magma_zgetvector((magma_int_t)t.size(), M.d, 1, t.data(), 1, q); std::vector<cd> h(t.size()); for (size_t i = 0; i < t.size(); i++) h[i] = cc(t[i]); return h; }
template <class T> static T* dev(const std::vector<T>& v) { T* d; magma_malloc((void**)&d, std::max<size_t>(v.size(), 1) * sizeof(T)); magma_setvector((magma_int_t)v.size(), sizeof(T), v.data(), 1, d, 1, q); return d; }
static cd opv(char t, Mat& A, int i, int j) { return t == 'N' ? A.at(i, j) : t == 'T' ? A.at(j, i) : std::conj(A.at(j, i)); }
static magma_trans_t TR(char t) { return t == 'N' ? MagmaNoTrans : t == 'T' ? MagmaTrans : MagmaConjTrans; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 6) return 1; std::string op = argv[1], o1 = argv[2], o2 = argv[3]; int mx = atoi(argv[4]), B = atoi(argv[5]);
  magma_init(); magma_queue_create(0, &q);
  std::vector<int> s1(B), s2(B), s3(B); for (int b = 0; b < B; b++) { s1[b] = (int)((r1() + 0.5) * (mx + 1)); s2[b] = (int)((r1() + 0.5) * (mx + 1)); s3[b] = (int)((r1() + 0.5) * (mx + 1)); }
  s1[0] = 0; s1[B - 1] = mx; s2[1 % B] = 0; s2[B - 1] = mx; s3[2 % B] = 0; s3[B - 1] = mx;
  for (int b = 0; b < B; b++) { s1[b] = std::min(s1[b], mx); s2[b] = std::min(s2[b], mx); s3[b] = std::min(s3[b], mx); }
  std::vector<Mat> A(B), Bm(B), Cm(B); std::vector<std::vector<cd>> R(B); std::vector<char> outTri(B, 0);   // R: expected full output buffer incl. sentinels
  cd al(0.7, -0.3), be(-0.4, 0.2); double alr = 0.7, ber = -0.4;
  std::vector<magma_int_t> M1(B + 1), N1(B + 1), K1(B + 1), LA(B + 1), LB(B + 1), LC(B + 1), IX(B + 1), IY(B + 1);
  char uplo = 'L', side = 'L', tr = 'N', dg = 'N';
  for (int b = 0; b < B; b++) {
    if (op == "gemm") { char ta = o1[0], tb = o2[0]; int m = s1[b], n = s2[b], k = s3[b]; A[b] = ta == 'N' ? mk(m, k) : mk(k, m); Bm[b] = tb == 'N' ? mk(k, n) : mk(n, k); Cm[b] = mk(m, n);
      R[b] = Cm[b].h; for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { cd s = 0; for (int l = 0; l < k; l++) s += opv(ta, A[b], i, l) * opv(tb, Bm[b], l, j); R[b][i + (size_t)j * Cm[b].ld] = al * s + be * Cm[b].at(i, j); }
      M1[b] = m; N1[b] = n; K1[b] = k; }
    else if (op == "gemv") { char t = o1[0]; int inc = atoi(o2.c_str()); int m = s1[b], n = s2[b]; A[b] = mk(m, n); int lx = t == 'N' ? n : m, ly = t == 'N' ? m : n;
      Bm[b] = mk(std::max(lx * inc, 0), 1); Cm[b] = mk(std::max(ly * inc, 0), 1); for (int i = 0; i < Bm[b].r; i++) if (i % inc) Bm[b].at(i, 0) = SENT; for (int i = 0; i < Cm[b].r; i++) if (i % inc) Cm[b].at(i, 0) = SENT;
      R[b] = Cm[b].h; if (m > 0 && n > 0) for (int i = 0; i < ly; i++) {   // reference BLAS: quick return (y untouched) when m = 0 or n = 0
        cd s = 0; for (int l = 0; l < lx; l++) s += (t == 'N' ? A[b].at(i, l) : t == 'T' ? A[b].at(l, i) : std::conj(A[b].at(l, i))) * Bm[b].at(l * inc, 0); R[b][(size_t)i * inc] = al * s + be * Cm[b].at(i * inc, 0); }
      M1[b] = m; N1[b] = n; IX[b] = inc; IY[b] = inc; }
    else if (op == "hemv") { uplo = o1[0]; int inc = atoi(o2.c_str()); int n = s1[b]; A[b] = mk(n, n); for (int i = 0; i < n; i++) A[b].at(i, i) = cd(A[b].at(i, i).real(), 0);
      Mat F = A[b]; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { bool st = uplo == 'L' ? i >= j : i <= j; if (!st) { F.at(i, j) = std::conj(A[b].at(j, i)); A[b].at(i, j) = SENT; } }
      Bm[b] = mk(n * inc, 1); Cm[b] = mk(n * inc, 1); for (int i = 0; i < n * inc; i++) if (i % inc) { Bm[b].at(i, 0) = SENT; Cm[b].at(i, 0) = SENT; }
      R[b] = Cm[b].h; for (int i = 0; i < n; i++) { cd s = 0; for (int l = 0; l < n; l++) s += F.at(i, l) * Bm[b].at(l * inc, 0); R[b][(size_t)i * inc] = al * s + be * Cm[b].at(i * inc, 0); }
      N1[b] = n; IX[b] = inc; IY[b] = inc; }
    else if (op == "hemm") { side = o1[0]; uplo = o2[0]; int m = s1[b], n = s2[b], na = side == 'L' ? m : n; A[b] = mk(na, na); for (int i = 0; i < na; i++) A[b].at(i, i) = cd(A[b].at(i, i).real(), 0);
      Mat F = A[b]; for (int j = 0; j < na; j++) for (int i = 0; i < na; i++) { bool st = uplo == 'L' ? i >= j : i <= j; if (!st) { F.at(i, j) = std::conj(A[b].at(j, i)); A[b].at(i, j) = SENT; } }
      Bm[b] = mk(m, n); Cm[b] = mk(m, n); R[b] = Cm[b].h;
      for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { cd s = 0; if (side == 'L') for (int l = 0; l < m; l++) s += F.at(i, l) * Bm[b].at(l, j); else for (int l = 0; l < n; l++) s += Bm[b].at(i, l) * F.at(l, j); R[b][i + (size_t)j * Cm[b].ld] = al * s + be * Cm[b].at(i, j); }
      M1[b] = m; N1[b] = n; }
    else if (op == "herk" || op == "syrk" || op == "her2k" || op == "syr2k") { uplo = o1[0]; tr = o2[0]; bool he = op[1] == 'e', two = op.find('2') != std::string::npos; int n = s1[b], k = s3[b];
      A[b] = tr == 'N' ? mk(n, k) : mk(k, n); if (two) Bm[b] = tr == 'N' ? mk(n, k) : mk(k, n); Cm[b] = mk(n, n); if (he) for (int i = 0; i < n; i++) Cm[b].at(i, i) = cd(Cm[b].at(i, i).real(), 0);
      for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { bool st = uplo == 'L' ? i >= j : i <= j; if (!st) Cm[b].at(i, j) = SENT; }
      R[b] = Cm[b].h; auto a = [&](Mat& X, int i, int l) { return tr == 'N' ? X.at(i, l) : X.at(l, i); };   // op(X)(i, l), op = N or the (conj-)transpose
      for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { bool st = uplo == 'L' ? i >= j : i <= j; if (!st) continue; cd s = 0;
        for (int l = 0; l < k; l++) { cd ai = a(A[b], i, l), aj = a(A[b], j, l); if (he && tr != 'N') { ai = std::conj(ai); aj = std::conj(aj); }
          if (!two) s += he ? (tr == 'N' ? ai * std::conj(aj) : ai * std::conj(aj)) : ai * aj;
          else { cd bi = a(Bm[b], i, l), bj = a(Bm[b], j, l); if (he && tr != 'N') { bi = std::conj(bi); bj = std::conj(bj); }
            s += he ? al * ai * std::conj(bj) + std::conj(al) * bi * std::conj(aj) : al * (ai * bj + bi * aj); } }
        cd v = two ? s : (he ? alr * s : al * s); v += (he ? cd(ber) : be) * Cm[b].at(i, j); if (he && i == j) v = cd(v.real(), 0); R[b][i + (size_t)j * Cm[b].ld] = v; }
      N1[b] = n; K1[b] = k; }
    else { side = o1[0]; uplo = o1[1]; tr = o2[0]; dg = o2[1]; bool sm = op == "trsm"; int m = s1[b], n = s2[b], na = side == 'L' ? m : n; A[b] = mk(na, na);
      for (int j = 0; j < na; j++) for (int i = 0; i < na; i++) { bool st = uplo == 'L' ? i >= j : i <= j; if (!st) A[b].at(i, j) = SENT; else if (i == j) A[b].at(i, j) += cd(sm ? 3 : 0, 0); else if (sm) A[b].at(i, j) *= 0.3; }
      Mat T = mk(na, na, false); for (int j = 0; j < na; j++) for (int i = 0; i < na; i++) { bool st = uplo == 'L' ? i >= j : i <= j; T.at(i, j) = st ? (i == j && dg == 'U' ? cd(1) : A[b].at(i, j)) : cd(0); }
      Bm[b] = mk(m, n); R[b] = Bm[b].h;
      // trmm: B := alpha op(T) B (L) or alpha B op(T) (R);  trsm: solve op(T) X = alpha B (L) or X op(T) = alpha B (R); reference for trsm checks the residual instead
      if (!sm) for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { cd s = 0; if (side == 'L') for (int l = 0; l < m; l++) s += opv(tr, T, i, l) * Bm[b].at(l, j); else for (int l = 0; l < n; l++) s += Bm[b].at(i, l) * opv(tr, T, l, j); R[b][i + (size_t)j * Bm[b].ld] = al * s; }
      Cm[b] = T; M1[b] = m; N1[b] = n; }
    LA[b] = A[b].ld; LB[b] = Bm[b].r ? Bm[b].ld : 1; LC[b] = Cm[b].r ? Cm[b].ld : 1;
  }
  std::vector<Z*> pa(B), pb(B), pc(B); for (int b = 0; b < B; b++) { up(A[b]); pa[b] = A[b].d; if (!Bm[b].h.empty()) { up(Bm[b]); pb[b] = Bm[b].d; } if (op != "trmm" && op != "trsm") { up(Cm[b]); pc[b] = Cm[b].d; } }
  Z** dpa = dev(pa); Z** dpb = dev(pb); Z** dpc = dev(pc);
  magma_int_t *dM = dev(M1), *dN = dev(N1), *dK = dev(K1), *dLA = dev(LA), *dLB = dev(LB), *dLC = dev(LC), *dIX = dev(IX), *dIY = dev(IY);
  // the ld of a gemv/hemv vector is irrelevant; LB / LC above are only for the matrix ops
  magma_uplo_t U = uplo == 'U' ? MagmaUpper : MagmaLower; magma_side_t S = side == 'L' ? MagmaLeft : MagmaRight;
  if (op == "gemm") magmablas_zgemm_vbatched(TR(o1[0]), TR(o2[0]), dM, dN, dK, zz(al), (const Z* const*)dpa, dLA, (const Z* const*)dpb, dLB, zz(be), dpc, dLC, B, q);
  else if (op == "gemv") magmablas_zgemv_vbatched(TR(o1[0]), dM, dN, zz(al), dpa, dLA, dpb, dIX, zz(be), dpc, dIY, B, q);
  else if (op == "hemv") magmablas_zhemv_vbatched(U, dN, zz(al), dpa, dLA, dpb, dIX, zz(be), dpc, dIY, B, q);
  else if (op == "hemm") magmablas_zhemm_vbatched(S, U, dM, dN, zz(al), dpa, dLA, dpb, dLB, zz(be), dpc, dLC, B, q);
  else if (op == "herk") magmablas_zherk_vbatched(U, TR(tr), dN, dK, alr, (const Z* const*)dpa, dLA, ber, dpc, dLC, B, q);
  else if (op == "syrk") magmablas_zsyrk_vbatched(U, TR(tr), dN, dK, zz(al), (const Z* const*)dpa, dLA, zz(be), dpc, dLC, B, q);
  else if (op == "her2k") magmablas_zher2k_vbatched(U, TR(tr), dN, dK, zz(al), (const Z* const*)dpa, dLA, (const Z* const*)dpb, dLB, ber, dpc, dLC, B, q);
  else if (op == "syr2k") magmablas_zsyr2k_vbatched(U, TR(tr), dN, dK, zz(al), (const Z* const*)dpa, dLA, (const Z* const*)dpb, dLB, zz(be), dpc, dLC, B, q);
  else if (op == "trmm") magmablas_ztrmm_vbatched(S, U, TR(tr), dg == 'U' ? MagmaUnit : MagmaNonUnit, dM, dN, zz(al), dpa, dLA, dpb, dLB, B, q);
  else magmablas_ztrsm_vbatched(S, U, TR(tr), dg == 'U' ? MagmaUnit : MagmaNonUnit, dM, dN, zz(al), dpa, dLA, dpb, dLB, B, q);
  magma_queue_sync(q);
  double emax = 0; int wrong = 0; long long sent = 0, otri = 0;   // otri: writes inside the m x n block that should stay untouched (other triangle, M246)
  for (int b = 0; b < B; b++) {
    bool tri = op == "trmm" || op == "trsm"; Mat& O = tri ? Bm[b] : Cm[b]; if (O.h.empty()) continue; std::vector<cd> X = dn(O); double e = 0, a = 0;
    if (op == "trsm") { int m = O.r, n = O.c; Mat& T = Cm[b];   // residual op(T) X - alpha B (L) or X op(T) - alpha B (R)
      for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { cd s = 0; if (side == 'L') for (int l = 0; l < m; l++) s += opv(tr, T, i, l) * X[l + (size_t)j * O.ld]; else for (int l = 0; l < n; l++) s += X[i + (size_t)l * O.ld] * opv(tr, T, l, j);
        cd ref = al * O.at(i, j); e = std::max(e, std::abs(s - ref)); a = std::max(a, std::abs(ref)); }
      for (int j = 0; j < std::max(n, 1); j++) for (int i = m; i < O.ld; i++) if (X[i + (size_t)j * O.ld] != SENT) sent++; }
    else for (size_t i = 0; i < X.size(); i++) { if (R[b][i] == SENT) { if (X[i] != SENT && (int)(i % O.ld) < O.r && (int)(i / O.ld) < O.c) { otri++; continue; } if (X[i] != SENT) { sent++; if (getenv("R78_DBG") && sent <= 6) fprintf(stderr, "sentinel mat %d (n %d k %d ld %d) idx %zu = (%d,%zu): %g %g\n", b, s1[b], s3[b], O.ld, i, (int)(i % O.ld), i / O.ld, X[i].real(), X[i].imag()); } continue; } e = std::max(e, std::abs(X[i] - R[b][i])); a = std::max(a, std::abs(R[b][i])); }
    double rel = a > 0 ? e / a : e; if (!(rel < 1e-12)) { wrong++; if (wrong <= 3) fprintf(stderr, "mat %d (sizes %d %d %d): rel %.3g\n", b, s1[b], s2[b], s3[b], rel); } if (!(rel <= emax)) emax = rel; }
  printf("r78,%s,%s,%s,%d,%d,%.3g,%d,%lld,%lld,%s\n", op.c_str(), o1.c_str(), o2.c_str(), mx, B, emax, wrong, sent, otri, wrong == 0 && sent == 0 && (otri == 0 || (op.find("rk") != std::string::npos || op.find("r2k") != std::string::npos)) ? (otri ? "OK_M246" : "OK") : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

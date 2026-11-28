// Round 61 (roxy): variable-size batched factorizations against LAPACK, matrix by matrix.
//   round61_vbatched <op> <set> <pad> <sing>
//   op:  getrf (magma_zgetrf_vbatched)  getrfmax (magma_zgetrf_vbatched_max_nocheck_work, workspace by query)
//        nopiv (magma_zgetrf_nopiv_vbatched, diagonally dominant input)  potrf (magma_zpotrf_vbatched, Lower, HPD)
//   set: 0 square {1, 7, 32, 33, 100, 257}; 1 rectangular {50x10, 10x50, 300x64, 64x300, 0x5, 5x0, 1x1};
//        2 large {512x512, 1000x200, 200x1000}; 3 mixed {257x257, 1x1, 0x0, 128x3, 3x128, 64x64}
//   pad: ldda_i = m_i + pad.  sing: if 1, column min(m,n)/2 of every second matrix is zero (getrf: info = col + 1 expected;
//        potrf: that diagonal set to -1, info expected at that position).
//   Per matrix: info against LAPACK; when both are 0, LU (or L) relative difference, and ipiv equal.
// CSV: r61,op,set,pad,sing,batch,bad_info,bad_ipiv,maxerr,verdict  (+ one detail line per bad matrix on stderr)
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
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 5) { printf("usage: see source\n"); return 1; }
  std::string op = argv[1]; int set = atoi(argv[2]), pad = atoi(argv[3]), sing = atoi(argv[4]);
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  std::vector<std::pair<int,int>> S;
  if (set == 0) for (int s : {1, 7, 32, 33, 100, 257}) S.push_back({s, s});
  if (set == 1) S = {{50,10},{10,50},{300,64},{64,300},{0,5},{5,0},{1,1}};
  if (set == 2) S = {{512,512},{1000,200},{200,1000}};
  if (set == 3) S = {{257,257},{1,1},{0,0},{128,3},{3,128},{64,64}};
  bool chol = op == "potrf", nop = op == "nopiv";
  if (chol) for (auto& s : S) s.second = s.first;   // square
  int bc = (int)S.size();
  std::vector<magma_int_t> hM(bc), hN(bc), hL(bc), hmn(bc), hinfo(bc);
  std::vector<std::vector<Z>> A(bc); std::vector<Z*> hdA(bc); std::vector<magma_int_t*> hdp(bc);
  magma_int_t maxm = 0, maxn = 0, maxmn = 0, maxmxn = 0;
  for (int b = 0; b < bc; b++) {
    magma_int_t m = S[b].first, n = S[b].second; hM[b] = m; hN[b] = n; hL[b] = std::max<magma_int_t>(1, m + pad); hmn[b] = std::min(m, n);
    maxm = std::max(maxm, m); maxn = std::max(maxn, n); maxmn = std::max(maxmn, hmn[b]); maxmxn = std::max(maxmxn, m * n);
    A[b].assign((size_t)hL[b] * std::max<magma_int_t>(n, 1), MAGMA_Z_ZERO);
    for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i < m; i++) A[b][i + j * hL[b]] = MAGMA_Z_MAKE(r1(), r1());
    if (chol) { for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i <= j; i++) { Z v = i == j ? MAGMA_Z_MAKE(n + r1(), 0) : MAGMA_Z_MAKE(r1(), r1()); A[b][i + j * hL[b]] = v; A[b][j + i * hL[b]] = MAGMA_Z_CONJ(v); } }
    if (nop) for (magma_int_t i = 0; i < hmn[b]; i++) A[b][i + i * hL[b]] = MAGMA_Z_ADD(A[b][i + i * hL[b]], MAGMA_Z_MAKE(std::max(m, n), 0));
    if (sing && (b % 2 == 1) && hmn[b] > 0) { magma_int_t c = hmn[b] / 2;
      if (chol) A[b][c + c * hL[b]] = MAGMA_Z_MAKE(-1, 0);
      else for (magma_int_t i = 0; i < m; i++) A[b][i + c * hL[b]] = MAGMA_Z_ZERO; }
    magma_zmalloc(&hdA[b], (size_t)hL[b] * std::max<magma_int_t>(n, 1)); magma_zsetmatrix(hL[b], std::max<magma_int_t>(n, 1), A[b].data(), hL[b], hdA[b], hL[b], q);
    magma_imalloc(&hdp[b], std::max<magma_int_t>(hmn[b], 1));
  }
  magma_int_t *dM, *dN, *dL, *dmn, *dinfo; magma_imalloc(&dM, bc + 1); magma_imalloc(&dN, bc + 1); magma_imalloc(&dL, bc + 1); magma_imalloc(&dmn, bc + 1); magma_imalloc(&dinfo, bc);
  magma_isetvector(bc, hM.data(), 1, dM, 1, q); magma_isetvector(bc, hN.data(), 1, dN, 1, q); magma_isetvector(bc, hL.data(), 1, dL, 1, q); magma_isetvector(bc, hmn.data(), 1, dmn, 1, q);
  { std::vector<magma_int_t> z(bc, 0); magma_isetvector(bc, z.data(), 1, dinfo, 1, q); }
  Z** dAa; magma_int_t** dpa; magma_malloc((void**)&dAa, bc * sizeof(Z*)); magma_malloc((void**)&dpa, bc * sizeof(magma_int_t*));
  magma_setvector(bc, sizeof(Z*), hdA.data(), 1, dAa, 1, q); magma_setvector(bc, sizeof(magma_int_t*), hdp.data(), 1, dpa, 1, q);
  magma_int_t ret = 0; void* work = NULL;
  if (op == "getrf") ret = magma_zgetrf_vbatched(dM, dN, dAa, dL, dpa, dinfo, bc, q);
  else if (op == "getrfmax") { magma_int_t lw = -1; magma_zgetrf_vbatched_max_nocheck_work(dM, dN, maxm, maxn, maxmn, maxmxn, NULL, dL, NULL, NULL, NULL, &lw, bc, q);
    magma_malloc(&work, std::max<magma_int_t>(lw, 1)); ret = magma_zgetrf_vbatched_max_nocheck_work(dM, dN, maxm, maxn, maxmn, maxmxn, dAa, dL, dpa, dinfo, work, &lw, bc, q); }
  else if (op == "nopiv") ret = magma_zgetrf_nopiv_vbatched(dM, dN, dAa, dL, dinfo, bc, q);
  else if (op == "potrf") ret = magma_zpotrf_vbatched(MagmaLower, dN, dAa, dL, dinfo, bc, q);
  magma_queue_sync(q); magma_igetvector(bc, dinfo, 1, hinfo.data(), 1, q);
  int bad_info = 0, bad_ipiv = 0; double maxerr = 0;
  for (int b = 0; b < bc; b++) {
    magma_int_t m = hM[b], n = hN[b], ld = hL[b], rinfo = 0; std::vector<Z> X((size_t)ld * std::max<magma_int_t>(n, 1)), R = A[b]; std::vector<magma_int_t> ip(std::max<magma_int_t>(hmn[b], 1)), rip(ip.size());
    magma_zgetmatrix(ld, std::max<magma_int_t>(n, 1), hdA[b], ld, X.data(), ld, q); if (!chol && !nop) magma_igetvector(std::max<magma_int_t>(hmn[b], 1), hdp[b], 1, ip.data(), 1, q);
    if (chol) { if (n > 0) lapackf77_zpotrf("L", &n, R.data(), &ld, &rinfo); }
    else if (m > 0 && n > 0) lapackf77_zgetrf(&m, &n, R.data(), &ld, rip.data(), &rinfo);
    double e = 0, a = 0;
    if (hinfo[b] != rinfo) { bad_info++; fprintf(stderr, "matrix %d (%lldx%lld): info %lld, LAPACK %lld\n", b, (long long)m, (long long)n, (long long)hinfo[b], (long long)rinfo); }
    else if (rinfo == 0) {
      for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i < m; i++) { if (chol && i < j) continue; double d = MAGMA_Z_ABS(MAGMA_Z_SUB(X[i + j * ld], R[i + j * ld])); if (!(d <= e)) e = d; a = std::max(a, MAGMA_Z_ABS(R[i + j * ld])); }
      if (a > 0) e /= a; maxerr = std::max(maxerr, e);
      if (!chol && !nop) for (magma_int_t i = 0; i < hmn[b]; i++) if (ip[i] != rip[i]) { bad_ipiv++; fprintf(stderr, "matrix %d (%lldx%lld): ipiv[%lld] %lld, LAPACK %lld\n", b, (long long)m, (long long)n, (long long)i, (long long)ip[i], (long long)rip[i]); break; }
      if (e > 1e-10) fprintf(stderr, "matrix %d (%lldx%lld): rel err %.3g\n", b, (long long)m, (long long)n, e);
    }
    // padding rows must stay untouched
    for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = m; i < ld; i++) if (MAGMA_Z_ABS(X[i + j * ld]) != 0) { maxerr = INFINITY; fprintf(stderr, "matrix %d: padding row %lld col %lld written\n", b, (long long)i, (long long)j); i = ld; j = n; }
  }
  bool ok = ret == 0 && bad_info == 0 && bad_ipiv == 0 && maxerr < 1e-10;
  printf("r61,%s,%d,%d,%d,%d,%d,%d,%.3g,%lld,%s\n", op.c_str(), set, pad, sing, bc, bad_info, bad_ipiv, maxerr, (long long)ret, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError()));
  for (int b = 0; b < bc; b++) { magma_free(hdA[b]); magma_free(hdp[b]); } if (work) magma_free(work);
  magma_free(dM); magma_free(dN); magma_free(dL); magma_free(dmn); magma_free(dinfo); magma_free(dAa); magma_free(dpa); magma_queue_destroy(q); magma_finalize(); return 0;
}

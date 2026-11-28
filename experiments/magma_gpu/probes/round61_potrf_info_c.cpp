// Round 61 (roxy), precision c, generated from round61_potrf_info.cpp by sed: position of the reported non-positive pivot in magma_cpotrf_vbatched (and magma_cpotrf_batched as control).
//   round61_potrf_info <vb|b> <n_big> <c_big> <n_small> <c_small>
//   batch of two HPD matrices (n_big, n_small; n_small = 0 drops it); diagonal entry c (0-based) set to -1 in each (c < 0: none).
//   vb: magma_cpotrf_vbatched (Lower; the large path runs when max n > 448); b: magma_cpotrf_batched (needs n_small = n_big).
// CSV: r61pic,op,n_big,c_big,n_small,c_small,info_big,expect_big,info_small,expect_small,return,untouched_matrices,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
typedef magmaFloatComplex Z;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 6) { printf("usage: see source\n"); return 1; }
  std::string op = argv[1]; int nb_ = atoi(argv[2]), cb = atoi(argv[3]), ns = atoi(argv[4]), cs = atoi(argv[5]);
  magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  std::vector<int> N = {nb_}, C = {cb}; if (ns > 0) { N.push_back(ns); C.push_back(cs); }
  int bc = (int)N.size(); std::vector<Z*> hp(bc); std::vector<magma_int_t> hn(bc), hl(bc), expect(bc), hinfo(bc);
  for (int b = 0; b < bc; b++) { int n = N[b]; hn[b] = n; hl[b] = n; std::vector<Z> A((size_t)n * n);
    for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { Z v = i == j ? MAGMA_C_MAKE(n + r1(), 0) : MAGMA_C_MAKE(r1(), r1()); A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = MAGMA_C_CONJ(v); }
    if (C[b] >= 0 && C[b] < n) A[C[b] + (size_t)C[b] * n] = MAGMA_C_MAKE(-1, 0);
    std::vector<Z> R = A; magma_int_t nn = n, ri = 0; lapackf77_cpotrf("L", &nn, R.data(), &nn, &ri); expect[b] = ri;
    magma_cmalloc(&hp[b], (size_t)n * n); magma_csetmatrix(n, n, A.data(), n, hp[b], n, q); }
  Z** dA; magma_malloc((void**)&dA, bc * sizeof(Z*)); magma_setvector(bc, sizeof(Z*), hp.data(), 1, dA, 1, q);
  magma_int_t *dn, *dl, *dinfo; magma_imalloc(&dn, bc + 1); magma_imalloc(&dl, bc + 1); magma_imalloc(&dinfo, bc);
  magma_isetvector(bc, hn.data(), 1, dn, 1, q); magma_isetvector(bc, hl.data(), 1, dl, 1, q); { std::vector<magma_int_t> z(bc, 0); magma_isetvector(bc, z.data(), 1, dinfo, 1, q); }
  std::vector<std::vector<Z>> before(bc); for (int b = 0; b < bc; b++) { before[b].resize((size_t)N[b] * N[b]); magma_cgetmatrix(N[b], N[b], hp[b], N[b], before[b].data(), N[b], q); }
  magma_int_t ret = op == "vb" ? magma_cpotrf_vbatched(MagmaLower, dn, dA, dl, dinfo, bc, q) : magma_cpotrf_batched(MagmaLower, nb_, dA, nb_, dinfo, bc, q);
  int untouched = 0; for (int b = 0; b < bc; b++) { std::vector<Z> after((size_t)N[b] * N[b]); magma_cgetmatrix(N[b], N[b], hp[b], N[b], after.data(), N[b], q);
    bool same = true; for (size_t i = 0; i < after.size(); i++) if (MAGMA_C_REAL(after[i]) != MAGMA_C_REAL(before[b][i]) || MAGMA_C_IMAG(after[i]) != MAGMA_C_IMAG(before[b][i])) { same = false; break; } untouched += same; }
  magma_queue_sync(q); magma_igetvector(bc, dinfo, 1, hinfo.data(), 1, q);
  bool ok = ret == 0 && untouched == 0; for (int b = 0; b < bc; b++) ok = ok && hinfo[b] == expect[b];
  printf("r61pic,%s,%d,%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%d,%s\n", op.c_str(), nb_, cb, ns, cs, (long long)hinfo[0], (long long)expect[0], bc > 1 ? (long long)hinfo[1] : -9LL, bc > 1 ? (long long)expect[1] : -9LL, (long long)ret, untouched, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError()));
  for (auto p : hp) magma_free(p); magma_free(dA); magma_free(dn); magma_free(dl); magma_free(dinfo); magma_queue_destroy(q); magma_finalize(); return 0;
}

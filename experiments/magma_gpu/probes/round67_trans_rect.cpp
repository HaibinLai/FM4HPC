// Round 67 (roxy): magma_zmtranspose / magma_zmtransposeconjugate on a rectangular CSR matrix (m x n, m != n).
//   round67_trans_rect <m> <n> <where host|dev|conjcpu|abscpu|structcpu> <conj 0|1>;  (conjcpu/abscpu/structcpu: the _cpu variants, conj ignored)
//   prints a marker before and after the call (a crash shows which)
// CSV: r67t,m,n,where,conj,ret,err,verdict
#include "magma_v2.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
  if (argc < 5) return 1; int m = atoi(argv[1]), n = atoi(argv[2]); std::string w = argv[3]; int cj = atoi(argv[4]);
  magma_init(); magma_queue_t Q; magma_queue_create(0, &Q);
  std::vector<magma_index_t> row(1, 0), col; std::vector<magmaDoubleComplex> val; std::vector<std::complex<double>> D((size_t)m * n);
  for (int i = 0; i < m; i++) { for (int k = 0; k < 3; k++) { int j = (i * 7 + k * 11) % n; if (k && j == ((i * 7 + (k - 1) * 11) % n)) continue; bool dup = false; for (magma_index_t p = row.back(); p < (magma_index_t)col.size(); p++) if (col[p] == j) dup = true; if (dup) continue;
      std::complex<double> v(1 + 0.01 * i, 0.3 + 0.02 * j); col.push_back(j); val.push_back(MAGMA_Z_MAKE(v.real(), v.imag())); D[i + (size_t)j * m] = v; }
    // keep column indices sorted within the row
    std::vector<std::pair<int, magmaDoubleComplex>> e; for (size_t p = row.back(); p < col.size(); p++) e.push_back({col[p], val[p]}); std::sort(e.begin(), e.end(), [](auto& a, auto& b) { return a.first < b.first; });
    for (size_t q = 0; q < e.size(); q++) { col[row.back() + q] = e[q].first; val[row.back() + q] = e[q].second; } row.push_back((magma_index_t)col.size()); }
  magma_z_matrix A = {Magma_CSR}, T = {Magma_CSR}, dA = {Magma_CSR}, dT = {Magma_CSR}, hT = {Magma_CSR};
  magma_index_t* r = (magma_index_t*)malloc(row.size() * sizeof(magma_index_t)); magma_index_t* c = (magma_index_t*)malloc(col.size() * sizeof(magma_index_t)); magmaDoubleComplex* v = (magmaDoubleComplex*)malloc(val.size() * sizeof(magmaDoubleComplex));
  memcpy(r, row.data(), row.size() * sizeof(magma_index_t)); memcpy(c, col.data(), col.size() * sizeof(magma_index_t)); memcpy(v, val.data(), val.size() * sizeof(magmaDoubleComplex));
  magma_zcsrset(m, n, r, c, v, &A, Q);
  magma_int_t ret; fprintf(stderr, "BEFORE %s %s\n", w.c_str(), cj ? "H" : "T");
  if (w == "host") { ret = cj ? magma_zmtransposeconjugate(A, &T, Q) : magma_zmtranspose(A, &T, Q); hT = T; }
  else if (w == "conjcpu") { cj = 1; ret = magma_zmtransposeconj_cpu(A, &T, Q); hT = T; }
  else if (w == "abscpu") { cj = 2; ret = magma_zmtransposeabs_cpu(A, &T, Q); hT = T; }
  else if (w == "structcpu") { cj = 3; ret = magma_zmtransposestruct_cpu(A, &T, Q); hT = T; }
  else { magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q); ret = cj ? magma_zmtransposeconjugate(dA, &dT, Q) : magma_zmtranspose(dA, &dT, Q); if (ret == 0) magma_zmtransfer(dT, &hT, Magma_DEV, Magma_CPU, Q); }
  fprintf(stderr, "AFTER ret %lld rows %lld cols %lld nnz %lld\n", (long long)ret, (long long)hT.num_rows, (long long)hT.num_cols, (long long)hT.nnz);
  double e = 0; if (ret == 0) { if (hT.num_rows != n || hT.num_cols != m) e = INFINITY; else { std::vector<std::complex<double>> X((size_t)n * m);
      for (int i = 0; i < n; i++) for (magma_index_t p = hT.row[i]; p < hT.row[i + 1]; p++) { int j = hT.col[p]; if (j < 0 || j >= m) { e = INFINITY; break; } X[i + (size_t)j * n] += cj == 3 ? std::complex<double>(1, 0) : std::complex<double>(MAGMA_Z_REAL(hT.val[p]), MAGMA_Z_IMAG(hT.val[p])); }   // structcpu: values unspecified, count stored entries
      for (int j = 0; j < m; j++) for (int i = 0; i < n; i++) { auto ref = D[j + (size_t)i * m]; if (cj == 1) ref = std::conj(ref); if (cj == 2) ref = std::abs(ref); if (cj == 3) { if ((ref != 0.0) != (X[i + (size_t)j * n] != 0.0)) { if (getenv("R67_DBG") && e == 0) fprintf(stderr, "struct mismatch T(%d,%d): ref %g X %g\n", i, j, std::abs(ref), std::abs(X[i + (size_t)j * n])); e = INFINITY; } continue; } e = std::max(e, std::abs(X[i + (size_t)j * n] - ref)); } } }
  printf("r67t,%d,%d,%s,%d,%lld,%.3g,%s\n", m, n, w.c_str(), cj, (long long)ret, e, ret == 0 && e < 1e-14 ? "OK" : "WRONG");
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

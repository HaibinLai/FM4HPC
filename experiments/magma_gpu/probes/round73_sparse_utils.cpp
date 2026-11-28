// Round 73 (roxy): sparse utilities never tested before, with complex values.
//   round73_sparse_utils tril|triu <n>        magma_zmatrix_tril / _triu on a random CSR matrix (lower / upper part incl. diagonal)
//   round73_sparse_utils diagadd <n>          magma_zmdiagadd: A + add*I (only stored diagonal entries exist in CSR)
//   round73_sparse_utils spmm <n> <k>         (first version: dense B, which magma_z_spmm rejects with -103: B must be CSR)
//   round73_sparse_utils spgemm <n>           magma_z_spmm with A, B both CSR on the device, alpha = 0.7-0.3i: err against alpha*A*B, err2 against A*B
//   round73_sparse_utils isort <len>          magma_zindexsort / magma_zindexsortval on random keys with duplicates
//   round73_sparse_utils stencil5|stencil27 <g>   magma_zm_5stencil / _27stencil: structure and values of the generated matrix
// CSV: r73,op,a,b,info,err,verdict
#include "magma_v2.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
typedef std::complex<double> cd;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static magma_queue_t Q; static int g_extra = 0, g_missing = 0;
static cd C(magmaDoubleComplex v) { return cd(MAGMA_Z_REAL(v), MAGMA_Z_IMAG(v)); }
static magma_z_matrix randcsr(int n, std::vector<cd>& D) { D.assign((size_t)n * n, 0.0); std::vector<magma_index_t> row(1, 0), col; std::vector<magmaDoubleComplex> val;
  for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) if (i == j || (r1() + 0.5) < 0.15) { cd v(r1() + (i == j ? 3 : 0), r1()); D[i + (size_t)j * n] = v; col.push_back(j); val.push_back(MAGMA_Z_MAKE(v.real(), v.imag())); } row.push_back((magma_index_t)col.size()); }
  magma_index_t* r = (magma_index_t*)malloc(row.size() * sizeof(magma_index_t)); magma_index_t* c = (magma_index_t*)malloc(col.size() * sizeof(magma_index_t)); magmaDoubleComplex* v = (magmaDoubleComplex*)malloc(val.size() * sizeof(magmaDoubleComplex));
  memcpy(r, row.data(), row.size() * sizeof(magma_index_t)); memcpy(c, col.data(), col.size() * sizeof(magma_index_t)); memcpy(v, val.data(), val.size() * sizeof(magmaDoubleComplex));
  magma_z_matrix A = {Magma_CSR}; magma_zcsrset(n, n, r, c, v, &A, Q); return A; }
static std::vector<cd> dense(const magma_z_matrix& A, int m, int n) { std::vector<cd> D((size_t)m * n, 0.0); for (int i = 0; i < (int)A.num_rows && i < m; i++) for (magma_index_t k = A.row[i]; k < A.row[i + 1]; k++) if (A.col[k] >= 0 && A.col[k] < n) D[i + (size_t)A.col[k] * m] += C(A.val[k]); return D; }
static double dif(const std::vector<cd>& X, const std::vector<cd>& R) { double e = 0; for (size_t i = 0; i < X.size(); i++) e = std::max(e, std::abs(X[i] - R[i])); return X.size() == R.size() ? e : INFINITY; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 3) return 1; std::string op = argv[1]; int a = atoi(argv[2]), b = argc > 3 ? atoi(argv[3]) : 0;
  magma_init(); magma_queue_create(0, &Q); magma_int_t info = 0; double err = -1;
  if (op == "tril" || op == "triu") { std::vector<cd> D; magma_z_matrix A = randcsr(a, D), L = {Magma_CSR}; info = op == "tril" ? magma_zmatrix_tril(A, &L, Q) : magma_zmatrix_triu(A, &L, Q);
    std::vector<cd> R = D; for (int j = 0; j < a; j++) for (int i = 0; i < a; i++) if ((op == "tril" && i < j) || (op == "triu" && i > j)) R[i + (size_t)j * a] = 0.0;
    err = info ? INFINITY : dif(dense(L, a, a), R); }
  else if (op == "diagadd") { std::vector<cd> D; magma_z_matrix A = randcsr(a, D); cd add(0.25, -0.5); info = magma_zmdiagadd(&A, MAGMA_Z_MAKE(add.real(), add.imag()), Q);
    std::vector<cd> R = D; for (int i = 0; i < a; i++) R[i + (size_t)i * a] += add; err = info ? INFINITY : dif(dense(A, a, a), R); }
  else if (op == "spmm") { std::vector<cd> D; magma_z_matrix A = randcsr(a, D), dA = {Magma_CSR}; magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q);
    int k = b; std::vector<magmaDoubleComplex> Bh((size_t)a * k); std::vector<cd> Bc((size_t)a * k); for (size_t i = 0; i < Bh.size(); i++) { Bc[i] = cd(r1(), r1()); Bh[i] = MAGMA_Z_MAKE(Bc[i].real(), Bc[i].imag()); }
    magma_z_matrix hB = {Magma_CSR}, dB = {Magma_CSR}, dC = {Magma_CSR}, hC = {Magma_CSR}; magma_zvset(a, k, Bh.data(), &hB, Q); hB.major = MagmaColMajor; magma_zmtransfer(hB, &dB, Magma_CPU, Magma_DEV, Q);
    magma_zvinit(&dC, Magma_DEV, a, k, MAGMA_Z_ZERO, Q); dC.major = MagmaColMajor; cd al(0.7, -0.3);
    info = magma_z_spmm(MAGMA_Z_MAKE(al.real(), al.imag()), dA, dB, &dC, Q); magma_queue_sync(Q); magma_zmtransfer(dC, &hC, Magma_DEV, Magma_CPU, Q);
    std::vector<cd> X((size_t)a * k), R((size_t)a * k, 0.0); for (size_t i = 0; i < X.size(); i++) X[i] = C(hC.val[i]);
    for (int c = 0; c < k; c++) for (int i = 0; i < a; i++) { cd s = 0; for (int j = 0; j < a; j++) s += D[i + (size_t)j * a] * Bc[j + (size_t)c * a]; R[i + (size_t)c * a] = al * s; }
    err = info ? INFINITY : dif(X, R); }
  else if (op == "spgemm") { std::vector<cd> DA, DB; magma_z_matrix A = randcsr(a, DA), B = randcsr(a, DB), dA = {Magma_CSR}, dB = {Magma_CSR}, dC = {Magma_CSR}, hC = {Magma_CSR};
    magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q); magma_zmtransfer(B, &dB, Magma_CPU, Magma_DEV, Q); cd al(0.7, -0.3);
    info = magma_z_spmm(MAGMA_Z_MAKE(al.real(), al.imag()), dA, dB, &dC, Q); magma_queue_sync(Q);
    if (!info) { magma_zmtransfer(dC, &hC, Magma_DEV, Magma_CPU, Q); std::vector<cd> X = dense(hC, a, a), R1((size_t)a * a, 0.0), R2 = R1;
      for (int j = 0; j < a; j++) for (int i = 0; i < a; i++) { cd t = 0; for (int l = 0; l < a; l++) t += DA[i + (size_t)l * a] * DB[l + (size_t)j * a]; R1[i + (size_t)j * a] = al * t; R2[i + (size_t)j * a] = t; }
      double n1 = 0; for (auto& v : R2) n1 = std::max(n1, std::abs(v)); err = dif(X, R1) / n1; double e2 = dif(X, R2) / n1;
      printf("r73,spgemm_vs_AB,%d,0,%lld,%.3g,%s\n", a, (long long)info, e2, e2 < 1e-13 ? "MATCHES_AB" : "NO"); } else err = INFINITY; }
  else if (op == "isort") { int len = a; std::vector<magma_index_t> x(len), x2(len); std::vector<magmaDoubleComplex> y(len); std::vector<std::pair<int, double>> ref(len);
    for (int i = 0; i < len; i++) { x[i] = (magma_index_t)((r1() + 0.5) * (len / 3 + 1)); x2[i] = x[i]; y[i] = MAGMA_Z_MAKE(x[i] * 10 + 0.5, -(double)x[i]); }   // value tied to its key
    info = magma_zindexsort(x.data(), 0, len - 1, Q); magma_int_t i2 = magma_zindexsortval(x2.data(), y.data(), 0, len - 1, Q); if (!info) info = i2;
    std::vector<magma_index_t> s = x; std::sort(s.begin(), s.end()); double e = 0; for (int i = 0; i < len; i++) { if (x[i] != s[i]) e = INFINITY; if (x2[i] != s[i]) e = INFINITY;
      if (MAGMA_Z_REAL(y[i]) != x2[i] * 10 + 0.5 || MAGMA_Z_IMAG(y[i]) != -(double)x2[i]) e = INFINITY; } err = e; }
  else if (op == "stencil5" || op == "stencil27") { magma_z_matrix A = {Magma_CSR}; info = op == "stencil5" ? magma_zm_5stencil(a, &A, Q) : magma_zm_27stencil(a, &A, Q);
    // expected: 5-point: 2-D grid a x a, diagonal 4, neighbours -1;  27-point: 3-D grid a^3, diagonal 26, neighbours -1
    bool three = op == "stencil27"; int N = three ? a * a * a : a * a; err = 0;
    if (info || (int)A.num_rows != N || (int)A.num_cols != N) err = INFINITY;
    else { std::vector<std::map<int, cd>> E(N); for (int r = 0; r < N; r++) for (magma_index_t k = A.row[r]; k < A.row[r + 1]; k++) E[r][A.col[k]] += C(A.val[k]);
      for (int r = 0; r < N; r++) { int x = r % a, y = (r / a) % a, z = three ? r / (a * a) : 0; std::map<int, cd> want;
        for (int dz = three ? -1 : 0; dz <= (three ? 1 : 0); dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
          if (!three && dx != 0 && dy != 0) continue; int X = x + dx, Y = y + dy, Zz = z + dz; if (X < 0 || Y < 0 || Zz < 0 || X >= a || Y >= a || Zz >= a) continue;
          int c = X + Y * a + Zz * a * a; want[c] = (dx == 0 && dy == 0 && dz == 0) ? cd(three ? 26 : 4, 0) : cd(-1, 0); }
        cd sc = 1.0; { auto it0 = E[0].find(0); if (it0 != E[0].end()) sc = it0->second / cd(three ? 26 : 4, 0); }   // the complex build scales the stencil by (1+i) on purpose
        for (auto& kv : want) { auto it = E[r].find(kv.first); err = std::max(err, it == E[r].end() ? 1.0 : std::abs(it->second - sc * kv.second)); }
        for (auto& kv : want) if (!E[r].count(kv.first) || std::abs(E[r][kv.first]) == 0) g_missing++;
        for (auto& kv : E[r]) if (!want.count(kv.first) && std::abs(kv.second) != 0) { err = std::max(err, 1.0); g_extra++; } }
      fprintf(stderr, "stencil n=%d: spurious couplings %d, missing %d\n", N, g_extra, g_missing); } }
  printf("r73,%s,%d,%d,%lld,%.3g,%s\n", op.c_str(), a, b, (long long)info, err, info == 0 && err == err && err < 1e-13 ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

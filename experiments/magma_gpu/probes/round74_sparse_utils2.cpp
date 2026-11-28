// Round 74 (roxy): more sparse utilities, complex data, CPU references written from the documentation.
//   round74_sparse_utils2 shift <csr|ell|ellpackt> <n> <alpha1 0|1>   magma_z_spmv_shift, documented y = alpha (A - lambda I) x + beta y
//                         (offset 0, blocksize n, add_rows NULL); err against the documented formula, err2 against alpha A x - lambda x + beta y
//   round74_sparse_utils2 conj <csr|ell> <n>        magma_zmconjugate on a device matrix (documented "conjugates a matrix")
//   round74_sparse_utils2 compress <n>              magma_zmcsrcompressor_gpu on a device CSR matrix with stored zeros, pure-imaginary and
//                         pure-real entries (only exact complex zeros may be removed)
//   round74_sparse_utils2 slice <n> <num_slices> <slice>   magma_zmslice: B (slice rows of A, identity elsewhere), ALOC/ANLOC, comm plan, start/end
//                         R74_FREE: after an error, free B, ALOC, ANLOC with magma_zmfree (markers show where it crashes)
//   round74_sparse_utils2 shrink <m> <n> <cpu|dev>  magma_zmshrink (m < n): keep the leading m x m block
//   round74_sparse_utils2 vtrans <m> <n> <cpu|dev>  magma_zvtranspose of a column-major dense m x n block
// CSV: r74,op,a,b,c,info,err,err2,verdict
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
typedef std::complex<double> cd;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static magma_queue_t Q;
static cd C(magmaDoubleComplex v) { return cd(MAGMA_Z_REAL(v), MAGMA_Z_IMAG(v)); }
static magmaDoubleComplex Z(cd v) { return MAGMA_Z_MAKE(v.real(), v.imag()); }
// random m x n CSR on the CPU; kind 0: ordinary values; kind 1: a third of the stored values are exact zeros, a third pure imaginary, a third pure real
static magma_z_matrix randcsr(int m, int n, std::vector<cd>& D, int kind = 0) { D.assign((size_t)m * n, 0.0); std::vector<magma_index_t> row(1, 0), col; std::vector<magmaDoubleComplex> val;
  for (int i = 0; i < m; i++) { for (int j = 0; j < n; j++) if (i == j || (r1() + 0.5) < 0.2) { cd v(r1() + (i == j ? 3 : 0), r1());
        if (kind == 1) { int t = (int)((r1() + 0.5) * 3); v = t == 0 ? cd(0, 0) : t == 1 ? cd(0, r1() + 0.7) : cd(r1() + 0.7, 0); }
        D[i + (size_t)j * m] = v; col.push_back(j); val.push_back(Z(v)); } row.push_back((magma_index_t)col.size()); }
  magma_index_t* r = (magma_index_t*)malloc(row.size() * sizeof(magma_index_t)); magma_index_t* c = (magma_index_t*)malloc(std::max<size_t>(col.size(), 1) * sizeof(magma_index_t)); magmaDoubleComplex* v = (magmaDoubleComplex*)malloc(std::max<size_t>(val.size(), 1) * sizeof(magmaDoubleComplex));
  memcpy(r, row.data(), row.size() * sizeof(magma_index_t)); memcpy(c, col.data(), col.size() * sizeof(magma_index_t)); memcpy(v, val.data(), val.size() * sizeof(magmaDoubleComplex));
  magma_z_matrix A = {Magma_CSR}; magma_zcsrset(m, n, r, c, v, &A, Q); return A; }
static std::vector<cd> dense(const magma_z_matrix& A, int m, int n, bool* bad = NULL) { std::vector<cd> D((size_t)m * n, 0.0);
  for (int i = 0; i < (int)A.num_rows && i < m; i++) for (magma_index_t k = A.row[i]; k < A.row[i + 1]; k++) { if (A.col[k] >= 0 && A.col[k] < n) D[i + (size_t)A.col[k] * m] += C(A.val[k]); else if (bad) *bad = true; } return D; }
static double dif(const std::vector<cd>& X, const std::vector<cd>& R) { if (X.size() != R.size()) return INFINITY; double e = 0, a = 0; for (size_t i = 0; i < X.size(); i++) { double d = std::abs(X[i] - R[i]); if (!(d <= e)) e = d; a = std::max(a, std::abs(R[i])); } return a > 0 ? e / a : e; }
static magma_z_matrix dvec(const std::vector<cd>& h, int m, int n) { std::vector<magmaDoubleComplex> t(h.size()); for (size_t i = 0; i < h.size(); i++) t[i] = Z(h[i]); magma_z_matrix x = {Magma_CSR}, dx = {Magma_CSR};
  magma_zvset(m, n, t.data(), &x, Q); magma_zmtransfer(x, &dx, Magma_CPU, Magma_DEV, Q); return dx; }
static std::vector<cd> hvec(magma_z_matrix dx) { magma_z_matrix x = {Magma_CSR}; magma_zmtransfer(dx, &x, Magma_DEV, Magma_CPU, Q); std::vector<cd> h((size_t)x.num_rows * x.num_cols); for (size_t i = 0; i < h.size(); i++) h[i] = C(x.val[i]); return h; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
  if (argc < 3) return 1; std::string op = argv[1]; magma_init(); magma_queue_create(0, &Q);
  long long info = 0; double err = -1, err2 = -1; std::string tag; bool ok = false;
  if (op == "shift") {
    std::string fmt = argv[2]; int n = atoi(argv[3]); bool a1 = atoi(argv[4]) == 1; std::vector<cd> D; magma_z_matrix A = randcsr(n, n, D), B = {Magma_CSR}, dA = {Magma_CSR};
    magma_storage_t st = fmt == "csr" ? Magma_CSR : fmt == "ell" ? Magma_ELL : Magma_ELLPACKT; if (st == Magma_CSR) B = A; else magma_zmconvert(A, &B, Magma_CSR, st, Q);
    magma_zmtransfer(B, &dA, Magma_CPU, Magma_DEV, Q);
    std::vector<cd> x(n), y(n); for (auto& v : x) v = cd(r1(), r1()); for (auto& v : y) v = cd(r1(), r1());
    cd al = a1 ? cd(1, 0) : cd(0.7, -0.3), la(0.4, 0.25), be(-0.4, 0.2); magma_z_matrix dx = dvec(x, n, 1), dy = dvec(y, n, 1);
    info = magma_z_spmv_shift(Z(al), dA, Z(la), dx, Z(be), 0, n, NULL, dy, Q); magma_queue_sync(Q); std::vector<cd> X = hvec(dy), R1(n), R2(n);
    for (int i = 0; i < n; i++) { cd s = 0; for (int j = 0; j < n; j++) s += D[i + (size_t)j * n] * x[j]; R1[i] = al * (s - la * x[i]) + be * y[i]; R2[i] = al * s - la * x[i] + be * y[i]; }
    err = dif(X, R1); err2 = dif(X, R2); ok = info == 0 && err < 1e-13; tag = fmt + "," + argv[3] + "," + argv[4];
  } else if (op == "conj") {
    std::string fmt = argv[2]; int n = atoi(argv[3]); std::vector<cd> D; magma_z_matrix A = randcsr(n, n, D), B = {Magma_CSR}, dA = {Magma_CSR}, hA = {Magma_CSR}, hC = {Magma_CSR};
    magma_storage_t st = fmt == "csr" ? Magma_CSR : Magma_ELL; if (st == Magma_CSR) B = A; else magma_zmconvert(A, &B, Magma_CSR, st, Q);
    magma_zmtransfer(B, &dA, Magma_CPU, Magma_DEV, Q); fprintf(stderr, "BEFORE conj %s\n", fmt.c_str());
    info = magma_zmconjugate(&dA, Q); magma_queue_sync(Q); fprintf(stderr, "AFTER conj cuda=%s\n", cudaGetErrorString(cudaGetLastError()));
    magma_zmtransfer(dA, &hA, Magma_DEV, Magma_CPU, Q); if (st == Magma_CSR) hC = hA; else magma_zmconvert(hA, &hC, st, Magma_CSR, Q);
    std::vector<cd> R = D; for (auto& v : R) v = std::conj(v); err = dif(dense(hC, n, n), R); err2 = dif(dense(hC, n, n), D); ok = info == 0 && err < 1e-15; tag = fmt + "," + argv[3] + ",-";
  } else if (op == "compress") {
    int n = atoi(argv[2]); std::vector<cd> D; magma_z_matrix A = randcsr(n, n, D, 1), dA = {Magma_CSR}, hA = {Magma_CSR}; long long nz = 0; for (auto& v : D) if (v != 0.0) nz++;
    magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q); info = magma_zmcsrcompressor_gpu(&dA, Q); magma_queue_sync(Q); magma_zmtransfer(dA, &hA, Magma_DEV, Magma_CPU, Q);
    bool bad = false; err = dif(dense(hA, n, n, &bad), D); long long stz = 0; for (magma_int_t k = 0; k < hA.nnz; k++) if (C(hA.val[k]) == 0.0) stz++;
    err2 = (double)(hA.nnz - nz); fprintf(stderr, "compress n=%d: nnz before %lld, nonzeros %lld, nnz after %lld, stored zeros after %lld, row[n] %lld\n", n, (long long)A.nnz, nz, (long long)hA.nnz, stz, (long long)hA.row[n]);
    ok = info == 0 && !bad && err == 0 && hA.nnz == nz && stz == 0 && hA.row[n] == nz; tag = std::string(argv[2]) + ",-,-";
  } else if (op == "slice") {
    int n = atoi(argv[2]), ns = atoi(argv[3]), sl = atoi(argv[4]); std::vector<cd> D; magma_z_matrix A = randcsr(n, n, D), B = {Magma_CSR}, AL = {Magma_CSR}, AN = {Magma_CSR};
    std::vector<magma_index_t> ci(n + 64, -7); std::vector<magmaDoubleComplex> cv(n + 64, MAGMA_Z_MAKE(-7, -7)); magma_int_t st = -1, en = -1;
    int s = (n + ns - 1) / ns, ls = std::min(sl * s, n), le = std::min((sl + 1) * s, n);   // documented intent: rows [ls, le) of a partition into num_slices pieces (empty at the end)
    fprintf(stderr, "BEFORE slice n=%d ns=%d sl=%d (raw lstart %d)\n", n, ns, sl, sl * s);
    info = magma_zmslice(ns, sl, A, &B, &AL, &AN, ci.data(), cv.data(), &st, &en, Q); fprintf(stderr, "AFTER info %lld start %lld end %lld B.nnz %lld\n", info, (long long)st, (long long)en, (long long)B.nnz);
    if (info != 0 && getenv("R74_FREE")) {   // a caller cleaning up after the error, as the testers do with magma_zmfree
      fprintf(stderr, "FREE B\n"); magma_zmfree(&B, Q); fprintf(stderr, "FREE ALOC\n"); magma_zmfree(&AL, Q); fprintf(stderr, "FREE ANLOC\n"); magma_zmfree(&AN, Q); fprintf(stderr, "FREED\n"); }
    if (info == 0) { std::vector<cd> RB((size_t)n * n, 0.0), RL, RN; for (int i = 0; i < n; i++) { if (i >= ls && i < le) for (int j = 0; j < n; j++) RB[i + (size_t)j * n] = D[i + (size_t)j * n]; else RB[i + (size_t)i * n] = 1.0; }
      bool bad = false; err = dif(dense(B, n, n, &bad), RB); if (bad || B.row[n] != B.nnz) err = INFINITY; int sz = le - ls; double e2 = 0;
      if (st != ls || en != le || AL.num_rows != sz || AN.num_rows != sz) e2 = INFINITY;
      else { RL.assign((size_t)sz * sz, 0.0); RN.assign((size_t)sz * n, 0.0); std::vector<int> wi(n, 0); std::vector<cd> wv(n, 0.0);
        for (int i = ls; i < le; i++) for (int j = 0; j < n; j++) { cd v = D[i + (size_t)j * n]; bool stored = false; for (magma_index_t k = A.row[i]; k < A.row[i + 1]; k++) if (A.col[k] == j) stored = true; if (!stored) continue;
            if (j >= ls && j < le) RL[(i - ls) + (size_t)(j - ls) * sz] = v; else { RN[(i - ls) + (size_t)j * sz] = v; wi[j] = 1; wv[j] += std::abs(v); } }
        e2 = std::max(dif(dense(AL, sz, sz, &bad), RL), dif(dense(AN, sz, n, &bad), RN)); if (bad) e2 = INFINITY;
        for (int j = 0; j < n; j++) { if (ci[j] != wi[j]) e2 = INFINITY; e2 = std::max(e2, std::abs(C(cv[j]) - wv[j])); } }
      err2 = e2; }
    ok = info == 0 && err < 1e-15 && err2 < 1e-13; tag = std::string(argv[2]) + "," + argv[3] + "," + argv[4];
  } else if (op == "shrink") {
    int m = atoi(argv[2]), n = atoi(argv[3]); bool dev = std::string(argv[4]) == "dev"; std::vector<cd> D; magma_z_matrix A = randcsr(m, n, D), dA = {Magma_CSR}, B = {Magma_CSR}, hB = {Magma_CSR};
    if (dev) magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q); info = magma_zmshrink(dev ? dA : A, &B, Q); if (dev) magma_zmtransfer(B, &hB, Magma_DEV, Magma_CPU, Q); else hB = B;
    std::vector<cd> R((size_t)m * m); for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) R[i + (size_t)j * m] = D[i + (size_t)j * m];
    if (info == 0) { bool bad = false; err = (hB.num_rows == m && hB.num_cols == m) ? dif(dense(hB, m, m, &bad), R) : INFINITY; if (bad) err = INFINITY; }
    fprintf(stderr, "shrink %dx%d %s: info %lld rows %lld cols %lld nnz %lld\n", m, n, argv[4], info, (long long)hB.num_rows, (long long)hB.num_cols, (long long)hB.nnz);
    ok = info == 0 && err < 1e-15; tag = std::string(argv[2]) + "," + argv[3] + "," + argv[4];
  } else if (op == "vtrans") {
    int m = atoi(argv[2]), n = atoi(argv[3]); bool dev = std::string(argv[4]) == "dev"; std::vector<cd> X((size_t)m * n); for (auto& v : X) v = cd(r1(), r1());
    std::vector<magmaDoubleComplex> t(X.size()); for (size_t i = 0; i < X.size(); i++) t[i] = Z(X[i]); magma_z_matrix x = {Magma_CSR}, dx = {Magma_CSR}, y = {Magma_CSR}, hy = {Magma_CSR};
    magma_zvset(m, n, t.data(), &x, Q); if (dev) magma_zmtransfer(x, &dx, Magma_CPU, Magma_DEV, Q); info = magma_zvtranspose(dev ? dx : x, &y, Q); magma_queue_sync(Q);
    if (dev) magma_zmtransfer(y, &hy, Magma_DEV, Magma_CPU, Q); else hy = y; double e = 0;
    if (info == 0) { if (hy.num_rows != m || hy.num_cols != n || hy.major != MagmaRowMajor) e = INFINITY; else for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) e = std::max(e, std::abs(C(hy.val[(size_t)i * n + j]) - X[i + (size_t)j * m])); }
    fprintf(stderr, "vtrans: rows %lld cols %lld major %d\n", (long long)hy.num_rows, (long long)hy.num_cols, (int)hy.major);
    err = info ? INFINITY : e; ok = info == 0 && e == 0; tag = std::string(argv[2]) + "," + argv[3] + "," + argv[4];
  }
  printf("r74,%s,%s,%lld,%.3g,%.3g,%s\n", op.c_str(), tag.c_str(), info, err, err2, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

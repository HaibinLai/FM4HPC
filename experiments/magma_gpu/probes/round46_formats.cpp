// Round 46 (roxy): every sparse storage format MAGMA converts to, on small edge-case matrices.  One process per
// (format, matrix) so that a crash is attributed (round 29 missed crashes; round 45 found M251 in the conversion code).
//   round46_formats <FORMAT> <matrix>     FORMAT in ELLPACKT ELL DENSE BCSR CSC COO ELLRT SELLP ELLD CSRLIST CSRD CSRL CSRU CSRCOO CUCSR CSR5
// Steps, each printed before it runs (the last STEP line of a crashed process names the culprit):
//   1 host CSR -> F      2 host F -> CSR, compare as dense (CSRL / CSRU: against tril / triu)
//   3 F to the device and y = A x with magma_z_spmv, compare with the host product
//   4 device CSR -> F (device path) and back to CSR on the device, compare
// CSV: r46,format,matrix,step,ret,maxerr,verdict     verdict OK | WRONG | UNSUPPORTED (ret = MAGMA_ERR_NOT_SUPPORTED)
#include "magma_v2.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
static magma_queue_t Q; static unsigned Sd = 1u;
static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
struct Dn { int m, n; std::vector<double> a; double& at(int i, int j) { return a[i + (size_t)j * m]; } };
static Dn mat(const std::string& nm) {
  Dn D; Sd = 3u + (unsigned)nm.size();
  auto mk = [&](int m, int n) { D.m = m; D.n = n; D.a.assign((size_t)m * n, 0); };
  if (nm == "n1") { mk(1, 1); D.at(0, 0) = 3; }
  else if (nm == "poisson") { int g = 6; mk(g * g, g * g); for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) { int i = y * g + x; D.at(i, i) = 4; if (x) D.at(i, i - 1) = -1; if (x + 1 < g) D.at(i, i + 1) = -1; if (y) D.at(i, i - g) = -1; if (y + 1 < g) D.at(i, i + g) = -1; } }
  else if (nm == "random") { mk(53, 53); for (int i = 0; i < 53; i++) { D.at(i, i) = 2 + r1(); for (int k = 0; k < 4; k++) { int j = (int)((r1() + 0.5) * 53) % 53; D.at(i, j) += r1(); } } }
  else if (nm == "emptyrow") { mk(40, 40); for (int i = 0; i < 40; i++) if (i != 17) { D.at(i, i) = 2; if (i + 1 < 40) D.at(i, i + 1) = r1(); if (i) D.at(i, i - 1) = r1(); } }
  else if (nm == "denserow") { mk(40, 40); for (int i = 0; i < 40; i++) D.at(i, i) = 3; for (int j = 0; j < 40; j++) D.at(5, j) = 0.5 + 0.01 * j; }
  else if (nm == "rect") { mk(30, 47); for (int i = 0; i < 30; i++) for (int k = 0; k < 3; k++) D.at(i, (i * 7 + k * 11) % 47) = 1 + r1(); }
  else if (nm == "zero") { mk(16, 16); }   // nnz = 0
  else if (nm == "emptyrow0") { mk(20, 20); for (int i = 1; i < 20; i++) { D.at(i, i) = 2; if (i > 1) D.at(i, i - 1) = r1(); } }   // row 0 empty
  return D; }
static magma_z_matrix toCsr(const Dn& D) { std::vector<magma_index_t> row(1, 0), col; std::vector<magmaDoubleComplex> val;
  for (int i = 0; i < D.m; i++) { for (int j = 0; j < D.n; j++) { double v = D.a[i + (size_t)j * D.m]; if (v != 0) { col.push_back(j); val.push_back(MAGMA_Z_MAKE(v, 0)); } } row.push_back((magma_index_t)col.size()); }
  if (col.empty()) { col.push_back(0); val.push_back(MAGMA_Z_ZERO); }   // keep valid pointers for nnz = 0
  magma_z_matrix A = {Magma_CSR}; magma_index_t* r = (magma_index_t*)malloc(row.size() * sizeof(magma_index_t)); magma_index_t* c = (magma_index_t*)malloc(col.size() * sizeof(magma_index_t)); magmaDoubleComplex* v = (magmaDoubleComplex*)malloc(val.size() * sizeof(magmaDoubleComplex));
  memcpy(r, row.data(), row.size() * sizeof(magma_index_t)); memcpy(c, col.data(), col.size() * sizeof(magma_index_t)); memcpy(v, val.data(), val.size() * sizeof(magmaDoubleComplex));
  magma_zcsrset(D.m, D.n, r, c, v, &A, Q); A.nnz = row.back(); return A; }
static Dn fromCsr(const magma_z_matrix& A, int m, int n) { Dn D; D.m = m; D.n = n; D.a.assign((size_t)m * n, 0);
  for (int i = 0; i < (int)A.num_rows && i < m; i++) for (magma_index_t k = A.row[i]; k < A.row[i + 1]; k++) { int j = (int)A.col[k]; if (j >= 0 && j < n) D.a[i + (size_t)j * m] += MAGMA_Z_REAL(A.val[k]); } return D; }
static double diff(const Dn& X, const Dn& Y) { double e = 0; for (size_t i = 0; i < X.a.size() && i < Y.a.size(); i++) { double d = fabs(X.a[i] - Y.a[i]); if (!(d <= e)) e = d; } return e; }
static void out(const char* f, const char* m, int step, long long ret, double e) {
  const char* v = ret == MAGMA_ERR_NOT_SUPPORTED ? "UNSUPPORTED" : (ret == 0 && e <= 1e-12) ? "OK" : "WRONG"; printf("r46,%s,%s,%d,%lld,%.3g,%s\n", f, m, step, ret, e, v); }

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
  if (argc < 3) { printf("usage: round46_formats <FORMAT> <matrix>\n"); return 0; }
  std::map<std::string, magma_storage_t> F = {{"ELLPACKT", Magma_ELLPACKT}, {"ELL", Magma_ELL}, {"DENSE", Magma_DENSE}, {"BCSR", Magma_BCSR}, {"CSC", Magma_CSC}, {"COO", Magma_COO}, {"ELLRT", Magma_ELLRT}, {"SELLP", Magma_SELLP},
    {"ELLD", Magma_ELLD}, {"CSRLIST", Magma_CSRLIST}, {"CSRD", Magma_CSRD}, {"CSRL", Magma_CSRL}, {"CSRU", Magma_CSRU}, {"CSRCOO", Magma_CSRCOO}, {"CUCSR", Magma_CUCSR}, {"CSR5", Magma_CSR5}};
  std::string fn = argv[1], mn = argv[2]; if (!F.count(fn)) { printf("unknown format\n"); return 0; }
  magma_storage_t fmt = F[fn]; magma_init(); magma_queue_create(0, &Q);
  Dn D = mat(mn); Dn ref = D;
  if (fn == "CSRL") for (int j = 0; j < D.n; j++) for (int i = 0; i < D.m; i++) if (i < j) ref.at(i, j) = 0;    // lower incl. diagonal
  if (fn == "CSRU") for (int j = 0; j < D.n; j++) for (int i = 0; i < D.m; i++) if (i > j) ref.at(i, j) = 0;    // upper incl. diagonal
  magma_z_matrix A = toCsr(D), B = {Magma_CSR}, C = {Magma_CSR}, dB = {Magma_CSR}, dA = {Magma_CSR}, dF = {Magma_CSR}, dC = {Magma_CSR}, hC = {Magma_CSR};
  B.blocksize = 8; B.alignment = 8; dF.blocksize = 8; dF.alignment = 8;   // SELL-P / ELLRT parameters (BCSR uses its own)
  fprintf(stderr, "STEP 1\n"); magma_int_t r = magma_zmconvert(A, &B, Magma_CSR, fmt, Q); out(fn.c_str(), mn.c_str(), 1, r, 0);
  if (r == 0) {
    fprintf(stderr, "STEP 2\n"); magma_int_t r2 = magma_zmconvert(B, &C, fmt, Magma_CSR, Q); double e = r2 == 0 ? diff(fromCsr(C, D.m, D.n), ref) : 0; out(fn.c_str(), mn.c_str(), 2, r2, e);
    if (fn != "CSRL" && fn != "CSRU") {
      fprintf(stderr, "STEP 3\n"); magma_int_t r3 = magma_zmtransfer(B, &dB, Magma_CPU, Magma_DEV, Q);
      magma_z_matrix x = {Magma_CSR}, y = {Magma_CSR}, hy = {Magma_CSR}; std::vector<magmaDoubleComplex> xv(D.n); for (int j = 0; j < D.n; j++) xv[j] = MAGMA_Z_MAKE(1.0 + 0.1 * j, 0);
      magma_zvset(D.n, 1, xv.data(), &hy, Q); magma_zmtransfer(hy, &x, Magma_CPU, Magma_DEV, Q); magma_zvinit(&y, Magma_DEV, D.m, 1, MAGMA_Z_MAKE(7, 0), Q);
      if (r3 == 0) r3 = magma_z_spmv(MAGMA_Z_ONE, dB, x, MAGMA_Z_ZERO, y, Q); magma_queue_sync(Q);
      double e3 = 0; if (r3 == 0) { magma_z_matrix gy = {Magma_CSR}; magma_zmtransfer(y, &gy, Magma_DEV, Magma_CPU, Q);
        for (int i = 0; i < D.m; i++) { double s = 0; for (int j = 0; j < D.n; j++) s += D.a[i + (size_t)j * D.m] * (1.0 + 0.1 * j); double d = fabs(MAGMA_Z_REAL(gy.val[i]) - s); if (!(d <= e3)) e3 = d;
          if (getenv("R46_DUMP") && d > 1e-12) fprintf(stderr, "  row %d: got %g expected %g (nnz in row %d)\n", i, MAGMA_Z_REAL(gy.val[i]), s, (int)(A.row[i + 1] - A.row[i])); }
        if (getenv("R46_DUMP")) fprintf(stderr, "  format fields: max_nnz_row=%lld alignment=%lld blocksize=%lld numblocks=%lld\n", (long long)dB.max_nnz_row, (long long)dB.alignment, (long long)dB.blocksize, (long long)dB.numblocks);
        magma_zmfree(&gy, Q); }
      out(fn.c_str(), mn.c_str(), 3, r3, e3);
      magma_zmfree(&x, Q); magma_zmfree(&y, Q); magma_zmfree(&hy, Q); }
  }
  fprintf(stderr, "STEP 4\n"); magma_int_t r4 = magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q); if (r4 == 0) r4 = magma_zmconvert(dA, &dF, Magma_CSR, fmt, Q);
  double e4 = 0; if (r4 == 0) { magma_int_t rb = magma_zmconvert(dF, &dC, fmt, Magma_CSR, Q); if (rb == 0) { magma_zmtransfer(dC, &hC, Magma_DEV, Magma_CPU, Q); e4 = diff(fromCsr(hC, D.m, D.n), ref); } else r4 = rb; }
  out(fn.c_str(), mn.c_str(), 4, r4, e4);
  fprintf(stderr, "END\n"); magma_queue_destroy(Q); magma_finalize(); return 0;
}

// Round 67 (roxy): the round-46 format sweep with genuinely complex values (round 46 used imaginary part 0 everywhere).
//   round67_formats_complex <FORMAT|TRANS> <matrix>
//   FORMAT: as round 46 (CSR -> F on host, F -> CSR round trip, SpMV on F on the device with a complex x, device CSR -> F -> CSR)
//   TRANS:  CSR only: magma_zmtranspose (A^T, no conjugation) and magma_zmtransposeconjugate (A^H), on the host and the device
//   matrices: n1 poisson random emptyrow denserow rect zero emptyrow0 (the round-46 patterns), every nonzero a + i b with b != 0
// CSV: r67,format,matrix,step,ret,err,verdict
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
typedef std::complex<double> cd;
static magma_queue_t Q;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
struct Dn { int m = 0, n = 0; std::vector<cd> a; cd& at(int i, int j) { return a[i + (size_t)j * m]; } };
static Dn mat(const std::string& nm) {
  Dn D; auto mk = [&](int m, int n) { D.m = m; D.n = n; D.a.assign((size_t)m * n, 0.0); };
  if (nm == "n1") { mk(1, 1); D.at(0, 0) = 3; }
  else if (nm == "poisson") { int g = 6; mk(g * g, g * g); for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) { int i = y * g + x; D.at(i, i) = 4; if (x) D.at(i, i - 1) = -1; if (x + 1 < g) D.at(i, i + 1) = -1; if (y) D.at(i, i - g) = -1; if (y + 1 < g) D.at(i, i + g) = -1; } }
  else if (nm == "random") { mk(53, 53); for (int i = 0; i < 53; i++) { D.at(i, i) = 2 + r1(); for (int k = 0; k < 4; k++) { int j = (int)((r1() + 0.5) * 53) % 53; D.at(i, j) += r1(); } } }
  else if (nm == "emptyrow") { mk(40, 40); for (int i = 0; i < 40; i++) if (i != 17) { D.at(i, i) = 2; if (i + 1 < 40) D.at(i, i + 1) = r1(); if (i) D.at(i, i - 1) = r1(); } }
  else if (nm == "denserow") { mk(40, 40); for (int i = 0; i < 40; i++) D.at(i, i) = 3; for (int j = 0; j < 40; j++) D.at(5, j) = 0.5 + 0.01 * j; }
  else if (nm == "rect") { mk(30, 47); for (int i = 0; i < 30; i++) for (int k = 0; k < 3; k++) D.at(i, (i * 7 + k * 11) % 47) = 1 + r1(); }
  else if (nm == "zero") { mk(16, 16); }
  else if (nm == "emptyrow0") { mk(20, 20); for (int i = 1; i < 20; i++) { D.at(i, i) = 2; if (i > 1) D.at(i, i - 1) = r1(); } }
  for (int j = 0; j < D.n; j++) for (int i = 0; i < D.m; i++) { cd& v = D.at(i, j); if (v != 0.0) v = cd(v.real(), 0.3 * v.real() + 0.05 * (i - 2 * j) + 0.11); }   // make every nonzero genuinely complex
  return D; }
static magmaDoubleComplex Z(cd v) { return MAGMA_Z_MAKE(v.real(), v.imag()); }
static cd C(magmaDoubleComplex v) { return cd(MAGMA_Z_REAL(v), MAGMA_Z_IMAG(v)); }
static magma_z_matrix toCsr(const Dn& D) { std::vector<magma_index_t> row(1, 0), col; std::vector<magmaDoubleComplex> val;
  for (int i = 0; i < D.m; i++) { for (int j = 0; j < D.n; j++) { cd v = D.a[i + (size_t)j * D.m]; if (v != 0.0) { col.push_back(j); val.push_back(Z(v)); } } row.push_back((magma_index_t)col.size()); }
  if (col.empty()) { col.push_back(0); val.push_back(MAGMA_Z_ZERO); }
  magma_z_matrix A = {Magma_CSR}; magma_index_t* r = (magma_index_t*)malloc(row.size() * sizeof(magma_index_t)); magma_index_t* c = (magma_index_t*)malloc(col.size() * sizeof(magma_index_t)); magmaDoubleComplex* v = (magmaDoubleComplex*)malloc(val.size() * sizeof(magmaDoubleComplex));
  memcpy(r, row.data(), row.size() * sizeof(magma_index_t)); memcpy(c, col.data(), col.size() * sizeof(magma_index_t)); memcpy(v, val.data(), val.size() * sizeof(magmaDoubleComplex));
  magma_zcsrset(D.m, D.n, r, c, v, &A, Q); A.nnz = row.back(); return A; }
static Dn fromCsr(const magma_z_matrix& A, int m, int n) { Dn D; D.m = m; D.n = n; D.a.assign((size_t)m * n, 0.0);
  for (int i = 0; i < (int)A.num_rows && i < m; i++) for (magma_index_t k = A.row[i]; k < A.row[i + 1]; k++) { int j = (int)A.col[k]; if (j >= 0 && j < n) D.a[i + (size_t)j * m] += C(A.val[k]); } return D; }
static double diff(const Dn& X, const Dn& Y) { double e = 0; for (size_t i = 0; i < X.a.size() && i < Y.a.size(); i++) { double d = std::abs(X.a[i] - Y.a[i]); if (!(d <= e)) e = d; } if (X.a.size() != Y.a.size()) e = INFINITY; return e; }
static void out(const char* f, const char* m, const char* step, long long ret, double e) {
  const char* v = ret == MAGMA_ERR_NOT_SUPPORTED ? "UNSUPPORTED" : (ret == 0 && e <= 1e-12) ? "OK" : "WRONG"; printf("r67,%s,%s,%s,%lld,%.3g,%s\n", f, m, step, ret, e, v); }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
  if (argc < 3) { printf("usage: round67_formats_complex <FORMAT|TRANS> <matrix>\n"); return 0; }
  std::map<std::string, magma_storage_t> F = {{"ELLPACKT", Magma_ELLPACKT}, {"ELL", Magma_ELL}, {"DENSE", Magma_DENSE}, {"BCSR", Magma_BCSR}, {"CSC", Magma_CSC}, {"COO", Magma_COO}, {"ELLRT", Magma_ELLRT}, {"SELLP", Magma_SELLP},
    {"ELLD", Magma_ELLD}, {"CSRLIST", Magma_CSRLIST}, {"CSRD", Magma_CSRD}, {"CSRL", Magma_CSRL}, {"CSRU", Magma_CSRU}, {"CSRCOO", Magma_CSRCOO}, {"CUCSR", Magma_CUCSR}, {"CSR5", Magma_CSR5}};
  std::string fn = argv[1], mn = argv[2]; magma_init(); magma_queue_create(0, &Q);
  Dn D = mat(mn); magma_z_matrix A = toCsr(D);
  if (fn == "TRANS") {
    Dn T, H; T.m = H.m = D.n; T.n = H.n = D.m; T.a.assign((size_t)D.m * D.n, 0.0); H.a = T.a;
    for (int j = 0; j < D.n; j++) for (int i = 0; i < D.m; i++) { T.a[j + (size_t)i * D.n] = D.a[i + (size_t)j * D.m]; H.a[j + (size_t)i * D.n] = std::conj(D.a[i + (size_t)j * D.m]); }
    magma_z_matrix At = {Magma_CSR}, Ah = {Magma_CSR}; magma_int_t r = magma_zmtranspose(A, &At, Q); out("TRANS", mn.c_str(), "host_T", r, r == 0 ? diff(fromCsr(At, D.n, D.m), T) : 0);
    r = magma_zmtransposeconjugate(A, &Ah, Q); out("TRANS", mn.c_str(), "host_H", r, r == 0 ? diff(fromCsr(Ah, D.n, D.m), H) : 0);
    magma_z_matrix dA = {Magma_CSR}, dT = {Magma_CSR}, dH = {Magma_CSR}, hT = {Magma_CSR}, hH = {Magma_CSR}; magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q);
    r = magma_zmtranspose(dA, &dT, Q); if (r == 0) { magma_zmtransfer(dT, &hT, Magma_DEV, Magma_CPU, Q); } out("TRANS", mn.c_str(), "dev_T", r, r == 0 ? diff(fromCsr(hT, D.n, D.m), T) : 0);
    r = magma_zmtransposeconjugate(dA, &dH, Q); if (r == 0) { magma_zmtransfer(dH, &hH, Magma_DEV, Magma_CPU, Q); } out("TRANS", mn.c_str(), "dev_H", r, r == 0 ? diff(fromCsr(hH, D.n, D.m), H) : 0);
    fprintf(stderr, "END\n"); magma_queue_destroy(Q); magma_finalize(); return 0; }
  if (!F.count(fn)) { printf("unknown format\n"); return 0; }
  magma_storage_t fmt = F[fn]; Dn ref = D;
  if (fn == "CSRL") for (int j = 0; j < D.n; j++) for (int i = 0; i < D.m; i++) if (i < j) ref.at(i, j) = 0.0;
  if (fn == "CSRU") for (int j = 0; j < D.n; j++) for (int i = 0; i < D.m; i++) if (i > j) ref.at(i, j) = 0.0;
  magma_z_matrix B = {Magma_CSR}, Cm = {Magma_CSR}, dB = {Magma_CSR}, dA = {Magma_CSR}, dF = {Magma_CSR}, dC = {Magma_CSR}, hC = {Magma_CSR};
  B.blocksize = 8; B.alignment = 8; dF.blocksize = 8; dF.alignment = 8;
  fprintf(stderr, "STEP 1\n"); magma_int_t r = magma_zmconvert(A, &B, Magma_CSR, fmt, Q); out(fn.c_str(), mn.c_str(), "1", r, 0);
  if (r == 0) {
    fprintf(stderr, "STEP 2\n"); magma_int_t r2 = magma_zmconvert(B, &Cm, fmt, Magma_CSR, Q); out(fn.c_str(), mn.c_str(), "2", r2, r2 == 0 ? diff(fromCsr(Cm, D.m, D.n), ref) : 0);
    if (fn != "CSRL" && fn != "CSRU") {
      fprintf(stderr, "STEP 3\n"); magma_int_t r3 = magma_zmtransfer(B, &dB, Magma_CPU, Magma_DEV, Q);
      magma_z_matrix x = {Magma_CSR}, y = {Magma_CSR}, hy = {Magma_CSR}; std::vector<magmaDoubleComplex> xv(D.n); for (int j = 0; j < D.n; j++) xv[j] = MAGMA_Z_MAKE(1.0 + 0.1 * j, 0.5 - 0.07 * j);
      magma_zvset(D.n, 1, xv.data(), &hy, Q); magma_zmtransfer(hy, &x, Magma_CPU, Magma_DEV, Q); magma_zvinit(&y, Magma_DEV, D.m, 1, MAGMA_Z_MAKE(7, 3), Q);
      if (r3 == 0) r3 = magma_z_spmv(MAGMA_Z_ONE, dB, x, MAGMA_Z_ZERO, y, Q); magma_queue_sync(Q);
      double e3 = 0; if (r3 == 0) { magma_z_matrix gy = {Magma_CSR}; magma_zmtransfer(y, &gy, Magma_DEV, Magma_CPU, Q);
        for (int i = 0; i < D.m; i++) { cd s = 0; for (int j = 0; j < D.n; j++) s += D.a[i + (size_t)j * D.m] * C(xv[j]); double d = std::abs(C(gy.val[i]) - s); if (!(d <= e3)) e3 = d; }
        magma_zmfree(&gy, Q); }
      out(fn.c_str(), mn.c_str(), "3", r3, e3); }
  }
  fprintf(stderr, "STEP 4\n"); magma_int_t r4 = magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q); if (r4 == 0) r4 = magma_zmconvert(dA, &dF, Magma_CSR, fmt, Q);
  double e4 = 0; if (r4 == 0) { magma_int_t rb = magma_zmconvert(dF, &dC, fmt, Magma_CSR, Q); if (rb == 0) { magma_zmtransfer(dC, &hC, Magma_DEV, Magma_CPU, Q); e4 = diff(fromCsr(hC, D.m, D.n), ref); } else r4 = rb; }
  out(fn.c_str(), mn.c_str(), "4", r4, e4);
  fprintf(stderr, "END\n"); magma_queue_destroy(Q); magma_finalize(); return 0;
}

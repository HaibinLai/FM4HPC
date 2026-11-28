// Round 49 (roxy): the sparse findings M251-M254 in single precision (magma_s* API; generated from round47_dsparse.cpp).  One check per process.
//   round47_dsparse coo      M252  CSR -> COO -> CSR on [1 0 2; 0 3 0; 4 0 5] (host)
//   round47_dsparse csrlist  M254  CSR -> CSRLIST -> CSR with an empty row 0 (heap overflow)
//   round47_dsparse ellrt    M253  ELLRT SpMV on a row with 40 entries (alignment 8)
//   round47_dsparse sellp    M251  device CSR -> SELL-P with blocksize / alignment set on the target
// Prints RESULT lines; a crash shows as a missing RESULT (exit code reported by the caller).
#include "magma_v2.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
static magma_queue_t Q;
static magma_s_matrix csr(int m, int n, const std::vector<double>& D) { std::vector<magma_index_t> row(1, 0), col; std::vector<float> val;
  for (int i = 0; i < m; i++) { for (int j = 0; j < n; j++) if (D[i + (size_t)j * m] != 0) { col.push_back(j); val.push_back(D[i + (size_t)j * m]); } row.push_back((magma_index_t)col.size()); }
  if (col.empty()) { col.push_back(0); val.push_back(0); }
  magma_index_t* r = (magma_index_t*)malloc(row.size() * sizeof(magma_index_t)); magma_index_t* c = (magma_index_t*)malloc(col.size() * sizeof(magma_index_t)); float* v = (float*)malloc(val.size() * sizeof(float));
  memcpy(r, row.data(), row.size() * sizeof(magma_index_t)); memcpy(c, col.data(), col.size() * sizeof(magma_index_t)); memcpy(v, val.data(), val.size() * sizeof(float));
  magma_s_matrix A = {Magma_CSR}; magma_scsrset(m, n, r, c, v, &A, Q); A.nnz = row.back(); return A; }
static double diffCsr(const magma_s_matrix& C, int m, int n, const std::vector<double>& D) { std::vector<double> G((size_t)m * n, 0);
  for (int i = 0; i < m; i++) for (magma_index_t k = C.row[i]; k < C.row[i + 1]; k++) G[i + (size_t)C.col[k] * m] += C.val[k];
  double e = 0; for (size_t i = 0; i < G.size(); i++) e = fmax(e, fabs(G[i] - D[i])); return e; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string g = argc > 1 ? argv[1] : "coo"; magma_init(); magma_queue_create(0, &Q);
  if (g == "coo") { std::vector<double> D = {1, 0, 4, 0, 3, 0, 2, 0, 5}; magma_s_matrix A = csr(3, 3, D), B = {Magma_CSR}, C = {Magma_CSR};
    magma_int_t e1 = magma_smconvert(A, &B, Magma_CSR, Magma_COO, Q), e2 = magma_smconvert(B, &C, Magma_COO, Magma_CSR, Q);
    printf("RESULT M252 dmconvert CSR->COO->CSR ret=%lld,%lld rowptr=[%d %d %d %d] maxerr=%g %s\n", (long long)e1, (long long)e2, (int)C.row[0], (int)C.row[1], (int)C.row[2], (int)C.row[3], diffCsr(C, 3, 3, D), diffCsr(C, 3, 3, D) > 0 ? "REPRODUCED" : "NOT"); }
  if (g == "csrlist") { int n = 20; std::vector<double> D((size_t)n * n, 0); for (int i = 1; i < n; i++) { D[i + (size_t)i * n] = 2; if (i > 1) D[i + (size_t)(i - 1) * n] = 0.5; }
    magma_s_matrix A = csr(n, n, D), B = {Magma_CSR}, C = {Magma_CSR}; magma_int_t e1 = magma_smconvert(A, &B, Magma_CSR, Magma_CSRLIST, Q); printf("converted to CSRLIST ret=%lld\n", (long long)e1);
    magma_int_t e2 = magma_smconvert(B, &C, Magma_CSRLIST, Magma_CSR, Q); double e = diffCsr(C, n, n, D); printf("RESULT M254 CSRLIST round trip with empty row 0: ret=%lld nnz_out=%d (in %d) maxerr=%g %s\n", (long long)e2, (int)C.row[n], (int)A.nnz, e, e > 0 ? "REPRODUCED" : "NOT");
    magma_smfree(&C, Q); magma_smfree(&B, Q); printf("freed\n"); }
  if (g == "ellrt") { int n = 40; std::vector<double> D((size_t)n * n, 0); for (int i = 0; i < n; i++) D[i + (size_t)i * n] = 3; for (int j = 0; j < n; j++) D[5 + (size_t)j * n] = 0.5 + 0.01 * j;
    magma_s_matrix A = csr(n, n, D), B = {Magma_CSR}, dB = {Magma_CSR}, x = {Magma_CSR}, y = {Magma_CSR}, hx = {Magma_CSR}, gy = {Magma_CSR}; B.blocksize = 8; B.alignment = 8;
    magma_smconvert(A, &B, Magma_CSR, Magma_ELLRT, Q); magma_smtransfer(B, &dB, Magma_CPU, Magma_DEV, Q); std::vector<float> xv(n); for (int j = 0; j < n; j++) xv[j] = 1 + 0.1 * j;
    magma_svset(n, 1, xv.data(), &hx, Q); magma_smtransfer(hx, &x, Magma_CPU, Magma_DEV, Q); magma_svinit(&y, Magma_DEV, n, 1, 0.0f, Q); magma_int_t e = magma_s_spmv(1.0f, dB, x, 0.0f, y, Q); magma_queue_sync(Q);
    magma_smtransfer(y, &gy, Magma_DEV, Magma_CPU, Q); double ref = 0; for (int j = 0; j < n; j++) ref += D[5 + (size_t)j * n] * xv[j];
    printf("RESULT M253 sgeellrtmv row 5: got %.6g expected %.6g ret=%lld %s\n", gy.val[5], ref, (long long)e, fabs(gy.val[5] - ref) > 1e-3 ? "REPRODUCED" : "NOT"); }
  if (g == "sellp") { std::vector<double> D = {1, 0, 4, 0, 3, 0, 2, 0, 5}; magma_s_matrix A = csr(3, 3, D), dA = {Magma_CSR}, dS = {Magma_CSR}; magma_smtransfer(A, &dA, Magma_CPU, Magma_DEV, Q); dS.blocksize = 8; dS.alignment = 8;
    printf("converting device CSR -> SELLP ...\n"); magma_int_t e = magma_smconvert(dA, &dS, Magma_CSR, Magma_SELLP, Q); printf("RESULT M251 dmconvert(device CSR->SELLP) returned %lld NOT\n", (long long)e); }
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

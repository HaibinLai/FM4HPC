// Second GPU round for the dense audit (HANDOFF §29.1 items 6, 7, 12, 13, 14) and the sparse ICC / LOBPCG items.
// Modes: vb (potrf_vbatched Upper + stale info), getri, gegqr, hgetrf, icc, lobpcg.   GPU 0 only.
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
static magma_queue_t Q;
typedef magmaDoubleComplex Z;
static Z R(double x) { return MAGMA_Z_MAKE(x, 0); }
static unsigned S = 777u; static double rnd() { S = S * 1664525u + 1013904223u; return ((S >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static int nonfin(const Z* v, size_t k) { int c = 0; for (size_t i = 0; i < k; i++) if (!std::isfinite(MAGMA_Z_REAL(v[i]))) c++; return c; }

static void spd(int n, std::vector<Z>& A, bool garbageLower) {      // SPD tridiagonal-plus; optional garbage below
  A.assign((size_t)n * n, R(0)); for (int i = 0; i < n; i++) A[i + (size_t)i * n] = R(4);
  for (int i = 0; i + 1 < n; i++) { A[i + (size_t)(i + 1) * n] = R(-1); A[i + 1 + (size_t)i * n] = R(-1); }
  if (garbageLower) for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) A[i + (size_t)j * n] = R(100 + i);   // "not referenced"
}
// residual of the Upper factor: max |A_upper - U^H U| over the upper triangle
static double upperResid(int n, const std::vector<Z>& A, const std::vector<Z>& F) {
  double e = 0; for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { double s = 0;
    for (int k = 0; k <= i; k++) s += MAGMA_Z_REAL(F[k + (size_t)i * n]) * MAGMA_Z_REAL(F[k + (size_t)j * n]);
    e = fmax(e, fabs(s - MAGMA_Z_REAL(A[i + (size_t)j * n]))); } return e; }

static void vb() {
  printf("== potrf_vbatched ==\n");
  int sizes[2] = {40, 300}; int batch = 2;
  for (int stale : {0, 7}) for (magma_uplo_t up : {MagmaUpper, MagmaLower}) {
    if (stale && up == MagmaUpper) continue;
    std::vector<std::vector<Z>> A(batch); std::vector<Z*> dA(batch); std::vector<magma_int_t> hn(batch + 1), hld(batch + 1), hinfo(batch, stale);
    for (int b = 0; b < batch; b++) { int n = sizes[b]; spd(n, A[b], up == MagmaUpper); magma_zmalloc(&dA[b], (size_t)n * n);
      magma_zsetmatrix(n, n, A[b].data(), n, dA[b], n, Q); hn[b] = n; hld[b] = n; }
    magma_int_t *dn, *dld, *dinfo; Z** dAarr; magma_imalloc(&dn, batch + 1); magma_imalloc(&dld, batch + 1); magma_imalloc(&dinfo, batch);
    magma_malloc((void**)&dAarr, batch * sizeof(Z*));
    magma_setvector(batch, sizeof(magma_int_t), hn.data(), 1, dn, 1, Q); magma_setvector(batch, sizeof(magma_int_t), hld.data(), 1, dld, 1, Q);
    magma_setvector(batch, sizeof(magma_int_t), hinfo.data(), 1, dinfo, 1, Q); magma_setvector(batch, sizeof(Z*), dA.data(), 1, dAarr, 1, Q);
    magma_int_t ret = magma_zpotrf_vbatched(up, dn, dAarr, dld, dinfo, batch, Q);
    magma_getvector(batch, sizeof(magma_int_t), dinfo, 1, hinfo.data(), 1, Q);
    for (int b = 0; b < batch; b++) { int n = sizes[b]; std::vector<Z> F((size_t)n * n); magma_zgetmatrix(n, n, dA[b], n, F.data(), n, Q);
      bool changed = false; for (size_t i = 0; i < F.size(); i++) if (MAGMA_Z_REAL(F[i]) != MAGMA_Z_REAL(A[b][i])) { changed = true; break; }
      double ur = up == MagmaUpper ? upperResid(n, A[b], F) : -1;
      printf("uplo=%s stale_info_in=%d  matrix n=%3d: ret=%lld info=%lld  modified=%d  %s\n", up == MagmaUpper ? "Upper" : "Lower", stale, n,
             (long long)ret, (long long)hinfo[b], (int)changed,
             up == MagmaUpper ? (std::string("max|A - U^H U| (upper) = ") + std::to_string(ur)).c_str() : "");
      magma_free(dA[b]); }
    magma_free(dn); magma_free(dld); magma_free(dinfo); magma_free(dAarr);
  }
}

static void getri() {
  printf("== getri_outofplace_batched on an exactly singular matrix ==\n");
  int n = 32, batch = 1; std::vector<Z> A((size_t)n * n); for (auto& a : A) a = R(rnd());
  for (int i = 0; i < n; i++) A[i + (size_t)5 * n] = A[i + (size_t)4 * n];   // column 5 = column 4
  Z *dA, *dI; magma_int_t* dpiv; magma_int_t* dinfo; Z** dAarr; Z** dIarr; magma_int_t** dParr;
  magma_zmalloc(&dA, (size_t)n * n); magma_zmalloc(&dI, (size_t)n * n); magma_imalloc(&dpiv, n); magma_imalloc(&dinfo, batch);
  magma_malloc((void**)&dAarr, sizeof(Z*)); magma_malloc((void**)&dIarr, sizeof(Z*)); magma_malloc((void**)&dParr, sizeof(magma_int_t*));
  magma_setvector(1, sizeof(Z*), &dA, 1, dAarr, 1, Q); magma_setvector(1, sizeof(Z*), &dI, 1, dIarr, 1, Q); magma_setvector(1, sizeof(magma_int_t*), &dpiv, 1, dParr, 1, Q);
  magma_zsetmatrix(n, n, A.data(), n, dA, n, Q);
  magma_int_t r1 = magma_zgetrf_batched(n, n, dAarr, n, dParr, dinfo, batch, Q); magma_int_t i1; magma_getvector(1, sizeof(magma_int_t), dinfo, 1, &i1, 1, Q);
  magma_int_t sentinel = 12345; magma_setvector(1, sizeof(magma_int_t), &sentinel, 1, dinfo, 1, Q);
  magma_int_t r2 = magma_zgetri_outofplace_batched(n, dAarr, n, dParr, dIarr, n, dinfo, batch, Q); magma_int_t i2; magma_getvector(1, sizeof(magma_int_t), dinfo, 1, &i2, 1, Q);
  std::vector<Z> I((size_t)n * n); magma_zgetmatrix(n, n, dI, n, I.data(), n, Q);
  printf("getrf_batched ret=%lld info=%lld | info_array set to %lld | getri_outofplace ret=%lld info_array after=%lld | nonfinite entries of inverse=%d\n",
         (long long)r1, (long long)i1, (long long)sentinel, (long long)r2, (long long)i2, nonfin(I.data(), I.size()));
}

static void gegqr() {
  printf("== gegqr_gpu: rank 0 reported as info = 0 ==\n");
  int m = 1000, n = 8;
  for (int ikind : {1, 3, 4}) for (double scale : {1.0, 1e-20}) {
    std::vector<Z> A((size_t)m * n); S = 99u; for (auto& a : A) a = R(scale * rnd());
    Z *dA, *dwork; magma_zmalloc(&dA, (size_t)m * n); magma_zmalloc(&dwork, 3 * n * n + n + 2);
    std::vector<Z> work(5 * n * n + 7 * n + 64); magma_zsetmatrix(m, n, A.data(), m, dA, m, Q);
    magma_int_t info = -999; magma_zgegqr_gpu(ikind, m, n, dA, m, dwork, work.data(), &info);
    std::vector<Z> Qm((size_t)m * n); magma_zgetmatrix(m, n, dA, m, Qm.data(), m, Q);
    double orth = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { double s = 0;
      for (int k = 0; k < m; k++) s += MAGMA_Z_REAL(Qm[k + (size_t)i * m]) * MAGMA_Z_REAL(Qm[k + (size_t)j * m]); orth = fmax(orth, fabs(s - (i == j))); }
    printf("ikind=%d scale=%-6g info=%lld  ||Q^T Q - I||_max=%.3g  nonfinite=%d\n", ikind, scale, (long long)info, orth, nonfin(Qm.data(), Qm.size()));
    magma_free(dA); magma_free(dwork);
  }
}

static void hgetrf() {
  printf("== hgetrf_gpu (FP16-TC LU): ldda = n (unaligned) vs ldda = roundup(n,32) ==\n");
  for (int n : {1000, 1024}) {
    std::vector<float> A((size_t)n * n); S = 5u; for (auto& a : A) a = (float)rnd(); for (int i = 0; i < n; i++) A[i + (size_t)i * n] += 4.0f;
    std::vector<std::vector<float>> F(2); std::vector<std::vector<magma_int_t>> P(2); magma_int_t info[2];
    int lds[2] = {n, (n + 31) / 32 * 32};
    for (int v = 0; v < 2; v++) { int ld = lds[v]; float* dA; magma_smalloc(&dA, (size_t)ld * n);
      magma_ssetmatrix(n, n, A.data(), n, dA, ld, Q); P[v].resize(n); info[v] = -999;
      magma_hgetrf_gpu(n, n, dA, ld, P[v].data(), &info[v]);
      F[v].resize((size_t)n * n); magma_sgetmatrix(n, n, dA, ld, F[v].data(), n, Q); magma_free(dA); }
    double d = 0; long nd = 0; for (size_t i = 0; i < F[0].size(); i++) { double e = fabs(F[0][i] - F[1][i]); if (e > 0) nd++; d = fmax(d, e); }
    long pd = 0; for (int i = 0; i < n; i++) if (P[0][i] != P[1][i]) pd++;
    printf("n=%d: info(ld=%d)=%lld info(ld=%d)=%lld | factor entries differing=%ld (max |diff|=%.3g) | pivots differing=%ld\n",
           n, lds[0], (long long)info[0], lds[1], (long long)info[1], nd, d, pd);
  }
}

// ---- sparse ----
static void csr_dev(int n, const std::vector<double>& A, magma_z_matrix* dA) {
  std::vector<magma_index_t> row(n + 1), col; std::vector<Z> val;
  for (int i = 0; i < n; i++) { row[i] = (magma_index_t)col.size(); for (int j = 0; j < n; j++) if (A[i * n + j] != 0) { col.push_back(j); val.push_back(R(A[i * n + j])); } }
  row[n] = (magma_index_t)col.size();
  magma_z_matrix h = {Magma_CSR}; magma_zcsrset(n, n, row.data(), col.data(), val.data(), &h, Q); magma_zmtransfer(h, dA, Magma_CPU, Magma_DEV, Q);
}
static void icc() {
  printf("== ICC (IC(0)) setup + PCG on an SPD matrix whose IC(0) breaks down (pivot 6 negative) ==\n");
  int n = 6; std::vector<double> A(n * n); FILE* f = fopen("icc_breakdown_A.txt", "r"); for (auto& a : A) if (fscanf(f, "%lf", &a) != 1) a = 0; fclose(f);
  magma_z_matrix dA = {Magma_CSR}, db = {Magma_CSR}, dx = {Magma_CSR};
  csr_dev(n, A, &dA);
  std::vector<Z> b(n, R(1)), x0(n, R(0)); magma_z_matrix h = {Magma_CSR};
  magma_zvset(n, 1, b.data(), &h, Q); magma_zmtransfer(h, &db, Magma_CPU, Magma_DEV, Q);
  magma_zvset(n, 1, x0.data(), &h, Q); magma_zmtransfer(h, &dx, Magma_CPU, Magma_DEV, Q);
  magma_z_solver_par sp; magma_z_preconditioner pp; memset(&sp, 0, sizeof sp); memset(&pp, 0, sizeof pp);
  sp.solver = Magma_PCG; sp.maxiter = 100; sp.atol = 1e-16; sp.rtol = 1e-10; pp.solver = Magma_ICC; pp.levels = 0;
  magma_zsolverinfo_init(&sp, &pp, Q);
  fprintf(stderr, "[icc] setup...\n");
  magma_int_t si = magma_z_precondsetup(dA, db, &sp, &pp, Q);
  fprintf(stderr, "[icc] setup returned %lld; pcg...\n", (long long)si);
  magma_int_t ret = magma_zpcg(dA, db, &dx, &sp, &pp, Q);
  magma_z_matrix hx = {Magma_CSR}; magma_zmtransfer(dx, &hx, Magma_DEV, Magma_CPU, Q);
  printf("ICC setup info=%lld | PCG ret=%lld info=%lld numiter=%lld final_res=%g | x[0]=%g\n", (long long)si, (long long)ret, (long long)sp.info,
         (long long)sp.numiter, sp.final_res, MAGMA_Z_REAL(hx.val[0]));
}
static void lobpcg() {
  printf("== LOBPCG with more eigenvectors than rows (n_ev = 8 > m = 6): parameter check is dead code ==\n");
  int n = 6; std::vector<double> A(n * n, 0); for (int i = 0; i < n; i++) A[i * n + i] = i + 1;
  magma_z_matrix dA = {Magma_CSR}; csr_dev(n, A, &dA);
  magma_z_solver_par sp; magma_z_preconditioner pp; memset(&sp, 0, sizeof sp); memset(&pp, 0, sizeof pp);
  sp.solver = Magma_LOBPCG; sp.maxiter = 50; sp.rtol = 1e-10; sp.num_eigenvalues = 8; pp.solver = Magma_NONE;
  magma_zsolverinfo_init(&sp, &pp, Q);
  sp.ev_length = n; magma_zeigensolverinfo_init(&sp, Q);
  fprintf(stderr, "[lobpcg] solve...\n");
  magma_int_t ret = magma_zlobpcg(dA, &sp, &pp, Q);
  printf("LOBPCG n_ev=8 m=6: ret=%lld info=%lld numiter=%lld\n", (long long)ret, (long long)sp.info, (long long)sp.numiter);
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  std::string m = argc > 1 ? argv[1] : "vb";
  magma_init(); magma_queue_create(0, &Q);
  if (m == "vb") vb(); else if (m == "getri") getri(); else if (m == "gegqr") gegqr(); else if (m == "hgetrf") hgetrf();
  else if (m == "icc") icc(); else if (m == "lobpcg") lobpcg();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

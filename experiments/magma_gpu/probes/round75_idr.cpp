// Round 75 (roxy): the IDR(s) solvers called directly (the tester maps IDR and PIDR to PIDRMERGE, so magma_zidr, magma_zpidr,
// magma_zidr_strms and magma_zpidr_strms were never run), and the sparse QR wrapper magma_zqr that they use.
//   round75_idr <idr|pidr|idrs|pidrs|idrm|pidrm> <gen|rot0|rot45|rot90> <n> <s>
//       gen:   random complex CSR, about 5 entries per row, diagonal 4 + random imaginary part
//       rotT:  e^{iT deg} S, S = tridiag(-1, 2.1, -1) (Hermitian positive definite, eigenvalues in [0.1, 4.1])
//       b = A x_true, x0 = 0, rtol 1e-10, maxiter 2n, restart = s; pidr*: preconditioner Magma_NONE
//       CSV: r75,solver,mat,n,s,info,numiter,reported_res/|b|,true_res/|b|,verdict
//   round75_idr qr <m> <n> <cpu|dev> <ldpad>   magma_zqr on a dense m x n block with lda = m + ldpad
//       CSV: r75q,m,n,loc,ldpad,info,|Q^H Q - I|,|Q R - A|/|A|,lower part of R,A changed (dev),verdict
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
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
  if (argc < 5) return 1; std::string op = argv[1]; magma_init(); magma_queue_create(0, &Q);
  if (op == "qr") {
    int m = atoi(argv[2]), n = atoi(argv[3]); bool dev = std::string(argv[4]) == "dev"; int ld = m + atoi(argv[5]);
    std::vector<magmaDoubleComplex> A((size_t)ld * n, MAGMA_Z_MAKE(77, 77)); for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) A[i + (size_t)j * ld] = MAGMA_Z_MAKE(r1(), r1());
    magma_z_matrix hA = {Magma_CSR}, dA = {Magma_CSR}, Qm = {Magma_CSR}, Rm = {Magma_CSR}, hQ = {Magma_CSR}, hR = {Magma_CSR}, hA2 = {Magma_CSR};
    magma_zvset(ld, n, A.data(), &hA, Q); if (dev) magma_zmtransfer(hA, &dA, Magma_CPU, Magma_DEV, Q);
    magma_int_t info = magma_zqr(m, n, dev ? dA : hA, ld, &Qm, &Rm, Q); magma_queue_sync(Q);
    double eo = -1, er = -1, el = -1, ech = -1; int k = std::min(m, n);
    if (info == 0) {
      if (dev) { magma_zmtransfer(Qm, &hQ, Magma_DEV, Magma_CPU, Q); magma_zmtransfer(Rm, &hR, Magma_DEV, Magma_CPU, Q); magma_zmtransfer(dA, &hA2, Magma_DEV, Magma_CPU, Q);
        ech = 0; for (size_t i = 0; i < A.size(); i++) ech = std::max(ech, std::abs(C(hA2.val[i]) - C(A[i]))); }
      else { hQ = Qm; hR = Rm; }
      auto q = [&](int i, int j) { return C(hQ.val[i + (size_t)j * ld]); }; auto r = [&](int i, int j) { return C(hR.val[i + (size_t)j * ld]); };
      int qc = std::min(n, m); eo = 0; for (int a = 0; a < qc; a++) for (int b = 0; b < qc; b++) { cd s = 0; for (int i = 0; i < m; i++) s += std::conj(q(i, a)) * q(i, b); eo = std::max(eo, std::abs(s - (a == b ? 1.0 : 0.0))); }
      double an = 0; er = 0; for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { cd s = 0; for (int l = 0; l < k; l++) s += q(i, l) * r(l, j); an = std::max(an, std::abs(C(A[i + (size_t)j * ld]))); er = std::max(er, std::abs(s - C(A[i + (size_t)j * ld]))); }
      er /= an; el = 0; for (int j = 0; j < n; j++) for (int i = j + 1; i < m; i++) el = std::max(el, std::abs(r(i, j))); }
    bool ok = info == 0 && eo < 1e-12 && er < 1e-12 && el == 0;
    printf("r75q,%d,%d,%s,%s,%lld,%.3g,%.3g,%.3g,%.3g,%s\n", m, n, argv[4], argv[5], (long long)info, eo, er, el, ech, ok ? "OK" : "WRONG");
  } else {
    std::string mat = argv[2]; int n = atoi(argv[3]), s = atoi(argv[4]);
    std::vector<magma_index_t> row(1, 0), col; std::vector<magmaDoubleComplex> val; std::vector<std::vector<std::pair<int, cd>>> R(n);
    if (mat == "gen") { for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) if (i == j || (r1() + 0.5) < 5.0 / n) R[i].push_back({j, i == j ? cd(4, r1()) : cd(r1(), r1())}); } }
    else { double th = atof(mat.c_str() + 3) * 3.14159265358979323846 / 180; cd e(cos(th), sin(th)); for (int i = 0; i < n; i++) { if (i) R[i].push_back({i - 1, -e}); R[i].push_back({i, 2.1 * e}); if (i + 1 < n) R[i].push_back({i + 1, -e}); } }
    for (int i = 0; i < n; i++) { for (auto& p : R[i]) { col.push_back(p.first); val.push_back(Z(p.second)); } row.push_back((magma_index_t)col.size()); }
    magma_index_t* rp = (magma_index_t*)malloc(row.size() * sizeof(magma_index_t)); magma_index_t* cp = (magma_index_t*)malloc(col.size() * sizeof(magma_index_t)); magmaDoubleComplex* vp = (magmaDoubleComplex*)malloc(val.size() * sizeof(magmaDoubleComplex));
    memcpy(rp, row.data(), row.size() * sizeof(magma_index_t)); memcpy(cp, col.data(), col.size() * sizeof(magma_index_t)); memcpy(vp, val.data(), val.size() * sizeof(magmaDoubleComplex));
    magma_z_matrix A = {Magma_CSR}, dA = {Magma_CSR}; magma_zcsrset(n, n, rp, cp, vp, &A, Q); magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, Q);
    std::vector<cd> xt(n), bh(n, 0.0); for (auto& v : xt) v = cd(r1(), r1()); for (int i = 0; i < n; i++) for (auto& p : R[i]) bh[i] += p.second * xt[p.first];
    std::vector<magmaDoubleComplex> bz(n); for (int i = 0; i < n; i++) bz[i] = Z(bh[i]); magma_z_matrix hb = {Magma_CSR}, db = {Magma_CSR}, dx = {Magma_CSR}, hx = {Magma_CSR};
    magma_zvset(n, 1, bz.data(), &hb, Q); magma_zmtransfer(hb, &db, Magma_CPU, Magma_DEV, Q); magma_zvinit(&dx, Magma_DEV, n, 1, MAGMA_Z_ZERO, Q);
    magma_z_solver_par sp; memset(&sp, 0, sizeof(sp)); sp.rtol = 1e-10; sp.atol = 0; sp.maxiter = 2 * n; sp.restart = s; sp.verbose = 0;
    magma_z_preconditioner pp; memset(&pp, 0, sizeof(pp)); pp.solver = Magma_NONE;
    std::string so = op; magma_int_t info;
    if (so == "idr") info = magma_zidr(dA, db, &dx, &sp, Q); else if (so == "pidr") info = magma_zpidr(dA, db, &dx, &sp, &pp, Q);
    else if (so == "idrs") info = magma_zidr_strms(dA, db, &dx, &sp, Q); else if (so == "pidrs") info = magma_zpidr_strms(dA, db, &dx, &sp, &pp, Q);
    else if (so == "idrm") info = magma_zidr_merge(dA, db, &dx, &sp, Q); else info = magma_zpidr_merge(dA, db, &dx, &sp, &pp, Q);
    magma_queue_sync(Q); magma_zmtransfer(dx, &hx, Magma_DEV, Magma_CPU, Q);
    double nb = 0, nr = 0; for (int i = 0; i < n; i++) { cd t = 0; for (auto& p : R[i]) t += p.second * C(hx.val[p.first]); nr += std::norm(bh[i] - t); nb += std::norm(bh[i]); }
    double tr = sqrt(nr / nb), rr = sp.final_res / sqrt(nb);
    printf("r75,%s,%s,%d,%d,%lld,%lld,%.3g,%.3g,%s\n", so.c_str(), mat.c_str(), n, s, (long long)info, (long long)sp.numiter, rr, tr, info == 0 && tr < 1e-8 ? "OK" : "WRONG");
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(Q); magma_finalize(); return 0;
}

// Round 69 (roxy): magmablas_zlange / magmablas_zlanhe / magma_zgeqr2_gpu with dwork at the documented size + sentinel guard.
//   round69_norms lange <m> <n> <M|O|I|F>        lwork: M or I -> m, O -> n (documented), F -> max(m, n) (not documented)
//   round69_norms lanhe <n> <M|O|I|F> <U|L>     lwork n (documented "LWORK >= N", max-norm included)
//   R69_DIAGIMAG (lanhe): diagonal imaginary parts set to 1e3, which the documentation says are assumed to be zero
//   round69_norms geqr2 <m> <n>                 dwork n doubles (documented "dimension (N)"); tau and R against LAPACK zgeqrf
// CSV: r69,op,m,n,opt,uplo,value,ref,rel,guard,verdict
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
  if (argc < 4) return 1; std::string op = argv[1]; magma_init(); magma_queue_t q; magma_queue_create(0, &q);
  const size_t G = 1 << 14; auto guard_alloc = [&](size_t doc) { double* d; magma_dmalloc(&d, doc + G); std::vector<double> s(doc + G, 555.5); magma_dsetvector((magma_int_t)(doc + G), s.data(), 1, d, 1, q); return d; };
  auto guard_count = [&](double* d, size_t doc) { std::vector<double> s(G); magma_dgetvector((magma_int_t)G, d + doc, 1, s.data(), 1, q); size_t w = 0; for (double x : s) if (x != 555.5) w++; return w; };
  if (op == "lange" || op == "lanhe") {
    bool he = op == "lanhe"; magma_int_t m = atoi(argv[2]), n = he ? m : atoi(argv[3]); char nc = argv[he ? 3 : 4][0], up = he && argc > 4 ? argv[4][0] : 'L';
    magma_norm_t norm = nc == 'M' ? MagmaMaxNorm : nc == 'O' ? MagmaOneNorm : nc == 'I' ? MagmaInfNorm : MagmaFrobeniusNorm;
    magma_int_t ld = magma_roundup(std::max<magma_int_t>(m, 1), 32); std::vector<Z> A((size_t)m * n);
    for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i < m; i++) A[i + j * m] = MAGMA_Z_MAKE(r1() * (1 + i % 3), r1() * (1 + j % 5));
    if (he) for (magma_int_t j = 0; j < n; j++) { A[j + j * m] = MAGMA_Z_MAKE(MAGMA_Z_REAL(A[j + j * m]), 0); for (magma_int_t i = j + 1; i < m; i++) A[j + i * m] = MAGMA_Z_CONJ(A[i + j * m]); }
    std::vector<Z> Ad = A; if (he) for (magma_int_t j = 0; j < n; j++) for (magma_int_t i = 0; i < m; i++) if ((up == 'L' && i < j) || (up == 'U' && i > j)) Ad[i + j * m] = MAGMA_Z_MAKE(1e6, 1e6);   // other triangle must not be read
    if (he && getenv("R69_DIAGIMAG")) for (magma_int_t j = 0; j < n; j++) Ad[j + j * m] = MAGMA_Z_MAKE(MAGMA_Z_REAL(Ad[j + j * m]), 1e3);   // doc: imaginary parts of the diagonal need not be set
    Z* dA; magma_zmalloc(&dA, (size_t)ld * n); magma_zsetmatrix(m, n, Ad.data(), m, dA, ld, q);
    size_t doc = he ? n : (nc == 'O' ? n : nc == 'F' ? std::max(m, n) : m); double* dw = guard_alloc(std::max<size_t>(doc, 1));
    double val = he ? magmablas_zlanhe(norm, up == 'U' ? MagmaUpper : MagmaLower, n, dA, ld, dw, (magma_int_t)std::max<size_t>(doc, 1), q)
                    : magmablas_zlange(norm, m, n, dA, ld, dw, (magma_int_t)std::max<size_t>(doc, 1), q);
    magma_queue_sync(q); size_t gw = guard_count(dw, std::max<size_t>(doc, 1));
    std::vector<double> wk(std::max(m, n) + 1); magma_int_t M = m, N = n; double ref = lapackf77_zlange(&nc, &M, &N, A.data(), &M, wk.data());
    double rel = ref != 0 ? fabs(val - ref) / ref : fabs(val);
    printf("r69,%s,%lld,%lld,%c,%c,%.15g,%.15g,%.3g,%zu,%s\n", op.c_str(), (long long)m, (long long)n, nc, up, val, ref, rel, gw, rel < 1e-12 && gw == 0 ? "OK" : "WRONG");
    magma_free(dA); magma_free(dw);
  } else {
    magma_int_t m = atoi(argv[2]), n = atoi(argv[3]), ld = magma_roundup(m, 32), mn = std::min(m, n), info = 0; std::vector<Z> A((size_t)m * n); for (auto& a : A) a = MAGMA_Z_MAKE(r1(), r1());
    Z *dA, *dt; magma_zmalloc(&dA, (size_t)ld * n); magma_zmalloc(&dt, std::max<magma_int_t>(mn, 1)); magma_zsetmatrix(m, n, A.data(), m, dA, ld, q); double* dw = guard_alloc(n);
    magma_zgeqr2_gpu(m, n, dA, ld, dt, dw, q, &info); magma_queue_sync(q); size_t gw = guard_count(dw, n);
    std::vector<Z> R = A, tr(mn), tau(mn), X((size_t)m * n); magma_int_t lw = -1, ri; Z wq; lapackf77_zgeqrf(&m, &n, R.data(), &m, tr.data(), &wq, &lw, &ri); lw = (magma_int_t)MAGMA_Z_REAL(wq); { std::vector<Z> w(lw); lapackf77_zgeqrf(&m, &n, R.data(), &m, tr.data(), w.data(), &lw, &ri); }
    magma_zgetvector(mn, dt, 1, tau.data(), 1, q); magma_zgetmatrix(m, n, dA, ld, X.data(), m, q); double e = 0, a = 0;
    for (int i = 0; i < mn; i++) e = std::max(e, MAGMA_Z_ABS(MAGMA_Z_SUB(tau[i], tr[i]))); for (size_t i = 0; i < X.size(); i++) { e = std::max(e, MAGMA_Z_ABS(MAGMA_Z_SUB(X[i], R[i]))); a = std::max(a, MAGMA_Z_ABS(R[i])); }
    printf("r69,geqr2,%lld,%lld,-,-,%lld,0,%.3g,%zu,%s\n", (long long)m, (long long)n, (long long)info, e / a, gw, info == 0 && e / a < 1e-10 && gw == 0 ? "OK" : "WRONG");
    magma_free(dA); magma_free(dt); magma_free(dw);
  }
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError())); magma_queue_destroy(q); magma_finalize(); return 0;
}

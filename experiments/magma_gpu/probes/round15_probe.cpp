// Fifteenth GPU round: findings of the per-file sweep over sparse/control and sparse/blas (HANDOFF §29.15).  GPU 0 only.
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magmasparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
typedef magmaDoubleComplex Z;
static magma_queue_t Q;
static Z C(double r, double i = 0) { return MAGMA_Z_MAKE(r, i); }
static Z* dev(const std::vector<Z>& v) { Z* d; magma_zmalloc(&d, std::max<size_t>(v.size(), 1)); magma_zsetvector(v.size(), v.data(), 1, d, 1, Q); return d; }
static magma_index_t* devi(const std::vector<magma_index_t>& v) { magma_index_t* d; magma_index_malloc(&d, v.size()); magma_index_setvector(v.size(), v.data(), 1, d, 1, Q); return d; }
static std::vector<Z> host(Z* d, int n) { std::vector<Z> v(n); magma_zgetvector(n, d, 1, v.data(), 1, Q); return v; }
static void pr(const char* t, const std::vector<Z>& v) { printf("  %s =", t); for (auto& z : v) printf(" (%g%+gi)", MAGMA_Z_REAL(z), MAGMA_Z_IMAG(z)); printf("\n"); }
static std::vector<std::vector<magma_index_t>> KR, KC; static std::vector<std::vector<Z>> KV;
static magma_z_matrix csr(int m, int n, std::vector<magma_index_t> r, std::vector<magma_index_t> c, std::vector<Z> v) { KR.push_back(r); KC.push_back(c); KV.push_back(v);
  magma_z_matrix h = {Magma_CSR}; magma_zcsrset(m, n, KR.back().data(), KC.back().data(), KV.back().data(), &h, Q); return h; }

static void csrmv_trans() {   // #18 transA ignored
  auto dv = dev({C(1)}); auto dr = devi({0, 1, 1}); auto dc = devi({1}); auto dx = dev({C(1), C(0)}); auto dy = dev({C(7), C(7)});
  magma_zgecsrmv(MagmaTrans, 2, 2, C(1), dv, dr, dc, dx, C(0), dy, Q); printf("#18 zgecsrmv(MagmaTrans), A=[[0,1],[0,0]], x=(1,0): expected A^T x = (0,1)\n"); pr("y", host(dy, 2)); }
static void elltmv_nan() {   // #21 padding multiplies x[0]
  magma_z_matrix h = csr(2, 2, {0, 2, 3}, {0, 1, 1}, {C(1), C(1), C(1)}), e = {Magma_CSR}, d = {Magma_CSR};
  magma_int_t c = magma_zmconvert(h, &e, Magma_CSR, Magma_ELLPACKT, Q); magma_zmtransfer(e, &d, Magma_CPU, Magma_DEV, Q);
  magma_z_matrix x = {Magma_CSR}, hx = {Magma_CSR}, y = {Magma_CSR}; std::vector<Z> xv = {C(INFINITY), C(1)}; magma_zvset(2, 1, xv.data(), &hx, Q); magma_zmtransfer(hx, &x, Magma_CPU, Magma_DEV, Q);
  magma_zvinit(&y, Magma_DEV, 2, 1, C(0), Q); magma_int_t r = magma_z_spmv(C(1), d, x, C(0), y, Q);
  printf("#21 ELLPACKT spmv, A=[[1,1],[0,1]], x=(Inf,1): convert=%lld ret=%lld; expected y = (Inf, 1)\n", (long long)c, (long long)r); pr("y", host(y.dval, 2)); }
static void mdotc() {        // #22/#27 unconjugated
  auto dv = dev({C(0, 1)}); auto dr = dev({C(0, 1)}); Z *d1, *d2; magma_zmalloc(&d1, 4096); magma_zmalloc(&d2, 4096); Z skp = C(-99);
  magma_zmdotc(1, 1, dv, dr, d1, d2, &skp, Q);
  printf("#22 zmdotc n=1, v=(i), r=(i): <v,r> = conj(i)*i = 1; got (%g%+gi)\n", MAGMA_Z_REAL(skp), MAGMA_Z_IMAG(skp)); }
static void mdot_shfl() {    // #26 reduction no-op
  int n = 64, k = 2; std::vector<Z> V(n * k, C(1)), R(n, C(1)); auto dv = dev(V); auto dr = dev(R); Z *d1, *d2; magma_zmalloc(&d1, 1 << 16); magma_zmalloc(&d2, 1 << 16);
  auto ds = dev({C(-99), C(-99)}); magma_zgemvmdot_shfl(n, k, dv, dr, d1, d2, ds, Q);
  printf("#26 zgemvmdot_shfl n=64, k=2, all ones: expected (64, 64)\n"); pr("skp", host(ds, 2)); }
static void mgecsrmv() {     // #28 y stride n
  int m = 2, n = 3, nv = 2;   // A = [[1,0,0],[0,1,0]], X = [x0 | x1] with x0 = (1,2,3), x1 = (4,5,6): AX = [(1,2) | (4,5)]
  auto dv = dev({C(1), C(1)}); auto dr = devi({0, 1, 2}); auto dc = devi({0, 1}); auto dx = dev({C(1), C(2), C(3), C(4), C(5), C(6)});
  auto dy = dev({C(7), C(7), C(7), C(7)}); magma_zmgecsrmv(MagmaNoTrans, m, n, nv, C(1), dv, dr, dc, dx, C(0), dy, Q);
  printf("#28 zmgecsrmv m=2 n=3 num_vecs=2: expected Y (column-major, ld 2) = 1 2 4 5\n"); pr("Y", host(dy, 4)); }
static void sellp_multi() {  // #31 second half of the vectors not computed
  magma_z_matrix h = csr(4, 4, {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {C(1), C(1), C(1), C(1)}), s = {Magma_CSR}, d = {Magma_CSR};
  s.blocksize = 4; s.alignment = 1; magma_int_t c = magma_zmconvert(h, &s, Magma_CSR, Magma_SELLP, Q); magma_zmtransfer(s, &d, Magma_CPU, Magma_DEV, Q);
  std::vector<Z> X(8); for (int i = 0; i < 8; i++) X[i] = C(i + 1); magma_z_matrix hx = {Magma_CSR}, x = {Magma_CSR}, y = {Magma_CSR};
  magma_zvset(4, 2, X.data(), &hx, Q); magma_zmtransfer(hx, &x, Magma_CPU, Magma_DEV, Q); magma_zvinit(&y, Magma_DEV, 4, 2, C(7), Q);
  magma_int_t r = magma_z_spmv(C(1), d, x, C(0), y, Q);
  printf("#31 SELLP 4x4 identity, X with 2 columns (1..8): convert=%lld ret=%lld; expected Y = X\n", (long long)c, (long long)r); pr("Y", host(y.dval, 8)); }
static void zsort_nan() { Z x[2] = {C(NAN), C(NAN)}; printf("#7 magma_zsort({NaN, NaN}, 0, 1) ...\n"); fflush(stdout); magma_int_t r = magma_zsort(x, 0, 1, Q); printf("  returned %lld\n", (long long)r); }
static void vread() { const char* f = "vec2col.txt"; FILE* fp = fopen(f, "w"); fprintf(fp, "1 0\n2 0\n"); fclose(fp);
  magma_z_matrix x = {Magma_CSR}; char fn[64]; strcpy(fn, f); magma_int_t r = magma_zvread(&x, 2, fn, Q);
  printf("#8 magma_zvread of the two-column file \"1 0 / 2 0\", length 2: ret=%lld num_rows=%lld", (long long)r, (long long)x.num_rows);
  if (x.val) { printf(" values:"); for (int i = 0; i < 2; i++) printf(" (%g%+gi)", MAGMA_Z_REAL(x.val[i]), MAGMA_Z_IMAG(x.val[i])); } printf(" (expected 1, 2)\n"); }
static void csrget() { magma_z_matrix h = csr(2, 2, {0, 1, 2}, {0, 1}, {C(3), C(4)}), d = {Magma_CSR}; magma_zmtransfer(h, &d, Magma_CPU, Magma_DEV, Q);
  magma_int_t m, n; magma_index_t *row, *col; Z* val; magma_int_t r = magma_zcsrget(d, &m, &n, &row, &col, &val, Q);
  void* junk = malloc(64); memset(junk, 0x5a, 64);   // reuse the freed chunk
  printf("#4 magma_zcsrget of a device CSR [[3,0],[0,4]]: ret=%lld m=%lld n=%lld; val[0], val[1] read from the returned pointer = (%g, %g) (expected 3, 4)\n",
         (long long)r, (long long)m, (long long)n, MAGMA_Z_REAL(val[0]), MAGMA_Z_REAL(val[1])); free(junk); }
static void compact() { auto dA = dev({C(1)}); double* dn; magma_dmalloc(&dn, 1); double nanv = NAN; magma_dsetvector(1, &nanv, 1, dn, 1, Q);
  magma_int_t* act; magma_imalloc(&act, 2); magma_int_t ones[2] = {1, 1}; magma_setvector(2, sizeof(magma_int_t), ones, 1, act, 1, Q); magma_int_t cb = -9;
  magma_zcompact(1, 1, dA, 1, dn, 1e-8, act, &cb, Q); magma_int_t a[2]; magma_getvector(2, sizeof(magma_int_t), act, 1, a, 1, Q);
  printf("#15 zcompact, one residual norm = NaN, tol 1e-8: active[0]=%lld cBlock=%lld (NaN treated as converged if 0 / 0)\n", (long long)a[0], (long long)cb); }
static void compressor() { magma_z_matrix h = csr(2, 2, {0, 2, 3}, {0, 1, 1}, {C(1), C(0), C(2)});
  magma_int_t r = magma_zmcsrcompressor_gpu(&h, Q); printf("#13 zmcsrcompressor_gpu on a CPU CSR with one stored zero: ret=%lld result %lld x %lld nnz=%lld (expected 2 x 2, nnz 2)\n",
         (long long)r, (long long)h.num_rows, (long long)h.num_cols, (long long)h.nnz); }
static void frob() { magma_z_matrix A = csr(1, 1, {0, 1}, {0}, {C(0, 1)}), B = csr(1, 1, {0, 1}, {0}, {C(0)}); real_Double_t res = -1;
  magma_int_t r = magma_zfrobenius(A, B, &res, Q); printf("#1 zfrobenius([i], [0]): ret=%lld res=%g (expected 1)\n", (long long)r, res); }
static void diagdom() { magma_z_matrix A = csr(2, 2, {0, 1, 1}, {0}, {C(2)}); double mn = -1, mx = -1, av = -1;
  magma_int_t r = magma_zmdiagdom(A, &mn, &mx, &av, Q); printf("#6 zmdiagdom on [[2,0],[0,0]] (row 1 empty): ret=%lld min=%g max=%g avg=%g\n", (long long)r, mn, mx, av); }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); std::string m = argc > 1 ? argv[1] : ""; magma_init(); magma_queue_create(0, &Q);
  if (m == "csrmv_trans") csrmv_trans(); else if (m == "elltmv") elltmv_nan(); else if (m == "mdotc") mdotc(); else if (m == "mdot_shfl") mdot_shfl();
  else if (m == "mgecsrmv") mgecsrmv(); else if (m == "sellp_multi") sellp_multi(); else if (m == "zsort") zsort_nan(); else if (m == "vread") vread();
  else if (m == "csrget") csrget(); else if (m == "compact") compact(); else if (m == "compressor") compressor(); else if (m == "frob") frob(); else if (m == "diagdom") diagdom();
  magma_queue_destroy(Q); magma_finalize(); return 0;
}

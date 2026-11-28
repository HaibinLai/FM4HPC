// Round 60 (roxy): workspace-query interfaces (expert_gpu_work, *_batched_work, *_native_work).  Each is called twice as
// MAGMA's own wrappers do: a query with lwork = -1, then the real call with host and device workspaces of exactly the
// queried byte counts, each followed by a 0xA5 sentinel guard.  guard = bytes of either guard that changed.
//   round60_query_guard <op> <m> <n> <p> <mode H|N> [uplo|trans]
//   geqrf  m n            magma_zgeqrf_expert_gpu_work (nb = magma_get_zgeqrf_nb), dT at geqrf_gpu's documented size (guarded too)
//   getrf  m n            magma_zgetrf_expert_gpu_work (recnb 32); P A = L U residual
//   potrf  n   uplo       magma_zpotrf_expert_gpu_work; against LAPACK zpotrf
//   getri  n              magma_zgetri_expert_gpu_work after magma_zgetrf_gpu; |A inv - I|
//   getrs  n nrhs trans   magma_zgetrs_expert_gpu_work; residual
//   potrs  n nrhs uplo    magma_zpotrs_expert_gpu_work; residual
//   trtri  n   uplo       magma_ztrtri_expert_gpu_work (non-unit); |A inv - I|
//   gegqr  m n ikind      magma_zgegqr_expert_gpu_work; |Q^H Q - I|
//   bgeqrf m n batch      magma_zgeqrf_batched_work (device workspace only)
//   gbtrf  n kl ku        magma_zgbtrf_native_work (device workspace only); gbsv  n kl ku: magma_zgbsv_native_work, nrhs 3
// CSV: r60,op,m,n,p,mode,opt,info,lhost,ldev,guard,err,verdict
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
typedef magmaDoubleComplex Z;
static unsigned Sd = 1u; static double r1() { Sd = Sd * 1664525u + 1013904223u; return ((Sd >> 8) & 0xffffff) / (double)0x1000000 - 0.5; }
static std::vector<Z> rnd(size_t k) { std::vector<Z> v(k); for (auto& x : v) x = MAGMA_Z_MAKE(r1(), r1()); return v; }
static magma_queue_t qs[2]; static magma_event_t ev[2];
static const size_t GB = 1 << 18;   // guard bytes
struct HG { unsigned char* p = nullptr; size_t n = 0; void* make(size_t b) { n = b; magma_malloc_pinned((void**)&p, b + GB); memset(p, 0, b); memset(p + b, 0xA5, GB); return b ? p : p; }
  size_t bad() { size_t w = 0; for (size_t i = 0; i < GB; i++) if (p[n + i] != 0xA5) w++; return w; } ~HG() { if (p) magma_free_pinned(p); } };
struct DG { unsigned char* p = nullptr; size_t n = 0; void* make(size_t b) { n = b; magma_malloc((void**)&p, b + GB); std::vector<unsigned char> s(GB, 0xA5); magma_setvector((magma_int_t)GB, 1, s.data(), 1, p + b, 1, qs[0]); magma_queue_sync(qs[0]); return p; }
  size_t bad() { std::vector<unsigned char> s(GB); magma_getvector((magma_int_t)GB, 1, p + n, 1, s.data(), 1, qs[0]); size_t w = 0; for (auto c : s) if (c != 0xA5) w++; return w; } ~DG() { if (p) magma_free(p); } };
static double maxabs(const std::vector<Z>& v) { double a = 0; for (auto& x : v) a = std::max(a, MAGMA_Z_ABS(x)); return a; }
// |A X - I| for n x n host matrices (A, X column-major, ld n)
static double invres(const std::vector<Z>& A, const std::vector<Z>& X, int n) { double e = 0; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { Z s = MAGMA_Z_ZERO;
  for (int k = 0; k < n; k++) s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(A[i + (size_t)k * n], X[k + (size_t)j * n])); if (i == j) s = MAGMA_Z_SUB(s, MAGMA_Z_ONE); e = std::max(e, MAGMA_Z_ABS(s)); } return e; }
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 6) { printf("usage: see source\n"); return 1; }
  std::string op = argv[1]; magma_int_t m = atoi(argv[2]), n = atoi(argv[3]), p = atoi(argv[4]); char md = argv[5][0], opt = argc > 6 ? argv[6][0] : 'L';
  magma_init(); magma_queue_create(0, &qs[0]); magma_queue_create(0, &qs[1]); magma_event_create(&ev[0]); magma_event_create(&ev[1]);
  magma_mode_t mode = md == 'N' ? MagmaNative : MagmaHybrid; magma_uplo_t uplo = opt == 'U' ? MagmaUpper : MagmaLower;
  magma_int_t info = 0, lh = -1, ld = -1; size_t guard = 0; double err = -1; HG hg; DG dg;
  magma_int_t ldda = magma_roundup(std::max<magma_int_t>(m, 1), 32);
  // generic: f(hwork, &lh, dwork, &ld) queried, then called with guarded buffers
  auto run = [&](auto f) { lh = -1; ld = -1; f((void*)NULL, (void*)NULL); void* hw = hg.make(std::max<magma_int_t>(lh, 0)); void* dw = dg.make(std::max<magma_int_t>(ld, 0));
    f(hw, dw); magma_queue_sync(qs[0]); magma_queue_sync(qs[1]); guard += hg.bad() + dg.bad(); };
  if (op == "geqrf") {
    magma_int_t nb = magma_get_zgeqrf_nb(m, n), mn = std::min(m, n); std::vector<Z> A = rnd((size_t)m * n), tau(std::max<magma_int_t>(mn, 1)); Z* dA; magma_zmalloc(&dA, (size_t)ldda * n);
    magma_zsetmatrix(m, n, A.data(), m, dA, ldda, qs[0]); DG tg; Z* dT = (Z*)tg.make((size_t)(2 * mn + magma_roundup(n, 32)) * nb * sizeof(Z));
    run([&](void* hw, void* dw) { magma_zgeqrf_expert_gpu_work(m, n, hw || dw ? dA : NULL, ldda, tau.data(), dT, &info, mode, nb, hw, &lh, dw, &ld, qs); });
    guard += tg.bad(); magma_free(dA);
  } else if (op == "getrf" || op == "getri") {
    magma_int_t nn = op == "getri" ? n : n, mm = op == "getri" ? n : m; std::vector<Z> A = rnd((size_t)mm * nn); if (op == "getri") for (int i = 0; i < n; i++) A[i + (size_t)i * n] = MAGMA_Z_ADD(A[i + (size_t)i * n], MAGMA_Z_MAKE(n, 0));
    magma_int_t lda2 = magma_roundup(mm, 32); Z* dA; magma_zmalloc(&dA, (size_t)lda2 * nn); magma_zsetmatrix(mm, nn, A.data(), mm, dA, lda2, qs[0]); std::vector<magma_int_t> ipiv(std::min(mm, nn));
    if (op == "getrf") { magma_int_t nb = magma_get_zgetrf_nb(mm, nn);
      run([&](void* hw, void* dw) { magma_zgetrf_expert_gpu_work(mm, nn, hw || dw ? dA : NULL, lda2, hw || dw ? ipiv.data() : NULL, &info, mode, nb, 32, hw, &lh, dw, &ld, ev, qs); });
      std::vector<Z> LU((size_t)mm * nn); magma_zgetmatrix(mm, nn, dA, lda2, LU.data(), mm, qs[0]);
      // residual of P A = L U on a random vector (square part only when m == n)
      if (mm == nn) { int N = (int)nn; std::vector<Z> x = rnd(N), y(N, MAGMA_Z_ZERO), u(N, MAGMA_Z_ZERO), z(N);
        for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) y[i] = MAGMA_Z_ADD(y[i], MAGMA_Z_MUL(A[i + (size_t)j * N], x[j]));
        for (int i = 0; i < N; i++) std::swap(y[i], y[ipiv[i] - 1]);
        for (int i = 0; i < N; i++) for (int j = i; j < N; j++) u[i] = MAGMA_Z_ADD(u[i], MAGMA_Z_MUL(LU[i + (size_t)j * N], x[j]));
        err = 0; for (int i = 0; i < N; i++) { z[i] = u[i]; for (int j = 0; j < i; j++) z[i] = MAGMA_Z_ADD(z[i], MAGMA_Z_MUL(LU[i + (size_t)j * N], u[j])); err = std::max(err, MAGMA_Z_ABS(MAGMA_Z_SUB(y[i], z[i]))); }
        err /= maxabs(A) * maxabs(x) * N; }
    } else { magma_zgetrf_gpu(n, n, dA, lda2, ipiv.data(), &info);
      run([&](void* hw, void* dw) { magma_zgetri_expert_gpu_work(n, hw || dw ? dA : NULL, lda2, hw || dw ? ipiv.data() : NULL, &info, mode, hw, &lh, dw, &ld, qs); });
      std::vector<Z> X((size_t)n * n); magma_zgetmatrix(n, n, dA, lda2, X.data(), n, qs[0]); err = invres(A, X, n); }
    magma_free(dA);
  } else if (op == "potrf" || op == "potrs" || op == "trtri" || op == "getrs") {
    magma_int_t nrhs = (op == "potrs" || op == "getrs") ? std::max<magma_int_t>(p, 1) : 0, lda2 = magma_roundup(n, 32);
    std::vector<Z> A((size_t)n * n); for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { Z v = i == j ? MAGMA_Z_MAKE(n + r1(), 0) : MAGMA_Z_MAKE(r1(), r1()); A[i + (size_t)j * n] = v; A[j + (size_t)i * n] = MAGMA_Z_CONJ(v); }
    if (op == "trtri") for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if ((opt == 'L' && i < j) || (opt == 'U' && i > j)) A[i + (size_t)j * n] = MAGMA_Z_ZERO;
    if (op == "getrs") for (auto& a : A) a = MAGMA_Z_ADD(a, MAGMA_Z_MAKE(r1(), r1()));   // general
    Z* dA; magma_zmalloc(&dA, (size_t)lda2 * n); magma_zsetmatrix(n, n, A.data(), n, dA, lda2, qs[0]);
    std::vector<Z> B = rnd((size_t)n * std::max<magma_int_t>(nrhs, 1)); Z* dB = NULL; if (nrhs) { magma_zmalloc(&dB, (size_t)lda2 * nrhs); magma_zsetmatrix(n, nrhs, B.data(), n, dB, lda2, qs[0]); }
    if (op == "potrf") { magma_int_t nb = magma_get_zpotrf_nb(n);
      run([&](void* hw, void* dw) { magma_zpotrf_expert_gpu_work(uplo, n, hw || dw ? dA : NULL, lda2, &info, mode, nb, 32, hw, &lh, dw, &ld, ev, qs); });
      std::vector<Z> L((size_t)n * n), R = A; magma_zgetmatrix(n, n, dA, lda2, L.data(), n, qs[0]); magma_int_t N = n, rinfo; lapackf77_zpotrf(&opt, &N, R.data(), &N, &rinfo);
      err = 0; double a = 0; for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { if ((opt == 'L' && i < j) || (opt == 'U' && i > j)) continue; err = std::max(err, MAGMA_Z_ABS(MAGMA_Z_SUB(L[i + (size_t)j * n], R[i + (size_t)j * n]))); a = std::max(a, MAGMA_Z_ABS(R[i + (size_t)j * n])); } err /= a; }
    else if (op == "trtri") { run([&](void* hw, void* dw) { magma_ztrtri_expert_gpu_work(uplo, MagmaNonUnit, n, hw || dw ? dA : NULL, lda2, &info, hw, &lh, dw, &ld, qs); });
      std::vector<Z> X((size_t)n * n); magma_zgetmatrix(n, n, dA, lda2, X.data(), n, qs[0]); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if ((opt == 'L' && i < j) || (opt == 'U' && i > j)) X[i + (size_t)j * n] = MAGMA_Z_ZERO; err = invres(A, X, n); }
    else {   // potrs / getrs: factor first (plain interfaces), then the expert solve
      std::vector<magma_int_t> ipiv(n); magma_trans_t trans = opt == 'C' ? Magma_ConjTrans : opt == 'T' ? MagmaTrans : MagmaNoTrans;
      if (op == "potrs") magma_zpotrf_gpu(uplo, n, dA, lda2, &info); else magma_zgetrf_gpu(n, n, dA, lda2, ipiv.data(), &info);
      if (op == "potrs") run([&](void* hw, void* dw) { magma_zpotrs_expert_gpu_work(uplo, n, nrhs, hw || dw ? dA : NULL, lda2, hw || dw ? dB : NULL, lda2, &info, hw, &lh, dw, &ld, qs[0]); });
      else run([&](void* hw, void* dw) { magma_zgetrs_expert_gpu_work(trans, n, nrhs, hw || dw ? dA : NULL, lda2, hw || dw ? ipiv.data() : NULL, hw || dw ? dB : NULL, lda2, &info, mode, hw, &lh, dw, &ld, qs[0]); });
      std::vector<Z> X((size_t)n * nrhs); magma_zgetmatrix(n, nrhs, dB, lda2, X.data(), n, qs[0]); err = 0;
      for (int k = 0; k < nrhs; k++) for (int i = 0; i < n; i++) { Z s = MAGMA_Z_ZERO; for (int j = 0; j < n; j++) { Z a = op == "getrs" && opt != 'N' ? A[j + (size_t)i * n] : A[i + (size_t)j * n]; if (opt == 'C') a = MAGMA_Z_CONJ(a); s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(a, X[j + (size_t)k * n])); }
        err = std::max(err, MAGMA_Z_ABS(MAGMA_Z_SUB(s, B[i + (size_t)k * n]))); }
      err /= maxabs(A) * maxabs(X) * n; }
    magma_free(dA); if (dB) magma_free(dB);
  } else if (op == "gegqr") {
    std::vector<Z> A = rnd((size_t)m * n); Z* dA; magma_zmalloc(&dA, (size_t)ldda * n); magma_zsetmatrix(m, n, A.data(), m, dA, ldda, qs[0]);
    run([&](void* hw, void* dw) { magma_zgegqr_expert_gpu_work(p, m, n, hw || dw ? dA : NULL, ldda, hw, &lh, dw, &ld, &info, qs[0]); });
    std::vector<Z> Q((size_t)m * n); magma_zgetmatrix(m, n, dA, ldda, Q.data(), m, qs[0]); err = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { Z s = MAGMA_Z_ZERO; for (int k = 0; k < m; k++) s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(MAGMA_Z_CONJ(Q[k + (size_t)i * m]), Q[k + (size_t)j * m])); if (i == j) s = MAGMA_Z_SUB(s, MAGMA_Z_ONE); err = std::max(err, MAGMA_Z_ABS(s)); }
    magma_free(dA);
  } else if (op == "bgeqrf") {
    magma_int_t bc = std::max<magma_int_t>(p, 1), mn = std::min(m, n); std::vector<Z*> hA(bc), ht(bc); for (int b = 0; b < bc; b++) { magma_zmalloc(&hA[b], (size_t)ldda * n); magma_zmalloc(&ht[b], std::max<magma_int_t>(mn, 1)); std::vector<Z> A = rnd((size_t)m * n); magma_zsetmatrix(m, n, A.data(), m, hA[b], ldda, qs[0]); }
    Z **dAa, **dta; magma_int_t* dinfo; magma_malloc((void**)&dAa, bc * sizeof(Z*)); magma_malloc((void**)&dta, bc * sizeof(Z*)); magma_imalloc(&dinfo, bc);
    magma_setvector(bc, sizeof(Z*), hA.data(), 1, dAa, 1, qs[0]); magma_setvector(bc, sizeof(Z*), ht.data(), 1, dta, 1, qs[0]);
    ld = -1; magma_zgeqrf_batched_work(m, n, NULL, ldda, NULL, NULL, NULL, &ld, bc, qs[0]); lh = 0; void* dw = dg.make(std::max<magma_int_t>(ld, 0)); hg.make(0);
    info = magma_zgeqrf_batched_work(m, n, dAa, ldda, dta, dinfo, dw, &ld, bc, qs[0]); magma_queue_sync(qs[0]); guard = dg.bad();
    for (int b = 0; b < bc; b++) { magma_free(hA[b]); magma_free(ht[b]); } magma_free(dAa); magma_free(dta); magma_free(dinfo);
  } else if (op == "gbtrf" || op == "gbsv") {
    magma_int_t kl = m, ku = p, N = n, ldab = 2 * kl + ku + 1, nrhs = 3; std::vector<Z> AB((size_t)ldab * N, MAGMA_Z_ZERO);
    for (int j = 0; j < N; j++) for (int i = std::max<int>(0, j - (int)ku); i <= std::min<int>(N - 1, j + (int)kl); i++) AB[kl + ku + i - j + (size_t)j * ldab] = i == j ? MAGMA_Z_MAKE(4 + r1(), 0) : MAGMA_Z_MAKE(r1(), r1());
    Z *dAB, *dB; magma_int_t *dipiv, *dinfo; magma_zmalloc(&dAB, (size_t)ldab * N); magma_imalloc(&dipiv, N); magma_imalloc(&dinfo, 1); magma_zmalloc(&dB, (size_t)N * nrhs);
    magma_zsetmatrix(ldab, N, AB.data(), ldab, dAB, ldab, qs[0]); std::vector<Z> B = rnd((size_t)N * nrhs); magma_zsetmatrix(N, nrhs, B.data(), N, dB, N, qs[0]);
    ld = -1; if (op == "gbtrf") magma_zgbtrf_native_work(N, N, kl, ku, NULL, ldab, NULL, &info, NULL, &ld, qs[0]); else magma_zgbsv_native_work(N, kl, ku, nrhs, NULL, ldab, NULL, NULL, N, &info, NULL, &ld, qs[0]);   // the query dereferences info, as magma_zgbsv_native passes it
    lh = 0; hg.make(0); void* dw = dg.make(std::max<magma_int_t>(ld, 0));
    if (op == "gbtrf") magma_zgbtrf_native_work(N, N, kl, ku, dAB, ldab, dipiv, &info, dw, &ld, qs[0]); else magma_zgbsv_native_work(N, kl, ku, nrhs, dAB, ldab, dipiv, dB, N, &info, dw, &ld, qs[0]);
    magma_queue_sync(qs[0]); guard = dg.bad();
    if (op == "gbsv") { std::vector<Z> X((size_t)N * nrhs); magma_zgetmatrix(N, nrhs, dB, N, X.data(), N, qs[0]); err = 0; double am = 0, xm = maxabs(X);
      for (int k = 0; k < nrhs; k++) for (int i = 0; i < N; i++) { Z s = MAGMA_Z_ZERO; for (int j = std::max<int>(0, i - (int)kl); j <= std::min<int>(N - 1, i + (int)ku); j++) { Z a = AB[kl + ku + i - j + (size_t)j * ldab]; am = std::max(am, MAGMA_Z_ABS(a)); s = MAGMA_Z_ADD(s, MAGMA_Z_MUL(a, X[j + (size_t)k * N])); }
        err = std::max(err, MAGMA_Z_ABS(MAGMA_Z_SUB(s, B[i + (size_t)k * N]))); } err /= am * xm * N; }
    magma_free(dAB); magma_free(dB); magma_free(dipiv); magma_free(dinfo);
  }
  bool ok = info == 0 && guard == 0 && (err < 0 || err < 1e-10);
  printf("r60,%s,%lld,%lld,%lld,%c,%c,%lld,%lld,%lld,%zu,%.3g,%s\n", op.c_str(), (long long)m, (long long)n, (long long)p, md, opt, (long long)info, (long long)lh, (long long)ld, guard, err, ok ? "OK" : "WRONG");
  fprintf(stderr, "DONE cuda=%s\n", cudaGetErrorString(cudaGetLastError()));
  magma_event_destroy(ev[0]); magma_event_destroy(ev[1]); magma_queue_destroy(qs[0]); magma_queue_destroy(qs[1]); magma_finalize(); return 0;
}

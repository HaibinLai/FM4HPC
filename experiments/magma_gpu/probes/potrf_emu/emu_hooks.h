// Round 50: hooks for running magma_[zd]potrf3_mgpu with ngpu "virtual devices" on one physical GPU.
// Included by the generated copies emu_[zd]potrf3_mgpu.cpp right after magma_internal.h / trace.h.
//  * magma_setdevice / magma_getdevice become no-ops: every virtual device's buffers and its 3 queues live on GPU 0.
//    Queue / event semantics (in-order queues, cross-queue event waits, host syncs) are the same as across devices.
//  * EMU_DELAY_KIND 1: before every stream-2 trsm whose triangular panel lies in dlP (the received diagonal
//    block), the stream sleeps EMU_DELAY_MS (cudaLaunchHostFunc).  This only changes timing, which CUDA leaves free.
//  * EMU_DELAY_KIND 2: the same sleep before every stream-1 copy of the factored diagonal block from the
//    pinned host workspace into the owner's dlA (host-buffer hazard).
//  * kinds 3 / 4: as 1 / 2 but only the first such operation on virtual device `dev` (one delay in the whole run),
//    so that no other delay lies on a chain the host waits for.
//  * kinds 5 / 6: as 1 / 2 but only the c-th such operation in the run (c = `dev`, counted from 0).
//  * EMU_FIX 1: after each such stream-2 trsm an event is recorded, and the next stream-1 copy into that
//    device's dlP diagonal block waits for it (the one ordering the source lacks).
#pragma once
#include <cuda_runtime.h>
#include <thread>
#include <chrono>
#include <cstdio>
struct EmuState {
  int ngpu = 0; char* pbase[8] = {}; size_t pbytes[8] = {}; char* abase[8] = {}; size_t abytes[8] = {};
  magma_queue_t q[8][3] = {}; int kind = 0, ms = 0, fix = 0, dev = 1, once = 0, cnt[2] = {0, 0};
  int tdev = -1, tset = 0; cudaEvent_t t0 = 0, tA = 0, tB[64] = {}; int nB = 0;   // timing of the delayed trsm vs later dlP copies on that device
  cudaEvent_t ev[8] = {}; int rec[8] = {};
  char* hbase = nullptr; size_t esz = 0; long long hld = 0; long long sr0 = 0, sr1 = 0, sc0 = 0, sc1 = 0; int hset = 0;   // host source block of the delayed copy
  long long n_delay = 0, n_fixwait = 0;
};
extern EmuState g_emu;
static inline void emu_setdevice(magma_device_t) {}
static inline int emu_find(magma_queue_t queue, int* s) { for (int d = 0; d < g_emu.ngpu; d++) for (int k = 0; k < 3; k++) if (g_emu.q[d][k] == queue) { *s = k; return d; } *s = -1; return -1; }
static inline bool emu_in(const void* p, char* const* base, const size_t* bytes, int d) { return d >= 0 && (const char*)p >= base[d] && (const char*)p < base[d] + bytes[d]; }
static void CUDART_CB emu_sleep(void* ms) { std::this_thread::sleep_for(std::chrono::milliseconds((size_t)ms)); }
static inline void emu_delay(magma_queue_t queue) { g_emu.n_delay++; cudaLaunchHostFunc(magma_queue_get_cuda_stream(queue), emu_sleep, (void*)(size_t)g_emu.ms); }
static inline bool emu_pick(int k, int d) { if (g_emu.kind == k) return true; if (g_emu.kind == k + 4) { bool hit = g_emu.cnt[k - 1] == g_emu.dev; g_emu.cnt[k - 1]++; if (hit) { fprintf(stderr, "delayed op #%d on vdev %d\n", g_emu.dev, d); g_emu.tdev = d; } return hit; } if (g_emu.kind == k + 2 && d == g_emu.dev && !g_emu.once) { g_emu.once = 1; return true; } return false; }
static inline void emu_pre_trsm(const void* dA, magma_queue_t queue) { int s, d = emu_find(queue, &s); if (s == 1 && emu_in(dA, g_emu.pbase, g_emu.pbytes, d) && emu_pick(1, d)) emu_delay(queue); }
static inline void emu_post_trsm(const void* dA, magma_queue_t queue) { int s, d = emu_find(queue, &s);
  if (s == 1 && emu_in(dA, g_emu.pbase, g_emu.pbytes, d) && g_emu.fix) { cudaEventRecord(g_emu.ev[d], magma_queue_get_cuda_stream(queue)); g_emu.rec[d] = 1; } }
static inline void emu_pre_set(const void* dB, magma_queue_t queue) { int s, d = emu_find(queue, &s); if (s != 0) return;
  if (emu_in(dB, g_emu.pbase, g_emu.pbytes, d) && g_emu.fix && g_emu.rec[d]) { g_emu.n_fixwait++; cudaStreamWaitEvent(magma_queue_get_cuda_stream(queue), g_emu.ev[d], 0); }
  if (emu_in(dB, g_emu.abase, g_emu.abytes, d) && emu_pick(2, d)) emu_delay(queue); }

// host-buffer timing: rows / columns (in elements) of a host block inside the pinned workspace
static inline void emu_hrange(const void* h, long long m, long long n, long long* r0, long long* r1, long long* c0, long long* c1) {
  long long o = ((const char*)h - g_emu.hbase) / (long long)g_emu.esz; *r0 = o % g_emu.hld; *r1 = *r0 + m; *c0 = o / g_emu.hld; *c1 = *c0 + n; }
static inline void emu_post_get(const void* hB, long long m, long long n, magma_queue_t q) {
  if (!g_emu.hset || !g_emu.hbase || g_emu.nB >= 64 || !g_emu.tB[g_emu.nB]) return; long long r0, r1, c0, c1; emu_hrange(hB, m, n, &r0, &r1, &c0, &c1);
  if (r0 < g_emu.sr1 && g_emu.sr0 < r1 && c0 < g_emu.sc1 && g_emu.sc0 < c1) cudaEventRecord(g_emu.tB[g_emu.nB++], magma_queue_get_cuda_stream(q)); }
#define magma_setdevice(d) emu_setdevice(d)
#define magma_getdevice(p) (*(p) = 0)
#ifdef PRECISION_z
static inline void emu_ztrsm(magma_side_t a, magma_uplo_t b, magma_trans_t c, magma_diag_t e, magma_int_t m, magma_int_t n, magmaDoubleComplex al,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_queue_t q) {
  int was = g_emu.tdev; emu_pre_trsm(dA, q); bool mine = was < 0 && g_emu.tdev >= 0 && !g_emu.tset;
  if (mine && g_emu.t0) cudaEventRecord(g_emu.t0, magma_queue_get_cuda_stream(q));
  magma_ztrsm(a, b, c, e, m, n, al, dA, ldda, dB, lddb, q); emu_post_trsm(dA, q);
  if (mine && g_emu.tA) { cudaEventRecord(g_emu.tA, magma_queue_get_cuda_stream(q)); g_emu.tset = 1; } }
static inline void emu_zset(magma_int_t m, magma_int_t n, const magmaDoubleComplex* h, magma_int_t lda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_queue_t q) {
  int was2 = g_emu.tdev; emu_pre_set(dB, q); bool mine2 = was2 < 0 && g_emu.tdev >= 0 && !g_emu.tset && g_emu.kind == 6;
  if (mine2 && g_emu.t0) { cudaEventRecord(g_emu.t0, magma_queue_get_cuda_stream(q)); emu_hrange(h, m, n, &g_emu.sr0, &g_emu.sr1, &g_emu.sc0, &g_emu.sc1); }
  magma_zsetmatrix_async_internal(m, n, h, lda, dB, lddb, q, __func__, __FILE__, __LINE__);
  if (mine2 && g_emu.tA) { cudaEventRecord(g_emu.tA, magma_queue_get_cuda_stream(q)); g_emu.tset = 1; g_emu.hset = 1; }
  int s, d = emu_find(q, &s); if (g_emu.tset && s == 0 && d == g_emu.tdev && emu_in(dB, g_emu.pbase, g_emu.pbytes, d) && g_emu.nB < 64 && g_emu.tB[g_emu.nB]) cudaEventRecord(g_emu.tB[g_emu.nB++], magma_queue_get_cuda_stream(q)); }
#define magma_ztrsm emu_ztrsm
static inline void emu_zget(magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex* h, magma_int_t ldb, magma_queue_t q) {
  magma_zgetmatrix_async_internal(m, n, dA, ldda, h, ldb, q, __func__, __FILE__, __LINE__); emu_post_get(h, m, n, q); }
#undef magma_zsetmatrix_async
#define magma_zsetmatrix_async emu_zset
#undef magma_zgetmatrix_async
#define magma_zgetmatrix_async emu_zget
#endif
#ifdef PRECISION_d
static inline void emu_dtrsm_work(magma_side_t a, magma_uplo_t b, magma_trans_t c, magma_diag_t e, magma_int_t m, magma_int_t n, double al,
    magmaDouble_const_ptr dA, magma_int_t ldda, magmaDouble_ptr dB, magma_int_t lddb, magmaDouble_ptr dX, magma_int_t lddx,
    magma_int_t flag, magmaDouble_ptr dinvA, magma_int_t len, magma_queue_t q) {
  emu_pre_trsm(dA, q); magmablas_dtrsm_work(a, b, c, e, m, n, al, dA, ldda, dB, lddb, dX, lddx, flag, dinvA, len, q); emu_post_trsm(dA, q); }
static inline void emu_dset(magma_int_t m, magma_int_t n, const double* h, magma_int_t lda, magmaDouble_ptr dB, magma_int_t lddb, magma_queue_t q) {
  int was2 = g_emu.tdev; emu_pre_set(dB, q); bool mine2 = was2 < 0 && g_emu.tdev >= 0 && !g_emu.tset && g_emu.kind == 6;
  if (mine2 && g_emu.t0) { cudaEventRecord(g_emu.t0, magma_queue_get_cuda_stream(q)); emu_hrange(h, m, n, &g_emu.sr0, &g_emu.sr1, &g_emu.sc0, &g_emu.sc1); }
  magma_dsetmatrix_async_internal(m, n, h, lda, dB, lddb, q, __func__, __FILE__, __LINE__);
  if (mine2 && g_emu.tA) { cudaEventRecord(g_emu.tA, magma_queue_get_cuda_stream(q)); g_emu.tset = 1; g_emu.hset = 1; } }
#define magmablas_dtrsm_work emu_dtrsm_work
static inline void emu_dget(magma_int_t m, magma_int_t n, magmaDouble_const_ptr dA, magma_int_t ldda, double* h, magma_int_t ldb, magma_queue_t q) {
  magma_dgetmatrix_async_internal(m, n, dA, ldda, h, ldb, q, __func__, __FILE__, __LINE__); emu_post_get(h, m, n, q); }
#undef magma_dsetmatrix_async
#define magma_dsetmatrix_async emu_dset
#undef magma_dgetmatrix_async
#define magma_dgetmatrix_async emu_dget
#endif
#ifdef PRECISION_c
static inline void emu_ctrsm(magma_side_t a, magma_uplo_t b, magma_trans_t c, magma_diag_t e, magma_int_t m, magma_int_t n, magmaFloatComplex al,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda, magmaFloatComplex_ptr dB, magma_int_t lddb, magma_queue_t q) {
  int was = g_emu.tdev; emu_pre_trsm(dA, q); bool mine = was < 0 && g_emu.tdev >= 0 && !g_emu.tset;
  if (mine && g_emu.t0) cudaEventRecord(g_emu.t0, magma_queue_get_cuda_stream(q));
  magma_ctrsm(a, b, c, e, m, n, al, dA, ldda, dB, lddb, q); emu_post_trsm(dA, q);
  if (mine && g_emu.tA) { cudaEventRecord(g_emu.tA, magma_queue_get_cuda_stream(q)); g_emu.tset = 1; } }
static inline void emu_cset(magma_int_t m, magma_int_t n, const magmaFloatComplex* h, magma_int_t lda, magmaFloatComplex_ptr dB, magma_int_t lddb, magma_queue_t q) {
  int was2 = g_emu.tdev; emu_pre_set(dB, q); bool mine2 = was2 < 0 && g_emu.tdev >= 0 && !g_emu.tset && g_emu.kind == 6;
  if (mine2 && g_emu.t0) { cudaEventRecord(g_emu.t0, magma_queue_get_cuda_stream(q)); emu_hrange(h, m, n, &g_emu.sr0, &g_emu.sr1, &g_emu.sc0, &g_emu.sc1); }
  magma_csetmatrix_async_internal(m, n, h, lda, dB, lddb, q, __func__, __FILE__, __LINE__);
  if (mine2 && g_emu.tA) { cudaEventRecord(g_emu.tA, magma_queue_get_cuda_stream(q)); g_emu.tset = 1; g_emu.hset = 1; }
  int s, d = emu_find(q, &s); if (g_emu.tset && s == 0 && d == g_emu.tdev && emu_in(dB, g_emu.pbase, g_emu.pbytes, d) && g_emu.nB < 64 && g_emu.tB[g_emu.nB]) cudaEventRecord(g_emu.tB[g_emu.nB++], magma_queue_get_cuda_stream(q)); }
#define magma_ctrsm emu_ctrsm
static inline void emu_cget(magma_int_t m, magma_int_t n, magmaFloatComplex_const_ptr dA, magma_int_t ldda, magmaFloatComplex* h, magma_int_t ldb, magma_queue_t q) {
  magma_cgetmatrix_async_internal(m, n, dA, ldda, h, ldb, q, __func__, __FILE__, __LINE__); emu_post_get(h, m, n, q); }
#undef magma_csetmatrix_async
#define magma_csetmatrix_async emu_cset
#undef magma_cgetmatrix_async
#define magma_cgetmatrix_async emu_cget
#endif
#ifdef PRECISION_s
static inline void emu_strsm_work(magma_side_t a, magma_uplo_t b, magma_trans_t c, magma_diag_t e, magma_int_t m, magma_int_t n, float al,
    magmaFloat_const_ptr dA, magma_int_t ldda, magmaFloat_ptr dB, magma_int_t lddb, magmaFloat_ptr dX, magma_int_t lddx,
    magma_int_t flag, magmaFloat_ptr dinvA, magma_int_t len, magma_queue_t q) {
  emu_pre_trsm(dA, q); magmablas_strsm_work(a, b, c, e, m, n, al, dA, ldda, dB, lddb, dX, lddx, flag, dinvA, len, q); emu_post_trsm(dA, q); }
static inline void emu_sset(magma_int_t m, magma_int_t n, const float* h, magma_int_t lda, magmaFloat_ptr dB, magma_int_t lddb, magma_queue_t q) {
  int was2 = g_emu.tdev; emu_pre_set(dB, q); bool mine2 = was2 < 0 && g_emu.tdev >= 0 && !g_emu.tset && g_emu.kind == 6;
  if (mine2 && g_emu.t0) { cudaEventRecord(g_emu.t0, magma_queue_get_cuda_stream(q)); emu_hrange(h, m, n, &g_emu.sr0, &g_emu.sr1, &g_emu.sc0, &g_emu.sc1); }
  magma_ssetmatrix_async_internal(m, n, h, lda, dB, lddb, q, __func__, __FILE__, __LINE__);
  if (mine2 && g_emu.tA) { cudaEventRecord(g_emu.tA, magma_queue_get_cuda_stream(q)); g_emu.tset = 1; g_emu.hset = 1; } }
#define magmablas_strsm_work emu_strsm_work
static inline void emu_sget(magma_int_t m, magma_int_t n, magmaFloat_const_ptr dA, magma_int_t ldda, float* h, magma_int_t ldb, magma_queue_t q) {
  magma_sgetmatrix_async_internal(m, n, dA, ldda, h, ldb, q, __func__, __FILE__, __LINE__); emu_post_get(h, m, n, q); }
#undef magma_ssetmatrix_async
#define magma_ssetmatrix_async emu_sset
#undef magma_sgetmatrix_async
#define magma_sgetmatrix_async emu_sget
#endif

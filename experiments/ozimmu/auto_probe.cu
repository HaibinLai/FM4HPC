// Probe of ozIMMU's fp64_int8_auto mode selection (pinned enp1s0/ozIMMU 08eea92), WITHOUT running the INT8 GEMM.
// Evidence collected per input (library code from split.cu/handle.cu/config.cu/... compiled for the local GPU):
//  (1) library decision via the public mtk::ozimmu::auto_mode_select (three repetitions);
//  (2) an exact host-side replica of calculate_mantissa_loss_kernel + the decision loop (the "correct" decision);
//  (3) the 16 device counters read back directly (the library allocates/zeroes only 8 = 13-6+1 of them);
//  (4) the decision after pre-filling the caller's stack with 0x00 vs 0xFF (the library reads host_buffer[8..15]
//      although only 8 entries exist and only 8 are copied back).
// Usage: auto_probe [threshold=1.5]
#include <ozimmu/ozimmu.hpp>
#include "handle.hpp"   // internal: to read d_mantissa_loss_counter_ptr
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Case { const char* name; std::size_t m, n, k; int spread; double zero_frac; };

static std::uint64_t ebits(double x) { std::uint64_t u; std::memcpy(&u, &x, 8); return (u >> 52) & 0x7ff; }

// Exact replica of calculate_mantissa_loss_kernel for one "row" of length len with stride inc.
static void replica_rows(const std::vector<double>& M, std::size_t rows, std::size_t len, bool row_is_strided,
                         std::size_t ld, unsigned bits, std::uint64_t cnt[16]) {
  for (std::size_t r = 0; r < rows; r++) {
    double maxmask = 0;
    for (std::size_t i = 0; i < len; i++) {
      const double v = M[row_is_strided ? (i * ld + r) : (r * ld + i)];
      const double mv = std::ldexp(1.0, (int)ebits(v) - 1023);   // mask_exponent reinterpreted as fp (normal inputs)
      if (v != 0 && mv > maxmask) maxmask = mv;
    }
    const double max_exp = maxmask * 2;                          // x2(get_exp_max_element(...))
    for (std::size_t i = 0; i < len; i++) {
      const double v = M[row_is_strided ? (i * ld + r) : (r * ld + i)];
      if (v == 0 || max_exp == 0) continue;
      const std::uint64_t req = (ebits(max_exp) - ebits(v)) + 53;
      for (unsigned s = 3; s <= 18; s++) {
        const std::uint64_t space = (std::uint64_t)s * bits;
        cnt[s - 3] += space < req ? req - space : 0;
      }
    }
  }
}

static std::string replica_decision(const std::uint64_t cnt[16], std::size_t m, std::size_t n, std::size_t k, double thr) {
  for (unsigned s = 3; s <= 18; s++)
    if (cnt[s - 3] / static_cast<double>(m * k + k * n) <= thr) return "fp64_int8_" + std::to_string(s);
  return "dgemm";
}

__attribute__((noinline)) static void fill_stack(unsigned char byte) {
  volatile unsigned char buf[16384];
  for (unsigned i = 0; i < sizeof(buf); i++) buf[i] = byte;
}

__attribute__((noinline)) static std::string lib_select(mtk::ozimmu::handle_t h, const Case& c, double* da, double* db, double thr) {
  const auto mode = mtk::ozimmu::auto_mode_select(h, mtk::ozimmu::op_n, mtk::ozimmu::op_n, c.m, c.n, c.k,
                                                   da, c.m, db, c.k, mtk::ozimmu::real, thr);
  cudaDeviceSynchronize();
  return mtk::ozimmu::get_compute_mode_name_str(mode);
}

int main(int argc, char** argv) {
  const double thr = argc > 1 ? std::atof(argv[1]) : 1.5;
  mtk::ozimmu::handle_t h;
  mtk::ozimmu::create(&h);
  mtk::ozimmu::set_auto_mantissa_loss_threashold(h, thr);
  std::printf("library counter length (allocated and zeroed) = %d, counters written by the kernel = 16 (splits 3..18)\n",
              (int)mtk::ozimmu::handle::mantissa_loss_counter_length);
  const std::vector<Case> cases = {
      {"easy_k64", 8, 8, 64, 0, 0.0},
      {"wide_k64", 8, 8, 64, 200, 0.0},
      {"large_k2^20", 8, 8, 1u << 20, 0, 0.0},
      {"zeros_k64", 8, 8, 64, 0, 0.3},          // zeros -> early return before __shfl_xor_sync(~0u)
      {"k33", 8, 8, 33, 0, 0.0},                // k not a multiple of 32 -> partial warp in the shuffle
      {"k33_wide", 8, 8, 33, 60, 0.0},
  };
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> U(-1.0, 1.0), Z(0.0, 1.0);
  for (const auto& c : cases) {
    std::uniform_int_distribution<int> E(0, c.spread);
    std::vector<double> a(c.m * c.k), b(c.k * c.n);
    for (auto& x : a) x = Z(rng) < c.zero_frac ? 0.0 : std::ldexp(U(rng), E(rng));
    for (auto& x : b) x = Z(rng) < c.zero_frac ? 0.0 : std::ldexp(U(rng), E(rng));
    double *da, *db;
    cudaMalloc(&da, a.size() * 8); cudaMalloc(&db, b.size() * 8);
    cudaMemcpy(da, a.data(), a.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size() * 8, cudaMemcpyHostToDevice);
    const unsigned bits = mtk::ozimmu::get_bits_per_int8(c.k);
    // (2) replica: A is col-major m x k (rows strided by ld=m); B is col-major k x n, its columns are contiguous
    std::uint64_t cnt[16] = {0};
    replica_rows(a, c.m, c.k, true, c.m, bits, cnt);
    replica_rows(b, c.n, c.k, false, c.k, bits, cnt);
    const std::string want = replica_decision(cnt, c.m, c.n, c.k, thr);
    // (1) library decision, 3 repetitions
    std::string got[3];
    for (int r = 0; r < 3; r++) got[r] = lib_select(h, c, da, db, thr);
    // (3) device counters after the last call (16 entries; only 8 belong to the allocation)
    unsigned long long dev[16] = {0};
    cudaError_t e = cudaMemcpy(dev, h->d_mantissa_loss_counter_ptr, sizeof(dev), cudaMemcpyDeviceToHost);
    // (4) decision with the caller's stack pre-filled with 0x00 / 0xFF
    fill_stack(0x00); const std::string z0 = lib_select(h, c, da, db, thr);
    fill_stack(0xFF); const std::string zf = lib_select(h, c, da, db, thr);
    std::printf("\n%-12s m=%zu n=%zu k=%zu spread=%d zeros=%.1f bits/slice=%u\n", c.name, c.m, c.n, c.k, c.spread, c.zero_frac, bits);
    std::printf("  replica (correct) decision : %s\n", want.c_str());
    std::printf("  library decision x3        : %s %s %s\n", got[0].c_str(), got[1].c_str(), got[2].c_str());
    std::printf("  library after stack=0x00   : %s   after stack=0xFF : %s\n", z0.c_str(), zf.c_str());
    std::printf("  avg loss replica  s=3..18  :");
    for (int s = 0; s < 16; s++) std::printf(" %.2f", cnt[s] / static_cast<double>(c.m * c.k + c.k * c.n));
    std::printf("\n  device counters / replica (after 3 calls; memcpy %s):", cudaGetErrorString(e));
    for (int s = 0; s < 16; s++) std::printf(" %s%.2f", s == 8 ? "| " : "", cnt[s] ? dev[s] / (double)cnt[s] : (dev[s] ? -1.0 : 0.0));
    std::printf("\n");
    cudaFree(da); cudaFree(db);
  }
  mtk::ozimmu::destroy(h);
  return 0;
}

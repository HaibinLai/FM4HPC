// End-to-end accuracy of ozIMMU (pinned 08eea92) fp64_int8_auto vs explicit modes vs DGEMM.
// Metric per output entry: |C - ref| / (|A||B|)_ij in units of u = 2^-53 (DGEMM's a-priori bound is about k*u).
// ref is accumulated in long double (64-bit mantissa): its own error is <= k*2^-64 relative to |A||B|.
// Cases:
//   bug_dgemm  : every row of A = [1, 2^-200 x (k-1)], every column of B = [2^-200, 1 x (k-1)].
//                The correct (replica) decision is dgemm; see what the library picks and what it returns.
//   avg_hides  : random A, B with a single bad row/column pair (row 0 of A = [1, 2^-100 ...], col 0 of B =
//                [2^-100, 1 ...]); the average loss over all m*k+k*n elements is small, so a mode <= 10 is
//                selected (counters 0..7 are the valid ones), but entry (0,0) is entirely lost.
//   rand_sX    : random entries 2^U(0..X) * U(-1,1), sweep of exponent spread X.
// Usage: gemm_probe [threshold=1.5]
#include <ozimmu/ozimmu.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace mtk::ozimmu;

static std::uint64_t ebits(double x) { std::uint64_t u; std::memcpy(&u, &x, 8); return (u >> 52) & 0x7ff; }

// replica of calculate_mantissa_loss_kernel (see auto_probe.cu)
static void replica_rows(const std::vector<double>& M, std::size_t rows, std::size_t len, bool strided,
                         std::size_t ld, unsigned bits, std::uint64_t cnt[16]) {
  for (std::size_t r = 0; r < rows; r++) {
    double maxmask = 0;
    for (std::size_t i = 0; i < len; i++) {
      const double v = M[strided ? (i * ld + r) : (r * ld + i)];
      const double mv = std::ldexp(1.0, (int)ebits(v) - 1023);
      if (v != 0 && mv > maxmask) maxmask = mv;
    }
    const double max_exp = maxmask * 2;
    for (std::size_t i = 0; i < len; i++) {
      const double v = M[strided ? (i * ld + r) : (r * ld + i)];
      if (v == 0 || max_exp == 0) continue;
      const std::uint64_t req = (ebits(max_exp) - ebits(v)) + 53;
      for (unsigned s = 3; s <= 18; s++) { const std::uint64_t sp = (std::uint64_t)s * bits; cnt[s - 3] += sp < req ? req - sp : 0; }
    }
  }
}

int main(int argc, char** argv) {
  const double thr = argc > 1 ? std::atof(argv[1]) : 1.5;
  handle_t h;
  create(&h);
  set_auto_mantissa_loss_threashold(h, thr);
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> U(-1.0, 1.0);

  struct Case { std::string name; std::size_t m, n, k; std::function<void(std::vector<double>&, std::vector<double>&, std::size_t, std::size_t, std::size_t)> fill; };
  std::vector<Case> cases;
  cases.push_back({"bug_dgemm", 8, 8, 64, [](auto& A, auto& B, std::size_t m, std::size_t n, std::size_t k) {
    for (std::size_t r = 0; r < m; r++) for (std::size_t i = 0; i < k; i++) A[i * m + r] = i == 0 ? 1.0 : std::ldexp(1.0, -200);
    for (std::size_t c = 0; c < n; c++) for (std::size_t i = 0; i < k; i++) B[c * k + i] = i == 0 ? std::ldexp(1.0, -200) : 1.0; }});
  cases.push_back({"avg_hides", 64, 64, 64, [&](auto& A, auto& B, std::size_t m, std::size_t n, std::size_t k) {
    for (auto& x : A) x = U(rng);
    for (auto& x : B) x = U(rng);
    for (std::size_t i = 0; i < k; i++) A[i * m + 0] = i == 0 ? 1.0 : std::ldexp(1.0, -100);
    for (std::size_t i = 0; i < k; i++) B[0 * k + i] = i == 0 ? std::ldexp(1.0, -100) : 1.0; }});
  // zero_lane0: random entries with spread 30, but every element whose index along k is a multiple of 32 is 0.
  // In calculate_mantissa_loss_core a zero returns before the full-mask __shfl_xor_sync and only lane 0 does the
  // atomicAdd, so every warp of the counting kernel may drop its contribution.
  cases.push_back({"zero_lane0", 64, 64, 256, [&](auto& A, auto& B, std::size_t m, std::size_t n, std::size_t k) {
    std::uniform_int_distribution<int> E(0, 30);
    for (std::size_t i = 0; i < k; i++) for (std::size_t r = 0; r < m; r++) A[i * m + r] = i % 32 == 0 ? 0.0 : std::ldexp(U(rng), E(rng));
    for (std::size_t c = 0; c < n; c++) for (std::size_t i = 0; i < k; i++) B[c * k + i] = i % 32 == 0 ? 0.0 : std::ldexp(U(rng), E(rng)); }});
  cases.push_back({"sparse30", 64, 64, 256, [&](auto& A, auto& B, std::size_t, std::size_t, std::size_t) {
    std::uniform_int_distribution<int> E(0, 30);
    std::uniform_real_distribution<double> Z(0, 1);
    for (auto& x : A) x = Z(rng) < 0.3 ? 0.0 : std::ldexp(U(rng), E(rng));
    for (auto& x : B) x = Z(rng) < 0.3 ? 0.0 : std::ldexp(U(rng), E(rng)); }});
  // tri: A upper triangular (row r is zero before index r), B lower triangular (column c is zero before index c);
  // in each row/column the warp that contains the diagonal has lane 0 = 0, so at most one warp per row is dropped.
  for (int X : {30, 60}) {
    cases.push_back({"tri_s" + std::to_string(X), 256, 256, 256, [&, X](auto& A, auto& B, std::size_t m, std::size_t n, std::size_t k) {
      std::uniform_int_distribution<int> E(0, X);
      for (std::size_t i = 0; i < k; i++) for (std::size_t r = 0; r < m; r++) A[i * m + r] = i < r ? 0.0 : std::ldexp(U(rng), E(rng));
      for (std::size_t c = 0; c < n; c++) for (std::size_t i = 0; i < k; i++) B[c * k + i] = i < c ? 0.0 : std::ldexp(U(rng), E(rng)); }});
  }
  for (int X : {0, 10, 20, 30, 40, 60, 80}) {
    cases.push_back({"rand_s" + std::to_string(X), 64, 64, 256, [&, X](auto& A, auto& B, std::size_t, std::size_t, std::size_t) {
      std::uniform_int_distribution<int> E(0, X);
      for (auto& x : A) x = std::ldexp(U(rng), E(rng));
      for (auto& x : B) x = std::ldexp(U(rng), E(rng)); }});
  }

  std::printf("threshold=%.3f   errors in units of u=2^-53 relative to (|A||B|)_ij; DGEMM a-priori bound ~ k\n", thr);
  for (auto& c : cases) {
    const std::size_t m = c.m, n = c.n, k = c.k;
    std::vector<double> A(m * k), B(k * n);
    c.fill(A, B, m, n, k);
    // reference and |A||B|
    std::vector<long double> ref(m * n), absAB(m * n);
    for (std::size_t j = 0; j < n; j++) for (std::size_t i = 0; i < m; i++) {
      long double s = 0, t = 0;
      for (std::size_t l = 0; l < k; l++) { const long double p = (long double)A[l * m + i] * B[j * k + l]; s += p; t += std::fabs(p); }
      ref[j * m + i] = s; absAB[j * m + i] = t;
    }
    std::uint64_t cnt[16] = {0};
    const unsigned bits = get_bits_per_int8(k);
    replica_rows(A, m, k, true, m, bits, cnt);
    replica_rows(B, n, k, false, k, bits, cnt);
    std::string want = "dgemm";
    for (unsigned s = 3; s <= 18; s++) if (cnt[s - 3] / static_cast<double>(m * k + k * n) <= thr) { want = "fp64_int8_" + std::to_string(s); break; }

    double *dA, *dB, *dC;
    cudaMalloc(&dA, A.size() * 8); cudaMalloc(&dB, B.size() * 8); cudaMalloc(&dC, m * n * 8);
    cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
    const std::string libsel = get_compute_mode_name_str(auto_mode_select(h, op_n, op_n, m, n, k, dA, m, dB, k, real, thr));

    auto run = [&](compute_mode_t mode, double& worst, double& c00) {
      reallocate_working_memory(h, gemm_list_t{gemm_params_t{op_n, op_n, m, n, k, real, mode}});
      const double alpha = 1, beta = 0;
      cudaMemset(dC, 0, m * n * 8);
      gemm(h, op_n, op_n, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m, mode, real);
      cudaDeviceSynchronize();
      std::vector<double> C(m * n);
      cudaMemcpy(C.data(), dC, m * n * 8, cudaMemcpyDeviceToHost);
      worst = 0;
      for (std::size_t q = 0; q < m * n; q++) {
        const long double e = std::fabs((long double)C[q] - ref[q]);
        const double r = absAB[q] > 0 ? (double)(e / absAB[q]) * std::ldexp(1.0, 53) : (e > 0 ? INFINITY : 0);
        if (r > worst) worst = r;
      }
      c00 = C[0];
    };
    std::printf("\n%-10s m=%zu n=%zu k=%zu bits/slice=%u  replica=%s  library auto_mode_select=%s\n", c.name.c_str(), m, n, k, bits, want.c_str(), libsel.c_str());
    std::printf("  ref C00 = %.6Le  |A||B|_00 = %.6Le\n", ref[0], absAB[0]);
    std::vector<std::pair<std::string, compute_mode_t>> modes = {{"auto", fp64_int8_auto}, {"dgemm", dgemm}};
    for (unsigned s : {3u, 5u, 8u, 10u, 11u, 13u, 15u, 18u}) modes.push_back({"int8_" + std::to_string(s), (compute_mode_t)(fp64_int8_3 + (s - 3))});
    for (auto& [nm, md] : modes) {
      double w, c00; run(md, w, c00);
      std::printf("  %-8s worst err/(u|A||B|) = %-12.4g C00 = %.6e  (%s)\n", nm.c_str(), w, c00, cudaGetErrorString(cudaGetLastError()));
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
  }
  destroy(h);
  return 0;
}

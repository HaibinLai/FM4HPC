// End-to-end accuracy probe of GEMMul8 (RIKEN-RCCS/GEMMul8 d8d3901, Ozaki scheme II, INT8 backend) on DGEMM inputs.
// For each case and each (num_moduli, fastmode) the worst entry is reported in two normalisations:
//   comp = |C - ref| / (u * (|A||B|)_ij)                 (componentwise, DGEMM a-priori bound ~ k)
//   bits = -log2( max |C - ref| / (||a_i||_2 ||b_j||_2) ) (normwise "effective bits"; README table: 14->55, 15->58, 16->62)
// plus the number of non-finite outputs and of entries that are finite in ref but wrong by > 100%.
// ref: long double (64-bit significand) accumulation.  Usage: gemmul8_probe [case-filter]
#include "gemmul8.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

struct Case {
  std::string name;
  std::size_t m, n, k;
  std::function<void(std::vector<double>&, std::vector<double>&, std::size_t, std::size_t, std::size_t)> fill;
};

int main(int argc, char** argv) {
  const std::string filter = argc > 1 ? argv[1] : "";
  cublasHandle_t h;
  cublasCreate(&h);
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> U(-1.0, 1.0), P(0.9, 1.0);
  auto ldexpU = [&](int lo, int hi) { std::uniform_int_distribution<int> E(lo, hi); return std::ldexp(U(rng), E(rng)); };

  std::vector<Case> cases;
  cases.push_back({"rand_s0", 128, 128, 512, [&](auto& A, auto& B, auto, auto, auto) {
    for (auto& x : A) x = U(rng); for (auto& x : B) x = U(rng); }});
  cases.push_back({"rand_s30", 128, 128, 512, [&](auto& A, auto& B, auto, auto, auto) {
    for (auto& x : A) x = ldexpU(0, 30); for (auto& x : B) x = ldexpU(0, 30); }});
  cases.push_back({"rand_s60", 128, 128, 512, [&](auto& A, auto& B, auto, auto, auto) {
    for (auto& x : A) x = ldexpU(0, 60); for (auto& x : B) x = ldexpU(0, 60); }});
  // Cauchy-Schwarz nearly tight: B = A^T, all entries positive -> C_ii = ||a_i||^2 (tests the CRT range margin)
  cases.push_back({"gram_pos", 128, 128, 4096, [&](auto& A, auto& B, std::size_t m, std::size_t, std::size_t k) {
    for (auto& x : A) x = P(rng);
    for (std::size_t i = 0; i < m; i++) for (std::size_t l = 0; l < k; l++) B[i * k + l] = A[l * m + i]; }});
  cases.push_back({"gram_ones", 64, 64, 1024, [&](auto& A, auto& B, auto, auto, auto) {
    for (auto& x : A) x = 1.0; for (auto& x : B) x = 1.0; }});
  // one bad row/column pair (row 0 of A = [1, 2^-100...], column 0 of B = [2^-100, 1...]); ref C00 = k*2^-100
  cases.push_back({"avg_hides", 64, 64, 64, [&](auto& A, auto& B, std::size_t m, std::size_t, std::size_t k) {
    for (auto& x : A) x = U(rng); for (auto& x : B) x = U(rng);
    for (std::size_t l = 0; l < k; l++) A[l * m] = l == 0 ? 1.0 : std::ldexp(1.0, -100);
    for (std::size_t l = 0; l < k; l++) B[l] = l == 0 ? std::ldexp(1.0, -100) : 1.0; }});
  // large magnitudes: A ~ 2^520, B ~ 2^-520 (C ~ O(1)); the squared row norm of A overflows binary64
  cases.push_back({"big520", 64, 64, 256, [&](auto& A, auto& B, auto, auto, auto) {
    for (auto& x : A) x = std::ldexp(U(rng), 520); for (auto& x : B) x = std::ldexp(U(rng), -520); }});
  // only row 0 of A is large: A[0,:] ~ 2^520
  cases.push_back({"big520_row0", 64, 64, 256, [&](auto& A, auto& B, std::size_t m, std::size_t, std::size_t k) {
    for (auto& x : A) x = U(rng); for (auto& x : B) x = U(rng);
    for (std::size_t l = 0; l < k; l++) A[l * m] = std::ldexp(U(rng), 520); }});
  // threshold sweep near 2^512 / sqrt(k)
  for (int e : {500, 505, 508, 511}) {
    cases.push_back({"big" + std::to_string(e), 64, 64, 256, [&, e](auto& A, auto& B, auto, auto, auto) {
      for (auto& x : A) x = std::ldexp(U(rng), e); for (auto& x : B) x = std::ldexp(U(rng), -e); }});
  }
  // small magnitudes: A ~ 2^-600, B ~ 2^600 (C ~ O(1)); the squared row norm of A underflows
  cases.push_back({"tinyA_bigB600", 64, 64, 256, [&](auto& A, auto& B, auto, auto, auto) {
    for (auto& x : A) x = std::ldexp(U(rng), -600); for (auto& x : B) x = std::ldexp(U(rng), 600); }});
  for (int e : {-540, -560, -580}) {
    cases.push_back({"tinyA_bigB" + std::to_string(-e), 64, 64, 256, [&, e](auto& A, auto& B, auto, auto, auto) {
      for (auto& x : A) x = std::ldexp(U(rng), e); for (auto& x : B) x = std::ldexp(U(rng), -e); }});
  }
  // pure underflow of the squared norm: A ~ 2^e (e very negative), B ~ O(1); C ~ 2^e is a normal binary64 number
  for (int e : {-530, -545, -600, -800, -1000}) {
    cases.push_back({"tinyA" + std::to_string(-e), 64, 64, 256, [&, e](auto& A, auto& B, auto, auto, auto) {
      for (auto& x : A) x = std::ldexp(U(rng), e); for (auto& x : B) x = U(rng); }});
  }
  // huge B only: A ~ O(1), B ~ 2^e with e near 512 (C ~ 2^e)
  for (int e : {505, 515}) {
    cases.push_back({"bigB" + std::to_string(e), 64, 64, 256, [&, e](auto& A, auto& B, auto, auto, auto) {
      for (auto& x : A) x = U(rng); for (auto& x : B) x = std::ldexp(U(rng), e); }});
  }
  // non-finite input: A[0,0] = +inf (DGEMM: row 0 of C is +-inf, or NaN where B[0,j] = 0)
  cases.push_back({"inf_a00", 64, 64, 256, [&](auto& A, auto& B, auto, auto, auto) {
    for (auto& x : A) x = U(rng); for (auto& x : B) x = U(rng); A[0] = INFINITY; }});
  cases.push_back({"nan_a00", 64, 64, 256, [&](auto& A, auto& B, auto, auto, auto) {
    for (auto& x : A) x = U(rng); for (auto& x : B) x = U(rng); A[0] = NAN; }});

  std::printf("GEMMul8 version header: see include/version.hpp; u = 2^-53\n");
  for (auto& c : cases) {
    if (!filter.empty() && c.name.find(filter) == std::string::npos) continue;
    const std::size_t m = c.m, n = c.n, k = c.k;
    std::vector<double> A(m * k), B(k * n);
    c.fill(A, B, m, n, k);
    std::vector<long double> ref(m * n), absAB(m * n);
    std::vector<long double> na(m, 0), nb(n, 0);  // long double: no overflow at 2^1022
    for (std::size_t i = 0; i < m; i++) { long double s = 0; for (std::size_t l = 0; l < k; l++) s += (long double)A[l * m + i] * A[l * m + i]; na[i] = sqrtl(s); }
    for (std::size_t j = 0; j < n; j++) { long double s = 0; for (std::size_t l = 0; l < k; l++) s += (long double)B[j * k + l] * B[j * k + l]; nb[j] = sqrtl(s); }
    for (std::size_t j = 0; j < n; j++) for (std::size_t i = 0; i < m; i++) {
      long double s = 0, t = 0;
      for (std::size_t l = 0; l < k; l++) { const long double p = (long double)A[l * m + i] * B[j * k + l]; s += p; t += std::fabs(p); }
      ref[j * m + i] = s; absAB[j * m + i] = t;
    }
    double *dA, *dB, *dC;
    cudaMalloc(&dA, A.size() * 8); cudaMalloc(&dB, B.size() * 8); cudaMalloc(&dC, m * n * 8);
    cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
    const double one = 1, zero = 0;
    std::printf("\n%-12s m=%zu n=%zu k=%zu  ref C00=%.6Le |A||B|00=%.3Le ||a0||=%.3Le ||b0||=%.3Le\n", c.name.c_str(), m, n, k,
                ref[0], absAB[0], na[0], nb[0]);
    auto evaluate = [&](const char* label) {
      std::vector<double> C(m * n);
      cudaMemcpy(C.data(), dC, m * n * 8, cudaMemcpyDeviceToHost);
      double comp = 0, normw = 0; long nonfinite = 0, wrong = 0, ref_nonfinite = 0;
      for (std::size_t j = 0; j < n; j++) for (std::size_t i = 0; i < m; i++) {
        const std::size_t q = j * m + i;
        if (!std::isfinite((double)ref[q])) { ref_nonfinite++; if (std::isfinite(C[q])) wrong++; continue; }
        if (!std::isfinite(C[q])) { nonfinite++; continue; }
        const long double e = std::fabs((long double)C[q] - ref[q]);
        if (absAB[q] > 0) comp = std::fmax(comp, (double)(e / absAB[q]) * std::ldexp(1.0, 53));
        else if (e > 0) comp = INFINITY;
        const long double den = na[i] * nb[j];
        if (den > 0) normw = std::fmax(normw, (double)(e / den));
        if (ref[q] != 0 && e >= std::fabs(ref[q])) wrong++;
      }
      std::printf("  %-10s comp=%-11.4g bits=%-6.1f nonfinite=%ld wrong>=100%%=%ld (ref nonfinite %ld)  C00=%.6e\n", label, comp,
                  normw > 0 ? -std::log2(normw) : INFINITY, nonfinite, wrong, ref_nonfinite, C[0]);
    };
    cudaMemset(dC, 0, m * n * 8);
    cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, dA, m, dB, k, &zero, dC, m);
    cudaDeviceSynchronize();
    evaluate("DGEMM");
    for (int fast = 1; fast >= 0; fast--) {
      for (int nm : {8, 12, 14, 16, 20}) {
        auto ws = gemmul8::gemm<double, gemmul8::Backend::INT8>(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, dA, m, dB, k, &zero, dC, m,
                                                                nm, fast == 1, nullptr);
        void* work = nullptr;
        cudaMalloc(&work, (size_t)ws[0]);
        cudaMemset(dC, 0, m * n * 8);
        gemmul8::gemm<double, gemmul8::Backend::INT8>(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, dA, m, dB, k, &zero, dC, m, nm,
                                                      fast == 1, work);
        cudaError_t err = cudaDeviceSynchronize();
        char label[32];
        std::snprintf(label, sizeof label, "%s_N%d", fast ? "fast" : "accu", nm);
        if (err != cudaSuccess) std::printf("  %s CUDA error: %s\n", label, cudaGetErrorString(err));
        evaluate(label);
        cudaFree(work);
      }
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
  }
  cublasDestroy(h);
  return 0;
}

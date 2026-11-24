// Forward-looking test of MAGMA mixed-precision iterative refinement (contract: CONTRACTS.md section 6).
// Contract prediction: MAGMA accepts when NOT (Rnrm > Xnrm*cte) (dsgesv_gpu.cpp:276, dxgesv_gmres_gpu.cpp:648/755),
// so a NaN residual or an Inf in X (Xnrm*cte = Inf) is accepted as converged.  Low-precision factorization only
// reports exactly-zero pivots, so overflow of the factors to Inf/NaN is not caught before the check.
// Inputs: same as cusolver_irs_probe.py / cusolver_irs_round2.py (scaled Wilkinson s*W(n), block-diagonal
// diag(s*W(m), R) with n=1024, random diagonally dominant control, NaN in b).  x_true = ones, b = A x_true (exact).
// Routines: magma_dgesv_gpu (FP64 reference), magma_dsgesv_gpu (classic FP32 IR),
//           magma_dsgesv_iteref_gpu (FP32 + GMRES-IR), magma_dhgesv_iteref_gpu (FP32 LU with FP16 tensor-core updates).
#include <magma_v2.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Case { std::string name; int n; std::vector<double> A; std::vector<double> b; };

static std::vector<double> wilkinson_block(int n, int m, double s, int ld) {  // embeds s*W(m) at top-left of n x n
  std::vector<double> A((size_t)ld * n, 0.0);
  for (int j = 0; j < m; j++)
    for (int i = 0; i < m; i++) {
      double v = (i == j) ? 1.0 : (i > j ? -1.0 : 0.0);
      if (j == m - 1) v = 1.0;
      A[(size_t)j * ld + i] = v * s;
    }
  return A;
}

int main() {
  magma_init();
  magma_queue_t q;
  magma_queue_create(0, &q);
  std::mt19937_64 rng(4);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  std::vector<Case> cases;
  auto add = [&](std::string name, int n, std::vector<double> A, bool nan_b = false) {
    std::vector<double> b(n, 0.0);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) b[i] += A[(size_t)j * n + i];  // A * ones
    if (nan_b) b[0] = NAN;
    cases.push_back({name, n, A, b});
  };
  for (auto [n, e] : std::vector<std::pair<int, int>>{{10, 10}, {20, 0}, {40, 0}, {16, 0}, {40, 80}, {40, 100}, {30, 120}})
    add("W(" + std::to_string(n) + ")*2^" + std::to_string(e) + " growth 2^" + std::to_string(n - 1 + e), n,
        wilkinson_block(n, n, std::ldexp(1.0, e), n));
  {
    int n = 256;
    std::vector<double> A((size_t)n * n);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) A[(size_t)j * n + i] = U(rng) + (i == j ? n : 0.0);
    add("random diag-dominant 256", n, A);
    add("random 256 with b[0]=NaN", n, A, true);
  }
  for (auto [m, e] : std::vector<std::pair<int, int>>{{0, 0}, {20, 0}, {40, 0}, {40, 100}}) {
    int n = 1024;
    std::vector<double> A = wilkinson_block(n, m, std::ldexp(1.0, e), n);
    for (int j = 0; j < n; j++)
      for (int i = 0; i < n; i++)
        if (i >= m && j >= m) A[(size_t)j * n + i] = 0.01 * U(rng) + (i == j ? n : 0.0);
    add(m ? "diag(W(" + std::to_string(m) + ")*2^" + std::to_string(e) + ", R) n=1024" : std::string("R only n=1024"), n, A);
  }

  for (auto& c : cases) {
    const int n = c.n, nrhs = 1, ldda = magma_roundup(n, 32);
    double *dA, *dB, *dX, *dWD; float* dWS; magma_int_t* dipiv;
    magma_dmalloc(&dA, (size_t)ldda * n); magma_dmalloc(&dB, ldda); magma_dmalloc(&dX, ldda);
    magma_dmalloc(&dWD, (size_t)n * nrhs); magma_smalloc(&dWS, (size_t)ldda * (n + nrhs)); magma_imalloc(&dipiv, n);
    std::vector<magma_int_t> ipiv(n);
    std::printf("\n%s\n", c.name.c_str());
    for (int r = 0; r < 4; r++) {
      magma_dsetmatrix(n, n, c.A.data(), n, dA, ldda, q);
      magma_dsetmatrix(n, 1, c.b.data(), n, dB, ldda, q);
      std::vector<double> fill(n, 777.0);
      magma_dsetmatrix(n, 1, fill.data(), n, dX, ldda, q);
      magma_int_t iter = 12345, info = -999;
      const char* label = "";
      if (r == 0) {
        label = "magma_dgesv_gpu (FP64)";
        magma_dgesv_gpu(n, 1, dA, ldda, ipiv.data(), dB, ldda, &info);
        magma_dcopymatrix(n, 1, dB, ldda, dX, ldda, q);
        iter = 0;
      } else if (r == 1) {
        label = "magma_dsgesv_gpu (FP32 IR)";
        magma_dsgesv_gpu(MagmaNoTrans, n, 1, dA, ldda, ipiv.data(), dipiv, dB, ldda, dX, ldda, dWD, dWS, &iter, &info);
      } else if (r == 2) {
        label = "magma_dsgesv_iteref_gpu";
        magma_dsgesv_iteref_gpu(MagmaNoTrans, n, 1, dA, ldda, ipiv.data(), dipiv, dB, ldda, dX, ldda, dWD, dWS, &iter, &info);
      } else {
        label = "magma_dhgesv_iteref_gpu (FP16-TC)";
        magma_dhgesv_iteref_gpu(MagmaNoTrans, n, 1, dA, ldda, ipiv.data(), dipiv, dB, ldda, dX, ldda, dWD, dWS, &iter, &info);
      }
      magma_queue_sync(q);
      std::vector<double> x(n);
      magma_dgetmatrix(n, 1, dX, ldda, x.data(), n, q);
      int bad = 0; double err = 0;
      for (int i = 0; i < n; i++) {
        if (!std::isfinite(x[i])) { bad++; continue; }
        err = std::fmax(err, std::fabs(x[i] - 1.0));
      }
      std::printf("  %-34s iter=%6lld info=%4lld  nonfinite=%4d/%d max|x-1|(finite)=%.3g  x[0]=%g x[n-1]=%g\n", label,
                  (long long)iter, (long long)info, bad, n, err, x[0], x[n - 1]);
      std::fflush(stdout);
    }
    magma_free(dA); magma_free(dB); magma_free(dX); magma_free(dWD); magma_free(dWS); magma_free(dipiv);
  }
  magma_queue_destroy(q);
  magma_finalize();
  return 0;
}

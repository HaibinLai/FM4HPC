// Twenty-fourth GPU round: finding #6, 32-bit size arithmetic in the divide-and-conquer eigensolver (HANDOFF §29.26).
// magma_zstedx_m(ngpu = 1) keeps Z on the host; only the D&C workspace dw = 3n(n/2+1) doubles goes to the GPU, and that
// size is computed in 32-bit magma_int_t (dlaex0_m.cpp l.151).  GPU 0 only.
#include "magma_v2.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
typedef magmaDoubleComplex Z;
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0); int n = argc > 1 ? atoi(argv[1]) : 1000; magma_init();
  long long want = 3LL * n * (n / 2 + 1); int as32 = (int)(unsigned)(want & 0xffffffffLL);
  long long lrw = 1LL + 4LL * n + 2LL * n * n; int lrw32 = (int)(unsigned)(lrw & 0xffffffffLL);
  printf("n=%d: D&C device workspace 3n(n/2+1) = %lld doubles, as 32-bit int = %d; lrwmin = %lld, as 32-bit int = %d\n", n, want, as32, lrw, lrw32);
  std::vector<double> d(n), e(n - 1); unsigned s = 12345u;
  for (auto& x : d) { s = s * 1664525u + 1013904223u; x = 1 + ((s >> 8) & 0xffffff) / (double)0x1000000; }
  for (auto& x : e) { s = s * 1664525u + 1013904223u; x = 0.5 * ((s >> 8) & 0xffffff) / (double)0x1000000; }
  std::vector<double> wref = d, eref = e; { magma_int_t nn = n, info; lapackf77_dsterf(&nn, wref.data(), eref.data(), &info); std::sort(wref.begin(), wref.end()); }
  Z* Zm = (Z*)malloc((size_t)n * n * sizeof(Z)); double* rw = (double*)malloc((size_t)lrw * sizeof(double)); std::vector<magma_int_t> iw(3 + 5 * (size_t)n);
  if (!Zm || !rw) { printf("host allocation failed\n"); return 0; }
  magma_int_t info = -999; double t0 = magma_wtime();
  magma_zstedx_m(1, MagmaRangeAll, n, 0, 0, 0, 0, d.data(), e.data(), Zm, n, rw, INT_MAX, iw.data(), (magma_int_t)iw.size(), &info);
  cudaError_t ce = cudaDeviceSynchronize(), le = cudaGetLastError();
  printf("magma_zstedx_m(ngpu=1, range All): info=%lld time=%.1f s  cudaDeviceSynchronize=%s lastError=%s\n", (long long)info, magma_wtime() - t0,
         cudaGetErrorString(ce), cudaGetErrorString(le));
  if (info == 0) {
    std::vector<double> w = d; std::sort(w.begin(), w.end()); double ew = 0; int nf = 0;
    for (int i = 0; i < n; i++) { if (!std::isfinite(w[i])) nf++; else ew = fmax(ew, fabs(w[i] - wref[i])); }
    double orth = 0; int cols[3] = {0, n / 2, n - 1}; int nfz = 0;
    for (int a : cols) for (int b : cols) { double re = 0, im = 0;
      for (size_t i = 0; i < (size_t)n; i++) { Z x = Zm[i + (size_t)a * n], y = Zm[i + (size_t)b * n];
        re += MAGMA_Z_REAL(x) * MAGMA_Z_REAL(y) + MAGMA_Z_IMAG(x) * MAGMA_Z_IMAG(y); im += MAGMA_Z_REAL(x) * MAGMA_Z_IMAG(y) - MAGMA_Z_IMAG(x) * MAGMA_Z_REAL(y);
        if (a == 0 && b == 0 && !std::isfinite(MAGMA_Z_REAL(x))) nfz++; }
      orth = fmax(orth, fabs(hypot(re, im) - (a == b ? 1.0 : 0.0))); }
    printf("  eigenvalues: non-finite %d, max|w - w_dsterf| = %.3g;  eigenvectors (columns 0, n/2, n-1): max|<z_a,z_b> - delta| = %.3g, non-finite in column 0: %d\n",
           nf, ew, orth, nfz);
  }
  magma_finalize(); return 0;
}

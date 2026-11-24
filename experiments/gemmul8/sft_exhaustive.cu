// Exhaustive check, on the GPU, of GEMMul8's floating-point shift computations (the library's own device functions),
// INT8 backend, real, num_moduli N = 2..20.  T[N] = round-down((P_N - 1)/2) as binary64 (P_N = product of the moduli).
//
// fast mode  (scaling/fast/calc_sft.hpp): sft = calc_sft(v), v = rounded-up squared row/column 2-norm.
//   The result depends on v only through (ex, vf) = (ilogb v, float_ru(v * 2^-ex)), vf in [1, 2].
//   Enumerated: every ex in [-1074, 1023] and every float vf in [1, 2) (2^23 values) plus vf = 2.0
//   (realised by v = (2 - 2^-40) * 2^ex).  Claim checked: 2^ex * vf * 4^sft < T[N] + 1, so the INTEGER sum of squares of the
//   scaled, truncated row is <= floor((P-1)/2) (also reported: how often 2^ex * vf * 4^sft > T[N]).
// accurate mode (scaling/accu/calc_sft.hpp): cor = calc_sft(amax), amax = int32 row/column maximum of the INT8 product
//   of rounded-up magnitudes.  Enumerated: every amax in [1, 2^31 - 1].  Claim checked: amax * 4^cor < T[N] + 1.
// Also reported: the largest and smallest value of lhs / T[N] (how tight the chosen shift is) and the range of shifts.
#include "oz2/scaling/fast/calc_sft.hpp"
#include "oz2/scaling/accu/calc_sft.hpp"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <utility>

__constant__ double Tc[21];

struct Stat {
  unsigned long long viol, count, viol_strict;
  unsigned long long max_ratio_bits, min_ratio_bits;  // positive doubles: order of bits == order of values
  int min_sft, max_sft;
  double first_bad;
};

struct Local {
  unsigned long long viol = 0, count = 0, strict = 0, rmax = 0ULL, rmin = ~0ULL;
  int smin = 1 << 30, smax = -(1 << 30);
  double bad = 0.0;
  __device__ void add(double lhs, double T, int s, double arg) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(lhs / T);
    rmax = b > rmax ? b : rmax;
    rmin = b < rmin ? b : rmin;
    smin = s < smin ? s : smin;
    smax = s > smax ? s : smax;
    // integer criterion: the integer sum of squares (resp. |c|) is <= floor(lhs), so lhs < T + 1 suffices
    if (lhs > T) strict++;
    if (lhs >= T + 1.0) { if (viol == 0) bad = arg; viol++; }
    count++;
  }
  __device__ void flush(Stat* st) {
    atomicAdd(&st->count, count);
    atomicAdd(&st->viol_strict, strict);
    if (count == 0) return;
    atomicMax(&st->max_ratio_bits, rmax);
    atomicMin(&st->min_ratio_bits, rmin);
    atomicMin(&st->min_sft, smin);
    atomicMax(&st->max_sft, smax);
    if (viol) { if (atomicAdd(&st->viol, viol) == 0ULL) st->first_bad = bad; }
  }
};

template <unsigned N>
__global__ void fast_kernel(Stat* st) {
  const double T = Tc[N];
  const long long per_ex = (1LL << 23) + 1;  // 2^23 floats in [1,2) plus vf = 2.0
  const long long total = 2098LL * per_ex;
  Local L;
  for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    const int ex = int(t / per_ex) - 1074;
    const long long i = t % per_ex;
    const double sig = i < (1LL << 23) ? 1.0 + double(i) * 0x1p-23 : 2.0 - 0x1p-40;
    const double v = ldexp(sig, ex);
    if (v == 0.0 || !isfinite(v)) continue;
    const int s = gemmul8::scaling::fast::calc_sft<gemmul8::Backend::INT8, N, double, false, false>(v);
    const int e2 = ilogb(v);
    const float vf = __double2float_ru(scalbn(v, -e2));
    L.add(ldexp((double)vf, e2 + 2 * s), T, s, v);
  }
  L.flush(st);
}

template <unsigned N>
__global__ void accu_kernel(Stat* st) {
  const double T = Tc[N];
  Local L;
  for (long long a = 1 + blockIdx.x * (long long)blockDim.x + threadIdx.x; a <= 0x7fffffffLL; a += (long long)gridDim.x * blockDim.x) {
    const int c = gemmul8::scaling::accu::calc_sft<N, gemmul8::Backend::INT8, false>(int32_t(a));
    L.add(ldexp((double)a, 2 * c), T, c, (double)a);
  }
  L.flush(st);
}

template <unsigned N>
void run_one() {
  Stat h0{0, 0, 0, 0ULL, ~0ULL, 1 << 30, -(1 << 30), 0.0};
  Stat* d;
  cudaMalloc(&d, sizeof(Stat));
  for (int mode = 0; mode < 2; mode++) {
    cudaMemcpy(d, &h0, sizeof(Stat), cudaMemcpyHostToDevice);
    if (mode == 0) fast_kernel<N><<<80 * 32, 256>>>(d); else accu_kernel<N><<<80 * 32, 256>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    Stat h;
    cudaMemcpy(&h, d, sizeof(Stat), cudaMemcpyDeviceToHost);
    double rmax, rmin;
    std::memcpy(&rmax, &h.max_ratio_bits, 8);
    std::memcpy(&rmin, &h.min_ratio_bits, 8);
    std::printf("N=%2u %-4s checked=%llu violations(lhs>=T+1)=%llu [lhs>T: %llu]  lhs/T in [%.6f, %.9f]  shift in [%d, %d]%s%s\n", N,
                mode == 0 ? "fast" : "accu", h.count, h.viol, h.viol_strict, rmin, rmax, h.min_sft, h.max_sft,
                e == cudaSuccess ? "" : "  CUDA ERROR ", e == cudaSuccess ? "" : cudaGetErrorString(e));
    if (h.viol) std::printf("      first violating argument: %a\n", h.first_bad);
    std::fflush(stdout);
  }
  cudaFree(d);
}

template <unsigned... Ns>
void run_all(std::integer_sequence<unsigned, Ns...>) { (run_one<Ns + 2>(), ...); }

int main() {
  // round-down((P_N - 1)/2) for N = 2..20 (computed exactly with Python big integers)
  const double T[21] = {0, 0, 0x1.fdfc000000000p+14, 0x1.f805fc0000000p+22, 0x1.ee2de1fc00000p+30, 0x1.dcce450dfc000p+38,
                        0x1.c0de2f022dfc0p+46, 0x1.a30f6de308f1fp+54, 0x1.7d690b03a3244p+62, 0x1.552ef6da40ef6p+70,
                        0x1.2e88a4e387944p+78, 0x1.078907a2331a3p+86, 0x1.bec64ef0faa26p+93, 0x1.703d73109e93dp+101,
                        0x1.1e3fc471eb44ep+109, 0x1.b88e245754181p+116, 0x1.4c232965d6662p+124, 0x1.ef9c77c5f5ec7p+131,
                        0x1.5e69a0aef6e02p+139, 0x1.ea07b6b4ad3d7p+146, 0x1.4b27367819128p+154};
  cudaMemcpyToSymbol(Tc, T, sizeof(T));
  run_all(std::make_integer_sequence<unsigned, 19>{});
  return 0;
}

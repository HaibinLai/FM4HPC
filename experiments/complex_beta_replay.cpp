#include <cstdio>
#include <cstddef>
#define __global__
struct Dim { unsigned x; };
Dim blockIdx{0}, blockDim{1}, threadIdx{0};
struct cuDoubleComplex { double x,y; };
cuDoubleComplex make_cuDoubleComplex(double x,double y) { return {x,y}; }
template <bool is_beta_zero>
__global__ void init_c_complex_kernel(const std::size_t m, const std::size_t n,
                                      cuDoubleComplex *const c_ptr,
                                      const std::size_t ldc,
                                      const cuDoubleComplex beta) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= m * n) {
    return;
  }

  const auto mi = tid % m;
  const auto ni = tid / m;

  const auto memory_index = ni * ldc + mi;

  if (is_beta_zero) {
    c_ptr[memory_index] = make_cuDoubleComplex(0, 0);
  } else {
    auto c = c_ptr[memory_index];
    c.x = c.x * beta.x - c.y * beta.y;
    c.y = c.y * beta.x + c.x * beta.y;

    c_ptr[memory_index] = c;
  }
}
int main() {
  cuDoubleComplex c{1,2}, beta{3,4};
  init_c_complex_kernel<false>(1,1,&c,1,beta);
  std::printf("%.17g %.17g\n",c.x,c.y);
}

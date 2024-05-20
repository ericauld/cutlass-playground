#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

__global__ static
void
gemm_kernel(cute::half_t const *d_q,
            cute::half_t const *d_K) {
  using namespace cute;

  return;
}

int main() {
  using namespace cute;

  int n = 256;
  int d = 128;

  using TA = half_t;

  cute::device_init(0);

  thrust::host_vector<TA> h_q(d);
  thrust::host_vector<TA> h_K(n*d);

  for (int j = 0; j < d; ++j) h_q[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*d; ++j) h_K[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );

  thrust::device_vector<TA> d_q = h_q;
  thrust::device_vector<TA> d_K = h_K;

  dim3 dimBlock(1);
  dim3 dimGrid(1);

  gemm_kernel<<<dimGrid, dimBlock>>>(d_q.data().get(), d_K.data().get());

  return 0;
}
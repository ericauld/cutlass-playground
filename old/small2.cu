#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// # define K
// # define bN
// # define M

__global__ static
void
gemm_kernel(cute::half_t const *d_q,
            cute::half_t const *d_K,
            cute::half_t *d_S) {
  using namespace cute;

  int i = blockIdx.x;
  

}

int main() {
  using namespace cute;

  int n = 256;
  int d = 128;

  using TA = half_t;

  cute::device_init(0);

  thrust::host_vector<TA> h_q(d);
  thrust::host_vector<TA> h_Kt(d*n);
  thrust::host_vector<TA> h_S(n);

  for (int j = 0; j < d; ++j) h_q[j] = 1;
  for (int j = 0; j < d*n; ++j) h_Kt[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n; ++j) h_S[j] = 0;

  thrust::device_vector<TA> d_q = h_q;
  thrust::device_vector<TA> d_Kt = h_Kt;
  thrust::device_vector<TA> d_S = h_S;

  dim3 dimBlock(256);
  dim3 dimGrid(128);

  gemm_kernel<<<dimGrid, dimBlock>>>(d_q.data().get(), d_Kt.data().get(), d_S.data().get());

  h_S = d_S;
  assert(h_S[0] == 1);
  return 0;
}
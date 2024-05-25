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

/*
I'd like to follow a tiled mma w blayout (1, 1, 4: whatever) and see how the
code takes care of adding the terms that wind up in the D registers.

I'd like to pay closer attention to which of the cutlass examples use tiled
mma's that operate on rmem and which on smem, and ask why.
*/

template <class TiledMma>
__global__ static
void
f(cute::half_t const *A,
  cute::half_t const *B,
  cute::half_t       *C,
  TiledMma            my_mma) {
  using namespace cute;

  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(8, 16));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(8, 16));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(8, 8)); 

  __shared__ half_t smemA[8*16];
  __shared__ half_t smemB[8*16];
  Tensor sA = make_tensor(make_smem_ptr(smemA), make_shape(8, 16));
  Tensor sB = make_tensor(make_smem_ptr(smemB), make_shape(8, 16));
  
  copy(mA, sA);
  copy(mB, sB);
  
#if 1
  if(thread0()) {
    print("my_mma: "); print(my_mma); print("\n");
    print("sA: "); print(sA); print("\n");
    print("my_mma.thrfrg_A(sA) : "); print(my_mma.thrfrg_A(sA)); print("\n");
  }
#endif

#if 0
  gemm(my_mma, sA, sB, rC);
#endif

}

int main() {
  using namespace cute;

  int M = 8;
  int N = 8;
  int K = 16;

  using TA = half_t;

  thrust::host_vector<TA> h_A(M*K);
  thrust::host_vector<TA> h_B(K*N);
  thrust::host_vector<TA> h_C(M*N);

  for (int j = 0; j < M*K; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < N*K; ++j) h_B[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < M*N; ++j) h_C[j] = 0;

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TA> d_B = h_B;
  thrust::device_vector<TA> d_C = h_C;

  using op = SM70_8x8x4_F16F16F16F16_NT;
  auto tiled_mma = make_tiled_mma(op{}, make_layout(make_shape(_1{}, _1{}, _4{}))); 

  dim3 dimGrid(1);
  dim3 dimBlock(32);
  f<<<dimGrid, dimBlock>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), tiled_mma);

  return 0;
}
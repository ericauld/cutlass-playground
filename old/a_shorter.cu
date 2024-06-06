#include <iostream>
#include <cassert>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

template <class TiledMma>
__global__ static
void
f(cute::half_t const *A,
  cute::half_t const *B,
  cute::half_t       *C,
  TiledMma            my_mma) {
  using namespace cute;

  // mA is k-major, i.e. "row major" i.e. "not transposed"
  Tensor mA = make_tensor(make_gmem_ptr(A), 
                          make_layout(make_shape(_16{}, _16{}), make_stride(_16{}, _1{})));
  // mB is k-major, i.e. "column major" i.e. "transposed"
  Tensor mB = make_tensor(make_gmem_ptr(B), 
                          make_layout(make_shape(_8{}, _16{}), make_stride(_16{}, _1{})));
  // mC is n-major, i.e. "row major"
  Tensor mC = make_tensor(make_gmem_ptr(C), 
                          make_layout(make_shape(_16{}, _8{}), make_stride(_8{}, _1{})));
  auto thrmma = my_mma.get_slice(threadIdx.x);

  auto rC = thrmma.partition_fragment_C(mC);
  clear(rC);
  auto rA = thrmma.partition_fragment_A(mA);
  auto rB = thrmma.partition_fragment_B(mB);
  auto tCmC = thrmma.partition_C(mC);
  auto tCmA = thrmma.partition_A(mA);
  auto tCmB = thrmma.partition_B(mB);

  copy(tCmA, rA);
  copy(tCmB, rB);
  gemm(my_mma, rA, rB, rC);
  copy(rC, tCmC);
  return;
}

int main() {
  using namespace cute;

  int m = 16;
  int n = 8;
  int k = 16;

  using TA = half_t;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TA> h_B(k*n);
  thrust::host_vector<TA> h_C(m*n);

  for (int j = 0; j < m*k; ++j) 
    h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) 
    h_B[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) 
    h_C[j] = 0;

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TA> d_B = h_B;
  thrust::device_vector<TA> d_C = h_C;

  using op = SM80_16x8x16_F16F16F16F16_TN;
  auto tiled_mma = make_tiled_mma(op{}, make_layout(make_shape(_1{}, _1{}, _1{}))); 

  dim3 dimGrid(1);
  dim3 dimBlock(32);
  
  f<<<dimGrid, dimBlock>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), 
                           tiled_mma);

  thrust::host_vector<TA> cute_result = d_C;
  return 0;
}

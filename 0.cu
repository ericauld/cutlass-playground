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

  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(_8{}, _16{}));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(_8{}, _16{}));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(_16{}, _8{})); 
  auto thrmma = my_mma.get_slice(threadIdx.x);

  auto rC = thrmma.partition_fragment_C(mC);
  auto rA = thrmma.partition_fragment_A(mA);
  auto rB = thrmma.partition_fragment_B(mB);
  auto tCmC = thrmma.partition_C(mC);
  auto tCmA = thrmma.partition_A(mA);
  auto tCmB = thrmma.partition_B(mB);

  gemm(my_mma, rA, rB, rC);
  copy(rC, tCmC);
  return;
}

int main() {
  using namespace cute;

  int M = 16;
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
  // Why am I copying a bunch of zeros from host to device?
  thrust::device_vector<TA> d_C = h_C;

  using op = SM80_16x8x16_F16F16F16F16_TN;
  auto tiled_mma = make_tiled_mma(op{}, make_layout(make_shape(_1{}, _1{}, _1{}))); 

  dim3 dimGrid(1);
  dim3 dimBlock(32);
  f<<<dimGrid, dimBlock>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), tiled_mma);
  return 0;
}
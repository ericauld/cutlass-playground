#include <iostream>
#include <cassert>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

void matrix_multiply_cpu(const cute::half_t* A, const cute::half_t* B, cute::half_t* C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      cute::half_t sum = static_cast<cute::half_t>(0.0f);
      for (int p = 0; p < k; ++p) {
        sum += A[i * k + p] * B[j * k + p];
      }
      C[i * n + j] = sum;
    }
  }
}

bool areMatricesEqual(const cute::half_t* C1, const cute::half_t* C2, int m, int n, float tolerance = 1e-2) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (std::fabs(static_cast<float>(C1[i * n + j]) - static_cast<float>(C2[i * n + j])) > tolerance) {
        return false;
      }
    }
  }
  return true;
}

template <class TiledMma>
__global__ static
void
f(cute::half_t const *A,
  cute::half_t const *B,
  cute::half_t       *C,
  int m, int n,
  TiledMma            my_mma) {
  using namespace cute;

  Tensor mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(m, _16{}), make_stride(_16{}, _1{})));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(n, _16{}), make_stride(_16{}, _1{})));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(m, n), make_stride(n, _1{})));

  auto thrmma = my_mma.get_slice(threadIdx.x);

  auto cta_tiler = make_shape(_16{}, _8{}, _16{});
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  Tensor tCgC = thrmma.partition_C(gC);

  auto rC = thrmma.partition_fragment_C(gC);
  clear(rC);
  auto rA = thrmma.partition_fragment_A(gA);
  auto rB = thrmma.partition_fragment_B(gB);
  auto tCgA = thrmma.partition_A(gA);
  auto tCgB = thrmma.partition_B(gB);

#if 1
  if (thread0()) {
    print("mA : "); print(mA); print("\n");
    print("mB : "); print(mB); print("\n");
    print("mC : "); print(mC); print("\n");
    print("gA : "); print(gA); print("\n");
    print("gB : "); print(gB); print("\n");
    print("gC : "); print(gC); print("\n");
    print("rA : "); print(rA); print("\n");
    print("rB : "); print(rB); print("\n");
    print("rC : "); print(rC); print("\n");
    print("tCgA : "); print(tCgA); print("\n");
    print("tCgB : "); print(tCgB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
  }
#endif
/*
mA : gmem_ptr[16b](0x7f280bc00000) o (80,_16):(_16,_1)
mB : gmem_ptr[16b](0x7f280bc00a00) o (48,_16):(_16,_1)
mC : gmem_ptr[16b](0x7f280bc01000) o (80,48):(48,_1)
gA : gmem_ptr[16b](0x7f280bc00000) o (_16,_16,_1):(_16,_1,_0)
gB : gmem_ptr[16b](0x7f280bc00a00) o (_8,_16,_1):(_16,_1,_0)
gC : gmem_ptr[16b](0x7f280bc01000) o (_16,_8):(48,_1)
rA : ptr[16b](0x7f282ffffcb0) o ((_2,_2,_2),_1,_1,_1):((_1,_2,_4),_0,_0,_0)
rB : ptr[16b](0x7f282ffffcc0) o ((_2,_2),_1,_1,_1):((_1,_2),_0,_0,_0)
rC : ptr[16b](0x7f282ffffca0) o ((_2,_2),_1,_1):((_1,_2),_0,_0)
tCgA : gmem_ptr[16b](0x7f280bc00000) o ((_2,_2,_2),_1,_1,_1):((_1,_128,_8),_0,_0,_0)
tCgB : gmem_ptr[16b](0x7f280bc00a00) o ((_2,_2),_1,_1,_1):((_1,_8),_0,_0,_0)
tCgC : gmem_ptr[16b](0x7f280bc01000) o ((_2,_2),_1,_1):((_1,384),_0,_0)

argument types are:

const cute::MMA_Atom<cute::SM80_16x8x16_F16F16F16F16_TN>,
    cute::Tensor<cute::ArrayEngine<cutlass::half_t, 4>,
                 cute::Layout<cute::tuple<cute::tuple<cute::_2, cute::_2>,
                                          cute::_1, cute::_1>,
                              cute::tuple<cute::tuple<cute::_1, cute::_2>,
                                          cute::_0, cute::C<0>>>>,
    const cute::Tensor<
        cute::ArrayEngine<cutlass::half_t, 8>,
        cute::Layout<cute::tuple<cute::tuple<cute::_2, cute::_2, cute::_2>,
                                 cute::_1, cute::_1, cute::_1>,
                     cute::tuple<cute::tuple<cute::_1, cute::_2, cute::_4>,
                                 cute::C<0>, cute::_0, cute::C<0>>>>,
    const cute::Tensor<
        cute::ArrayEngine<cutlass::half_t, 4>,
        cute::Layout<cute::tuple<cute::tuple<cute::_2, cute::_2>, cute::_1,
                                 cute::_1, cute::_1>,
                     cute::tuple<cute::tuple<cute::_1, cute::_2>, cute::C<0>,
                                 cute::_0, cute::C<0>>>>,
    cute::Tensor<
        cute::ArrayEngine<cutlass::half_t, 4>,
        cute::Layout<
            cute::tuple<cute::tuple<cute::_2, cute::_2>, cute::_1, cute::_1>,
            cute::tuple<cute::tuple<cute::_1, cute::_2>, cute::_0, cute::C<0>>>>

During instantiation of:

    void
    cute::gemm(const cute::MMA_Atom<MMA> &, const cute::Tensor<TA, ALayout> &,
               const cute::Tensor<TB, BLayout> &, cute::Tensor<TC, CLayout> &)

with

        MMA = cute::SM80_16x8x16_F16F16F16F16_TN

    TA = cute::ArrayEngine<cutlass::half_t, 8>

    ALayout =
        cute::Layout<cute::tuple<cute::tuple<cute::_2, cute::_2, cute::_2>,
                                 cute::_1, cute::_1, cute::_1>,
                     cute::tuple<cute::tuple<cute::_1, cute::_2, cute::_4>,
                                 cute::C<0>, cute::_0, cute::C<0>>>

    TB = cute::ArrayEngine<cutlass::half_t, 4>

    BLayout = cute::Layout<cute::tuple<cute::tuple<cute::_2, cute::_2>,
                                       cute::_1, cute::_1, cute::_1>,
                           cute::tuple<cute::tuple<cute::_1, cute::_2>,
                                       cute::C<0>, cute::_0, cute::C<0>>>

    TC = cute::ArrayEngine<cutlass::half_t, 4>

    CLayout = cute::Layout<
        cute::tuple<cute::tuple<cute::_2, cute::_2>, cute::_1, cute::_1>,
        cute::tuple<cute::tuple<cute::_1, cute::_2>, cute::_0, cute::C<0>>>
*/
#if 1
  copy(tCgA, rA);
  copy(tCgB, rB);
  gemm(my_mma, rA, rB, rC);
  copy(rC, tCgC);
#endif
  return;
}

void printMatrix(const cute::half_t* data, int m, int n) {
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < m; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < n; ++j) {
            std::cout << static_cast<float>(data[i * n + j]) << " ";
        }
        std::cout << "]" << std::endl;
    }
}

int main() {
  using namespace cute;

  int Am = 16;
  int An = 8;
  int Ak = 16;
  int m1 = 5;
  int n1 = 6;
  int k1 = 1;
  int m = Am * m1;
  int n = An * n1;
  int k = Ak * k1;

  using TA = half_t;

  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TA> h_B(n * k);
  thrust::host_vector<TA> h_C(m * n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = 0;

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TA> d_B = h_B;
  thrust::device_vector<TA> d_C = h_C;

  using op = SM80_16x8x16_F16F16F16F16_TN;
  auto tiled_mma = make_tiled_mma(op{}, make_layout(make_shape(_1{}, _1{}, _1{}))); 

  dim3 dimGrid(m1, n1);
  dim3 dimBlock(32);
  
  f<<<dimGrid, dimBlock>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(),
                           m, n, tiled_mma);

  thrust::host_vector<TA> cute_result = d_C;
#if 0
  matrix_multiply_cpu(h_A.data(), h_B.data(), h_C.data(), m, n, k);
#endif
#if 0
  print("h_A : "); printMatrix(h_A.data(), m, k); print("\n\n");
  print("h_B : "); printMatrix(h_B.data(), k, n); print("\n\n");
  print("h_C : "); printMatrix(h_C.data(), m, n); print("\n\n");
  print("cute_result : "); printMatrix(cute_result.data(), m, n); print("\n\n");
#endif
#if 0
  assert(areMatricesEqual(cute_result.data(), h_C.data(), m, n));
  std::cout << "Success!" << std::endl;
#endif
  return 0;
}

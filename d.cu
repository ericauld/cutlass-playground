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

bool areMatricesEqual(const cute::half_t* C1, const cute::half_t* C2, int m, int n, float tolerance = 1e-1) {
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
  int k1,
  TiledMma            my_mma) {
  using namespace cute;

  int k = k1 * 16;

  Tensor mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(_16{}, k), make_stride(k, _1{})));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(_8{}, k), make_stride(k, _1{})));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(_16{}, _8{}), make_stride(_8{}, _1{})));
  auto thrmma = my_mma.get_slice(threadIdx.x);

  // No need for gA, gB, or gC...only one CTA

  // Our single CTA has blockIdx.x = 0, blockIdx.y = 0
  auto cta_coord = make_coord(0, 0, _);
  auto cta_tiler = make_shape(_16{}, _8{}, _16{});
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});

  Tensor tCgA = thrmma.partition_A(gA);
  Tensor tCgB = thrmma.partition_B(gB);
  Tensor tCmC = thrmma.partition_C(mC);

  auto rC = thrmma.make_fragment_C(tCmC);
  clear(rC);

  auto rA = thrmma.make_fragment_A(tCgA(_, _, _, 0));
  auto rB = thrmma.make_fragment_B(tCgB(_, _, _, 0));

#if 0
  if (thread0()) {
    print(my_mma);
    print("mA : "); print(mA); print("\n");
    print("mB : "); print(mB); print("\n");
    print("mC : "); print(mC); print("\n");
    print("gA : "); print(gA); print("\n");
    print("gB : "); print(gB); print("\n");
    print("tCgA : "); print(tCgA); print("\n");
    print("tCgB : "); print(tCgB); print("\n");
    print("tCmC : "); print(tCmC); print("\n");
    print("rA : "); print(rA); print("\n");
    print("rB : "); print(rB); print("\n");
    print("rC : "); print(rC); print("\n");
  }
#endif
#if 1
  for (int p1 = 0; p1 < k1; ++p1) {
    copy(tCgA(_, _, _, p1), rA);
    copy(tCgB(_, _, _, p1), rB);
    gemm(my_mma, rA, rB, rC);
  }
  copy(rC, tCmC);
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

  int Xm = 16;
  int Xn = 8;
  int Xk = 16;

  int k1 = 7;

  int m = Xm;
  int n = Xn;
  int k = Xk * k1;

  using TA = half_t;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TA> h_B(n*k);
  thrust::host_vector<TA> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = 0;

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TA> d_B = h_B;
  thrust::device_vector<TA> d_C = h_C;

  using op = SM80_16x8x16_F16F16F16F16_TN;
  auto tiled_mma = make_tiled_mma(op{}, make_layout(make_shape(_1{}, _1{}, _1{}))); 

  dim3 dimGrid(1);
  dim3 dimBlock(32);
  
  f<<<dimGrid, dimBlock>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), k1, tiled_mma);

  thrust::host_vector<TA> cute_result = d_C;
#if 1
  matrix_multiply_cpu(h_A.data(), h_B.data(), h_C.data(), m, n, k);
#endif
#if 0
  print("h_A : "); printMatrix(h_A.data(), m, k); print("\n\n");
  print("h_B : "); printMatrix(h_B.data(), k, n); print("\n\n");
  print("h_C : "); printMatrix(h_C.data(), m, n); print("\n\n");
  print("cute_result : "); printMatrix(cute_result.data(), m, n); print("\n\n");
#endif
# if 1
  assert(areMatricesEqual(cute_result.data(), h_C.data(), m, n));
  std::cout << "Success!" << std::endl;
#endif
  return 0;
}

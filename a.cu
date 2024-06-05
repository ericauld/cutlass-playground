#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

// Simple CPU implementation of matrix multiplication, assuming A is transposed
void matrix_multiply_cpu(const cute::half_t* A, const cute::half_t* B, cute::half_t* C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      cute::half_t sum = static_cast<cute::half_t>(0.0f);
      for (int p = 0; p < k; ++p) {
        // A is transposed, access it as A[p + i*k]
        sum += A[p + i * k] * B[p + j * k];
      }
      C[i * n + j] = sum;
    }
  }
}

bool areMatricesEqual(const cute::half_t* C1, const cute::half_t* C2, int m, int n, float tolerance = 1e-3) {
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
  TiledMma            my_mma) {
  using namespace cute;

  Tensor mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(_16{}, _16{}), make_stride(_16{}, _1{})));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(_8{}, _16{}));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(_16{}, _8{})); 
  auto thrmma = my_mma.get_slice(threadIdx.x);

  auto rC = thrmma.partition_fragment_C(mC);
  clear(rC);
  auto rA = thrmma.partition_fragment_A(mA);
  auto rB = thrmma.partition_fragment_B(mB);
  auto tCmC = thrmma.partition_C(mC);
  auto tCmA = thrmma.partition_A(mA);
  auto tCmB = thrmma.partition_B(mB);

#if 0
  print("mA : "); print(mA); print("\n");
  print("mB : "); print(mB); print("\n");
  print("mC : "); print(mC); print("\n");
  print("rA : "); print(rA); print("\n");
  print("rB : "); print(rB); print("\n");
  print("rC : "); print(rC); print("\n");
  print("tCmA : "); print(tCmA); print("\n");
  print("tCmB : "); print(tCmB); print("\n");
  print("tCmC : "); print(tCmC); print("\n");
#endif
#if 1
  copy(tCmA, rA);
  copy(tCmB, rB);
  gemm(my_mma, rA, rB, rC);
  copy(rC, tCmC);
#endif
  return;
}

void printMatrix(const cute::half_t* data, int rows, int cols) {
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < rows; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < cols; ++j) {
            std::cout << static_cast<float>(data[i * cols + j]) << " ";
        }
        std::cout << "]" << std::endl;
    }
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
  thrust::host_vector<TA> h_C_ref(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = 0;

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TA> d_B = h_B;
  thrust::device_vector<TA> d_C = h_C;

  matrix_multiply_cpu(h_A.data(), h_B.data(), h_C_ref.data(), m, n, k);

  using op = SM80_16x8x16_F16F16F16F16_TN;
  auto tiled_mma = make_tiled_mma(op{}, make_layout(make_shape(_1{}, _1{}, _1{}))); 

  dim3 dimGrid(1);
  dim3 dimBlock(32);
  f<<<dimGrid, dimBlock>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), tiled_mma);
  thrust::copy(d_C.begin(), d_C.end(), h_C.begin());
#if 1
  print("h_A : "); printMatrix(h_A.data(), m, k); print("\n\n");
  print("h_B : "); printMatrix(h_B.data(), k, n); print("\n\n");
  print("h_C : "); printMatrix(h_C.data(), m, n); print("\n\n");
  print("h_C_ref : "); printMatrix(h_C_ref.data(), m, n); print("\n\n");
#endif
  assert(areMatricesEqual(h_C.data(), h_C_ref.data(), m, n));

  std::cout << "Success!" << std::endl;
  return 0;
}

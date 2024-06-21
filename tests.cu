#include <iostream>
#include <cassert>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include "e.h"

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

bool areMatricesEqual(const cute::half_t* C1, const cute::half_t* C2, int m, int n, float tolerance = 1e-1f) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (std::fabs(static_cast<float>(C1[i * n + j]) - static_cast<float>(C2[i * n + j])) > tolerance) {
        return false;
      }
    }
  }
  return true;
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

void simplest() {
  using namespace cute;

  int Xm = 16;
  int Xn = 8;
  int Xk = 16;
  int Tm = 1;
  int Tn = 1;
  int Tk = 1;
  int m = Xm * Tm;
  int n = Xn * Tn;
  int k = Xk * Tk;

  using TA = half_t;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TA> h_B(k*n);
  thrust::host_vector<TA> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = 0;

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TA> d_B = h_B;
  thrust::device_vector<TA> d_C = h_C;

  using op = SM80_16x8x16_F16F16F16F16_TN;
  auto tiled_mma = make_tiled_mma(op{}, make_layout(make_shape(_1{}, _1{}, _1{}))); 

  dim3 dimGrid(1, 1);
  dim3 dimBlock(32);
  
  auto shape = make_shape(m, n, k);
  auto dA = make_stride(k, _1{});
  auto dB = make_stride(k, _1{});
  auto dC = make_stride(n, _1{});
  f<<<dimGrid, dimBlock>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, tiled_mma);

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
#if 1
  assert(areMatricesEqual(cute_result.data(), h_C.data(), m, n));
  std::cerr << "Simplest succeeded" << std::endl;
#endif
  return;
}

void second_simplest() {
  using namespace cute;

  int Xm = 16;
  int Xn = 8;
  int Xk = 16;
  int Tm = 4;
  int Tn = 5;
  int Tk = 1;
  int m = Xm * Tm;
  int n = Xn * Tn;
  int k = Xk * Tk;

  using TA = half_t;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TA> h_B(k*n);
  thrust::host_vector<TA> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = 0;

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TA> d_B = h_B;
  thrust::device_vector<TA> d_C = h_C;

  using op = SM80_16x8x16_F16F16F16F16_TN;
  auto tiled_mma = make_tiled_mma(op{}, make_layout(make_shape(_1{}, _1{}, _1{}))); 

  dim3 dimGrid(1, 1);
  dim3 dimBlock(32);
  
  auto shape = make_shape(m, n, k);
  auto dA = make_stride(k, _1{});
  auto dB = make_stride(k, _1{});
  auto dC = make_stride(n, _1{});
  f<<<dimGrid, dimBlock>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, tiled_mma);

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
#if 1
  assert(areMatricesEqual(cute_result.data(), h_C.data(), m, n));
  std::cerr << "Second simplest succeeded" << std::endl;
#endif
  return;
}

int main() {
  using namespace cute;

  simplest();
  second_simplest();

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
  
  f_local<<<dimGrid, dimBlock>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), k1, tiled_mma);

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
#if 1
  assert(areMatricesEqual(cute_result.data(), h_C.data(), m, n));
#endif
#if 1
  std::cerr << "Main succeeded" << std::endl;
#endif
  return 0;
}

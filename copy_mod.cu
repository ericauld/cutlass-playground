// Adapted from cutlass/examples/cute/tutorial/tiled_copy.cu

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, ThreadLayout, VecLayout)
{
  using namespace cute;
  using Element = typename TensorS::value_type;

  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);

  using AccessType = cutlass::AlignedArray<Element, size(VecLayout{})>;
  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

  auto tiled_copy =
    make_tiled_copy(
      Atom{},
      ThreadLayout{},
      VecLayout{});

  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);

  Tensor fragment = make_fragment_like(thr_tile_D);

  // gmem -> rmem
  copy(tiled_copy, thr_tile_S, fragment);
  // rmem -> gmem
  copy(tiled_copy, fragment, thr_tile_D);
}

int main(int argc, char** argv)
{
  using namespace cute;
  using Element = float;

  auto tensor_shape = make_shape(256, 512);

  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));
  Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape));

  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {
    std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
    return -1;
  }
  // Equivalent check to the above
  if (not weakly_compatible(block_shape, tensor_shape)) {
    std::cerr << "Expected the tensors to be weakly compatible with the block_shape." << std::endl;
    return -1;
  }

  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);

  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
  Layout vec_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

  dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));
  dim3 blockDim(size(thr_layout));

  copy_kernel_vectorized<<< gridDim, blockDim >>>(
    tiled_tensor_S,
    tiled_tensor_D,
    thr_layout,
    vec_layout);

  cudaError result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  h_D = d_D; // I guess this copies back to host...

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  return 0;
}

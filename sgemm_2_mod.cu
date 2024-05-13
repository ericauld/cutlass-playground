// Adapted from cutlass/examples/cute/tutorial/sgemm_2.cu

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta,
            int do_print = 0)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of partitioning via a TiledCopy

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
  // Allocate registers same shape/layout as partitioned data
  Tensor tArA = make_fragment_like(tAsA);                              // (CPY,CPY_M,CPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
  // Allocate registers same shape/layout as partitioned data
  Tensor tBrB = make_fragment_like(tBsB);                              // (CPY,CPY_N,CPY_K)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K

  // Copy gmem to rmem for k_tile=0
  copy(copy_a, tAgA(_,_,_,0), tArA);
  copy(copy_b, tBgB(_,_,_,0), tBrB);
  //
  // Define A/B partitioning and C accumulators
  //

  // TUTORIAL: Example of partitioning via a TiledMMA

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K

  // Clear the accumulators
  clear(tCrC);
  if (do_print == 1) {
    if(thread0()) {
      print("  mA : "); print(  mA); print("\n");
      print("  gA : "); print(  gA); print("\n");
      print("  sA : "); print(  sA); print("\n");
      print("tAgA : "); print(tAgA); print("\n");
      print("tAsA : "); print(tAsA); print("\n");
      print("tArA : "); print(tArA); print("\n");
    }
    if(thread0()) {
      print("  mB : "); print(  mB); print("\n");
      print("  gB : "); print(  gB); print("\n");
      print("  sB : "); print(  sB); print("\n");
      print("tBgB : "); print(tBgB); print("\n");
      print("tBsB : "); print(tBsB); print("\n");
      print("tBrB : "); print(tBrB); print("\n");
    }
    if(thread0()) {
      print("  mC : "); print(  mC); print("\n");
      print("  gC : "); print(  gC); print("\n");
      print("tCsA : "); print(tCsA); print("\n");
      print("tCsB : "); print(tCsB); print("\n");
      print("tCgC : "); print(tCgC); print("\n");
      print("tCrC : "); print(tCrC); print("\n");
    }
  }
#if 1

/*
M = 5120
N = 5120
K = 4096

TiledCopy
  Tiler_MN:       (_128,_8)
  TiledLayout_TV: (_256,_4):(_4,_1)

Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_4):(_0,_1)
  ValLayoutDst: (_1,_4):(_0,_1)
  ValLayoutRef: (_1,_4):(_0,_1)
  ValueType:    32b

TiledCopy
  Tiler_MN:       (_128,_8)
  TiledLayout_TV: (_256,_4):(_4,_1)

Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_4):(_0,_1)
  ValLayoutDst: (_1,_4):(_0,_1)
  ValLayoutRef: (_1,_4):(_0,_1)
  ValueType:    32b

TiledMMA
  ThrLayoutVMNK:  (_1,_16,_16,_1):(_0,_1,_16,_0)
  PermutationMNK: (_,_,_)

MMA_Atom
  ThrID:      _1:_0
  Shape_MNK:  (_1,_1,_1)
  LayoutA_TV: (_1,_1):(_0,_0)
  LayoutB_TV: (_1,_1):(_0,_0)
  LayoutC_TV: (_1,_1):(_0,_0)

  mA : (5120,4096): col

  gA : (_128:_1, _8:5120, 512:40960)

  sA : (_128,_8): col

tAgA : ((_4,_1):(_1,_0), _1:_0, _1:_0, 512:40960)

tAsA : ((_4,_1):(_1,_0), _1:_0, _1:_0)

tArA : ((_4,_1):(_1,_0), _1:_0, _1:_0)

  mB : (5120, 4096): col

  gB : (_128:_1, _8:5120, 512:40960)

  sB : (_128, _8): col

tBgB : ((_4,_1):(_1,_0), _1:_0, _1:_0, 512:40960)

tBsB : ((_4,_1):(_1,_0), _1:_0, _1:_0)

tBrB : ((_4,_1):(_1,_0), _1:_0, _1:_0)

  mC : (5120,5120) : col

  gC : (_128,_128) : (_1, 5120)

tCsA : (_1,_8,_8):(_0,_16,_128)

tCsB : (_1,_8,_8):(_0,_16,_128)

tCgC : (_1,_8,_8):(_0,_16,81920)

tCrC : (_1,_8,_8):(_0,_1,_8)
*/

  // TUTORIAL: Example of an inner loop that pipelines compute with reads
  //           from global memory by staging through register and shared memory.
  //   Data is read from global to registers, then to shared via the TiledCopy partitions
  //   gemm(.) operates on the shared memory directly via the TiledMMA partitions

  auto K_TILE_MAX = size<3>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy rmem to smem with tA|tB thread-partitioned tensors
    __syncthreads();         // Wait for all threads to consume smem
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();         // Wait for all threads to consume smem

    // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
    // TUTORIAL: The above call to copy(copy_a, tAgA(_,_,_,k_tile_next), tArA) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       copy_a.call(tAgA(_,m,k), tArA(_,m,k);
    //     }
    //   }

    // Compute gemm on mma-partitioned smem
    gemm(mma, tCsA, tCsB, tCrC);
    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       CUTE_UNROLL
    //       for (int n = 0; n < size<1>(tCrC); ++n) {
    //         mma.call(tCsA(_,m,k), tCsB(_,n,k), tCrC(_,m,n);
    //       }
    //     }
    //   }
  }

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0,
        int do_print = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)

  // TUTORIAL: Construct TiledCopy with a particular Copy_Atom to use and
  //           define the partitioning pattern to apply.
  // Each thread will (try to) copy 4x1 elements of type TA using 128-bit copy.
  // Use 32x8 of these threads.

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                                    Layout<Shape<_32,_8>>{},  // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{}); // Val layout  4x1 m-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
                                    Layout<Shape<_32,_8>>{},  // Thr layout 32x8 n-major
                                    Layout<Shape< _4,_1>>{}); // Val layout  4x1 n-major

  // TUTORIAL: Construct TiledMMA with a particular MMA_Atom to use and
  //           define the partitioning pattern to apply.
  // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single thread.
  // Reproduce that atom 16x16x1 times (m-major) across threads so that we use 256 threads.

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 UniversalFMA

  if (do_print == 1) {
    print(copyA);
    print(copyB);
    print(mmaC);
  }

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC,
       alpha, beta, do_print);
}

int main(int argc, char** argv)
{
  int m = 5120;
  int n = 5120;
  int k = 4096;

  char transA = 'N';
  char transB = 'T';

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  TI alpha = 1.0;
  TI beta  = 0.0;

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  cute::device_init(0);

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;

  ldA = m;
  ldB = n;

  // Run once
  d_C = h_C;
  gemm_nt(m, n, k, alpha, 
          d_A.data().get(), ldA, 
          d_B.data().get(), ldB, 
          beta, 
          d_C.data().get(), ldC, /*stream*/ 0, /*do_print*/ 1);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm_nt(m, n, k, alpha, 
            d_A.data().get(), ldA, 
            d_B.data().get(), ldB, 
            beta, 
            d_C.data().get(), ldC, /*stream*/ 0);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

  return 0;
}

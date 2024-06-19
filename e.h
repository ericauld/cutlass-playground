#include <cute/tensor.hpp>

template <class TiledMma, class ProblemShape,
          class AStride, class BStride, class CStride>
__global__ static
void
f(cute::half_t const *A,
  cute::half_t const *B,
  cute::half_t       *C,
  ProblemShape problem_shape,
  TiledMma            my_mma,
  AStride dA, BStride dB, CStride dC) {

  using namespace cute;
  // mA is k-major, i.e. "row major" i.e. "not transposed"
  Tensor mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(_16{}, _16{}), make_stride(_16{}, _1{})));
  // mB is k-major, i.e. "column major" i.e. "transposed"
  Tensor mB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(_8{}, _16{}), make_stride(_16{}, _1{})));
  // mC is n-major, i.e. "row major"
  Tensor mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(_16{}, _8{}), make_stride(_8{}, _1{})));
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
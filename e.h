#include <cute/tensor.hpp>

template <class TiledMma>
__global__ static
void
block_outer_product(cute::half_t const *A,
                    cute::half_t const *B,
                    cute::half_t       *C,
                    int m, int n,
                    TiledMma            my_mma) {
  using namespace cute;

  // Assumes block outer product structure, i.e. k = Tk1

  auto Tm1 = my_mma.template tile_size_mnk<0>();
  auto Tn1 = my_mma.template tile_size_mnk<1>();
  auto Tk1 = my_mma.template tile_size_mnk<2>();

  Tensor mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(m, Tk1), make_stride(Tk1, _1{})));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(n, Tk1), make_stride(Tk1, _1{})));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(m, n), make_stride(n, _1{})));

  auto thrmma = my_mma.get_slice(threadIdx.x);

  auto cta_tiler = make_shape(Tm1, Tn1, Tk1);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, 0);

  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  auto tCgA = thrmma.partition_A(gA);
  auto tCgB = thrmma.partition_B(gB);
  auto tCgC = thrmma.partition_C(gC);

  auto rA = thrmma.make_fragment_A(tCgA);
  auto rB = thrmma.make_fragment_B(tCgB);
  auto rC = thrmma.make_fragment_C(tCgC); 
  clear(rC);

  copy(tCgA, rA);
  copy(tCgB, rB);
  gemm(my_mma, rA, rB, rC);
  copy(rC, tCgC);
  return;
}

template <class TiledMma>
__global__ static
void
blocked_inner_product_tilek(cute::half_t const *A,
  cute::half_t const *B,
  cute::half_t       *C,
  int k1,
  TiledMma            my_mma) {
  using namespace cute;

  // Assumes block inner product structure, i.e. m = Tm1, n = Tn1

  auto Tm1 = my_mma.template tile_size_mnk<0>();
  auto Tn1 = my_mma.template tile_size_mnk<1>();
  auto Tk1 = my_mma.template tile_size_mnk<2>();

  int k = k1 * 16;

  Tensor mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(_16{}, k), make_stride(k, _1{})));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(_8{}, k), make_stride(k, _1{})));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(_16{}, _8{}), make_stride(_8{}, _1{})));
  auto thrmma = my_mma.get_slice(threadIdx.x);

  // Our single CTA has blockIdx.x = 0, blockIdx.y = 0
  auto cta_coord = make_coord(0, 0, _);
  auto cta_tiler = make_shape(_16{}, _8{}, _16{});
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});

  Tensor tCgA = thrmma.partition_A(gA);
  Tensor tCgB = thrmma.partition_B(gB);
  Tensor tCmC = thrmma.partition_C(mC);

  auto rA = thrmma.make_fragment_A(tCgA(_, _, _, 0));
  auto rB = thrmma.make_fragment_B(tCgB(_, _, _, 0));
  auto rC = thrmma.make_fragment_C(tCmC);
  clear(rC);

  for (int p1 = 0; p1 < k1; ++p1) {
    copy(tCgA(_, _, _, p1), rA);
    copy(tCgB(_, _, _, p1), rB);
    gemm(my_mma, rA, rB, rC);
  }
  copy(rC, tCmC);
  return;
}
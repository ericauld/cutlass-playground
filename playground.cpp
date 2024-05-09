#include <cute/tensor.hpp>
#include <iostream>

int f()
{
  using namespace cute;

  auto layout = Layout<Shape<Int<16>,_4>, Stride<_1,Int<16>>>{};
  auto shape = Shape<Int<64>, Int<16>>{};

  auto result = tile_to_shape(layout, shape);

  std::cout << "tile_to_shape(" << layout << ", " << shape << ")  =>  " << result << "\n";

  return 0;
}

int g() {
  using namespace cute;
  auto layout = Layout<Shape<_2,_4,_3>, Stride<_4, _1, _8>>{};
  auto B = Layout<Shape<_4>, Stride<_2>>{};
  std::cout << layout.compose(B) << "\n";
  std::cout << logical_divide(layout, B) <<  "\n";
  return 0;
}

int h1() {
  using namespace cute;
  auto layout = Layout<Shape<_4, _2>, Stride<Int<1>, Int<16>>>{};
  auto AtomLayoutMNK = Layout<Shape<_2,_2,_1>, Stride<_2,_1,_0>>{};
  std::cout << zipped_product(layout, AtomLayoutMNK) << "\n";
  std::cout << tiled_product(layout, AtomLayoutMNK) << "\n";
  return 0;
}

int h() {
  using namespace cute;
  TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                              Layout<Shape <_2,_2>,
                                     Stride<_2,_1>>{});
  auto ref_C = Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>{};
  auto x = mma.thrfrg_C(ref_C);
  std::cout << x << "\n";
  return 0;
}

int h2() {
  using namespace cute;
  TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                              Layout<Shape <_2,_2>,
                                     Stride<_2,_1>>{});
  auto ref_C = Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>{};
  auto a_tile = make_tile(make_layout(Int<8>{}),
                          make_layout(Int<8>{}));
  auto x = zipped_divide(ref_C, a_tile);
  std::cout << x << "\n";
  return 0;
}

int h5() {
  using namespace cute;
  auto l1 = Layout<Shape<Int<32>, Int<8>>>{};
  auto l2 = Layout<Shape<Int<4>, Int<1>>>{};
  auto rp = raked_product(l1, l2);
  std::cout << "raked_product(" << l1 << "," << l2 << ") = "<< rp << "\n";
  std::cout << coalesce(rp) << "\n";
  std::cout << right_inverse(rp) << "\n";
  std::cout << right_inverse(rp).with_shape(Shape<Int<256>, Int<4>>{}) << "\n";
  return 0;
}

int h4() {
  using namespace cute;
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 m-major
  auto x = typename decltype(copyA)::AtomLayoutSrc{};
  auto y = typename decltype(copyA)::AtomLayoutDst{};
  std::cout << "get_layoutS_TV() => " << copyA.get_layoutS_TV() << "\n";
  return 0;
}

int h3() {
  using namespace cute;
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 m-major
  auto x = typename decltype(copyA)::AtomLayoutSrc{};
  auto ref_S = make_layout(make_shape(shape(typename decltype(copyA)::Tiler_MN{}), Int<1>{}));
  std::cout << "ref_S => " << ref_S << "\n";
  std::cout << "right_inverse(x).compose(x) => " << right_inverse(x).compose(x) << "\n";
  auto y = copyA.tile2thrfrg(ref_S, right_inverse(x).compose(x));
  std::cout << "copyA.tile2thrfrg(ref_S, right_inverse(x).compose(x)) => " << y << "\n";
  return 0;
}

int h6() {
  using namespace cute;
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 m-major
  auto x = typename decltype(copyA)::TiledLayout_TV{};
  std::cout << zipped_divide(x, Shape<_1, _4>{}) << "\n";
  return 0;
}

int main() {
  using namespace cute;
  auto ref2trg = Layout<Shape<_1, _4>>{};
  using AtomNumThr = _1;
  using AtomNumVal = _4;
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 m-major
  using TiledLayout_TV = typename decltype(copyA)::TiledLayout_TV;

  // Take the thrs/vals that the atom is interested in
  // NOTE: Assumes the AtomNumThr are contiguous and identity within TiledThrID
  auto atom_layout_TV = zipped_divide(TiledLayout_TV{}, make_shape(AtomNumThr{}, AtomNumVal{}));
  // ((atom_tid,atom_val),(rest_tid,rest_val)) -> (m,n)

  // Transform to the trg layout
  auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
  // ((trg_tid,trg_val),(rest_tid,rest_val)) -> (m,n)

  // Transform the thrs mode from thrid to thr_idx
  // NOTE: Assumes the AtomNumThr are contiguous and identity within TiledThrID
  auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1,Shape<_1,_1>>{});
  // ((trg_tid,rest_tid),(trg_val,rest_val)) -> (m,n)

  std::cout << "thrval2mn => " << thrval2mn << "\n";
  return 0;
}
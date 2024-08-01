cd /home/ericauld/cup/build/cutlass/examples/cute/tutorial && \
      /usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler -DCUTLASS_ENABLE_CUBLAS=1 \
      -I/home/ericauld/cup/cutlass/include -I/home/ericauld/cup/cutlass/examples/common \
      -I/home/ericauld/cup/build/cutlass/include -I/include -I/examples \
      -I/home/ericauld/cup/cutlass/tools/util/include -isystem=/usr/local/cuda-12.4/include \
      -DCUTLASS_VERSIONS_GENERATED -O3 -DNDEBUG \
      --generate-code=arch=compute_90a,code=[sm_90a] \
      --generate-code=arch=compute_90a,code=[compute_90a] -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
      --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 \
      -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -ptx -src-in-ptx -std=c++17 \
      -x cu -c /home/ericauld/cup/cutlass/examples/cute/tutorial/wgmma_sm90.cu
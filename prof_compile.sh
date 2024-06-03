#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source-file>"
    exit 1
fi

SOURCE_FILE="$1"
BASE_NAME=$(basename "$SOURCE_FILE" .cu)
OBJECT_FILE="build/${BASE_NAME}.cu.o"
BINARY_NAME="build/${BASE_NAME}"

/usr/local/cuda/bin/nvcc --generate-line-info -forward-unknown-to-host-compiler --options-file /teamspace/studios/this_studio/cup/cutlass/build/examples/cute/tutorial/CMakeFiles/sgemm_1.dir/includes_CUDA.rsp -DCUTLASS_VERSIONS_GENERATED -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_86,code=[sm_86]"  -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -x cu -c "$SOURCE_FILE" -o "$OBJECT_FILE"
/usr/bin/g++ "$OBJECT_FILE" -o "${BINARY_NAME}_prof" @/teamspace/studios/this_studio/cup/cutlass/build/examples/cute/tutorial/CMakeFiles/sgemm_1.dir/linkLibs.rsp -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda/targets/x86_64-linux/lib"
rm "$OBJECT_FILE"

# linkLibs.rsp is:
#  -Wl,-rpath,'$ORIGIN' -Wl,-rpath,'$ORIGIN/../lib64' -Wl,-rpath,'$ORIGIN/../lib' -Wl,-rpath,'/usr/local/cuda/lib64' -Wl,-rpath,'/usr/local/cuda/lib' -lcuda -lcudadevrt -lcudart

# /teamspace/studios/this_studio/cup/cutlass/build/examples/cute/tutorial/CMakeFiles/sgemm_1.dir/includes_CUDA.rsp is:
# -I/teamspace/studios/this_studio/cup/cutlass/include -I/teamspace/studios/this_studio/a/cutlass/examples/common -I/teamspace/studios/this_studio/cup/cutlass/build/include -I/include -I/examples -I/teamspace/studios/this_studio/cup/cutlass/tools/util/include -isystem /usr/local/cuda/include
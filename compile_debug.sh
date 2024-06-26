#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source-file>"
    exit 1
fi

SOURCE_FILE="$1"
BASE_NAME=$(basename "$SOURCE_FILE" .cu)
OBJECT_FILE="build/${BASE_NAME}.cu.o"
BINARY_NAME="build/${BASE_NAME}"

/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler --options-file /teamspace/studios/this_studio/cup/includes_CUDA.rsp -DCUTLASS_VERSIONS_GENERATED -g -G -O0 -std=c++17 "--generate-code=arch=compute_86,code=[sm_86]" -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -x cu -c "$SOURCE_FILE" -o "$OBJECT_FILE"
/usr/bin/g++ "$OBJECT_FILE" -o "$BINARY_NAME" @/teamspace/studios/this_studio/cup/linkLibs.rsp -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda/targets/x86_64-linux/lib"
rm "$OBJECT_FILE"

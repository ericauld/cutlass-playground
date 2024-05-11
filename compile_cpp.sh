#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source-file>"
    exit 1
fi

SOURCE_FILE="$1"
BASE_NAME=$(basename "$SOURCE_FILE" .cpp)
OBJECT_FILE="build/${BASE_NAME}.cpp.o"
BINARY_NAME="build/${BASE_NAME}"

/usr/bin/c++ -I/teamspace/studios/this_studio/a/cutlass/include -I/teamspace/studios/this_studio/a/cutlass/test/unit/common -I/teamspace/studios/this_studio/a/cutlass/build/include -I/include -I/examples -isystem /usr/local/cuda/include -isystem /teamspace/studios/this_studio/a/cutlass/build/_deps/googletest-src/googletest/include -isystem /teamspace/studios/this_studio/a/cutlass/build/_deps/googletest-src/googletest -DCUTLASS_VERSIONS_GENERATED -O3 -DNDEBUG -std=c++17 -fPIE -o "$OBJECT_FILE" -c "$SOURCE_FILE"
/usr/bin/c++ "$OBJECT_FILE" -o "$BINARY_NAME" -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64: -Wl,-rpath,'$ORIGIN' -Wl,-rpath,'$ORIGIN/../lib64' -Wl,-rpath,'$ORIGIN/../lib' -Wl,-rpath,'/usr/local/cuda/lib64' -Wl,-rpath,'/usr/local/cuda/lib' /teamspace/studios/this_studio/a/cutlass/build/test/unit/cute/core/../../../../lib/libgtest.a -Wl,-rpath,'$ORIGIN' -Wl,-rpath,'$ORIGIN/../lib64' -Wl,-rpath,'$ORIGIN/../lib' -Wl,-rpath,'/usr/local/cuda/lib64' -Wl,-rpath,'/usr/local/cuda/lib' -lpthread
rm "$OBJECT_FILE"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
llama=third_party/llama.cpp
build_dir=${LITTLEBIT_BUILD_DIR:-"$PWD/build"}
git submodule update --init -- "$llama"

bash "$PWD/scripts/apply-patches.sh" "$PWD/$llama"

cmake -S "$llama" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_NATIVE="${LITTLEBIT_NATIVE:-ON}" -DGGML_CPU=ON -DGGML_BLAS=OFF \
    -DGGML_CUDA="${LITTLEBIT_CUDA:-OFF}" -DGGML_METAL="${LITTLEBIT_METAL:-OFF}" \
    -DGGML_VULKAN="${LITTLEBIT_VULKAN:-OFF}" -DGGML_OPENCL=OFF \
    -DGGML_SYCL=OFF -DGGML_WEBGPU=OFF -DGGML_RPC=OFF \
    -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF
cmake --build "$build_dir" --parallel "${LITTLEBIT_JOBS:-2}" \
    --target llama-cli llama-bench llama-quantize llama-perplexity

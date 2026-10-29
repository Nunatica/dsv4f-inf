#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
backend=${BACKEND:-auto}
if [[ $backend == auto ]]; then
    if command -v hipcc >/dev/null; then backend=hip
    elif command -v nvcc >/dev/null; then backend=cuda
    else echo 'HIP or CUDA toolkit is required.' >&2; exit 1; fi
fi
args=(-S "$ROOT" -B "${BUILD_DIR:-$ROOT/build}" -G Ninja -DCMAKE_BUILD_TYPE=Release
      -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF)
case "$backend" in
    hip) args+=(-DGGML_HIP=ON -DGGML_CUDA=OFF -DCMAKE_HIP_COMPILER="${HIP_COMPILER:-/opt/rocm/llvm/bin/clang++}"
                -DCMAKE_HIP_ARCHITECTURES="${GPU_ARCH:-gfx1100}" -DGGML_HIP_GRAPHS="${HIP_GRAPHS:-ON}");;
    cuda) args+=(-DGGML_CUDA=ON -DGGML_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH:-native}");;
    *) echo "Unknown BACKEND: $backend" >&2; exit 1;;
esac
cmake "${args[@]}"
extra_targets=()
if [[ $backend == hip ]]; then extra_targets+=(dsv4f-qnorm-rope-test dsv4f-indexer-top512-test dsv4f-indexer-test dsv4f-prefill-attn-test dsv4f-mmq-workspace-test); fi
cmake --build "${BUILD_DIR:-$ROOT/build}" --target llama-server dsv4f-load-pipeline-test dsv4f-sparse-attn-test dsv4f-kv-stream-test dsv4f-moe-test dsv4f-graph-test dsv4f-service-test dsv4f-router-test dsv4f-web-http-test dsv4f-budget-test dsv4f-cache-policy-test dsv4f-mxfp4-rows-test dsv4f-prefill-test "${extra_targets[@]}" -j "${BUILD_JOBS:-12}"

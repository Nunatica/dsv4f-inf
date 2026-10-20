#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MODEL="${MODEL:-$ROOT/models/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00001-of-00005.gguf}"
export CTX_SIZE=${CTX_SIZE:-262144}
export DSV4F_KV_RESIDENT=${DSV4F_KV_RESIDENT:-65536}
#export DSV4F_KV_RESIDENT=${DSV4F_KV_RESIDENT:-0}
if [[ ! $DSV4F_KV_RESIDENT =~ ^[0-9]{1,10}$ ]] ||
   ((10#$DSV4F_KV_RESIDENT > 2147483647 || (10#$DSV4F_KV_RESIDENT != 0 && 10#$DSV4F_KV_RESIDENT < 2048))); then
    echo 'DSV4F_KV_RESIDENT must be 0 or an integer in 2048..2147483647.' >&2; exit 1
fi
export GGML_HIP_SPARSE_FA=${GGML_HIP_SPARSE_FA:-1}
case "$GGML_HIP_SPARSE_FA" in 0|1) ;; *) echo 'GGML_HIP_SPARSE_FA must be 0 or 1.' >&2; exit 1;; esac
export GGML_HIP_PREFILL_FA=${GGML_HIP_PREFILL_FA:-2}
case "$GGML_HIP_PREFILL_FA" in 0|1|2) ;; *) echo 'GGML_HIP_PREFILL_FA must be 0, 1 or 2.' >&2; exit 1;; esac
export GGML_CUDA_MOE_CACHE=0
export DSV4F_CPU_THREADS=${DSV4F_CPU_THREADS:-12}
export DSV4F_VRAM_HEADROOM_MIB=${DSV4F_VRAM_HEADROOM_MIB:-256}
export DSV4F_VRAM_GUARD_MIB=${DSV4F_VRAM_GUARD_MIB:-256}
export DSV4F_WHOLE_EXPERT=${DSV4F_WHOLE_EXPERT:-1}
export DSV4F_BATCH_TRANSFERS=${DSV4F_BATCH_TRANSFERS:-1}
export DSV4F_GPU_JOIN=${DSV4F_GPU_JOIN:-1}
export DSV4F_PINNED_COMPUTE=${DSV4F_PINNED_COMPUTE:-1}
export DSV4F_PINNED_WEIGHTS=${DSV4F_PINNED_WEIGHTS:-$DSV4F_PINNED_COMPUTE}
export DSV4F_EXPERT_GRAPHS=${DSV4F_EXPERT_GRAPHS:-1}
export DSV4F_FUSED_HC_MIX=${DSV4F_FUSED_HC_MIX:-1}
export DSV4F_GPU_SERVICE=${DSV4F_GPU_SERVICE:-1}
export DSV4F_GPU_WAIT=${DSV4F_GPU_WAIT:-callback}
export DSV4F_CPU_SPIN_US=${DSV4F_CPU_SPIN_US:-300}
export DSV4F_CPU_FAST_SYNC=${DSV4F_CPU_FAST_SYNC:-1}
export DSV4F_CPU_AFFINITY=${DSV4F_CPU_AFFINITY:-none}
export DSV4F_CPU_SCHEDULE=${DSV4F_CPU_SCHEDULE:-dynamic}
export DSV4F_CPU_ROW_CHUNK=${DSV4F_CPU_ROW_CHUNK:-32}
export DSV4F_CPU_MXFP4_ROWS=${DSV4F_CPU_MXFP4_ROWS:-1}
export DSV4F_CPU_MXFP4_ISA=${DSV4F_CPU_MXFP4_ISA:-avx512}
case "$DSV4F_CPU_AFFINITY" in none|physical) ;; *) echo 'DSV4F_CPU_AFFINITY must be none or physical.' >&2; exit 1;; esac
case "$DSV4F_CPU_SCHEDULE" in dynamic|static) ;; *) echo 'DSV4F_CPU_SCHEDULE must be dynamic or static.' >&2; exit 1;; esac
export DSV4F_HC_WARP=${DSV4F_HC_WARP:-1}
export DSV4F_HC_FULL=${DSV4F_HC_FULL:-1}
export DSV4F_HC_POST_FULL=${DSV4F_HC_POST_FULL:-1}
export DSV4F_PREFILL_STREAM=${DSV4F_PREFILL_STREAM:-1}
export DSV4F_PREFILL_GROUP=${DSV4F_PREFILL_GROUP:-16}
export DSV4F_PREFILL_CACHE=${DSV4F_PREFILL_CACHE:-0}
export DSV4F_PREFILL_CACHE_MIN_TOKENS=${DSV4F_PREFILL_CACHE_MIN_TOKENS:-4096}
case "$DSV4F_PREFILL_CACHE" in 0|1) ;; *) echo 'DSV4F_PREFILL_CACHE must be 0 or 1.' >&2; exit 1;; esac
if [[ ! $DSV4F_PREFILL_CACHE_MIN_TOKENS =~ ^[0-9]{1,10}$ ]] || ((10#$DSV4F_PREFILL_CACHE_MIN_TOKENS < 1 || 10#$DSV4F_PREFILL_CACHE_MIN_TOKENS > 2147483647)); then
    echo 'DSV4F_PREFILL_CACHE_MIN_TOKENS must be an integer in 1..2147483647.' >&2; exit 1
fi
case "$DSV4F_PREFILL_STREAM" in 0|1) ;; *) echo 'DSV4F_PREFILL_STREAM must be 0 or 1.' >&2; exit 1;; esac
if [[ ! $DSV4F_PREFILL_GROUP =~ ^[0-9]{1,3}$ ]] || ((10#$DSV4F_PREFILL_GROUP < 1 || 10#$DSV4F_PREFILL_GROUP > 256)); then
    echo 'DSV4F_PREFILL_GROUP must be an integer in 1..256.' >&2; exit 1
fi
export DSV4F_PHASE_WORKSPACE=${DSV4F_PHASE_WORKSPACE:-1}
export DSV4F_DECODE_BATCH=${DSV4F_DECODE_BATCH:-512}
export DSV4F_DECODE_UBATCH=${DSV4F_DECODE_UBATCH:-256}
export DSV4F_CACHE_POLICY=${DSV4F_CACHE_POLICY:-aged}
case "$DSV4F_CACHE_POLICY" in legacy|scored|aged) ;; *) echo 'DSV4F_CACHE_POLICY must be legacy, scored or aged.' >&2; exit 1;; esac
export DSV4F_MXFP4_FUSED=${DSV4F_MXFP4_FUSED:-1}
case "$DSV4F_GPU_WAIT" in poll|callback) ;; *) echo 'DSV4F_GPU_WAIT must be poll or callback.' >&2; exit 1;; esac
# Pinned host weights reduce the first large prefill's expert-upload cost.
# Decode cache misses stay on CPU. Both host buffer kinds share one allocator.
case "$DSV4F_PINNED_WEIGHTS" in 0|1) ;; *) echo 'DSV4F_PINNED_WEIGHTS must be 0 or 1.' >&2; exit 1;; esac
if [[ $DSV4F_PINNED_COMPUTE == 1 || $DSV4F_PINNED_WEIGHTS == 1 ]] || ((10#$DSV4F_KV_RESIDENT > 0)); then
    unset GGML_CUDA_NO_PINNED
else
    export GGML_CUDA_NO_PINNED=1
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$DSV4F_CPU_THREADS}
export HOST=${HOST:-0.0.0.0} PORT=${PORT:-8080}
export API_KEY=${API_KEY:-dummykey}
export DSV4F_MTP=${DSV4F_MTP:-1}
export MTP_MODEL="${MTP_MODEL:-$ROOT/models/DeepSeek-V4-Flash-MTP-Q4_0.gguf}"
export DSV4F_DRAFT_N_MAX=${DSV4F_DRAFT_N_MAX:-1}
export DSV4F_DRAFT_P_MIN=${DSV4F_DRAFT_P_MIN:-0.25}
case "$DSV4F_MTP" in 0|1) ;; *) echo 'DSV4F_MTP must be 0 or 1.' >&2; exit 1;; esac
# Cache ceiling. The native server loads target/MTP contexts first, then sizes
# the actual expert cache from live VRAM while preserving the requested headroom.
if [[ $DSV4F_MTP == 1 ]]; then
    [[ -f "$MTP_MODEL" ]] || { echo "Missing MTP model: $MTP_MODEL" >&2; exit 1; }
    # Phase sizing measures both contexts before allocating the cache. Leave
    # its ceiling automatic so reclaimed workspace can be used with MTP too.
    # An explicitly supplied cache ceiling is always retained.
    if [[ $DSV4F_PHASE_WORKSPACE == 0 ]]; then
        export DSV4F_EXPERT_CACHE_MIB=${DSV4F_EXPERT_CACHE_MIB:-10240}
    fi
fi
BUILD_DIR=${BUILD_DIR:-$ROOT/build}
args=(--model "$MODEL" --ctx-size "$CTX_SIZE" --cache-type-k q8_0 --cache-type-v q8_0
      --load-mode none --lazy-mode off --no-repack --cpu-moe --n-gpu-layers 999 --fit off
      --threads "$DSV4F_CPU_THREADS" --threads-batch "$DSV4F_CPU_THREADS"
      --batch-size "${BATCH_SIZE:-5120}" --ubatch-size "${UBATCH_SIZE:-5120}"
      --flash-attn on --parallel 1 --host "$HOST" --port "$PORT" --api-key "$API_KEY"
      --jinja --reasoning off --reasoning-budget 0 --no-context-shift
      --alias dsv4f --ui --path "$ROOT/web")
if [[ $DSV4F_PINNED_WEIGHTS == 0 ]]; then args+=(--no-host); fi
if [[ $DSV4F_MTP == 1 ]]; then
    args+=(--model-draft "$MTP_MODEL" --spec-type draft-mtp
           --spec-draft-n-max "$DSV4F_DRAFT_N_MAX" --spec-draft-p-min "$DSV4F_DRAFT_P_MIN"
           --spec-draft-ngl 999 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0
           --spec-draft-threads "$DSV4F_CPU_THREADS" --spec-draft-threads-batch "$DSV4F_CPU_THREADS")
fi
if [[ -n ${MMPROJ:-} ]]; then args+=(--mmproj "$MMPROJ"); fi
if [[ ${1:-} == --dry-run ]]; then
    python3 "$ROOT/scripts/model_info.py" "$MODEL"
    printf '%q ' "$BUILD_DIR/bin/dsv4f-server" "${args[@]}" "${@:2}"
    printf '\n'
    exit 0
fi
if ! python3 "$ROOT/scripts/model_info.py" --swap-check; then
    if command -v systemd-run >/dev/null; then
        exec systemd-run --user --scope --quiet --unit="dsv4f-$$" -p MemorySwapMax=0 "$ROOT/run-server.sh" "$@"
    fi
    echo 'Cannot forbid swap. Disable system swap (sudo swapoff -a) or enable user systemd cgroups.' >&2
    exit 1
fi
[[ -x "$BUILD_DIR/bin/dsv4f-server" ]] || "$ROOT/build-engine.sh"
mkdir -p "$ROOT/logs" "$ROOT/state"
exec python3 "$ROOT/scripts/serve.py" --state-dir "$ROOT/state" --log-dir "$ROOT/logs" -- \
    "$BUILD_DIR/bin/dsv4f-server" "${args[@]}" "$@"

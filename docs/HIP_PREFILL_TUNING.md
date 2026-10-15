# HIP D512 prefill tuning

The RX 7900 XTX (gfx1100) implementation has three process-level modes:

| `GGML_HIP_PREFILL_FA` | Path |
|---|---|
| `0` | Original HIP tile kernel |
| `1` | Original arithmetic, with fully masked KV chunks skipped |
| `2` (default) | RDNA3 WMMA, with Q in LDS and FP32 VKQ accumulation |

Decode continues to use the existing selected-KV gather. These prefill modes
apply only to D512, at least 32 queries and 2048 KV rows, one sequence, a shared
F16 mask, F16/Q8_0 KV, a GQA ratio divisible by eight, and no ALiBi/softcap.
KV rows must be padded to 256. WMMA also requires the existing MMA stride
alignment. Unsupported inputs retain existing dispatch. This D512 path currently
supports RDNA3.0; the implementation was tested on gfx1100.

The skip checks the union of four query masks in each KV chunk. It skips only
exact negative infinity, retaining finite mask biases and the ordinary path for
NaN/positive-infinity entries. It does not gather a union across the whole
microbatch. Tile mode uses 128-row chunks; WMMA uses 32-row chunks.

The original D512 WMMA host symbols exist, but their device bodies were disabled.
Only the 4-query, 8-head, 32-column RDNA3 body is now enabled. The 64-column
configuration remains disabled: its Q staging alone needs 66560 bytes, exceeding
gfx1100's LDS capacity. Q in LDS lowers register pressure compared with keeping
all D512 Q fragments in registers. FP32 VKQ accumulation and scaling Q before
FP16 conversion improve accuracy. The final shared-memory request is 50496 bytes
plus compiler/static LDS. Register spilling remains: rocprofv3 reports 256 VGPRs
and Scratch_Size=1148, compared with 1576 for the initial Q-in-register/FP16
prototype. This is an improvement, not a spill-free WMMA implementation.

## Validation

Build normally with `./build-engine.sh`, then run:

```bash
ctest --test-dir build \
  -R 'gpu-prefill-attention|gpu-sparse-attention|gpu-streamed-prefill|gpu-graph-replay' \
  --output-on-failure
```

Both new modes exercise seven cases and three graph executions per case with
changing Q/K/V, masks and sinks. Cases cover F16/Q8_0, 33-query partial tiles,
one/two KV heads, shared/separate KV, strided storage, all-masked rows with
sinks, dense masks without sinks, finite mask biases and small-query fallback.
All outputs are checked for finiteness. Skip-only output is compared in full
against the original tile and is exactly equal in these tests. WMMA uses a
double-precision CPU oracle for sampled query/head rows, with NMSE < 1e-5 and
max absolute error < 0.002. The original tile is still compared for diagnostics,
but its FP16 VKQ arithmetic is not treated as an exact oracle.

On 2026-10-15, all five selected CTest suites passed. Both new suites also passed
after strengthening masks to exercise long empty KV intervals, a fully masked
four-query tile and an empty partial final tile. WMMA's eligible cases then had
CPU NMSE from 9.28e-8 to 1.60e-7 and maximum absolute error up to 8.13e-5.
The skip-only path matched the original tile exactly in every case. The existing
sparse decode test, streamed MoE prefill test and graph replay test also passed.
These are numerical tests, not a broad model-quality evaluation.

## Measuring the full model

Use the entire repository prompt with MTP disabled. Fix cache capacity, group
size and batch settings across server restarts:

```bash
DSV4F_MTP=0 DSV4F_EXPERT_CACHE_MIB=6324 GGML_HIP_PREFILL_FA=2 \
  GGML_HIP_PREFILL_FA_DEBUG=1 ./run-server.sh
# In another terminal:
python3 scripts/benchmark.py --prompt-file PROMPT_TEST.md --tokens 128 --runs 3 \
  --warmup-tokens 0 --output logs/prefill-wmma.json
```

Repeat with modes 0 and 1. Each request uses `cache_prompt=false`, temperature 0
and seed 42. Compare medians with identical 496-slot expert caches and 4096-token
prefill batch/ubatch. Keep `DSV4F_PREFILL_PROFILE` and rocprof disabled for final
throughput measurements because profiling changes synchronization/timing.

The earlier synthetic 4096-query/8960-KV-row/Q8_0 experiment measured about
302 ms for tile, 187 ms for skip-only tile and 153 ms for LDS/FP32 WMMA with
mask skipping. Those are isolated attention-call times with a synthetic mask,
not full-model prefill throughput. Experiments and raw results are in ignored
`logs/hip-prefill-tuning/`.

## Full-model results (2026-10-15)

RX 7900 XTX, the full `PROMPT_TEST.md` (17778 prompt tokens), 128 output tokens,
3 runs per mode, no additional warmup, MTP disabled, context 65536, batch/ubatch
4096, streamed MoE group16/ring2, fixed 6324MiB expert cache (496 slots in every
prefill/decode phase). Each server was started with `run-server.sh`.

| Mode | Median prefill tok/s | Change vs original | Median decode tok/s |
|---|---:|---:|---:|
| 0: original tile | 237.52 | +0.00% | 14.56 |
| 1: skip-only tile | 353.07 | +48.65% | 14.48 |
| 2: LDS/FP32 WMMA (default) | 369.98 | +55.77% | 14.70 |

Median prefill time decreased from 74.85s to 50.35s with mode1 and 48.05s with
mode2. Decode measurements stayed in a similar range; no decode improvement is
claimed by this prefill change. The original baseline binary was measured before
rebuilding; optimized modes used the same rebuilt binaries.

All three modes emitted the same first nine complete function names in all three
128-token outputs. Introductory wording and how much of the next function fit
inside the token limit varied. This check does not establish general model-quality
parity. Mode2 is the default in both the launcher and library; mode1 remains
available to retain the original tile arithmetic.

Raw API responses, server logs, fixed-cache checks, build hashes and per-run
measurements are in `logs/hip-prefill-tuning/comparison.json` and adjacent files.
The prompt SHA256 is
`46ea3e967d86ee8b1c51c770cbe2704aefedd2b0ef56c4c08a253ad6bb7861c0`.

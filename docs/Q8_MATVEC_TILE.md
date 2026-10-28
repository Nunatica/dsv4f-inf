# V4 Q8 matvec tile experiment — 2026-10-26

`GGML_HIP_Q8_TILE=1` selects an experimental single-column Q8_0 matvec path on
RDNA3.0. Following the experiment, the user requested enabling it by default;
the launcher now defaults to **1** with automatic configuration selection.
Direct binary use requires explicitly setting the variable. The experiment keeps router fusion and
the all-hit completion optimization enabled, and disables MTP.

## Dispatch and tuning

Several independent output rows share a thread block. Each row can use fewer
warps than the general matvec kernel; the shared gate/up variant retains the
existing clamped SwiGLU epilogue. Weights, activation quantization, bias values
and the clamp formula are unchanged. Changing reduction width can change FP32
rounding, so not every configuration is bit-exact.

| Matrix, input → output | Epilogue | Warps/row | Rows/block | Config ID |
| --- | --- | ---: | ---: | ---: |
| 4096 → 1024 | Plain | 2 | 4 | 2 |
| 4096 → 2048 | Plain | 1 | 4 | 1 |
| 4096 → 2048 | Fused gate/up + clamped SwiGLU | 8 | 1 | 4 |
| 2048 → 4096 | Plain | 1 | 4 | 1 |

Only these matrix dimensions, Q8_0 weights, no routed ID gather, one input
column, and supported epilogues select the new kernel. Bias/scale fusion and
shared routed epilogues retain the original dispatch. Channel/sample broadcast
and tensor strides are carried through. Larger token batches and other matrix
shapes retain the original path.

The development override `GGML_HIP_Q8_TILE_TUNE=1..7` selects a measured
configuration for supported shapes; unset or 0 uses the table above. IDs 1–7
map to `(warps,row count)` = `(1,4), (2,4), (4,2), (8,1), (1,8), (2,8), (1,16)`.
Normal model comparisons use automatic selection (override 0). Debug logs can
be enabled with `GGML_HIP_Q8_TILE_DEBUG=1` to verify actual kernel dispatch.

## Component measurements

Seven configurations and the original path were compared in three independent
processes on RX 7900 XTX/gfx1100. Each process interleaves modes and reports a
median. The table takes the median of those three process medians. Separate
weight buffers rotate through approximately 192 MiB of storage, exceeding the
GPU cache. Weights are reused within each process, so these are still synthetic
component measurements, not a model speedup prediction.

| Operation | Original ms | Selected ms | Latency reduction |
| --- | ---: | ---: | ---: |
| 4096 → 1024 | 0.009426 | 0.008430 | 10.57% |
| 4096 → 2048, plain | 0.017670 | 0.013510 | 23.54% |
| 2048 → 4096 | 0.018598 | 0.017146 | 7.81% |
| 4096 → 2048, fused gate/up | 0.028591 | 0.027604 | 3.45% |

Plain matvec timings use prequantized Q8_1 input. The fused timings call the
actual backend entry point and include the same input quantizer in both modes.
Those rows must not be added together or treated as equally scoped timings.
The 8-warps/row fused configuration matched the original output bitwise in the
probe and was slightly faster than other fused candidates. The 1-warp plain
winner was not assumed to be the best fused configuration.

## Numerical and graph checks

The rotating-weight probe compares each configuration against the original
GPU implementation and an independently computed, sampled FP64 dot/SwiGLU
oracle. Coverage includes negative/zero/small Q8 scales and a seven-row tail.
The relative-error limits remain 2e-6 and maximum absolute oracle error 0.001;
no tolerance was widened. Three final probe processes passed every mode.

An initial fused oracle readback omitted synchronization of its nondefault
stream and read unfinished quantizer output. Synchronization was added before
readback; the initial failure log is preserved as `probe-r1.log`. The corrected
`probe-r2.log` through `probe-r4.log` are the final numerical/tuning records.
This was a probe readback issue, not a change to inference arithmetic.

`gpu-q8-tile-parity` checks actual GGML backend dispatch and graph fusion,
changing input amplitudes, graph capture/replay, fused and separate clamp/GLU,
channel/sample broadcast, padded weight/input views, and bias epilogues.
Multi-token and unselected-shape comparisons require bitwise equality against
the preceding dispatch. Other operator comparisons retain relative RMS <2e-6.
The complete suite with tile and router fusion enabled passed **36/36** tests;
Python loader checks passed **2/2**.

## Initial whole-model numerical comparison

Actual dispatch logs confirmed all three target matrix shapes and fused shared
gate/up using config 4. With expert cache disabled and the same 64 teacher
prefixes, all **64 returned prediction tokens matched**. Returned probability
records did not match: 25 steps had a changed top-16 set; 995 common entries
had relative RMS probability difference **0.0237833**, mean absolute logprob
difference **0.136396**, and maximum absolute logprob difference **0.889584**.
For the returned top-1 probabilities, mean absolute difference was **0.007807**
and maximum **0.051677**. These are API-returned probabilities/logprobs, not
raw logits or operator error measures. The complete probability records are
preserved in both validation reports.

A separate actual-model run then forced config 4 (8 warps/row, one row/block)
for every selected shape, using the same binary and cache=0 teacher inputs.
All 64 returned tokens and all 1,024 top-16 logprob entries matched the
original; the top-16 ID sets also matched. This supports attributing the fast
automatic configuration's numerical differences to changed reduction order
and its downstream effects. It is not a general model-quality equivalence
study. Whole-model throughput of this uniform config-4 alternative was not
measured. The launcher now enables the automatic configuration by default;
the uniform config-4 alternative remains an explicit override.

## Whole-model method

Both modes use the same final binary and library files, the five-shard 144.44
GiB V4 Flash Vision model, 12 CPU workers, context 262144, batch/ubatch 5120,
Q8_0 KV, resident KV 65536, aged expert cache and 256 MiB headroom/guard.
`DSV4F_MTP=0`, `DSV4F_ROUTER_FUSED=1`, `DSV4F_SKIP_NO_MISS=1` are fixed. Each
server runs in a no-swap systemd scope and is stopped afterwards.

Cache=0 validation servers compare 64 growing prefixes of the same teacher
sequence. Each request returns one token and top-16 probabilities. This avoids
feeding diverging generated tokens back into the numerical comparison. This
is a numerical comparison for one input sequence, not a broad quality study.

Performance uses fresh servers in **0 → 1 → 1 → 0** order with profiling and
debug logs disabled. The exact short prompt is
`Explain story of the romance of three kingdoms.` The archived fixed teacher
plan from the router experiment is reused unchanged: a 12-token chat prefix,
128 warmup evaluations and 256 measured one-token evaluations. The next input
always follows the teacher, independent of returned generation tokens.
Native evaluation time is `prompt_ms + predicted_ms`; request wall time is
also retained. No teacher generation occurs on any measured server.

Each run then processes all of `PROMPT_TEST.md`, prompt cache off, reasoning
off, temperature 0, seed 42 and a 64-token generation limit. Raw short results
and a secondary comparison excluding the same outlier positions from every
run are retained. The exclusion set is the union of positions whose wall time
exceeds three times that run's median. Asynchronous expert-cache admission is
not frozen. Two runs per mode do not provide a confidence interval or a
universal speedup estimate.

## Whole-model performance results

| Run | Short native ms/token | Short tok/s | Long prefill tok/s | Long decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Control 1 | 54.7980 | 18.2488 | 400.6760 | 12.9950 |
| Tile 1 | 54.6386 | 18.3021 | 401.2347 | 13.0371 |
| Tile 2 | 55.6055 | 17.9838 | 399.3051 | 13.6987 |
| Control 2 | 54.5954 | 18.3166 | 399.7437 | 13.6274 |

| Both runs per mode | Original | Automatic tile | Throughput change |
| --- | ---: | ---: | ---: |
| Short, all 512 samples/mode | 18.2826 | 18.1416 | -0.77% |
| Short, common outlier position excluded | 18.2743 | 18.2729 | -0.01% |
| Long decode | 13.3112 | 13.3679 | +0.43% |
| Long prefill | 400.2098 | 400.2699 | +0.02% |

The short aggregation is reciprocal mean native latency. The long aggregation
is the mean of the two reported native throughput values. All four long
prompts contained 17,778 tokens, followed by 64 generated tokens.

Step **328** was the common excluded position: native evaluation was
48.467 / 47.978 / **264.259** / 48.251 ms in execution order. Its cause was not
established. The secondary comparison excludes that position from every run
(510 samples/mode); the raw averages remain above. After this same-step
exclusion, the first comparison favored the tile by about 0.29%, while the
reverse-order pair favored the original by about 0.30%.

A repeatable whole-model short-input benefit was not demonstrated. The
observed +0.43% long decode difference is small relative to the variation
between runs and is not a statistically established improvement. Long
responses were not all equal; measured short prediction tokens also varied
within each mode. The teacher inputs themselves were identical across modes.
The raw generated outputs remain in the reports.

The automatic fast configuration changes returned model probabilities and did
not show a meaningful whole-model throughput gain in this trial. Following the
user's subsequent request, the launcher default is now **1**; use
`GGML_HIP_Q8_TILE=0` to restore the original kernel. The measurements above are
unchanged. Component speedups can still inform future work with a different
critical-path mix; no full-model bottleneck profile was collected here.

All performance-process swap checks were zero, and server/library/launcher
hashes were unchanged through both comparisons. All owned servers were
stopped; `state/server.json` was removed and `amd-smi process` reported no
running GPU processes.

## Reproduction

```bash
cmake --build build --target dsv4f-q8-tile-test dsv4f-q8-tile-probe llama-server
DSV4F_MTP=0 build/bin/dsv4f-q8-tile-probe
DSV4F_MTP=0 ctest --test-dir build -R gpu-q8-tile-parity --output-on-failure
DSV4F_MTP=0 GGML_HIP_Q8_TILE=1 ./run-server.sh
```

Use `GGML_HIP_Q8_TILE=0` for the original kernel. Raw data, configuration sweep,
model logs, fixed-input records, per-run settings, swap checks and library
hashes are saved under `logs/q8-tile-v4-20261026/`.

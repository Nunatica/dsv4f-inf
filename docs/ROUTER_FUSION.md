# V4 router fusion experiment — 2026-10-26

`DSV4F_ROUTER_FUSED=1` is an experimental HIP path for V4's small decode
batches. Following the experiment, the user requested enabling it by default;
the launcher now defaults to **1**. Set `DSV4F_ROUTER_FUSED=0` for the original
route. Direct binary use requires explicitly setting the variable. MTP was disabled throughout this
experiment; the existing all-hit completion optimization stayed enabled in
both performance modes.

## Implementation

The fused operation computes sqrt-softplus scores, selection-only bias,
top-k IDs, unbiased score gathering, optional normalization, and final weight
scaling in one GPU kernel. The CPU fallback uses an independent partial sort.
IDs and probabilities are views into one packed allocation.

V4 normalization remains `max(sum, 6.103515625e-5)`. This differs from V4.1's
`sum + 1e-20`; the V4.1 denominator was not imported. A zero weight-scale
parameter retains V4's existing meaning of omitting scaling. Equal selection
scores choose the smaller expert ID, and bias never enters returned weights.

The graph selects fusion only for V4, HIP, 1–4 tokens, at most 512 experts and
32 selected experts, an ordinary contiguous F32 selection bias, no expert
groups, and no evaluation callback. Existing `selected_experts_in` always
bypasses fusion, preserving token-ID hash routing in the first three text
layers. Media routing retains its chosen bias. Larger prefill batches and
other backends retain the original graph.

GPU placement is explicit on the packed router operation. A host weight or
bias buffer is not a reliable indication of where a decode operation should
execute under `--cpu-moe`. The operation retains ordinary scheduler transfers
for its actual inputs; it does not use the expert-weight descriptor exemption.

Sources: `src/router.cpp`, `src/moe.h`,
`vendor/llama.cpp/src/llama-graph.cpp`,
`vendor/llama.cpp/ggml/src/ggml-cuda/dsv4f-service.cu`, and the custom-op dispatch
in `ggml-cuda.cu`. The implementation was adapted from `../dsv41f-inf`.

## Correctness checks

- `gpu-v4-router-parity`: 168 combinations, each with 16 changing inputs.
  Expert counts 6/32/255/256/512, top-k 1/6/32 where valid, 1/2/4 tokens,
  normalization on/off, and contiguous/strided rows. Inputs include ties,
  underflow-to-zero scores, small scores and large positive logits. IDs and
  weights match the original GPU composition **bitwise**. The independent CPU
  fallback passes the existing 2e-6 relative-floor comparison.
- The GPU suite passed **35/35** tests; Python loader checks passed **2/2**.
  Launcher values 0/1 and rejection of invalid values were also checked.
- Actual-model activation logs confirm fused layers **3–42**, no fused hash
  layers 0–2, and execution of the 256-expert/top-6 GPU kernel.
- Fresh cache=0 servers generated 64 tokens with each router mode. Tokens and
  all returned top-16 probability records were exactly equal. This compares
  serialized probabilities, not a dump of all vocabulary logits or a broad
  model-quality evaluation. Both processes retained VmSwap=0.

The isolated 256-expert/top-6/one-token normalized graph measured approximately
67.987 → 35.346 microseconds in the first operator test. It includes graph
execution and synchronization, and is not an estimate of whole-model gain.

## Whole-model protocol

The native V4 Flash Vision GGUF (five shards, 144.44 GiB), HIP/gfx1100 build,
12 CPU workers, context 262144, batch/ubatch 5120, Q8_0 KV, resident KV 65536,
aged expert cache and 256 MiB headroom/guard were retained. Each run used a
fresh server in a no-swap systemd scope. The order was **0 → 1 → 1 → 0**.
Throughput runs disabled profiling and router diagnostic logs. Binary/library
and launcher SHA-256 values were checked throughout the experiment.

The short prompt is exactly `Explain story of the romance of three kingdoms.`
An archived story response was retokenized with the current V4 tokenizer into
a fixed teacher sequence. Every measured server receives the same 12-token
chat prefix and growing teacher continuation, with 128 warmup evaluations and
256 measured one-token evaluations. Returned generation tokens are not used
as subsequent inputs. The teacher file was prepared on a separate validation
server, so its preparation did not warm one measured server more than another.
Reported native time is `prompt_ms + predicted_ms`; request wall time is also
preserved.

Each server then processes the entire `PROMPT_TEST.md` with prompt reuse off,
reasoning off, temperature 0, seed 42, and a 64-token generation limit. The
router change targets decode; a prefill throughput difference is not by itself
evidence of router acceleration.

Raw short results and a secondary comparison excluding the same outlier step
positions from every run are retained. The exclusion set is the union of
positions whose request wall time exceeds three times that run's median. This
secondary view never replaces the original averages. Asynchronous expert-cache
admission is not frozen, and two runs per mode do not establish a universal
speedup or confidence interval.

## Whole-model results

| Run | Short native ms/token | Short tok/s | Long prefill tok/s | Long decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Control 1 | 55.3844 | 18.0556 | 399.6488 | 13.5167 |
| Fused 1 | 54.4096 | 18.3791 | 400.2702 | 13.6619 |
| Fused 2 | 54.5644 | 18.3270 | 398.9979 | 13.7043 |
| Control 2 | 56.4318 | 17.7205 | 399.2817 | 13.4130 |

| Both runs per mode | Original | Fused | Throughput change |
| --- | ---: | ---: | ---: |
| Short, all samples | 17.8865 | 18.3530 | +2.61% |
| Short, common outlier position excluded | 18.0118 | 18.3435 | +1.84% |
| Long decode | 13.4649 | 13.6831 | +1.62% |
| Long prefill | 399.4653 | 399.6341 | +0.04% |

Short aggregation uses reciprocal mean native latency, with 512 samples per
mode (510 in the secondary common-step comparison). Long aggregation is the mean of the two native throughput reports. The
long prompt was 17,778 tokens in every run, followed by 64 generated tokens.
Prefill was effectively unchanged, as expected for a decode-only change.

The common excluded position is step **220**. Native evaluation there was
48.150 / 47.098 / 47.421 / **262.073 ms** in execution order. Its cause was not
established; it is not described as a delay removed by router fusion. The raw
averages remain above, and the same position is excluded from every run in
the secondary view. The two individual short comparisons both favored fusion:
about +1.79% for Control 1 → Fused 1, and +3.42% for Control 2 → Fused 2; after
common-step exclusion they were about +1.79% and +1.89%.

Cached full-model output equality is **not** established. Three long responses
(Control 1, Fused 2, Control 2) were identical; Fused 1 differed. Among measured
fixed-input short predictions, Control 1/Control 2 differed at 3 positions and
Fused 1/Fused 2 at 4 positions out of 256. These returned tokens never changed
the next teacher input. Normal asynchronous expert admission permits different
CPU/GPU expert placements and rounding; the experiment did not isolate the
cause of each output difference. The strict cache=0 equality check and the
operator-level bitwise check are separate evidence.

All process swap checks were zero. Server/library/launcher hashes were
unchanged across the comparison. All owned servers were stopped, and
`amd-smi process` reported no running GPU processes afterwards.

The result supports an **optional, small decode optimization** for this
configuration. The user subsequently requested launcher default 1; the
two-run-per-mode experiment does not establish statistical significance or model-wide
quality equivalence. Raw data and complete generation responses are retained.

## Reproduction and records

```bash
cmake --build build --target dsv4f-router-test llama-server
DSV4F_MTP=0 ctest --test-dir build -R gpu-v4-router-parity --output-on-failure
DSV4F_MTP=0 DSV4F_ROUTER_FUSED=1 ./run-server.sh
```

Use `DSV4F_ROUTER_FUSED=0` for the original route. To inspect graph selection,
set `DSV4F_ROUTER_DEBUG=1` and pass `--log-verbosity 4`; normal server verbosity
hides library-level informational layer logs. The GPU-kernel diagnostic is
also recorded directly on stderr. Diagnostics are disabled for throughput runs.

Final records are in `logs/router-fusion-v4-20261026/`: activation and native
parity JSON, per-run reports and logs, the fixed teacher plan, process mappings,
comparison JSON, and pre/post hashes. Earlier activation attempts and their
logs are preserved; layer logs hidden by normal verbosity were not evidence
that a kernel failed to execute. Performance comparisons use only the runs
whose activation check completed.

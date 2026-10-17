# Prefill expert cache experiment

The streamed MXFP4 prefill path can borrow existing service-cache slots. It
reads resident experts directly through an optional MMQ pointer table and
uploads missing experts into two alternating groups of replacement slots.
The existing `ready` / `used` events order copies and consumers. Group size
continues to use `DSV4F_PREFILL_GROUP` (default 16).

The cache path is opt-in (`DSV4F_PREFILL_CACHE=1`); its default is **off**.
`DSV4F_PREFILL_CACHE_MIN_TOKENS` sets the minimum **physical ubatch** length,
with a default of 4096. The request's total prompt length is not the trigger.
A threshold larger than `UBATCH_SIZE` will not activate the cache path.
`DSV4F_PREFILL_CACHE=0` selects the existing staging path for comparison.
Both settings are validated by the launcher and recorded in the server manifest.
This feature requires `DSV4F_PREFILL_STREAM=1`, `DSV4F_GPU_SERVICE=1`, and the
fused MXFP4 service cache on GPU 0. Unsupported geometry, an empty cache, or
insufficient replacement slots falls back to the existing streamed path.

A lease pins all selected resident experts and borrows only the ring positions
used by misses. Empty and low-score slots are preferred. Routing rows are
regrouped to compute resident experts first; the final reduction preserves the
original selected-expert order. Prefill does not add its dense routing counts to
the decode cache's frequency scores. Final ring contents remain valid cache
entries after the layer finishes.

The implementation drains service admission before leasing and completes the
layer before publishing slots back to decode. Memory reclamation cannot free
leased weights. This conservative synchronization makes source invalidation,
phase changes, and subsequent decode graph capture safe. It also introduces
some overhead, which is included in performance measurements. The original
staging reservation remains available for fallback; this experiment does not
reduce the scheduler's reserved workspace.

## Validation

```bash
cmake --build build --target llama-server dsv4f-prefill-test dsv4f-service-test -j 8
ctest --test-dir build -R 'gpu-(streamed-prefill|prefill-.*|service-.*)-parity' --output-on-failure
```

The prefill tests compare with GPU `MUL_MAT_ID`, including dense and sparse
routing, resident/missing experts, all-resident reuse, insufficient and zero
cache capacity, repeated inputs, shared multi-layer scratch, the default
4095/4096 boundary, and prefill-to-decode handoff. Profile logs include the
selected path, resident expert count, borrowed slots, and submitted H2D bytes.
Profiling synchronizes execution and must be disabled for performance results.

## Reproduction

Run one server at a time. The fixed 6324 MiB cache budget ensures equal capacity
between configurations on the RX 7900 XTX; both use MTP off and ubatch 4096.

```bash
DSV4F_MTP=0 DSV4F_EXPERT_CACHE_MIB=6324 DSV4F_PREFILL_CACHE=0 ./run-server.sh
# Repeat with DSV4F_PREFILL_CACHE=1.
python3 scripts/benchmark.py --prompt-file PROMPT_TEST.md --tokens 1 \
  --warmup-tokens 0 --runs 3 --output logs/prefill-cache-comparison.json
python3 scripts/benchmark.py --tokens 128 --runs 3 \
  --output logs/prefill-cache-decode-comparison.json
```

## RX 7900 XTX results — 2026-10-17

The optimization works, but this experiment did **not** establish a useful
full-model throughput improvement. It remains **disabled by default**.

Conditions: Ryzen 9900X, RX 7900 XTX, 192 GiB RAM, the repository's Vision GGUF,
12 CPU threads, context 65536, Q8_0 KV, pinned weights, existing default attention
settings, group 16, batch/ubatch 4096, fixed **496 slots / 6324 MiB** in every phase,
MTP off, reasoning off, temperature 0, seed 42, and `cache_prompt=false`.
`PROMPT_TEST.md` produced **17778 native prompt tokens**, with `cache_n=0` in
all runs. Each server had normal model warmup and no additional API warmup
before its first long request. Three long requests generated one token each;
then the AGENTS.md short prompt used a 32-token warmup and three 128-token runs.
Servers ran sequentially. For the same-binary ON/OFF comparison, no other GPU
test or compilation ran during requests. The extra geometry test completed during
control-server loading, before readiness. The original pre-change baseline is an
auxiliary reference: its first request overlapped a short CPU test rebuild.

| Configuration | Prefill median, tok/s | Decode median, tok/s |
| --- | ---: | ---: |
| Original implementation before changes | 372.49 | 16.05 |
| New binary, cache OFF | 371.84 | 16.49 |
| Same new binary, cache ON | 372.28 | 15.91 |

ON versus OFF changes median prefill throughput by **+0.12%**,
about 56 ms across a roughly 48-second prefill. This is too small to treat as a
confirmed benefit from three runs. The measured decode median changes by
**-3.51%**; cache contents differ after prefill and the decode
sample is small, so this does not establish a general decode regression rate.
It does rule out claiming an overall improvement in this test.

Prefill rates, first request followed by two repeats:

- Cache OFF: 363.27, 371.84, 371.92 tok/s.
- Cache ON: 368.08, 372.28, 372.62 tok/s.

All three one-token outputs matched between configurations (`Here`, `Based`,
`Based`). This is a smoke check, not proof of full-model logit identity.
The first cached layer reused 5 experts and borrowed 32 slots. Existing streaming
already overlaps transfers and computation; additional cache reuse provided no
material gain here. Separate GPU copy/compute timing was not collected, so this
experiment does not identify a definitive hardware bottleneck.

The synthetic 4096-by-2048 projection test with eight resident experts did avoid
all weight upload bytes after its first call (102 MiB to 0 MiB), and its prefill
outputs matched the GPU reference exactly. That high-residency test is not the
full-model workload. All **27 CTests** passed; tested prefill relative RMSE was 0,
and prefill-to-decode relative RMSE stayed below 0.003 (tolerance 0.04).

Both model runs retained `VmSwap=0`. Whole-process counters included about 34 MiB
of reads and 161 major faults in each run; these counters do not identify the
files involved. Observed whole-run physical VRAM peaks were
21319 MiB OFF and 21505 MiB ON, including other GPU allocations.
These are samples, not exact per-allocation memory accounting.

Raw reports are `logs/prefill-cache-control.json`,
`logs/prefill-cache-candidate.json`, their `-decode.json`, `-runtime.json`, and `-launch.log`
companions. The consolidated report is `logs/prefill-cache-comparison.json`.
Geometry coverage is in `logs/prefill-cache-geometry-parity.log`; the complete
final test run is in `logs/prefill-cache-final-ctest.log`. Comparison binaries had
identical hashes; the final rebuild only changes the default enable flag to OFF
and updates tests. Explicit ON/OFF settings retain the measured behavior.
The final full-suite rerun exposed a timing race in the existing pressure-regrowth
test: six tiny replays could finish before asynchronous admission completed.
That test now drains admission between its final replays before requiring a
restored hit, preserving graph replay and cache capacity. Poll and callback
headroom modes were each repeated five times after the fix; diagnostics and
repeat results are in `logs/prefill-cache-headroom-*.log`.
MTP-on execution was not validated in this experiment.

To opt in with the default threshold:

```bash
DSV4F_MTP=0 DSV4F_PREFILL_CACHE=1 DSV4F_PREFILL_CACHE_MIN_TOKENS=4096 ./run-server.sh
```

# CPU-first routed expert scheduling

`DSV4F_ROUTE_FIRST=1 ./run-server.sh` enables the experimental decode path.
The launcher currently defaults to `DSV4F_ROUTE_FIRST=0`. Existing router fusion,
Q8 tiles, and all-hit completion skipping keep their configured settings.

The path applies to supported GPU-service MoE operations with 1–4 tokens and no
`cb_eval`. It preserves V4 routing, including the first three hash-routing layers,
and does not change expert math. Streamed prefill keeps its existing implementation.

A separate prepare operation publishes all tokens' CPU miss requests and saves
per-token cache plans and probabilities before GPU expert work. These plans retain
GPU reader leases until their corresponding expert kernels finish. The shared
expert FFN runs before the deferred CPU-result join.

By default, `DSV4F_CPU_START_WAIT=1` makes GPU expert work wait until a CPU worker
has completed its first miss gate/up tile. This acknowledges actual routed-expert
computation, rather than only service-thread detection or work submission. It does
not wait for all CPU experts to finish. All-hit requests bypass this start wait,
including when all-hit completion skipping is disabled.

V4.1's prepare/submit split publishes CPU requests early but does not acknowledge
actual CPU computation. To measure that publication-only variant, use
`DSV4F_ROUTE_FIRST=1 DSV4F_CPU_START_WAIT=0 ./run-server.sh`.
`DSV4F_ROUTE_FIRST_DEBUG=1` prints graph activation per layer and token count;
leave diagnostics disabled during timing.

The acknowledgement belongs to the monotonically increasing ring sequence, so
captured graph replays cannot accept an acknowledgement from an earlier request.
The CPU entry-point ABI now accepts an optional acknowledgement descriptor. CPU
fallbacks pass no descriptor and keep computing the complete routed operation.

## Validation

The GPU regression cases cover poll/callback joins, 1/2/4 tokens, mixed misses and
trailing all-hit tokens, cache=0, full-hit CPU-answer mode, ring reuse, captured graph
replays, cache shrink/regrowth, and publication-only mode. A delayed CPU callback
checks acknowledgement sequence reuse and publication. The pure CPU graph fallback
is checked separately.

Performance results are recorded in `logs/route-first-v4-20261026/`. Measurements
use MTP=0, router fusion=1, Q8 tiles=1 with automatic configuration, all-hit skip=1,
and 12 CPU workers. The short prompt is
`Explain story of the romance of three kingdoms.`; the long prompt is the complete
`PROMPT_TEST.md`. The fixed teacher continuation is reused from the Q8 experiment.

On the Ryzen 9 9900X / RX 7900 XTX (`gfx1100`), all 45 CTest cases and both
Python tests passed. The 64-step cache=0 comparison produced identical predicted
tokens, identical top-16 sets, and exactly equal values for all 1,024 compared
logprobs. Activation was confirmed for all 43 layers, including hash-routing layers
0–2. This is a fixed-input numerical check; cached CPU/GPU placement can still
change rounding and generated continuations.

## ABBA measurements

Each process starts with a fresh model/cache. The short workload replays an
identical token prefix, warms up for 128 steps, then measures 256 one-token steps
using native `prompt_ms + predicted_ms`. Long requests use all 17,778 prompt tokens
and generate 64 tokens. All four long responses were identical. Context size is
262,144 with Q8 K/V; the launcher uses 65,536 resident KV tokens. Timings exclude
model loading and HTTP overhead. No profiling or diagnostic flags were enabled.

| Run | Route first | Short tok/s | Short tok/s, common delay excluded | Long prefill tok/s | Long decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| control-r1 | off | 18.0131 | 18.2852 | 398.9722 | 13.6493 |
| candidate-r1 | CPU start ACK | 18.1981 | 18.1960 | 399.8925 | 13.0288 |
| candidate-r2 | CPU start ACK | 18.2910 | 18.2883 | 401.5664 | 13.0589 |
| control-r2 | off | 18.1550 | 18.1521 | 382.9208 | 13.5153 |

Aggregated short throughput is **18.0838 → 18.2444 tok/s (+0.89%)** with all
measurements. Control-r1 step 313 took 266.196 ms; it is the sole step whose wall
time exceeded three times its run median. Excluding this same input step from
all four processes gives **18.2184 → 18.2420 tok/s (+0.13%)**. The two trimmed
pairs have opposite signs, so this does not establish a short-decode gain.

Long decode averages **13.5823 → 13.0438 tok/s (-3.96%)**. It is slower in both
pairs despite identical generated text. Prefill averages 390.9465 → 400.7294 tok/s
(+2.50%), but control-r2 is substantially slower than the other three prefill
runs. The feature changes only small decode graphs; this prefill difference
should not be attributed to CPU-start waiting without further evidence.

The current V4 model uses one MXFP4 expert format. The V4.1 advantage of publishing
several formats' requests before any format's GPU work does not directly carry
over. Waiting for actual CPU progress adds a dependency beyond V4.1's publication
step. These results support keeping `DSV4F_ROUTE_FIRST=0` by default. The
publication-only variant (`DSV4F_CPU_START_WAIT=0`) passed regression checks but
was not included in the full-model performance comparison. The 2/4-token paths
were functionally tested; MTP was disabled throughout performance measurements.

All process swap measurements were zero. Binary/launcher hashes stayed unchanged
throughout the experiment, and the owned test servers were stopped. Raw reports,
server logs, teacher inputs, the benchmark driver, comparison JSON, and verification
are under `logs/route-first-v4-20261026/`. The initial interrupted regression log
and initial compilation diagnostics are retained separately from the passing
final build and 45-test regression run.

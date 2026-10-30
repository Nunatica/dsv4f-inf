# Adaptive expert pruning

For the currently collected **DeepSeek V4 Flash Vision Exp UD-Q4_K_XL** history, start evaluation at **`DSV4F_ADAPTIVE_PRUNING=15`**. This removes **20.57 GiB** of resident routed expert weights while excluding **0.198%** of recorded expert selections overall and at most **0.376%** within any layer. Use `10` as a more conservative candidate and `20` when the extra memory savings matter. These are history-coverage judgments, not measured output-quality thresholds.

## Usage

First collect representative history with pruning disabled. Counting and persistence also work with the default full resident loading (`DSV4F_RAM_BUDGET_MIB=0`):

```bash
DSV4F_RAM_BUDGET_MIB=0 DSV4F_ADAPTIVE_ROUTING=1 DSV4F_ADAPTIVE_PRUNING=0 ./run-server.sh
```

Then restart using that history in read-only mode:

```bash
# Initial evaluation: 15% routed expert byte target, resident loading.
DSV4F_RAM_BUDGET_MIB=0 DSV4F_ADAPTIVE_ROUTING=0 DSV4F_ADAPTIVE_PRUNING=15 ./run-server.sh

# Memory-priority candidate: 20%, with an explicit 128 GiB process RAM ceiling.
DSV4F_RAM_BUDGET_MIB=131072 DSV4F_ADAPTIVE_ROUTING=0 DSV4F_ADAPTIVE_PRUNING=20 ./run-server.sh
```

`DSV4F_ADAPTIVE_PRUNING=N` accepts integers from `0` to `50`; `0` disables pruning. **N targets the percentage of main-model routed gate/up/down weight bytes to remove, across all layers.** Shared experts, attention, other weights, KV/compute buffers and MTP drafter weights are outside that target. Routing collection and pruning cannot be enabled together. Changing the history requires a server restart; the original GGUF is unchanged.

Pruning startup rejects a missing, malformed or mismatched history, a main-model layer with no recorded calls, invalid percentages, and simultaneous routing collection and pruning.

With non-mmap resident loading and O_DIRECT tiering, excluded expert planes are neither allocated nor read. Tiered descriptors pack active expert IDs and map reads back to the original GGUF offsets. RAM preload loads only retained experts, stopping after the retained set is resident if it fits the cache ceiling. Explicit mmap loading retains the original layout and masks routing. Total process RSS also includes other weights, alignment padding and runtime buffers; a positive RAM budget remains a ceiling, not a target allocation.

## Analyzed snapshot

The repository contained one `*.routing.json` at analysis time. This report uses that file and the corresponding five GGUF shard headers, without loading the full model or modifying the history.

- Analysis date: **2026-10-30, 17:56 JST**.
- History: [`models/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00001-of-00005.routing.json`](models/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL-00001-of-00005.routing.json).
- Schema version: `1`; history snapshot size: **181,911 bytes**.
- Snapshot SHA-256: `9a7dc831e06f49df37aa9ef2ece6b4b138f4fb05d740d86ca29395ffaa82a2b2`.
- Main model: **43 layers**, **256 routed experts per layer**, **top-k 6**; **11,008 layer/expert pairs** in total.
- Recorded selections: **3,017,401**; per-layer totals range from **70,136 to 70,176**. All 43 main-model rows have nonzero history. There is no MTP-layer row in this snapshot.
- Observed layer/expert pairs: **10,601 / 11,008 (96.303%)**. The remaining **407 (3.697%)** have zero recorded calls, representing **5.07 GiB** of routed weights.
- All **129 routed projection tensors** are GGML type `39` (**MXFP4**), despite the model filename containing `UD-Q4_K_XL`. Each complete expert's gate/up/down payload is **13,369,344 bytes = 12.75 MiB**.
- Total routed expert weights: **147,169,738,752 bytes = 137.0625 GiB**. Total five-shard file size: **144.44 GiB**; the remaining **7.38 GiB** includes other tensors, headers and alignment.

The history is cumulative and has no prompt, request, timestamp, modality or quality labels. The per-layer maximum divided by six is 11,696, but this is not a verified generated-token count. The observer records nonzero-weight selections in decode-style batches of up to four tokens; this can also include short prefill batches and main-model speculative verification. Startup warmup and weight transfers are excluded. The file cannot establish how diverse the collection workload was.

## Selection rule and metrics

The engine and [`scripts/model_info.py`](scripts/model_info.py) use the same global greedy selection rule:

```text
cost(layer, expert) = (historical calls / total calls in that layer)
                      / (gate + up + down weight bytes for that expert)
```

Lower-cost candidates are removed first. Missing expert counters within a valid layer count as zero; equal costs use smaller layer index, then smaller global expert ID. Every layer retains at least its original top-k count. The planner also preserves enough removable byte capacity to reach the global target under that constraint, and rejects infeasible targets. Here all expert payloads are the same size, so the target removes `ceil(11008 * N / 100)` whole experts. The rounding excess is less than one **12.75 MiB** expert.

The percentage is global, not a per-layer exclusion quota. For example, at `15`, layer 18 loses 74/256 experts (28.906%) while layers 0–2 lose none. Normal and vision routing apply the mask before top-k, refill top-k from active experts, and normalize weights as configured by the model. V4 text hash routes retain surviving choices and replace excluded choices with the most-used active experts without duplicates. **For this snapshot, layers 0–2 lose no experts at any integer setting from 1 through 30.** This is a result of their recorded usage, not a hard-coded exemption for hash layers.

The table uses these metrics:

```text
Overall historical call share excluded
  = calls belonging to all excluded experts / all recorded calls

Maximum per-layer share excluded
  = max over layers(calls belonging to excluded experts / that layer's calls)
```

The overall share is weighted by recorded calls, not an unweighted average of layer percentages. Neither metric weights selections by their routing probability or activation contribution.

## Results

All integer targets `1..30` were calculated from the same immutable snapshot. The excluded expert IDs were also checked against the native C++ planner: **exact agreement for all 30 targets, all 43 layers**. Selected targets are shown below; memory columns use actual whole-expert byte counts.

| Byte target | Experts excluded | Historical calls excluded | Maximum per-layer share (layer) | Resident expert weights saved | Routed weights remaining |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3% | 331 | 0.000% | 0.000% (—) | 4.12 GiB | 132.94 GiB |
| 4% | 441 | 0.001% | 0.010% (4) | 5.49 GiB | 131.57 GiB |
| 5% | 551 | 0.005% | 0.017% (21) | 6.86 GiB | 130.20 GiB |
| 7% | 771 | 0.018% | 0.038% (15) | 9.60 GiB | 127.46 GiB |
| 10% | 1,101 | 0.060% | 0.110% (30) | 13.71 GiB | 123.35 GiB |
| 12% | 1,321 | 0.104% | 0.194% (20) | 16.45 GiB | 120.61 GiB |
| 15% | 1,652 | 0.198% | 0.376% (26) | 20.57 GiB | 116.49 GiB |
| 20% | 2,202 | 0.446% | 0.735% (36) | 27.42 GiB | 109.65 GiB |
| 25% | 2,752 | 0.824% | 1.315% (36) | 34.27 GiB | 102.80 GiB |
| 30% | 3,303 | 1.356% | 1.996% (29) | 41.13 GiB | 95.94 GiB |

At `7`, the excluded set contains **771 experts** and **536 recorded selections (0.017764%)**; the highest per-layer excluded share is **0.038480% in layer 15**. At `12`, it contains **1,321 experts** and **3,135 selections (0.103897%)**; the highest per-layer share is **0.193798% in layer 20**. These provide intermediate choices between the previously shown byte targets.

`3` is the highest supported integer target that removes only zero-call experts in this snapshot: **331 experts, 4.12 GiB, zero recorded calls**. There are 407 zero-call experts in total, but the next integer setting, `4`, removes 441 experts and includes **34 recorded selections (0.001127%)**. At `5`, the excluded set accounts for **144 selections (0.004772%)**. The table rounds these percentages; zero-call status is based on exact counters.

## History-based starting points

| Candidate | Setting | Interpretation |
| --- | ---: | --- |
| Small zero-call-only trial | `3` | Saves 4.12 GiB and excludes no recorded selections in this snapshot. |
| Lower-pruning candidate | `7` | Saves 9.60 GiB; 0.018% overall and 0.038% maximum per-layer excluded share. |
| More conservative | `10` | Saves 13.71 GiB; 0.060% overall and 0.110% maximum per-layer excluded share. |
| Conservative intermediate | `12` | Saves 16.45 GiB; 0.104% overall and 0.194% maximum per-layer excluded share. |
| **Initial evaluation** | **`15`** | **Saves 20.57 GiB; 0.198% overall and 0.376% maximum per-layer excluded share.** |
| Memory priority | `20` | Saves 27.42 GiB; 0.446% overall and 0.735% maximum per-layer excluded share. |
| Explicit quality experiments | `25..30` | At `25`, the maximum per-layer excluded share already reaches 1.315%; losses of historical coverage rise further afterward. |

Moving from `10` to `15` saves another **6.86 GiB** while the maximum per-layer excluded share stays below **0.4%**. Moving from `15` to `20` saves another **6.85 GiB**, but the overall excluded share grows by about **2.25 times**, from **0.198% to 0.446%**, and the maximum per-layer share almost doubles. This supports evaluating `15` first and using `20` when its additional savings are useful.

Moving from `20` to `25` saves another **6.85 GiB**, while the overall share rises to **0.824%** and the maximum per-layer share reaches **1.315%**. At `30`, those figures are **1.356%** and **1.996%**. These are substantial increases in the amount of historical routing being replaced. The 0.4% and 1% figures are descriptive coverage heuristics, not established quality boundaries.

At `15`, the routed tensors retain **116.49 GiB**; at `20`, **109.65 GiB**. Subtracting saved routed bytes from the original five-shard file size gives **123.87 GiB** and **117.03 GiB**, respectively. These are file-based accounting figures, not process RSS estimates: MTP weights, KV caches, compute buffers, allocation overhead and the supervisor are additional, and GPU placement changes host residency. A positive RAM budget must be set explicitly; the launcher's default is `0`.

## Per-layer view

Layer IDs are zero-based. Each layer originally has 256 routed experts. The first three hash layers are grouped because their exclusions are identical. These counts illustrate how the global byte target distributes pruning rather than imposing the same percentage everywhere.

| Layer | Experts excluded at 15 | Historical calls excluded at 15 | Experts excluded at 20 | Historical calls excluded at 20 |
| ---: | ---: | ---: | ---: | ---: |
| 0–2 (each) | 0 | 0.000% | 0 | 0.000% |
| 3 | 25 | 0.154% | 33 | 0.311% |
| 4 | 50 | 0.305% | 64 | 0.566% |
| 5 | 33 | 0.222% | 49 | 0.549% |
| 6 | 37 | 0.254% | 52 | 0.556% |
| 7 | 27 | 0.170% | 42 | 0.472% |
| 8 | 34 | 0.205% | 55 | 0.634% |
| 9 | 23 | 0.172% | 37 | 0.439% |
| 10 | 35 | 0.215% | 49 | 0.479% |
| 11 | 36 | 0.237% | 49 | 0.489% |
| 12 | 54 | 0.229% | 68 | 0.485% |
| 13 | 35 | 0.145% | 49 | 0.406% |
| 14 | 62 | 0.262% | 78 | 0.554% |
| 15 | 73 | 0.255% | 85 | 0.485% |
| 16 | 72 | 0.291% | 87 | 0.562% |
| 17 | 56 | 0.211% | 66 | 0.390% |
| 18 | 74 | 0.294% | 84 | 0.479% |
| 19 | 22 | 0.125% | 28 | 0.247% |
| 20 | 65 | 0.256% | 79 | 0.510% |
| 21 | 59 | 0.192% | 64 | 0.291% |
| 22 | 41 | 0.175% | 50 | 0.335% |
| 23 | 41 | 0.199% | 51 | 0.403% |
| 24 | 47 | 0.241% | 68 | 0.640% |
| 25 | 40 | 0.222% | 58 | 0.579% |
| 26 | 53 | 0.376% | 70 | 0.675% |
| 27 | 34 | 0.219% | 43 | 0.406% |
| 28 | 44 | 0.286% | 62 | 0.646% |
| 29 | 47 | 0.264% | 63 | 0.577% |
| 30 | 43 | 0.251% | 58 | 0.553% |
| 31 | 25 | 0.084% | 34 | 0.271% |
| 32 | 45 | 0.268% | 61 | 0.590% |
| 33 | 30 | 0.160% | 51 | 0.579% |
| 34 | 26 | 0.141% | 35 | 0.311% |
| 35 | 29 | 0.172% | 46 | 0.512% |
| 36 | 38 | 0.262% | 61 | 0.735% |
| 37 | 31 | 0.198% | 45 | 0.467% |
| 38 | 33 | 0.174% | 43 | 0.378% |
| 39 | 17 | 0.077% | 27 | 0.268% |
| 40 | 43 | 0.214% | 66 | 0.650% |
| 41 | 39 | 0.197% | 47 | 0.359% |
| 42 | 34 | 0.150% | 45 | 0.351% |

At `15`, layer **26** has the highest excluded call share (**0.376%**); layer **18** loses the most experts (**74**). At `20`, these are layer **36** (**0.735%**) and layer **16** (**87 experts**), respectively. Expert count removed and historical call share removed measure different things; neither directly measures layer sensitivity.

## Reproducing the calculations

Run from the repository root. This uses the production launcher planner and reads GGUF headers, not full model weights. A temporary copy keeps every target for each history on one snapshot. If the live history has accumulated more calls, its SHA-256 and results will differ from this report.

```bash
PYTHONPATH=scripts python3 - <<'PY'
import hashlib
import tempfile
from pathlib import Path
from model_info import inspect, resident_pruning_plan

for history in sorted(Path('models').glob('*.routing.json')):
    contents = history.read_bytes()
    model = history.with_name(history.name.removesuffix('.routing.json') + '.gguf')
    info = inspect(model)
    print(history.name, hashlib.sha256(contents).hexdigest())
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / history.name).write_bytes(contents)
        for target in (3, 4, 5, 7, 10, 12, 15, 20, 25, 30):
            plan = resident_pruning_plan(info, target, tmp)
            layers = plan['layers']
            calls = sum(sum(layer['counts']) for layer in layers)
            excluded = sum(sum(layer['counts'][i] for i in layer['removed'])
                           for layer in layers)
            maximum = max(layer['excluded_call_share'] for layer in layers)
            print(f'{target:2d}% saved={plan["skipped_bytes"]/2**30:.2f} GiB '
                  f'excluded={100*excluded/calls:.6f}% max_layer={100*maximum:.6f}%')
PY
```

## Limits of the conclusions

These call shares are **not quality-loss percentages, percentages of affected tokens, cache miss rates or speedup predictions**. Every token visits many layers, excluded choices are replaced with other experts, and changes in hidden states can alter later routes. The counters omit selection rank, probability mass, activation magnitude, temporal order, prompt domain and quality labels. A rare or previously unobserved expert can still matter on another task; the vision-model filename alone does not prove that the collected history covers image inputs.

This report performs **offline history and weight-byte analysis only**. It does not measure real-model generation quality, perplexity, RAM/RSS changes or throughput at these settings. Keep `0` as the baseline, collect representative text and image workloads as applicable, and compare the same workloads before adopting a nonzero setting. For performance comparisons, disable MTP (`DSV4F_MTP=0`), as required by this repository's testing instructions.

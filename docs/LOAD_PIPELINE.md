# Pipelined resident weight loading

The launcher enables `DSV4F_LOAD_MODE=dio` and `DSV4F_LOAD_PIPELINE_MIB=1024`.
Pinned host weights are
allocated in tensor batches: a single reader loads the previous batch while the
main thread allocates the next one. Tensors are read in shard/offset order.
Each tensor stays intact, so a tensor larger than the target batch size receives
its own buffer. GPU weights load first to keep staging memory inexpensive.

Every buffer stays owned by the model. Cancellation, allocation failures, read
errors and validation failures join the active reader before buffers can be
released. The server becomes ready only after all weights have loaded. Mapped,
explicitly locked, custom-data and ordinary pageable CPU loading retain their
existing paths.

Set `DSV4F_LOAD_PIPELINE_MIB=0` to disable the pipeline. Valid values are
0..65536 MiB. Direct invocations of `build/bin/dsv4f-server` must set this
environment variable explicitly; the default is supplied by `run-server.sh`.
Set `DSV4F_LOAD_MODE=none` to restore buffered reads. The launcher accepts
`none` and `dio`; DIO is the default.

## Validation

`ctest --test-dir build -R '^model-load-pipeline$' --output-on-failure`
checks actual overlap, exact tensor contents, an oversized tensor, monotonic
progress, cancellation before/during/after loading, corrupt tensor validation,
truncated reads, and allocation failure while a reader is active.

The real-model comparison on 2026-10-29 JST used all five GGUF shards
(144.44 GiB), `DSV4F_MTP=0`, `CTX_SIZE=4096`, `BATCH_SIZE=512`,
`UBATCH_SIZE=512`, and the launcher's remaining defaults. Each run generated
32 tokens at temperature 0 and seed 42 with:

> Explain story of the romance of three kingdoms.

Raw results, memory readings, generated text and native log paths are saved in
`logs/load-pipeline-comparison.json`. Loading time is the supervisor's elapsed
time until its health check and resident-memory verification pass. These are
ordinary starts without flushing the filesystem cache; cache and storage state
affect the absolute times.

| Run order | Batch target | Ready time |
| --- | --- | ---: |
| 1 | Disabled | 98.090 s |
| 2 | 1024 MiB | 62.844 s |
| 3 | 1024 MiB | 47.221 s |
| 4 | Disabled | 67.521 s |

The two-run means were 82.806 s without the pipeline and 55.033 s with it
(33.54% less time). The reverse-order comparison also improved: 67.521 s versus
47.221 s. All four starts reported about 138.25 GiB RSS and zero process swap.
All four generation requests completed. Generated text matched within each
comparison pair (runs 1/2 and 3/4), with a different ending across the two pairs.
This is a startup comparison, not a sustained inference-throughput benchmark.

## Buffered I/O versus Direct I/O

On the same machine, `scripts/benchmark_load_io.py` compared `--load-mode none`
and `--load-mode dio` in the order none, dio, dio, none. MTP remained disabled,
the pipeline target remained 1024 MiB, and the same 32-token generation check
ran after every load. The filesystem cache was not flushed between runs.

| Run | Load mode | Ready time | Model file-cache peak | Memory PSI some | Direct reclaim scan |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Buffered | 53.567 s | 49.783 GiB | 884.355 ms | 250,800 pages |
| 2 | DIO | 56.383 s | 0.000 GiB | 224.279 ms | 2,560 pages |
| 3 | DIO | 53.296 s | 0.000 GiB | 58.541 ms | 512 pages |
| 4 | Buffered | 48.352 s | 58.550 GiB | 786.639 ms | 101,253 pages |

Buffered loading averaged 50.960 s and DIO averaged 54.839 s. DIO was 7.61%
slower by the two-run means (5.26% and 10.22% in the paired comparisons), but
it removed the model's page-cache copy and sharply reduced reclaim and memory
pressure. All five shards retained `O_DIRECT` throughout both DIO runs, no
buffered fallback was logged, all starts reached about 138.25 GiB RSS with zero
swap, and every generation check completed. Raw samples are in
`logs/load-io-comparison.json` when generated locally.

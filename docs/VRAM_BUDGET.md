# VRAM Budget

## Configuration

The GPU expert cache budget works together with two settings that reserve free GPU memory:

| Environment variable | Purpose | Default / minimum |
| --- | --- | --- |
| `DSV4F_EXPERT_CACHE_MIB` | Sets an upper limit on memory used for MoE experts cached on the GPU. | Automatic sizing when unset; see launcher behavior below. |
| `DSV4F_VRAM_HEADROOM_MIB` | Reserves free GPU memory. | 256 MiB in the launcher; minimum 128 MiB. |
| `DSV4F_VRAM_GUARD_MIB` | Preserves an internal memory guard at allocation and graph boundaries for library workspaces and GPU code loaded on first use. | 256 MiB; set to `0` to disable. |

## Launcher Behavior

By default, `run-server.sh` enables `DSV4F_PHASE_WORKSPACE=1` and leaves `DSV4F_EXPERT_CACHE_MIB` unset unless the user supplies a value. The native server loads the target model and MTP contexts first, then sizes the expert cache from the available VRAM while accounting for headroom and the internal guard.

An explicitly supplied `DSV4F_EXPERT_CACHE_MIB` value is preserved and used as a cache ceiling. The actual allocation can be smaller when available VRAM is insufficient to reach that ceiling while preserving the memory reserve.

When `DSV4F_MTP=1` and `DSV4F_PHASE_WORKSPACE=0`, the launcher sets a default ceiling of **10240 MiB (10 GiB)** if no value is supplied. This ceiling does not apply to the default configuration, where phase workspace management is enabled.

## Memory Reclamation

With the default GPU service enabled, memory can be reclaimed from both the initial expert cache and cache expansions made during decoding. This allows compute buffers, temporary pools, and resident KV storage to grow while preserving the configured memory reserve.

MXFP4 cache entries use a stable GPU pointer table, allowing captured graphs to continue running after cache memory is reclaimed.

If a required allocation still cannot fit after reclaiming the cache, the allocation fails rather than consuming the configured reserve.

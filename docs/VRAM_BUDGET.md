# VRAM Budget

## Configuration

The GPU expert cache budget works together with two settings that reserve free GPU memory:

| Environment variable | Purpose | Default / minimum |
| --- | --- | --- |
| `DSV4F_EXPERT_CACHE_MIB` | Sets the memory budget for MoE experts cached on the GPU. | — |
| `DSV4F_VRAM_HEADROOM_MIB` | Reserves free GPU memory. | 256 MiB in the launcher; minimum 128 MiB. |
| `DSV4F_VRAM_GUARD_MIB` | Preserves an internal memory guard at allocation and graph boundaries for library workspaces and GPU code loaded on first use. | 256 MiB; set to `0` to disable. |

## Memory Reclamation

With the default GPU service enabled, memory can be reclaimed from both the initial expert cache and cache expansions made during decoding. This allows compute buffers, temporary pools, and resident KV storage to grow while preserving the configured memory reserve.

MXFP4 cache entries use a stable GPU pointer table, allowing captured graphs to continue running after cache memory is reclaimed.

If a required allocation still cannot fit after reclaiming the cache, the allocation fails rather than consuming the configured reserve.

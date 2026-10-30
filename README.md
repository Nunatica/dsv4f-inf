# dsv4f-inf
Almost fully vibe-coded, highly experimental inference engine for DeepSeek V4 Flash.

## Target system
9900x (or any zen5 cpus such as 9800x3d) + 192GiB ddr5 5600mhz + 7900xtx (or nvidia gpus such 3090, 4090 which has at least 24gb vram, not tested)

## Target model
- [main model](https://huggingface.co/unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF/tree/main/UD-Q4_K_XL)
- [mtp drafter](https://huggingface.co/ddh0/DeepSeek-V4-Flash-GGUF/tree/a79b345f46fad371767f6a57cc3be31f28e41205)

## Dependencies and setup

The launcher requires Linux (`/proc` and cgroup v2 are used for memory/swap checks), Bash, Python 3, and a HIP or CUDA GPU backend. Run the commands below from the repository root.

### Common build tools

- A C/C++ compiler with C++17 support (GCC/G++ is used on the target system).
- CMake **3.24 or newer** and Ninja.
- OpenMP headers/runtime for the CPU backend (provided by GCC/G++ on Ubuntu/Debian; Clang builds need the matching OpenMP development package).
- Python 3 for model inspection and the server launcher. These scripts use only the Python standard library; no `pip install` or virtual environment is required.

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build python3
cmake --version  # Must be 3.24 or newer; upgrade if your distro ships an older version.
ninja --version
python3 --version
```

The llama.cpp source is included under `vendor/llama.cpp`. The provided build disables OpenSSL and uses the included `web/` assets, so no OpenSSL development package, Node.js, or npm install is needed for serving.

### AMD GPU (ROCm / HIP)

Install an AMD GPU driver compatible with your GPU and ROCm release, plus **ROCm/HIP 6.1 or newer**. The build requires the HIP compiler/device libraries and the **hipBLAS and rocBLAS development libraries**, including their CMake package files.

Follow the [official ROCm installation guide](https://rocm.docs.amd.com/en/latest/install/rocm.html) to configure the repository for your distribution and install the driver. On Ubuntu/Debian, after configuring AMD's package repository, the HIP SDK can be installed with:

```bash
sudo apt install rocm-hip-sdk
```

For the default `/opt/rocm` installation, expose the tools and check that ROCm can see your GPU:

```bash
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
hipcc --version
rocminfo
BACKEND=hip GPU_ARCH=gfx1100 ./build-engine.sh
```

`gfx1100` is the default target for the RX 7900 XTX. For another GPU, set `GPU_ARCH` to its ROCm architecture. If ROCm is installed elsewhere, set `ROCM_PATH`, add its `bin` directory to `PATH`, and set `HIP_COMPILER` to its `llvm/bin/clang++` path. Ensure your user has access to `/dev/kfd` and the GPU render devices; follow AMD's post-installation instructions for group permissions if `rocminfo` cannot access the GPU.

### NVIDIA GPU (CUDA, untested)

Install a compatible NVIDIA driver and the full CUDA Toolkit, including **nvcc, CUDA runtime development files, and cuBLAS**, using the [official CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). A driver-only installation does not provide the compiler or development libraries. Use a host GCC/G++ version supported by your CUDA Toolkit.

For a toolkit installed at `/usr/local/cuda`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvidia-smi
nvcc --version
BACKEND=cuda ./build-engine.sh
```

The CUDA build defaults to `GPU_ARCH=native`, so the GPU should be visible during compilation. Set `GPU_ARCH` explicitly when needed. `build-engine.sh` otherwise selects HIP first if `hipcc` is on `PATH`, then CUDA if `nvcc` is available; use `BACKEND` to choose when both are installed. Use separate `BUILD_DIR` values when building both backends (and pass the same `BUILD_DIR` to `run-server.sh`).

## How to run

After installing the dependencies, put **all five main-model GGUF shards** and `DeepSeek-V4-Flash-MTP-Q4_0.gguf` from the links above into `models/`. `MODEL` must point to the first main-model shard. Set `MODEL` and `MTP_MODEL` if your files are elsewhere or have different names. MTP is enabled by default; set `DSV4F_MTP=0` to run without the MTP model.

```bash
./build-engine.sh
./run-server.sh
```

The launcher enforces swap protection and, with a positive RAM budget, a cgroup v2 process-tree RAM ceiling. It uses an inherited suitable cgroup or creates a systemd user scope with `MemoryMax` and `MemorySwapMax=0`. A working systemd user session with memory control is required if these limits are not already enforced. It refuses an unbounded launch if the limits cannot be applied.

### RAM budget and adaptive experts

| Variable | Launcher default | Behavior |
| --- | ---: | --- |
| `DSV4F_RAM_BUDGET_MIB` | `0` | Total process-tree RAM ceiling in MiB. `0` loads weights into resident memory; positive values enable VRAM/RAM/O_DIRECT SSD expert tiering. |
| `DSV4F_RAM_PRELOAD` | `0` | `1` fills available expert RAM cache after contexts and warmup are ready. Requires a positive RAM budget. |
| `DSV4F_ADAPTIVE_ROUTING` | `0` | `1` records decode expert usage, saves it after responses, and uses history for RAM/VRAM preload and eviction. |
| `DSV4F_ADAPTIVE_PRUNING` | `0` | Integer `1..50`: target percentage of main-model routed expert weight bytes to exclude using historical usage per byte. `0` disables pruning. |

Run from the repository root. Routing history is saved atomically as `models/<first GGUF shard stem>.routing.json` and reused after restarting. Startup warmup and weight transfers do not increment the counters. Routing and pruning cannot be enabled together.

```bash
# Collect representative usage and preload frequently used experts.
DSV4F_RAM_BUDGET_MIB=131072 DSV4F_RAM_PRELOAD=1 DSV4F_ADAPTIVE_ROUTING=1 ./run-server.sh

# Restart with read-only history and a 20% routed-weight exclusion target.
DSV4F_RAM_BUDGET_MIB=131072 DSV4F_ADAPTIVE_ROUTING=0 DSV4F_ADAPTIVE_PRUNING=20 ./run-server.sh

# Resident loading, with physically compacted expert weights when pruning is enabled.
DSV4F_RAM_BUDGET_MIB=0 DSV4F_ADAPTIVE_PRUNING=20 ./run-server.sh
```

A positive budget keeps expert payloads on SSD and reads complete expert planes through O_DIRECT into a bounded RAM cache. Other weights and runtime buffers count against the same process ceiling. `DSV4F_RAM_RESERVE_MIB` (default `1024`) preserves runtime headroom; `DSV4F_RAM_CACHE_MIB` can impose a smaller expert-cache ceiling. Tiering disables pinned resident weights and caps prefill batches at 1024. `DSV4F_LOAD_PIPELINE_MIB` controls the resident allocation/read and RAM-preload pipelines; `0` disables pipelining. Direct native-library loading defaults to budget `0` when the environment variable is unset.

Pruning preserves at least top-k experts in every layer and applies the exclusion mask before normal and vision top-k selection. In V4 hash-routed text layers, surviving hash choices stay intact; excluded choices are replaced with the most-used active experts without duplicates. Shared experts, attention, KV/compute buffers and the MTP drafter are excluded from the percentage target. Resident non-mmap loading omits excluded expert planes from allocation and reads, while mmap and tiered loading retain the original file layout and mask routing. The GGUF is unchanged. Missing, malformed, mismatched or empty per-layer history causes pruning startup to fail. Choose pruning percentages by validating output quality for your workload; V4.1 quality settings are not assumed to transfer to V4.

With budget `0`, the startup RAM check requires the model sizes minus compacted expert bytes plus 2 GiB available, including the MTP model when enabled.

See [Adaptive expert pruning](ADAPTIVE_EXPERT_PRUNING.md) for analysis of the collected V4 routing history, byte savings by pruning target, per-layer coverage and history-based starting settings.

Once ready, open `http://localhost:8080`. The default API key is `dummykey`; set `API_KEY`, `HOST`, and `PORT` before starting to override the defaults.

For access from another device on your LAN, open `http://<server-local-ip>:8080` (for example, `http://192.168.0.64:8080`). The default `HOST=0.0.0.0` listens on all interfaces, and the web UI supports plain HTTP without SSL. Chat IDs use `crypto.getRandomValues()` when `crypto.randomUUID()` is unavailable on HTTP.

## Performance

| Implementation | Prefill (tok/s) | Decode (tok/s) |
| --- | ---: | ---: |
| dsv4f-inf | 445 | 19.69 |
| llama cpp - moe expert cache + mtp | 197 | 12.43 |
| llama cpp - full expert offloading + mtp | 195 | 11.33 |

Data was measured at target system. test was done with same prompt with PROMPT_TEST.md

## Credits
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Strata](https://github.com/Niko1221/Strata)
- [project-maya](https://github.com/mw00/project-maya)
- [ds4](https://github.com/antirez/ds4)

## License
Project itself is under MIT License. included llama.cpp source follows its own license.

## Disclaimer
This project is not intended to use for serious tasks. Use it at your own risk.

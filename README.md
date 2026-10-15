# dsv4f-inf
Almost fully vibe-coded, highly experimental serving engine for DeepSeek V4 Flash.

## Target system
9900x (or any zen5 cpus such as 9800x3d) + 192GiB ddr5 5600mhz + 7900xtx (or nvidia gpus such 3090, 4090 which has at least 24gb vram, not tested)

## Target model
- main [https://huggingface.co/unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF/tree/main/UD-Q4_K_XL]
- mtp [https://huggingface.co/ddh0/DeepSeek-V4-Flash-GGUF/tree/a79b345f46fad371767f6a57cc3be31f28e41205]

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

The launcher requires swap protection: either system swap must already be disabled, or a working systemd user session with cgroup v2 memory control must allow `systemd-run --user --scope -p MemorySwapMax=0` (the launcher applies this automatically). It refuses to start if neither is available. Keep enough free RAM for resident model loading; the initial check requires at least the total main-model file size plus 2 GiB, with additional memory needed for MTP and runtime buffers.

Once ready, open `http://localhost:8080`. The default API key is `dummykey`; set `API_KEY`, `HOST`, and `PORT` before starting to override the defaults.

For access from another device on your LAN, open `http://<server-local-ip>:8080` (for example, `http://192.168.0.64:8080`). The default `HOST=0.0.0.0` listens on all interfaces, and the web UI supports plain HTTP without SSL. Chat IDs use `crypto.getRandomValues()` when `crypto.randomUUID()` is unavailable on HTTP.

## Performance
HIP sparse attention is enabled by default in `run-server.sh`. For 512-dimensional
Q8_0 or FP16 KV caches, it gathers the union of visible rows for up to eight
queries, converts only selected Q8_0 rows to FP16, and runs the existing HIP tile
kernel. Shared K/V caches use one gathered buffer. The path requires an explicit
`n_kv_max` mask bound, at least 2048 source rows, and at least a 2:1 reduction after
padding the union bound to 256 rows. Other shapes and larger query batches use
the existing attention path.

Set `GGML_HIP_SPARSE_FA=0` before launching to compare against dense attention;
`GGML_HIP_SPARSE_FA_DEBUG=1` prints the first selected sparse shape. Restart the
server after changing these options. To benchmark the repository's test prompt:

```bash
DSV4F_MTP=0 DSV4F_EXPERT_CACHE_MIB=6324 ./run-server.sh
# In another terminal:
python3 scripts/benchmark.py --prompt-file PROMPT_TEST.md --tokens 128 --runs 3 \
    --warmup-tokens 0 --output logs/prompt-test.json
```

`ctest --test-dir build -R gpu-sparse-attention-parity --output-on-failure`
checks sparse versus dense results, including Q8_0/FP16, shared and separate K/V,
MTP-sized query groups, strided caches, attention sinks, changing masks during
graph replay, and dense fallback.

Performance comparisons disable MTP with `DSV4F_MTP=0`, as required by
`AGENTS.md`. Test MTP compatibility separately. Fix the expert cache capacity
when comparing runs if desktop VRAM usage changes between server starts.

On 2026-10-15, the command above using `PROMPT_TEST.md` (17,778 prompt tokens),
`cache_prompt=false`, and three runs per mode measured median decode throughput
of **13.97 → 14.54 tok/s (+4.12%)** with sparse off/on. Median prefill was
**237.30 → 237.09 tok/s (-0.09%)**. Both modes used 496 expert slots throughout
prefill and decode. Generated introductory wording varied within the original
path as well as between modes; the nine complete function names emitted within
the 128-token output limit matched in all three pairs. The numerical parity
test passed all 11 cases, including graph replay and MTP-sized query groups.
These MTP-disabled, fixed-cache figures are separate from the earlier default
configuration measurements below.

At target system, it reached to 230 tok/s for prefill and 18.93 tok/s for decode.
For comparison, llama cpp gives you 197 tok/s for prefill and 12.43 tok/s for decode with moe expert cache + mtp.
Performance test is done with same prompt from PROMPT_TEST.md

## Credits
- llama.cpp [https://github.com/ggml-org/llama.cpp]
- Strata [https://github.com/Niko1221/Strata]
- project-maya [https://github.com/mw00/project-maya]

## License
Project itself is under MIT License. included llama.cpp source follows its own license.

## Disclaimer
This project is not intended to use for serious tasks. Use it at your own risk.

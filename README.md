# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## gfx906 (Vega 20) branch: `qwen4exp-rccl-downfuse`

Fork of [milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp) (itself on ggml-org mainline) with three additive changes for MI50/MI60/Radeon VII (gfx906), validated on 8x MI60 (32 GB each, no P2P — every card behind its own PCIe switch) with the Qwen3.8-Flash-Next MoE model (512 experts, Q4_K_XL ~107 GB / Q8_0 ~188 GB).

### Changes on this branch

| Commit | Change | How to disable |
| --- | --- | --- |
| `efcad20` | **downfuse** — fused MoE down-projection weighted-reduction decode kernel (`ggml-cuda/moe-weighted-reduction.{cu,cuh}` + wiring in `ggml-cuda.cu`). Fuses the down-matvec, expert-weight multiply, and weighted sum/reduction (the `k_bin_bcast` add/mul + scale + sigmoid glue around `mul_mat_vec_q_moe`) into one HIP kernel, removing the launch overhead that was ~65% of decode wall on gfx906. Compiled unconditionally into the ggml-cuda build. Measured **+10.5% token generation** (19.68 vs 17.81 t/s; qwen4exp Q4_K_XL, 4 cards, 16k ctx). | runtime: `GGML_CUDA_DISABLE_FUSION=1` |
| `256e13e` | **KFD 2D-DMA wedge workaround** — large `hipMemcpy2D` (2D-strided) H2D/D2H transfers deadlock in `kfd_wait_on_events` on gfx906 (reproduced without llama.cpp; 1D copies of identical size pass). Tensor-split model load writes every shard via this path. When the macro is defined, `buffer_set/get_tensor_2d` emits per-shard 1D copies instead. **Opt-in at build time** (preprocessor macro; no CMake option yet) — the RCCL test builds had it **enabled** via `CMAKE_CXX_FLAGS`. | just don't define `GGML_WORKAROUND_2D_DMA_WEDGE` |
| `eff9bc0` | **Lazy NCCL/RCCL init** — `ncclCommInitAll` during a large concurrent H2D model load wedges the same KFD path; when the macro is defined, comm creation is deferred to the first allreduce (post-load) and falls back to the internal allreduce on init failure. **Opt-in at build time**; the RCCL test builds did *not* have it compiled in (layer mode doesn't trigger the load wedge). | just don't define `GGML_LAZY_NCCL_INIT` |

### Build

```sh
# Base (downfuse + RCCL; wedge workarounds OFF)
cmake -B build \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DGGML_HIP_RCCL=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm-6.3.4        # host RCCL install
cmake --build build -j

# + the KFD-wedge workarounds — this is how the tested RCCL builds were compiled
cmake -B build ... \
  -DCMAKE_CXX_FLAGS="-DGGML_WORKAROUND_2D_DMA_WEDGE -DGGML_LAZY_NCCL_INIT"
```

Runtime:

```sh
HSA_OVERRIDE_GFX_VERSION=9.0.6 HIP_VISIBLE_DEVICES=0,1,2,3 \
  LD_LIBRARY_PATH=/opt/rocm-6.3.4/lib \
  ./build/bin/llama-bench -m model-Q4_K_XL.gguf -ngl 99 -fa on
```

**Layer-split is the reliable multi-GPU mode on this topology.** 4 cards outperforms 8 (allreduce over PCIe without P2P is a net loss beyond 4). Tensor-split is blocked by the KFD 2D-DMA driver bug at multi-GB scale — the workaround reduces the exposure but the durable fix is a ROCm/KFD driver update.

### Reference results (Qwen3.8-Flash-Next Q4_K_XL, 4 cards 0–3, 16k ctx)

Paired A/B, identical harness (`llama-bench -ngl 99 -t 8 -fa on`):

| Config | tg128 | pp512 |
| --- | ---: | ---: |
| downfuse ON (default) | **19.68 ± 0.29** | 89.82 ± 2.94 |
| downfuse OFF (`GGML_CUDA_DISABLE_FUSION=1`) | 17.81 ± 0.13 | 87.74 ± 5.77 |

Fork control before the kernel: 16.73 t/s (different harness flags — not comparable to the paired A/B). Full evidence: `~/tuning/TIER1-A-B-RESULTS.md` on the build host.

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

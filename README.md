# llama.cpp

## gfx906 (Vega 20 / MI60) branch: `qwen4exp-rccl-downfuse`

Fork of [milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp) (itself on ggml-org mainline) with additive changes for MI50/MI60/Radeon VII (gfx906), validated on the 8× MI60 box **radllm** with the Qwen3.8-Flash-Next MoE model (177B / ~3B active, 512 experts top-10, Q4_K_XL ~107 GB / Q8_0 ~188 GB).

### The host: radllm — hardware & PCIe topology

This is what dictates the performance ceiling, so it's documented here rather than in a wiki.

| | |
| --- | --- |
| Motherboard | **BIOSTAR TB360-BTC Pro 2.0** (Intel B360 chipset) |
| CPU | Intel Core i3-8350K — 4 cores / 4 threads @ 4.0 GHz, 8 MiB L3, 1 NUMA node |
| RAM | 16 GiB DDR4 (1 node), 4 GiB swap, no hugepages |
| GPU | **8× AMD Radeon Instinct MI60 (Vega 20, `0x66a0`)**, 32 GiB HBM2 each = **256 GiB aggregate** |
| GPU link | gfx906, **no P2P, no Infinity Fabric** between cards |
| Storage | 2.7 TiB ZFS (`rpool`), 418 GiB of models |
| OS | Ubuntu 24.04.4 LTS, kernel 6.17.0-14-generic |
| ROCm | 6.3.4 / 7.0.2 / 7.1.1 present; builds target `gfx906` |

**The topology, and why it matters.** The B360 is a *consumer* chipset on the Cannon Lake PCH. From `lspci -tv`:

```
-[0000:00]-+-1b.0-[02-04]----00.0-[03-04]----00.0-[04]----00.0  Vega 20  (GPU0)
           +-1b.5-[05-07]----00.0-[06-07]----00.0-[07]----00.0  Vega 20  (GPU1)
           +-1c.5-[09-0b]----00.0-[0a-0b]----00.0-[0b]----00.0  Vega 20  (GPU2)
           +-1c.6-[0c-0e]----00.0-[0d-0e]----00.0-[0e]----00.0  Vega 20  (GPU3)
           +-1c.7-[0f-11]----00.0-[10-11]----00.0-[11]----00.0  Vega 20  (GPU4)
           +-1d.0-[12-14]----00.0-[13-14]----00.0-[14]----00.0  Vega 20  (GPU5)
           +-1d.1-[15-17]----00.0-[16-17]----00.0-[17]----00.0  Vega 20  (GPU6)
           +-1d.3-[18-1a]----00.0-[19-1a]----00.0-[1a]----00.0  Vega 20  (GPU7)
```

Every GPU is a **leaf behind its own PCH PCIe bridge**, and all eight bridges hang off the same CPU-attached PCH root. There is **no GPU-to-GPU path** — not a switch, not Infinity Fabric, not a peer route. Consequences:

- **No P2P.** `cuMemGetPeerAccess` / RCCL P2P is unavailable (`VMM: no`); any cross-card data movement must be staged through host DRAM via the PCH.
- **Allreduce is host-mediated.** With 4 CPU cores and a single NUMA node, the cross-card reduce at every layer boundary is the dominant off-GPU cost (measured ~17.6 ms/tok = ~34% of decode wall).
- **More cards ≠ faster.** Layer-split serializes over the PCH, so 8 cards is *slower* than 4 (14.26 vs 20.79 t/s at 16k). **4 cards is the practical optimum**; 2 cards OOM on the 107 GB model.
- **Tensor-split is blocked** by a gfx906 KFD 2D-DMA driver deadlock at multi-GB scale (see workarounds below).
- **The 16 GiB host RAM** caps how much can be staged for CPU offload / model loading, and is why KV-cache-in-RAM experiments are infeasible.

This is a *topology wall*, not a tuning problem: even with zero off-GPU overhead the on-GPU ceiling at 16k is ~29.6 t/s, already below the 40 t/s target. See `~/tuning/WALL-CHARACTERIZATION.md`.

### Changes on this branch

| Commit | Change | How to disable |
| --- | --- | --- |
| `efcad20` | **downfuse** — fused MoE down-projection weighted-reduction decode kernel (`ggml-cuda/moe-weighted-reduction.{cu,cuh}` + wiring in `ggml-cuda.cu`). Fuses the down-matvec, expert-weight multiply, and weighted sum/reduction (the `k_bin_bcast` add/mul + scale + sigmoid glue around `mul_mat_vec_q_moe`) into one HIP kernel, removing the launch overhead that was ~65% of decode wall on gfx906. Compiled unconditionally into the ggml-cuda build. Measured **+10.5% token generation** (19.68 vs 17.81 t/s; qwen4exp Q4_K_XL, 4 cards, 16k ctx). | runtime: `GGML_CUDA_DISABLE_FUSION=1` |
| `256e13e` | **KFD 2D-DMA wedge workaround** — large `hipMemcpy2D` (2D-strided) H2D/D2H transfers deadlock in `kfd_wait_on_events` on gfx906 (reproduced without llama.cpp; 1D copies of identical size pass). Tensor-split model load writes every shard via this path. When the macro is defined, `buffer_set/get_tensor_2d` emits per-shard 1D copies instead. **Opt-in at build time** (preprocessor macro; no CMake option yet) — the RCCL test builds had it **enabled** via `CMAKE_CXX_FLAGS`. | just don't define `GGML_WORKAROUND_2D_DMA_WEDGE` |
| `eff9bc0` | **Lazy NCCL/RCCL init** — `ncclCommInitAll` during a large concurrent H2D model load wedges the same KFD path; when the macro is defined, comm creation is deferred to the first allreduce (post-load) and falls back to the internal allreduce on init failure. **Opt-in at build time**; the RCCL test builds did *not* have it compiled in (layer mode doesn't trigger the load wedge). | just don't define `GGML_LAZY_NCCL_INIT` |

RCCL (ROCm collective comms) is available via `GGML_HIP_RCCL=ON` and provides a working 4-GPU allreduce path on this box (verified: `allreduce[0][0]=6.0`), replacing llama.cpp's internal fallback that deadlocked.

### MTP / speculative decoding on qwen4exp — why it's a net loss here

The Qwen3.8-Flash-Next MTP sidecar (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, 2.79 GB) was investigated as a TG lever. **It does not help on this model on this box — it hurts.** Findings:

- **The sidecar is tiny, not a 50-layer model.** `strings` shows it contains only `blk.48.*` (one A3B MoE layer) + the `nextn` head, plus borrowed `token_embd` / `lm_head` from the target. So draft *compute* is negligible — GPU-draft vs CPU-draft moved the cost by only ~50 ms/token. The overhead is not in the draft.
- **~450 ms/step is *fixed* overhead in the MTP verification path**, independent of draft depth and draft compute:

  | Config | tok/step | ms/step | t/s |
  | --- | ---: | ---: | ---: |
  | non-MTP | 1 | **70** | **14.26** |
  | MTP d=1 (GPU draft) | 2 | 527 | 4.88 |
  | MTP d=2 (GPU draft) | 3 | 513 | 5.94 |
  | MTP d=2 (CPU draft) | 3 | ~660 | 4.53 |

  d=1 and d=2 cost ~the same per step → the cost is a **fixed per-step constant, not a per-verified-token term**.
- **The break-even math never works.** MTP only wins if `step_cost < tokens_verified × non-MTP_step`. At 70 ms/token, d=2 needs <210 ms but costs ~513; d=1 needs <140 ms but costs ~527. **Non-MTP 14.26 t/s is the real ceiling; MTP is a net loss.**
- **A second bug in the tree was fixed** on the way: a draft-only export declares the full block count but ships only the MTP block, so trunk tensors load null and context reservation segfaulted. The borrow-check / draft-only-load guard (commits `48e638e`, `2364180`) catches this cleanly.

Bottom line: for qwen4exp on gfx906, run **non-MTP** and spend the optimization budget on the decode path (downfuse) and the multi-GPU topology (4-card layer-split), not on speculative decoding.

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

Runtime (4-card layer-split, the reliable mode on this topology):

```sh
HSA_OVERRIDE_GFX_VERSION=9.0.6 HIP_VISIBLE_DEVICES=0,1,2,3 \
  LD_LIBRARY_PATH=/opt/rocm-6.3.4/lib \
  ./build/bin/llama-bench -m model-Q4_K_XL.gguf -ngl 99 -fa on
```

**Layer-split is the reliable multi-GPU mode on this topology.** 4 cards outperforms 8 (host-mediated allreduce over the PCH is a net loss beyond 4). Tensor-split is blocked by the KFD 2D-DMA driver bug at multi-GB scale — the workaround reduces the exposure but the durable fix is a ROCm/KFD driver update.

### Reference results (Qwen3.8-Flash-Next Q4_K_XL, 4 cards 0–3, 16k ctx)

Paired A/B, identical harness (`llama-bench -ngl 99 -t 8 -fa on`):

| Config | tg128 | pp512 |
| --- | ---: | ---: |
| downfuse ON (default) | **19.68 ± 0.29** | 89.82 ± 2.94 |
| downfuse OFF (`GGML_CUDA_DISABLE_FUSION=1`) | 17.81 ± 0.13 | 87.74 ± 5.77 |

Fork control before the kernel: 16.73 t/s (different harness flags — not comparable to the paired A/B). Full evidence: `~/tuning/TIER1-A-B-RESULTS.md`, `~/tuning/WALL-CHARACTERIZATION.md`, `~/tuning/qfmtp/TG-FINDINGS.md` on the build host.


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

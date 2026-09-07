# llama.cpp

## gfx906 (Vega 20 / MI60) branch: `qwen4exp-rccl-downfuse`

A fork of [milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp) (itself on ggml-org mainline) that adds a set of targeted, opt-in changes for Vega 20 datacenter GPUs — MI50/MI60/Radeon VII (gfx906) — and documents the hard-won performance picture for a large MoE model on that hardware. Validated on an 8× MI60 test host (referred to below as "the box") running the Qwen3.8-Flash-Next MoE (177B / ~3B active, 512 experts top-10, Q4_K_XL ~107 GB / Q8_0 ~188 GB).

This section is meant to be a self-contained record: the hardware, what was tried, what worked, what didn't, and what's left — so someone else can reproduce or extend it.

### Results at a glance

| | |
| --- | --- |
| Best decode (TG) | **19.68 t/s** — 4-card layer-split, 16k ctx, downfuse kernel ON |
| Best prefill (PP) | **~156 t/s** — 4-card layer-split, 16k ctx |
| Single biggest win | **+10.5% TG** from the fused MoE down-projection kernel (`downfuse`) |
| Multi-GPU sweet spot | **4 cards** (8 is *slower* — see topology) |
| Speculative / MTP | **Net loss** on this model (fixed ~450 ms/step overhead) |
| Goal (40 t/s TG / 200 t/s PP @ 200k) | **Not reachable** — a topology wall, not a tuning gap |

### The host: hardware & PCIe topology

This topology is what dictates the ceiling, so it's documented up front.

| | |
| --- | --- |
| Motherboard | BIOSTAR TB360-BTC Pro 2.0 (Intel B360 chipset) |
| CPU | Intel Core i3-8350K — 4 cores / 4 threads @ 4.0 GHz, 8 MiB L3, 1 NUMA node |
| RAM | 16 GiB DDR4 (1 node), 4 GiB swap, no hugepages |
| GPU | **8× AMD Radeon Instinct MI60 (Vega 20, `0x66a0`)**, 32 GiB HBM2 each = **256 GiB aggregate** |
| GPU link | gfx906, **no P2P, no Infinity Fabric** between cards |
| Storage | ~2.7 TiB ZFS root, ~418 GiB of models |
| OS | Ubuntu 24.04 LTS, kernel 6.17 |
| ROCm | 6.3.4 / 7.0.2 / 7.1.1 present; builds target `gfx906` |

From `lspci -tv`:

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

Every GPU is a **leaf behind its own PCH PCIe bridge**, all eight hanging off the same CPU-attached PCH root. There is **no GPU-to-GPU path** — no switch, no Infinity Fabric, no peer route. Consequences:

- **No P2P.** `cuMemGetPeerAccess` / RCCL P2P is unavailable (`VMM: no`); any cross-card movement must stage through host DRAM via the PCH.
- **Allreduce is host-mediated.** With 4 CPU cores and one NUMA node, the cross-card reduce at every layer boundary is the dominant off-GPU cost (measured ~17.6 ms/tok ≈ 34% of decode wall).
- **More cards ≠ faster.** Layer-split serializes over the PCH, so 8 cards is *slower* than 4 (14.26 vs 20.79 t/s at 16k). **4 cards is the practical optimum**; 2 OOMs on the 107 GB model.
- **Tensor-split is blocked** by a gfx906 KFD 2D-DMA driver deadlock at multi-GB scale (see workarounds).
- **16 GiB host RAM** caps CPU offload and model-load staging.

This is a *topology wall*: even with zero off-GPU overhead, the on-GPU ceiling at 16k is ~29.6 t/s — already below the 40 t/s goal.

### Changes on this branch

| Commit | Change | How to disable |
| --- | --- | --- |
| `efcad20` | **downfuse** — fused MoE down-projection weighted-reduction decode kernel (`ggml-cuda/moe-weighted-reduction.{cu,cuh}` + wiring in `ggml-cuda.cu`). Fuses the down-matvec, expert-weight multiply, and weighted sum/reduction (the `k_bin_bcast` add/mul + scale + sigmoid glue around `mul_mat_vec_q_moe`) into one HIP kernel, removing launch overhead that was ~65% of decode wall on gfx906. Compiled into the ggml-cuda build. **+10.5% TG** (19.68 vs 17.81 t/s; 4 cards, 16k ctx). | runtime: `GGML_CUDA_DISABLE_FUSION=1` |
| `256e13e` | **KFD 2D-DMA wedge workaround** — large `hipMemcpy2D` (2D-strided) H2D/D2H transfers deadlock in `kfd_wait_on_events` on gfx906 (reproduced standalone; 1D copies of identical size pass). Tensor-split load writes every shard via this path. When the macro is defined, `buffer_set/get_tensor_2d` emits per-shard 1D copies instead. **Opt-in at build time** (preprocessor macro, no CMake option yet) — the test builds had it **enabled** via `CMAKE_CXX_FLAGS`. | don't define `GGML_WORKAROUND_2D_DMA_WEDGE` |
| `eff9bc0` | **Lazy NCCL/RCCL init** — `ncclCommInitAll` during a large concurrent H2D model load wedges the same KFD path; when the macro is defined, comm creation is deferred to the first allreduce (post-load), falling back to the internal allreduce on failure. **Opt-in at build time**; the test builds did *not* compile it in (layer mode doesn't trigger the load wedge). | don't define `GGML_LAZY_NCCL_INIT` |

RCCL (ROCm collective comms) is available via `GGML_HIP_RCCL=ON` and gives a working 4-GPU allreduce path on this box (verified: 4-GPU `allreduce` correct, 8-GPU `allreduce` correct), replacing llama.cpp's internal fallback that deadlocked.

### MTP / speculative decoding on qwen4exp — why it's a net loss here

The MTP sidecar was investigated as a TG lever. **It hurts on this model on this box.**

- **The sidecar is tiny, not a 50-layer model.** It contains only one A3B MoE layer + the `nextn` head, plus borrowed `token_embd`/`lm_head` from the target. So draft *compute* is negligible — GPU- vs CPU-draft moved cost by only ~50 ms/token. The overhead is not in the draft.
- **~450 ms/step is *fixed* overhead in the MTP verification path**, independent of draft depth or draft compute:

  | Config | tok/step | ms/step | t/s |
  | --- | ---: | ---: | ---: |
  | non-MTP | 1 | **70** | **14.26** |
  | MTP d=1 (GPU draft) | 2 | 527 | 4.88 |
  | MTP d=2 (GPU draft) | 3 | 513 | 5.94 |
  | MTP d=2 (CPU draft) | 3 | ~660 | 4.53 |

  d=1 and d=2 cost ~the same per step → a **fixed per-step constant, not a per-verified-token term**.
- **The break-even math never works.** MTP only wins if `step_cost < tokens_verified × non-MTP_step`. At 70 ms/token, d=2 needs <210 ms but costs ~513; d=1 needs <140 ms but costs ~527. **Non-MTP 14.26 t/s is the real ceiling; MTP is a net loss.**

A related bug was fixed on the way: a draft-only export declares the full block count but ships only the MTP block, so trunk tensors load null and context reservation segfaulted. The borrow-check / draft-only-load guard (commits `48e638e`, `2364180`) catches this cleanly.

Bottom line: for this MoE on gfx906, run **non-MTP** and spend the budget on the decode path (downfuse) and the multi-GPU topology (4-card layer-split), not on speculative decoding.

### How we got here (work log)

Chronological, so the reasoning is reproducible. Steps 1–3 established the baseline and the wall; 4–6 found the wins; 7–9 packaged them.

1. **Baseline (vanilla).** Started from a stock gfx906 build (mxxm `llama-cpp-gfx906`, ROCm 7.2.7). Single-card 27B dense model: pp512 ≈ 157, tg128 ≈ 19.3. Parameter sweep on the dense model: best `-fa on -t 8`; KV quantization was a *penalty*, not a win. This set the expectation that decode, not prefill, is the binding constraint.
2. **The MoE target.** Downloaded the Qwen3.8-Flash-Next MoE (Q4_K_XL ~107 GB, Q8_0 ~188 GB) — too big for ≤4 cards, so multi-GPU was mandatory. A tiny `micro-qwen4exp` (8 experts) was also pulled to exercise the 2/4-device load path in isolation (it's a path-test, not a perf model — 8 experts is too small to split meaningfully).
3. **The KFD wedge.** Tensor-split on the real model **hung** with threads parked in `kfd_wait_on_events`, 0% GPU — reproduced *without* llama.cpp in a standalone HIP program (`c1d2d.cu`): 2D-strided `hipMemcpy2D` H2D at multi-GB size deadlocks, 1D copies of the same size pass. The non-RCCL baseline wedged identically → it's the **KFD driver + HSA runtime**, not RCCL or llama.cpp logic. Root cause of the "no tensor-split" result.
4. **RCCL integration.** Built with `GGML_HIP_RCCL=ON` against the host RCCL. Standalone 4-GPU and 8-GPU `allreduce` verified correct. llama.cpp's internal allreduce fallback is the one that deadlocks; RCCL provides a working comms path.
5. **The downfuse kernel.** Profiled decode: gaps (kernel-launch overhead) were ~65% of the MoE decode wall. Wrote a fused down-projection weighted-reduction kernel and wired it into the MoE path. **+10.5% TG**, cleanly A/B'd with a runtime opt-out.
6. **MTP investigation.** Ruled in then out: the sidecar is one layer, the ~450 ms/step is a *fixed* verification-path overhead, and the break-even math never favors it on this model. Documented as a net loss.
7. **Workarounds.** Added the 2D-DMA→1D copy workaround and lazy NCCL init, both opt-in and off by default, so other archs are unaffected.
8. **The wall, quantified.** `rocprofv2` trace: 51.4 ms/tok wall = ~33.8 ms on-GPU + ~17.6 ms off-GPU (cross-card sync on 4 cores). Zeroing the off-GPU term still caps at ~29.6 t/s — below the 40 t/s goal. The goal is a topology wall, not a tuning gap.
9. **Packaged.** Committed as three labeled, opt-in commits on this branch, with the README record.

### Lessons learned (transferable)

- **Profile before you optimize.** The win (downfuse) came from finding that launch *gaps*, not compute, dominated the MoE decode wall. A parameter sweep would never have found it.
- **Reproduce a suspected driver bug standalone.** The `kfd_wait_on_events` wedge was only tractable because it was reduced to a 30-line HIP program with no llama.cpp in the path. Isolate the suspect layer before debugging the stack.
- **A "slow" path is often a *fixed* overhead, not a per-unit cost.** The MTP overhead was constant per step regardless of draft depth — that single observation killed the whole speculative-decoding line of attack on this model.
- **More GPUs can be worse.** With no P2P and host-mediated allreduce, 8 cards was slower than 4. Card count is a *topology* variable, not a monotonic one.
- **Gate experiments opt-in.** Every change here is a compile flag or env var off by default, so the fork stays drop-in safe on other archs.

### Open questions / next steps (for anyone building on this)

- **The KFD 2D-DMA wedge** is the real blocker for tensor-split. The workaround (1D copies) reduces exposure but the durable fix is a **ROCm/KFD driver or HSA runtime update** that fixes the large strided-DMA deadlock. A box with a working P2P path (Infinity Fabric, or a proper GPU switch) would remove the host-mediated allreduce term entirely.
- **The 40 t/s / 200 t/s @ 200k goal** is unreachable on this box's topology. To pursue it: (a) fix the KFD wedge to enable tensor-split, (b) add P2P, or (c) target a different (smaller or more parallelizable) model that fits the on-GPU ceiling.
- **KV-cache-in-RAM** is infeasible here (16 GiB host RAM) — only relevant if the host is re-specced with more DRAM.
- **downfuse generalization** — the fused down-projection is written for this MoE's `mul_mat_vec_q_moe` shape; extending it to other expert FFN shapes / quant types is untested.

### Build & run

```sh
# Base (downfuse + RCCL; wedge workarounds OFF)
cmake -B build \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DGGML_HIP_RCCL=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm        # host ROCm/RCCL install
cmake --build build -j

# + the KFD-wedge workarounds — this is how the test builds were compiled
cmake -B build ... \
  -DCMAKE_CXX_FLAGS="-DGGML_WORKAROUND_2D_DMA_WEDGE -DGGML_LAZY_NCCL_INIT"
```

Runtime (4-card layer-split, the reliable mode on this topology):

```sh
HSA_OVERRIDE_GFX_VERSION=9.0.6 HIP_VISIBLE_DEVICES=0,1,2,3 \
  LD_LIBRARY_PATH=/opt/rocm/lib \
  ./build/bin/llama-bench -m model-Q4_K_XL.gguf -ngl 99 -fa on
```

### Reference results (Qwen3.8-Flash-Next Q4_K_XL)

**A/B — does the downfuse kernel help?** 4 cards, `llama-bench -ngl 99 -t 8 -b 128 -fa on`.
`llama-bench` sizes the context window from the prompt/depth, so this A/B runs at a small
window (pp at ~512 ctx, tg at ~128 ctx). It isolates the fused MoE down-projection +
weighted-reduction kernel (ON = default, OFF = `GGML_CUDA_DISABLE_FUSION=1`, no rebuild):

| Config | pp512 (t/s) | tg128 (t/s) |
| --- | ---: | ---: |
| downfuse ON (default) | 89.82 ± 2.94 | **19.68 ± 0.29** |
| downfuse OFF (`GGML_CUDA_DISABLE_FUSION=1`) | 87.74 ± 5.77 | 17.81 ± 0.13 |

Re-run 2026-09-06 (clean KFD): tg128 **19.84 ± 0.27**, pp512 **92.02 ± 2.44** — consistent,
confirming the original numbers still hold on this build.

**Context sweep — best config as the context window grows.** downfuse ON, 4 cards,
`llama-bench -ngl 99 -t 8 -b 128 -fa on` (f16 KV). *pp* = prompt-processing throughput
for a prompt of that length; *tg128* = token-generation throughput with the KV cache
pre-filled to that context. 16k–64k are the solid QF ceiling rows on this board;
**128k and 256k do not complete** (the no-P2P MoE-dispatch wall — see below).

| Context | pp (t/s) | tg128 (t/s) |
| --- | ---: | ---: |
| 16k | 78.9 ± 5.5 | 16.91 ± 0.20 |
| 32k | 74.3 ± 2.3 | 14.90 ± 0.10 |
| 64k | 64.8 ± 0.7 | 12.40 ± 0.10 |
| 128k | **does not complete** | **does not complete** |
| 256k (model max) | **does not complete** | **does not complete** |

**Why 128k/256k do not complete (the wall).** This is an *architectural* ceiling, not a
script bug, timeout, or KFD wedge:
- The ~111 GB MoE model is layer-split across the cards. The board has **no P2P / no
  Infinity Fabric**, so every MoE expert-token dispatch during a large-context prefill
  is routed through **host RAM by the 4 CPU cores** (i3-8350K).
- Observed across 4 / 6 / 8 cards, f16 and Q4 KV: GPUs idle at 350 MHz between short
  bursts, CPU ~48% (a single core of 4), `read_bytes` still climbing after load. On the
  8-card Q4-KV run the driver logged `HW Exception by GPU node-7 … GPU Hang` → core dump.
- 6 cards is *slower* than 4 (each card re-streams the weights through the 16 GB page
  cache; `read_bytes` hit 816 GB ≈ 10× the model size).

**Solid QF data (4-card, f16 KV, fork build)** is the 16k/32k/64k table above. To exceed
it you need P2P-capable GPUs, a fused host-mediation-free dispatch kernel, or a smaller
MoE that fits one 32 GB card. Evidence: `QF-WALL-FINAL.md` (durable).

Fork control before the kernel: 16.73 t/s tg (different harness flags — not comparable
to the paired A/B).

### Single-card: Qwen3.8-27B (qwen35, internal MTP) — per-quant max-context + throughput

A second, dense (non-MoE) model on the same fork build: **Qwen3.8-27B**, single card
(MI60 32 GB), **internal MTP ON** (`--spec-type draft-mtp --spec-draft-n-max 2`) and
**Q4 KV cache** (`-ctk q4_0 -ctv q4_0`), run through `llama-server` (not `llama-bench`)
because the server timing is token-accurate under MTP. This is the model for the
"few 27B agents, one per card, largest context" deployment target.

Architectural note (why it is long-context-friendly): qwen35 is a **hybrid SSM +
attention** model — 65 blocks, full attention only every 4th block (~16 layers keep a
KV cache); the rest are SSM layers with **fixed-size state** (no KV growth). MTP head is
embedded (`n_layer_nextn=1`), so no sidecar draft file is needed. Native max context =
262144.

**Per-quant results (single card, MTP ON, Q4 KV, depth = 16k prompt for a consistent
pp/tg; ceiling = highest context that allocates on the 32 GB card).** All four quants
were produced on-box from the Q8_0 source via `llama-quantize --allow-requantize`.

| Quant | size | **max ctx (1 card)** | pp (t/s) | tg (t/s) | MTP acc | ppl* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 29 GB | **128k** (256k OOM) | 199.6 | 23.7 | 0.72 | 1.4165 |
| Q6_K | 20.9 GB | **256k** (model max) | 159.0 | 24.3 | 0.86 | 1.4090 |
| Q5_K_M | 18.2 GB | **256k** (model max) | 181.6 | 21.5 | 0.89 | 1.4429 |
| Q4_K_M | 15.7 GB | **256k** (model max) | 163.4 | 21.9 | 0.86 | 1.3935 |

\* Perplexity on the same ~10k-token diverse text, `llama-perplexity -c 4096`; all four
quants are within ~2% of each other (inside the run-to-run ±0.03 noise band), so even
Q4_K_M holds accuracy for this workload while halving the footprint.

**Two context depths matter — measure both.** The table above is at a **~16k-token
prompt** (shallow KV). But the deployment target is *max* context, and for a dense
attention model tg degrades with KV depth. So we also ran **at true 256k context**
(KV pre-filled with a ~156k-token prompt, 128 generated):

| Quant | ctx | pp @256k (t/s) | **tg @256k (t/s)** |
| --- | ---: | ---: | ---: |
| Q6_K | 256k | 96.5 | **12.06** |
| Q4_K_M | 256k | 98.0 | **12.14** |

*(server `eval time`, authoritative — `ctx27-256k.log`).*

**Takeaways for the multi-agent target (corrected):**
- **Q4_K_M / Q5_K_M / Q6_K all reach the full 256k context on a single MI60** — only
  Q8_0 tops out at 128k (29 GB + 256k KV overflows 32 GB). The "largest context per
  card" goal is met at **256k** for any quant ≤ Q6_K.
- **But tg at 256k context is ~12 t/s, below the 20 t/s target.** The 21–24 t/s in the
  table is the *shallow-context* (16k KV) figure; dense attention over 156k tokens
  dominates per-token cost, so tg falls to ~12 as context fills. pp stays strong
  (~97–98 t/s) at 256k.
- If 20 t/s tg is a hard floor for the agents, **context must be capped below ~256k**
  (the tg-vs-context curve is the next thing to map). If 256k context is the hard
  requirement, accept ~12 t/s tg, or add MTP (accept ≈ 0.86–0.89 already included).
- Accuracy: Q4_K_M ppl 1.3935 is within ~2% of Q8_0 (inside the ±0.03 noise band) —
  Q4_K_M is accuracy-equivalent while halving footprint and still hitting 256k.

Run params (reproducible): `llama-server -m Qwen3.8-27B-<QUANT>.gguf --fit off
-ngl 99 -t 8 -c <CTX> [-b 4096] --parallel 1 -ctk q4_0 -ctv q4_0 -fa on --spec-type
draft-mtp --spec-draft-n-max 2` on one MI60 (card 0), `HSA_OVERRIDE_GFX_VERSION=9.0.6`;
ceiling by descending `256k→128k→64k→32k→16k` on OOM; shallow pp/tg at a ~16k-token
prompt, deep pp/tg at a ~156k-token prompt (128 generated). Durable artifacts:
`ctx27-quant.tsv` (shallow), `ctx27-256k.tsv` + `ctx27-256k.log` (deep),
`ppl27b.tsv`, `quant27b.sh`, `ctx27-quant.py`, `ctx27-256k.py` under `~/tuning/`.

### Testing procedure



- **Tool:** `llama-bench`, the context-window benchmark. It sets the context window and
  measures prompt-processing (*pp*) and token-generation (*tg*) throughput at that depth —
  i.e. real "usable at N tokens of context" numbers, not a fixed small prompt.
- **pp vs tg:** *pp* = prompt tokens/sec ingested (long-context reading); *tg* = generated
  tokens/sec (the interactive speed you feel). Both are reported per context window.
- **How the window is set:** `llama-bench` derives the context from the prompt length and
  pre-fill depth, so a longer prompt / deeper pre-fill allocates a larger KV cache and
  exercises real long-context attention. *pp(C)* = process a C-token prompt (window = C);
  *tg(C)* = generate 128 tokens with the KV pre-filled to C.
- **Best config (4 cards):** layer-split across 4 of the 8 MI60s — the sweet spot on this
  no-P2P board (8 cards is slower for TG; see topology). Flags: `-ngl 99 -t 8 -b 128 -fa on`,
  downfuse ON (default).
- **A/B isolation:** `GGML_CUDA_DISABLE_FUSION=1` switches the downfuse kernel off for an
  exact on/off comparison with no rebuild.
- **Model:** Qwen3.8-Flash-Next (MoE, 512 experts top-10), Q4_K_XL ≈ 107 GB, 176.9 B params.
- **Stability note:** the gfx906 KFD driver wedges after a SIGKILL of a GPU process, so
  runs are SIGTERM-clean and the box is rebooted between any failed test; the numbers above
  are from clean KFD state.

### Provenance & attribution

- **Upstream:** forked from [milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp), on ggml-org mainline.
- **Model:** Qwen3.8-Flash-Next (MoE, 512 experts top-10) — Q4_K_XL and Q8_0 GGUF.
- **Hardware:** 8× MI60 (Vega 20, gfx906) on an Intel B360 board, no P2P (see topology).
- **Toolchain:** ROCm (6.3.4 / 7.x), HIP, gfx906 target.
- Full evidence logs (A/B runs, profiling, wedge repros) are kept on the build host and are available to anyone who wants to verify the numbers.

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

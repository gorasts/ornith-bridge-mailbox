# Strata setup and tuning audit — 7 October 2026

The earlier claim that GORAST has no pathway to 200–300 PP was unsupported. We measured one narrow configuration family, not a hardware ceiling. Published results establish that Strata can run much faster, including on legacy AMD. They do not establish the speed of three Vega 64s on this exact real prompt.

The most actionable configuration discrepancy is **fixed prefill 512**. Upstream's normal setup uses **prefill auto**, and the closest legacy-AMD report uses **4096**. Further layer balancing should wait until the actual prefill phases and a larger safe chunk have been tested.

No benchmark, model download, build, service restart, or GPU tuning change was performed during this audit. Ornith remains inactive; gpu-pinguard remains active.

## Published comparisons

| Setup | Model / quant | Published result | Important qualification |
|---|---|---|---|
| GORAST: three Vega 64 8 GB, Xeon E5-2680 v4 | Swift 1.5 IQ2_XS | 69.27 PP / 17.98 TG, real 7501 tokens, TG512 | Prefill 512; split 21,35; one x16 path and two x1 paths; decode cache hit 89.6% |
| Two MI50 16 GB, Xeon E5-2666 v3 | Coder IQ1_M | 321 PP / 50.1 TG at 4096 tokens; 523 PP / 47.8 TG at 32768 | Both links x16 Gen3, measured 13.8 GB/s; prefill 4096; all experts in VRAM and 100% decode cache hits; synthetic code requests, TG256 |
| RTX 5070 12 GB, Ryzen 5 7600 | Swift 1.5 IQ2_XS | 811 PP / 73.8 TG at 4K, engine 0.1.26 | Normal automatic prefill; modern GPU, DDR5; one code-agent prompt per length, TG256 |
| RTX 5090 32 GB, Core Ultra 9 285K | Original Flash-Next IQ2_XS | 4269.8 PP / 179.4 TG at 4096 tokens | Measured transfer 50 GB/s; 23.44 GiB expert cache; synthetic prompts, no reused tokens, TG256 |
| Two RTX 3090 24 GB, EPYC 7453 VM | Coder IQ1_M | 98.6 TG at 32K with automatic layer split | 48 GB total VRAM; different harness; the report expressly limits cross-upstream comparisons |
| RDNA4 gfx1201 ROCm contributor setup | Swift IQ2_XS | Reported 80 PP on a 39-token chunk versus 944 PP for a 2047-token chunk | Contributor development baseline; short/long prompt comparison, not a paired GORAST measurement |

Sources: [MI50 report](https://github.com/Niko1221/Strata/blob/main/bench/results/2026-10-03-community-2x-mi50/README.md), [measured upstream tables](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md), [RTX 5090 report](https://github.com/Niko1221/Strata/blob/main/bench/results/2026-09-30-community-rtx-5090/README.md), [RTX 3090 report](https://github.com/Niko1221/Strata/blob/main/bench/results/2026-09-29-rtx3090-epyc-milan/README.md), [ROCm development notes](https://github.com/Maxritz/Strata-rocm/blob/main/docs/ROCM_PORTING.md).

The large published numbers are real measurements, although some other GPU tables are explicitly estimates. The comparison above separates measurements from different prompt/model/hardware conditions. None is an equivalent reference for the exact GORAST request.

## What the accepted source and live host confirm

1. **Chunk sizing has not been tuned.** `generate.cpp:431–437` states that auto prefill selects the largest chunk, normally up to 8192, whose buffers the expert cache can lend. Its stated benefit is streaming each routed expert once per chunk and reducing bytes per token. The current run used 512 and borrowed only 287/275/268 cache slots across the three GPUs. This is a credible major performance lever; the gain remains unmeasured.
2. **The x1 bottlenecks are genuine, despite endpoint status.** Every GPU endpoint reports x16 Gen3. Walking the complete sysfs path finds PCI 0a and root port 00:1c.4 at x1 Gen2 on the 0c path, and PCI 0d and root port 00:1c.5 at x1 Gen2 on the 0f path. PCI 05's entire path is x16 Gen3. These upstream bottlenecks agree with the engine's 12.2 / 0.4 / 0.4 GB/s probes. Bare endpoint status would give a wrong diagnosis.
3. **The build is Release and uses the separate legacy HIP backend.** `STRATA_HIP_GFX906=ON`, gfx900 target, ROCm 6.3.4, and `STRATA_PORTABLE=OFF`. In this backend CMake deliberately keeps `STRATA_ENABLE_CUDA=ON` and `STRATA_ENABLE_HIP=OFF`; that combination is not evidence of accidentally running CUDA or omitting HIP. The accepted executable is unchanged.
4. **Host pinning needs inspection, not an assumed speedup.** The SSH environment has no STRATA/HSA/HIP/ROCR tuning overrides and a memlock limit of 10,272,620 KiB, about 9.8 GiB. The MI50 report used unlimited memlock and `STRATA_ARENA_MMAP=1`. The accepted source supports that flag on legacy HIP/Linux and describes locking mapped expert spans to avoid ROCclr staging stalls. Its relevant code paths must be shown to engage; current logs do not prove a pinning failure. Adaptation is disabled in the current configuration, which also affects applicability of swap-related optimizations.
5. **Correct profiling exists.** `STRATA_PREFILL_TIMING` in `src/prefill/prefill.cpp` reports phases including host grouping, copy waits, dequantization, gate/up/down GEMMs, attention, and PLE. `STRATA_SPLIT_TIMING` in `generate.cpp:7427` is for verification-stage host time; it is not a substitute for prefill profiling. GPU busy averages alone cannot establish the PP bottleneck.
6. **Exactness mode is not a demonstrated 3× slowdown.** `STRATA_IQ_MT_MIN=1` uses the multi-token CPU path even for singleton expert groups. The source describes a small measured decode cost on another quant/CPU, not a comparable Vega PP measurement. Removing it before investigating output differences would weaken correctness control.

## Tuning applicability

| Control | Audit conclusion |
|---|---|
| Prefill 512 → auto | Highest-priority same-model test; automatic cache borrowing plans the buffers. Verify the actual chosen chunk and all-card headroom. |
| `STRATA_ARENA_MMAP=1` and adequate process memlock | Supported source path; investigate copy/staging time first, then test independently if relevant. This is distinct from the resident-experts flag. |
| CPU expert pool | Current 15 workers exceed the 14 physical cores before other engine threads. Profiling may justify a 13-worker comparison; oversubscription is a hypothesis, not a proven cause. |
| `--pcie-frac` | Current probes select 0.33 / 0.01 / 0.01. An explicit fraction applies to every GPU and bypasses probes; a value copied from an x16-only setup can harm the x1 stages. |
| Fused tensor-core prefill | The inspected CUDA fused path requires compute capability >=80. Its advertised gains cannot be assumed for gfx900. Modern HIP acceleration reports require their own supported backend/architecture. |
| `STRATA_PREFILL_MMQ` | Currently off. CMake requires the separate standard HIP backend and rejects it with the legacy configuration. It is not a safe flag to flip on this build. |
| RDNA4 WMMA | gfx12-specific; unavailable as a gfx900 tuning switch. |
| KV / reserve / stage trimming | Already int8 / 700 MiB / enabled. The full run retained about 450 MiB minimum headroom. Reducing the reserve blindly is not the first lever. |
| MTP spec | Current spec2 primarily affects generation. Changing it cannot establish a 200–300 PP path. |
| `--resident-experts` with layer split | The saved warm log expressly says it does not enable resident mode here. That historical run is an OS-cache baseline. |
| Tensor splitting | Outside the accepted implementation; not established as necessary or faster. Frequent communication across the confirmed x1 upstream paths is a concern. |
| Quant format | A separate research branch. Upstream's measured 4K RTX 5070 table gives Q2_0 1007 PP versus IQ2_XS 811 PP. The ROCm notes cite a larger model-card advantage, but that original table was not independently verified here. Swift Q2_0 availability/packing and answer quality must be established before changing the model quant. |

Architecture and implementation references: [HIP documentation](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md), [multi-GPU documentation](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md), [ROCm development notes](https://github.com/Maxritz/Strata-rocm/blob/main/docs/ROCM_PORTING.md). Source checks used the exported accepted tree with provenance commit a1641e9f77aacad4d201b53c8a7ae8fa21059ebb.

## Sequential next steps

1. Hold model, split 21,35, context, KV, spec, safety settings, and prompt fixed. Capture one controlled `STRATA_PREFILL_TIMING=1` diagnostic; inspect CPU grouping, copy waits, dequantization, GEMMs, and PLE. Keep profiling results separate from ordinary throughput rankings.
2. Change only prefill 512 to auto. Use the same real 2048-token screening prompt and TG128. Record the selected chunk, PP/TG/MTP, output, faults, and peak memory. If it wins with acceptable correctness and headroom, run the exact original 7501 tokens and TG512.
3. If the diagnosis shows copy or host-pinning stalls, examine process memlock and mmap pinning engagement, then run one isolated pinning comparison. If CPU expert work dominates, examine workers and quantization instead. No blind sweep.
4. Establish the normal engine's reference on the identical workload before claiming Strata is a replacement. Keep the Ornith service stopped; any separate reference benchmark must be explicitly identified and controlled.
5. Preserve the unresolved screening greedy-token divergence. Coherent output, successful completion, and no GPU faults do not establish numerical equivalence. Larger chunks can also change rounding, so correctness must accompany the speed result.

**Decision:** 69 PP is a valid measurement of the tested configuration, not a demonstrated ceiling. There are verified, previously untested execution-path controls that warrant investigation. A 200–300 PP result on GORAST is neither established nor ruled out by the existing evidence.

## Scope

The audit covered upstream community reports, bundled speed/prefill/cache/layer/CPU-GPU benchmark documentation, NVIDIA and legacy/modern AMD implementation notes, and relevant ROCm/dual-GPU contributor setups. It does not claim to exhaust every fork, unpublished test, or hardware combination. Advertised estimates and unverified third-party benchmark citations were not treated as measured GORAST results.

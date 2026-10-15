# GORAST Q4 CPU-expert pipeline: forensic repair design + gated Grok test suite
**Author:** ChatGPT supervisor research, 2026-10-15  
**Status:** DESIGN ONLY; NO TESTS OR REMOTE COMMANDS AUTHORIZED OR SENT. Grok's work is PAUSED at owner's request.  
**Scope:** Ornith 1.5 Q4_K/IQ4_XS with expert tensors from blocks 5,8,20,35 on CPU, three Vega64 Vulkan GPUs, CPU+GPU concurrent, correct long-context output. Do not conflate all-GPU Q4 scores with this target.

## 0. Read first; immutable controls
Supervisor: `docs/SUPERVISOR_RULES.md`. Grok's complete telemetry:
- `outbox/result_grok-q4-cpu-offload-stall-20261015.txt`
- `outbox/result_grok-q4-cpu-offload-attempts-telemetry-20261015.txt`

**Absolutely no benchmark execution, compilation, SSH/session actions, service restarts, changing clocks/undervolt, GPU resets, memory policy, or GORAST modification until Gorast explicitly says to resume.** This is a design for Grok, not a work dispatch. Do not invoke any workers merely because this file exists.

Production must not be touched: /opt/llama/llama-c1-v3mx2-20261013, /etc/ornith/threecard.cmd, ornith-server inactive by operator choice; do not copy experimental binaries over production. Storj files/services off limits. All candidate builds in a separate named worktree, isolated port 8205, manifest/source+lib SHA before/after, explicit rollback. Preserve clocks/voltage/thermal firmware settings. Passive thermal telemetry; **NO user-unapproved 110C software cutoff and NO arbitrary thermal kill limit**; nonthermal faults/OOM/VRAM guards remain.

### Exact target/control inputs
Host Ubuntu GORAST Xeon E5-2680v4 14C/28T AVX2, 80GB RAM; Vulkan0 PCI05 x16, Vulkan1 PCI0c Gen2 x1, Vulkan2 PCI0f Gen2 x1 (with MESA_VK_DEVICE_SELECT UNSET; never infer Vulkan indices from sysfs cardN names). All Vega64 8176MiB.
GGUF `/opt/ornith-models/Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf`.
Model and flags **identical for primary A/B**:
```text
-dev Vulkan0,Vulkan1,Vulkan2 -sm layer -ts 0.8,1,1 -ngl 99
-ot 'blk.(5|8|20|35).ffn_(up|down|gate)_exps.weight=CPU'
-c 131072 -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --no-op-offload
-t 14 -tb 14 -fa on --load-mode dio -np 1 -lv 4 --temp 0.1
--jinja --no-webui --cache-ram 0
```
Use source `/home/ubuntu/claude-ornith-logs/q32_512.json`: 31619 actual prompt tokens, cache_prompt=false, n_predict=512, preserve same sampling/prompt IDs. Pipeline experimental env:
```text
ORNITH_PIPELINE_OT=1 ORNITH_CPU_WORKER=1
GGML_VK_XDEV_ASYNC=1 GGML_VK_ASYNC_READBACK=1 TURBOPREFILL=0
```
Binary previously `/home/ubuntu/c1-xdev/build/bin/llama-server`, dirty tree 10bf611e5. Runner `/tmp/grok-q4-async.py`; use **copy** of it in isolated worktree, not edit old proof. `MESA_VK_DEVICE_SELECT` remains unset.
Reference CPU experts, pipe OFF 365.368 PP /25.609 TG at 31619/512, `-b2048 -ub2048` (not a strictly matched -ub512 control). Historical b10919 426.490 PP/26.644 TG (other binary). Experimental pipe ON `-ub512`: 270.85 /22.96, 270.24 /22.96, 269.77/23.00, 269.79/22.95, 267.75/22.96; outputs coherent short excerpt, not independently oracle verified. **Do not compare as if all baselines have identical UB/binary.**
Other rejected configs: op offload ON 14GB+ GPU allocation (OOM); `--no-op-offload` UB2048 four copies exceeds V1 (9731MB); UB1024 four-copy Q8 configuration missed fit by ~90MiB on V1; even split + Q4 KV yielded 292.827/25.105 without overlap; GPU-all-Q4 q4_0 KV -b32768 -ub1024 reached 906.693/27.592 at 32K and is NOT this CPU target.
Build29 headroom after run: V0 free 2045MiB; V1 **559MiB**; V2 865MiB. Before any change, reserve/measure all four copy slots and inspect allocations. No draft.

## 1. Evidence-based bottleneck and **ambiguity**
- In build29, 31619 prefill=118.089s; graph_compute accumulated 94.519s; host graph-build 0.487s. Three `CPU MUL_MAT_ID` input-copy/sync waits are **26.121, 29.910, 34.865s** = 90.896s (~77% of prompt wall, 96% of measured graph_compute by that checkpoint).
- Frame stack: `ggml_cpu_worker_sync -> libggml-vulkan -> libggml-base -> llama_context::graph_compute`. Those three CPU splits report copy_ms=whole wait, graph_ms=0. GPU graph submits ~1ms after warmup.
- GPU busy 15s snapshots mostly one GPU (occasionally two); zero clean all-3-at-least-20%+CPU overlap. CPU host utilization ~8-12% total, not necessarily % of worker cores. The ring expansion from 32 to 96 active slots (256 cap, 768MB budget) removed input-ring host waits but **PP stayed ~268**.
- `input ring flush miss`: attempted early submit was not the command buffer owning the ring completion signal; ring slots still owned by the GPU graph completion.
- **Ambiguity that MUST be resolved before patch:** Does `ggml_cpu_worker_sync` spend 90s waiting for (A) legitimately executing CPU MoE compute, (B) GPU->CPU activation / expert-ID readiness, (C) stalled worker queue or over-synchronization, or a combination? A wait stack proves the calling thread blocks; it DOES NOT prove CPU compute is free or that deletion of the wait is safe. Three giant waits could be accumulated queued work, not one 30s individual expert matmul. CPU 8-12% aggregate alone is not enough to identify which.
- Existing build30 (ring signal / histogram) was compiled to 100% AFTER operator paused, **never executed**. Do not call it a fix. Prefer source inspection + tiny proof before any full prompt.

## 2. Relevant upstream research — reference, not blind cherry-pick
- https://github.com/ggml-org/llama.cpp/pull/29963 (open, upstream pipeline parallelism with MoE host weights; head at review 6325cf42174a75887c98bae8bd387f9e8aa723ce)
- https://github.com/ggml-org/llama.cpp/discussions/29964
- https://github.com/ggml-org/llama.cpp/pull/11397 (origin of override/pipeline interaction)
- https://github.com/ggml-org/llama.cpp/issues/20757 (alternative GPU expert caching; NOT near-term CPU worker fix)
- https://github.com/ggml-org/llama.cpp/discussions/10182 (scheduler copy and reserve semantics)

PR #29963 uses several separable mechanisms:
  1. Preupload **user inputs once per device** with per-copy input completion events, instead of duplicating across 4 sched copies: major VRAM savings on high-context masks.
  2. Keep a **constant worst-case reserved graph topology** in pipeline, to prevent graph realloc+global sync at ubatch boundaries.
  3. **Single reuseable per-device host-weight copy**, ordered on the split stream; pin/immutable lifetime rules apply.
  4. Route host-weight ops to the GPU owning each layer and avoid synchronously reading MoE IDs during in-flight prefill; for sufficiently large batches use all expert rows instead of selective ID reads.
  5. Optional balanced fit layout (NOT appropriate as a first experiment on x16/x1/x1).

**Critical applicability boundary:** Upstream PR computes GPU-offloaded experts from host RAM. Grok explicitly sets `--no-op-offload`, executing `MUL_MAT_ID` on CPU. The upstream speedups on 3x fast PCIe5x16 Nvidia GPUs are **not transferable** to Vega x1 risers; upstream comment reports asymmetric x16/x4 regression 1100 ->600 PP, even heavily skewed splits did not beat serial. Use upstream as a design reference / individual cherry-picked concept in a NEW branch after diff review. Do not cherry-pick all files or enable blanket expert uploads to x1 cards. `ggml_cpu_worker_sync` appears custom to Grok's fork, not an upstream function; first inspect its actual definition/callers.

## 3. Ranked repair hypotheses (one change per branch/test)
### H0 — forensic timestamps and CPU work accounting (mandatory first)
Before code surgery, instrument **existing CPU worker** with monotonic timestamps and a unique `req/ubatch/split/slot/generation` token:
- ENQUEUE CPU graph and its immutable input slots; GPU producer fence/ready; worker dequeue/start; CPU matmul enter/leave; worker publish; scheduler wait enter/leave; GPU consumer submit; release/reuse slot; queued CPU jobs and active CPU jobs.
- For each wait: waited object, source backend, producer fence/event ID, whether producer complete, CPU worker state and CPU `thread CPU time` while wall time grows; use per-thread CPU usage (`pidstat -t` / `/proc/PID/task` / optional nonprivileged perf) as well as host total.
- Classify `ggml_cpu_worker_sync`: CPU_EXEC_BUSY vs WAIT_FOR_GPU_INPUT vs WAIT_FOR_CPU_RESULT vs WAIT_FOR_SLOT vs DRAIN_ALL_JOBS / global barrier. Publish aggregate wall **exclusive** critical-path buckets (avoid adding overlapping wall times). Identify whether the three 26/30/35s records are individual waits or one backlog drain at boundaries.
- Inspect current `ggml_backend_sched_compute_splits`, `ggml_cpu_worker_sync`, `ORNITH_CPU_WORKER`, `MUL_MAT_ID` split, `sched_copy_experts`, `get_tensor_async`, `event_record/wait`, ring semaphore ownership. Save source-line+hash evidence. **No code fix until causal classification.**

### H1 — CPU worker correctly asynchronous with explicit dependencies
If the wait is a queue-drain/global barrier that can be deferred, implement a bounded 3–4-slot CPU-work queue (start with existing four scheduler copies) and **per-ubatch producer / CPU-complete / consumer-done state**, with immutable graph metadata and data for each in-flight slot:
```text
GPU_PRODUCER_SUBMITTED -> GPU_INPUT_READY -> CPU_QUEUED ->
CPU_RUNNING -> CPU_RESULT_READY -> GPU_CONSUMER_SUBMITTED ->
GPU_CONSUMER_DONE -> SLOT_FREE
```
- Scheduler may submit an **independent later ubatch** on V0 while CPU works on an earlier ubatch, but must NOT submit a GPU consumer using unready CPU result or overwrite a producer's still-in-flight buffers.
- The CPU worker receives its own stable tensor pointers and ID/activation data. Release input slot only after CPU has finished reading; release output slot only after consuming GPU has finished. Record per-slot generation to reject ABA/reuse races.
- Limit queue depth and keep a clean blocking fallback for small batches/errors. CPU-side condition variables/events plus actual GPU completion fences; **do not put GPU-side waits on host-signaled VkEvent** (the previous unsafe v4 hung GPU05). Avoid cyclic waits and enforce timeout -> cancel/error (not GPU reset/reboot).
- Crucial: returning from `CPU_GRAPH_COMPUTE` asynchronously alone is insufficient if the *next dependent GPU split* blocks the single dispatcher. Ensure the host can dispatch an independent UB or backend while CPU job is pending, e.g. a ready-queue scheduler. A future/worker_thread without a runnable-graph queue is not a performance fix.
- Gate with producer/consumer event timestamp timeline proving no read-before-write, no hidden global sync, at least two UBs alive concurrently; preserve 4-slot allocation headroom.

### H2 — remove unintended synchronous MoE IDs / graph reservation resets
If forensic trace shows GPU->CPU readback/synchronize is forcing the queue drain, inspect selective expert-ID path and the PR #29963 alternative (upload-all) **but do not globally upload all experts** over x1 links. In this `--no-op-offload` mode, CPU MoE IDs may be required for CPU execution; support asynchronous GPU D2H into registered/pinned per-source buffers, with completion before CPU reads IDs, per-UB immutable storage. Validate GPU input ready vs CPU worker wait.
Test independently whether `graph_reserve` / `alloc_splits` / checkpoint creates a global sync or realloc for non-full graph. Enable one-copy-per-device user-input preupload only with correct events; capture compute allocation changes and *prove* it reduces waits, not just memory.
If CPU worker spends nearly all measured time in real expert kernel execution, H1 merely moves the wait to consumer; prioritize H3 instead.

### H3 — speed up actual CPU work / placement when H1 cannot
Measure CPU `MUL_MAT_ID` op time with inputs already resident and complete; per-thread utilization, effective memory bandwidth, page faults, frequency and AVX2 kernels, pinning/NUMA placement **without changing clocks/hardware**. Sweep only reproducible CPU thread count 8/14 (perhaps 20 if evidence justifies), one setting at a time, one exact prompt, matched control. Check avoid threadpool oversubscription and that the 2.18GiB host expert weights are resident and not faulting. Do not assume more threads help.
Alternate **separate, explicitly labeled product configuration**: relocate ONE CPU expert block 5 or 8 to Vulkan0/PCI05 (free ~2045MiB), keeping other CPU blocks. This is NOT success on the 4-CPU-bank assignment and changes CPU/GPU placement; requires separate owner approval, measured fit, and new correctness gate. V1 559MiB free prohibits casually shifting large banks. Similarly q4 KV / c40960 / UB1024 are diagnostic memory tradeoffs, not comparable to 131K Q8 requirement.
If actual CPU matmul floor is dominant, consider staying with pipe OFF for four CPU blocks (365/25.61), or eventually full-GPU no draft as independent option; do NOT claim artificial pipeline win.

### H4 — submit Vulkan ring copy separately (only if critical after H0)
Existing build30 forces ring signal on the last submission. Only test if source review proves the last submission actually owns the copy and the event chain is monotonic. Instrument `record -> submit -> fence signal -> slot reuse`; if a separate submit is valid, it must preserve transfer-write-to-shader-read barrier, queue-family rules, and immutable source. No unsignaled waits, no deletion of synchronization. Earlier ring expansion removed waits with zero gain, so this ranks behind CPU worker root cause.

## 4. Owner-gated test-suite (no launch until APPROVED)
**Rule:** one source change per arm, one unchanged comparator per branch, cache_prompt=false, real prompt IDs, save full stdout/stderr/telemetry/command/environment/sha/host timestamps/HTTP result. Keep an isolated port, a process-time cap and memory/OOM/fault watcher. Do not introduce an owner-unapproved 110C or other software hotspot cutoff; do preserve existing GPU driver protections. Preserve no-draft, -ngl99, all four CPU expert banks for the PRIMARY line.

| Stage | Workload | Change | Required result to proceed |
|---|---|---|---|
| 0 | **Read-only**, saved build29 and build30 sources/logs | Diff build29 vs30, map 26/30/35s call sites + GPU and CPU worker states; SHA, renderer and device map | Causal hypothesis with explicit stack/event trace; no experiment yet |
| 1 | B2048/UB512, 2048 genuine prefix, n_predict32 diagnostic | Baseline instrument-only; compare to byte-identical build with tracing disabled; preserve original model/flags | Matched PP within ~5% (tracing overhead). CPU worker begin/end and wait attribution. Nonnegative VRAM; 0 GPU errors |
| 2 | B2048/UB512, 8196 genuine-prefix tokens, n_predict32 diagnostic | One H1 or H2 minimal surgical patch per build; matched OFF/ON, same -ub and same CPU banks | No deadlock, all 3 GPUs demonstrate actual overlap, CPU worker not globally waiting at checkpoint; coherent/finite output; reason if no improvement |
| 3 | Same real 8196 prefix, n_predict512 | Stronger correctness: token/logit comparisons against matched serial for deterministic temp0 and fixed seed; exact oracle task for non-identical numerics | TG512 completed, no repeated junk or NaN, 20/20 ledger/oracle or equivalent |
| 4 | 31619 real prompt, n_predict512 | Matched pipeline-OFF vs candidate-ON, SAME binary family, -b2048 -ub512, Q8 KV, 4 CPU banks; repeat candidate at least twice only if first run passes | PP/TG result reproducible, same prompt IDs; target **PP >= 426.49**, **TG >= 26.2**, overlap and correctness, positive VRAM margin, 0 driver faults |
| 5 | 62410 and 88407 genuine prompts, TG512, then 131K capacity | **Only if** 32K repeat stable and user authorizes long-context work | Correctness, VRAM and speed slope; abort on real faults/OOM; preserve rollback |
If tracing proves H1/H2 impossible or no benefit at short stage, STOP and report, do NOT burn 32K tests. Stage 3 correctness should include a ledger task beyond merely a coherent `<think>` prelude. For CPU-vs-GPU placement, identical token arrays may not be expected due FP order; oracle exact result plus fixed-prefix logits/tokens and finiteness are separate checks. Gate signal: run with 500ms–1s common-clock GPU telemetry, and record per-device event start/end. At minimum user's criterion one clean prefill sample all 3 GPUs >=20% **AND** CPU >=40% by the runner's defined denominator; show real timeline overlap, not utilization alone. Existing 15-second snapshots are too sparse.

**Baseline fairness:** historical pipe-OFF 365.37 at -ub2048 vs ON ~270 at -ub512 differ in microbatch. First obtain a genuine pipeline-OFF -ub512 control using the same build and same request, so timing delta may be attributable to code. The historical 426.49 is a user performance goal / older reference, NOT the matched counterfactual. Do not claim a new speedup against unmatched variants.

**Diagnostics:** for each split/ubatch log ID, CPU-work-ready time, CPU-op actual execution, dispatcher thread blocked time, input copies vs GPU wait, queue and slot occupancy, CPU system vs per-worker % and runqueue, backend completion-fence age, GPU busy and kernel event intervals, VRAM (real min free by BDF), PCIe H2D/D2H bytes, host RAM, temperature/junction/power unchanged, CPU/GPU faults from accessible logs, last successful output token and stop reason. Correlate clocks; aggregates must not double-count overlapping time. A wrong/garbled generation is a FAIL whatever PP.

### Immediate STOP conditions
- Wrong tokens/garbage/nonfinite logits, unexpected 0-token hang, GPU job timeout/reset/VM fault/OOM, negative projected headroom, memory continually rising, broken CPU/GPU fence, missing required per-run provenance, any production process involvement.
- *No arbitrary hotspot temperature threshold.* Log temperature and leave hardware/driver safety intact; stop only for actual driver failures/throttle anomalies with explicit evidence, or by owner.
- No GPU reset, reboot, PSU/hardware action, changes to clocks/power/fans, or production restart without explicit owner approval.

## 5. Suggested experiments in priority order
1. **READ ONLY H0**: source audit actual worker state/async CPU backend, event ownership, loop, 26/30/35s, gather no new scores. Build30 untested.
2. **A trace-only short run (if owner resumes)** plus exact same-build pipeline-OFF UB512 control. Validate whether CPU needs 90s to compute or is just waiting.
3. If wait is unnecessary queue-drain, implement H1 *per-UB stable slots + dependency-aware ready queue*. This is a sizable backend project, not a one-line barrier removal.
4. If id readback or reserve churn dominates, H2 isolated tweaks drawing from #29963 (Vulkan-aware, no GPU0-only assumptions).
5. If real CPU work dominates, H3 CPU AVX2/thread/host locality, not more staging slots. Consider alternate placement only as explicitly different workload.
6. Evaluate H4 build30 only after CPU root cause and ring ownership proof.
7. No production promotion before 32K TG512 repeat plus longer-context stability and owner authorization.

## 6. Grok's requested return format (after authorization)
```text
status: PAUSED | DIAGNOSTIC_ONLY | FAIL | PASS_CANDIDATE | OWNER_DECISION_REQUIRED
causal_branch: CPU_COMPUTE | GPU_INPUT | WORKER_QUEUE | BACKEND_EVENT | RESERVE | MIXED | UNKNOWN
binary + ggml library SHA / source changes: ...
exact command + env + prompt token hash + ctx/ub/b: ...
arm / PP / TG512 / prompt_ms / predicted_ms / generated_n / stop_reason:
trace table: CPU actual execute vs wait, CPU worker queue depths, CPU and GPU timeline:
overlap: all3 >=20 + CPU >=40 common timestamp and event-based proof (or NONE):
VRAM free minima by 05/0c/0f + host memory:
correctness: oracle, first diverging token/logit, full completion, finite:
faults: sysfs/driver process and complete reason:
next bounded action / owner decision needed: ...
```

## 7. Open research questions and caution
- `ggml_cpu_worker_sync` is custom; upstream #29963 cannot answer its implementation. Source inspection is essential.
- Upstream op-offload patch intentionally sends host expert weights to GPUs and may be **catastrophically slower over GORAST's x1 risers**; never enable wholesale GPU expert streaming as a generic fix.
- CPU work may be fundamentally serialized within each layer; overlap only spans *different ubatches* when their slots/data are distinct and GPU queue accepts independent submissions.
- 26/30/35s waiting may be accumulated CPU work rather than one per-ubatch matmul; classification by per-thread CPU time and queued work is required.
- The current best FOUR-CPU-BANK experiment (~268 PP/23 TG) misses the owner threshold and must not be promoted. All-GPU q4_0 KV results are different tests.

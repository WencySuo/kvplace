# GH200 follow-up experiments (started 2026-11-14 evening)

Box: Lambda GH200 480GB (`ssh ubuntu@192.222.51.59`), Ellie's code at `~/cs2430_project`
(her `main` d968c90, pushed over by rsync because her repo is private). Main rerun of her
L40S midterm sweep: `tmux attach -t run`, status in `~/cs2430_project/logs/job_status.txt`,
watchdog log on the Mac at `kvplace/logs/gh200_watch.log`, results pulled to
`kvplace/results/gh200/`. Follow-up code is on branch `gh200-followups` of
`~/code/cs2430_project` (Mac) and staged on the box in `~/followups_staging/`.

## What the first GH200 numbers say (pilot + linkbench, before the follow-ups)

| | L40S (Ellie, PCIe4) | GH200 (this run) |
|---|---|---|
| pinned H2D, 16 MiB-1 GiB copies | 13.5 GB/s | 340-367 GB/s (D2H 275-295) |
| per-layer KV fetch at 16k (64 MiB) | 4.95 ms | ~0.19 ms |
| `hbm` ordinary decode, 16k, ms/token | 34.6 | **71** (?) |
| `hbm` ordinary decode, ~100-token context | 34.2 | **39** |
| `ours` R=0 (everything offloaded), 16k, ordinary / speculative | 179 / 98 | **47 / 32** |
| GPU utilization during the run | n/a | ~30% |

Two things are wrong or at least unexplained, and they change what the follow-ups must do first:

1. **The step is CPU-bound, on both machines.** On the L40S the all-HBM step is 34.5 ms at 100
   tokens and at 16k tokens alike, which cannot be GPU time (weights alone are ~19 ms there, KV
   adds ~3 ms at 16k). On the GH200 the GPU is 4x faster yet the step got slower, and the GPU is
   busy 30% of the time. Eager transformers + DFlash's Python loop is launch-bound. Our pipeline
   model (fetch hidden behind per-layer *GPU* compute) is therefore being validated against wall
   time that is mostly CPU. Fetches are "hidden" behind Python, not behind compute.
2. **The `hbm` baseline is pathological on the GH200** (71 ms at 16k vs 47 ms for the fully
   offloaded arm). Only `hbm` uses a stock `DynamicCache` (torch.cat growth every step); every
   other arm uses preallocated buffers. Suspect allocator churn; the profile will say.
3. `hf_offload` (both block sizes) and `ondemand` (block 1) produced different tokens than
   `hbm` on this box; on the L40S all 72 offloaded runs matched. `ours` matched. Not our arm,
   but worth one sentence in the report.

## The follow-ups (serial, bounded; `bash followups.sh` on the box, ~1.5-2 GPU hours)

| # | stage | question | GPU time |
|---|---|---|---|
| F0 | `hostmem`, `smread` | Can SMs read mapped pinned host memory in place, and at what bandwidth vs the copy engine (DirectKV's premise)? Also: SDPA with K/V *in host memory* vs in HBM, q=1 and q=16, 8k/16k/32k, the exact op the zero-copy arm runs. | 3 min |
| F1 | `profile` | torch.profiler of 10 verify steps for `hbm`, `ours` R=36 (all resident, preallocated), `ours` R=0, `zerocopy` R=0, block 1 and 16: GPU-busy fraction, launches/step, cudaMalloc/Free per step, top kernels. Explains (1) and (2) and gives the GPU-side step time the model should use. | 5 min |
| F2 | `zc16`, `zc8` | **DirectKV-style zero-copy arm** vs Ellie's copy pipeline vs HBM, R in {0, 18}, block 1/16, 4 prompts. The simulator on the zero-copy branch predicted "ties copy for speculative, loses for ordinary"; this measures it. | 30 min |
| F3 | `layers` | `measure_layers.py` on the GH200: per-layer attention vs rest-of-layer GPU time (CUDA events, so CPU overhead is excluded). Feeds `simulate.py` with GH200 numbers and the real link bandwidth. | 15 min |
| F4 | `evict` | **Eviction policy v1** = (which layers stay resident, how many recent tokens per layer stay in HBM). Runs: tail window 2048 and 8192 at R=0; R=18 resident-*last* vs resident-first (from F2). Metric: ms/token vs HBM KV footprint, copy vs zero-copy. | 25 min |
| F5 | `zc32` | 32k context (SPEED-Bench `throughput_32k`): hbm / ours R=0 / zerocopy R=0, block 1/16, 3 prompts. Attention time per layer doubles, the fetch per layer doubles; does the ratio hold and does zero-copy start to win? | 15 min |
| | `summarize` | `results/followups_summary.md` | |

Datasets: SPEED-Bench throughput 8k/16k/32k (what Ellie used; tau 1.6-2.4 at 16k, DFlash is
*slower* than ordinary decoding there on HBM), gsm8k as the high-acceptance control (tau ~5, in
the main job's `repro` stage).

## What is deliberately not run

- Multi-request / batched serving: Ellie's cache is batch-1 by construction. The capacity
  argument is made from the measured per-request footprints (`kv_gpu_bytes`, staging bytes)
  instead of a serving benchmark.
- CUDA graphs / torch.compile to remove the CPU bound: a real fix but a multi-hour engineering
  change to DFlash's loop; F1 + F3 quantify the GPU-only step so the model can be re-fit without it.
- DirectKV's fused projection+attention TMA kernel (SM90a, CUTLASS, SGLang-bound): the
  transferable idea is the in-place read, which `zerocopy` implements with PyTorch SDPA.

## Results so far

### Main rerun (Ellie's sweep on the GH200, 2026-11-15 02:28-03:37 UTC; `kvplace/results/gh200/`)

ms per generated token, mean over 6 SPEED-Bench prompts (`sweep_*.jsonl`); step = ms/token x tau.

| config | 8k ordinary | 8k spec (tau 2.5) | 16k ordinary | 16k spec (tau 1.86) |
|---|---|---|---|---|
| `hbm` (DynamicCache) | 52.8 | 43.5 | 53.3 | 53.1 |
| `ours` R=36 (all resident, preallocated) | 40.0 | 23.2 | 39.7 | 30.4 |
| `ours` R=18 (half offloaded) | 43.4 | 24.9 | 42.9 | 30.7 |
| `ours` R=0 (all offloaded, 2.4 GB/step at 16k) | 60.3 | 26.1 | 60.6 | 31.4 |
| L40S, Ellie: `ours` R=0 | 179 | 98 | 179 | 98 |

- **Speculative verification hides the whole host read.** At 16k, offloading every layer's KV
  costs 57.1 vs 55.3 ms per verify step (+3%); at 8k, +13%. Ordinary decoding pays +53% at 16k.
  This is the project's thesis in one row, but see the caveat below.
- **The `hbm` baseline is broken on this box**: 13 ms/step slower than the same model with
  `ours` R=36. The decode-only profile shows identical GPU time for the two (10.1 vs 10.2 ms/step),
  so the gap is CPU-side Python in the stock `DynamicCache` path, not GPU work (both still pay the
  `torch.cat` cache growth: 1.3 ms/step ordinary, 8.6 ms/step speculative at 16k). Use `ours` R=36
  as the all-HBM reference, not `hbm`.
- **Caveat: the step is CPU-bound.** `ours` R=36 ordinary is 40 ms/token at 8k and 16k alike,
  and 42 ms on gsm8k at 100 tokens; GPU-side work is a fraction of that. So "hidden" means hidden
  behind Python dispatch as much as behind GPU compute. Ellie's per-layer pipeline model mis-predicts
  the GH200 by 44% mean error (it fit the L40S to 1.3%). The GPU-only step from the layer timing
  stage is what a GH200 model needs.
- The ablation (slots 1/2/3, phase prefetch on/off) is flat within 1.4 ms/token at 16k spec: with
  0.19 ms fetches, lookahead barely matters.
- Block-1 runs of every offloaded arm mismatch `hbm` on ~half the prompts (block 16 matches 100%).
  Only the q=1 path hands SDPA a strided KV view (no mask -> fused GQA kernel); the kernel/split
  choice differs from the contiguous `DynamicCache` case and bf16 argmax flips. Not a data bug.
- Link: pinned H2D 350 GB/s, D2H ~290 GB/s; copies slow a concurrent GEMM by ~0.55x on this box.

### Follow-up F0: in-place host reads (`results/sm_read_NVIDIA_GH200_480GB.csv`)

| | GB/s |
|---|---|
| copy engine, pinned host -> HBM, 1 GiB | 350 |
| SM kernel reading mapped host memory, >= 264 blocks | **332** (95% of DMA); 78 at 16 blocks |
| same kernel from HBM | 3570 |
| SM writes into host memory | 298 |

DirectKV's premise holds: kernels can read host memory at link speed, no staging needed.
SDPA with K/V physically in host memory, Qwen3-8B geometry, 16k context:

| op | from HBM | from host | note |
|---|---|---|---|
| q=1, fused GQA kernel | 0.034 ms | 0.203 ms (331 GB/s) | reads KV exactly once |
| q=16, fused GQA, no mask | 0.199 ms | 0.648 ms (104 GB/s) | kernel re-reads KV ~3x across the query/head tiles |
| q=16, transformers path (repeat_kv + mask) | 0.710 ms | 1.346 ms | materialized 4x GQA expansion; 3.5x slower than fused even from HBM |

Two consequences. (a) A naive zero-copy arm pays +0.64 ms/layer at block 16 (the path transformers
actually takes), i.e. ~+23 ms per verify step at 16k, unless a KV-stationary kernel is used; this is
precisely the "CPU-memory-aware tiling" argument in the DirectKV paper. (b) The copy pipeline is also
leaving 0.5 ms/layer on the table at block 16 because of the repeat_kv expansion; a mask-free fused
GQA call would cut the verify step's attention time 3.5x on HBM.

### Follow-ups F2-F5 (all ran 03:45-04:30 UTC; `results/gh200/results/{zc_8k,zc_16k,zc_32k,evict_16k,layer_times_*}.jsonl`, tables in `followups_summary.md`)

Same 4 SPEED-Bench prompts per context (3 at 32k), 256 new tokens. ms per verify **step**
(ms/token x tau) so ordinary and speculative are comparable; `hbm` is the broken DynamicCache
baseline and is listed only for reference.

| context | mode | copy pipeline R=0 | **zero-copy R=0** | copy R=18 | zero-copy R=18 | hbm |
|---|---|---|---|---|---|---|
| 8k | ordinary | 70.3 | **43.8** | 45.7 | 43.0 | 62.1 |
| 8k | speculative (tau 2.0) | 60.3 | **55.2** | 57.2 | 54.5 | 107 |
| 16k | ordinary | 68.1 | **45.0** | 47.1 | 44.7 | 59.6 |
| 16k | speculative (tau 1.9) | **62.5** | 67.8 | 59.2 | 63.1 | 103 |
| 32k | ordinary | 70.6 | **39.0** | - | - | 63.2 |
| 32k | speculative (tau 1.7) | **79.3** | 112 | - | - | 144 |

**F2, zero-copy (DirectKV-style) vs the copy pipeline.**
- For *ordinary* decoding zero-copy wins everywhere, by 1.6x at 16k and 1.8x at 32k, and its
  step does not grow with context (39-45 ms at 8k/16k/32k). The in-kernel read is 0.2-0.4 ms per
  layer at 331 GB/s, 7-15 ms of GPU time per step, and it hides under the ~40 ms CPU-bound step.
  The copy pipeline's ordinary step is not link-bound either; it pays ~28 ms of exposed waits
  (85 memcpys + event waits per step) over the all-resident 40 ms.
- For *speculative* verification zero-copy ties at 8k, loses 8% at 16k and 41% at 32k. This is the
  kernel effect from F0: the fused-GQA kernel re-reads the 16-row block's KV ~3x from host, so at
  32k the verify attention alone is ~2.7 ms/layer from host vs 0.7 ms from HBM, and that is GPU
  time the CPU overhead cannot hide. A KV-stationary kernel (DirectKV's Q-inner loop) would remove
  the 3x and should put zero-copy at parity or better for speculative decoding too.
- Zero-copy keeps 18 MB of KV in HBM per request at any context (vs 155 MB staging for the copy
  pipeline at 16k, 289 MB at 32k, and 2.4-4.8 GB all-resident): the capacity story at its cleanest.
- **Prefill is the price**: time to first token 1.9 s vs 0.6 s at 16k and 6.8 s vs 1.5 s at 32k,
  because the prefix attention also reads host memory with heavy re-reads. Fix is mechanical:
  prefill through HBM, write the prefix to host once, switch to host views for decode.
- The simulator on the zero-copy branch predicted the opposite ("ties for speculative, loses for
  ordinary"). It assumed a GPU-bound step and a kernel that reads KV once; both assumptions are
  wrong here, in opposite directions.

**F4, eviction policy v1** (16k, copy pipeline unless stated; HBM KV = bytes resident + staging):

| policy | HBM KV per request | ordinary step | speculative step |
|---|---|---|---|
| everything on host, 64-token tail (R=0) | 155 MB | 68.1 | 62.5 |
| keep last 2048 tokens of every layer in HBM | 448 MB | 62.6 | **56.1** |
| keep last 8192 tokens of every layer in HBM | 1354 MB | 62.0 | 57.9 |
| keep first 18 layers resident (R=18) | 1352 MB | **47.1** | 59.2 |
| keep last 18 layers resident | 1352 MB | 61.6 | 57.8 |
| all resident (R=36, from the sweep) | 2400 MB | 39.7 | 55.3 |

- For speculative decoding a *token window* is the better eviction unit: 2048 recent tokens per
  layer (448 MB) beats half the layers resident (1352 MB) by 3 ms/step at a third of the memory.
  The layer choice (first vs last) is within noise for speculative decoding.
- For ordinary decoding the opposite: resident-*first* layers are worth 14 ms/step over
  resident-last, because the copy pipeline needs the slack of resident layers to prefetch; a token
  window barely helps (the exposed cost is per-layer waits, not bytes).
- For the zero-copy arm the tail window is a no-op by construction (the kernel reads the whole
  prefix from host; the tail is written there too), so a zero-copy eviction policy needs a split
  attention (host prefix + HBM tail, merged by log-sum-exp) to benefit from any HBM budget.

**F1, decode-only step profile** (torch.profiler, 16k, one prompt, 10 verify steps; prefill
subtracted; `results/gh200/results/profile_16k.jsonl`, traces in `profile_traces/`). Wall step =
clean ms/token x that run's own acceptance (1.36 tokens/step for this prompt; the sweep average is
1.9, which an earlier version of this table used and which overstated the speculative wall step):

| config | wall step (clean) | GPU kernel ms/step | GPU busy | of which: attention | GQA expand copies | cache cat | H2D memcpy (side stream) |
|---|---|---|---|---|---|---|---|
| ordinary, all resident (R=36) | 48 | 10.2 | 21% | 0.9 | - | 1.3 | 0 |
| ordinary, copy pipeline R=0 | 81 | 9.1 | 11% | 0.8 | - | 0 | 6.5 |
| ordinary, zero-copy R=0 | 50 | 14.8 | 30% | **6.7** (0.19 ms/layer, link-bound) | - | 0 | 0 |
| speculative, all resident (R=36) | 64 | 45.0 | 70% | 15.0 | 10.3 | 8.6 | 0 |
| speculative, copy pipeline R=0 | 72 | 39.4 | 55% | 15.0 | 12.1 | 1.4 | 6.5 |
| speculative, zero-copy R=0 | 67 | 56.4 | 84% | 14.9 | **29.3** (reads host at ~80 GB/s) | 1.4 | 0 |

- Ordinary decoding is 10 ms of GPU work inside a 48 ms step (weights GEMMs 4.6 ms). A GPU-bound
  engine (CUDA graphs) would make the step ~5x faster, and then zero-copy's +5.8 ms of in-kernel
  host reads would be a 1.6x slowdown rather than invisible. The copy pipeline's 6.5 ms of DMA
  overlaps, but its wall step is the worst because of per-layer event waits on the CPU side.
- Speculative verification is mostly GPU-bound already (45 of 64 ms, 70%). 19 of those 45 ms are
  avoidable: `repeat_kv` materializing the 4x GQA expansion (10.3) and DynamicCache's cat (8.6).
  With a mask-free fused-GQA attention call and a preallocated cache the verify step's GPU time
  would be ~26 ms.
- The zero-copy arm's extra cost at block 16 is the expansion copy reading host memory with a
  strided access pattern at ~80 GB/s (29 ms), not the attention kernel (14.9 ms, same as HBM,
  because it runs on the expanded HBM copy). One contiguous bulk read at 330 GB/s would cost 7 ms;
  a KV-stationary fused-GQA kernel reading host once would cost ~14 ms with no expansion at all.

**F3, per-layer timing (`layer_times_*.jsonl`)**: inconclusive as a GPU-time measurement. CUDA
events around module forwards measure wall time when the GPU is starved by the CPU, and the sum
over layers reproduces the step time (57 of 59 ms), so "attention = 71% of layer time" is mostly
Python in the attention module. The decode-only torch.profiler kernel sum (F1) is the GPU number.

## Status

- [x] main rerun finished 03:37 UTC; snapshot `~/cs243/cs2430_main_results_1115_0338` on the box
- [x] follow-ups F0-F5 finished 04:33 UTC (first attempt aborted on a missing CUDA include, fixed)
- [x] results pulled to `kvplace/results/gh200/` and summarized above
- [ ] instance still running (kept on purpose); stop it from the Lambda console when done

## Recommended next steps, in order of value per GPU hour

1. **Fix the baseline's wasted GPU time** (mask-free fused-GQA attention for the verify block,
   preallocated cache): ~19 ms of 45 ms per speculative step at 16k. Benefits every arm.
2. **Zero-copy prefill through HBM + split attention (host prefix, HBM tail, LSE merge).** Removes
   the 3x TTFT penalty and makes the token-window eviction policy apply to zero-copy.
3. **A KV-stationary host-read attention for the verify block** (DirectKV's Q-inner loop, or a
   bulk contiguous read into a scratch tile per layer). This is what decides whether zero-copy wins
   for speculative decoding at 32k+.
4. **CUDA-graph the target forward** so the step is GPU-bound; only then do the link/compute ratios
   in the report mean what they say.

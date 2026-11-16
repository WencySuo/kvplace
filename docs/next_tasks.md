# Next tasks after the GH200 run (brief shared by the engineer and PM agents)

## Original problem statement (the thing every task must serve)

CS243 project (Mary Guan, Wency Suo, Ellie Vitanov). Question: on a GH200, can the *verifier's*
(target model's) KV cache live in Grace host memory and be read during DFlash speculative
verification at close to HBM step latency, so that HBM holds more or longer contexts? The core
strategy is **offloading the verifier KV cache to host memory** and either (a) staging it back
through a layer-ahead copy pipeline (Ellie's `ours` arm) or (b) reading it in place from
GPU-mapped pinned memory (the `zerocopy` arm, DirectKV-style). Eviction policy v1 = which layers
stay resident and how many recent tokens per layer stay in HBM.

Fixed setup, not up for change: Qwen/Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16 (block 16), batch
size 1, transformers 5.15 eager backend driven by DFlash's generate loop, SPEED-Bench prompts,
Ellie's `run_arms.py` methodology so new numbers stay comparable with
`kvplace/results/gh200/`. No SGLang/vLLM port, no new model, no new benchmark suite.

## What the GH200 run found (see kvplace/docs/gh200_followups.md for tables)

- SM reads of mapped host memory run at 332 GB/s (95% of DMA): the DirectKV premise holds.
- The decode step is CPU-bound (ordinary: 10 ms GPU inside 48 ms; speculative: 45 of 90 ms).
- Full offload with the copy pipeline costs +3%/speculative step at 16k, +53% ordinary.
- Zero-copy wins ordinary decoding, ties speculative at 8k, loses 8% at 16k and 41% at 32k
  because transformers' `repeat_kv` GQA expansion reads host memory at ~80 GB/s; its prefill is
  3-4.5x slower because prefill also reads host.
- 19 of 45 GPU ms per speculative step are waste: `repeat_kv` (10.3) + DynamicCache `torch.cat` (8.6).
- Token-window eviction (last 2048 tokens resident, 448 MB) beats half-the-layers resident
  (1352 MB) for speculative decoding; for zero-copy the window is a no-op until attention is split.

## Code base

Ellie's clone: `/Users/wencysuo/code/cs2430_project`, branch `gh200-followups` (base for all
tasks). Key files: `kvoffload.py` (cache arms, `OffloadedKVCache`, `HostOffloadLayer`,
`make_cache`), `hostmem.py` (mapped pinned tensors, CUDA ext via load_inline; CPU fallback),
`run_arms.py` (driver), `test_kvoffload.py` (CPU correctness: outputs identical across arms;
MUST stay green), `profile_step.py`, `followups.sh` (unattended GPU stages with
`results/.done/f_<stage>` markers), `analyze.py`/`simulate.py` (cost model).
Python: `/Users/wencysuo/code/cs2430_project/.venv/bin/python` (torch 2.13, transformers 5.15,
dflash editable). No GPU, no triton, no flash_attn on this machine.

Each task works in its own git worktree:
`git -C /Users/wencysuo/code/cs2430_project worktree add /Users/wencysuo/code/cs2430_wt/<task> -b task/<task> gh200-followups`
and commits on `task/<task>`. Run tests from the worktree with the venv python above.

## Tasks

### T1 attention-and-cache (1a + 1b)
1a. Mask-free fused-GQA attention for the verify block. Register a custom attention function
    with transformers' `AttentionInterface` (name it e.g. `dflash_gqa`) that, for q_len > 1,
    avoids `repeat_kv` materialization: SDPA with `enable_gqa=True` plus a bottom-right-aligned
    causal mask (or `is_causal` only when q_len == kv_len), and the existing q_len == 1 path
    unchanged. Selected by a flag in `run_arms.py` (`--attn dflash_gqa`), default off so old
    numbers stay reproducible. CPU equivalence test vs the stock path (max abs diff reported,
    tolerance 1e-3 in fp32) for q_len 1, 16 and a prefill-size chunk, with GQA 32:8 heads.
1b. Preallocated resident cache: resident layers of `OffloadedKVCache` (and the `hbm`
    reference if cheap) stop doing per-step `torch.cat`; use fixed `[B,H,max_len,D]` buffers,
    index writes, and narrow views. Report whether this removes the block-1 output mismatch
    noted in the GH200 doc (offloaded arms vs hbm at block size 1).
Done when: tests green, a `profile` follow-up stage line exists in `followups.sh` that runs
`profile_step.py` with and without `--attn dflash_gqa`, and the commit message states exactly
what was and was not tested (no GPU here).

### T2 zerocopy-prefill-and-split (2a + 2b)
2a. Zero-copy prefill through HBM: in the `zerocopy` arm, prefill attends on HBM K/V, then the
    prefix is written to the mapped host buffer once (one D2H copy per layer), and decode
    switches to the host views. Must not change outputs. TTFT should match `ours`.
2b. Split attention with log-sum-exp merge: attention over host prefix and HBM tail as two
    calls merged by LSE (a small math-path implementation for the tail is fine since tail <=
    2048; do NOT add flash_attn/flex dependencies). This makes `tail_keep` meaningful for
    `zerocopy`. CPU test: split result equals single full attention to 1e-4 in fp32 for several
    prefix/tail splits.
Done when: tests green, `zc8/zc16/zc32` and `evict` stages in `followups.sh` include zerocopy
rows with `tail_keep` in {64, 2048, 8192}, commit message states what was not tested.

### T3 host-read-kernel (3)
KV-stationary host-read attention for the verify block (the DirectKV port). Deliver:
(i) a pure-torch reference of KV-stationary split-K attention with LSE merge for q_len <= 16,
    GQA 4:1 (32 q heads, 8 kv heads, D=128), K/V in `[S,B,H,D]` layout, CPU-tested against SDPA;
(ii) a Triton kernel `kv_host_attn.py` implementing the same tiling (all 4 q-heads x 16 block
    positions = 64 query rows per KV head as the inner loop; KV tiles streamed once; split-K
    across programs; LSE merge), reading K/V from whatever pointer it is given (host-mapped or
    HBM); a test runnable under `TRITON_INTERPRET=1` on Linux. It cannot be executed on this
    machine: say so plainly, and keep the kernel small enough to debug in one GPU session;
(iii) a `followups.sh` stage `kernel` that runs the test and a timing at 8k/16k/32k vs SDPA.
Do not wire it into the model forward yet; that is a follow-up after it is validated on GPU.

### T4 cost-model (5, modeling part only)
Ellie's cost model (`analyze.py` / `simulate.py`) mispredicts the GH200 by 44% because it
assumes a GPU-bound step. Add a CPU-side per-step term fitted from
`kvplace/results/gh200/results/profile_16k.jsonl` and the `arms_*`/`sweep_*` rows, re-run the
model against the GH200 measurements, and report the new error. Keep the L40S fit unchanged.
No pushes, no edits to the report text, no new figures beyond what `analyze.py` already emits.

## Rules for every task
- Stay inside the task. No refactors of untouched code, no new dependencies, no renaming of
  Ellie's arms or CLI flags, no changes to measurement methodology.
- `test_kvoffload.py` must end with `outputs identical across arms: True`.
- Report honestly: what was run, what passed, what could not be run here (GPU), and any
  deviation from this brief with the reason.
- Final report (returned to the orchestrator): branch name, files touched, test output tail,
  what is untested, open questions. Under 400 words.

## Status 2026-11-15 (after engineer + PM pass; nothing merged, nothing pushed, no GPU run)

| task | branch (worktree under ~/code/cs2430_wt/) | head | PM verdict | revisions |
|---|---|---|---|---|
| T1 attention-and-cache | task/attention-and-cache | 0c306ee | REVISE -> fixed | fold-variant prefill OOM bug, `prealloc` block-1 stage, `profile_gqa` trimmed 40->24 configs, help text |
| T2 zerocopy-prefill-and-split | task/zerocopy-prefill-and-split | fb5e562 | REVISE -> fixed | `split_attn` in summary groupby, e2e test hits prefill path, split rows run with `hbm`, fp32-upcast-of-host-prefix reverted |
| T3 host-read-kernel | task/host-read-kernel | 2bdc5b5 | APPROVE WITH NITS -> fixed | fused-GQA SDPA bench row, mask hoisted, cpu_view fill, interpreter test first, fail-not-skip |
| T4 cost-model | task/cost-model | 3cbb9c4 | REVISE -> fixed | identifiability stated (a,u exactly determined), `ref36` column (43.8 -> 20.9 from the reference fix alone, -> 12.0 with CPU term), fit data labelled, fit() args |

Reviews: `kvplace/docs/reviews/<task>.md`. All CPU tests green on every branch
(`test_kvoffload.py` identical across arms). Triton kernel (T3) has never executed anywhere.

### GPU session plan (one relaunch, serial, bounded)

Merge order into `gh200-followups`: T4 (no conflicts), T3, T2, T1 (T1/T2/T3 each add a stage to
the `STAGES=` line in followups.sh; trivial conflict). Then on the box:

| stage | branch | what it answers | est. GPU min |
|---|---|---|---|
| `kernel` | T3 | does the Triton kernel compile/pass; host-read GB/s vs SDPA at 8k/16k/32k | 10 (+ debug time if it fails) |
| `profile_gqa` | T1 | which attention variant removes the 19 ms of waste; does `enable_gqa`+mask hit the math kernel | 30 |
| `prealloc` | T1 | does preallocation fix the block-1 mismatch | 10 |
| `zc8 zc16 zc32` (re-run with split rows) | T2 | zero-copy TTFT at parity? does split attention + tail window make zero-copy win speculative at 16k/32k | 45-55 |
| `evict` (re-run with split rows) | T2 | eviction v1 for the zero-copy arm | 15 |

~2 hours plus kernel debugging. Pick a T1 winner from `profile_gqa` before applying `--attn` /
`--prealloc` to the zc*/evict stages (PM decision: later, not now).

## Status 2026-11-15 (evening): A40 validation done, kernel-16k merged

All of T1-T4 validated on a RunPod A40 (see `docs/a40_validation.md`): fold attention variant
is the winner (-24% GPU/step, -33% with prealloc), T3 kernel runs and matches fused SDPA from HBM
at 16k+, zero-copy prefill TTFT at parity, every arm matches hbm. Follow-up `task/kernel-16k`
(fused merge, honest bench stats, runtime split count; review `docs/reviews/kernel-16k.md`)
merged into `gh200-followups` (pushed) and synced to the pod. Open: 40-75 ms host-side stall
seen only with mapped pinned memory allocated (deferred until GH200 shows it); block-1 mismatch
is GH200-specific; split attention token mismatches until the kernel replaces it.
Next: CUDA graphs + torch.compile (task 4) on the A40, then GH200 session for timings.

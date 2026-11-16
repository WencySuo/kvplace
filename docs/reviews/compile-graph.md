# Review: T5 compile-graph (7ea8ec0, 63cd7d5 vs gh200-followups)

**Tests I re-ran:** all pass on CPU (`test_graph` skips) and on the A40 (graph 6/6, logit diff <=2.4e-7). Both new tests run in fp32, so they cannot catch the bf16 token changes below.

## Quality: acceptable
- `STATIC` branches are inert by default; `test_kvoffload` is green on CPU and on the A40. `new_zeros` also runs on the non-graph prealloc path: a ~1-2 GB memset per cache, a few ms of TTFT. Negligible; state it.
- **The block-16 explanation is wrong.** `profile_step` generates only 20 tokens, so most verify steps have `verify_size < 16`. Those sizes are not in `sizes={1,16}` and run eager: `hbm_R0_b16.json` has 5 `cudaGraphLaunch` over ~12 steps, and the 30 ms `target_cpu_ms_per_step` is inside the target forward. The drafter is only ~700 calls.
- Not reported: the graph adds GPU work at bs1 (37.5 -> 40.4 ms), so on the A40 compile+graph is slower than compile alone (tpot 34.6 vs 32.0).

## Scope: acceptable
- New flags default off, and no existing flag or arm changed.
- **The q=1 two-bmm attention changes the method.** It rounds the scores to bf16 before the softmax (the defect flagged in T2) and makes graph-only `hbm` at bs1 generate 256 tokens instead of the stock 119. Graph-only `zerocopy:0` at bs1 uses folded SDPA and reproduces the stock hash `355814778ce4`, so the fix is already in the code.
- `--help` only implies that the copy pipeline raises.

## Alignment: acceptable
- At bs1, host time in the target forward drops from 39 to 0.45 ms and launch calls from 2072 to 4. That supports the GH200 claim, where GPU work is 10 ms of a 48 ms step.
- `ours:0` cannot be graphed, so `--compile` is the only mode that compares copy and zero-copy fairly.
- `--compile` changes the tokens: 159 vs 119, both stopped at EOS, with acceptance 2.36 vs 2.19. Compare modes on step ms (tpot x tau), not tpot.

## Verdict: REVISE
1. Delete the q=1 bmm branch in the graph path and use folded SDPA for all layers; show that graph-only `hbm` at bs1 reproduces `355814778ce4` at 8k.
2. Record `graph_replays` and the replayed fraction of verify steps in every `run_arms`/`profile_step` row.
3. Make every profiled bs16 step replay a graph (e.g. capture sizes 1..bs in `profile_step` only), re-run the 16k bs16 graph rows, and correct the "eager drafter" claim.
4. Try `torch._inductor.config.emulate_precision_casts = True` once; if it does not restore the stock hash, say in the `--compile` help that tokens change and report the first divergent token index.
5. State in the `--cuda-graph` help that `ours`/`ondemand` with offloaded layers and split attention raise.

## GH200 stage (~40 GPU min)
- `profile_step` at 16k, fold+prealloc, arms `hbm`, `ours:36`, `ours:0`, `zerocopy:0`, bs1 and bs16.
- Modes: stock, compile, graph, compile+graph (`ours:0` gets stock and compile only).
- Plus `run_arms` at 16k, 4 prompts, `--compile`.
- Question: does zero-copy's ordinary-decoding win survive a GPU-bound step?

## Later, not now
Graph the drafter as its own task (it is DFlash's code). Find out why folded SDPA at q=1 is faster from host (tpot 140 vs 235 ms). Graph the copy pipeline.

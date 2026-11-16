# Review: T1 attention-and-cache (task/attention-and-cache @ a954d69)

**Verdict: REVISE** (small fixes; the core work is good)

## What I re-ran (venv python)
- `test_gqa_attn.py`: exit 0. Diff 0.0 for every case. The mask equals dflash's mask. Tiny Qwen3 end to end: 0 and 1.94e-7.
- `test_kvoffload.py`: the 4 prealloc variants pass, `outputs identical across arms: True`.
- `profile_step.py --tiny --attn dflash_gqa --prealloc` (and `_fold`): completes, and rows carry `attn`/`prealloc`. It needs `--device cpu`, as it already did before this branch.

## Quality: acceptable
- The fold index math is correct. Head kv*n_rep+r maps to row r*q+i, and that row gets mask row i.
- **Bug:** fold has no `q_len == kv_len` branch. A 16k prefill materializes a `[1,1,64k,16k]` bool mask (1 GiB) plus a bf16 additive copy (2 GiB) per layer. At 32k that becomes 4+8 GiB: an OOM, or at least a distorted TTFT. The CPU tests at 512 cannot catch it.
- `PreallocLayer.crop` matches `DynamicLayer.crop`, including crop(0). Overflow raises an error; nothing falls back silently. The commit message is accurate.

## Scope: acceptable
- Fold is a reasonable hedge, not creep. On CUDA a mask rules out flash, and enable_gqa is likely unsupported on mem-efficient, so `dflash_gqa` at block 16 may hit math (which expands K/V). Fold is plain MHA with a mask, which mem-efficient accepts, so it is the variant most likely to read K/V once. It is about 20 lines and tested. Keep it, labelled speculative.
- `--prealloc` is off by default, so the old numbers stay reproducible. Rows record `prealloc`.
- max_len: run_arms uses prompt + max_new_tokens + block + 32; profile uses prompt + 2*steps*16 + 64. Not a fixed 32k. Good.

## Alignment: weak until fixed
- **1b has no GPU path.** No stage runs `run_arms.py --prealloc`. `--prealloc` also changes the `hbm` reference, so `matches_hbm` would compare prealloc against prealloc.
- **profile_gqa** runs 40 configs: 5x the old stage, roughly 50-65 GPU min. About half are redundant. At bs=1 both variants delegate to stock sdpa, and prealloc does nothing on `ours:0`.

## Required changes
1. In `dflash_gqa_fold_attention`, when `q_len == kv_len`, use the `is_causal=True, enable_gqa=True` call instead of the folded mask.
2. Add a bounded stage: `run_arms.py`, block 1, `--max-samples 2`, arms `hbm,ours`, run with and without `--prealloc` into one file (about 10 min). Answer 1b by comparing `out_hash` against stock `hbm`.
3. Trim `profile_gqa`: the no-prealloc variants run `--block-sizes 16` only, and the prealloc variants run `--arms hbm,ours:36` only. That is 22 configs, about 30 min.
4. Say in the `--prealloc` help text that it also changes the `hbm` reference.

## Open questions from the engineer
- `--attn`/`--prealloc` in the zc*/evict stages: **later, not now.** Pick a winner from `profile_gqa` first. Adding the flags now doubles GPU time on unvalidated paths and breaks comparability with `results/gh200`.
- pyc files: leave them uncommitted. Untracking them is separate housekeeping.

## Later, not now
- torch `causal_lower_right` as a mask-free bottom-right causal option. Check whether it composes with enable_gqa.
- Record the SDPA backend that was actually selected in the profile rows.

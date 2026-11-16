# Review: T2 zerocopy-prefill-and-split (`task/zerocopy-prefill-and-split` @ c2fa625)

**Verdict: REVISE.** Two small fixes. The code is sound, but the follow-up stage as written would silently mix split rows into the existing zero-copy averages.

Re-ran on CPU: `test_split_attention.py` gives a max diff of 3.58e-06 and prefill with zero host reads. `test_kvoffload.py` passes crop stress for every variant (zc-split diff about 3e-7) and ends with `outputs identical across arms: True`. The report's test claims hold.

## Quality: acceptable
- **2a correctness is covered, though not where the report says.** Crop stress uses `prompt_len=300`, which is above the 256 threshold. So every zerocopy variant takes the prefill path and matches DynamicCache logits exactly (diff 0). The token-level e2e test never takes that path, which is an honest gap and cheap to close. I re-ran e2e with `prefill_flush_threshold=128` on all zerocopy variants and got **True**.
- **Large q_len cannot reach the explicit-softmax path.** The prefill branch requires `h == 0`, and with `h == 0` no prefix is registered, so the call falls back to stock SDPA. `kv_split` math runs only when `h > 0`, and inside DFlash's loop that means q_len is the block size (1 or 16). At 32k with block 16 the fp32 score matrix is [1,8,64,32k], about 67 MB. Counting intermediates it is about 0.4 GB per layer, transient. That is acceptable.
- **No repeat_kv-style copy of the host prefix.** GQA is folded into rows (`q.reshape`), with no `repeat_kv`. A CPU profile of `_attn_lse` on a permuted `[S,B,H,D]` view shows only views (`_reshape_alias`/`_unsafe_view`) in front of `bmm`, with no clone or contiguous call. The strides (ld=1024, unit inner stride) are cuBLAS-legal, so an in-place read is likely. This is unverified on GPU.
- **bf16 numerics risk.** The scores come out of a bf16 matmul and are rounded before `.float()`, which stock SDPA does not do. Expect `matches_hbm` to drop on GPU. The commit message only says "bf16 untested".
- Nits: a stray blank line before `install()`. `install()` does not call `restore_attention()` for non-split arms, so attention selection stays switched for the rest of the process. That is harmless in the planned stages because split runs last in each grid.

## Scope: strong
Only the files the task needs are touched. The flag is new and defaults to off. No renames and no new dependencies. Folding 2a and 2b into one commit is disclosed. On (4): `kv_split` is set on the target config, so every layer goes through the wrapper, and resident or prefix-less layers fall through to the original function. That is the minimum transformers allows, so it is in scope.

## Alignment: acceptable
2a removes the 3-4.5x TTFT penalty, and 2b makes `tail_keep` meaningful for zero-copy, both as asked.
- **Stage cost:** about 90 extra rows: zc8 +24, zc16 +24, zc32 +18, evict +24. That is roughly 45-55 GPU minutes at N=4, using the old per-run times of about 22 s at 8k/16k and about 50 s at 32k. Bounded, and worth it.
- **Summary bug:** `summarize_followups.py` groups by `(file, block_size, arm, R, mode, tail)` and does not include `split_attn`. Split tail=64 rows will be averaged into the plain zerocopy rows, and evict split 0/1 rows will merge.
- **No match check in zc stages:** split rows in zc8/zc16/zc32 run without `hbm` in the same call, so `matches_hbm` is None there.
- **Interpretation risk:** the split path issues about 60 aten ops per layer against about 3 for SDPA. In a CPU-bound step that could add 10-20 ms of launch overhead and be misread as a memory-placement cost.

## Required changes
1. Add `split_attn` (fillna False) to the groupby in `summarize_followups.py` so split rows never average with non-split rows.
2. In `test_kvoffload.end_to_end`, run the zerocopy variants with `prefill_flush_threshold=128` (or add `/pf128` copies) so that token-identical e2e covers the 2a prefill path.

Nits, optional: fold the tail-64 split into each zc stage's main call (`--zc-split 0,1`) so it gets `matches_hbm` at no extra cost, and drop the evict@64 split row, which duplicates zc16. State the bf16 score rounding and the launch count in the commit message.

## Later, not now
- A zc-split row in the `profile` stage, to separate GPU time from launch overhead.
- An LSE-returning fused kernel for the prefix, either T3's kernel or efficient-attention with expanded K/V.
- Fp32-accumulated scores.

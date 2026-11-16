# Review: task/kernel-16k (b406f21), the 16k kernel-bench cliff

**Verdict: REVISE** (three small changes; the diagnosis and the merge kernel are sound)

## What I re-ran (A40 pod; the three files rsynced, GPU idle afterwards)
- `python test_kv_host_attn.py` on CUDA: all 10 cases pass, max diff <= 2.6e-4 (new 16384/nsplit 37, 16384/default and 32768 cases included).
- `TRITON_INTERPRET=1`: passes. The commit message calls this untested.
- Local CPU tests pass; `test_kvoffload.py`: identical across arms.
- Bench, HBM kernel: 0.128 / 0.177 / 0.293 ms at 8k/16k/32k. Fused SDPA: 0.100 / 0.181 / 0.348. Host-mapped: 1.99 / 4.43 / 9.68, unchanged. At 8k the kernel is 28% slower than fused SDPA in my run; the report says 17%.

## Quality: acceptable
- Merge kernel is correct by reading. B>1: the partial indexing `(s*B*Hkv + bh)` and the output offset `(bh*R + r)*D` match the layouts (`out = empty_like(q)`, and `q` is asserted contiguous). nsplit=1: weight 1. A -inf LSE gets weight 0, and an all-masked row returns 0, not NaN. nsplit=1 and Q=1/5 are tested on CUDA; B>1 is untested but outside the fixed setup.
- `NS: tl.constexpr` is a latent stall. nsplit = cdiv(S, chunk) changes as S grows, giving 11 distinct values on the A40 and 22 on the GH200 over 8k-32k. Each new value is a JIT compile in the middle of decode. `range(NS)` works with a runtime int.
- The module docstring still says "UNTESTED ON GPU" and "the merge is torch".
- The stall evidence (0/14 vs 3-5/14) comes from scratch scripts that were not committed.

## Scope: acceptable
The fused merge goes beyond the task, but it is 27 lines, tested, and the report states it. It is justified because on the GH200 the host read at 16k is ~0.2 ms, so a 0.13 ms torch merge would be ~40% overhead on exactly the path the project is about. Nothing renamed or deleted. nsplit: keep one heuristic and the existing HBM (nsplit=8) comment; no separate default.

## Alignment: acceptable, with one gap
The bench now produces kernel numbers that can be interpreted, which is what the GH200 `kernel` stage needs. But switching to median-only hides a cost that may be real. The stall appears only once mapped pinned memory exists, which the zerocopy arm always has, so a median-only bench would hide it from the GH200 kernel stage. There is no GH200 `kernel_bench.csv` yet, so changing columns now is free.

## Required changes
1. Have `ev_time` return median, mean and max per row, plus a count of iterations above 5x the median, and write all of them to `kernel_bench.csv`. Use a fresh file name, or write a header when the columns change.
2. Make `NS` a runtime argument of `_merge_kernel` instead of `tl.constexpr`.
3. Update the module docstring and the commit message to the current state: GPU-tested on the A40, fused merge on CUDA, interpreter path passes, and 8k is ~17-28% slower than fused SDPA.

## Later, not now
- Root-cause the stall (sync APIs vs `cudaHostAlloc`/`cudaFreeHost` of the previous S's buffers, NUMA, or pod noise). Do this only if the GH200 `kernel` stage shows stalls, or if per-step zerocopy decode times show 40-75 ms outliers.
- A separate HBM nsplit default, if the kernel is ever used for resident layers.

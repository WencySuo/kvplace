# A40 validation of the task branches (RunPod, 2026-11-15)

Pod: RunPod 1x NVIDIA A40 46 GB (Ampere sm86, x86 host, 96 cores), Ubuntu 24.04, CUDA 12.8 toolkit,
torch 2.13.0+cu130, transformers 5.15.0, triton 3.7.1. Code: `gh200-followups` @ adb8db4 (all four
task branches merged). Queue `pod/dev_a40.sh`, 22 min wall, every step passed. Raw results:
`kvplace/results/a40/`. Host-memory reads here cross PCIe (~15 GB/s in-kernel), so host-side
timings are NOT comparable with the GH200; everything below is correctness plus HBM-side timing.

## What passed
- GPU correctness gate (`test_kvoffload.py` on CUDA): all 17 cache variants identical to the
  stock cache, including zerocopy, zc-split (3.8e-7 logit diff), prealloc.
- T3 Triton kernel: first execution ever. Interpreter and CUDA tests pass; bf16 error vs SDPA
  1.4e-4 to 2.6e-4 up to S=16385, Q=16, GQA 32:8.
- T1 attention variants and preallocation profiled at 16k (24 configs).
- T1b block-1 preallocation check, T2 zero-copy prefill/split correctness, summary.

## T1: which attention path avoids the GQA expansion on CUDA (speculative, block 16, 16k)

GPU kernel ms per verify step (A40; wall ~= GPU here, the step is GPU-bound on this box):

| attention | prealloc | hbm | ours R=36 | ours R=0 copy | attention kernel | GQA expand | cache cat |
|---|---|---|---|---|---|---|---|
| stock sdpa (repeat_kv + mask) | no | 124 | 125 | 116 | 33 | 35 | 13 |
| `dflash_gqa` (enable_gqa + mask) | no | **233** | 233 | 227 | math path | - | 13 |
| `dflash_gqa_fold` | no | **94** | 95 | 85 | 35 (cutlass fmha) | 1 | 13 |
| stock sdpa | yes | 114 | 114 | - | 33 | 35 | 2 |
| `dflash_gqa_fold` | yes | **84** | 83 | - | 35 | 1 | 2 |

- The PM's prediction held exactly: `enable_gqa=True` with a mask falls to PyTorch's math
  kernel (top kernel `elementwise_kernel<128,2>`, +650 launches) and is 1.9x slower. Drop it.
- The fold variant is the winner: it dispatches to the memory-efficient cutlass FMHA with no
  expansion copy, -24% GPU time per speculative step. With preallocation -33% (124 -> 84 ms).
- Preallocation alone: cache cat 13 -> 2 ms. For ordinary decoding it cuts hbm from 47.6 to
  39.3 ms/token (-17%) on this box.
- Zero-copy over PCIe: 967 ms/step stock, 159 with fold (the expansion copy was reading host at
  2.4 GB/s). Confirms the mechanism, not a usable number.

## T1b: block-1 mismatch
On the A40 all arms match `hbm` at block 1 with and without `--prealloc` (hashes identical per
prompt). The GH200 mismatch is therefore specific to its (cuDNN) attention dispatch, not to the
cache layout, and the A40 cannot settle it. Re-check on the GH200 with `--prealloc`.

## T2: zero-copy prefill on HBM + split attention (8k, 1 prompt, correctness)
- Prefill-on-HBM works: zerocopy TTFT 1.54 s vs copy pipeline 1.50 s (on the GH200 it was 3-4.5x).
- zerocopy without split: identical tokens to hbm at block 1 and 16.
- zc-split: 3 of 4 rows differ from hbm at the token level (tail 64 b1/b16, tail 2048 b1); the
  tail-2048 block-16 row matches. Expected: kv_split rounds scores to bf16 before the softmax,
  which SDPA does not (PM flagged it). The fp32-accumulating T3 kernel is the fix; until then
  split rows are timing-only and `matches_hbm` must be reported alongside.

## T3: kernel bench (kernel vs SDPA, K/V in HBM; host rows are PCIe and only show the ratio)
| S | kernel HBM | sdpa fused-GQA HBM | sdpa repeat_kv HBM | kernel host | sdpa fused host | sdpa repeat_kv host |
|---|---|---|---|---|---|---|
| 8k | 0.28 ms (121 GB/s) | 0.10 (336) | 0.83 (40) | 2.2 (15.4) | 10.7 (3.1) | 13.9 (2.4) |
| 16k | 3.33 (20) ?? | 0.18 (370) | 1.89 (36) | 4.6 (14.6) | 21.7 (3.1) | 27.6 (2.4) |
| 32k | 0.37 (368) | 0.35 (386) | 3.76 (36) | 10.4 (12.9) | 47.5 (2.8) | 55.7 (2.4) |

- From host the kernel reads 5x more efficiently than either SDPA path (reads each byte once),
  which is the KV-stationary premise; absolute host numbers are PCIe-bound.
- From HBM at 32k the kernel matches fused SDPA; at 8k it is 2.8x slower (launch/merge
  overhead dominates small S). The 16k row (3.3 ms) is anomalous and under investigation.

## Environment notes for future pods
- `setup.sh` does not install `ninja`; hostmem's extension needs it (`pip install ninja`).
- torch 2.13 cu130 wheel + system nvcc 12.8 works for the hostmem extension (torch ships nvcc).
- Never `pkill -f` a pattern that appears in the ssh command line itself; use `[p]attern`.

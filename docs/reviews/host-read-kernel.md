# Review: T3 host-read-kernel (`task/host-read-kernel` @ 905bf10)

**Verdict: APPROVE WITH NITS**

I re-ran the tests here. Ref vs SDPA 4.77e-07, uneven merge 3.58e-07, kernel tests SKIP (no triton), `test_kvoffload.py` ends `outputs identical across arms: True`, `bash -n followups.sh` clean. All match the report.

## 1. Quality: acceptable
- **Tiling is right.**
  - Grid `(B*Hkv, NSPLIT)`. Each program holds all 64 rows (4 grouped q heads x 16 positions) for one KV head.
  - It streams `[64,128]` K/V tiles over a key range no other program touches, never reloading them. Each K/V byte is read once.
  - Head mapping (`h*G+g`) and the partial-output indexing are correct.
- **Masking has no NaN.** A row fully masked within a split gets `m_safe=0`, `alpha=0`, lse=-inf. The merge gives it weight 0, and keys `0..S-Q` are visible to every row.
- **Debuggable in one session.** About 60 lines that mirror the reference; small first test case (S=300).
- **Honesty.** The commit message and docstrings say plainly what was not run. The tolerances are labelled as guesses.

## 2. Scope: strong
- Five files, all in T3. Not wired into the forward pass, no new dependencies, Ellie's code untouched.
- **Default stage list: in scope.** The brief asks for a stage, and every existing stage is in the default list.
  - On the box every other stage already has a `.done` marker, so an unattended run is `kernel` + `summarize`, a few minutes.
- **Fails safely.** The runner does not use `set -e`. `.done/f_kernel` is written only if test and bench both succeed. A failure is logged and the run moves on.
- Expect a trivial `STAGES` line conflict with T1/T2.

## 3. Alignment: strong
- **Baseline matches the measured path.** `sdpa_transformers_like` is repeat_kv on the permuted host view plus a mask. That is the access pattern behind F0's 1.346 ms host row and F1's 29+15 ms.
- **Comparable with `results/gh200`.** Same layout and S values as F0, and the same lack of an L2 flush.
- **Gap.** The `enable_gqa` no-mask SDPA row (F0: 0.648 ms host) is missing. That is the post-T1 bar the kernel actually has to beat.
- The stage is bounded at about 2-3 GPU minutes.

## Nits
1. Add an `enable_gqa=True` no-mask SDPA row to the bench, as `sm_read_bench.py` does.
2. Build the mask once, outside the timed lambda, to match F0.
3. Fill the host tensors via `kx.cpu_view.copy_(...)`, as `sm_read_bench.py` and `hostmem.selftest` do.
4. Start `stage_kernel` with `TRITON_INTERPRET=1 python test_kv_host_attn.py`, to separate semantic bugs from compile bugs.
5. In `test_kernel`, fail instead of SKIP when CUDA is present but triton is not.

## Later, not now
- **L2 caveat.** At 8k, K+V (32 MB) fits in the GH200's L2, so host bandwidth at 8k may be optimistic in both F0 and this bench. Trust 32k.
- Merge in-kernel: the fp32 partials at 16k are about 17 MB plus around 5 launches.
- Add `kernel_bench.csv` to the summary.
- Wire the kernel into the zerocopy arm after it is validated on GPU.

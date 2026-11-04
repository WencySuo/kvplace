# kvplace / kvdiag

CS243 project. Team: Mary Guan, Wency Suo, Ellie Vitanov.

Diagnostic suite: can a shared KV-cache prefix held in GH200 host (Grace) DRAM be read during
speculative-decoding verification at HBM speed, and at what fan-out? Three stages, one JSONL.

| stage | command | needs | answers |
|---|---|---|---|
| A capacity | `python -m kvdiag.audit --hbm-gb 96` | nothing | Q1, Q2: bytes released, requests admitted, by number of warm prefixes M |
| L link | `python -m kvdiag.bench_link` | GH200 | beta_dma (memcpy), beta_sm (SM reads), L2-caches-host-lines test, latency |
| C attention | `python -m kvdiag.bench_attn --S 32768 --arms hbm host_direct ...` | GH200 + FlashInfer | Q3, Q4: t_host/t_hbm vs R = B*q; effective link GB/s |
| B profile | `python -m kvdiag.profile --S 8192 32768 --B 1 8` | GH200 + transformers | Q5: verification step time and attention share |
| figures | `python -m kvdiag.plot` ; `python -m kvdiag.summarize` | results.jsonl | fig_capacity, fig_attention_*, fig_profile |

Predictions without a GPU: `python -m kvdiag.bench_attn --dry-run && python -m kvdiag.plot`
(rows carry status `predicted` and plot dashed).

## Setup

    uv venv --python 3.13 .venv && uv pip install -r requirements.txt     # Mac / CPU
    .venv/bin/python -m pytest tests/ -q                                  # reference + shapes tests

On the GH200 (aarch64), use the NVIDIA PyTorch container, then `pip install flashinfer-python
transformers`. First task there:

    python -c "import kvdiag.hostmem as h; h.selftest()"      # host view readable by a kernel?
    python -m pytest tests/test_arms.py -q                     # correctness gate, all arms

## Status (2026-11-03)

Verified on Mac: `shapes.py`, `audit.py`, `plot.py`, `summarize.py`, `bench_attn.py --dry-run`,
numpy reference and LSE merge tests. **Unverified** (no CUDA here): `hostmem.py`, `arms.py`,
`bench_link.py`, `profile.py`, the real path of `bench_attn.py`. Each carries a STATUS header.

Model configs and revisions are committed under `kvdiag/model_configs/`.
Every results row has run_id, case_id, stage, status; non-ok rows (oom, failed, unsupported)
are written, never dropped. `results/manifest.json` records the environment.

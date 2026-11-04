"""Stage B: target verification pass, all-HBM, HF transformers. No drafter, no generation.

STATUS: UNVERIFIED on GH200 as of 2026-11-03.

    python -m kvdiag.profile --S 8192 32768 --B 1 8 --out results/results.jsonl

Per case: fills a KV cache with S synthetic tokens per sequence (chunked prefill, labeled
synthetic; acceptance rates are NOT measured here), then times the verification step: a forward
of q=16 new positions per sequence against the cache. Reports step_ms (CUDA events, clean),
attn_ms (from one torch.profiler trace: sum of attention kernels, overhead-inflated, labeled),
reserved bytes after load (fills Stage A 'workspace'), and link_ms_if_host = S*kv/token/beta.
HF keeps one KV copy per sequence (no prefix dedup) - fine for step composition, said in report.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # allow `python kvdiag/profile.py` as well as `python -m kvdiag.profile`

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

import torch

from kvdiag.audit import manifest
from kvdiag.shapes import load_dims, GH200_ASSUMED

ATTN_KEYS = ("attn", "attention", "flash", "fmha", "sdpa", "scaled_dot", "mem_efficient")


def load_model(name, attn_impl):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation=attn_impl)
    m.eval()
    return m


@torch.no_grad()
def prefill(model, B, S, chunk=2048, seed=0):
    from transformers import DynamicCache
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (B, S), generator=g, device="cuda")
    cache = DynamicCache()
    for a in range(0, S, chunk):
        b = min(S, a + chunk)
        model(input_ids=ids[:, a:b], past_key_values=cache, use_cache=True,
              position_ids=torch.arange(a, b, device="cuda").expand(B, -1))
    return cache


@torch.no_grad()
def verify_step(model, cache, B, S, q, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (B, q), generator=g, device="cuda")
    pos = torch.arange(S, S + q, device="cuda").expand(B, -1)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, position_ids=pos)
    # drop the q appended positions so the step is repeatable
    cache.crop(S)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--model-key", default="qwen3-8b")
    ap.add_argument("--attn", default="flash_attention_2", help="fallback: sdpa")
    ap.add_argument("--S", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--q", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--beta-gbps", type=float, default=GH200_ASSUMED.beta_dma / 1e9)
    ap.add_argument("--out", default="results/results.jsonl")
    a = ap.parse_args(argv)

    dims = load_dims(a.model_key)
    man = manifest()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = load_model(a.model, a.attn)
    except Exception as e:  # noqa: BLE001
        print("falling back to sdpa:", repr(e)[:200]); a.attn = "sdpa"; model = load_model(a.model, a.attn)
    torch.cuda.synchronize()
    reserved_after_load = torch.cuda.memory_reserved()
    print(f"loaded {a.model} attn={a.attn} reserved={reserved_after_load/2**30:.1f} GiB")

    with out.open("a") as f:
        i = 0
        for S in a.S:
            for B in a.B:
                row = {"run_id": run_id, "case_id": f"B-{i:03d}", "stage": "B", "model": a.model_key,
                       "model_revision": dims.revision, "S": S, "B": B, "q": a.q, "attn_impl": a.attn,
                       "inputs": "synthetic random token ids (shape diagnostics only)",
                       "prefix_dedup": False, "graph_mode": "eager", "tp": 1,
                       "reserved_after_load_bytes": reserved_after_load,
                       "link_ms_if_host": S * dims.kv_bytes_per_token / (a.beta_gbps * 1e9) * 1e3,
                       "gpu": man.get("gpu"), "torch_version": man.get("torch"), "git_sha": man.get("git_sha"),
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                i += 1
                try:
                    t0 = time.perf_counter()
                    cache = prefill(model, B, S)
                    torch.cuda.synchronize()
                    row["prefill_s"] = time.perf_counter() - t0
                    for _ in range(a.warmup):
                        verify_step(model, cache, B, S, a.q)
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(a.trials):
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record(); verify_step(model, cache, B, S, a.q); e1.record(); torch.cuda.synchronize()
                        samples.append(e0.elapsed_time(e1))
                    row["step_ms"] = statistics.median(samples); row["samples_ms"] = samples
                    row["mem_reserved_bytes"] = torch.cuda.memory_reserved()
                    # one profiler trace for attribution (overhead-inflated; kept separate from step_ms)
                    from torch.profiler import profile, ProfilerActivity
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        verify_step(model, cache, B, S, a.q); torch.cuda.synchronize()
                    attn_us = total_us = 0.0
                    for ev in prof.key_averages():
                        t = getattr(ev, "device_time_total", getattr(ev, "cuda_time_total", 0.0))
                        total_us += t
                        if any(k in ev.key.lower() for k in ATTN_KEYS):
                            attn_us += t
                    row.update({"attn_ms": attn_us / 1e3, "profiled_total_ms": total_us / 1e3,
                                "attn_share_profiled": attn_us / total_us if total_us else None,
                                "attn_kernel_match": list(ATTN_KEYS), "status": "ok"})
                    del cache
                except torch.cuda.OutOfMemoryError as e:
                    row.update({"status": "oom", "error": str(e)[:200]})
                except Exception as e:  # noqa: BLE001
                    row.update({"status": "failed", "error": repr(e)[:300]})
                torch.cuda.empty_cache()
                f.write(json.dumps(row) + "\n"); f.flush()
                print({k: row.get(k) for k in ("S", "B", "status", "step_ms", "attn_ms", "attn_share_profiled", "link_ms_if_host")})


if __name__ == "__main__":
    main()

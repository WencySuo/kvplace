"""Tier 0: host->GPU bandwidth, two paths, plus the L2-caches-host-lines test and load latency.

STATUS: UNVERIFIED on GH200 as of 2026-11-03.

    python -m kvdiag.bench_link --out results/results.jsonl

Rows (stage "L"):
  test=memcpy_h2d     chunk_bytes, gbps            copy engine; feeds host_staged / host_copy (beta_dma)
  test=sm_read        blocks, threads, src, gbps   SM loads, disjoint regions; src in {host, hbm}; feeds host_direct (beta_sm)
  test=sm_read_shared blocks, footprint, gbps      every block reads the same host region; gbps >> C2C peak => L2 caches host lines
  test=chase          src, ns_per_hop              single-thread dependent loads
  test=nvbandwidth    raw stdout, parsed h2d gbps  cross-check (optional binary)
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # allow `python kvdiag/bench_link.py` as well as `python -m kvdiag.bench_link`

import argparse
import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import torch

from kvdiag import hostmem
from kvdiag.audit import manifest

MiB = 1 << 20


def timed(fn, warmup=3, trials=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1))
    out.sort()
    return out[len(out) // 2], out


def memcpy_sweep(total=256 * MiB, chunks=(256 * 1024, 1 * MiB, 4 * MiB, 8 * MiB, 16 * MiB, 32 * MiB, 64 * MiB, 256 * MiB)):
    src = torch.empty(total, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(total, dtype=torch.uint8, device="cuda")
    for c in chunks:
        n = total // c
        def fn():
            for i in range(n):
                dst[i * c:(i + 1) * c].copy_(src[i * c:(i + 1) * c], non_blocking=True)
        ms, samples = timed(fn)
        yield {"test": "memcpy_h2d", "chunk_bytes": c, "n_copies": n, "bytes": total, "median_ms": ms,
               "gbps": total / (ms * 1e-3) / 1e9, "samples_ms": samples}


def sm_read_sweep(total=256 * MiB, threads=256, blocks_list=(8, 16, 33, 66, 132, 264, 528, 1056)):
    host = hostmem.host_tensor((total,), torch.uint8)
    host.cpu_view.fill_(1)
    hbm = torch.ones(total, dtype=torch.uint8, device="cuda")
    for src_name, t in (("host", host), ("hbm", hbm)):
        for b in blocks_list:
            per = total // b // 16 * 16
            ms, samples = timed(lambda: hostmem.read_reduce(t, b, threads, per, False))
            yield {"test": "sm_read", "src": src_name, "blocks": b, "threads": threads, "bytes": per * b,
                   "median_ms": ms, "gbps": per * b / (ms * 1e-3) / 1e9, "samples_ms": samples}
    # L2 test: all blocks read the same footprint; compare aggregate GB/s to the disjoint number
    for fp in (4 * MiB, 16 * MiB, 64 * MiB):
        for b in (132, 528):
            ms, samples = timed(lambda: hostmem.read_reduce(host, b, threads, fp, True))
            yield {"test": "sm_read_shared", "src": "host", "blocks": b, "threads": threads, "footprint_bytes": fp,
                   "bytes": fp * b, "median_ms": ms, "gbps": fp * b / (ms * 1e-3) / 1e9, "samples_ms": samples}


def chase(n_elems=32 * MiB, hops=200_000):
    perm = torch.randperm(n_elems, dtype=torch.int64)
    nxt = torch.empty(n_elems, dtype=torch.int64)
    nxt[perm] = torch.roll(perm, -1)           # one cycle through all elements, random order
    host = hostmem.host_tensor((n_elems,), torch.int64); host.cpu_view.copy_(nxt)
    hbm = nxt.cuda()
    for src_name, t in (("host", host), ("hbm", hbm)):
        ms, samples = timed(lambda: hostmem.pointer_chase(t, 0, hops), warmup=1, trials=5)
        yield {"test": "chase", "src": src_name, "hops": hops, "working_set_bytes": n_elems * 8,
               "median_ms": ms, "ns_per_hop": ms * 1e6 / hops, "samples_ms": samples}


def nvbandwidth():
    exe = shutil.which("nvbandwidth")
    if not exe:
        return {"test": "nvbandwidth", "status": "unsupported", "reason": "binary not on PATH"}
    p = subprocess.run([exe, "-t", "host_to_device_memcpy_ce"], capture_output=True, text=True, timeout=300)
    m = re.search(r"SUM\s+host_to_device_memcpy_ce\s+([\d.]+)", p.stdout)
    return {"test": "nvbandwidth", "raw": p.stdout[-2000:], "h2d_ce_gbps": float(m.group(1)) if m else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/results.jsonl")
    ap.add_argument("--skip", nargs="*", default=[], choices=["memcpy", "sm", "chase", "nvbandwidth"])
    a = ap.parse_args(argv)
    man = manifest()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    gens = []
    if "memcpy" not in a.skip: gens.append(memcpy_sweep())
    if "sm" not in a.skip: gens.append(sm_read_sweep())
    if "chase" not in a.skip: gens.append(chase())
    if "nvbandwidth" not in a.skip: gens.append(iter([nvbandwidth()]))
    with out.open("a") as f:
        i = 0
        for g in gens:
            for row in g:
                rec = {"run_id": run_id, "case_id": f"L-{i:03d}", "stage": "L", "status": row.pop("status", "ok"),
                       "gpu": man.get("gpu"), "git_sha": man.get("git_sha"),
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **row}
                f.write(json.dumps(rec) + "\n"); f.flush(); i += 1
                print({k: v for k, v in rec.items() if k not in ("samples_ms", "raw", "run_id", "git_sha", "timestamp")})


if __name__ == "__main__":
    main()

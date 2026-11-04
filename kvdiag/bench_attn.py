"""Stage C: grouped shared-prefix attention, HBM vs host-resident K/V, swept over R and S.

    python -m kvdiag.bench_attn --dry-run                      # predicted rows, no GPU
    python -m kvdiag.bench_attn --arms hbm host_direct --S 32768   # GH200

One row per (model, S, R, arm, rep). Each timed sample is one sweep over n_caches distinct
layer caches (rotated so the working set exceeds L2); median_ms is per layer (sweep / n_caches).
Private-suffix attention is NOT included (identical in both modes; added analytically in report).
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # allow `python kvdiag/bench_attn.py` as well as `python -m kvdiag.bench_attn`

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

from kvdiag.shapes import (load_dims, make_shapes, t_pred_s, link_bytes, q_tiles, GH200_ASSUMED, Machine, ARMS, PRESETS)
from kvdiag.audit import manifest

DEFAULT_R = [1, 16, 128, 1024, 4096]
DEFAULT_S = [8192, 32768]


def machine_from(path: str | None) -> Machine:
    if not path:
        return GH200_ASSUMED
    if path in PRESETS:
        return PRESETS[path]
    d = json.loads(Path(path).read_text())
    return Machine(**{**GH200_ASSUMED.__dict__, **d, "source": d.get("source", path)})


def base_row(run_id, i, dims, sh, arm, chunk_tokens, m: Machine, man):
    return {
        "run_id": run_id, "case_id": f"C-{i:04d}", "stage": "C", "model": dims.name,
        "model_revision": dims.revision, "arm": arm, "placement": "hbm" if arm == "hbm" else "host_pinned",
        "R": sh.R, "S": sh.L, "L": sh.L, "q": 16, "B": sh.R / 16, "G": sh.G, "h_q": sh.h_q, "h_kv": sh.h_kv,
        "d": sh.d, "chunk_tokens": chunk_tokens if arm in ("host_direct_chunked", "host_staged") else None,
        "kv_bytes": sh.kv_bytes, "flops": sh.flops, "intensity": sh.intensity,
        "link_bytes_est": link_bytes(sh, arm, m), "q_tiles_est": q_tiles(sh, m.tile_rows),
        "graph_mode": "eager", "dtype": "bfloat16", "gpu": man.get("gpu"), "nvidia_smi": man.get("nvidia_smi"),
        "torch_version": man.get("torch"), "flashinfer_version": man.get("flashinfer"),
        "git_sha": man.get("git_sha"), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": m.__dict__,
    }


def finish(row, median_ms, p95_ms, samples_ms, status, **extra):
    sh_flops, kv = row["flops"], row["kv_bytes"]
    row.update({
        "status": status, "median_ms": median_ms, "p95_ms": p95_ms, "samples_ms": samples_ms,
        "achieved_tflops": sh_flops / (median_ms * 1e-3) / 1e12,
        "effective_kv_gbps": kv / (median_ms * 1e-3) / 1e9,
        **extra,
    })
    return row


def dry_run(a, dims, m, man, run_id, out):
    n = 0
    with out.open("a") as f:
        for S in a.S:
            for R in a.R:
                for arm in a.arms:
                    sh = make_shapes(dims, R, S)
                    t = t_pred_s(sh, arm, m, chunk_tokens=a.chunk_tokens) * 1e3
                    row = finish(base_row(run_id, n, dims, sh, arm, a.chunk_tokens, m, man),
                                 t, t, [], "predicted", rep=0, trials=0, warmup=0, n_caches=0)
                    f.write(json.dumps(row) + "\n"); n += 1
    print(f"dry run: wrote {n} predicted rows to {out} using {m.source} constants")


def real_run(a, dims, m, man, run_id, out):
    # UNVERIFIED on GH200 as of 2026-11-03; exercised only by tests on a non-GH200 GPU.
    import torch
    from kvdiag import arms as A
    n = 0
    with out.open("a") as f:
        for S in a.S:
            for R in a.R:
                sh = make_shapes(dims, R, S)
                caches = [A.make_kv(sh, "hbm", seed=s) for s in range(a.n_caches)]
                hcaches = None
                Q = A.make_q(sh, seed=123)
                for rep in range(a.reps):
                    order = list(a.arms) if rep % 2 == 0 else list(reversed(a.arms))
                    for arm in order:
                        row = base_row(run_id, n, dims, sh, arm, a.chunk_tokens, m, man); n += 1
                        try:
                            if arm != "hbm" and hcaches is None:
                                hcaches = [A.make_kv(sh, "host", seed=s) for s in range(a.n_caches)]
                            kvs = caches if arm == "hbm" else hcaches
                            obj = A.make_arm(arm, sh, chunk_tokens=a.chunk_tokens)
                            mem0 = torch.cuda.memory_allocated()
                            for _ in range(a.warmup):
                                for K, V in kvs:
                                    obj.run(Q, K, V)
                            torch.cuda.synchronize()
                            samples = []
                            for _ in range(a.trials):
                                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                w0 = time.perf_counter()
                                e0.record()
                                for K, V in kvs:
                                    obj.run(Q, K, V)
                                e1.record(); torch.cuda.synchronize()
                                samples.append(e0.elapsed_time(e1) / len(kvs))
                                row.setdefault("wall_ms_samples", []).append((time.perf_counter() - w0) * 1e3 / len(kvs))
                            mem1 = torch.cuda.memory_allocated()
                            samples_sorted = sorted(samples)
                            finish(row, statistics.median(samples), samples_sorted[int(0.95 * (len(samples) - 1))],
                                   samples, "ok", rep=rep, trials=a.trials, warmup=a.warmup, n_caches=a.n_caches,
                                   mem_allocated_delta_bytes=mem1 - mem0, backend=getattr(obj, "backend", None),
                                   kv_ptr_attrs=A.pointer_attrs(kvs[0][0]))
                        except torch.cuda.OutOfMemoryError as e:
                            finish(row, float("nan"), float("nan"), [], "oom", rep=rep, error=str(e)[:200])
                        except Exception as e:  # noqa: BLE001
                            finish(row, float("nan"), float("nan"), [], "failed", rep=rep, error=repr(e)[:300])
                        f.write(json.dumps(row, allow_nan=True) + "\n"); f.flush()
                        print(f"{arm:20s} S={S:6d} R={R:5d} rep={rep} {row['status']:9s} "
                              f"{row['median_ms']:8.3f} ms/layer  {row['effective_kv_gbps']:7.0f} GB/s  {row['achieved_tflops']:6.1f} TFLOP/s")
                del caches, hcaches
                torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--R", type=int, nargs="+", default=DEFAULT_R)
    ap.add_argument("--S", type=int, nargs="+", default=DEFAULT_S)
    ap.add_argument("--chunk-tokens", type=int, default=4096, help="16 MiB per layer for h_kv=8, d=128")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-caches", type=int, default=8, help="distinct layer caches rotated per sweep")
    ap.add_argument("--machine", default=None, help="preset name (gh200, h100_pcie5, l40s_pcie4) or JSON of measured Machine constants")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="results/results.jsonl")
    a = ap.parse_args(argv)

    dims = load_dims(a.model)
    m = machine_from(a.machine)
    man = manifest()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    (dry_run if a.dry_run else real_run)(a, dims, m, man, run_id, out)


if __name__ == "__main__":
    main()

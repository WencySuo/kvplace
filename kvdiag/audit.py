"""Stage A: environment manifest + config-derived capacity table. Runs without a GPU.

    python -m kvdiag.audit --hbm-gb 96 --out results/results.jsonl

Writes manifest.json next to the JSONL and appends one row per (S, P, M) with stage="A".
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # allow `python kvdiag/audit.py` as well as `python -m kvdiag.audit`

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from kvdiag.shapes import load_dims, capacity_rows, row_dict, GH200_ASSUMED

GiB = 1 << 30


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def manifest(extra: dict | None = None) -> dict:
    m = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.sm,clocks.max.sm,pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader"]),
        "cpu": _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or _run(["lscpu"]),
        "tp": 1, "graph_mode": "eager", "dtype": "bfloat16", "background_load": "idle (assumed; record if not)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import torch  # noqa
        m["torch"] = torch.__version__
        m["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            m["gpu"] = p.name
            m["hbm_bytes"] = p.total_memory
            m["l2_bytes"] = getattr(p, "L2_cache_size", None)
    except ImportError:
        m["torch"] = None
    try:
        import flashinfer  # noqa
        m["flashinfer"] = flashinfer.__version__
    except ImportError:
        m["flashinfer"] = None
    m.update(extra or {})
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="qwen3-8b")
    ap.add_argument("--drafter", default="qwen3-8b-dflash-b16")
    ap.add_argument("--hbm-gb", type=float, default=96.0, help="GH200 variant: 96 or 144")
    ap.add_argument("--workspace-gb", type=float, default=2.0, help="allocator reserve after load; measure in Stage B")
    ap.add_argument("--S", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--P", type=int, nargs="+", default=[256, 4096])
    ap.add_argument("--M", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--q", type=int, default=16)
    ap.add_argument("--beta-gbps", type=float, default=GH200_ASSUMED.beta_dma / 1e9)
    ap.add_argument("--out", default="results/results.jsonl")
    ap.add_argument("--print", action="store_true", dest="do_print")
    a = ap.parse_args(argv)

    target, drafter = load_dims(a.target), load_dims(a.drafter)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    man = manifest({"run_id": run_id, "stage": "A", "target": row_dict(target), "drafter": row_dict(drafter),
                    "hbm_gb_arg": a.hbm_gb, "workspace_gb_arg": a.workspace_gb, "beta_gbps_arg": a.beta_gbps})
    (out.parent / "manifest.json").write_text(json.dumps(man, indent=2))

    rows = capacity_rows(target, drafter, int(a.hbm_gb * GiB), int(a.workspace_gb * GiB),
                         a.S, a.P, a.M, a.q, a.beta_gbps * 1e9)
    with out.open("a") as f:
        for i, r in enumerate(rows):
            rec = {"run_id": run_id, "case_id": f"A-{i:03d}", "stage": "A", "status": "ok",
                   "timestamp": man["timestamp"], **row_dict(r)}
            f.write(json.dumps(rec) + "\n")

    print(f"target {target.name}: {target.layers}L h_q={target.h_q} h_kv={target.h_kv} d={target.d} "
          f"G={target.G} kv/token={target.kv_bytes_per_token/1024:.0f} KiB weights={target.weight_bytes/GiB:.1f} GiB "
          f"params={target.params/1e9:.2f}B rev={target.revision}")
    print(f"drafter {drafter.name}: {drafter.layers}L kv/token={drafter.kv_bytes_per_token/1024:.0f} KiB "
          f"weights={drafter.weight_bytes/GiB:.2f} GiB block={drafter.block_size} rev={drafter.revision}")
    print(f"free HBM = {rows[0].free_hbm_bytes/GiB:.1f} GiB  (hbm {a.hbm_gb} - weights - workspace {a.workspace_gb})")
    print(f"{'S':>6} {'P':>5} {'M':>3} {'shared GiB':>10} {'per-req MiB':>11} {'B_res':>5} {'B_off':>5} {'extra':>5} {'link ms/step':>12}  note")
    for r in rows:
        print(f"{r.S:>6} {r.P:>5} {r.M:>3} {r.shared_bytes/GiB:>10.2f} {r.per_req_bytes/2**20:>11.0f} "
              f"{r.B_max_resident:>5} {r.B_max_offloaded:>5} {r.extra_requests:>5} "
              f"{r.link_s_per_step_per_prefix*1e3:>12.1f}  {r.note}")
    print(f"wrote {len(rows)} rows to {out}; manifest at {out.parent/'manifest.json'}")


if __name__ == "__main__":
    main()

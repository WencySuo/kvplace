"""Print the report tables from results/results.jsonl. Reads only; never recomputes a measurement.

    python -m kvdiag.summarize --results results/results.jsonl
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # allow `python kvdiag/summarize.py` as well as `python -m kvdiag.summarize`

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

GiB = 1 << 30


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def stage_a(rows):
    a = [r for r in rows if r["stage"] == "A"]
    if not a:
        return print("Stage A: no rows")
    print(f"\nStage A capacity ({a[0]['target']} + {a[0]['drafter']}, free HBM {a[0]['free_hbm_bytes']/GiB:.1f} GiB)")
    print(f"{'S':>6} {'P':>5} {'M':>3} {'shared GiB':>10} {'B_res':>5} {'B_off':>5} {'extra':>5}  note")
    for r in a:
        print(f"{r['S']:>6} {r['P']:>5} {r['M']:>3} {r['shared_bytes']/GiB:>10.1f} {r['B_max_resident']:>5} "
              f"{r['B_max_offloaded']:>5} {r['extra_requests']:>5}  {r['note']}")


def stage_l(rows):
    l = [r for r in rows if r["stage"] == "L"]
    if not l:
        return print("\nTier 0: no rows")
    print("\nTier 0 link")
    for r in l:
        if r["test"] == "memcpy_h2d":
            print(f"  memcpy  chunk {r['chunk_bytes']/2**20:7.2f} MiB  {r['gbps']:7.1f} GB/s")
        elif r["test"] == "sm_read":
            print(f"  sm_read {r['src']:4s} blocks {r['blocks']:5d}  {r['gbps']:7.1f} GB/s")
        elif r["test"] == "sm_read_shared":
            print(f"  shared  footprint {r['footprint_bytes']/2**20:4.0f} MiB blocks {r['blocks']:4d}  {r['gbps']:7.1f} GB/s aggregate")
        elif r["test"] == "chase":
            print(f"  chase   {r['src']:4s}  {r['ns_per_hop']:7.1f} ns/hop")
        elif r["test"] == "nvbandwidth":
            print(f"  nvbandwidth h2d_ce {r.get('h2d_ce_gbps')} GB/s  ({r['status']})")


def stage_c(rows):
    c = [r for r in rows if r["stage"] == "C"]
    if not c:
        return print("\nStage C: no rows")
    by = defaultdict(list)
    for r in c:
        by[(r["model"], r["S"], r["R"], r["arm"], r["status"])].append(r["median_ms"])
    med = {k: statistics.median(v) for k, v in by.items()}
    print("\nStage C attention (median over reps of per-layer median ms; ratio vs hbm same status)")
    print(f"{'model':>9} {'S':>6} {'R':>5} {'arm':>20} {'status':>9} {'ms/layer':>9} {'GB/s eff':>9} {'TFLOP/s':>8} {'ratio':>6}")
    for (m, S, R, arm, st), ms in sorted(med.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][4], kv[0][3])):
        base = med.get((m, S, R, "hbm", st))
        r0 = next(r for r in c if (r["model"], r["S"], r["R"], r["arm"], r["status"]) == (m, S, R, arm, st))
        ratio = ms / base if base else float("nan")
        print(f"{m:>9} {S:>6} {R:>5} {arm:>20} {st:>9} {ms:>9.3f} {r0['kv_bytes']/(ms*1e-3)/1e9:>9.0f} "
              f"{r0['flops']/(ms*1e-3)/1e12:>8.1f} {ratio:>6.2f}")
    bad = [r for r in c if r["status"] not in ("ok", "predicted")]
    if bad:
        print(f"  {len(bad)} non-ok rows:", {(r['arm'], r['S'], r['R'], r['status']) for r in bad})


def stage_b(rows):
    b = [r for r in rows if r["stage"] == "B"]
    if not b:
        return print("\nStage B: no rows")
    print("\nStage B verification step (all-HBM, no dedup, synthetic ids)")
    print(f"{'S':>6} {'B':>3} {'status':>7} {'step ms':>8} {'attn ms*':>9} {'share*':>7} {'link ms if host':>16}")
    for r in b:
        print(f"{r['S']:>6} {r['B']:>3} {r['status']:>7} {r.get('step_ms', float('nan')):>8.2f} "
              f"{r.get('attn_ms', float('nan')):>9.2f} {r.get('attn_share_profiled') or float('nan'):>7.2f} "
              f"{r['link_ms_if_host']:>16.1f}")
    print("  * from one profiler trace (overhead-inflated); step ms is clean CUDA-event time")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.jsonl")
    a = ap.parse_args(argv)
    rows = load(a.results)
    print(f"{len(rows)} rows; stages: {sorted({r['stage'] for r in rows})}; "
          f"statuses: {sorted({r['status'] for r in rows})}")
    stage_a(rows); stage_l(rows); stage_c(rows); stage_b(rows)


if __name__ == "__main__":
    main()

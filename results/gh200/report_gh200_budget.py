"""Redraw Ellie's offload-budget figure with the GH200 sweeps, in her style, but with the honest
all-resident baseline (ours R=36) as T for the pipeline model and as the 5% band, and the stock
DynamicCache arm shown as a hollow marker. Output: docs/midterm_report/figs/fig_offload_budget_gh200_{8k,16k}.png
"""
import json, os, sys, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, os.path.expanduser("~/cs243-kv-offload/cs2430_project"))
from analyze import pipeline_step, LAYERS

R_DIR = os.path.expanduser("~/cs243-kv-offload/kvplace/results/gh200/results")
OUT = os.path.expanduser("~/cs243-kv-offload/kvplace/docs/midterm_report/figs")
COLORS = {"ordinary": "tab:blue", "speculative": "tab:orange"}
BW_GH200 = 341.5   # measured pinned H2D at 1 GiB, linkbench_NVIDIA_GH200_480GB.csv
TOL = 0.05
RS = [36, 32, 27, 18, 9, 0]

def rows(f): return [json.loads(l) for l in open(f)]
def mode(r): return "speculative" if r["block_size"] > 1 else "ordinary"
def step(r): return r["tpot_ms"] * (r["mean_accept"] if r["block_size"] > 1 else 1.0)
def n_ctx(r): return r["prompt_tokens"] + r["output_tokens"] / 2.0

for ctx in ("8k", "16k"):
    rs = rows(f"{R_DIR}/sweep_{ctx}.jsonl")
    b = rs[0]["kv_bytes_per_token"]
    f_full = (LAYERS - 0) / LAYERS * 0.99  # R=0 keeps a 64-token tail resident
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
    base_rows = [r for r in rs if r["arm"] == "ours" and r["resident_layers"] == 36 and mode(r) == "ordinary"]
    baseline = st.fmean(r["tpot_ms"] for r in base_rows)
    ax.axhspan(baseline, baseline * (1 + TOL), color="0.6", alpha=0.35, lw=0, zorder=0)
    for m in ("ordinary", "speculative"):
        c = COLORS[m]
        # measured: ours at each R
        xs, ys, es = [], [], []
        for R in RS:
            g = [r["tpot_ms"] for r in rs if r["arm"] == "ours" and r["resident_layers"] == R and mode(r) == m]
            xs.append((LAYERS - R) / LAYERS * 0.99); ys.append(st.fmean(g)); es.append(st.stdev(g) / len(g) ** 0.5 if len(g) > 1 else 0)
        ax.errorbar(xs, ys, yerr=es, fmt="o", color=c, capsize=3, ms=6, zorder=3)
        # stock DynamicCache arm (hbm): hollow marker at f=0
        hb = [r["tpot_ms"] for r in rs if r["arm"] == "hbm" and mode(r) == m]
        ax.errorbar([0.0], [st.fmean(hb)], yerr=[st.stdev(hb) / len(hb) ** 0.5], fmt="D", mfc="white", mec=c, color=c, capsize=3, ms=6, zorder=3)
        # pipeline model with T = all-resident (R=36) step of the same prompt, link = measured GH200 H2D
        per_prompt = {r["prompt_idx"]: r for r in rs if r["arm"] == "ours" and r["resident_layers"] == 36 and mode(r) == m}
        fx, fy = [], []
        for R in range(LAYERS, -1, -1):
            vals = []
            for p, r in per_prompt.items():
                T, tau, n = step(r), (r["mean_accept"] if r["block_size"] > 1 else 1.0), n_ctx(r)
                t = n * b / LAYERS / (BW_GH200 * 1e9) * 1e3
                vals.append(pipeline_step(T, R, t, 2) / tau)
            fx.append((LAYERS - R) / LAYERS * 0.99); fy.append(float(np.mean(vals)))
        ax.plot(fx, fy, color=c, lw=1.4, ls="--", zorder=2)
    ax.set_yscale("log"); ax.set_xlabel("fraction of KV cache offloaded"); ax.set_ylabel("ms per generated token")
    ax.set_xlim(-0.03, 1.03); ax.grid(True, which="both", alpha=0.3)
    handles = [Line2D([], [], color=COLORS[m], marker="o", ls="", label=m) for m in ("ordinary", "speculative")]
    handles += [Line2D([], [], color="0.3", ls="--", label="pipeline model"),
                Line2D([], [], color="0.3", marker="D", mfc="white", ls="", label="stock cache"),
                Patch(facecolor="0.6", alpha=0.35, label=f"baseline (+{int(TOL*100)}%)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_offload_budget_gh200_{ctx}.png"); plt.close(fig)
    print(ctx, "baseline", round(baseline, 1), "written")

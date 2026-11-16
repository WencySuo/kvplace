"""Midterm-report figures from the GH200 results (and Ellie's L40S midterm sweep).
Usage: python report_figs.py   -> results/gh200/figs_report/fig*.{png,pdf}
"""
import json, csv, os, collections, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figs_report")
ELLIE = os.path.expanduser("~/code/cs2430_project/results")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 200, "savefig.bbox": "tight",
})

def rows(path):
    return [json.loads(l) for l in open(path)]

def group(rs, key):
    g = collections.defaultdict(list)
    for r in rs:
        g[key(r)].append(r)
    return g

def step_ms(r):  # wall time per verify step (ms); tpot is per output token
    return r["tpot_ms"] * (r["mean_accept"] if r["block_size"] > 1 else 1.0)

def mean(xs): return st.fmean(xs)
def sd(xs): return st.pstdev(xs) if len(xs) > 1 else 0.0

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1: offload cost vs resident layers, GH200 vs L40S
def fig1():
    gh = rows(f"{R}/sweep_16k.jsonl")
    l4 = rows(f"{ELLIE}/midterm_16k.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    for ax, bs, title in zip(axes, (1, 16), ("Ordinary decoding (block 1)", "Speculative verify (DFlash block 16)")):
        for i, (rs, color, label) in enumerate(((gh, BLUE, "GH200 (C2C, 350 GB/s)"), (l4, ORANGE, "L40S (PCIe, 13.5 GB/s)"))):
            g = group([r for r in rs if r["arm"] == "ours" and r["block_size"] == bs], lambda r: r["resident_layers"])
            xs = sorted(g)
            ys = [mean([r["tpot_ms"] for r in g[x]]) for x in xs]
            es = [sd([r["tpot_ms"] for r in g[x]]) for x in xs]
            ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)], color=color, alpha=0.12, lw=0)
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=4, label=label)
            hb = [r["tpot_ms"] for r in rs if r["arm"] == "hbm" and r["block_size"] == bs]
            ax.plot([36], [mean(hb)], marker="D", ms=6, mfc=SURF, mec=color, mew=1.6, ls="none")
            ax.annotate(f"{ys[0]:.0f}", (xs[0], ys[0]), xytext=(5, 3), textcoords="offset points", color=color, fontsize=8)
            ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(-16, 7 if i == 0 else -11), textcoords="offset points", color=color, fontsize=8)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xticks([0, 9, 18, 27, 36]); ax.set_ylim(0, None)
        ax.set_xlabel("layers resident in HBM (of 36)")
    axes[0].set_ylabel("ms per output token, 16k context")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="D", ms=6, mfc=SURF, mec=INK2, mew=1.4, ls="none")); l.append("stock HBM cache (hbm arm)")
    axes[1].legend(h, l, loc="upper right")
    fig.text(0.01, -0.02, "x = 0: entire verifier KV cache in CPU memory;  x = 36: nothing offloaded.  Band = ±1 s.d. over 6 SPEED-Bench prompts.", fontsize=7.5, color=INK2)
    save(fig, "fig1_offload_cost_vs_resident")

# ---------------------------------------------------------------- fig 2: effective host-read bandwidth by access pattern
def fig2():
    sm = list(csv.DictReader(open(f"{R}/sm_read_NVIDIA_GH200_480GB.csv")))
    def get(test, place=None, blocks=None):
        for r in sm:
            if r["test"] == test and (place is None or r["place"] == place) and (blocks is None or int(r["blocks"]) == blocks):
                return float(r["gbps"])
    items = [
        ("DMA copy host→HBM (copy engine, what the copy pipeline uses)", get("dma_h2d")),
        ("SM bulk read of host memory inside a kernel (264 blocks)", get("sm_read", blocks=264)),
        ("attention q=1, fused GQA, K/V read from host", get("sdpa_q1", "host", 16384)),
        ("attention q=16, fused GQA, K/V read from host", get("sdpa_q16_gqa", "host", 16384)),
        ("attention q=16 via repeat_kv expand (transformers default), K/V in host", get("sdpa_q16_repeatkv", "host", 16384)),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.6), layout="constrained")
    ys = list(range(len(items)))[::-1]
    ax.barh(ys, [v for _, v in items], color=BLUE, height=0.55)
    for y, (lab, v) in zip(ys, items):
        ax.text(v + 5, y, f"{v:.0f} GB/s", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(ys); ax.set_yticklabels([l for l, _ in items], color=INK, fontsize=8)
    ax.set_xlim(0, 400); ax.grid(axis="y", visible=False)
    ax.set_xlabel("effective bandwidth reading one 64 MB KV layer (16k tokens) from Grace memory, GB/s")
    ax.set_title("Reading host memory: the access pattern, not the link, sets the speed", color=INK, loc="left")
    save(fig, "fig2_host_read_bandwidth_by_access_pattern")

# ---------------------------------------------------------------- fig 3: where the time goes (decode-only profile)
def fig3():
    prof = rows(f"{R}/profile_16k.jsonl")
    def comp(r):
        n = r["steps_clean"]; att = exp = cat = 0.0
        for k, ms, _cnt in r["top_kernels"]:  # already ms per step

            if "sdpa" in k: att += ms
            elif "elementwise_kernel<128, 4" in k and "nocast" in k: exp += ms  # repeat_kv expand copy
            elif "CatArrayBatchedCopy" in k: cat += ms
        gpu = r["gpu_ms_per_step"]
        wall = r["tpot_ms_clean"] * (r["mean_accept"] if r["block_size"] > 1 else 1)  # unprofiled step of the same run
        return dict(att=att, exp=exp, cat=cat, other=max(gpu - att - exp - cat, 0), gpu=gpu, wall=wall, memcpy=r["memcpy_ms_per_step"], launches=r["launches_per_step"])
    order = [("ours", 36, 1, "ordinary · all in HBM"), ("ours", 0, 1, "ordinary · copy pipeline, all offloaded"), ("ours", 0, 1, None),
             ("zerocopy", 0, 1, "ordinary · zero-copy, all offloaded"),
             ("ours", 36, 16, "speculative · all in HBM"), ("ours", 0, 16, "speculative · copy pipeline, all offloaded"), ("zerocopy", 0, 16, "speculative · zero-copy, all offloaded")]
    order = [o for o in order if o[3]]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    ys = []; y = 0
    for i, (arm, R_, bs, lab) in enumerate(order):
        if i == 3: y -= 0.6  # gap between the two groups
        r = next(r for r in prof if r["arm"] == arm and r["resident_layers"] == R_ and r["block_size"] == bs)
        c = comp(r); ys.append((y, lab))
        ax.barh(y, c["wall"], color="none", edgecolor=INK2, lw=1.2, height=0.62, zorder=3)
        left = 0
        for key, col in (("att", BLUE), ("exp", ORANGE), ("cat", AQUA), ("other", MUTED)):
            ax.barh(y, c[key], left=left, color=col, height=0.62, lw=0, zorder=2); left += c[key]
        txt = f"wall {c['wall']:.0f} ms · GPU {c['gpu']:.0f} ms ({c['gpu']/c['wall']*100:.0f}% busy) · {c['launches']:.0f} launches"
        if c["memcpy"] > 0.5: txt += f" · +{c['memcpy']:.1f} ms DMA overlapped"
        ax.text(c["wall"] + 1.5, y, txt, va="center", fontsize=7.5, color=INK)
        y -= 1
    ax.set_yticks([y for y, _ in ys]); ax.set_yticklabels([l for _, l in ys], color=INK, fontsize=8)
    ax.grid(axis="y", visible=False); ax.set_xlim(0, 150)
    ax.set_xlabel("ms per verify step at 16k context (decode only, prefill subtracted; one prompt, ~10 steps)")
    ax.set_title("GPU kernels fill only part of each step: the loop is CPU-bound", color=INK, loc="left")
    fig.legend(handles=[Patch(color=BLUE, label="attention kernel"), Patch(color=ORANGE, label="GQA expand copy (repeat_kv)"),
                        Patch(color=AQUA, label="cache torch.cat"), Patch(color=MUTED, label="other GPU (GEMMs, norms)"),
                        Patch(facecolor="none", edgecolor=INK2, label="wall-clock step")], loc="outside lower center", ncol=5, fontsize=7.5)
    save(fig, "fig3_where_the_time_goes")

# ---------------------------------------------------------------- fig 4: zero-copy vs copy pipeline vs context
def fig4():
    ctxs = [8, 16, 32]
    series = (("hbm", None, MUTED, "stock HBM cache (hbm arm)"), ("ours", 0, ORANGE, "copy pipeline, all offloaded"), ("zerocopy", 0, BLUE, "zero-copy, all offloaded"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), layout="constrained")
    panels = ((1, step_ms, "ms per step", "Ordinary decoding step", "{:.0f}"), (16, step_ms, "ms per step", "Speculative verify step", "{:.0f}"), (1, lambda r: r["ttft_s"], "seconds", "Prefill (time to first token)", "{:.1f}"))
    for ax, (bs, metric, ylab, title, fmt) in zip(axes, panels):
        for arm, R_, col, lab in series:
            ys, es = [], []
            for c in ctxs:
                rs = [r for r in rows(f"{R}/zc_{c}k.jsonl") if r["arm"] == arm and r["block_size"] == bs and (R_ is None or r["resident_layers"] == R_)]
                ys.append(mean([metric(r) for r in rs])); es.append(sd([metric(r) for r in rs]))
            ax.errorbar(ctxs, ys, yerr=es, color=col, lw=2, marker="o", ms=4, capsize=2, label=lab)
            ax.annotate(fmt.format(ys[-1]), (ctxs[-1], ys[-1]), xytext=(5, -3), textcoords="offset points", fontsize=8, color=col)
        ax.set_xticks(ctxs); ax.set_xticklabels([f"{c}k" for c in ctxs]); ax.set_xlabel("context length")
        ax.set_xlim(6, 36); ax.set_title(title, color=INK, loc="left"); ax.set_ylabel(ylab); ax.set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3)
    save(fig, "fig4_zerocopy_vs_copy_vs_context")

# ---------------------------------------------------------------- fig 5: eviction policy v1, HBM budget vs step time
def fig5():
    sw = rows(f"{R}/sweep_16k.jsonl"); ev = rows(f"{R}/evict_16k.jsonl"); zc = rows(f"{R}/zc_16k.jsonl")
    # zc_16k and evict_16k share the same 4 prompts; only the all-resident row comes from the 6-prompt sweep
    pol = [  # (label, HBM KV MB per request, selector, source), ordered by HBM budget
        ("all on host, 64-token tail", 155, lambda r: r["arm"] == "ours" and r["resident_layers"] == 0 and r.get("tail_keep", 64) == 64, zc, BLUE),
        ("last 2048 tokens of every layer in HBM", 448, lambda r: r["arm"] == "ours" and r["resident_layers"] == 0 and r.get("tail_keep") == 2048, ev, BLUE),
        ("first 18 layers resident", 1352, lambda r: r["arm"] == "ours" and r["resident_layers"] == 18, zc, ORANGE),
        ("last 18 layers resident", 1352, lambda r: r["arm"] == "ours" and r["resident_layers"] == 18 and r.get("resident_mode") == "last", ev, ORANGE),
        ("last 8192 tokens of every layer in HBM", 1354, lambda r: r["arm"] == "ours" and r["resident_layers"] == 0 and r.get("tail_keep") == 8192, ev, BLUE),
        ("all 36 layers resident (nothing offloaded; 6-prompt sweep)", 2400, lambda r: r["arm"] == "ours" and r["resident_layers"] == 36, sw, ORANGE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained", sharey=True)
    ys = list(range(len(pol)))[::-1]
    for ax, bs, title in zip(axes, (16, 1), ("Speculative (block 16)", "Ordinary (block 1)")):
        for y, (lab, mb, sel, src, col) in zip(ys, pol):
            rs = [r for r in src if r["block_size"] == bs and sel(r)]
            v = mean([step_ms(r) for r in rs]); e = sd([step_ms(r) for r in rs])
            ax.plot([v - e, v + e], [y, y], color=col, lw=1, alpha=0.5)
            ax.plot(v, y, "o", color=col, ms=7, mec=SURF, mew=1, zorder=3)
            ax.text(v + 1.2, y + 0.22, f"{v:.0f}", fontsize=7.5, color=col)
        ax.set_title(title, color=INK, loc="left"); ax.grid(axis="y", visible=False)
        ax.set_xlabel("ms per step")
    fig.supxlabel("copy pipeline, 16k context, mean ±1 s.d. over 4-6 prompts; HBM MB = resident KV + staging buffers per request", fontsize=7.5, color=INK2)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([f"{lab}\n{mb:,} MB HBM" for lab, mb, *_ in pol], fontsize=7.5, color=INK)
    axes[0].set_xlim(50, 70); axes[1].set_xlim(35, 80)
    axes[1].legend(handles=[Patch(color=BLUE, label="token-window policies"), Patch(color=ORANGE, label="layer policies")], loc="center left", bbox_to_anchor=(0.0, 0.3), fontsize=7.5)
    save(fig, "fig5_eviction_budget_vs_step")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()

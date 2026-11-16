"""budget_both.py -- Figure 1 of the midterm report: L40S and GH200 offload sweeps on one axis.

y = time per generated token divided by ordinary decoding with nothing offloaded on the
same machine (L40S: stock cache; GH200: all-resident configuration of our cache, see
report).  Points: measured mean over six prompts (bars: standard error).  Lines: Ellie's
parameter-free pipeline model (analyze.py) at that machine's measured link bandwidth.

Run from this directory with the cs2430_project venv:
  ~/code/cs2430_project/.venv/bin/python budget_both.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HOME = os.path.expanduser("~")
ELLIE = os.path.join(HOME, "code", "cs2430_project")
GH200 = os.path.join(HOME, "code", "kvplace", "results", "gh200", "results")
sys.path.insert(0, ELLIE)
import analyze  # noqa: E402  (Ellie's model + loaders)

COLOR = {"GH200": "#2a78d6", "L40S": "#eb6834"}  # validated palette: blue, orange
MARK = {"ordinary": dict(marker="o", mfc=None), "speculative": dict(marker="D", mfc="white")}
LS = {"ordinary": "-", "speculative": "--"}
TOL = 0.05


def machine(name, runs, linkbench, base_arm):
    df = analyze.load_runs([runs])
    _, bw = analyze.link_bandwidth(linkbench)
    b = int(df.kv_bytes_per_token.iloc[0])
    model = analyze.Model(df, b, slots=2)
    base = df[(df.arm == base_arm) & (df.R == 36)]
    if base_arm == "ours":  # GH200: model T from the all-resident configuration of our cache
        model.prompts = {m: g[["step_ms", "tau", "n_ctx"]].to_numpy() for m, g in base.groupby("mode")}
    baseline = float(base[base["mode"] == "ordinary"].tpot_ms.mean())
    pts = df[(df.arm == "ours") | ((df.arm == "hbm") & (base_arm == "hbm"))]
    if base_arm == "ours":
        pts = df[df.arm == "ours"]
    meas = analyze.measured_curves(pts)
    return dict(name=name, bw=bw, model=model, baseline=baseline, meas=meas)


def panel(ax, machines, title):
    ax.axhspan(1.0, 1 + TOL, color="0.6", alpha=0.3, lw=0, zorder=0)
    for m in machines:
        c = COLOR[m["name"]]
        for mode in ("ordinary", "speculative"):
            f, y = m["model"].curve(mode, m["bw"])
            ax.plot(f, y / m["baseline"], color=c, lw=1.2, ls=LS[mode], alpha=0.8, zorder=2)
            s = m["meas"][mode]
            mk = MARK[mode]
            ax.errorbar(s.f, s.t / m["baseline"], yerr=s.se / m["baseline"], color=c, ls="none",
                        marker=mk["marker"], ms=6, mfc=mk["mfc"] or c, mec=c, mew=1.4,
                        capsize=2, lw=1, zorder=3)
            last = s.iloc[-1]
            ax.annotate(f"{last.t / m['baseline']:.2f}x", (last.f, last.t / m["baseline"]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=c)
    ax.set_yscale("log")
    ax.set_yticks([0.75, 1, 1.5, 2, 3, 5])
    ax.set_yticklabels(["0.75", "1", "1.5", "2", "3", "5"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(-0.03, 1.12)
    ax.set_xlabel("fraction of KV cache offloaded")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="0.9", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    out = os.path.dirname(os.path.abspath(__file__))
    cfg = [
        ("8K", os.path.join(ELLIE, "results", "midterm_8k.jsonl"), os.path.join(GH200, "sweep_8k.jsonl")),
        ("16K", os.path.join(ELLIE, "results", "midterm_16k.jsonl"), os.path.join(GH200, "sweep_16k.jsonl")),
    ]
    lb_l40s = os.path.join(ELLIE, "results", "linkbench_NVIDIA_L40S.csv")
    lb_gh200 = os.path.join(GH200, "linkbench_NVIDIA_GH200_480GB.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, constrained_layout=True)
    for ax, (ctx, l40s_runs, gh_runs) in zip(axes, cfg):
        ms = [machine("L40S", l40s_runs, lb_l40s, "hbm"), machine("GH200", gh_runs, lb_gh200, "ours")]
        panel(ax, ms, f"{ctx} context")
        for m in ms:
            print(ctx, m["name"], f"link {m['bw']:.1f} GB/s, baseline {m['baseline']:.1f} ms/token")
    axes[0].set_ylabel("time per token / all-GPU ordinary")
    handles = [
        Line2D([], [], color=COLOR["L40S"], marker="o", ls="-", label="L40S, ordinary"),
        Line2D([], [], color=COLOR["L40S"], marker="D", mfc="white", ls="--", label="L40S, speculative"),
        Line2D([], [], color=COLOR["GH200"], marker="o", ls="-", label="GH200, ordinary"),
        Line2D([], [], color=COLOR["GH200"], marker="D", mfc="white", ls="--", label="GH200, speculative"),
        Patch(color="0.6", alpha=0.3, label="baseline +5%"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=8, frameon=False,
               handlelength=2.2, columnspacing=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_budget_both.{ext}"), dpi=250)
    print("wrote", os.path.join(out, "fig_budget_both.{png,pdf}"))


if __name__ == "__main__":
    main()

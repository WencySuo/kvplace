"""Figures from results/results.jsonl. Reads only JSONL; never recomputes a measurement.

    python -m kvdiag.plot --results results/results.jsonl --out results/plots

fig_capacity.png   Stage A: feasible batch under resident vs offloaded prefix, by M, per (S, P)
fig_attention.png  Stage C: roofline (achieved TFLOP/s vs intensity) + ratio t_arm/t_hbm vs R
fig_profile.png    Stage B: verification step time split, with link time if prefix were on host
Measured rows are solid; rows with status "predicted" are dashed and labeled as such.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # allow `python kvdiag/plot.py` as well as `python -m kvdiag.plot`

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (validated palette, light surface). Color follows the arm, never rank.
ARM_COLOR = {
    "hbm": "#2a78d6", "host_direct": "#eb6834", "host_direct_chunked": "#1baf7a",
    "host_staged": "#eda100", "host_copy": "#e87ba4",
}
ARM_LABEL = {
    "hbm": "HBM prefix", "host_direct": "host, direct (stock kernel)",
    "host_direct_chunked": "host, direct, chunked + merge", "host_staged": "host, staged memcpy + merge",
    "host_copy": "host, copy then compute (serial)",
}
INK, INK2, GRID = "#1f1f1e", "#5f5f5c", "#e4e4e0"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 150, "lines.linewidth": 2, "lines.markersize": 5,
})


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _style(row):
    pred = row.get("status") == "predicted"
    return dict(linestyle="--" if pred else "-", marker="" if pred else "o", alpha=0.9 if pred else 1.0)


# ---------------------------------------------------------------- fig_capacity

def fig_capacity(rows, out: Path):
    a = [r for r in rows if r["stage"] == "A" and r["status"] == "ok"]
    if not a:
        return None
    combos = sorted({(r["S"], r["P"]) for r in a})
    fig, axes = plt.subplots(1, len(combos), figsize=(3.2 * len(combos), 3.2), sharey=True)
    axes = list(axes) if len(combos) > 1 else [axes]
    for ax, (S, P) in zip(axes, combos):
        sub = sorted([r for r in a if (r["S"], r["P"]) == (S, P)], key=lambda r: r["M"])
        xs = range(len(sub))
        w = 0.38
        ax.bar([x - w / 2 for x in xs], [r["B_max_resident"] for r in sub], w, color=ARM_COLOR["hbm"],
               label="prefixes resident in HBM", linewidth=0)
        ax.bar([x + w / 2 for x in xs], [r["B_max_offloaded"] for r in sub], w, color=ARM_COLOR["host_staged"],
               label="prefixes offloaded (zero-overhead bound)", linewidth=0)
        for x, r in zip(xs, sub):
            if not r["fits_resident"]:
                ax.text(x - w / 2, 1, "does\nnot fit", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(list(xs), [str(r["M"]) for r in sub])
        ax.set_xlabel("distinct warm prefixes M")
        ax.set_title(f"S={S//1024}K prefix, P={P} private", fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("max concurrent requests B")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    free = a[0]["free_hbm_bytes"] / 2**30
    fig.suptitle(f"Capacity: {a[0]['target']} + {a[0]['drafter']}, free HBM {free:.0f} GiB, q={a[0]['q']} "
                 "(config-derived; drafter KV not deduplicated)", fontsize=9, color=INK)
    fig.tight_layout()
    p = out / "fig_capacity.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- fig_attention

def fig_attention(rows, out: Path, S: int, model: str):
    c = [r for r in rows if r["stage"] == "C" and r["S"] == S and r["model"] == model
         and r["status"] in ("ok", "predicted")]
    if not c:
        return None
    # prefer measured rows; predicted rows only for arms/R with no measurement
    by = defaultdict(dict)
    for r in sorted(c, key=lambda r: r["status"] == "predicted"):
        by[r["arm"]].setdefault((r["R"], r["status"]), r)
    hbm = {k[0]: v for k, v in by.get("hbm", {}).items()}
    machine = next((r.get("machine") for r in c if r.get("machine")), {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    # roofs
    if machine:
        import numpy as np
        I = np.logspace(0, 4.5, 100)
        F, Bh, bd, bs = machine["F_roof"], machine["B_hbm"], machine["beta_dma"], machine["beta_sm"]
        ax1.plot(I, np.minimum(Bh * I, F) / 1e12, color=INK2, lw=1, ls=":", label="HBM roof")
        ax1.plot(I, np.minimum(bd * I, F) / 1e12, color=INK2, lw=1, ls="-.", label="host roof (beta_dma)")
    for arm in ARM_COLOR:
        for status in ("ok", "predicted"):
            pts = sorted([r for (R, s), r in by.get(arm, {}).items() if s == status], key=lambda r: r["R"])
            if not pts:
                continue
            lbl = ARM_LABEL[arm] + (" (predicted)" if status == "predicted" else "")
            ax1.plot([r["intensity"] for r in pts], [r["achieved_tflops"] for r in pts],
                     color=ARM_COLOR[arm], label=lbl, **_style(pts[0]))
            if arm != "hbm":
                ratio = [r["median_ms"] / hbm[r["R"]]["median_ms"] for r in pts if r["R"] in hbm]
                Rs = [r["R"] for r in pts if r["R"] in hbm]
                ax2.plot(Rs, ratio, color=ARM_COLOR[arm], label=lbl, **_style(pts[0]))
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("arithmetic intensity G*R  [FLOP per K/V byte]")
    ax1.set_ylabel("achieved TFLOP/s (one layer)")
    ax1.set_title("Roofline by arm", fontsize=9, color=INK)
    ax2.set_xscale("log", base=2)
    ax2.axhline(1.1, color=INK2, lw=1, ls=":")
    ax2.text(ax2.get_xlim()[0] if ax2.lines else 1, 1.12, "1.1 = crossed", fontsize=7, color=INK2)
    ax2.set_xlabel("query rows R = B * q")
    ax2.set_ylabel("t_arm / t_hbm")
    ax2.set_yscale("log")
    sec = ax2.secondary_xaxis("top", functions=(lambda R: R / 16, lambda B: B * 16))
    sec.set_xlabel("requests B at q=16 (DFlash block)", fontsize=8, color=INK2)
    if machine:
        Rstar = machine["F_roof"] / (machine["beta_dma"] * (c[0]["h_q"] // c[0]["h_kv"]))
        ax2.axvline(Rstar, color=INK2, lw=1, ls="--")
        ax2.text(Rstar, ax2.get_ylim()[1] if ax2.lines else 10, f" R*={Rstar:.0f}", fontsize=7, color=INK2, va="top")
    ax2.set_title("Host arms relative to HBM", fontsize=9, color=INK)
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=7.5)
    src = machine.get("source", "n/a")
    fig.suptitle(f"{model}, S={S}, one layer, {c[0].get('gpu') or 'GPU n/a'}; predictions use {src} constants", fontsize=9, color=INK)
    fig.tight_layout()
    p = out / f"fig_attention_{model}_S{S}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- fig_profile

def fig_profile(rows, out: Path):
    b = [r for r in rows if r["stage"] == "B" and r["status"] == "ok"]
    if not b:
        return None
    b = sorted(b, key=lambda r: (r["S"], r["B"]))
    labels = [f"B={r['B']}, S={r['S']//1024}K" for r in b]
    attn = [r["attn_ms"] for r in b]
    other = [r["step_ms"] - r["attn_ms"] for r in b]
    link = [r["link_ms_if_host"] for r in b]
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(b) + 1.4))
    ys = range(len(b))
    ax.barh(ys, attn, color=ARM_COLOR["hbm"], label="attention kernels", linewidth=0)
    ax.barh(ys, other, left=[a_ + 0.02 for a_ in attn], color="#c9c9c5", label="rest of verification step", linewidth=0)
    ax.scatter(link, ys, marker="|", s=120, color=ARM_COLOR["host_direct"], label="link time if prefix on host (S*kv/beta)", zorder=3)
    ax.set_yticks(list(ys), labels)
    ax.set_xlabel("ms per verification step (q=16), all layers")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("Where the verification step goes (measured; prefill excluded)", fontsize=9, color=INK)
    fig.tight_layout()
    p = out / "fig_profile.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.jsonl")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--model", default="qwen3-8b")
    a = ap.parse_args(argv)
    rows = load(Path(a.results))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    made = [fig_capacity(rows, out), fig_profile(rows, out)]
    for S in sorted({r["S"] for r in rows if r["stage"] == "C"}):
        made.append(fig_attention(rows, out, S, a.model))
    for p in made:
        if p:
            print("wrote", p)


if __name__ == "__main__":
    main()

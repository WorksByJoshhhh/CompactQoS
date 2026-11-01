"""
Figures and summary for the Review 3 experiments (run_review3_experiments.py).

    python plot_review3.py                    # reads review3_runs.csv
    python plot_review3.py review3_quick.csv  # or any other results file
Writes figures/fig9_dpu_speed.png, figures/fig10_scenarios.png, review3_summary.csv
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = sys.argv[1] if len(sys.argv) > 1 else "review3_runs.csv"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

COLORS = {"fcfs": "#2a78d6", "fairshare": "#eb6834", "cqos_dpu": "#1baf7a"}
LABELS = {"fcfs": "FCFS", "fairshare": "Fair-share", "cqos_dpu": "CompactQoS"}
POLS = ["fcfs", "fairshare", "cqos_dpu"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})

d = pd.read_csv(RUNS)
metrics = ["calm_p99_ms_worst", "calm_p99_ms_mean", "calm_slow_s_mean", "flood_goodput",
           "dpu_share_flooder", "decision_us"]
summary = d.groupby(["experiment", "scenario", "policy"])[metrics].agg(["mean", "min", "max"]).round(2)
summary.to_csv("review3_summary.csv")
print("wrote review3_summary.csv")

# how often CompactQoS was best, per scenario (same seed compared)
for (e, sc), g in d.groupby(["experiment", "scenario"]):
    for m in ["calm_p99_ms_worst", "calm_slow_s_mean"]:
        p = g.pivot(index="seed", columns="policy", values=m)
        if set(POLS) <= set(p.columns):
            wins = int((p["cqos_dpu"] <= p[["fcfs", "fairshare"]].min(axis=1)).sum())
            print(f"{e} {sc:<20} {m:<18} CompactQoS best in {wins}/{len(p)} runs")


def panel(ax, sub, scenarios, metric, ylabel, log):
    width = 0.8 / len(POLS)
    top = 0
    for k, pol in enumerate(POLS):
        for i, sc in enumerate(scenarios):
            vals = sub[(sub.scenario == sc) & (sub.policy == pol)][metric]
            if vals.empty:
                continue
            x = i + (k - 1) * width
            m = vals.mean()
            ax.bar(x, m, width * 0.92, color=COLORS[pol], label=LABELS[pol] if i == 0 else None)
            ax.scatter([x] * len(vals), vals, s=14, color=INK, zorder=3, linewidths=0)
            top = max(top, vals.max())
            txt = f"{m:.0f}" if (log and m >= 10) else f"{m:.2f}" if m < 10 else f"{m:.1f}"
            ax.annotate(txt, (x, max(m, vals.max())), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.5, max(top, 1) * 10)
    else:
        ax.set_ylim(0, max(top, 0.1) * 1.3)


def two_panel(sub, scenarios, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    panel(axes[0], sub, scenarios, "calm_p99_ms_worst", "worst calm-tenant p99, ms (log scale)", True)
    panel(axes[1], sub, scenarios, "calm_slow_s_mean", "seconds in slowdown per calm tenant", False)
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    n = sub.seed.nunique()
    fig.suptitle(f"{title} (bars = mean, dots = {n} individual runs)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close(fig)
    print("wrote", fname)


if "E1" in set(d.experiment):
    order = [s for s in ["DPU 2x faster", "Baseline DPU", "DPU 2x slower"] if s in set(d.scenario)]
    two_panel(d[d.experiment == "E1"], order,
              "Real RocksDB, 3 tenants: sensitivity to the emulated DPU speed", "fig9_dpu_speed.png")

others = d[d.experiment.isin(["E2", "E3"])]
if not others.empty:
    order = [s for s in ["6 tenants, 1 noisy", "4 tenants, 2 noisy"] if s in set(others.scenario)]
    two_panel(others, order, "Real RocksDB: more tenants and two noisy tenants", "fig10_scenarios.png")

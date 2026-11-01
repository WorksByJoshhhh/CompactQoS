"""
Turn results_runs.csv (+ trace_*.csv) into report figures and summary tables.

    python make_plots.py            # reads results_runs.csv, writes figures/ and results_summary.csv
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = sys.argv[1] if len(sys.argv) > 1 else "results_runs.csv"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Colour follows the policy everywhere (validated categorical palette).
COLORS = {"FCFS": "#2a78d6", "Fair-share": "#eb6834", "CompactQoS": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})

d = pd.read_csv(RUNS)

# ------------------------------------------------------------------ tables
metrics = ["calm_p99_ms_worst", "calm_p99_ms_mean", "calm_slow_s_mean", "calm_stop_s_mean",
           "calm_episodes_mean", "flood_goodput", "jain_all", "dpu_util", "host_util",
           "overhead_us_per_decision"]
summary = d.groupby(["n_tenants", "policy"])[metrics].agg(["mean", "std"]).round(3)
summary.to_csv("results_summary.csv")
by_pattern = (d.groupby(["pattern", "n_tenants", "policy"])["calm_p99_ms_worst"]
              .mean().round(2).unstack("policy"))
by_pattern.to_csv("results_by_pattern.csv")
print("wrote results_summary.csv, results_by_pattern.csv")


def grouped_bars(policies, labels, metric, fname, title, ylabel, log=False, fmt="{:.0f}"):
    m = d.groupby(["n_tenants", "policy"])[metric].mean().unstack("policy")
    ns = list(m.index)
    width = 0.8 / len(policies)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for k, (pol, lab) in enumerate(zip(policies, labels)):
        xs = [i + (k - (len(policies) - 1) / 2) * width for i in range(len(ns))]
        vals = [m.loc[n, pol] for n in ns]
        bars = ax.bar(xs, vals, width * 0.92, color=COLORS[lab], label=lab)
        for b, v in zip(bars, vals):
            txt = f"{v:.2f}" if (fmt == "{:.0f}" and v < 10) else fmt.format(v)
            ax.annotate(txt, (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"{n} tenants" for n in ns])
    top = max(m.loc[n, p] for n in ns for p in policies)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.1, top * 8)          # headroom so the legend clears the bars
    else:
        ax.set_ylim(0, top * 1.3)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(loc="upper right", ncol=len(policies))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close(fig)
    print("wrote", fname)


# Fig 1: Experiment A, equal hardware (shared DPU slot only)
grouped_bars(["fcfs", "fairshare", "cqos_dpu"], ["FCFS", "Fair-share", "CompactQoS"],
             "calm_p99_ms_worst", "fig1_expA_calm_p99.png",
             "Worst calm-tenant p99 write latency, shared DPU only (mean of all runs)",
             "p99 write latency, ms (log scale)", log=True)

# Fig 2: Experiment A, time spent in slowdown by calm tenants
grouped_bars(["fcfs", "fairshare", "cqos_dpu"], ["FCFS", "Fair-share", "CompactQoS"],
             "calm_slow_s_mean", "fig2_expA_calm_slowdown.png",
             "Seconds each calm tenant spent in write slowdown (of 120 s)",
             "seconds in slowdown", fmt="{:.2f}")

# Fig 3: Experiment A, flooder goodput (cost of protecting calm tenants)
grouped_bars(["fcfs", "fairshare", "cqos_dpu"], ["FCFS", "Fair-share", "CompactQoS"],
             "flood_goodput", "fig3_expA_flooder_goodput.png",
             "Noisy tenant: fraction of offered writes served",
             "served / offered", fmt="{:.2f}")

# Fig 4: ablation (Experiment A, 3 and 8 tenants)
abl = ["fcfs", "fairshare", "cqos_dpu_no_ledger", "cqos_dpu_no_urgency", "cqos_dpu"]
abl_lab = ["FCFS baseline", "Fair-share baseline", "without ledger",
           "without urgency", "CompactQoS (full)"]
m = d[d.policy.isin(abl)].groupby(["n_tenants", "policy"])["calm_p99_ms_worst"].mean()
ns = [n for n in (3, 8) if n in m.index.get_level_values(0)]
fig, axes = plt.subplots(1, len(ns), figsize=(8.5, 3.4), sharey=True)
if len(ns) == 1:
    axes = [axes]
for ax, n in zip(axes, ns):
    vals = [m.loc[(n, p)] for p in abl]
    cols = ["#2a78d6", "#eb6834", "#86d7b8", "#86d7b8", "#1baf7a"]
    ax.barh(range(len(abl)), vals, color=cols, height=0.7)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f} ms", (v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(abl)))
    ax.set_yticklabels(abl_lab)
    ax.set_title(f"{n} tenants", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("worst calm-tenant p99, ms")
    ax.set_xlim(0, max(vals) * 1.3)
fig.suptitle("Ablation, shared DPU only: which component protects calm tenants?",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_ablation.png"), dpi=200)
plt.close(fig)
print("wrote fig4_ablation.png")

# Fig 5/6: Experiment B (DPU + host fallback). Two charts, not a dual axis.
b_pols = ["fcfs_2slot", "fairshare_2slot", "cqos"]
if set(b_pols) <= set(d.policy):
    grouped_bars(b_pols, ["FCFS", "Fair-share", "CompactQoS"], "host_util",
                 "fig5_expB_host_cpu.png",
                 "Fraction of time the host CPU was running compaction (lower is better)",
                 "host slot busy fraction", fmt="{:.2f}")
    grouped_bars(b_pols, ["FCFS", "Fair-share", "CompactQoS"], "calm_p99_ms_worst",
                 "fig6_expB_calm_p99.png",
                 "Worst calm-tenant p99 with a host fallback slot",
                 "p99 write latency, ms", fmt="{:.2f}")

# Fig 7: L0 over time (one run: 3 tenants, spike, seed 1, shared DPU only)
names = {"fcfs": "FCFS", "fairshare": "Fair-share", "cqos_dpu": "CompactQoS"}
if all(os.path.exists(f"trace_{p}.csv") for p in names):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, (p, lab) in zip(axes, names.items()):
        tr = pd.read_csv(f"trace_{p}.csv")
        ax.axhspan(8, 16, color="#fde9c4", zorder=0)
        ax.axhspan(16, 18, color="#f9c9c0", zorder=0)
        ax.plot(tr.t, tr.tenant0_l0, color="#8f8e88", lw=1.4, label="Tenant A (noisy)")
        ax.plot(tr.t, tr.tenant2_l0, color="#2a78d6", lw=2, label="Tenant C (calm)")
        ax.set_title(lab, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("time, s")
        ax.set_ylim(0, 18)
    axes[0].set_ylabel("L0 files")
    axes[0].text(2, 12, "slowdown zone", fontsize=8, color=MUTED)
    axes[0].text(2, 16.6, "stop zone", fontsize=8, color=MUTED)
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("L0 backlog over one run (flood from t = 24 s to 84 s)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_l0_trace.png"), dpi=200)
    plt.close(fig)
    print("wrote fig7_l0_trace.png")

# Fig 8: real RocksDB runs (rocksdb_harness.py), shared emulated DPU slot, 3 tenants
if os.path.exists("rocksdb_runs.csv"):
    r = pd.read_csv("rocksdb_runs.csv")
    rk = ["calm_p99_ms_worst", "calm_p99_ms_mean", "calm_p50_ms_mean", "calm_slow_s_mean",
          "calm_episodes_mean", "flood_goodput", "jain_all", "dpu_share_flooder", "decision_us"]
    r.groupby("policy")[rk].agg(["mean", "std"]).round(3).to_csv("rocksdb_summary.csv")
    print("wrote rocksdb_summary.csv")

    order = [("fcfs", "FCFS"), ("fairshare", "Fair-share"), ("cqos_dpu", "CompactQoS")]
    order = [(p, lab) for p, lab in order if p in set(r.policy)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, metric, ylab, log in [
            (axes[0], "calm_p99_ms_worst", "worst calm-tenant p99, ms (log scale)", True),
            (axes[1], "calm_slow_s_mean", "seconds in slowdown per calm tenant", False)]:
        means = [r[r.policy == p][metric].mean() for p, _ in order]
        ax.bar(range(len(order)), means, 0.6, color=[COLORS[lab] for _, lab in order])
        for i, (p, _) in enumerate(order):                       # one dot per seed
            pts = r[r.policy == p][metric]
            ax.scatter([i] * len(pts), pts, s=18, color=INK, zorder=3, linewidths=0)
        top = max(r[metric].max(), max(means))
        for i, v in enumerate(means):
            ax.annotate(f"{v:.1f}" if v >= 10 or not log else f"{v:.2f}", (i, v),
                        xytext=(14, 2), textcoords="offset points", fontsize=8, color=INK)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([lab for _, lab in order])
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1, top * 4)
        else:
            ax.set_ylim(0, top * 1.25)
        ax.set_ylabel(ylab)
    n_seeds = r.seed.nunique()
    fig.suptitle(f"Real RocksDB, 3 tenants, shared emulated DPU slot "
                 f"(bars = mean, dots = {n_seeds} individual runs)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_rocksdb.png"), dpi=200)
    plt.close(fig)
    print("wrote fig8_rocksdb.png")

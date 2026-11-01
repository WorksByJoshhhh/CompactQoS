"""
Review 3: extra experiments on real RocksDB
===========================================
Three questions the main experiment (3 tenants, 1 noisy, baseline DPU speed) leaves open:

  E1  DPU speed:     does the result depend on the emulated DPU throttle?
                     3 tenants, 1 noisy, DPU 2x faster / baseline / 2x slower
  E2  More tenants:  6 tenants, 1 noisy (the DPU is sized for 6 tenants)
  E3  Two noisy:     4 tenants, 2 of them noisy at the same time

Each scenario runs FCFS, fair-share and CompactQoS with the same seeds.

Usage
-----
    python run_review3_experiments.py                  # all, 3 seeds x 60 s, about 50 min
    python run_review3_experiments.py --only E1        # one experiment (E1 about 28 min, E2/E3 about 10 min)
    python run_review3_experiments.py --quick          # 20 s runs, 1 seed: checks everything works (about 5 min)

Results are appended to review3_runs.csv after every run, so nothing is lost if you
stop it. Runs already in the file are skipped, so you can restart where you left off.
Then:  python plot_review3.py
"""
import argparse
import csv
import os
import shutil
import time

import rocksdb_harness as h

EXPERIMENTS = {
    "E1": [dict(label="DPU 2x faster", n_tenants=3, n_noisy=1, dpu_scale=0.5),
           dict(label="Baseline DPU", n_tenants=3, n_noisy=1, dpu_scale=1.0),
           dict(label="DPU 2x slower", n_tenants=3, n_noisy=1, dpu_scale=2.0)],
    "E2": [dict(label="6 tenants, 1 noisy", n_tenants=6, n_noisy=1, dpu_scale=1.0)],
    "E3": [dict(label="4 tenants, 2 noisy", n_tenants=4, n_noisy=2, dpu_scale=1.0)],
}
POLICIES = ["fcfs", "fairshare", "cqos_dpu"]


def main():
    ap = argparse.ArgumentParser(description="Review 3 RocksDB experiments")
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--quick", action="store_true", help="20 s runs, 1 seed")
    ap.add_argument("--out", default="review3_runs.csv")
    ap.add_argument("--dir", default="rocks_run_r3")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.duration, args.out = 1, 20.0, "review3_quick.csv"

    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["experiment"], r["scenario"], r["policy"], int(r["seed"])))

    todo = [(e, sc, pol, seed)
            for e in args.only for sc in EXPERIMENTS[e]
            for seed in range(1, args.seeds + 1) for pol in POLICIES
            if (e, sc["label"], pol, seed) not in done]
    print(f"{len(todo)} runs to do, about {len(todo) * (args.duration + 3) / 60:.0f} min "
          f"({len(done)} already in {args.out})", flush=True)

    for i, (e, sc, pol, seed) in enumerate(todo, 1):
        cfg = dict(h.CFG, duration=args.duration, n_tenants=sc["n_tenants"],
                   n_noisy=sc["n_noisy"], dpu_scale=sc["dpu_scale"])
        t0 = time.time()
        row = h.run(pol, seed, cfg, args.dir, [])
        row = dict(experiment=e, scenario=sc["label"], **row)
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[{i}/{len(todo)}] {e} {sc['label']:<20} seed {seed} {pol:<10} "
              f"calm p99 worst {row['calm_p99_ms_worst']:8.1f} ms | "
              f"calm slowdown {row['calm_slow_s_mean']:5.2f} s | "
              f"noisy served {row['flood_goodput']:.2f} | ({time.time() - t0:.0f}s)", flush=True)

    shutil.rmtree(args.dir, ignore_errors=True)
    print("done, results in", args.out)


if __name__ == "__main__":
    main()

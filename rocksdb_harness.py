"""
CompactQoS on real RocksDB
==========================
Runs the Fairness Manager against N real RocksDB instances (via rocksdict),
one per tenant, sharing ONE emulated "DPU" compaction slot.

What is real
------------
* RocksDB instances, memtable flushes, L0 file counts (rocksdb.num-files-at-level0)
* Compaction work: the manager calls db.compact_range() on the chosen tenant
* Write latency is measured with a wall clock (open loop: from the moment a write
  was due until RocksDB accepted it)

What is emulated (say this in the report)
-----------------------------------------
* Automatic compaction is OFF so the manager owns all compaction. RocksDB does not
  apply its own L0 write stall in that mode (see check_stall.py), so each writer
  applies RocksDB's L0 rule itself before every write:
      L0 >= slowdown_trigger -> write paced to slowdown_rate writes/s
      L0 >= stop_trigger     -> write blocked until compaction brings L0 down
* The shared "DPU" is a single job slot that waits a throttle delay before each
  real compaction (dpu_base + dpu_per_file * L0 files), standing in for a slower,
  shared processor. There is no real DPU and no cgroup.
* compact_range() clears a tenant's whole L0 at once, so each tenant has at most
  one open request. The DPU's emulated work for a request is done in chunks of
  batch_files files with a scheduling decision between chunks (like the
  simulator's 4-file jobs); the real compact_range() runs after the last chunk.

Usage
-----
    python rocksdb_harness.py --seeds 5                 # main experiment, about 16 min
    python rocksdb_harness.py --duration 20 --seeds 1   # quick check, about 1 min
    python rocksdb_harness.py --tenants 6 --noisy 2 --dpu-scale 2.0   # other scenarios
The extra Review 3 experiments are run by run_review3_experiments.py.
Writes rocksdb_runs.csv (one row per run) and rocksdb_jobs.csv (one row per compaction).
"""
import argparse
import csv
import math
import os
import shutil
import threading
import time

import numpy as np
from rocksdict import Options, Rdict

CFG = dict(
    n_tenants=3,
    n_noisy=1,                 # tenants 0..n_noisy-1 flood; the rest are calm
    dpu_scale=1.0,             # multiplier on the DPU throttle: 2.0 = DPU twice as slow
    duration=60.0,
    slowdown_trigger=8,
    stop_trigger=16,
    slowdown_rate=80.0,        # writes/s while in slowdown
    batch_files=4,             # Interceptor raises a request once L0 >= this
    write_buffer=128 * 1024,   # memtable size -> roughly 100+ writes per L0 file
    value_size=1024,
    keyspace=20000,            # Zipf-like keys over a bounded keyspace
    calm_rate=100.0,           # calm tenants: 50..150 writes/s (heterogeneous)
    flood_rate=3000.0,         # each noisy tenant floods from 20% to 70% of the run
    calm_burst_prob=0.05,      # per second
    calm_burst_len=3.0,
    calm_burst_mult=3.0,
    dpu_base=0.05,             # emulated DPU throttle: seconds per job
    dpu_per_file=0.15,         #   ... plus seconds per L0 file
    ledger_half_life=5.0,
    share_slack=0.25,
    poll=0.005,                # manager poll interval (s)
)

POLICIES = {"fcfs": "fcfs", "fairshare": "rr", "cqos_dpu": "cqos",
            "cqos_dpu_no_ledger": "cqos_noledger"}


# --------------------------------------------------------------------------- #
# Workload: pre-generated open-loop arrival times per tenant
# --------------------------------------------------------------------------- #
def make_arrivals(cfg, tid, n, rng):
    d = cfg["duration"]
    noisy = tid < cfg["n_noisy"]
    bursts = []
    if not noisy:
        t = 0.0
        while t < d:
            if rng.random() < cfg["calm_burst_prob"]:
                bursts.append(t)
                t += cfg["calm_burst_len"]
            t += 1.0

    def rate(t):
        if noisy:
            return cfg["flood_rate"] if 0.2 * d <= t < 0.7 * d else cfg["calm_rate"]
        k, n_calm = tid - cfg["n_noisy"], n - cfg["n_noisy"]
        r = cfg["calm_rate"] * (0.5 + k / max(n_calm - 1, 1))   # calm rates spread 0.5x..1.5x
        if any(a <= t < a + cfg["calm_burst_len"] for a in bursts):
            r *= cfg["calm_burst_mult"]
        return r

    rmax = max(cfg["flood_rate"], cfg["calm_rate"] * 1.5 * cfg["calm_burst_mult"])
    out, t = [], 0.0
    while True:                                  # thinning of a Poisson process
        t += rng.exponential(1.0 / rmax)
        if t >= d:
            return np.array(out)
        if rng.random() < rate(t) / rmax:
            out.append(t)


# --------------------------------------------------------------------------- #
# Tenant
# --------------------------------------------------------------------------- #
class Tenant:
    def __init__(self, tid, path, cfg):
        self.tid = tid
        o = Options()
        o.create_if_missing(True)
        o.set_disable_auto_compactions(True)
        o.set_write_buffer_size(cfg["write_buffer"])
        self.db = Rdict(path, o)
        self.l0 = 0                 # refreshed by the manager every poll
        self.lat = []               # seconds
        self.offered = 0
        self.served = 0
        self.slow_s = 0.0
        self.stop_s = 0.0
        self.episodes = 0
        self.in_trouble = False
        self.ledger = 0.0           # Usage Tracker: decayed DPU seconds
        self.ledger_t = 0.0
        self.dpu_s = 0.0
        self.pending = False        # request queued or running
        self.work_left = None       # DPU files left in the current request
        self.job_files = 0
        self.chunks = 0
        self.t_req = 0.0
        self.t_first = 0.0

    def read_l0(self):
        return self.db.property_int_value("rocksdb.num-files-at-level0") or 0

    def ledger_at(self, now, half_life):
        return self.ledger * math.exp(-math.log(2) * (now - self.ledger_t) / half_life)


def writer(ten, arr, keys, cfg, t_start, stop_flag):
    val = b"x" * cfg["value_size"]
    gap = 1.0 / cfg["slowdown_rate"]
    i, n = 0, len(arr)
    ten.offered = n
    while i < n and not stop_flag.is_set():
        due = t_start + arr[i]
        now = time.perf_counter()
        if now < due:
            time.sleep(min(due - now, 0.01))
            continue
        while ten.l0 >= cfg["stop_trigger"] and not stop_flag.is_set():   # stop zone
            time.sleep(0.002)
        if stop_flag.is_set():
            break
        if ten.l0 >= cfg["slowdown_trigger"]:                              # slowdown zone
            time.sleep(gap)
        ten.db[keys[i]] = val
        ten.lat.append(time.perf_counter() - due)
        ten.served += 1
        i += 1
    end = t_start + cfg["duration"]              # censored writes: due but never served
    for j in range(i, n):
        due = t_start + arr[j]
        if due <= end:
            ten.lat.append(max(end - due, 0.0))


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def run(policy, seed, cfg, base_dir, job_rows):
    kind = POLICIES[policy]
    n = cfg["n_tenants"]
    shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    tenants = [Tenant(i, os.path.join(base_dir, f"tenant{i}"), cfg) for i in range(n)]
    arrs = [make_arrivals(cfg, i, n, rng) for i in range(n)]
    keys = [[f"user{int(z) % cfg['keyspace']:08d}".encode() for z in rng.zipf(1.1, size=len(a))]
            for a in arrs]

    queue = []                  # tenant ids with an open request, in request order
    lock = threading.Lock()
    stop_flag = threading.Event()
    dpu_free = threading.Event()
    dpu_free.set()
    rr_last = [-1]
    decide = [0, 0]             # total ns, count

    def pick(now):
        """Index into `queue` of the next tenant to compact (the Scheduler)."""
        if kind == "fcfs":
            return 0
        if kind == "rr":
            for k in range(1, n + 1):
                tid = (rr_last[0] + k) % n
                if tid in queue:
                    rr_last[0] = tid
                    return queue.index(tid)
        use_ledger = kind == "cqos"
        led = [t.ledger_at(now, cfg["ledger_half_life"]) for t in tenants]
        total = sum(led)
        fair = 1.0 / n
        best_i, best_s = 0, -1e18
        for i, tid in enumerate(queue):
            over = use_ledger and total > 1e-9 and led[tid] / total > fair * (1 + cfg["share_slack"])
            s = (0.0 if over else 10.0) + tenants[tid].l0 / cfg["stop_trigger"]
            if s > best_s + 1e-12:
                best_s, best_i = s, i
        return best_i

    def dpu_chunk(tid, chunk, t_run):
        """Run one chunk of a tenant's compaction on the shared DPU slot.
        The DPU's work is done in chunks of batch_files files, so the Scheduler
        gets a decision point between chunks (like RocksDB splitting a large
        compaction into subcompactions). The real compact_range() runs once the
        tenant's last chunk is done."""
        ten = tenants[tid]
        # emulated DPU throttle; the DPU is sized for the tenant count (n/3), as in
        # the simulator, and dpu_scale makes it faster (<1) or slower (>1)
        time.sleep((cfg["dpu_base"] + cfg["dpu_per_file"] * chunk)
                   * cfg["dpu_scale"] * 3.0 / n)
        with lock:
            ten.work_left -= chunk
            if ten.work_left <= 0:
                # charge for every file the real compaction will clear: files
                # flushed since the request started become more DPU work first
                extra = ten.read_l0() - ten.job_files
                if extra > 0:
                    ten.work_left = extra
                    ten.job_files += extra
            done = ten.work_left <= 0
        real_ms = 0.0
        if done:
            c0 = time.perf_counter()
            ten.db.compact_range(None, None)                           # real RocksDB compaction
            real_ms = (time.perf_counter() - c0) * 1000
        t_end = time.perf_counter()
        with lock:
            used = t_end - t_run
            ten.ledger = ten.ledger_at(t_end, cfg["ledger_half_life"]) + used  # measured-usage feedback
            ten.ledger_t = t_end
            ten.dpu_s += used
            if done:
                ten.l0 = ten.read_l0()
                job_rows.append(dict(policy=policy, seed=seed, tenant=tid, files=ten.job_files,
                                     chunks=ten.chunks + 1,
                                     wait_s=round(ten.t_first - ten.t_req, 4),
                                     real_compact_ms=round(real_ms, 2),
                                     total_job_s=round(t_end - ten.t_req, 4)))
                queue.remove(tid)
                ten.work_left = None
                ten.pending = False
            else:
                ten.chunks += 1
        dpu_free.set()

    t_start = time.perf_counter() + 0.2
    threads = [threading.Thread(target=writer, daemon=True,
                                args=(tenants[i], arrs[i], keys[i], cfg, t_start, stop_flag))
               for i in range(n)]
    for th in threads:
        th.start()

    end = t_start + cfg["duration"]
    last = time.perf_counter()
    job_thread = None
    while time.perf_counter() < end:
        now = time.perf_counter()
        dt, last = now - last, now
        with lock:
            for ten in tenants:
                ten.l0 = ten.read_l0()
                trouble = ten.l0 >= cfg["slowdown_trigger"]
                if now >= t_start:
                    if ten.l0 >= cfg["stop_trigger"]:
                        ten.stop_s += dt
                    elif trouble:
                        ten.slow_s += dt
                    if trouble and not ten.in_trouble:
                        ten.episodes += 1
                ten.in_trouble = trouble
                # Interceptor: one request per tenant once L0 reaches batch_files
                if not ten.pending and ten.l0 >= cfg["batch_files"]:
                    ten.pending = True
                    queue.append(ten.tid)
                    ten.t_req = now
            if dpu_free.is_set() and queue:
                d0 = time.perf_counter_ns()
                tid = queue[pick(now)]
                decide[0] += time.perf_counter_ns() - d0
                decide[1] += 1
                ten = tenants[tid]
                if ten.work_left is None:              # first chunk of this request
                    ten.work_left = ten.job_files = max(ten.l0, 1)
                    ten.t_first = now
                    ten.chunks = 0
                chunk = min(cfg["batch_files"], ten.work_left)
                dpu_free.clear()
                job_thread = threading.Thread(target=dpu_chunk, daemon=True, args=(tid, chunk, now))
                job_thread.start()
        time.sleep(cfg["poll"])

    stop_flag.set()
    for th in threads:
        th.join(timeout=10)
    if job_thread is not None:
        job_thread.join(timeout=30)
    for ten in tenants:
        ten.db.close()
    return summarize(policy, seed, cfg, tenants, decide)


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else 0.0


def jain(xs):
    s2 = sum(x * x for x in xs)
    return (sum(xs) ** 2) / (len(xs) * s2) if s2 else 1.0


def summarize(policy, seed, cfg, tenants, decide):
    k = cfg["n_noisy"]
    noisy, calm = tenants[:k], tenants[k:]
    gp = [t.served / max(t.offered, 1) for t in tenants]
    tot_dpu = sum(t.dpu_s for t in tenants) or 1e-9
    return dict(
        policy=policy, seed=seed, n_tenants=cfg["n_tenants"], n_noisy=k,
        dpu_scale=cfg["dpu_scale"], duration=cfg["duration"],
        calm_p99_ms_worst=max(1000 * pct(t.lat, 99) for t in calm),
        calm_p99_ms_mean=float(np.mean([1000 * pct(t.lat, 99) for t in calm])),
        calm_p50_ms_mean=float(np.mean([1000 * pct(t.lat, 50) for t in calm])),
        calm_slow_s_mean=float(np.mean([t.slow_s for t in calm])),
        calm_stop_s_mean=float(np.mean([t.stop_s for t in calm])),
        calm_episodes_mean=float(np.mean([t.episodes for t in calm])),
        calm_goodput_mean=float(np.mean(gp[k:])),
        flood_goodput=float(np.mean(gp[:k])),          # mean over the noisy tenants
        flood_p99_ms=max(1000 * pct(t.lat, 99) for t in noisy),
        jain_all=jain(gp),
        dpu_share_flooder=sum(t.dpu_s for t in noisy) / tot_dpu,
        decision_us=(decide[0] / max(decide[1], 1)) / 1000.0,
        noisy_offered=sum(t.offered for t in noisy), noisy_served=sum(t.served for t in noisy),
        calm_offered=sum(t.offered for t in calm), calm_served=sum(t.served for t in calm),
    )


def main():
    ap = argparse.ArgumentParser(description="CompactQoS on real RocksDB")
    ap.add_argument("--policies", nargs="+", default=["fcfs", "fairshare", "cqos_dpu"],
                    choices=list(POLICIES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--duration", type=float, default=CFG["duration"])
    ap.add_argument("--tenants", type=int, default=CFG["n_tenants"])
    ap.add_argument("--noisy", type=int, default=CFG["n_noisy"], help="number of noisy tenants")
    ap.add_argument("--dpu-scale", type=float, default=CFG["dpu_scale"],
                    help="DPU throttle multiplier: 0.5 = twice as fast, 2.0 = twice as slow")
    ap.add_argument("--dir", default="rocks_run", help="scratch folder for the databases")
    ap.add_argument("--out", default="rocksdb_runs.csv")
    ap.add_argument("--jobs-out", default="rocksdb_jobs.csv")
    args = ap.parse_args()

    cfg = dict(CFG, duration=args.duration, n_tenants=args.tenants, n_noisy=args.noisy,
               dpu_scale=args.dpu_scale)
    rows, jobs = [], []
    print(f"{len(args.policies)} policies x {args.seeds} seeds x {cfg['duration']:.0f} s "
          f"(about {len(args.policies) * args.seeds * (cfg['duration'] + 3) / 60:.0f} min)", flush=True)
    for seed in range(1, args.seeds + 1):
        for pol in args.policies:
            t0 = time.time()
            r = run(pol, seed, cfg, args.dir, jobs)
            rows.append(r)
            print(f"seed {seed} {pol:<20} calm p99 worst {r['calm_p99_ms_worst']:8.1f} ms | "
                  f"calm slowdown {r['calm_slow_s_mean']:5.2f} s | "
                  f"flooder served {r['flood_goodput']:.2f} | ({time.time() - t0:.0f}s)", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(args.jobs_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0]))
        w.writeheader()
        w.writerows(jobs)
    shutil.rmtree(args.dir, ignore_errors=True)
    print("wrote", args.out, "and", args.jobs_out)


if __name__ == "__main__":
    main()

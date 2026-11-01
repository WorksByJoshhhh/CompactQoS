"""
CompactQoS simulator
====================
Discrete-time simulation of N tenants sharing ONE emulated "DPU" compaction slot
(plus an optional host-CPU fallback slot).

What is modelled
----------------
* Each tenant writes into a memtable; every `flush_writes` writes become one L0 file.
* Stall rule (the same one the RocksDB harness will use):
    L0 <  slowdown_trigger : writes served at full speed
    L0 >= slowdown_trigger : writes served at `slowdown_rate` writes/s
    L0 >= stop_trigger     : writes blocked until compaction reduces L0
* Every time a tenant has `batch_files` L0 files not yet covered by a compaction
  request, the Interceptor emits one request ("compact these files").
* A compaction job clears `batch_files` L0 files. It runs on:
    - the shared DPU slot (one job at a time, throttled), or
    - the host fallback slot (faster, but while it runs EVERY tenant's write
      service rate is multiplied by `host_contention` and each write pays
      `host_penalty_ms` extra, because it uses the tenants' own CPU).
* Policies (see POLICIES) decide which pending request goes next and where.
  Experiment A, shared DPU slot only:
    fcfs       first-come-first-served (what published systems do)
    fairshare  round-robin across tenants (strict equal turns)
    cqos_dpu   CompactQoS scheduler (ledger tiers + urgency)
    + ablations cqos_dpu_no_urgency, cqos_dpu_no_ledger
  Experiment B, DPU slot + host fallback slot:
    fcfs_2slot, fairshare_2slot   baselines that use the host whenever free
    cqos       full CompactQoS (ledger + urgency + Spillover Controller)
    + ablations cqos_no_urgency, cqos_no_ledger, spill_only

Honest limitations (say these in the report)
--------------------------------------------
* This is a simulator, not real DPU hardware. Absolute numbers mean nothing;
  only the comparison between policies is meaningful.
* Key distributions (Zipfian etc.) do not change L0 dynamics, so the simulator's
  load generator only produces arrival counts. Zipfian keys belong in the
  RocksDB harness.
"""
import argparse
import csv
import math
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    n_tenants: int = 3
    duration: float = 120.0       # simulated seconds
    dt: float = 0.01              # tick length (s)
    seed: int = 1
    pattern: str = "spike"        # spike | ramp | oscillate

    # LSM / stall model (same numbers to be used in the RocksDB harness)
    flush_writes: int = 100       # writes per memtable flush -> one L0 file
    batch_files: int = 4          # L0 files cleared per compaction job
    slowdown_trigger: int = 8
    stop_trigger: int = 16
    normal_rate: float = 5000.0   # max writes/s a tenant can be served when healthy
    slowdown_rate: float = 80.0   # writes/s served while in slowdown
    base_latency_ms: float = 0.35

    # offered load (writes/s). Tenant 0 is the flooder, others are calm.
    calm_rate: float = 100.0
    flood_rate: float = 3000.0
    calm_burst_prob: float = 0.05  # per-second chance a calm tenant starts a burst
    calm_burst_len: float = 3.0    # seconds
    calm_burst_mult: float = 3.0

    # shared "DPU" slot (throttled) and host fallback slot
    dpu_base: float = 0.05        # seconds per job
    dpu_per_file: float = 0.15    # seconds per file
    host_base: float = 0.02
    host_per_file: float = 0.07
    host_contention: float = 0.6  # tenants' write service multiplier while host slot busy
    host_penalty_ms: float = 2.0  # extra latency on every write served while host slot busy
    capacity_scale: float = 1.0   # >1 means a bigger shared worker (used for scaling runs)

    # CompactQoS knobs
    ledger_half_life: float = 5.0  # seconds
    share_slack: float = 0.25      # "over share" means > (1 + slack) x fair share
    spill_l0: int = 8              # an over-share tenant at/above this L0 may be rescued on the host
    spill_l0_under: int = 4        # an under-share tenant is rescued earlier, at/above this L0
    hetero_calm: bool = True       # calm tenants get different steady rates (0.5x-1.5x)


# host modes: "none"   = DPU slot only
#             "always" = host slot used whenever free (extra capacity, no fairness)
#             "spill"  = host slot used only to rescue a tenant at/above spill_l0
POLICIES = {
    #                        pick    urgency ledger host
    # --- Experiment A: shared DPU slot only (equal hardware) ---
    "fcfs":                 ("fcfs", False, False, "none"),
    "fairshare":            ("rr",   False, False, "none"),
    "cqos_dpu":             ("cqos", True,  True,  "none"),
    "cqos_dpu_no_urgency":  ("cqos", False, True,  "none"),
    "cqos_dpu_no_ledger":   ("cqos", True,  False, "none"),
    # --- Experiment B: DPU slot + host fallback slot (equal hardware) ---
    "fcfs_2slot":           ("fcfs", False, False, "always"),
    "fairshare_2slot":      ("rr",   False, False, "always"),
    "cqos":                 ("cqos", True,  True,  "spill"),
    "cqos_no_urgency":      ("cqos", False, True,  "spill"),
    "cqos_no_ledger":       ("cqos", True,  False, "spill"),
    "spill_only":           ("cqos", False, False, "spill"),
}


# --------------------------------------------------------------------------- #
# Load generator (YCSB-style burst patterns on tenant 0)
# --------------------------------------------------------------------------- #
def offered_rate(cfg: Config, tid: int, t: float, bursting: bool = False) -> float:
    if tid != 0:
        r = cfg.calm_rate
        if cfg.hetero_calm and cfg.n_tenants > 2:
            frac = (tid - 1) / (cfg.n_tenants - 2)      # 0..1 across calm tenants
            r *= (0.5 + frac)
        return r * (cfg.calm_burst_mult if bursting else 1.0)
    d = cfg.duration
    if cfg.pattern == "spike":
        return cfg.flood_rate if 0.2 * d <= t < 0.7 * d else cfg.calm_rate
    if cfg.pattern == "ramp":
        lo, hi = 0.1 * d, 0.7 * d
        if lo <= t < hi:
            frac = (t - lo) / (hi - lo)
            return cfg.calm_rate + (cfg.flood_rate - cfg.calm_rate) * frac
        return cfg.calm_rate
    if cfg.pattern == "oscillate":
        return cfg.flood_rate if int(t / (0.1 * d)) % 2 == 1 else cfg.calm_rate
    raise ValueError(cfg.pattern)


# --------------------------------------------------------------------------- #
# Simulation state
# --------------------------------------------------------------------------- #
class Tenant:
    def __init__(self, tid):
        self.tid = tid
        self.l0 = 0
        self.memtable = 0
        self.covered = 0              # L0 files already covered by a request
        self.backlog = deque()        # [arrival_time, count]
        self.credit = 0.0
        self.lat = []                 # (latency_s, count)
        self.offered = 0
        self.served = 0
        self.slow_time = 0.0
        self.stop_time = 0.0
        self.episodes = 0
        self.in_trouble = False
        self.ledger = 0.0             # decayed DPU seconds used (Usage Tracker)
        self.dpu_time = 0.0
        self.host_time = 0.0
        self.burst_until = -1.0


class Job:
    __slots__ = ("tid", "files", "finish", "slot")

    def __init__(self, tid, files):
        self.tid, self.files, self.finish, self.slot = tid, files, 0.0, ""


def weighted_percentile(pairs, q):
    if not pairs:
        return 0.0
    arr = np.array(pairs, dtype=float)
    order = np.argsort(arr[:, 0])
    lat, cnt = arr[order, 0], arr[order, 1]
    cum = np.cumsum(cnt)
    idx = min(np.searchsorted(cum, q * cum[-1]), len(lat) - 1)
    return float(lat[idx])


def jain(xs):
    xs = [x for x in xs]
    if not xs or sum(x * x for x in xs) == 0:
        return 1.0
    return sum(xs) ** 2 / (len(xs) * sum(x * x for x in xs))


def run(cfg: Config, policy: str, trace_every: float = 0.0):
    pick_kind, use_urgency, use_ledger, host_mode = POLICIES[policy]
    rng = np.random.default_rng(cfg.seed)
    tenants = [Tenant(i) for i in range(cfg.n_tenants)]
    queue = deque()                    # pending Jobs in request order
    dpu = None                         # running Job or None
    host = None
    rr_last = -1
    decay = math.exp(-math.log(2) * cfg.dt / cfg.ledger_half_life)
    dpu_busy = host_busy = 0.0
    decision_ns = 0
    decisions = 0
    trace = []
    next_trace = 0.0

    def job_time(files, slot):
        if slot == "dpu":
            return (cfg.dpu_base + cfg.dpu_per_file * files) / cfg.capacity_scale
        return (cfg.host_base + cfg.host_per_file * files) / cfg.capacity_scale

    def pick():
        """Return index in `queue` of the next job to run, per policy."""
        nonlocal rr_last
        if pick_kind == "fcfs":
            return 0
        if pick_kind == "rr":
            pending = {j.tid for j in queue}
            for k in range(1, cfg.n_tenants + 1):
                tid = (rr_last + k) % cfg.n_tenants
                if tid in pending:
                    rr_last = tid
                    for i, j in enumerate(queue):
                        if j.tid == tid:
                            return i
        # CompactQoS scoring (two-tier):
        #   tier 1 = tenants at or under their fair share of recent DPU time
        #            (Usage Tracker ledger); tier 2 = tenants over their share.
        #   Within a tier, the most urgent tenant (highest L0 / stop_trigger)
        #   goes first. Ties keep request order (oldest first).
        # Without the ledger, everyone is tier 1 (pure urgency).
        # Without urgency, a tier is served oldest-first.
        total = sum(t.ledger for t in tenants)
        fair = 1.0 / cfg.n_tenants
        best_i, best_s, seen = 0, -1e18, set()
        for i, j in enumerate(queue):
            if j.tid in seen:          # first (oldest) job of each tenant only
                continue
            seen.add(j.tid)
            t = tenants[j.tid]
            over = use_ledger and total > 1e-9 and t.ledger / total > fair * (1 + cfg.share_slack)
            urg = t.l0 / cfg.stop_trigger if use_urgency else 0.0
            s = (0.0 if over else 10.0) + urg
            if s > best_s + 1e-12:
                best_s, best_i = s, i
        return best_i

    def take(i):
        j = queue[i]
        del queue[i]
        return j

    steps = int(round(cfg.duration / cfg.dt))
    for step in range(steps):
        t = step * cfg.dt

        # 1. arrivals (calm tenants occasionally burst, YCSB-style)
        for ten in tenants:
            if ten.tid != 0:
                if ten.burst_until <= t and rng.random() < cfg.calm_burst_prob * cfg.dt:
                    ten.burst_until = t + cfg.calm_burst_len
            n = rng.poisson(offered_rate(cfg, ten.tid, t, ten.burst_until > t) * cfg.dt)
            if n:
                ten.backlog.append([t, n])
                ten.offered += n

        # 2. service (stall rule) + flush into L0
        contention = cfg.host_contention if host is not None else 1.0
        penalty = cfg.host_penalty_ms / 1000.0 if host is not None else 0.0
        for ten in tenants:
            if ten.l0 >= cfg.stop_trigger:
                rate = 0.0
            elif ten.l0 >= cfg.slowdown_trigger:
                rate = cfg.slowdown_rate
            else:
                rate = cfg.normal_rate
            rate *= contention
            ten.credit = min(ten.credit + rate * cfg.dt, rate * cfg.dt + 1.0)
            can = int(ten.credit)
            while can > 0 and ten.backlog:
                head = ten.backlog[0]
                m = min(can, head[1])
                ten.lat.append(((t - head[0]) + cfg.base_latency_ms / 1000.0 + penalty, m))
                head[1] -= m
                can -= m
                ten.credit -= m
                ten.served += m
                ten.memtable += m
                if head[1] == 0:
                    ten.backlog.popleft()
            while ten.memtable >= cfg.flush_writes:
                ten.memtable -= cfg.flush_writes
                ten.l0 += 1

            # stall accounting
            trouble = ten.l0 >= cfg.slowdown_trigger
            if ten.l0 >= cfg.stop_trigger:
                ten.stop_time += cfg.dt
            elif trouble:
                ten.slow_time += cfg.dt
            if trouble and not ten.in_trouble:
                ten.episodes += 1
            ten.in_trouble = trouble

            # Interceptor: emit compaction requests for uncovered files
            while ten.l0 - ten.covered >= cfg.batch_files:
                ten.covered += cfg.batch_files
                queue.append(Job(ten.tid, cfg.batch_files))

            ten.ledger *= decay

        # 3. job completion
        if dpu is not None and t >= dpu.finish:
            ten = tenants[dpu.tid]
            ten.l0 -= dpu.files
            ten.covered -= dpu.files
            dpu = None
        if host is not None and t >= host.finish:
            ten = tenants[host.tid]
            ten.l0 -= host.files
            ten.covered -= host.files
            host = None

        # 4. dispatch (timed: this is the Fairness Manager's decision overhead)
        t0 = time.perf_counter_ns()
        if dpu is None and queue:
            j = take(pick())
            j.slot, j.finish = "dpu", t + job_time(j.files, "dpu")
            dpu = j
            decisions += 1
        if host is None and queue and host_mode != "none":
            i = None
            if host_mode == "always":
                i = pick()
            else:
                # Spillover Controller:
                #   * a tenant within its fair DPU share is rescued early,
                #     once it is about to stall (L0 >= spill_l0_under)
                #   * a tenant over its share is rescued only once it is
                #     already in trouble (L0 >= spill_l0)
                # Among candidates: under-share first, then deepest L0
                # (or oldest request, if urgency is off).
                total = sum(x.ledger for x in tenants)
                fair = 1.0 / cfg.n_tenants
                best = None
                for k, j in enumerate(queue):
                    ten = tenants[j.tid]
                    over = use_ledger and total > 1e-9 and ten.ledger / total > fair * (1 + cfg.share_slack)
                    need = cfg.spill_l0 if (over or not use_ledger) else cfg.spill_l0_under
                    if ten.l0 >= need:
                        key = (0 if over else 1, ten.l0 if use_urgency else 0)
                        if best is None or key > best:
                            best, i = key, k
            if i is not None:
                j = take(i)
                j.slot, j.finish = "host", t + job_time(j.files, "host")
                host = j
                decisions += 1
        decision_ns += time.perf_counter_ns() - t0

        # 5. usage accounting
        if dpu is not None:
            dpu_busy += cfg.dt
            tenants[dpu.tid].dpu_time += cfg.dt
            tenants[dpu.tid].ledger += cfg.dt
        if host is not None:
            host_busy += cfg.dt
            tenants[host.tid].host_time += cfg.dt

        if trace_every and t >= next_trace:
            trace.append([round(t, 3)] + [x.l0 for x in tenants])
            next_trace += trace_every

    # censored: writes still waiting at the end count with their current wait
    end = steps * cfg.dt
    for ten in tenants:
        for arr, cnt in ten.backlog:
            ten.lat.append(((end - arr) + cfg.base_latency_ms / 1000.0, cnt))

    return summarize(cfg, policy, tenants, dpu_busy, host_busy, decision_ns, decisions), trace


def summarize(cfg, policy, tenants, dpu_busy, host_busy, decision_ns, decisions):
    calm = tenants[1:]
    per = []
    for t in tenants:
        per.append(
            dict(
                tid=t.tid,
                p50_ms=1000 * weighted_percentile(t.lat, 0.50),
                p99_ms=1000 * weighted_percentile(t.lat, 0.99),
                stop_s=t.stop_time,
                slow_s=t.slow_time,
                episodes=t.episodes,
                norm_goodput=t.served / max(t.offered, 1),
                dpu_s=t.dpu_time,
            )
        )
    calm_p = per[1:]
    tot_dpu = sum(p["dpu_s"] for p in per) + 1e-9
    return dict(
        policy=policy,
        pattern=cfg.pattern,
        n_tenants=cfg.n_tenants,
        seed=cfg.seed,
        calm_p99_ms_worst=max(p["p99_ms"] for p in calm_p),
        calm_p99_ms_mean=float(np.mean([p["p99_ms"] for p in calm_p])),
        calm_stop_s_mean=float(np.mean([p["stop_s"] for p in calm_p])),
        calm_slow_s_mean=float(np.mean([p["slow_s"] for p in calm_p])),
        calm_episodes_mean=float(np.mean([p["episodes"] for p in calm_p])),
        calm_goodput_mean=float(np.mean([p["norm_goodput"] for p in calm_p])),
        flood_goodput=per[0]["norm_goodput"],
        flood_p99_ms=per[0]["p99_ms"],
        jain_all=jain([p["norm_goodput"] for p in per]),
        jain_calm=jain([p["norm_goodput"] for p in calm_p]),
        dpu_util=dpu_busy / cfg.duration,
        host_util=host_busy / cfg.duration,
        dpu_share_flooder=per[0]["dpu_s"] / tot_dpu,
        overhead_us_per_decision=(decision_ns / max(decisions, 1)) / 1000.0,
    )


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def make_cfg(base: Config, n, pattern, seed):
    # Scaling runs: the shared worker is sized in proportion to the tenant count
    # (a DPU for 3 tenants is not the same machine as a DPU for 20). Assumption,
    # state it in the report.
    return replace(base, n_tenants=n, pattern=pattern, seed=seed, capacity_scale=n / 3.0)


def main():
    ap = argparse.ArgumentParser(description="CompactQoS simulator")
    ap.add_argument("--tenants", type=int, nargs="+", default=[3])
    ap.add_argument("--patterns", nargs="+", default=["spike", "ramp", "oscillate"])
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--out", default="results_runs.csv")
    ap.add_argument("--trace", action="store_true",
                    help="also write trace_<policy>.csv (L0 over time, 3 tenants, spike, seed 1)")
    args = ap.parse_args()

    base = Config(duration=args.duration)
    rows = []
    t0 = time.time()
    for n in args.tenants:
        for pat in args.patterns:
            for pol in args.policies:
                for s in range(1, args.seeds + 1):
                    res, _ = run(make_cfg(base, n, pat, s), pol)
                    rows.append(res)
            print(f"done n={n} pattern={pat}  ({time.time() - t0:.0f}s)")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out, len(rows), "rows")

    if args.trace:
        for pol in ["fcfs", "fairshare", "cqos_dpu"]:
            _, tr = run(make_cfg(base, 3, "spike", 1), pol, trace_every=0.25)
            with open(f"trace_{pol}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t"] + [f"tenant{i}_l0" for i in range(3)])
                w.writerows(tr)
        print("wrote trace_*.csv")


if __name__ == "__main__":
    main()

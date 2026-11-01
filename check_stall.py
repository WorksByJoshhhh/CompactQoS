import shutil, time, statistics
from rocksdict import Rdict, Options

PATH = "stalldb"
shutil.rmtree(PATH, ignore_errors=True)

opts = Options()
opts.create_if_missing(True)
opts.set_disable_auto_compactions(True)
opts.set_write_buffer_size(256 * 1024)
opts.set_level_zero_slowdown_writes_trigger(8)
opts.set_level_zero_stop_writes_trigger(1000)   # high on purpose: never block forever

db = Rdict(PATH, opts)
val = "x" * 65536   # 64 KB values so flushes (and any slowdown) are visible

def l0():
    return db.property_int_value("rocksdb.num-files-at-level0")

lat_before, lat_after = [], []
for i in range(3000):
    t = time.perf_counter()
    db[f"k{i}"] = val
    ms = (time.perf_counter() - t) * 1000
    (lat_after if l0() >= 8 else lat_before).append(ms)
    if len(lat_after) >= 200:
        break

def summary(name, xs):
    if xs:
        print(f"{name}: n={len(xs)} median={statistics.median(xs):.3f} ms  max={max(xs):.3f} ms")
    else:
        print(f"{name}: no samples")

summary("L0 < 8 ", lat_before)
summary("L0 >= 8", lat_after)
print("L0 now:", l0())

db.compact_range(None, None)
print("L0 after compact:", l0())

t = time.perf_counter()
for i in range(50):
    db[f"z{i}"] = val
print(f"50 writes after compaction: {(time.perf_counter() - t) * 1000 / 50:.3f} ms avg")

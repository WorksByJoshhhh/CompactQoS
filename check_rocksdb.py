import shutil, os, time

PATH = "testdb"
shutil.rmtree(PATH, ignore_errors=True)

def step(name, fn):
    try:
        out = fn()
        print(f"PASS  {name}  -> {out}")
        return out
    except Exception as e:
        print(f"FAIL  {name}  -> {type(e).__name__}: {e}")
        return None

from rocksdict import Rdict, Options

opts = Options()
opts.create_if_missing(True)

# Step A: turn off automatic compaction, make the memtable small so flushes happen
step("disable auto compaction", lambda: opts.set_disable_auto_compactions(True))
step("small write buffer", lambda: opts.set_write_buffer_size(256 * 1024))

db = step("open db", lambda: Rdict(PATH, opts))

# Step B: write enough data to create several flushes
def load():
    for i in range(20000):
        db[f"key{i}"] = "x" * 200
    db.flush()
    return "wrote 20000 keys"
step("write + flush", load)

# Step C: read the L0 file count
for prop in ["rocksdb.num-files-at-level0", "rocksdb.num-files-at-level1"]:
    step(f"property {prop}", lambda p=prop: db.property_value(p))

# Step D: trigger a manual compaction, then re-read L0
step("manual compact", lambda: db.compact_range(None, None))
step("L0 after compact", lambda: db.property_value("rocksdb.num-files-at-level0"))

# If any step failed, show what the object actually exposes
print("\nAvailable methods on db:")
print([m for m in dir(db) if not m.startswith("_")])
print("\nAvailable methods on Options:")
print([m for m in dir(opts) if "compact" in m or "level_zero" in m or "write_buffer" in m])

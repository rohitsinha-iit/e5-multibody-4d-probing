#!/usr/bin/env python3
"""Add or update tasks in tasks.json atomically.  Usage: python3 enqueue.py task.json [task2.json ...]
A task file holds one task dict or a list. An existing task with the same id is replaced only if it has not
started (state pending / waiting_gate) unless --force is given."""
import fcntl, json, os, sys
Q = os.path.abspath(os.environ.get("E5_QUEUE", os.path.join(os.path.expanduser(os.environ.get("E5_ROOT", "~/e5_multibody")), "queue")))
force = "--force" in sys.argv
new = []
for a in sys.argv[1:]:
    if a == "--force":
        continue
    x = json.load(open(a)); new += x if isinstance(x, list) else [x]
with open(os.path.join(Q, ".tasks.lock"), "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    p = os.path.join(Q, "tasks.json")
    tasks = json.load(open(p)) if os.path.exists(p) else []
    state = json.load(open(os.path.join(Q, "state.json"))) if os.path.exists(os.path.join(Q, "state.json")) else {}
    for t in new:
        assert {"id", "script"} <= set(t), t
        ids = [x["id"] for x in tasks]
        if t["id"] in ids:
            st = state.get(t["id"], {}).get("status", "pending")
            if st in ("pending", "waiting_gate") or force:
                tasks[ids.index(t["id"])] = t; print("updated", t["id"])
            else:
                print("kept (already", st + ")", t["id"])
        else:
            tasks.append(t); print("added", t["id"])
    json.dump(tasks, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)

#!/usr/bin/env python3
"""E5 experiment queue supervisor (stdlib only).

Runs the tasks in tasks.json one after the other. Safe to start any number of times: a lock file
guarantees a single live instance, and cron restarts it every 5 minutes and at boot, so a server
crash or reboot only costs the interrupted task's progress since its last per-scene checkpoint
(every task script is written to skip work that is already on disk).

Files (all under the queue directory, default ~/e5_multibody/queue):
  tasks.json        ordered list of tasks: {id, title, script, deps[], gate, max_attempts, min_free_gb}
  state.json        per-task status: pending | waiting_gate | running | done | failed  (+ attempts, times, rc, pushed)
  tasks/<script>    task script; runs with cwd = work/<id>, env TASK_ID, RESULT_DIR, QDIR, SRC
  results/<id>/     files the task wants pushed to GitHub (results/queue/<id>/ in the repo)
  approvals/<gate>  touch this file to open a gate (e.g. after visual approval of a pilot dataset)
  logs/<id>.log     task stdout/stderr (appended across attempts); logs/supervisor.log, logs/push.log
  events.log        one line per state change; STATUS.md is a table regenerated every loop
  STOP              if present, the supervisor finishes the current task and exits
"""
import fcntl, json, os, signal, subprocess, time

Q = os.path.abspath(os.environ.get("E5_QUEUE", os.path.expanduser("~/e5_multibody/queue")))
POLL = 30


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def jload(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def jsave(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def event(msg):
    line = f"{now()}  {msg}"
    print(line, flush=True)
    with open(os.path.join(Q, "events.log"), "a") as fh:
        fh.write(line + "\n")


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_group(pid):
    try:
        os.killpg(pid, signal.SIGTERM); time.sleep(10); os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, TypeError):
        pass


def push(tid, st):
    """Push results/<tid> to GitHub. Failures are retried every 10 minutes."""
    with open(os.path.join(Q, "logs", "push.log"), "a") as lf:
        lf.write(f"\n== {now()} push {tid}\n"); lf.flush()
        rc = subprocess.call(["bash", os.path.join(Q, "push.sh"), tid], stdout=lf, stderr=subprocess.STDOUT)
    st["pushed"] = rc == 0; st["push_rc"] = rc
    event(f"pushed {tid}" if rc == 0 else f"push {tid} not done (rc {rc}), will retry")
    return rc == 0


def read_rc(rcfile, fallback):
    try:
        return int(open(rcfile).read().strip())
    except (OSError, ValueError):
        return fallback if fallback is not None else 99


def status_push():
    """Rebuild and push the dashboard without touching any task's push state."""
    with open(os.path.join(Q, "logs", "push.log"), "a") as lf:
        lf.write(f"\n== {now()} status push\n"); lf.flush()
        subprocess.call(["bash", os.path.join(Q, "push.sh"), "_status"], stdout=lf, stderr=subprocess.STDOUT)


def write_status(tasks, state):
    lines = ["# E5 queue status", "", f"updated {now()}", "",
             "| # | task | status | attempts | started | finished | pushed |", "|---|---|---|---|---|---|---|"]
    for i, t in enumerate(tasks):
        s = state.get(t["id"], {})
        lines.append(f"| {i+1} | {t['id']}: {t.get('title','')} | {s.get('status','pending')} | {s.get('attempts',0)} | "
                     f"{s.get('started','')} | {s.get('finished','')} | {s.get('pushed','')} |")
    with open(os.path.join(Q, "STATUS.md.tmp"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(os.path.join(Q, "STATUS.md.tmp"), os.path.join(Q, "STATUS.md"))


def main():
    os.makedirs(os.path.join(Q, "logs"), exist_ok=True)
    lockf = open(os.path.join(Q, ".supervisor.lock"), "a+")  # never truncate before holding the lock
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return  # another supervisor is alive
    lockf.seek(0); lockf.truncate(); lockf.write(str(os.getpid())); lockf.flush()
    event(f"supervisor start pid {os.getpid()}")
    spath = os.path.join(Q, "state.json")
    state = jload(spath, {})
    adopt = None
    for tid, st in state.items():  # crash recovery
        if st.get("status") == "running":
            if alive(st.get("pid")) and adopt is None:
                adopt = tid; event(f"adopting still-running task {tid} (pid {st['pid']})")
            else:
                if alive(st.get("pid")):
                    kill_group(st["pid"])
                st["status"] = "pending"; st["pid"] = None; event(f"recovered {tid} (was running) -> pending")
    jsave(spath, state)

    if adopt:
        st = state[adopt]; tasks = jload(os.path.join(Q, "tasks.json"), [])
        t = next((x for x in tasks if x["id"] == adopt), {})
        last_status = 0
        while alive(st["pid"]):
            with open(os.path.join(Q, "heartbeat"), "w") as fh:
                fh.write(f"{now()} running {adopt} (adopted)\n")
            if time.time() - last_status > 3600:
                status_push(); last_status = time.time()
            time.sleep(POLL)
        rc = read_rc(os.path.join(Q, "work", adopt + ".rc"), None)
        st.update(rc=rc, finished=now(), pid=None)
        if rc == 0:
            st["status"] = "done"; event(f"done {adopt} (adopted)")
        elif st.get("attempts", 1) < t.get("max_attempts", 2):
            st["status"] = "pending"; st["not_before"] = time.time() + 60; event(f"{adopt} ended rc {rc} (adopted), will retry")
        else:
            st["status"] = "failed"; event(f"{adopt} FAILED rc {rc} (adopted)")
        jsave(spath, state)
        if st["status"] in ("done", "failed"):
            st["last_push_try"] = time.time(); push(adopt, st); jsave(spath, state)

    while True:
        if os.path.exists(os.path.join(Q, "STOP")):
            event("STOP file found, exiting"); return
        tasks = jload(os.path.join(Q, "tasks.json"), [])
        for t in tasks:
            state.setdefault(t["id"], {"status": "pending", "attempts": 0})
            if state[t["id"]].get("status") == "failed" and os.path.exists(os.path.join(Q, "approvals", "retry_" + t["id"])):
                os.remove(os.path.join(Q, "approvals", "retry_" + t["id"]))
                state[t["id"]].update(status="pending", attempts=0); event(f"manual retry {t['id']}")
        for t in tasks:  # retry pushes of finished tasks
            st = state[t["id"]]
            if st["status"] in ("done", "failed") and not st.get("pushed") and time.time() - st.get("last_push_try", 0) > 600:
                st["last_push_try"] = time.time(); push(t["id"], st)
        nxt = None
        for t in tasks:
            st = state[t["id"]]
            if st["status"] not in ("pending", "waiting_gate"):
                continue
            if not all(state.get(d, {}).get("status") == "done" for d in t.get("deps", [])):
                continue
            gate = t.get("gate")
            if gate and not os.path.exists(os.path.join(Q, "approvals", gate)):
                if st["status"] != "waiting_gate":
                    st["status"] = "waiting_gate"; event(f"{t['id']} waiting for approval gate '{gate}'")
                continue
            if time.time() < st.get("not_before", 0):
                continue
            dpath = os.path.expanduser(t.get("disk_path", "~")); need = t.get("min_free_gb", 5)
            have = free_gb(dpath if os.path.exists(dpath) else os.path.expanduser("~"))
            if have < need:
                if not st.get("disk_warned"):
                    event(f"{t['id']} blocked: free disk {have:.1f} GB < {need} GB"); st["disk_warned"] = True
                continue
            nxt = t; break
        write_status(tasks, state); jsave(spath, state)
        if nxt is None:
            time.sleep(POLL); continue

        tid = nxt["id"]; st = state[tid]
        rcfile = os.path.join(Q, "work", tid + ".rc")
        if os.path.exists(rcfile):
            os.remove(rcfile)
        work = os.path.join(Q, "work", tid); res = os.path.join(Q, "results", tid)
        os.makedirs(work, exist_ok=True); os.makedirs(res, exist_ok=True)
        env = dict(os.environ, TASK_ID=tid, RESULT_DIR=res, QDIR=Q, SRC=os.path.join(Q, "src"), PYTHONUNBUFFERED="1",
                   PATH=os.path.expanduser("~/.local/bin") + ":/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", ""))
        log = open(os.path.join(Q, "logs", f"{tid}.log"), "a")
        log.write(f"\n===== {now()} attempt {st['attempts'] + 1}\n"); log.flush()
        script = os.path.join(Q, "tasks", nxt["script"])
        p = subprocess.Popen(["bash", "-c", f'bash "{script}"; echo $? > "{rcfile}"'], cwd=work, env=env,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        st.update(status="running", pid=p.pid, started=now(), finished="", attempts=st["attempts"] + 1, disk_warned=False, pushed=False)
        jsave(spath, state); write_status(tasks, state); event(f"start {tid} pid {p.pid} attempt {st['attempts']}")
        status_push(); last_status = time.time()
        while p.poll() is None:
            if time.time() - last_status > 3600:  # hourly dashboard refresh with partial results
                status_push(); last_status = time.time()
            with open(os.path.join(Q, "heartbeat"), "w") as fh:
                fh.write(f"{now()} running {tid}\n")
            time.sleep(POLL)
        log.close()
        rc = read_rc(rcfile, p.returncode)
        st.update(rc=rc, finished=now(), pid=None)
        if rc == 0:
            st["status"] = "done"; event(f"done {tid}")
        elif st["attempts"] < nxt.get("max_attempts", 2):
            st["status"] = "pending"; st["not_before"] = time.time() + 120; event(f"{tid} failed rc {rc}, will retry")
        else:
            st["status"] = "failed"; event(f"{tid} FAILED rc {rc} after {st['attempts']} attempts")
        jsave(spath, state); write_status(tasks, state)
        if st["status"] in ("done", "failed"):
            st["last_push_try"] = time.time(); push(tid, st); jsave(spath, state)


if __name__ == "__main__":
    main()

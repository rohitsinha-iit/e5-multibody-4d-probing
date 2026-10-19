#!/usr/bin/env python3
"""List remaining hardcoded 4090-server path literals in code (exit 1 if any).

A literal is allowed only (a) inside src/e5_paths.py, or (b) on a line that also names E5_ROOT or E5_BIG,
i.e. as the default of ${E5_ROOT:-...} / ${E5_BIG:-...} / os.environ.get("E5_...", ...).
Records (results/, provenance/, data/, env/, docs/, dashboard/, logs) are not code and are skipped.
Usage: python3 scripts/check_hardcoded_paths.py [repo_root]
"""
import os, re, sys
root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
PAT = re.compile(r"/home/rohit/e5_multibody|/media/workbench/zone_x/e5_multibody|~/e5_multibody|\$HOME/e5_multibody|/home/rohit/\.cache|~/\.cache/torch")
SKIP_DIRS = {".git", "results", "provenance", "data", "env", "docs", "dashboard", "figures", "state_snapshot", "__pycache__"}
SKIP_FILES = {os.path.join("src", "e5_paths.py"), os.path.join("scripts", "check_hardcoded_paths.py")}
EXT = {".py", ".sh", ".json", ".bak_gpu", ".txt", ""}
bad = 0
for d, dirs, files in os.walk(root):
    dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
    for f in files:
        p = os.path.join(d, f); rel = os.path.relpath(p, root)
        if rel in SKIP_FILES or os.path.splitext(f)[1] not in EXT or f.endswith(".log"):
            continue
        try:
            lines = open(p, errors="replace").read().splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if PAT.search(line) and not re.search(r"E5_ROOT|E5_BIG|TORCH_HOME", line):
                print(f"{rel}:{i}: {line.strip()[:160]}"); bad += 1
print(f"remaining hardcoded path literals: {bad}")
sys.exit(1 if bad else 0)

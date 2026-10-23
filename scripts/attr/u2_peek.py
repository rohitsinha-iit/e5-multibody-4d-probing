#!/usr/bin/env python3
"""Quick look at U2 rows: python u2_peek.py <rows csv> [model] [columns...]"""
import csv, sys
R = list(csv.DictReader(open(sys.argv[1])))
m = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
cols = sys.argv[3].split(",") if len(sys.argv) > 3 else None
for r in R:
    if m and r.get("model") != m:
        continue
    ks = cols or list(r)
    print(" ".join(f"{k}={r.get(k, '')[:8]}" for k in ks if r.get(k, "") != ""))

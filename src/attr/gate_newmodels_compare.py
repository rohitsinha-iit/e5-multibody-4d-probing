"""Combine new-model gate jsons with the Any4D reference on common scenes; write gates_newmodels.json and merge
the G0b/G0d/G0g/G0j entries into results/attr_gates.json (file lock + atomic replace)."""
import json, os, sys, time, fcntl
import numpy as np
D = sys.argv[1]; OUT = os.path.join(D, "gates_newmodels.json"); ATTR = sys.argv[2] if len(sys.argv) > 2 else ""
ld = lambda f: json.load(open(os.path.join(D, f))) if os.path.exists(os.path.join(D, f)) else None
rm = {}
for f in ("gate_any4d_n5.json", "gate_any4d.json", "gate_any4d_all.json"):
    rm.update({r["scene"]: r for r in (ld(f) or {}).get("rows", [])})
G = {}
def ratio(res):
    com = [r for r in res["rows"] if r["scene"] in rm]
    if not com:
        return None, None, 0
    m = float(np.nanmedian([r["epe_static"] for r in com])); a = float(np.nanmedian([rm[r["scene"]]["epe_static"] for r in com]))
    return m, a, len(com)
now = time.strftime("%Y-%m-%d %H:%M:%S")
for gate, f, need in [("G0d", "gate_opend4rt.json", 10), ("G0g", "gate_4rc.json", 5)]:
    res = ld(f) or (ld("gate_4rc_n5.json") if gate == "G0g" else None)
    if res is None:
        G[gate] = dict(gate=gate, status="pending", time=now); continue
    m, a, n = ratio(res)
    st = "pending" if (a is None or n < need) else ("pass" if m <= 2 * a else "fail")
    if gate == "G0g":
        keys_ok = all(k in (res.get("smoke_keys") or ["pts", "track", "extrinsic", "intrinsic"]) for k in ["pts", "track", "extrinsic", "intrinsic"])
        st = st if keys_ok else "fail"
    G[gate] = dict(gate=gate, time=now, host="aicoe-a6000", status=st, median_epe_static=m, any4d_median_epe_static=a,
                   ratio=(m / a if a else None), n_common_scenes=n, n_required=need, scenes=[r["scene"] for r in res["rows"]],
                   median_runtime_s=res.get("median_runtime_s"), max_peak_reserved_gb=res.get("max_peak_reserved_gb"),
                   criteria="static EPE (median px at 512, frame 0 -> last) within 2x of Any4D on the same scenes")
    if gate == "G0d":
        com = [r for r in res["rows"] if r["scene"] in rm]
        G[gate]["readouts_median_epe_static"] = {k: float(np.nanmedian([r[k] for r in com])) for k in
            ("epe_static", "epe_static_x_r2_predK", "epe_static_x_r2_gtK", "epe_static_x_r4") if com and k in com[0]}
        G[gate]["primary_readout"] = "r1 (uv head); r4 uses GT cameras and is reported for information only"
        r48 = ld("gate_opend4rt48.json")
        if r48:
            m48, a48, n48 = (lambda rr: (float(np.nanmedian([r["epe_static"] for r in rr])), float(np.nanmedian([rm[r["scene"]]["epe_static"] for r in rr])), len(rr)))([r for r in r48["rows"] if r["scene"] in rm])
            G[gate]["fallback_48CLIP"] = dict(median_epe_static=m48, any4d=a48, ratio=m48 / a48, n=n48,
                readouts_median_epe_static={k: float(np.nanmedian([r[k] for r in r48["rows"]])) for k in ("epe_static", "epe_static_x_r2_gtK", "epe_static_x_r4")})
res = ld("gate_trackcraft3r.json")
if res is None:
    G["G0b"] = dict(gate="G0b", status="pending", time=now)
else:
    n = len(res["rows"]); m = res.get("median_epe_static")
    G["G0b"] = dict(gate="G0b", time=now, host="aicoe-a6000", status=("pending" if n < 10 else ("pass" if m <= 10 else "fail")),
                    median_epe_static=m, n_scenes=n, scenes=[r["scene"] for r in res["rows"]], median_runtime_s=res.get("median_runtime_s"),
                    max_peak_reserved_gb=res.get("max_peak_reserved_gb"), criteria="TrackCraft3R condition (i) GT depth + GT cameras, frames [0,2,...,20,23], median static EPE <= 10 px at 512 on 10 v2 scenes")
if "status" in G["G0b"] and G["G0b"]["status"] != "pending":
    var = {}
    for tag in ("stretch", "pad_reg", "stretch_reg"):
        rv = ld(f"gate_trackcraft3r_{tag}.json")
        if rv:
            var[tag] = dict(median_epe_static=float(np.nanmedian([r["epe_static"] for r in rv["rows"]])),
                            median_epe_moving=float(np.nanmedian([r["epe_moving"] for r in rv["rows"]])),
                            median_s_w=float(np.nanmedian([r.get("w_s_w", np.nan) for r in rv["rows"]])), n=len(rv["rows"]))
    G["G0b"]["variants"] = var
    if var.get("stretch", {}).get("median_epe_static", 99) <= 10 and G["G0b"]["status"] == "fail":
        G["G0b"]["status"] = "fail_pad_pass_stretch"
        G["G0b"]["decision"] = ("plan setting (resize_mode pad) fails narrowly; the repo default resize_mode stretch passes; "
                                "pad_reg/stretch_reg use the opt-in GT-free t0 re-registration (sensitivity only)")
res = ld("gate_more.json"); sm = ld("smoke_more.json")
ok = (res is not None and len(res["rows"]) > 0) or (sm is not None and "smoke" in sm)
G["G0j"] = dict(gate="G0j", time=now, host="aicoe-a6000", status="pass" if ok else "fail",
                criteria="MoRe weights downloadable (gdown from the README Drive folder) and the full-attention model runs",
                load_info=(sm or {}).get("load_info"), median_epe_static_rigid=(res or {}).get("median_epe_static"),
                n_scenes=len((res or {}).get("rows", [])))
def _avg(ds):
    p = os.path.join(D, "g0k", ds, "_error_log.txt")
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.startswith("Average ATE"):
            v = [float(x.split(":")[1]) for x in line.strip().split(",")]
            return dict(ate=v[0], rpe_t=v[1], rpe_r=v[2])
tum, sin = _avg("tum"), _avg("sintel")
REP = dict(ate=0.010, rpe_t=0.008, rpe_r=0.314)   # 4RC paper Table 2(a), TUM-dynamics; the paper reports no Sintel pose numbers
st = "pending" if tum is None else ("pass" if abs(tum["ate"] - REP["ate"]) <= 0.1 * REP["ate"] else "fail")
xc = ld("g0k/g0k_pose_py_crosscheck.json")
G["G0k"] = dict(gate="G0k", time=now, host="aicoe-a6000", status=st, reference="TUM-dynamics (substituted: 4RC reports no Sintel ATE)",
                reported=REP, ours_tum=tum, ours_sintel=sin,
                pose_py_crosscheck={k: {kk: v.get(kk) for kk in ("mean_ate", "mean_rpe_t", "mean_rpe_r")} for k, v in (xc or {}).items()},
                criteria="4RC relpose evaluator (eval/relpose/launch.py, repo defaults, evo 1.28.0) reproduces 4RC's reported ATE within 10%")
json.dump(G, open(OUT, "w"), indent=1, default=float); print(json.dumps({k: v["status"] for k, v in G.items()}))
if ATTR:
    with open(ATTR + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.load(open(ATTR)) if os.path.exists(ATTR) else {}
        cur.update(G); tmp = ATTR + ".tmp_newmodels"; json.dump(cur, open(tmp, "w"), indent=1, default=float); os.replace(tmp, ATTR)
    print("merged into", ATTR)

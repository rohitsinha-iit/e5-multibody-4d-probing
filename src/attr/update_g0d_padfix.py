"""Merge the G0d re-run (pad-to-clip fix, 2026-10-20) into results/attr_gates.json with before/after values."""
import json, os, time, fcntl
import numpy as np
O = "/data2/DATA/rohit_sinha/e5/big/results/newmodels"; ATTR = "/data2/DATA/rohit_sinha/e5/big/results/attr_gates.json"
ld = lambda f: json.load(open(os.path.join(O, f)))
rm = {}
for f in ("gate_any4d_n5.json", "gate_any4d.json", "gate_any4d_all.json"):
    if os.path.exists(os.path.join(O, f)):
        rm.update({r["scene"]: r for r in ld(f).get("rows", [])})
KEYS = ("epe_static", "epe_static_x_r2_predK", "epe_static_x_r2_gtK", "epe_static_x_r4")


def summ(res):
    com = [r for r in res["rows"] if r["scene"] in rm]
    m = float(np.nanmedian([r["epe_static"] for r in com])); a = float(np.nanmedian([rm[r["scene"]]["epe_static"] for r in com]))
    out = dict(median_epe_static=m, any4d_median_epe_static=a, ratio=m / a, n_common_scenes=len(com),
               scenes=[r["scene"] for r in com], status="pass" if (len(com) >= 10 and m <= 2 * a) else "fail",
               readouts_median_epe_static={k: float(np.nanmedian([r[k] for r in com])) for k in KEYS if k in com[0]},
               pose_ate_median=float(np.nanmedian([r.get("pose_ate", np.nan) for r in com])),
               median_runtime_s=res.get("median_runtime_s"), max_peak_reserved_gb=res.get("max_peak_reserved_gb"))
    return out


before, after = summ(ld("gate_opend4rt.json")), summ(ld("gate_opend4rt_pad32.json"))
b48, a48 = summ(ld("gate_opend4rt48.json")), summ(ld("gate_opend4rt48_pad48.json"))
assert before["scenes"] == after["scenes"], "scene sets differ"
now = time.strftime("%Y-%m-%d %H:%M:%S")
G = dict(gate="G0d", time=now, host="aicoe-a6000", status=after["status"], median_epe_static=after["median_epe_static"],
         any4d_median_epe_static=after["any4d_median_epe_static"], ratio=after["ratio"], n_common_scenes=after["n_common_scenes"],
         n_required=10, scenes=after["scenes"], median_runtime_s=after["median_runtime_s"],
         max_peak_reserved_gb=after["max_peak_reserved_gb"],
         criteria="static EPE (median px at 512, frame 0 -> last) within 2x of Any4D on the same scenes",
         readouts_median_epe_static=after["readouts_median_epe_static"],
         primary_readout="r1 (uv head); r4 uses GT cameras and is reported for information only",
         fix=("2026-10-20 readout check: the 24-frame clip is now padded to the 32-frame training clip length by repeating "
              "the last frame (as Open-d4rt training does for 24-frame Kubric scenes); before, the clip was fed unpadded, "
              "as the repo's inference helpers do. OpenD4RTAdapter(pad_to_clip=True) default; --no_pad_clip restores the old input."),
         before_unpadded=before, after_pad32=after,
         fallback_48CLIP=dict(before_unpadded=b48, after_pad48=a48),
         source_files=["results/newmodels/gate_opend4rt.json (before)", "results/newmodels/gate_opend4rt_pad32.json (after)",
                       "results/newmodels/gate_opend4rt48.json", "results/newmodels/gate_opend4rt48_pad48.json"],
         report="opend4rt_readout_check.md")
with open(ATTR + ".lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    A = json.load(open(ATTR)) if os.path.exists(ATTR) else {}
    A["G0d"] = G
    tmp = ATTR + ".tmp"; json.dump(A, open(tmp, "w"), indent=1, default=float); os.replace(tmp, ATTR)
    fcntl.flock(lk, fcntl.LOCK_UN)
print(json.dumps(dict(before={k: before[k] for k in ("median_epe_static", "ratio", "status", "readouts_median_epe_static", "pose_ate_median")},
                      after={k: after[k] for k in ("median_epe_static", "ratio", "status", "readouts_median_epe_static", "pose_ate_median")},
                      b48={k: b48[k] for k in ("median_epe_static", "ratio", "readouts_median_epe_static", "pose_ate_median")},
                      a48={k: a48[k] for k in ("median_epe_static", "ratio", "readouts_median_epe_static", "pose_ate_median")},
                      any4d=after["any4d_median_epe_static"]), default=float))
per = {r["scene"]: [round(r["epe_static"], 2)] for r in ld("gate_opend4rt.json")["rows"]}
for r in ld("gate_opend4rt_pad32.json")["rows"]:
    per[r["scene"]] += [round(r["epe_static"], 2), round(r["epe_static_x_r2_gtK"], 2), round(r["epe_static_x_r4"], 2), round(rm[r["scene"]]["epe_static"], 2)]
print(json.dumps(per))

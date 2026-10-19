"""Pilot diagnostic for G2 calibration: held-out vs training-clip EPE, zero-motion baseline, split by stride."""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3_train as pt
from p3_decoder import token_positions
from x2_eval import load_decoder
B = pt.E5_BIG; clips = os.path.join(B, "p3", "clips"); feat = os.path.join(B, "p3", "feat")
run = sys.argv[1]; dev = torch.device("cuda")
model, args, step = load_decoder(os.path.join(run, "ckpt_latest.pt"), dev)
tp = token_positions(device=dev)
va = pt.clip_list(clips, feat, "val"); tr = pt.clip_list(clips, feat, "train")
tr_s = [tr[i] for i in np.random.default_rng(0).choice(len(tr), 36, replace=False)]
out = {}
for tag, names in (("val", va), ("train36", tr_s)):
    for tt in (24, 12, 4):
        agg, rows = pt.evaluate(model, clips, feat, names, tp, dev, t_tgt=tt)
        for st in (1, 2):
            rr = [r for r in rows if f"_s{st}_" in r["clip"]]
            out[f"{tag}_t{tt}_s{st}"] = dict(epe_static=float(np.nanmean([r["epe_static"] for r in rr])), epe_moving=float(np.nanmean([r["epe_moving"] for r in rr])), n=len(rr))
        out[f"{tag}_t{tt}"] = agg
# zero-motion baseline (predict the frame-0 pixel) on val, frame 0 -> 24
zb = []
for n in va:
    lab = pt.load_clip(clips, n); co = lab["coords"]; v = lab["vis"]
    m = v[:, 0] & v[:, 24] & (co[:, 24, 0] >= 0) & (co[:, 24, 0] < 512) & (co[:, 24, 1] >= 0) & (co[:, 24, 1] < 512)
    d = np.linalg.norm(lab["Xw"][m, 24] - lab["Xw"][m, 0], axis=1) < pt.STATIC_M
    e = np.linalg.norm(co[m, 24] - co[m, 0], axis=1)
    zb.append((float(np.median(e[d])) if d.sum() > 10 else np.nan, "_s2_" in n))
out["zero_motion_val_static"] = dict(all=float(np.nanmean([a for a, _ in zb])), s1=float(np.nanmean([a for a, s in zb if not s])), s2=float(np.nanmean([a for a, s in zb if s])))
json.dump(dict(run=run, step=step, **out), open(os.path.join(run, "diag.json"), "w"), indent=1)
print("DIAG", json.dumps(out))

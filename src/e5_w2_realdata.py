"""Round 2, review item 1 (W2): RAFT on KITTI 2015 and MPI Sintel (2 views, native resolution) and the alpha/beta
decomposition for RAFT and for Any4D (cached round-1 2-view predictions), with the round-1 real-data metrics.

  predict (xm venv, GPU):  python e5_w2_realdata.py predict --gt $R/gt/kitti15 --pass rgb --pred <dir>
      torchvision raft_large, Raft_Large_Weights.C_T_V2 (Chairs + Things only; never trained on KITTI or Sintel),
      24 updates, frames meta["frames"][pass]["0"] -> ["1"] at native resolution (replicate-padded to a multiple of 8).
      Saves <pred>/<sample>.npz with flow (H,W,2) float32. Resumable.
  eval (Any4D venv, CPU):  python e5_w2_realdata.py eval --gt ... --pass ... --model raft|any4d --pred <dir> --out x.json
      Rows are e5_realdata_eval.evaluate(...) (slope, epe_static, epe_moving, sep_ratio; same sample list and seed=index
      as e5_realdata_run.py) plus, on exactly the same evaluation points:
        alpha, beta from x_pred - x1 = alpha (x_true - x_static) + beta (x_static - x1) on moving pixels with
        |x_true - x_static| > 2 px; cos_AB, cond_AB; beta_static on static pixels with |x_static - x1| > 2 px.
      RAFT is passed to evaluate() as direct correspondences (x_pred) with an identity resize geometry.
      Any4D predictions are read from the round-1 cache (e5_realdata_run.py format).
"""
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["predict", "eval"])
ap.add_argument("--gt", required=True); ap.add_argument("--pass", dest="pas", required=True)
ap.add_argument("--pred", required=True); ap.add_argument("--model", default="raft"); ap.add_argument("--out", default="")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--raft_iters", type=int, default=24)
a = ap.parse_args()

files = sorted(f for f in glob.glob(os.path.join(a.gt, "*.npz")) if not f.endswith(".tmp.npz"))
sel = list(range(len(files)))
if a.limit:
    sel = sel[:: max(1, len(files) // a.limit)][: a.limit]


def load_gt(f):
    z = np.load(f, allow_pickle=False); meta = json.loads(str(z["meta"]))
    gt = {k: z[k] for k in ("x2_true", "x2_static", "group", "valid", "label") if k in z.files}
    return meta, gt


if a.mode == "predict":
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    dev = torch.device("cuda"); torch.manual_seed(0)
    RAFT = raft_large(weights=Raft_Large_Weights.C_T_V2, progress=False).to(dev).eval()
    os.makedirs(a.pred, exist_ok=True); t0 = time.time(); n = 0
    for fi in sel:
        f = files[fi]; name = os.path.basename(f)[:-4]; pp = os.path.join(a.pred, name + ".npz")
        if os.path.exists(pp):
            continue
        meta, _ = load_gt(f); fr = meta["frames"].get(a.pas, {})
        if "0" not in fr or "1" not in fr:
            continue
        ims = [np.asarray(Image.open(fr[k]).convert("RGB")) for k in ("0", "1")]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1.0
        H, W = x.shape[-2:]; ph, pw = (-H) % 8, (-W) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        with torch.no_grad():
            fl = RAFT(x[:1], x[1:], num_flow_updates=a.raft_iters)[-1][0, :, :H, :W]
        tmp = pp[:-4] + ".tmp.npz"; np.savez_compressed(tmp, flow=fl.permute(1, 2, 0).cpu().numpy().astype(np.float32)); os.replace(tmp, pp)
        n += 1
    print("raft", a.gt, a.pas, "new", n, f"{time.time() - t0:.0f}s")
    sys.exit(0)

# ---------------------------------------------------------------- eval
import e5_realdata_eval as ev  # noqa: E402
from e5_models import native_to_model, model_to_native  # noqa: E402
from e5_w2_eval import fit_ab, pslope  # noqa: E402


def build_P(name, meta):
    pp = os.path.join(a.pred, name + ".npz")
    if not os.path.exists(pp):
        return None
    q = np.load(pp, allow_pickle=False)
    if a.model == "raft":
        H, W = meta["H"], meta["W"]; g = dict(W=W, H=H, w=W, h=H, sx=1.0, sy=1.0, left=0, top=0, ow=W, oh=H)
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64); grid = np.stack([uu, vv], -1)
        return dict(x_pred=np.stack([grid, grid + q["flow"].astype(np.float64)]), geom=[g, g])
    return dict(pts3d_0=q["pts3d_0"], scene_flow=np.stack([np.zeros_like(q["pts3d_0"]), q["sf_t"]]),
                cam_pose=q["cam_pose"][[0, 1]], K=q["K"][[0, 1]], geom=json.loads(str(q["geom"])))


def decomposition(gt, P, ti=1):
    """Same evaluation points and predicted positions as e5_realdata_eval.evaluate."""
    g0, gt_ = P["geom"][0], P["geom"][ti]
    out = ev.pred_target_px(P, ti); xm = (out[0] if isinstance(out, tuple) else out).astype(np.float64)
    h, w = xm.shape[:2]
    band = ev.boundary_band(gt["label"]) if "label" in gt else np.zeros(gt["valid"].shape, bool)
    vv, uu = np.nonzero(gt["valid"] & (gt["group"] >= 0) & ~band)
    um, vm = native_to_model(g0, uu.astype(float), vv.astype(float))
    inside = (um >= -0.5) & (um <= w - 0.5) & (vm >= -0.5) & (vm <= h - 0.5)
    uu, vv, um, vm = uu[inside], vv[inside], um[inside], vm[inside]
    grp = gt["group"][vv, uu].astype(int)
    x1 = np.c_[uu, vv].astype(float); x2 = gt["x2_true"][vv, uu].astype(float); xs = gt["x2_static"][vv, uu].astype(float)
    xp_m, _ = ev.sample_pred(xm, um, vm); xp = np.stack(model_to_native(gt_, xp_m[:, 0], xp_m[:, 1]), 1)
    fin = np.isfinite(xp).all(1); mv = grp > 0
    A = x2 - xs; B = xs - x1; y = xp - x1
    big = mv & (np.linalg.norm(A, axis=1) > 2) & fin
    r = fit_ab(A[big], B[big], y[big])
    st = (~mv) & fin & (np.linalg.norm(B, axis=1) > 2)
    r["beta_static"] = pslope(y, B, st); r["n_static_b"] = int(st.sum())
    r["median_A_px"] = float(np.median(np.linalg.norm(A[big], axis=1))) if big.any() else float("nan")
    r["median_B_px"] = float(np.median(np.linalg.norm(B[big], axis=1))) if big.any() else float("nan")
    r["slope_check"] = pslope(xp - xs, A, big)
    return r


part = a.out + ".partial"
state = json.load(open(part)) if os.path.exists(part) else dict(rows=[], skipped=[])
done = {r["scene"] for r in state["rows"]} | {s["scene"] for s in state["skipped"]}
for fi in sel:
    f = files[fi]; name = os.path.basename(f)[:-4]
    if name in done:
        continue
    meta, gt = load_gt(f)
    fr = meta["frames"].get(a.pas, {})
    if "0" not in fr or "1" not in fr:
        state["skipped"].append(dict(scene=name, reason="missing frames")); continue
    P = build_P(name, meta)
    if P is None:
        state["skipped"].append(dict(scene=name, reason="no prediction")); continue
    row = ev.evaluate(gt, meta, P, ti=1, seed=fi)
    row.update(decomposition(gt, P))
    row.update(model=a.model, dataset=meta["dataset"], pass_=a.pas, ego_rot_deg=meta.get("ego_rot_deg", float("nan")))
    state["rows"].append(row)
    json.dump(state, open(part + ".tmp", "w"), default=float); os.replace(part + ".tmp", part)
json.dump(dict(model=a.model, gt=a.gt, pass_=a.pas, pred=a.pred, **state), open(a.out + ".tmp", "w"), indent=0, default=float)
os.replace(a.out + ".tmp", a.out)
v = np.array([r["slope"] for r in state["rows"]], float)
print(a.model, a.gt.split("/")[-1], a.pas, "rows", len(state["rows"]), "skipped", len(state["skipped"]),
      "mean slope", round(float(np.nanmean(v)), 4) if len(v) else None)

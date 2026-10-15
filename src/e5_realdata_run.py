"""Item 8 driver: run a model on real-data samples and write per-scene metric rows.

Run from the Any4D repo root (cwd) with this file in Any4D/scripts/ and --src pointing at queue/src:
  python scripts/e5_realdata_run.py --src $SRC --gt $R/gt/kitti15 --pass rgb --offsets 0,1 --resize square \
      --pred $R/pred/any4d_kitti15_2v_square --out $RESULT_DIR/kitti15_2v_square.csv
Views are [t0 + o for o in offsets]; offsets[0] must be 0 (reference) and offsets[1] must be 1 (target, evaluated).
Samples whose frames are missing are skipped and listed. Resumable: predictions are cached per sample (npz) and
rows are checkpointed to <out>.partial.json after every sample; the CSV is rewritten at the end.
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--gt", required=True); ap.add_argument("--pass", dest="pas", required=True)
ap.add_argument("--offsets", default="0,1"); ap.add_argument("--resize", default="square", choices=["square", "wide"])
ap.add_argument("--model", default="any4d"); ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--check_resize", type=int, default=1)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, a.src)
import e5_models, e5_realdata_eval as ev  # noqa: E402

OFF = [int(x) for x in a.offsets.split(",")]; assert OFF[:2] == [0, 1], OFF
os.makedirs(a.pred, exist_ok=True)
files = sorted(glob.glob(os.path.join(a.gt, "*.npz")))
files = [f for f in files if not f.endswith(".tmp.npz")]
if a.limit:
    files = files[:: max(1, len(files) // a.limit)][: a.limit]
part = a.out + ".partial.json"
state = json.load(open(part)) if os.path.exists(part) else dict(rows=[], skipped=[], resize_check=[])
done = {r["scene"] for r in state["rows"]} | {s["scene"] for s in state["skipped"]}
model = None
mode = "square" if a.resize == "square" else "fixed_mapping"


def denorm_check(view, paths0, g):
    """Exactness check of the native -> model mapping: sample the native image at model pixel centres (bilinear)
    and compare with the de-normalised model input. Returns mean abs difference in 0..255 units."""
    from PIL import Image
    import cv2
    img = np.asarray(Image.open(paths0).convert("RGB")).astype(np.float32)
    vm, um = np.mgrid[0:g["h"], 0:g["w"]].astype(np.float32)
    un, vn = e5_models.model_to_native(g, um, vm)
    samp = cv2.remap(img, un.astype(np.float32), vn.astype(np.float32), cv2.INTER_LINEAR)
    x = view["img"][0].float().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    x = (x * std + mean) * 255
    shifted = cv2.remap(img, (un + 1).astype(np.float32), vn.astype(np.float32), cv2.INTER_LINEAR)
    return float(np.abs(samp - x).mean()), float(np.abs(shifted - x).mean())


t_start = time.time()
for fi, f in enumerate(files):
    name = os.path.basename(f)[:-4]
    if name in done:
        continue
    z = np.load(f, allow_pickle=False); meta = json.loads(str(z["meta"]))
    gt = {k: z[k] for k in ("x2_true", "x2_static", "group", "valid", "label") if k in z.files}
    fr = meta["frames"].get(a.pas, {})
    if not all(str(o) in fr for o in OFF):
        state["skipped"].append(dict(scene=name, reason="missing frames")); continue
    paths = [fr[str(o)] for o in OFF]
    pp = os.path.join(a.pred, name + ".npz")
    if os.path.exists(pp):
        q = np.load(pp, allow_pickle=False)
        P = dict(pts3d_0=q["pts3d_0"], scene_flow=np.stack([np.zeros_like(q["pts3d_0"]), q["sf_t"]]),
                 cam_pose=q["cam_pose"][[0, 1]], K=q["K"][[0, 1]], geom=json.loads(str(q["geom"])))
    else:
        if model is None:
            model = e5_models.get_model(a.model, resize_mode=mode)
        R = model.predict(paths)
        if a.check_resize and len(state["resize_check"]) < 3:
            mad, mad_shift = denorm_check(R["_views"][0], paths[0], R["geom"][0])
            state["resize_check"].append(dict(scene=name, mad=mad, mad_shift1px=mad_shift, geom=R["geom"][0]))
            print("resize check", name, round(mad, 2), "(1 px shift:", round(mad_shift, 2), ")", flush=True)
        sv = dict(pts3d_0=R["pts3d_0"].astype(np.float32), sf_t=R["scene_flow"][1].astype(np.float32),
                  cam_pose=R["cam_pose"].astype(np.float32), K=R["K"].astype(np.float32),
                  geom=np.array(json.dumps(R["geom"])), paths=np.array(json.dumps(paths)))
        if "conf_0" in R:
            sv["conf_0"] = R["conf_0"].astype(np.float16)
        np.savez_compressed(pp[:-4] + ".tmp.npz", **sv); os.replace(pp[:-4] + ".tmp.npz", pp)
        P = dict(pts3d_0=R["pts3d_0"], scene_flow=R["scene_flow"][[0, 1]], cam_pose=R["cam_pose"][[0, 1]],
                 K=R["K"][[0, 1]], geom=R["geom"])
    row = ev.evaluate(gt, meta, P, ti=1, seed=fi)
    row.update(dataset=meta["dataset"], pass_=a.pas, offsets=a.offsets, resize=a.resize, model=a.model,
               n_views=len(OFF), tau_px=meta.get("tau_px"),
               ego_rot_deg=meta.get("ego_rot_deg", float("nan")), ego_trans_m=meta.get("ego_trans_m", meta.get("cam_trans")))
    state["rows"].append(row)
    json.dump(state, open(part + ".tmp", "w")); os.replace(part + ".tmp", part)
    if len(state["rows"]) % 20 == 1:
        print(f"[{time.time() - t_start:.0f}s] {name} slope {row['slope']:.3f} sep_ratio {row['sep_ratio']:.3f} "
              f"epe s/m {row['epe_static']:.2f}/{row['epe_moving']:.2f}", flush=True)

rows = state["rows"]
keys = ["scene", "dataset", "pass_", "model", "resize", "offsets", "n_views", "slope", "slope_inst_median", "epe_static",
        "epe_moving", "sep_gt", "sep_pred", "sep_ratio", "n_eval", "n_moving_eval", "n_moving_total", "frac_moving_in_view",
        "frac_edge", "n_inst", "n_slope_px", "n_groups_sep", "d_true_moving_med", "tau_px", "ego_rot_deg", "ego_trans_m", "pixel_offset"]
with open(a.out + ".tmp", "w", newline="") as fh:
    wr = csv.DictWriter(fh, keys, extrasaction="ignore"); wr.writeheader()
    for r in rows:
        wr.writerow({k: (f"{r[k]:.6g}" if isinstance(r.get(k), float) else r.get(k)) for k in keys})
os.replace(a.out + ".tmp", a.out)
json.dump(dict(skipped=state["skipped"], resize_check=state["resize_check"]), open(a.out[:-4] + "_info.json", "w"), indent=1)
print("rows", len(rows), "skipped", len(state["skipped"]))

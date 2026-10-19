#!/usr/bin/env python3
"""U1 MotionFactorial GT export (unified plan 2.1) and gate G0f (plan section 4).

Per base scene and variant V in {A, B, C, D}, writes <factorial>/<set>/<seed>/<V>/gt_factorial.npz with
  K (3,3); frames (T,); c2w_blender (T,4,4) camera matrix_world (Blender: camera looks along -Z, +Y up);
  w2c_cv (T,4,4) OpenCV world->camera; rel_w2c_cv_to0 (T,4,4) = w2c_cv[t] @ inv(w2c_cv[0]) (identity if camera frozen)
  obj_pose (N,T,4,4) object matrix_world; obj_group (N,); obj_disp (N,T,4,4) = M_t @ inv(M_0) (identity if frozen)
  pts0 (H,W,3) float32 world points of frame 0 from GT depth (e5_kubric_post.backproject, the E5 GT code)
  group0 (H,W) int16 motion group of each frame-0 pixel (0 static / dome, k moving body)
  moving_mask (T,H,W) bool: pixel shows a moving-group object (group > 0) in view t (GT segmentation of that variant)
  query set (identical pixels in A, B, C and D): the E5 query set of D (e5_xm_eval.query_set, rng = default_rng(si),
    si = index of the D scene in the sorted D dataset dir), stored as q_pix (N,2) pixel (u,v), q_group (N,), q_idx (D only)
  for target views tv (default the E5 round-2 views 4, 8, 12, 16, 20, 23):
    x_true (V,N,2) projection in view t of the frame-0 point moved by its object's rigid motion 0->t
    x_static (V,N,2) projection in view t of the unmoved frame-0 point (camera motion only)
    A = x_true - x_static (object-motion image displacement); B = x_static - x1 (camera-induced displacement)
    world_disp (V,N,3) = X_t - X_0 in world coordinates; vis (V,N) visible in view t (depth test as e5_kubric_post)
Usage:
  gt_factorial.py --sets v2,v3float [--only n2_generic_s401,...] [--limit N] [--gate G0f --gate_scenes 10]
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC); sys.path.insert(0, HERE)
import e5_kubric_post as kp          # noqa: E402
import e5_xm_eval as xm              # noqa: E402
import render_factorial as rf        # noqa: E402

E5_BIG = rf.E5_BIG
VIEWS = [0, 4, 8, 12, 16, 20, 23]
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # Blender camera axes -> OpenCV


def scene_index(ds, name):
    """Index of the scene in the sorted list of ALL scenes of the dataset (= e5_xm_eval.scene_dirs order once the set is
    complete; v2 = out100). Taken from the canonical scene list so it does not depend on which renders have finished."""
    names = sorted(f"{s}_s{seed}" for s, seed in rf.scene_list(ds))
    return names.index(name), len(names)


def load(d):
    meta = json.load(open(os.path.join(d, "e5_meta.json"))); gt = np.load(os.path.join(d, "gt.npz"))
    return meta, {k: gt[k] for k in ("depth", "segmentation")}


def export(ds, split, seed, v, q, tv):
    d = rf.target(ds, split, seed, v)
    meta, gt = load(d)
    fr = [str(f) for f in meta["frames"]]; T = len(fr)
    K = np.array(meta["K"], float)
    c2w = np.stack([np.array(meta["camera_matrix_world"][f], float) for f in fr])
    w2c = np.stack([np.linalg.inv(M @ FLIP) for M in c2w])
    # exact identity when the pose is bit-identical to frame 0 (frozen factor), avoiding inverse round-off (~1e-16)
    rel = np.stack([np.eye(4) if np.array_equal(c2w[t], c2w[0]) else w2c[t] @ np.linalg.inv(w2c[0]) for t in range(T)])
    obj = np.stack([np.stack([np.array(o["matrix_world"][f], float) for f in fr]) for o in meta["objects"]])
    disp = np.stack([np.stack([np.eye(4) if np.array_equal(obj[k, t], obj[k, 0]) else obj[k, t] @ np.linalg.inv(obj[k, 0])
                               for t in range(T)]) for k in range(len(obj))])
    grp_obj = np.array([o["group"] for o in meta["objects"]], int)
    depth = gt["depth"][..., 0]; seg = gt["segmentation"][..., 0]
    s2g = kp.seg_to_group(meta)
    lut = np.zeros(int(seg.max()) + 1, np.int16)
    for sid, (g, k) in s2g.items():
        if sid < len(lut):
            lut[sid] = g
    grp = lut[seg]                       # (T,H,W)
    X = kp.backproject(meta, depth[0], 0)  # (H,W,3) world
    u, vv = q[:, 0].astype(int), q[:, 1].astype(int)
    Xq = X[vv, u]; segq = seg[0][vv, u]; gq = grp[0][vv, u]
    xt, xs, wd, vis = [], [], [], []
    H, W = depth.shape[1:]
    for t in tv:
        Xt = Xq.copy()
        for sid, (g, k) in s2g.items():
            m = segq == sid
            if k >= 0 and m.any():
                Tk = disp[k, t]; Xt[m] = Xq[m] @ Tk[:3, :3].T + Tk[:3, 3]
        pt, rt = kp.project(meta, Xt, t); ps, _ = kp.project(meta, Xq, t)
        inside = (pt[:, 0] >= 0) & (pt[:, 0] < W - 1) & (pt[:, 1] >= 0) & (pt[:, 1] < H - 1)
        ui = np.clip(np.round(pt[:, 0]).astype(int), 0, W - 1); vi = np.clip(np.round(pt[:, 1]).astype(int), 0, H - 1)
        vis.append(inside & (np.abs(depth[t][vi, ui] - rt) < 0.02 * rt))
        xt.append(pt); xs.append(ps); wd.append(Xt - Xq)
    xt, xs, wd = np.array(xt, np.float64), np.array(xs, np.float64), np.array(wd, np.float64)
    out = dict(K=K, frames=np.array(meta["frames"]), c2w_blender=c2w, w2c_cv=w2c, rel_w2c_cv_to0=rel, obj_pose=obj,
               obj_group=grp_obj, obj_disp=disp, pts0=X.astype(np.float32), group0=grp[0].astype(np.int16),
               moving_mask=(grp > 0), q_pix=q.astype(np.int32), q_group=gq.astype(np.int16), target_views=np.array(tv),
               x1=q.astype(np.float64), x_true=xt, x_static=xs, A=xt - xs, B=xs - q[None].astype(np.float64),
               world_disp=wd, vis=np.array(vis), variant=np.array(v), scene=np.array(f"{split}_s{seed}"))
    # D's directory is the shared dataset render, so its factorial GT goes to <factorial>/<set>/<seed>/D_gt/
    np.savez_compressed(os.path.join(rf.f_dir(ds, seed, "D_gt") if v == "D" else d, "gt_factorial.npz"), **out)
    return out, meta, depth


def scene(ds, split, seed, tv):
    name = f"{split}_s{seed}"
    Dd = rf.d_dir(ds, split, seed)
    si, n = scene_index(ds, name)
    metaD, gtD = load(Dd)
    q, q_xy, c, idx_sep = xm.query_set(metaD, np.load(os.path.join(Dd, "gt.npz")), si)   # unchanged E5 query set of D
    qpix = c["x1"][q].astype(int)
    os.makedirs(rf.f_dir(ds, seed, "D_gt"), exist_ok=True)
    res = {}
    for v in "DABC":
        if not os.path.exists(os.path.join(rf.target(ds, split, seed, v), "e5_meta.json")):
            continue
        out, meta, depth = export(ds, split, seed, v, qpix, tv)
        if v == "D":
            np.save(os.path.join(rf.f_dir(ds, seed, "D_gt"), "q_idx.npy"), q)
        res[v] = (out, meta, depth)
    json.dump(dict(scene=name, set=ds, scene_index=si, n_scenes_in_set=n, n_query=int(len(q)), views=VIEWS, target_views=tv,
                   variants=sorted(res), time=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(rf.f_dir(ds, seed, "D_gt"), "gt_factorial_info.json"), "w"))
    return res


def g0f_checks(ds, split, seed, res):
    """Gate G0f per scene: copy done, frame-0 PSNR >= 35 dB, frozen camera -> identity, frozen objects -> zero displacement."""
    name = f"{split}_s{seed}"; r = dict(scene=name, set=ds)
    Dd = rf.d_dir(ds, split, seed)
    d0 = open(os.path.join(Dd, "rgba_00000.png"), "rb").read()
    r["copy_done"] = all(open(os.path.join(rf.f_dir(ds, seed, v), "rgba_00000.png"), "rb").read() == d0 for v in "ABC")
    r["psnr_db"] = {v: json.load(open(os.path.join(rf.f_dir(ds, seed, v), "frame0_copy.json")))["psnr_rgb_db"] for v in "ABC"}
    oD = res["D"][0]
    mov = oD["obj_group"] > 0
    cam = {}
    for v in "AC":
        o = res[v][0]
        cam[v] = dict(max_abs_c2w_minus_c2w0=float(np.abs(o["c2w_blender"] - o["c2w_blender"][:1]).max()),
                      max_abs_rel_minus_I=float(np.abs(o["rel_w2c_cv_to0"] - np.eye(4)).max()))
    objd = {}
    for v in "AB":
        o = res[v][0]
        objd[v] = dict(max_abs_objpose_minus_pose0=float(np.abs(o["obj_pose"] - o["obj_pose"][:, :1]).max()),
                       max_abs_world_disp=float(np.abs(o["world_disp"]).max()),
                       max_abs_A_px=float(np.abs(o["A"]).max()))
    r["frozen_camera"] = cam; r["frozen_objects"] = objd
    # same-scene sanity: unfrozen factors must equal D's exactly; frame-0 state identical
    r["same_scene"] = dict(
        assets_equal=all([x["asset_id"] for x in res[v][1]["objects"]] == [x["asset_id"] for x in res["D"][1]["objects"]] for v in "ABC"),
        B_cam_vs_D=float(np.abs(res["B"][0]["c2w_blender"] - oD["c2w_blender"]).max()),
        C_obj_vs_D=float(np.abs(res["C"][0]["obj_pose"] - oD["obj_pose"]).max()),
        frame0_pose_vs_D=float(max(np.abs(res[v][0]["c2w_blender"][0] - oD["c2w_blender"][0]).max() for v in "ABC")),
        frame0_obj_vs_D=float(max(np.abs(res[v][0]["obj_pose"][:, 0] - oD["obj_pose"][:, 0]).max() for v in "ABC")),
        frame0_depth_max_abs_vs_D=float(max(np.nanmax(np.abs(res[v][2][0] - res["D"][2][0])) for v in "ABC")),
        D_moving_objects=int(mov.sum()), D_median_A_px_t23=float(np.median(np.linalg.norm(oD["A"][-1][oD["q_group"] > 0], axis=1)))
        if (oD["q_group"] > 0).any() else float("nan"))
    r["pass"] = dict(copy=bool(r["copy_done"]), psnr=bool(min(r["psnr_db"].values()) >= 35.0),
                     cam_identity=bool(all(x["max_abs_c2w_minus_c2w0"] == 0.0 and x["max_abs_rel_minus_I"] == 0.0 for x in cam.values())),
                     obj_zero=bool(all(x["max_abs_objpose_minus_pose0"] == 0.0 and x["max_abs_world_disp"] == 0.0 for x in objd.values())))
    return r


def write_gate(entry):
    for path in (os.path.join(E5_BIG, "results", "attr_gates.json"), os.path.join(SRC, "..", "results", "attr_gates.json")):
        path = os.path.abspath(path); os.makedirs(os.path.dirname(path), exist_ok=True)
        g = json.load(open(path)) if os.path.exists(path) else {}
        g[entry["gate"]] = entry
        tmp = path + ".tmp"; json.dump(g, open(tmp, "w"), indent=1, default=float); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="v2,v3float"); ap.add_argument("--only", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target_views", default="4,8,12,16,20,23"); ap.add_argument("--gate", default="")
    ap.add_argument("--skip_done", type=int, default=1)
    a = ap.parse_args(); tv = [int(x) for x in a.target_views.split(",")]
    only = set(x for x in a.only.split(",") if x); rows = []
    for ds in a.sets.split(","):
        sl = rf.scene_list(ds)
        if only:
            sl = [x for x in sl if f"{x[0]}_s{x[1]}" in only]
        if a.limit:
            sl = sl[: a.limit]
        for split, seed in sl:
            have = [v for v in "DABC" if os.path.exists(os.path.join(rf.target(ds, split, seed, v), "e5_meta.json"))]
            if "D" not in have:
                print("skip (no D)", ds, split, seed); continue
            info = os.path.join(rf.f_dir(ds, seed, "D_gt"), "gt_factorial_info.json")
            if a.skip_done and not a.gate and os.path.exists(info) and sorted(json.load(open(info))["variants"]) == sorted(have):
                continue
            res = scene(ds, split, seed, tv)
            print("gt", ds, split, seed, sorted(res), flush=True)
            if a.gate == "G0f" and all(v in res for v in "ABCD"):
                rows.append(g0f_checks(ds, split, seed, res)); print(json.dumps(rows[-1]), flush=True)
    if a.gate == "G0f":
        allp = {k: all(r["pass"][k] for r in rows) for k in ("copy", "psnr", "cam_identity", "obj_zero")} if rows else {}
        ps = [p for r in rows for p in r["psnr_db"].values()]
        entry = dict(gate="G0f", time=time.strftime("%Y-%m-%d %H:%M:%S"), host="aicoe-a6000", n_scenes=len(rows),
                     scenes=[r["scene"] for r in rows], checks=allp, status="pass" if rows and all(allp.values()) else "fail",
                     psnr_db_min=float(min(ps)) if ps else None, psnr_db_median=float(np.median(ps)) if ps else None,
                     criteria="frame-0 copy done; rendered-vs-copied frame-0 PSNR >= 35 dB; frozen camera: c2w[t] == c2w[0] exactly; "
                              "frozen objects: object pose[t] == pose[0] exactly and query world displacement == 0",
                     per_scene=rows)
        write_gate(entry); print("G0f", entry["status"], allp, entry["psnr_db_min"], flush=True)


if __name__ == "__main__":
    main()

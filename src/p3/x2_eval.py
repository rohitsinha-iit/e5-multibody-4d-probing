"""P3 X2 / U3 evaluation of trained decoders on E5 Kubric splits and MotionFactorial variants.

Per scene (scene dirs in the E5 format: e5_meta.json, gt.npz, rgba_*.png; 24 frames at 512 px):
  1. Wan features: 25 frames (frame 23 repeated) at 480 px, t = 5, final block; cached once per scene in
     $E5_BIG/p3/feat_e5/<tag>/<scene>.npy and shared by all runs.
  2. Queries: the E5 query set e5_xm_eval.query_set(meta, gt, si) (si = index in the sorted scene list of the split,
     or of --si_ref when given, matched by scene name). Decoder queried at (u, v, t_src = 0, t_tgt = 23, t_cam = 23)
     and read out through the 2D projection of X with the GT intrinsics (pixel-index convention as kp.project).
  3. Scoring: e5_w2_eval.eval_scene on the track-format prediction (x2_q on the 518 grid, q_idx): slope2d_q, alpha,
     beta, cond_AB, median_A_px, median_B_px, epe_static_q, epe_moving_q (w2_summary definitions).
     pull index PI = beta - alpha. For camera-frozen variants (--variant C or A), B = 0 and alpha := slope2d_q.
  4. World axis: w_disp = X(0, 23, t_cam = 0) - X(0, 0, t_cam = 0) in frame-0 camera coordinates; GT world displacement
     of the query (rigid object motion, e5_xm_eval.true_disp3d) rotated into frame-0 OpenCV camera coordinates; scale
     aligned per scene by the median GT/pred depth ratio on static queries. s_w = through-origin slope over moving
     queries with a GT displacement of at least 2 px-equivalent (|d| f / z). static_residual = median |w_disp| on static
     queries / median |GT disp| on moving queries. For L2D cells depth is not supervised, so s_w is descriptive only.
  5. Boundary rule: *_nb columns exclude query points within 8 px (frame 0, 512 px) of a moving-object boundary.
Summary: per run and split, the scene table plus means/medians; alpha/beta/PI use the w2 identifiability guard
(cond_AB <= 10, median |A| and |B| >= 2 px) except camera-frozen variants.
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import e5_xm_eval as xe  # noqa: E402
import e5_w2_eval as w2  # noqa: E402
import e5_kubric_post as kp  # noqa: E402
from p3_decoder import P3Decoder, token_positions  # noqa: E402
from p3_train import patches  # noqa: E402

E5_BIG = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")
NB_PX = 8.0
T_TGT = 23 / 24.0


def scene_features(wf, d, cache):
    p = os.path.join(cache, os.path.basename(d) + ".npy")
    if os.path.exists(p):
        return np.load(p)
    from wan_feats import to_video_tensor
    _, _, rgb = kp.load_scene(d)
    fr = np.concatenate([rgb, rgb[-1:]], 0) if len(rgb) == 24 else rgb
    assert len(fr) == 25, len(fr)
    f = wf.run(wf.encode(to_video_tensor(fr)), want=("final",))["final"].to(torch.float16).cpu().numpy()
    os.makedirs(cache, exist_ok=True); np.save(p + ".tmp.npy", f); os.replace(p + ".tmp.npy", p)
    return f


def cv_cam(meta, fi):
    C = np.array(meta["camera_matrix_world"][str(meta["frames"][fi])])
    return C[:3, :3], C[:3, 3]


def world_to_cv(meta, X, fi):
    R, t = cv_cam(meta, fi)
    return ((X - t) @ R) * np.array([1.0, -1.0, -1.0])


def load_decoder(ck, dev):
    s = torch.load(ck, map_location=dev, weights_only=False); a = s["args"]
    m = P3Decoder(param=a["param"], width=a["width"], depth=a["depth"]).to(dev).eval()
    m.load_state_dict(s["model"]); return m, a, s["step"]


@torch.no_grad()
def query_model(model, feat, tok_pos, uv, pt, ts, tt, tc, dev):
    Q = uv.shape[0]
    f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(torch.from_numpy(feat)[None].to(dev, torch.bfloat16), tok_pos, f(uv), f(pt),
                    f(np.full(Q, ts)), f(np.full(Q, tt)), f(np.full(Q, tc)))
    return out["X"][0].float().cpu().numpy()


def boundary_dist(meta, gt):
    from scipy import ndimage
    seg0 = gt["segmentation"][0, ..., 0]; s2g = kp.seg_to_group(meta)
    mov = np.zeros(seg0.shape, bool)
    for sid, (g, k) in s2g.items():
        if k >= 0 and g > 0:
            mov |= seg0 == sid
    b = np.zeros(seg0.shape, bool)
    b[:, 1:] |= mov[:, 1:] != mov[:, :-1]; b[1:, :] |= mov[1:, :] != mov[:-1, :]
    return ndimage.distance_transform_edt(~b)


def eval_scene(model, feat, tok_pos, d, si, dev, variant):
    meta, gt, rgb = kp.load_scene(d)
    W = meta["resolution"][0]; b = len(meta["frames"]) - 1; K = np.array(meta["K"])
    q, q_xy, c, idx_sep = xe.query_set(meta, gt, si)
    x1 = c["x1"][q]
    uv = ((x1 + 0.5) / W).astype(np.float32)
    pt = patches(rgb, np.zeros(len(q), int), x1[:, 0], x1[:, 1])
    X = query_model(model, feat, tok_pos, uv, pt, 0.0, T_TGT, T_TGT, dev)
    z = np.where(np.abs(X[:, 2]) < 1e-3, 1e-3, X[:, 2])
    xp = np.c_[K[0, 0] * X[:, 0] / z + W / 2 - 0.5, K[1, 1] * X[:, 1] / z + W / 2 - 0.5]
    s518 = xe.M518 / W
    P = dict(q_idx=q, q_xy=q_xy, x2_q=((xp + 0.5) * s518 - 0.5).astype(np.float32))
    row, objs, arr = w2.eval_scene(d, P, si)
    # camera-frozen variants: alpha is the image slope
    if variant in ("C", "A"):
        row["alpha_raw_fit"] = row.get("alpha"); row["alpha"] = row["slope2d_q"]; row["beta"] = float("nan")
    row["PI"] = row["beta"] - row["alpha"] if np.isfinite(row.get("beta", np.nan)) else float("nan")
    # boundary rule
    bd = boundary_dist(meta, gt); xi = arr["x1"].astype(int); far = bd[xi[:, 1], xi[:, 0]] >= NB_PX
    A = arr["x2_true"] - arr["x2_static"]; Bc = arr["x2_static"] - arr["x1"]; y = arr["x2_pred"] - arr["x1"]
    mv = arr["group"] > 0; fin = np.isfinite(y).all(1); big = mv & (np.linalg.norm(A, axis=1) > 2) & fin
    err = np.linalg.norm(arr["x2_pred"] - arr["x2_true"], axis=1)
    ab = w2.fit_ab(A[big & far], Bc[big & far], y[big & far])
    row.update(alpha_nb=ab["alpha"] if variant not in ("C", "A") else w2.pslope(y - 0 * Bc, A, big & far),
               beta_nb=ab["beta"] if variant not in ("C", "A") else float("nan"), cond_AB_nb=ab["cond_AB"],
               epe_static_nb=float(np.median(err[(~mv) & far])) if ((~mv) & far).any() else float("nan"),
               epe_moving_nb=float(np.median(err[mv & far])) if (mv & far).any() else float("nan"),
               n_far=int(far.sum()))
    row["PI_nb"] = row["beta_nb"] - row["alpha_nb"] if np.isfinite(row["beta_nb"]) else float("nan")
    # world axis
    Pt = query_model(model, feat, tok_pos, uv, pt, 0.0, T_TGT, 0.0, dev)
    Ps = query_model(model, feat, tok_pos, uv, pt, 0.0, 0.0, 0.0, dev)
    wd = Pt - Ps
    X0, X0b = xe.true_disp3d(meta, gt, 0, b)
    xq = x1.astype(int); Xg = X0[xq[:, 1], xq[:, 0]]; dt = world_to_cv(meta, X0b[xq[:, 1], xq[:, 0]], 0) - world_to_cv(meta, Xg, 0)
    zg = world_to_cv(meta, Xg, 0)[:, 2]
    grp = c["group"][q]; st = grp == 0
    ok = st & (Ps[:, 2] > 1e-3)
    sc = float(np.median(zg[ok] / Ps[ok, 2])) if ok.sum() > 20 else float("nan")
    wds = wd * sc
    pxeq = np.linalg.norm(dt, axis=1) * K[0, 0] / np.maximum(zg, 1e-6)
    mm = (grp > 0) & (pxeq >= 2)
    row["s_w"] = float((wds[mm] * dt[mm]).sum() / (dt[mm] ** 2).sum()) if mm.sum() > 20 else float("nan")
    row["static_residual"] = (float(np.median(np.linalg.norm(wds[st], axis=1)) / np.median(np.linalg.norm(dt[mm], axis=1)))
                              if (st.any() and mm.sum() > 20) else float("nan"))
    mmf = mm & far[: len(mm)] if len(far) == len(mm) else mm
    row["s_w_nb"] = float((wds[mmf] * dt[mmf]).sum() / (dt[mmf] ** 2).sum()) if mmf.sum() > 20 else float("nan")
    row["depth_scale"] = sc; row["n_sw"] = int(mm.sum()); row["variant"] = variant
    return row


def summarise(rows, variant):
    col = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    ok = np.ones(len(rows), bool) if variant in ("C", "A") else ((col("cond_AB") <= 10) & (col("median_A_px") >= 2) & (col("median_B_px") >= 2))
    out = dict(n_scenes=len(rows), n_identifiable=int(ok.sum()))
    for k in ("alpha", "beta", "PI", "alpha_nb", "beta_nb", "PI_nb"):
        v = col(k)[ok]; v = v[np.isfinite(v)]
        out[k] = dict(mean=float(v.mean()) if len(v) else None, median=float(np.median(v)) if len(v) else None, n=int(len(v)))
    for k in ("slope2d_q", "s_w", "s_w_nb", "static_residual", "epe_static_q", "epe_moving_q", "epe_static_nb", "epe_moving_nb"):
        v = col(k); v = v[np.isfinite(v)]
        out[k] = dict(mean=float(v.mean()) if len(v) else None, median=float(np.median(v)) if len(v) else None, n=int(len(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of run dirs (each with ckpt_latest.pt)")
    ap.add_argument("--data", required=True, help="dir of E5-format scene dirs")
    ap.add_argument("--tag", required=True, help="split tag, e.g. v2, v3float, fact_v2_C")
    ap.add_argument("--variant", default="D", choices=["A", "B", "C", "D"])
    ap.add_argument("--si_ref", default="", help="original split dir whose sorted scene list defines si (factorial)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(E5_BIG, "results", "x2_eval"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    scenes = xe.scene_dirs(a.data, a.limit)
    ref = {os.path.basename(d): i for i, d in enumerate(xe.scene_dirs(a.si_ref))} if a.si_ref else None
    cache = os.path.join(E5_BIG, "p3", "feat_e5", a.tag)
    from wan_feats import WanFeat
    wf = None
    if any(not os.path.exists(os.path.join(cache, os.path.basename(d) + ".npy")) for d in scenes):
        wf = WanFeat(dev, dtype=torch.bfloat16)
    feats = {}
    for d in scenes:
        feats[d] = scene_features(wf, d, cache)
    del wf; torch.cuda.empty_cache()
    tok_pos = token_positions(device=dev)
    for rd in a.runs.split(","):
        model, args, step = load_decoder(os.path.join(rd, "ckpt_latest.pt"), dev)
        od = os.path.join(a.out, os.path.basename(rd.rstrip("/")), a.tag); os.makedirs(od, exist_ok=True)
        rows = []
        for si, d in enumerate(scenes):
            n = os.path.basename(d); pj = os.path.join(od, n + ".json")
            if os.path.exists(pj):
                rows.append(json.load(open(pj))); continue
            sii = ref[n] if ref is not None else si
            try:
                r = eval_scene(model, feats[d], tok_pos, d, sii, dev, a.variant)
            except Exception as e:
                import traceback; r = dict(scene=n, error=repr(e), tb=traceback.format_exc()[-1500:])
            r.update(run=os.path.basename(rd.rstrip("/")), step=step, param=args["param"], loss=args["loss"], seed=args["seed"])
            json.dump(r, open(pj + ".tmp", "w"), default=float); os.replace(pj + ".tmp", pj); rows.append(r)
        good = [r for r in rows if "error" not in r]
        summ = dict(run=os.path.basename(rd.rstrip("/")), tag=a.tag, variant=a.variant, step=step, n_errors=len(rows) - len(good),
                    **summarise(good, a.variant))
        json.dump(dict(summary=summ, rows=rows), open(os.path.join(od, "_combined.json"), "w"), indent=0, default=float)
        print(json.dumps(summ, default=float)[:1500], flush=True)


if __name__ == "__main__":
    main()

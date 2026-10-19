
import sys, os, json, numpy as np
sys.path += ["/data2/DATA/rohit_sinha/e5/repo/src/attr", "/data2/DATA/rohit_sinha/e5/repo/src"]
import e5_models as em, e5_kubric_post as kp, gate_newmodels as gn, e5_xm_eval as xe
A = em.TrackCraft3RAdapter()
out = {}
fr = em.TrackCraft3RAdapter.KUBRIC_FRAMES
for sc in ["n2_generic_s401", "camrot_s305", "n2_generic_s301"]:
    d = f"/data2/DATA/rohit_sinha/e5/big/data/v2/out/{sc}"; meta, gt, _ = kp.load_scene(d)
    paths = [os.path.join(d, f"rgba_{f:05d}.png") for f in fr]
    dz, w2c, intr = A.make_inputs("gt", meta, gt, fr)
    c = kp.correspondences(meta, gt, 0, fr[-1]); x1 = c["x1"]
    st = (c["group"] == 0) & xe.static_mask(meta, gt, 0)[x1[:, 1].astype(int), x1[:, 0].astype(int)]
    for mode in ["pad", "stretch"]:
        A.p.resize_mode = mode
        P = A.predict(paths, dz, w2c, intr); g = P["geom"][0]
        row = gn.score_scene("trackcraft3r", P, meta, gt, fr)
        x0, _ = gn.sample(P["x_pred"][0], g, x1[:, 0], x1[:, 1]); x0 = np.stack(em.model_to_native(g, x0[:, 0], x0[:, 1]), -1)
        dd = (x0 - x1)[st]; L = x1[st, 0] < 256; T_ = x1[st, 1] < 256
        res = dict(epe_static=row["epe_static"], epe_moving=row["epe_moving"], s_w=row.get("w_s_w"),
                   t0_self_epe=float(np.median(np.linalg.norm(dd, axis=1))), t0_dx_med=float(np.median(dd[:, 0])), t0_dy_med=float(np.median(dd[:, 1])),
                   t0_dx_left=float(np.median(dd[L, 0])), t0_dx_right=float(np.median(dd[~L, 0])),
                   t0_dy_top=float(np.median(dd[T_, 1])), t0_dy_bottom=float(np.median(dd[~T_, 1])), runtime_s=P["_runtime_s"])
        out[f"{sc}/{mode}"] = res; print(sc, mode, json.dumps({k: round(v, 2) for k, v in res.items()}), flush=True)
json.dump(out, open("/data2/DATA/rohit_sinha/e5/big/results/newmodels/diag_tc3r_2.json", "w"), indent=1)

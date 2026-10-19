
import sys, os, json, numpy as np
sys.path += ["/data2/DATA/rohit_sinha/e5/repo/src/attr", "/data2/DATA/rohit_sinha/e5/repo/src"]
import e5_models as em, e5_kubric_post as kp, gate_newmodels as gn, e5_xm_eval as xe
A = em.TrackCraft3RAdapter()
out = {}
for sc in ["n2_generic_s401", "camrot_s305", "n2_generic_s301"]:
    d = f"/data2/DATA/rohit_sinha/e5/big/data/v2/out/{sc}"
    meta, gt, _ = kp.load_scene(d)
    for name, fr in [("kubric12", em.TrackCraft3RAdapter.KUBRIC_FRAMES), ("stride2", list(range(0, 23, 2)))]:
        paths = [os.path.join(d, f"rgba_{f:05d}.png") for f in fr]
        dz, w2c, intr = A.make_inputs("gt", meta, gt, fr)
        P = A.predict(paths, dz, w2c, intr)
        row = gn.score_scene("trackcraft3r", P, meta, gt, fr)
        T = len(fr) - 1; g = P["geom"][0]
        c = kp.correspondences(meta, gt, 0, fr[-1]); x1, x2, grp = c["x1"], c["x2"], c["group"]
        st = (grp == 0) & xe.static_mask(meta, gt, 0)[x1[:, 1].astype(int), x1[:, 0].astype(int)]
        # (a) input pointmap projected with input camera T (should match GT up to resampling)
        Pj, _ = gn.sample(P["pts3d_0"], g, x1[:, 0], x1[:, 1])
        xa = em._project_cv(em._apply_T(w2c[T] @ np.linalg.inv(w2c[0]), Pj), P["K"][T])
        xa = np.stack(em.model_to_native(g, xa[:, 0], xa[:, 1]), -1)
        # (b) t=0 self projection of the prediction
        x0, _ = gn.sample(P["x_pred"][0], g, x1[:, 0], x1[:, 1]); x0 = np.stack(em.model_to_native(g, x0[:, 0], x0[:, 1]), -1)
        # (c) |P0(T) - Pj0| on static points relative to depth
        PT, _ = gn.sample(P["tracks_cam0"][T], g, x1[:, 0], x1[:, 1])
        rel = np.linalg.norm(PT - Pj, axis=1) / np.maximum(Pj[:, 2], 1e-6)
        res = dict(epe_static=row["epe_static"], epe_moving=row["epe_moving"], s_w=row.get("w_s_w"),
                   input_pointmap_proj_epe_static=float(np.median(np.linalg.norm(xa - x2, axis=1)[st])),
                   t0_self_epe_static=float(np.median(np.linalg.norm(x0 - x1, axis=1)[st])),
                   static_track_drift_rel_depth=float(np.median(rel[st])), runtime_s=P["_runtime_s"])
        out[f"{sc}/{name}"] = res; print(sc, name, json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
json.dump(out, open("/data2/DATA/rohit_sinha/e5/big/results/newmodels/diag_tc3r.json", "w"), indent=1)

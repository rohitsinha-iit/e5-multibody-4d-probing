"""Smoke check of the GT scene-flow convention used for training: base Any4D vs scene_flow_gt on a few scenes.
Usage (cwd = Any4D root): python scripts/ft_smoke_check.py SRC SCENE_DIR [SCENE_DIR ...]"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, sys.argv[1])
import e5_ft_lib as L, e5_kubric_post as kp
from demo_inference import init_inference_model
model = init_inference_model(dict(L.CFG, checkpoint_path=L.BASE_CKPT), L.BASE_CKPT, torch.device("cuda"))
out = []
for si, d in enumerate(sys.argv[2:]):
    ds = L.KubricFT([d], fixed_frames=L.TEST_FRAMES); s = ds[0]
    with torch.no_grad():
        pr = L.forward(model, L.to_device(s["views"], "cuda"))
    P = L.preds_to_P(pr)
    mov = s["moving"].numpy(); nm = s["norm_mask"].numpy(); Fg = s["F"].numpy(); X0 = s["X0"].numpy()
    pts = P["pts3d_0"]
    cosX = float(np.median(np.sum(pts[nm] * X0[nm], -1) / (np.linalg.norm(pts[nm], axis=-1) * np.linalg.norm(X0[nm], axis=-1) + 1e-9)))
    sc = float(np.median(np.linalg.norm(X0[nm], axis=-1) / (np.linalg.norm(pts[nm], axis=-1) + 1e-9)))
    fp = P["scene_flow"][6][mov]; fg = Fg[6][mov]
    big = np.linalg.norm(fg, axis=-1) > 0.05
    cosF = float(np.median(np.sum(fp[big] * fg[big], -1) / (np.linalg.norm(fp[big], axis=-1) * np.linalg.norm(fg[big], axis=-1) + 1e-9))) if big.any() else float("nan")
    ratio = float(np.median(np.linalg.norm(fp[big], axis=-1) / np.linalg.norm(fg[big], axis=-1))) if big.any() else float("nan")
    st = np.linalg.norm(P["scene_flow"][6][~mov & nm], axis=-1)
    loss, parts = L.sf_loss(pr, s)
    meta, gt, _ = kp.load_scene(d)
    m = L.kubric_metrics(meta, gt, P, si, L.TEST_FRAMES); g = L.geometry_errors(meta, gt, P, L.TEST_FRAMES)
    out.append(dict(scene=os.path.basename(d), cos_pts_vs_gt=cosX, gt_over_pred_scale=sc, cos_flow_moving=cosF,
                    flow_mag_ratio_pred_over_gt=ratio, static_flow_median_m=float(np.median(st)), moving_frac=float(mov.mean()),
                    sf_loss=float(loss), **parts, **m, **g))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[-1].items()}), flush=True)
json.dump(out, open("ft_smoke_check.json", "w"), indent=1)

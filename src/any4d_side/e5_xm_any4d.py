"""Item 7: Any4D predictions in the cross-model format (e5_xm_eval.py). Place in Any4D/scripts/ (Hydra configs).
Same model call as e5_any4d_infer.py (frames 0,4,8,12,16,20,23, square 518, dinov2 norm, AMP). Saves <out>/<scene>.npz with
  pts0, ptsL       pointmaps (pred "pts3d") of view 0 and the last view, model world frame
  sfL              scene flow of view-0 pixels to the last view
  pts3d_0, scene_flow (V,H,W,3), cam_pose, K   as in e5_any4d_infer.py, so the dense 2D correspondences are rebuilt
                   exactly as in e5_causal.py
  q_idx, q_xy      query set (e5_xm_eval.query_set)
Resumable (skips scenes whose npz exists; atomic writes).
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from demo_inference import init_inference_model, sample_inference  # noqa: E402
from any4d.utils.image import load_images  # noqa: E402
from any4d.utils.geometry import quaternion_to_rotation_matrix, recover_pinhole_intrinsics_from_ray_directions  # noqa: E402
from any4d.utils.misc import seed_everything  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.add_argument("--e5", required=True)
p.add_argument("--frames", default="0,4,8,12,16,20,23"); p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
sys.path.insert(0, args.e5)
import e5_xm_eval as xe  # noqa: E402
import e5_kubric_post as kp  # noqa: E402
seed_everything(0)
FRAMES = [int(x) for x in args.frames.split(",")]; V = len(FRAMES)
os.makedirs(args.out, exist_ok=True)
cfg = {"path": "configs/train.yaml",
       "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
       "checkpoint_path": "checkpoints/any4d_4v_combined.pth", "trained_with_amp": True, "data_norm_type": "dinov2"}
device = torch.device("cuda")
model = init_inference_model(cfg, cfg["checkpoint_path"], device)
for si, d in enumerate(xe.scene_dirs(args.data, args.limit)):
    name = os.path.basename(d); outp = os.path.join(args.out, f"{name}.npz")
    if os.path.exists(outp):
        continue
    t0 = time.time()
    meta, gt, _ = kp.load_scene(d)
    q_idx, q_xy, _, _ = xe.query_set(meta, gt, si)
    views = load_images([os.path.join(d, f"rgba_{f:05d}.png") for f in FRAMES], resize_mode="square", size=518,
                        norm_type=cfg["data_norm_type"], patch_size=14)
    res = sample_inference(model, views, device, use_amp=True)
    pr = [res[f"pred{i+1}"] for i in range(V)]
    out = {"pts3d_0": pr[0]["pts3d"][0].float().cpu().numpy()}
    out["scene_flow"] = np.stack([np.zeros_like(out["pts3d_0"])] + [pr[i]["scene_flow"][0].float().cpu().numpy() for i in range(1, V)])
    poses, Ks = [], []
    for i in range(V):
        R = quaternion_to_rotation_matrix(pr[i]["cam_quats"][0].float().cpu())
        T = np.eye(4); T[:3, :3] = R.numpy(); T[:3, 3] = pr[i]["cam_trans"][0].float().cpu().numpy(); poses.append(T)
        Ks.append(recover_pinhole_intrinsics_from_ray_directions(pr[i]["ray_directions"][0].float().cpu()).numpy())
    out["cam_pose"] = np.stack(poses); out["K"] = np.stack(Ks)
    out["pts0"] = out["pts3d_0"]; out["ptsL"] = pr[V - 1]["pts3d"][0].float().cpu().numpy(); out["sfL"] = out["scene_flow"][V - 1]
    if "conf" in pr[0]:
        out["conf0"] = pr[0]["conf"][0].float().cpu().numpy().squeeze(); out["confL"] = pr[V - 1]["conf"][0].float().cpu().numpy().squeeze()
    out = {k: v.astype(np.float32) for k, v in out.items()}
    tmp = outp + ".tmp.npz"
    np.savez_compressed(tmp, q_idx=q_idx, q_xy=q_xy, frames=np.array(FRAMES), **out); os.replace(tmp, outp)
    print(name, round(time.time() - t0, 2), "s", flush=True)
print("peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

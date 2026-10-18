"""E5 Step 3: run Any4D on E5 Kubric scenes and save the outputs needed for E5-C.

Place in Any4D/scripts/ (Hydra config paths are resolved relative to this file, as in demo_inference.py).
For each scene: views = frames FRAMES (frame 0 is the reference view, as in the Any4D demo).
Saves <out>/<scene>.npz with, at model resolution (518 x 518):
  pts3d_0     (H,W,3)  view-0 pointmap in the model's world frame
  scene_flow  (V,H,W,3) allocentric scene flow from view 0 to each view (index 0 = view 0 itself)
  cam_pose    (V,4,4)  camera-to-world for each view
  K           (V,3,3)  pinhole intrinsics recovered from ray directions
  depth_ray   (V,H,W)  depth along ray
  conf        (V,H,W)  confidence if provided
  frames      (V,)     Kubric frame indices used
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from demo_inference import init_inference_model, sample_inference  # noqa: E402
from any4d.utils.image import load_images  # noqa: E402
from any4d.utils.geometry import quaternion_to_rotation_matrix, recover_pinhole_intrinsics_from_ray_directions  # noqa: E402
from any4d.utils.misc import seed_everything  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="directory containing scene folders")
p.add_argument("--out", required=True)
p.add_argument("--frames", default="0,4,8,12,16,20,23")
p.add_argument("--checkpoint_path", default="checkpoints/any4d_4v_combined.pth")
p.add_argument("--config_dir", default="configs")
args = p.parse_args()
seed_everything(0)
FRAMES = [int(x) for x in args.frames.split(",")]
os.makedirs(args.out, exist_ok=True)

cfg = {
    "path": f"{args.config_dir}/train.yaml",
    "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
    "checkpoint_path": args.checkpoint_path, "trained_with_amp": True, "data_norm_type": "dinov2",
}
device = torch.device("cuda")
model = init_inference_model(cfg, cfg["checkpoint_path"], device)

scenes = sorted(d for d in glob.glob(os.path.join(args.data, "*")) if os.path.exists(os.path.join(d, "e5_meta.json")))
log = []
for si, d in enumerate(scenes):
    name = os.path.basename(d)
    if os.path.exists(os.path.join(args.out, f"{name}.npz")):
        continue  # resumable
    paths = [os.path.join(d, f"rgba_{f:05d}.png") for f in FRAMES]
    t0 = time.time()
    views = load_images(paths, resize_mode="square", size=518, norm_type=cfg["data_norm_type"], patch_size=14)
    res = sample_inference(model, views, device, use_amp=True)
    if si == 0:
        print("pred keys:", {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                             for k, v in res["pred1"].items()}, flush=True)
        print("pred2 keys:", list(res["pred2"].keys()), "img shape", tuple(views[0]["img"].shape), flush=True)
    V = len(FRAMES)
    pr = [res[f"pred{i+1}"] for i in range(V)]
    out = {}
    out["pts3d_0"] = pr[0]["pts3d"][0].float().cpu().numpy()
    sf = [np.zeros_like(out["pts3d_0"])] + [pr[i]["scene_flow"][0].float().cpu().numpy() for i in range(1, V)]
    out["scene_flow"] = np.stack(sf)
    poses, Ks, dr, cf = [], [], [], []
    for i in range(V):
        R = quaternion_to_rotation_matrix(pr[i]["cam_quats"][0].float().cpu())
        T = np.eye(4); T[:3, :3] = R.numpy(); T[:3, 3] = pr[i]["cam_trans"][0].float().cpu().numpy()
        poses.append(T)
        Ks.append(recover_pinhole_intrinsics_from_ray_directions(pr[i]["ray_directions"][0].float().cpu()).numpy())
        dr.append(pr[i]["depth_along_ray"][0].float().cpu().numpy().squeeze(-1))
        if "conf" in pr[i]:
            cf.append(pr[i]["conf"][0].float().cpu().numpy().squeeze())
    out["cam_pose"] = np.stack(poses); out["K"] = np.stack(Ks); out["depth_ray"] = np.stack(dr)
    if cf:
        out["conf"] = np.stack(cf)
    out["frames"] = np.array(FRAMES)
    np.savez_compressed(os.path.join(args.out, f"{name}.npz"), **{k: v.astype(np.float32) for k, v in out.items()})
    log.append(dict(scene=name, seconds=round(time.time() - t0, 2), shape=list(out["pts3d_0"].shape)))
    print(name, log[-1]["seconds"], "s", flush=True)
json.dump(log, open(os.path.join(args.out, "infer_log.json"), "w"), indent=1)
print("peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

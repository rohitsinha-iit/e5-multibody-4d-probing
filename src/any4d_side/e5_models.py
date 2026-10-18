"""Model adapters for item 8 (real-data output-level evaluation). Model-agnostic interface:

    model = get_model(name, **kw)
    P = model.predict(image_paths)          # list of V paths, index 0 = reference view
P is a dict (numpy, model resolution h x w):
    pts3d_0    (h,w,3)    reference-view pointmap in the model's world frame
    scene_flow (V,h,w,3)  allocentric scene flow from view 0 to each view (index 0 zeros), or None
    cam_pose   (V,4,4)    camera-to-world
    K          (V,3,3)    pinhole intrinsics at model resolution
    conf_0     (h,w)      optional confidence
    x_pred     (V,h,w,2)  optional: direct predicted correspondences of view-0 pixels in each view (model px);
                          if present it is used instead of the pointmap + scene flow projection
    geom       list of V dicts mapping native pixels to model pixels:
               u_model = (u_native + 0.5) * sx - 0.5 - left,  v_model = (v_native + 0.5) * sy - 0.5 - top
               (W, H native size; w, h model size). This is exactly Any4D's load_images resize (isotropic rescale
               so the image covers the target, floor of the scaled size, then a centred crop).
A new model only needs a class with predict() returning this dict.
"""
import math, os, sys
import numpy as np


def resize_geom(W, H, tw, th):
    """Replicates any4d.utils.cropping.crop_resize_if_necessary (no intrinsics given)."""
    scale = max(tw / W, th / H) + 1e-8
    ow, oh = int(math.floor(W * scale)), int(math.floor(H * scale))
    left, top = (ow - tw) // 2, (oh - th) // 2
    return dict(W=W, H=H, w=tw, h=th, sx=ow / W, sy=oh / H, left=left, top=top, ow=ow, oh=oh)


def native_to_model(g, u, v):
    return (u + 0.5) * g["sx"] - 0.5 - g["left"], (v + 0.5) * g["sy"] - 0.5 - g["top"]


def model_to_native(g, um, vm):
    return (um + g["left"] + 0.5) / g["sx"] - 0.5, (vm + g["top"] + 0.5) / g["sy"] - 0.5


class Any4D:
    """Any4D (MapAnything fine-tuned for 4D). Same loading as e5_any4d_infer.py. Run with cwd = Any4D repo root and
    this file importable next to scripts/demo_inference.py."""
    name = "any4d"

    def __init__(self, resize_mode="square", size=518, checkpoint_path="checkpoints/any4d_4v_combined.pth",
                 config_dir="configs"):
        import torch
        from demo_inference import init_inference_model, sample_inference
        from any4d.utils.image import load_images, find_closest_aspect_ratio
        from any4d.utils.geometry import quaternion_to_rotation_matrix, recover_pinhole_intrinsics_from_ray_directions
        from any4d.utils.misc import seed_everything
        seed_everything(0)
        self.torch = torch; self.sample_inference = sample_inference; self.load_images = load_images
        self.q2R = quaternion_to_rotation_matrix; self.rec_K = recover_pinhole_intrinsics_from_ray_directions
        self.fcar = find_closest_aspect_ratio
        self.resize_mode = resize_mode; self.size = size
        cfg = {"path": f"{config_dir}/train.yaml",
               "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
               "checkpoint_path": checkpoint_path, "trained_with_amp": True, "data_norm_type": "dinov2"}
        self.cfg = cfg; self.device = torch.device("cuda")
        self.model = init_inference_model(cfg, checkpoint_path, self.device)

    def target_size(self, W, H):
        if self.resize_mode == "square":
            s = (self.size // 14) * 14; return s, s
        if self.resize_mode == "fixed_mapping":
            return self.fcar(W / H, 518)
        raise ValueError(self.resize_mode)

    def predict(self, paths):
        from PIL import Image
        torch = self.torch
        sizes = [Image.open(p).size for p in paths]
        if self.resize_mode == "fixed_mapping":   # load_images uses the mean aspect ratio over the list
            ar = sum(w / h for w, h in sizes) / len(sizes); tw, th = self.fcar(ar, 518)
        else:
            tw, th = self.target_size(*sizes[0])
        views = self.load_images(list(paths), resize_mode=self.resize_mode,
                                 size=self.size if self.resize_mode == "square" else None,
                                 norm_type=self.cfg["data_norm_type"], patch_size=14)
        geom = [resize_geom(w, h, tw, th) for (w, h) in sizes]
        for vw, g in zip(views, geom):
            assert tuple(vw["img"].shape[-2:]) == (g["h"], g["w"]), (vw["img"].shape, g)
        with torch.no_grad():
            res = self.sample_inference(self.model, views, self.device, use_amp=True)
        V = len(paths); pr = [res[f"pred{i+1}"] for i in range(V)]
        out = {"pts3d_0": pr[0]["pts3d"][0].float().cpu().numpy()}
        out["scene_flow"] = np.stack([np.zeros_like(out["pts3d_0"])] +
                                     [pr[i]["scene_flow"][0].float().cpu().numpy() for i in range(1, V)])
        poses, Ks = [], []
        for i in range(V):
            T = np.eye(4); T[:3, :3] = self.q2R(pr[i]["cam_quats"][0].float().cpu()).numpy()
            T[:3, 3] = pr[i]["cam_trans"][0].float().cpu().numpy(); poses.append(T)
            Ks.append(self.rec_K(pr[i]["ray_directions"][0].float().cpu()).numpy())
        out["cam_pose"] = np.stack(poses); out["K"] = np.stack(Ks)
        if "conf" in pr[0]:
            out["conf_0"] = pr[0]["conf"][0].float().cpu().numpy().squeeze()
        out["geom"] = geom
        out["_views"] = views
        return out


def get_model(name, **kw):
    if name == "any4d":
        return Any4D(**kw)
    raise KeyError(f"unknown model {name}; add an adapter class with predict(paths) to e5_models.py")

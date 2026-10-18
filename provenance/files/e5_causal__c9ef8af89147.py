"""Item 4: causal story for the five object-motion supply heads (E8 follow-up). Place in Any4D/scripts/.

Heads (block:head, from the single-unit screen, replicated on v2): 2:1, 22:9, 17:10 (frame block), 14:6, 8:11.
Head intervention (pre-hook on attn.proj, per head h):  x_h <- m_h + alpha * (x_h - m_h), m_h = token mean
  alpha = 0  : mean ablation (as in the screen);  alpha > 1 : amplification.
Routing (path patching). Any4D's decoders read four taps of the residual stream: [encoder, block 11, block 17, final].
  The scene-flow DPT head reads all four; the geometry DPT head reads all four; the pose head reads the final tap.
  *_sfonly : clean transformer, but the scene-flow head receives the intervened run's taps
             (the intervention acts only through the scene-flow decoder)
  *_nonsf  : intervened transformer, but the scene-flow head receives the clean taps
             (the intervention acts only through the geometry and pose decoders)
  abl5_tap{k}: clean run; scene-flow head receives the ablated run's tap k only (k = 0 enc, 1 b11, 2 b17, 3 final)
Controls: 3 random sets of 5 heads matched in block type (4 global-attention heads + 1 frame-attention head).
Metrics per scene (frame 0 -> 23, 512 px GT frame), as in e5_e8_knockout.py:
  slope (fraction of object motion recovered), epe_static, epe_moving (median px), sep_ratio.
Resumable: per-scene rows are checkpointed to <out>.partial.
"""
import argparse, dataclasses, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from demo_inference import init_inference_model, sample_inference  # noqa: E402
from any4d.utils.image import load_images  # noqa: E402
from any4d.utils.geometry import quaternion_to_rotation_matrix, recover_pinhole_intrinsics_from_ray_directions  # noqa: E402
from any4d.utils.misc import seed_everything  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--e5", required=True); p.add_argument("--out", required=True)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--heads", default="2:1,22:9,17:10,14:6,8:11")
args = p.parse_args()
sys.path.insert(0, args.e5)
import e5_kubric_post as kp  # noqa: E402
import e5_step3_analysis as s3  # noqa: E402
seed_everything(0)
FRAMES = [0, 4, 8, 12, 16, 20, 23]; TV = 6
cfg = {"path": "configs/train.yaml",
       "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
       "checkpoint_path": "checkpoints/any4d_4v_combined.pth", "trained_with_amp": True, "data_norm_type": "dinov2"}
model = init_inference_model(cfg, cfg["checkpoint_path"], torch.device("cuda"))
blocks = model.info_sharing.self_attention_blocks
NB = len(blocks); NH = blocks[0].attn.num_heads; HD = blocks[0].attn.head_dim
FIVE = [tuple(int(v) for v in x.split(":")) for x in args.heads.split(",")]

rng = np.random.default_rng(0)
glob_pool = [(b, h) for b in range(0, NB, 2) for h in range(NH) if (b, h) not in FIVE]
frame_pool = [(b, h) for b in range(1, NB, 2) for h in range(NH) if (b, h) not in FIVE]
n_frame = sum(b % 2 for b, _ in FIVE)
RAND = []
for r in range(3):
    g = [glob_pool[i] for i in rng.choice(len(glob_pool), len(FIVE) - n_frame, replace=False)]
    f = [frame_pool[i] for i in rng.choice(len(frame_pool), n_frame, replace=False)]
    RAND.append(g + f)

# ------------------------------------------------ hooks
active = {"scale": {}, "capture": None, "sf_in": None}   # scale: {block: [(head, alpha)]}


def make_proj_prehook(l):
    def h(mod, inp):
        hs = active["scale"].get(l)
        if not hs:
            return None
        x = inp[0].clone()
        for hh, a in hs:
            sl = x[..., hh * HD:(hh + 1) * HD]
            m = sl.mean(dim=-2, keepdim=True)
            x[..., hh * HD:(hh + 1) * HD] = m + a * (sl - m)
        return (x,)
    return h


for l, b in enumerate(blocks):
    b.attn.proj.register_forward_pre_hook(make_proj_prehook(l))


def sf_prehook(mod, inp):
    x = inp[0]
    if active["capture"] is not None:
        active["capture"][:] = [t.detach().clone() for t in x.list_features]
    if active["sf_in"] is not None:
        return (dataclasses.replace(x, list_features=list(active["sf_in"])),)
    return None


model.scene_flow_dense_head.register_forward_pre_hook(sf_prehook)


def run(views):
    res = sample_inference(model, views, torch.device("cuda"), use_amp=True)
    pr = [res[f"pred{i+1}"] for i in range(len(FRAMES))]
    out = {"pts3d_0": pr[0]["pts3d"][0].float().cpu().numpy()}
    out["scene_flow"] = np.stack([np.zeros_like(out["pts3d_0"])] + [pr[i]["scene_flow"][0].float().cpu().numpy() for i in range(1, len(FRAMES))])
    poses, Ks = [], []
    for i in range(len(FRAMES)):
        R = quaternion_to_rotation_matrix(pr[i]["cam_quats"][0].float().cpu())
        Tm = np.eye(4); Tm[:3, :3] = R.numpy(); Tm[:3, 3] = pr[i]["cam_trans"][0].float().cpu().numpy(); poses.append(Tm)
        Ks.append(recover_pinhole_intrinsics_from_ray_directions(pr[i]["ray_directions"][0].float().cpu()).numpy())
    out["cam_pose"] = np.stack(poses); out["K"] = np.stack(Ks)
    return out


def scale_spec(heads, alpha):
    d = {}
    for b, h in heads:
        d.setdefault(b, []).append((h, alpha))
    return d


INTERVENTIONS = {"abl5": scale_spec(FIVE, 0.0)}
for r, hs in enumerate(RAND):
    INTERVENTIONS[f"rand5_r{r}"] = scale_spec(hs, 0.0)
for a in (1.5, 2.0, 3.0):
    INTERVENTIONS[f"amp5_a{a:g}"] = scale_spec(FIVE, a)
for r, hs in enumerate(RAND):
    INTERVENTIONS[f"amprand_a2_r{r}"] = scale_spec(hs, 2.0)
ROUTED = ["abl5", "amp5_a2"]

scenes = sorted(d for d in glob.glob(os.path.join(args.data, "*")) if os.path.exists(os.path.join(d, "e5_meta.json")))
if args.limit:
    scenes = scenes[: args.limit]
rows = json.load(open(args.out + ".partial"))["rows"] if os.path.exists(args.out + ".partial") else []
done = {r["scene"] for r in rows}
for si, d in enumerate(scenes):
    n = os.path.basename(d)
    if n in done:
        continue
    meta, gt, _ = kp.load_scene(d)
    W = meta["resolution"][0]; s = 518 / W; b = len(meta["frames"]) - 1
    c = kp.correspondences(meta, gt, 0, b)
    X = kp.backproject(meta, gt["depth"][0, ..., 0], 0); xs_dense, _ = kp.project(meta, X, b)
    x1 = c["x1"]; grp = c["group"]
    xs = xs_dense[x1[:, 1].astype(int), x1[:, 0].astype(int)]
    u = np.clip(np.round((x1[:, 0] + 0.5) * s - 0.5).astype(int), 0, 517); v = np.clip(np.round((x1[:, 1] + 0.5) * s - 0.5).astype(int), 0, 517)
    x1p = (np.c_[u, v] + 0.5) / s - 0.5
    rs = np.random.default_rng(si); idx = s3.sample_by_group(grp, 300, rs)
    sep_gt, _ = s3.sep_and_noise(x1[idx], c["x2"][idx], grp[idx])
    views = load_images([os.path.join(d, f"rgba_{f:05d}.png") for f in FRAMES], resize_mode="square", size=518,
                        norm_type=cfg["data_norm_type"], patch_size=14)

    def measure(cname, P):
        off = np.array(s3.pixel_offset(P))
        xp = s3.pred_correspondences(P, TV, off)[v, u]; xp = (xp + 0.5) / s - 0.5
        err = np.linalg.norm(xp - c["x2"], axis=1); mv = grp > 0
        dt = c["x2"] - xs; dp = xp - xs; big = mv & (np.linalg.norm(dt, axis=1) > 2)
        sep_pr, _ = s3.sep_and_noise(x1p[idx], xp[idx], grp[idx])
        rows.append(dict(scene=n, split=meta["split"], condition=cname,
                         slope=float((dp[big] * dt[big]).sum() / (dt[big] ** 2).sum()) if big.sum() > 20 else float("nan"),
                         epe_static=float(np.median(err[~mv])), epe_moving=float(np.median(err[mv])) if mv.any() else float("nan"),
                         sep_gt=float(sep_gt), sep_pred=float(sep_pr), sep_ratio=float(sep_pr / sep_gt) if sep_gt > 1 else float("nan")))

    def go(cname, scale=None, capture=None, sf_in=None):
        active["scale"] = scale or {}; active["capture"] = capture; active["sf_in"] = sf_in
        with torch.no_grad():
            P = run(views)
        active["scale"] = {}; active["capture"] = None; active["sf_in"] = None
        if cname:
            measure(cname, P)

    clean = []; go("base", capture=clean)
    taps = {}
    for name, spec in INTERVENTIONS.items():
        cap = [] if name in ROUTED else None
        go(name, scale=spec, capture=cap)
        if cap is not None:
            taps[name] = cap
    for name in ROUTED:
        go(f"{name}_sfonly", sf_in=taps[name])
        go(f"{name}_nonsf", scale=INTERVENTIONS[name], sf_in=clean)
    for k in range(len(clean)):
        mix = [taps["abl5"][j] if j == k else clean[j] for j in range(len(clean))]
        go(f"abl5_tap{k}", sf_in=mix)
    json.dump(dict(rows=rows), open(args.out + ".partial.tmp", "w")); os.replace(args.out + ".partial.tmp", args.out + ".partial")
    print(n, len(rows), flush=True)

json.dump(dict(rows=rows, five=FIVE, random_sets=RAND, taps=["encoder", "block11", "block17", "final"]), open(args.out, "w"))
print("done", len(rows))

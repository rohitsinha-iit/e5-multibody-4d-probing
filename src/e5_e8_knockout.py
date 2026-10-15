"""E8 for E5 claim C2: where does the pull of moving bodies toward the camera-induced motion arise?

Place in Any4D/scripts/. Inputs: E5 v1 scenes (data/out), per-patch GT (gt_patches), attention summaries (internals/*_att.npz).
Conditions (all on the same 7 views, frame 0 reference, target view 6 = frame 23):
  base                      no intervention
  skip_b{a}-{b}             skip blocks a..b (block output := block input) for windows of 4 blocks
  ko_track_k12 / ko_static_k12 / ko_rand{r}_k12
                            zero 12 heads in global-attention blocks (heads concatenated before attn.proj).
                            Heads ranked by per-head hit rate on the OTHER half of the scenes (2-fold by seed parity):
                            track = moving patches' argmax lands on the true moved location; static = static patches' argmax
                            lands on their GT match. Random = 12 random global heads (3 draws, same for all scenes).
Per scene and condition (pair frame 0 -> 23, 512 px GT frame):
  slope      fraction of object motion recovered on moving pixels: sum(d_pred . d_true) / sum(|d_true|^2), d = x - x_static
  epe_static / epe_moving   median endpoint error (px)
  sep_ratio  multibody separation of predicted correspondences / GT separation (GT groups, 300 pts per group)
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
p.add_argument("--data", required=True); p.add_argument("--gt", required=True); p.add_argument("--att", required=True)
p.add_argument("--e5", required=True, help="dir with e5_multibody.py, e5_kubric_post.py, e5_step3_analysis.py")
p.add_argument("--out", required=True); p.add_argument("--limit", type=int, default=0)
p.add_argument("--screen", type=int, default=0, help="1 = single-unit mean-ablation screen (all heads, all blocks)")
p.add_argument("--per_split", type=int, default=0, help="screen: scenes per non-degenerate split")
args = p.parse_args()
sys.path.insert(0, args.e5)
import e5_kubric_post as kp  # noqa: E402
import e5_step3_analysis as s3  # noqa: E402
seed_everything(0)
FRAMES = [0, 4, 8, 12, 16, 20, 23]; TV = 6; G = 37; T = G * G; PATCH = 14
cfg = {"path": "configs/train.yaml",
       "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
       "checkpoint_path": "checkpoints/any4d_4v_combined.pth", "trained_with_amp": True, "data_norm_type": "dinov2"}
model = init_inference_model(cfg, cfg["checkpoint_path"], torch.device("cuda"))
blocks = model.info_sharing.self_attention_blocks
NB = len(blocks); NH = blocks[0].attn.num_heads; HD = blocks[0].attn.head_dim

scenes = sorted(d for d in glob.glob(os.path.join(args.data, "*")) if os.path.exists(os.path.join(d, "e5_meta.json")))
if args.limit:
    scenes = scenes[: args.limit]
if args.per_split:
    keep, cnt = [], {}
    for d in scenes:
        sp = os.path.basename(d).rsplit("_s", 1)[0]
        if sp == "epi_aligned":
            continue
        if cnt.get(sp, 0) < args.per_split:
            keep.append(d); cnt[sp] = cnt.get(sp, 0) + 1
    scenes = keep
names = [os.path.basename(d) for d in scenes]
seed_of = {n: int(n.split("_s")[-1]) for n in names}

# ------------------------------------------------ per-head hit rates (global blocks), per scene
def cheb_hit(idx, target):
    cx, cy = idx % G, idx // G
    tx, ty = (target[:, 0] - 6.5) / PATCH, (target[:, 1] - 6.5) / PATCH
    return (np.abs(cx - tx) <= 1) & (np.abs(cy - ty) <= 1)

track = np.full((len(names), NB // 2, NH), np.nan); stat = np.full_like(track, np.nan)
for i, n in enumerate(names):
    g = dict(np.load(os.path.join(args.gt, f"{n}_gt.npz"), allow_pickle=True))
    pure = (g["purity"] >= 0.8) & g["vis"]
    M = pure & (g["body"] > 0) & (np.linalg.norm(g["x2_true"] - g["x2_static"], axis=1) > 2 * PATCH)
    S = pure & (g["body"] == 0)
    am = np.load(os.path.join(args.att, f"{n}_att.npz"))["argmax"].astype(int)
    for gi in range(NB // 2):
        for h in range(NH):
            if M.sum() >= 5:
                track[i, gi, h] = cheb_hit(am[gi, h][M], g["x2_true"][M]).mean()
            stat[i, gi, h] = cheb_hit(am[gi, h][S], g["x2_true"][S]).mean()

def top_heads(score, k=12):
    flat = np.argsort(-np.nan_to_num(score, nan=-1).ravel())[:k]
    return [(int(f // NH) * 2, int(f % NH)) for f in flat]  # (block index, head)

heads_for = {}
for par in (0, 1):
    other = [i for i, n in enumerate(names) if seed_of[n] % 2 != par]
    heads_for[par] = dict(track=top_heads(np.nanmean(track[other], 0)), static=top_heads(np.nanmean(stat[other], 0)))
rng = np.random.default_rng(0)
all_heads = [(2 * gi, h) for gi in range(NB // 2) for h in range(NH)]
rand_sets = [[all_heads[j] for j in rng.choice(len(all_heads), 12, replace=False)] for _ in range(3)]

# ------------------------------------------------ interventions
active = {"skip": set(), "ko": {}, "mean_head": {}, "mean_block": set()}

def make_block_hook(l):
    def h(mod, inp, out):
        if l in active["skip"]:
            return inp[0]
        if l in active["mean_block"]:  # replace this block's residual update by its token-mean
            delta = out - inp[0]
            return inp[0] + delta.mean(dim=-2, keepdim=True)
        return out
    return h

def make_proj_prehook(l):
    def h(mod, inp):
        hs = active["ko"].get(l); hm = active["mean_head"].get(l)
        if not hs and not hm:
            return None
        x = inp[0].clone()
        for hh in (hs or []):
            x[..., hh * HD:(hh + 1) * HD] = 0
        for hh in (hm or []):  # mean-ablation: head output replaced by its mean over tokens
            sl = x[..., hh * HD:(hh + 1) * HD]
            x[..., hh * HD:(hh + 1) * HD] = sl.mean(dim=-2, keepdim=True).expand_as(sl)
        return (x,)
    return h

for l, b in enumerate(blocks):
    b.register_forward_hook(make_block_hook(l))
    b.attn.proj.register_forward_pre_hook(make_proj_prehook(l))

conditions = [("base", set(), [])]
if args.screen:
    conditions = [("base", set(), [])] + [(f"mh_b{l}_h{h}", ("mh", l, h), []) for l in range(NB) for h in range(NH)] \
                 + [(f"mb_b{l}", ("mb", l), []) for l in range(NB)]
for a in (range(0, NB, 4) if not args.screen else []):
    conditions.append((f"skip_b{a}-{a+3}", set(range(a, a + 4)), []))
if not args.screen:
    conditions += [("ko_track_k12", set(), "track"), ("ko_static_k12", set(), "static")]
    conditions += [(f"ko_rand{r}_k12", set(), rand_sets[r]) for r in range(3)]

def run(views):
    res = sample_inference(model, views, torch.device("cuda"), use_amp=True)
    pr = [res[f"pred{i+1}"] for i in range(len(FRAMES))]
    out = {"pts3d_0": pr[0]["pts3d"][0].float().cpu().numpy(),
           "scene_flow": np.stack([np.zeros_like(pr[0]["pts3d"][0].float().cpu().numpy())] +
                                  [pr[i]["scene_flow"][0].float().cpu().numpy() for i in range(1, len(FRAMES))])}
    poses, Ks = [], []
    for i in range(len(FRAMES)):
        R = quaternion_to_rotation_matrix(pr[i]["cam_quats"][0].float().cpu())
        Tm = np.eye(4); Tm[:3, :3] = R.numpy(); Tm[:3, 3] = pr[i]["cam_trans"][0].float().cpu().numpy(); poses.append(Tm)
        Ks.append(recover_pinhole_intrinsics_from_ray_directions(pr[i]["ray_directions"][0].float().cpu()).numpy())
    out["cam_pose"] = np.stack(poses); out["K"] = np.stack(Ks)
    return out

rows = []
if os.path.exists(args.out + ".partial"):
    rows = json.load(open(args.out + ".partial"))["rows"]
done = {r["scene"] for r in rows}
for si, (d, n) in enumerate(zip(scenes, names)):
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
    for cname, skip, heads in conditions:
        if isinstance(heads, str):
            heads = heads_for[seed_of[n] % 2][heads]
        active["skip"] = skip if isinstance(skip, set) else set(); active["ko"] = {}
        active["mean_head"] = {}; active["mean_block"] = set()
        if isinstance(skip, tuple):
            if skip[0] == "mh":
                active["mean_head"] = {skip[1]: [skip[2]]}
            else:
                active["mean_block"] = {skip[1]}
        for (bl, hh) in heads:
            active["ko"].setdefault(bl, []).append(hh)
        P = run(views)
        off = np.array(s3.pixel_offset(P))
        xp = s3.pred_correspondences(P, TV, off)[v, u]; xp = (xp + 0.5) / s - 0.5
        err = np.linalg.norm(xp - c["x2"], axis=1)
        mv = (grp > 0)
        dt = c["x2"] - xs; dp = xp - xs
        big = mv & (np.linalg.norm(dt, axis=1) > 2)
        sep_pr, _ = s3.sep_and_noise(x1p[idx], xp[idx], grp[idx])
        rows.append(dict(scene=n, split=meta["split"], condition=cname,
                         slope=float((dp[big] * dt[big]).sum() / (dt[big] ** 2).sum()) if big.sum() > 20 else np.nan,
                         epe_static=float(np.median(err[~mv])), epe_moving=float(np.median(err[mv])) if mv.any() else np.nan,
                         sep_gt=float(sep_gt), sep_pred=float(sep_pr), sep_ratio=float(sep_pr / sep_gt) if sep_gt > 1 else np.nan))
    active["skip"] = set(); active["ko"] = {}; active["mean_head"] = {}; active["mean_block"] = set()
    json.dump(dict(rows=rows), open(args.out + ".partial.tmp", "w")); os.replace(args.out + ".partial.tmp", args.out + ".partial")
    print(n, flush=True)
json.dump(dict(rows=rows, heads_for={str(k): v for k, v in heads_for.items()}, rand_sets=rand_sets,
               track_head_mean=np.nanmean(track, 0).tolist(), static_head_mean=np.nanmean(stat, 0).tolist()),
          open(args.out, "w"))
print("done", len(rows))

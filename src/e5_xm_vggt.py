"""Item 7: VGGT-family models (VGGT-1B static, PAGE-4D dynamic fine-tune with the VGGT architecture) on E5 Kubric scenes.

One forward pass per scene (frames 0,4,8,12,16,20,23 at 518 x 518, view 0 = reference) saves
  <out>/pred/<scene>.npz      in the e5_xm_eval.py format: pts0/ptsL from depth + predicted cameras (VGGT's recommended
                              route), pts0_head/ptsL_head from the point head, cam_pose (V,4,4 camera-to-world),
                              K (V,3,3), conf, tracks of the query set (x2_q, x2_q0 = frame-0 echo, vis/conf) and of the
                              37x37 patch centres (x2_patch) from the tracking head
  <out>/internals/<scene>_tok.npy  float16 (L, 2, 1369, 1024): residual stream of views 0 and last at the stored layers.
       Layer numbering over the 48 alternating blocks (frame block i = block 2i, global block i = block 2i+1):
       layer 0 = aggregator input (patch tokens), layer l = output of block l-1. Default layers 0,8,...,48 = relative
       depth 0, 1/6, ..., 1. Only patch tokens are stored (the 5 camera/register tokens per frame are dropped).
  <out>/internals/<scene>_att.npz  for every global block g (24) and head h (16): argmax / soft / pmax of the
       attention of view-0 patch queries over last-view patch keys (same definitions as e5_any4d_extract.py), with the
       query/key normalisation and 2D RoPE of the model applied before the dot product.
  stored_layers.json, att_blocks.json (block index of each global block), grid.json.
Resumable: a scene is skipped when both its pred npz and att npz exist; files are written atomically.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5_xm_eval as xe  # noqa: E402
import e5_kubric_post as kp  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", required=True, help="path to a VGGT-architecture checkpoint")
ap.add_argument("--frames", default="0,4,8,12,16,20,23"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--layers", default="0,8,16,24,32,40,48"); ap.add_argument("--no_internals", action="store_true")
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--track_ckpt", default="", help="optional checkpoint whose track_head.* weights replace the model's (e.g. facebook/VGGT_tracker_fixed)")
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
FR = [int(x) for x in a.frames.split(",")]; V = len(FR); TV = V - 1
STORE = [int(x) for x in a.layers.split(",")]
PD, ID = os.path.join(a.out, "pred"), os.path.join(a.out, "internals")
os.makedirs(PD, exist_ok=True); os.makedirs(ID, exist_ok=True)
dev = torch.device("cuda")
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT()
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
    sd = sd["model"]
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
res = model.load_state_dict(sd, strict=False)
print("missing", len(res.missing_keys), res.missing_keys[:8], "unexpected", len(res.unexpected_keys), res.unexpected_keys[:8], flush=True)
assert not [k for k in res.missing_keys if not k.startswith("track_head")], "checkpoint does not match the VGGT architecture"
HAS_TRACK = model.track_head is not None and not any(k.startswith("track_head") for k in res.missing_keys)
if a.track_ckpt:
    tsd = torch.load(a.track_ckpt, map_location="cpu", weights_only=False)
    tsd = tsd.get("model", tsd) if isinstance(tsd, dict) else tsd
    tsd = {k[7:] if k.startswith("module.") else k: v for k, v in tsd.items()}
    tsd = {k[len("track_head."):]: v for k, v in tsd.items() if k.startswith("track_head.")}
    r2 = model.track_head.load_state_dict(tsd, strict=True); HAS_TRACK = True
    print("track head replaced from", a.track_ckpt, len(tsd), "tensors", flush=True)
model = model.to(dev).eval()
agg = model.aggregator
assert list(getattr(agg, "aa_order", ["frame", "global"])) == ["frame", "global"] and getattr(agg, "aa_block_size", 1) == 1
FB, GB = agg.frame_blocks, agg.global_blocks; NBLK = len(FB) + len(GB); NG = len(GB)
NH = GB[0].attn.num_heads; HD = GB[0].attn.head_dim
G = 37; T = G * G
json.dump(STORE, open(os.path.join(ID, "stored_layers.json"), "w"))
json.dump([2 * g + 1 for g in range(NG)], open(os.path.join(ID, "att_blocks.json"), "w"))
json.dump(dict(grid=G, patch=14, size=518, n_blocks=NBLK, n_global=NG, heads=NH, dim=GB[0].attn.qkv.in_features,
               layer_def="0 = aggregator input; l = output of block l-1; frame block i = block 2i, global block i = 2i+1",
               has_track=bool(HAS_TRACK)), open(os.path.join(ID, "grid.json"), "w"), indent=1)
print("blocks", NBLK, "global", NG, "heads", NH, "head_dim", HD, "track head", HAS_TRACK, flush=True)
grid = torch.stack(torch.meshgrid(torch.arange(G), torch.arange(G), indexing="xy"), -1).reshape(T, 2).float().to(dev)
st = {}
PS = [None]


def views_of(x, frame):
    """frame-block tensors are (S, P, C); global-block tensors (1, S*P, C). Returns (2, T, C) patch tokens of views 0, TV."""
    x = x.reshape(V, -1, x.shape[-1]) if not frame else x
    ps = x.shape[1] - T; PS[0] = ps
    return torch.stack([x[0, ps:], x[TV, ps:]]).half().cpu()


def pre0(mod, inp):
    if 0 in STORE:
        st["tok"][0] = views_of(inp[0], True)


def out_hook(l, frame):
    def h(mod, inp, out):
        if l + 1 in STORE:
            st["tok"][l + 1] = views_of(out, frame)
    return h


def wrap_attn(gi, attn):
    orig = attn.forward

    def fwd(x, pos=None, *args, **kw):
        if st.get("on"):
            with torch.no_grad():
                B, N, C = x.shape; P = N // V; ps = P - T
                qkv = attn.qkv(x).reshape(B, N, 3, NH, HD).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                q, k = attn.q_norm(q), attn.k_norm(k)
                if getattr(attn, "rope", None) is not None and pos is not None:
                    q = attn.rope(q, pos); k = attn.rope(k, pos)
                q0 = q[0, :, ps:P].float(); kL = k[0, :, TV * P + ps:(TV + 1) * P].float()
                lg = (q0 @ kL.transpose(1, 2)) * attn.scale; pr = lg.softmax(-1)
                st["argmax"][gi] = lg.argmax(-1).short().cpu(); st["soft"][gi] = (pr @ grid).half().cpu()
                st["pmax"][gi] = pr.max(-1).values.half().cpu()
        return orig(x, pos=pos, *args, **kw)
    attn.forward = fwd


if not a.no_internals:
    FB[0].register_forward_pre_hook(pre0)
    for i in range(len(FB)):
        FB[i].register_forward_hook(out_hook(2 * i, True)); GB[i].register_forward_hook(out_hook(2 * i + 1, False))
        wrap_attn(i, GB[i].attn)


def load(d):
    ims = [np.asarray(Image.open(os.path.join(d, f"rgba_{f:05d}.png")).convert("RGB").resize((518, 518), Image.BICUBIC)) for f in FR]
    return torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().div(255).to(dev)


def save_npz(path, **kw):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **kw); os.replace(tmp, path)


scenes = xe.scene_dirs(a.data, a.limit)
PC = xe.patch_centres(G, 14)
for si, d in enumerate(scenes):
    n = os.path.basename(d); pz = os.path.join(PD, n + ".npz"); az = os.path.join(ID, n + "_att.npz")
    if os.path.exists(pz) and (a.no_internals or os.path.exists(az)):
        continue
    t0 = time.time()
    meta, gt, _ = kp.load_scene(d)
    q_idx, q_xy, _, _ = xe.query_set(meta, gt, si)
    st.clear(); st.update(tok=[None] * (NBLK + 1), argmax=[None] * NG, soft=[None] * NG, pmax=[None] * NG, on=not a.no_internals)
    img = load(d)[None]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype):
            toks, ps_idx = agg(img)
        st["on"] = False
        toks = [t.float() if t is not None else None for t in toks]   # newer VGGT keeps only the layers the heads read
        with torch.autocast("cuda", enabled=False):
            pose_enc = model.camera_head(toks)[-1]
            depth, dconf = model.depth_head(toks, images=img, patch_start_idx=ps_idx)
            wpts, wconf = model.point_head(toks, images=img, patch_start_idx=ps_idx)
            tr = {}
            if HAS_TRACK:
                allq = torch.from_numpy(np.concatenate([q_xy, PC])).to(dev)
                outs, viss, confs = [], [], []
                for j in range(0, len(allq), a.chunk):
                    tl, vis, cf = model.track_head(toks, images=img, patch_start_idx=ps_idx, query_points=allq[None, j:j + a.chunk])
                    outs.append(tl[-1][0]); viss.append(vis[0]); confs.append(cf[0] if cf is not None else vis[0])
                trk = torch.cat(outs, 1).cpu().numpy(); vis = torch.cat(viss, 1).cpu().numpy(); cf = torch.cat(confs, 1).cpu().numpy()
                nq = len(q_xy)
                tr = dict(x2_q=trk[TV, :nq], x2_q0=trk[0, :nq], vis_q=vis[TV, :nq], conf_q=cf[TV, :nq], x2_patch=trk[TV, nq:], vis_patch=vis[TV, nq:])
    extr, intr = pose_encoding_to_extri_intri(pose_enc, img.shape[-2:])
    extr = extr[0].cpu().numpy(); intr = intr[0].cpu().numpy()
    dep = depth[0].cpu().numpy()
    pts = unproject_depth_map_to_point_map(dep, extr, intr)
    c2w = np.tile(np.eye(4), (V, 1, 1)); 
    for i in range(V):
        c2w[i, :3, :3] = extr[i, :, :3].T; c2w[i, :3, 3] = -extr[i, :, :3].T @ extr[i, :, 3]
    wp = wpts[0].cpu().numpy()
    save_npz(pz, pts0=pts[0].astype(np.float32), ptsL=pts[TV].astype(np.float32), pts0_head=wp[0].astype(np.float32),
             ptsL_head=wp[TV].astype(np.float32), cam_pose=c2w.astype(np.float32), K=intr.astype(np.float32),
             conf0=dconf[0, 0].cpu().numpy().astype(np.float16), confL=dconf[0, TV].cpu().numpy().astype(np.float16),
             q_idx=q_idx, q_xy=q_xy, frames=np.array(FR), **{k: v.astype(np.float32) for k, v in tr.items()})
    if not a.no_internals:
        assert all(st["tok"][l] is not None for l in STORE) and all(x is not None for x in st["argmax"]), "hooks did not fire"
        tmp = os.path.join(ID, n + "_tok.tmp.npy"); np.save(tmp, torch.stack([st["tok"][l] for l in STORE]).numpy())
        os.replace(tmp, os.path.join(ID, n + "_tok.npy"))
        save_npz(az, argmax=torch.stack(st["argmax"]).numpy(), soft=torch.stack(st["soft"]).numpy(), pmax=torch.stack(st["pmax"]).numpy())
    if si == 0:
        print("special tokens per frame", PS[0], "depth", tuple(depth.shape), "tracks", {k: v.shape for k, v in tr.items()}, flush=True)
    print(n, f"{time.time() - t0:.1f}s", flush=True)
print("peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

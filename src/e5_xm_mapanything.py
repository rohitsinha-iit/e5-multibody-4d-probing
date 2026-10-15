"""Item 7: MapAnything (static-trained; the model Any4D was fine-tuned from) on E5 Kubric scenes.

Same views as Any4D (frames 0,4,8,12,16,20,23, square 518, dinov2 normalisation). Saves
  <out>/pred/<scene>.npz      e5_xm_eval.py format: pts0/ptsL (world pointmaps of view 0 and the last view, from the
                              model's pts3d output), cam_pose (camera-to-world), K, conf. No 2D motion output.
  <out>/internals/<scene>_tok.npy float16 (L, 2, 1369, 768) residual stream of views 0 and last, layers as in
                              e5_any4d_extract.py (0 = transformer input, l = output of block l-1), default 0,4,...,24
  <out>/internals/<scene>_att.npz argmax/soft/pmax for the 12 global blocks x 12 heads, computed exactly as in
                              e5_any4d_extract.py (q.k from the qkv projection of view-0 queries vs last-view keys)
Resumable (per-scene files, atomic writes).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5_xm_eval as xe  # noqa: E402
import e5_kubric_post as kp  # noqa: E402
from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import load_images  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", required=True, help="local snapshot dir of facebook/map-anything")
ap.add_argument("--frames", default="0,4,8,12,16,20,23"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--layers", default="0,4,8,12,16,20,24")
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
FR = [int(x) for x in a.frames.split(",")]; V = len(FR); TV = V - 1
STORE = [int(x) for x in a.layers.split(",")]
PD, ID = os.path.join(a.out, "pred"), os.path.join(a.out, "internals")
os.makedirs(PD, exist_ok=True); os.makedirs(ID, exist_ok=True)
dev = torch.device("cuda")
model = MapAnything.from_pretrained(a.ckpt).to(dev).eval()
blocks = model.info_sharing.self_attention_blocks
NB = len(blocks); NG = (NB + 1) // 2; NH = blocks[0].attn.num_heads
G = 37; T = G * G
_a0 = blocks[0].attn
FLAGS = {k: bool(getattr(_a0, k, None)) for k in ["custom_positional_encoding", "use_scalable_softmax", "use_entropy_scaling"]}
FLAGS["qk_norm"] = not isinstance(_a0.q_norm, torch.nn.Identity)
print("attention flags (all must be False for raw q.k summaries to be exact)", FLAGS, flush=True)
assert not any(FLAGS.values()), FLAGS
json.dump(STORE, open(os.path.join(ID, "stored_layers.json"), "w"))
json.dump([2 * g for g in range(NG)], open(os.path.join(ID, "att_blocks.json"), "w"))
json.dump(dict(grid=G, patch=14, size=518, n_blocks=NB, n_global=NG, heads=NH, dim=blocks[0].attn.qkv.in_features,
               layer_def="0 = transformer input; l = output of block l-1; even blocks global, odd frame (as Any4D)"),
          open(os.path.join(ID, "grid.json"), "w"), indent=1)
grid = torch.stack(torch.meshgrid(torch.arange(G), torch.arange(G), indexing="xy"), -1).reshape(T, 2).float().to(dev)
st = {}
SHAPES = []


def take_views(x):
    SHAPES.append(tuple(x.shape))
    if x.dim() == 3 and x.shape[0] == V:
        assert x.shape[1] == T, x.shape
        return torch.stack([x[0], x[TV]])
    x = x[0, : V * T].reshape(V, T, -1)
    return torch.stack([x[0], x[TV]])


def pre0(mod, inp):
    st["tok"][0] = take_views(inp[0]).half().cpu()


def out_hook(l):
    def h(mod, inp, out):
        st["tok"][l + 1] = take_views(out[0] if isinstance(out, tuple) else out).half().cpu()
    return h


def qkv_hook(gi, attn):
    def h(mod, inp, out):
        x = out[0, : V * T].float(); C = x.shape[-1] // 3
        q = x[:T, :C].reshape(T, attn.num_heads, attn.head_dim).transpose(0, 1)
        k = x[TV * T:(TV + 1) * T, C:2 * C].reshape(T, attn.num_heads, attn.head_dim).transpose(0, 1)
        lg = (q @ k.transpose(1, 2)) * attn.scale; pr = lg.softmax(-1)
        st["argmax"][gi] = lg.argmax(-1).short().cpu(); st["soft"][gi] = (pr @ grid).half().cpu(); st["pmax"][gi] = pr.max(-1).values.half().cpu()
    return h


blocks[0].register_forward_pre_hook(pre0)
for l, b in enumerate(blocks):
    b.register_forward_hook(out_hook(l))
    if l % 2 == 0:
        b.attn.qkv.register_forward_hook(qkv_hook(l // 2, b.attn))
print("blocks", NB, "heads", NH, "attn children", {k: type(v).__name__ for k, v in blocks[0].attn.named_children()}, flush=True)


def save_npz(path, **kw):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **kw); os.replace(tmp, path)


def load(d):
    paths = [os.path.join(d, f"rgba_{f:05d}.png") for f in FR]
    try:
        views = load_images(paths, resize_mode="square", size=518, norm_type="dinov2", patch_size=14)
    except TypeError:
        views = load_images(paths)
    assert tuple(views[0]["img"].shape[-2:]) == (518, 518), views[0]["img"].shape
    return views


for si, d in enumerate(xe.scene_dirs(a.data, a.limit)):
    n = os.path.basename(d); pz = os.path.join(PD, n + ".npz"); az = os.path.join(ID, n + "_att.npz")
    if os.path.exists(pz) and os.path.exists(az):
        continue
    t0 = time.time()
    meta, gt, _ = kp.load_scene(d)
    q_idx, q_xy, _, _ = xe.query_set(meta, gt, si)
    st.clear(); st.update(tok=[None] * (NB + 1), argmax=[None] * NG, soft=[None] * NG, pmax=[None] * NG); SHAPES.clear()
    views = load(d)
    with torch.no_grad():
        pr = model.infer(views, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16", apply_mask=False,
                         mask_edges=False, apply_confidence_mask=False)
    if si == 0:
        print("pred keys", {k: tuple(v.shape) for k, v in pr[0].items() if hasattr(v, "shape")}, "hook shapes", SHAPES[:2], flush=True)
    g = lambda i, k: pr[i][k][0].float().cpu().numpy()
    out = dict(pts0=g(0, "pts3d"), ptsL=g(TV, "pts3d"), cam_pose=np.stack([g(i, "camera_poses") for i in range(V)]),
               K=np.stack([g(i, "intrinsics") for i in range(V)]))
    if "conf" in pr[0]:
        out["conf0"] = g(0, "conf").squeeze(); out["confL"] = g(TV, "conf").squeeze()
    assert out["pts0"].shape == (518, 518, 3), out["pts0"].shape
    assert all(t is not None for t in st["tok"]) and all(x is not None for x in st["argmax"]), "hooks did not fire"
    save_npz(pz, q_idx=q_idx, q_xy=q_xy, frames=np.array(FR), **{k: v.astype(np.float32) for k, v in out.items()})
    tmp = os.path.join(ID, n + "_tok.tmp.npy"); np.save(tmp, torch.stack([st["tok"][l] for l in STORE]).numpy())
    os.replace(tmp, os.path.join(ID, n + "_tok.npy"))
    save_npz(az, argmax=torch.stack(st["argmax"]).numpy(), soft=torch.stack(st["soft"]).numpy(), pmax=torch.stack(st["pmax"]).numpy())
    print(n, f"{time.time() - t0:.1f}s", flush=True)
print("peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

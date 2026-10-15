"""E5 Steps 4-6: extract Any4D internals for the E5 Kubric scenes. Place in Any4D/scripts/.

Views = frames 0,4,8,12,16,20,23 (view 0 = reference, view 6 = frame 23), 518x518, 37x37 patch tokens per view.
info_sharing = 24-block alternating-attention transformer (even block idx = global attention, odd = frame attention).
Per scene saves <out>/<scene>_tok.npy  float16 (25, 2, T, 768): residual stream for views 0 and 6
                 (index 0 = transformer input = encoder output + view PE; index l+1 = output of block l)
               <out>/<scene>_att.npz  for each global block g (12 of them) and head h (12):
                 argmax (12,12,T) int16      view-6 key patch with max q.k for each view-0 query patch
                 soft   (12,12,T,2) float16  softmax-weighted (col,row) patch position over view-6 keys
                 pmax   (12,12,T) float16    max of that conditional softmax
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from demo_inference import init_inference_model, sample_inference  # noqa: E402
from any4d.utils.image import load_images  # noqa: E402
from any4d.utils.misc import seed_everything  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--out", required=True)
p.add_argument("--frames", default="0,4,8,12,16,20,23"); p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
seed_everything(0)
FRAMES = [int(x) for x in args.frames.split(",")]; V = len(FRAMES); TV = V - 1
os.makedirs(args.out, exist_ok=True)
cfg = {"path": "configs/train.yaml",
       "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
       "checkpoint_path": "checkpoints/any4d_4v_combined.pth", "trained_with_amp": True, "data_norm_type": "dinov2"}
model = init_inference_model(cfg, cfg["checkpoint_path"], torch.device("cuda"))
info = model.info_sharing
blocks = info.self_attention_blocks
G = 37; T = G * G
grid = torch.stack(torch.meshgrid(torch.arange(G), torch.arange(G), indexing="xy"), -1).reshape(T, 2).float().cuda()  # (col,row)

store = {}

def take_views(x):
    """x: (1, V*T + extra, C) or (V, T, C) -> (2, T, C) for views 0 and TV."""
    if x.dim() == 3 and x.shape[0] == V:
        return torch.stack([x[0], x[TV]])
    x = x[0, : V * T].reshape(V, T, -1)
    return torch.stack([x[0], x[TV]])

def pre0(mod, inp):
    store["tok"][0] = take_views(inp[0]).half().cpu()

def make_out_hook(l):
    def h(mod, inp, out):
        store["tok"][l + 1] = take_views(out).half().cpu()
    return h

def make_qkv_hook(gi, attn):
    def h(mod, inp, out):
        x = out[0, : V * T].float()  # (V*T, 3C)
        C = x.shape[-1] // 3
        q = x[:T, :C].reshape(T, attn.num_heads, attn.head_dim).transpose(0, 1)               # view 0 queries
        k = x[TV * T:(TV + 1) * T, C:2 * C].reshape(T, attn.num_heads, attn.head_dim).transpose(0, 1)  # view TV keys
        lg = (q @ k.transpose(1, 2)) * attn.scale                                             # (H,T,T)
        pr = lg.softmax(-1)
        store["argmax"][gi] = lg.argmax(-1).short().cpu()
        store["soft"][gi] = (pr @ grid).half().cpu()
        store["pmax"][gi] = pr.max(-1).values.half().cpu()
    return h

handles = [blocks[0].register_forward_pre_hook(pre0)]
for l, b in enumerate(blocks):
    handles.append(b.register_forward_hook(make_out_hook(l)))
    if l % 2 == 0:
        handles.append(b.attn.qkv.register_forward_hook(make_qkv_hook(l // 2, b.attn)))

scenes = sorted(d for d in glob.glob(os.path.join(args.data, "*")) if os.path.exists(os.path.join(d, "e5_meta.json")))
if args.limit:
    scenes = scenes[: args.limit]
for d in scenes:
    name = os.path.basename(d); t0 = time.time()
    store.update(tok=[None] * 25, argmax=[None] * 12, soft=[None] * 12, pmax=[None] * 12)
    views = load_images([os.path.join(d, f"rgba_{f:05d}.png") for f in FRAMES], resize_mode="square", size=518,
                        norm_type=cfg["data_norm_type"], patch_size=14)
    sample_inference(model, views, torch.device("cuda"), use_amp=True)
    assert all(t is not None for t in store["tok"]) and all(a is not None for a in store["argmax"])
    np.save(os.path.join(args.out, f"{name}_tok.npy"), torch.stack(store["tok"]).numpy())
    np.savez_compressed(os.path.join(args.out, f"{name}_att.npz"), argmax=torch.stack(store["argmax"]).numpy(),
                        soft=torch.stack(store["soft"]).numpy(), pmax=torch.stack(store["pmax"]).numpy())
    print(name, f"{time.time()-t0:.1f}s", flush=True)
print("peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

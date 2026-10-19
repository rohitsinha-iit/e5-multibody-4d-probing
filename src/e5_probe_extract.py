"""Round 2, review W4 (plan item 6): internals of control models for the probe controls. Place in Any4D/scripts/.

Variants (same architecture and the same model call as e5_any4d_extract.py: frames 0,4,8,12,16,20,23, square 518,
dinov2 norm, AMP; views 0 and 6 stored, same token layout, float16):
  trained : Any4D checkpoint any4d_4v_combined.pth (only used to dump the raw encoder tokens, and as a check that this
            script reproduces the round-1 internals exactly).
  randtf  : pretrained DINOv2 ViT-L/14 encoder (the public LVD-142M weights from the torch hub cache, which is what
            Any4D is initialised from) + the rest of Any4D (input projection, fusion norm, 24-block transformer,
            heads) at the model's own random initialisation, seeded. No Any4D checkpoint is loaded.
  randall : everything random: as randtf, but the DINOv2 encoder is also replaced by a freshly initialised
            ViT-L/14 (dinov2 hub constructor with pretrained=False), seeded.
Outputs (resumable, atomic):
  <out>/<scene>_tok.npy      (len(layers), 2, T, 768) residual stream at --layers (0 = transformer input =
                             projected encoder output + reference-view PE, l = output of block l-1), as round 1
  <out>/stored_layers.json
  <enc_out>/<scene>_tok.npy  (1, 2, T, 1024) raw encoder patch tokens (before fusion norm, projection and view PE)
  <enc_out>/stored_layers.json = [0]
  <out>/norms.json           per scene and layer: mean token L2 norm and non-finite count (sanity)
Only scenes that have <gt>/<scene>_gt.npz are processed (the probes need the GT patches).
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from demo_inference import init_inference_model, sample_inference  # noqa: E402
from any4d.utils.image import load_images  # noqa: E402
from any4d.utils.misc import seed_everything  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--gt", required=True)
p.add_argument("--variant", required=True, choices=["trained", "randtf", "randall"])
p.add_argument("--out", default=""); p.add_argument("--enc_out", default="")
p.add_argument("--layers", default="0,4,8,12,16,20,24"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
FRAMES = [0, 4, 8, 12, 16, 20, 23]; V = len(FRAMES); TV = V - 1
STORE = [int(x) for x in args.layers.split(",")] if args.layers else []
seed_everything(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
cfg = {"path": "configs/train.yaml",
       "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
       "checkpoint_path": "checkpoints/any4d_4v_combined.pth", "trained_with_amp": True, "data_norm_type": "dinov2"}
dev = torch.device("cuda")
info = {"variant": args.variant, "seed": args.seed}
if args.variant == "trained":
    model = init_inference_model(cfg, cfg["checkpoint_path"], dev)
else:
    model = init_inference_model(cfg, None, dev)  # encoder: pretrained DINOv2 from the hub cache; rest: random init
    if args.variant == "randall":
        torch.manual_seed(args.seed + 1)
        hub = os.path.join(os.path.expanduser(os.environ.get("TORCH_HOME", "~/.cache/torch")), "hub", "facebookresearch_dinov2_main")
        fresh = torch.hub.load(hub, "dinov2_vitl14", source="local", pretrained=False)
        sd = {k: v for k, v in fresh.state_dict().items() if k in model.encoder.model.state_dict()}
        r = model.encoder.model.load_state_dict(sd, strict=False)
        info["encoder_reinit_missing"] = list(r.missing_keys); info["encoder_reinit_loaded"] = len(sd)
        assert len(r.missing_keys) == 0, r.missing_keys
        model.to(dev).eval()
# record how far the encoder is from the public DINOv2 weights (is Any4D's encoder fine-tuned?)
pub = torch.load(os.path.join(os.path.expanduser(os.environ.get("TORCH_HOME", "~/.cache/torch")), "hub", "checkpoints", "dinov2_vitl14_pretrain.pth"), map_location="cpu")
msd = model.encoder.model.state_dict(); diffs = []
for k, v in pub.items():
    if k in msd and msd[k].shape == v.shape:
        diffs.append(float((msd[k].float().cpu() - v.float()).abs().max()))
info["encoder_max_abs_diff_to_public_dinov2"] = max(diffs) if diffs else None
info["encoder_n_params_compared"] = len(diffs)
blocks = model.info_sharing.self_attention_blocks
G = 37; T = G * G
for d in [args.out, args.enc_out]:
    if d:
        os.makedirs(d, exist_ok=True)
if args.out:
    json.dump(STORE, open(os.path.join(args.out, "stored_layers.json"), "w"))
    json.dump(info, open(os.path.join(args.out, "variant_info.json"), "w"), indent=1)
if args.enc_out:
    json.dump([0], open(os.path.join(args.enc_out, "stored_layers.json"), "w"))
    json.dump(info, open(os.path.join(args.enc_out, "variant_info.json"), "w"), indent=1)
print(json.dumps(info), flush=True)

store = {}


def take_views(x):
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


def enc_hook(mod, inp, out):
    f = out.features if hasattr(out, "features") else out  # (V, 1024, 37, 37)
    f = f.reshape(f.shape[0], f.shape[1], -1).transpose(1, 2)  # (V, T, 1024)
    store["enc"] = torch.stack([f[0], f[TV]]).float().cpu()


blocks[0].register_forward_pre_hook(pre0)
for l, b in enumerate(blocks):
    b.register_forward_hook(make_out_hook(l))
model.encoder.register_forward_hook(enc_hook)

scenes = sorted(d for d in glob.glob(os.path.join(args.data, "*")) if os.path.exists(os.path.join(d, "e5_meta.json"))
                and os.path.exists(os.path.join(args.gt, os.path.basename(d) + "_gt.npz")))
if args.limit:
    scenes = scenes[: args.limit]
normp = os.path.join(args.out or args.enc_out, "norms.json")
norms = json.load(open(normp)) if os.path.exists(normp) else {}
for d in scenes:
    name = os.path.basename(d)
    to = os.path.join(args.out, f"{name}_tok.npy") if args.out else None
    te = os.path.join(args.enc_out, f"{name}_tok.npy") if args.enc_out else None
    if (to is None or os.path.exists(to)) and (te is None or os.path.exists(te)):
        continue
    t0 = time.time()
    store.update(tok=[None] * 25, enc=None)
    views = load_images([os.path.join(d, f"rgba_{f:05d}.png") for f in FRAMES], resize_mode="square", size=518,
                        norm_type=cfg["data_norm_type"], patch_size=14)
    sample_inference(model, views, dev, use_amp=True)
    rec = {}
    if to:
        arr = torch.stack([store["tok"][l] for l in STORE]).numpy()
        rec["tok_norm"] = [float(np.linalg.norm(arr[i].astype(np.float32), axis=-1).mean()) for i in range(len(STORE))]
        rec["tok_nonfinite"] = int((~np.isfinite(arr)).sum())
        np.save(to + ".tmp.npy", arr); os.replace(to + ".tmp.npy", to)
    if te:
        e = store["enc"].numpy()
        rec["enc_norm"] = float(np.linalg.norm(e, axis=-1).mean()); rec["enc_absmax"] = float(np.abs(e).max())
        e16 = e.astype(np.float16)[None]
        rec["enc_nonfinite"] = int((~np.isfinite(e16)).sum())
        np.save(te + ".tmp.npy", e16); os.replace(te + ".tmp.npy", te)
    norms[name] = rec
    json.dump(norms, open(normp + ".tmp", "w")); os.replace(normp + ".tmp", normp)
    print(name, f"{time.time()-t0:.1f}s", flush=True)
print("peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

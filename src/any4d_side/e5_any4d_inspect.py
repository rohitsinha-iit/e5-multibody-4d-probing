import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from demo_inference import init_inference_model
cfg = {"path": "configs/train.yaml",
       "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
       "checkpoint_path": "checkpoints/any4d_4v_combined.pth", "trained_with_amp": True, "data_norm_type": "dinov2"}
m = init_inference_model(cfg, cfg["checkpoint_path"], torch.device("cuda"))
print(type(m).__name__)
for n, c in m.named_children():
    print("child", n, type(c).__name__, sum(p.numel() for p in c.parameters()) / 1e6)
info = [c for n, c in m.named_modules() if type(c).__name__.startswith("MultiViewAlternatingAttentionTransformer")]
t = info[0]
print("info module", type(t).__name__, "depth", t.depth, "dim", t.dim)
b0 = t.self_attention_blocks[0]
print("block", type(b0).__name__, [n for n, _ in b0.named_children()])
a = b0.attn
print("heads", a.num_heads, "head_dim", a.head_dim, "custom_pe", a.custom_positional_encoding, "qnorm", type(a.q_norm).__name__,
      "latent", a.latent_attn, "scalable_softmax", a.use_scalable_softmax, "entropy_scaling", a.use_entropy_scaling, "fused", a.fused_attn)
for k in ["distinguish_ref_and_non_ref_views", "use_pe_for_non_reference_views", "use_rand_idx_pe_for_non_reference_views", "custom_positional_encoding"]:
    print(k, getattr(t, k, None))
names = [n for n, c in m.named_modules() if c is t]
print("info module path", names)
import inspect
src = inspect.getsource(type(a).forward)
print(src[src.find("q, k, v"):][:1800])

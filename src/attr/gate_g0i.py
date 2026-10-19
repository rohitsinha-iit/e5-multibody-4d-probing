#!/usr/bin/env python3
"""Gate G0i (unified plan section 4) and hook smoke/timing for Any4D (any4d_hooks.py).

G0i: patching every component (288 heads + 24 residual states, all positions) from D into D reproduces D exactly, and
B into B likewise: max abs output diff <= 1e-4 in fp32 (pts3d_0, scene_flow, cam_pose).
Also recorded (diagnostics, not gate criteria):
  - repeat determinism of the plain forward (fp32 and AMP);
  - each residual state and 24 sampled heads patched alone D->D (fp32);
  - AMP identity patch (bf16 activations, cache in activation dtype);
  - positive control: residual states of all blocks patched B->D move scene flow toward B;
  - masking: empty key mask = identity; GT key mask on all 12 heads of one global block changes the output;
  - timings and peak memory: base, capture, patch, masking 1 head / 12 heads, two-pass.
Writes $E5_BIG/results/attr_mech/g0i.json and updates $E5_BIG/results/attr_gates.json (key G0i) atomically.
"""
import argparse, json, os, sys, time, fcntl
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import any4d_hooks as ah  # noqa

p = argparse.ArgumentParser()
p.add_argument("--scene", default="/data2/DATA/rohit_sinha/e5/big/attr/factorial/v2/469")
p.add_argument("--out", default="/data2/DATA/rohit_sinha/e5/big/results/attr_mech/g0i.json")
p.add_argument("--gates", default="/data2/DATA/rohit_sinha/e5/big/results/attr_gates.json")
a = p.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
t0 = time.time()
model = ah.load_any4d()
H = ah.Any4DHooks(model)
R = {"scene": a.scene, "host": os.uname().nodename, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
     "load_s": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
views = {v: ah.load_views(os.path.join(a.scene, v)) for v in ("D", "B")}
V = len(views["D"]); ALL = torch.arange(V * ah.P_TOK + 1, device="cuda")
comps = ah.all_components()


def timed(fn):
    torch.cuda.synchronize(); s = time.time(); r = fn(); torch.cuda.synchronize(); return r, time.time() - s


def full_patch(cache):
    return [ah.Patch(c, ALL, cache) for c in comps]


gate = {}
for vn in ("D", "B"):
    base, cache = H.run(views[vn], capture=True, use_amp=False)
    rep, _ = H.run(views[vn], use_amp=False)
    pat, _ = H.run(views[vn], patches=full_patch(cache), use_amp=False)
    d = ah.max_abs_diff(base, pat); r = ah.max_abs_diff(base, rep)
    gate[vn] = {"patched_all_vs_base": d, "repeat_vs_base": r, "max": max(d.values())}
    if vn == "D":
        single = {}
        rng = np.random.default_rng(0)
        heads = [comps[i] for i in rng.choice([i for i, c in enumerate(comps) if c[0] == "h"], 24, replace=False)]
        for cc in [c for c in comps if c[0] == "r"] + heads:
            o, _ = H.run(views[vn], patches=[ah.Patch(cc, ALL, cache)], use_amp=False)
            single[ah.comp_name(cc)] = max(ah.max_abs_diff(base, o).values())
        gate["D"]["single_component_max"] = max(single.values()); gate["D"]["n_single"] = len(single)
        cacheD32, baseD32 = cache, base
    else:
        baseB32 = base
    del cache; torch.cuda.empty_cache()
passed = gate["D"]["max"] <= 1e-4 and gate["B"]["max"] <= 1e-4 and gate["D"]["single_component_max"] <= 1e-4
R["G0i_fp32"] = gate

# positive control: all residual states B -> D (fp32). Encoder tap is still D's, so not expected to equal B.
_, cacheB = H.run(views["B"], capture=True, use_amp=False)
pc, _ = H.run(views["D"], patches=[ah.Patch(c, ALL, cacheB) for c in comps if c[0] == "r"], use_amp=False)
dDB = float((baseD32["scene_flow"] - baseB32["scene_flow"]).norm(dim=-1).mean())
R["positive_control_resid_B_into_D"] = {
    "sf_dist_to_B": float((pc["scene_flow"] - baseB32["scene_flow"]).norm(dim=-1).mean()),
    "sf_dist_to_D": float((pc["scene_flow"] - baseD32["scene_flow"]).norm(dim=-1).mean()), "sf_dist_D_B": dDB}
del cacheB, cacheD32; torch.cuda.empty_cache()

# AMP (the screen setting)
torch.cuda.reset_peak_memory_stats()
(baseA, cacheA), t_cap = timed(lambda: H.run(views["D"], capture=True))
mem_cap = torch.cuda.max_memory_allocated() / 1e9
(o, _), t_base = timed(lambda: H.run(views["D"]))
R["amp_repeat_vs_base"] = ah.max_abs_diff(baseA, o)
(o, _), t_patch_all = timed(lambda: H.run(views["D"], patches=full_patch(cacheA)))
R["amp_patched_all_vs_base"] = ah.max_abs_diff(baseA, o)
(o, _), t_patch1 = timed(lambda: H.run(views["D"], patches=[ah.Patch(("h", 2, 1), ah.view_positions([0], V), cacheA)]))
cache_gb = sum(v.numel() * v.element_size() for v in cacheA.values()) / 1e9
# masking
gtD = np.load(os.path.join(a.scene, "D_gt", "gt_factorial.npz"))
mm = gtD["moving_mask"][ah.FRAMES]
km = ah.token_mask_from_pixels(mm, 0.25).reshape(-1).cuda()
(o, _), t_mask0 = timed(lambda: H.run(views["D"], masks={0: ah.KeyMask([0], torch.zeros_like(km))}))
R["mask_empty_vs_base"] = ah.max_abs_diff(baseA, o)
torch.cuda.reset_peak_memory_stats()
(o1, _), t_mask1 = timed(lambda: H.run(views["D"], masks={10: ah.KeyMask([3], km)}))
(o12, _), t_mask12 = timed(lambda: H.run(views["D"], masks={10: ah.KeyMask(list(range(12)), km)}))
(oall, _), t_maskall = timed(lambda: H.run(views["D"], masks={b: ah.KeyMask(list(range(12)), km) for b in range(24)}))
mem_mask = torch.cuda.max_memory_allocated() / 1e9
R["mask_effects"] = {"b10h3": ah.max_abs_diff(baseA, o1), "b10_all12": ah.max_abs_diff(baseA, o12),
                     "all288": ah.max_abs_diff(baseA, oall),
                     "n_flagged_tokens": int(km.sum()), "frac_flagged": float(km.float().mean())}
R["timing_s"] = {"capture": round(t_cap, 3), "base": round(t_base, 3), "patch_all": round(t_patch_all, 3),
                 "patch_one_head_view0": round(t_patch1, 3), "mask_empty": round(t_mask0, 3), "mask_1head": round(t_mask1, 3),
                 "mask_12heads_1block": round(t_mask12, 3), "mask_all288": round(t_maskall, 3)}
R["memory_gb"] = {"peak_capture_amp": round(mem_cap, 2), "peak_mask_all": round(mem_mask, 2), "cache_gb_amp": round(cache_gb, 3),
                  "cache_dtype_amp": str(next(iter(cacheA.values())).dtype)}
R["G0i"] = {"gate": "G0i", "status": "pass" if passed else "fail", "max_abs_diff_D": gate["D"]["max"],
            "max_abs_diff_B": gate["B"]["max"], "single_component_max": gate["D"]["single_component_max"],
            "criteria": "fp32, all 288 heads + 24 residual states patched at all positions, D->D and B->B; max abs diff of pts3d_0, scene_flow, cam_pose <= 1e-4",
            "time": R["time"], "host": R["host"], "scene": a.scene}
json.dump(R, open(a.out, "w"), indent=1)
# merge into attr_gates.json under a lock
with open(a.gates + ".lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    g = json.load(open(a.gates)) if os.path.exists(a.gates) else {}
    g["G0i"] = R["G0i"]
    json.dump(g, open(a.gates + ".tmp", "w"), indent=1); os.replace(a.gates + ".tmp", a.gates)
print(json.dumps(R["G0i"]), json.dumps(R["timing_s"]), json.dumps(R["memory_gb"]))

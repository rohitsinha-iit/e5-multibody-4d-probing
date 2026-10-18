"""Item 10: fine-tune Any4D (one variant). Place in Any4D/scripts/ next to e5_ft_lib.py; run with cwd = Any4D root.

  python scripts/e5_ft_train.py --variant heads --data DIR1,DIR2 --e5 $SRC --out RUN_DIR [--max_steps N]

Training scenes: every scene dir (with e5_meta.json) under the --data dirs; 10% are held out for early stopping
(every 10th scene in sorted order within each dataset dir), written to RUN_DIR/split.json.
One optimizer step = --accum samples (one scene, 7 views: frame 0 + 6 sorted random frames of 1..23).
Validation every --val_every steps on the held-out scenes with the test-time frames 0,4,..,20,23: training loss
plus the round-1 metrics (slope, epe_static, epe_moving, mask_iou) for frame 0 -> 23.
Early stopping: best validation loss; stop after --patience evaluations without improvement, or after --max_hours.
Resumable: RUN_DIR/last.pt (trainable weights, optimizer, scheduler, step, history) is written atomically every
--ckpt_every steps and at every validation; RUN_DIR/best.pt holds the best trainable weights. At the end the best
weights are merged into a full checkpoint RUN_DIR/any4d_ft.pth in the base format ({"model": state_dict}).
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True); ap.add_argument("--data", required=True); ap.add_argument("--e5", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--base_ckpt", default="checkpoints/any4d_4v_combined.pth")
ap.add_argument("--max_steps", type=int, default=2000); ap.add_argument("--steps_file", default="")
ap.add_argument("--accum", type=int, default=2); ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--val_every", type=int, default=200); ap.add_argument("--ckpt_every", type=int, default=200)
ap.add_argument("--patience", type=int, default=4); ap.add_argument("--max_hours", type=float, default=3.5)
ap.add_argument("--lr_lora", type=float, default=2e-4); ap.add_argument("--lr_sf", type=float, default=2e-5)
ap.add_argument("--lr_gain", type=float, default=5e-3); ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=16); ap.add_argument("--lam_geo", type=float, default=1.0)
ap.add_argument("--dyn_w", type=float, default=10.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit_scenes", type=int, default=0)
ap.add_argument("--n_val", type=int, default=0); ap.add_argument("--export", type=int, default=1)
ap.add_argument("--log_every", type=int, default=20); ap.add_argument("--smoke_max_steps", type=int, default=0)
ap.add_argument("--export_from", default="best", choices=["best", "last"], help="last: smoke tests only")
ap.add_argument("--select", default="loss", choices=["loss", "sf"],
                help="checkpoint selection: total val loss, or the scene-flow val loss with a static-EPE guard. With a "
                     "geometry-distillation term the total loss always keeps step 0, because that term is 0 for the base model")
ap.add_argument("--static_tol", type=float, default=0.25)
a = ap.parse_args()
sys.path.insert(0, a.e5)
import e5_ft_lib as L  # noqa: E402
import e5_kubric_post as kp  # noqa: E402
from demo_inference import init_inference_model  # noqa: E402
from any4d.utils.misc import seed_everything  # noqa: E402

os.makedirs(a.out, exist_ok=True)
seed_everything(a.seed)
if a.steps_file and os.path.exists(a.steps_file):
    a.max_steps = int(json.load(open(a.steps_file))["max_steps"]); print("max_steps from", a.steps_file, a.max_steps)
if a.smoke_max_steps:
    a.max_steps = a.smoke_max_steps

# ------------------------------------------------------------------ scenes and split
train, val = [], []
for D in a.data.split(","):
    sc = sorted(d for d in glob.glob(os.path.join(D, "*")) if os.path.exists(os.path.join(d, "e5_meta.json")))
    if a.limit_scenes:
        sc = sc[: a.limit_scenes]
    for i, d in enumerate(sc):
        (val if i % 10 == 9 else train).append(d)
if a.limit_scenes and not val:
    val = train[-1:]
if a.n_val:
    val = val[: a.n_val]
assert train and val, (len(train), len(val))
json.dump(dict(train=[os.path.basename(x) for x in train], val=[os.path.basename(x) for x in val],
               data=a.data.split(",")), open(os.path.join(a.out, "split.json"), "w"), indent=1)
print(f"train {len(train)} scenes, val {len(val)} scenes", flush=True)

# ------------------------------------------------------------------ model
dev = torch.device("cuda")
cfg = dict(L.CFG, checkpoint_path=a.base_ckpt)
model = init_inference_model(cfg, a.base_ckpt, dev)
groups, info = L.apply_variant(model, a.variant, r=a.rank, alpha=a.alpha, lr_lora=a.lr_lora, lr_sf=a.lr_sf, lr_gain=a.lr_gain)
L.enable_checkpointing(model)
model.eval()  # no dropout / drop-path; adapters are trained in eval mode
need_teacher = a.variant != "sf" and a.lam_geo > 0
info.update(vars(a)); info["need_teacher"] = need_teacher
json.dump(info, open(os.path.join(a.out, "config.json"), "w"), indent=1)
print("variant", a.variant, "heads", info["heads"], "trainable params", info["n_trainable"], flush=True)
opt = torch.optim.AdamW([dict(params=g["params"], lr=g["lr"]) for g in groups], weight_decay=0.0)


def lr_lambda(step):
    if step < a.warmup:
        return (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, a.max_steps - a.warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
state = dict(step=0, history=[], best=float("inf"), best_step=-1, bad=0, elapsed_h=0.0, stopped=None)
LAST, BEST = os.path.join(a.out, "last.pt"), os.path.join(a.out, "best.pt")
if os.path.exists(LAST):
    ck = torch.load(LAST, map_location="cpu", weights_only=False)
    L.load_trainable_state(model, ck["trainable"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    state = ck["state"]; print("resumed at step", state["step"], flush=True)


def save_last():
    torch.save(dict(trainable=L.trainable_state(model), opt=opt.state_dict(), sched=sched.state_dict(), state=state),
               LAST + ".tmp")
    os.replace(LAST + ".tmp", LAST)


def compute_loss(s, train_mode=True):
    views = L.to_device(s["views"], dev)
    teach = None
    if need_teacher:
        L.set_adapters(model, False)
        with torch.no_grad():
            teach = L.forward(model, views)
        L.set_adapters(model, True)
    with torch.set_grad_enabled(train_mode):
        preds = L.forward(model, views)
        lsf, parts = L.sf_loss(preds, s, a.dyn_w)
        loss = lsf
        parts["sf"] = float(lsf)
        if teach is not None:
            lg, gp = L.geo_loss(preds, teach)
            loss = loss + a.lam_geo * lg; parts.update(gp); parts["geo"] = float(lg)
    parts["loss"] = float(loss)
    return loss, parts, preds


val_ds = L.KubricFT(val, fixed_frames=L.TEST_FRAMES)


def validate():
    rows = []
    for k in range(len(val_ds)):
        s = val_ds[k]
        _, parts, preds = compute_loss(s, train_mode=False)
        P = L.preds_to_P(preds)
        meta, gt, _ = kp.load_scene(val[k])
        m = L.kubric_metrics(meta, gt, P, k, L.TEST_FRAMES)
        rows.append(dict(parts, **{f"val_{x}": m[x] for x in ("slope", "epe_static", "epe_moving", "mask_iou")}))
    out = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], float)
        out[key] = float(np.nanmean(vals)) if key.startswith("val_slope") or key in ("loss", "sf", "geo", "sf_dyn", "sf_static") \
            or key.startswith("geo") else float(np.nanmedian(vals))
    return out


def score(v):
    # checkpoint-selection score, lower is better (see --select)
    if a.select == "loss":
        return v["loss"]
    s0 = state["history"][0].get("val_epe_static", float("nan")) if state["history"] else float("nan")
    ok = not (v.get("val_epe_static", 0.0) > s0 + a.static_tol)
    return v["sf"] if ok else float("inf")


# ------------------------------------------------------------------ loop
t0 = time.time(); t_prev = state["elapsed_h"]
if state["step"] == 0 and not state["history"]:
    v = validate(); v.update(step=0, hours=0.0); state["history"].append(v); state["best"] = score(v); state["best_step"] = 0
    torch.save(L.trainable_state(model), BEST); save_last()
    print("val step 0", json.dumps({k: round(x, 4) for k, x in v.items()}), flush=True)
ds = L.KubricFT(train, seed=a.seed)
start_k = state["step"] * a.accum
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(start_k, a.max_steps * a.accum)), batch_size=None,
                                     shuffle=False, num_workers=a.workers, collate_fn=None, prefetch_factor=2,
                                     persistent_workers=False) if state["step"] < a.max_steps and state["stopped"] is None else []
it = iter(loader); run = {}
torch.cuda.reset_peak_memory_stats()
while state["step"] < a.max_steps and state["stopped"] is None:
    opt.zero_grad(set_to_none=True)
    batch = [next(it) for _ in range(a.accum)]
    for attempt in range(12):  # the GPU is shared with other jobs: on OOM, free memory, wait and redo this step
        try:
            acc = {}
            for s in batch:
                loss, parts, _ = compute_loss(s)
                (loss / a.accum).backward()
                for key, x in parts.items():
                    acc[key] = acc.get(key, 0.0) + x / (a.accum * a.log_every)
            break
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); loss = None; torch.cuda.empty_cache()
            print(f"OOM at step {state['step'] + 1}, retry {attempt + 1} in 60 s", flush=True); time.sleep(60)
    else:
        save_last(); raise SystemExit("repeated CUDA OOM (GPU shared); resumable from last.pt")
    for key, x in acc.items():
        run[key] = run.get(key, 0.0) + x
    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
    opt.step(); sched.step(); state["step"] += 1
    hours = t_prev + (time.time() - t0) / 3600; state["elapsed_h"] = hours
    if state["step"] % a.log_every == 0:
        print(f"step {state['step']} {hours:.2f} h lr {sched.get_last_lr()[0]:.2e} " +
              " ".join(f"{k} {x:.4f}" for k, x in run.items()) +
              f" peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        run = {}
    if state["step"] % a.val_every == 0 or state["step"] == a.max_steps:
        v = validate(); v.update(step=state["step"], hours=hours); state["history"].append(v)
        if score(v) < state["best"]:
            state["best"], state["best_step"], state["bad"] = score(v), state["step"], 0
            torch.save(L.trainable_state(model), BEST + ".tmp"); os.replace(BEST + ".tmp", BEST)
        else:
            state["bad"] += 1
        print("val", json.dumps({k: round(x, 4) for k, x in v.items()}), "best step", state["best_step"], flush=True)
        if state["bad"] >= a.patience:
            state["stopped"] = f"early stop at step {state['step']} (best {state['best_step']})"
        save_last()
    elif state["step"] % a.ckpt_every == 0:
        save_last()
    if os.environ.get("FT_DIE_AT_STEP") and state["step"] == int(os.environ["FT_DIE_AT_STEP"]):
        save_last(); print("simulated crash at step", state["step"], flush=True); os._exit(3)  # resume test only
    if hours > a.max_hours and state["stopped"] is None:
        state["stopped"] = f"time limit {a.max_hours} h at step {state['step']}"; save_last()
if state["stopped"] is None:
    state["stopped"] = f"max_steps {a.max_steps}"; save_last()
print("stopped:", state["stopped"], flush=True)
json.dump(dict(history=state["history"], best_step=state["best_step"], best=state["best"], stopped=state["stopped"],
               config=info), open(os.path.join(a.out, "train_history.json"), "w"), indent=1, default=str)

# ------------------------------------------------------------------ export merged checkpoint
FINAL = os.path.join(a.out, "any4d_ft.pth")
if a.export and not os.path.exists(FINAL):
    if a.export_from == "best":
        L.load_trainable_state(model, torch.load(BEST, map_location="cpu", weights_only=False))
    s = val_ds[0]; views = L.to_device(s["views"], dev)
    with torch.no_grad():  # fp32 so that the check measures the merge, not bf16 rounding
        ref = L.forward(model, views, amp=False)
    L.merge_adapters(model)
    with torch.no_grad():
        mrg = L.forward(model, views, amp=False)
    d_sf = max(float((r["scene_flow"] - m["scene_flow"]).abs().max()) for r, m in zip(ref, mrg))
    d_pt = max(float((r["pts3d"] - m["pts3d"]).abs().max()) for r, m in zip(ref, mrg))
    print(f"merge check: max |d scene_flow| {d_sf:.2e}, max |d pts3d| {d_pt:.2e}", flush=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(model=sd, e5_ft=dict(variant=a.variant, best_step=state["best_step"], heads=info["heads"], export_from=a.export_from,
                                          merge_check=dict(d_sf=d_sf, d_pts=d_pt))), FINAL + ".tmp")
    os.replace(FINAL + ".tmp", FINAL)
    json.dump(dict(d_sf=d_sf, d_pts=d_pt), open(os.path.join(a.out, "merge_check.json"), "w"))
print("done", FINAL, flush=True)

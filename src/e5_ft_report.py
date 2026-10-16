"""Item 10: per-task reports (<task>_summary.md + PNG in RESULT_DIR). numpy + matplotlib only (runs in .venv-post).

  e5_ft_report.py render DATASET_DIR RESULT_DIR TASK_ID EXPECTED
  e5_ft_report.py pilot  RUN_DIR RESULT_DIR TASK_ID STEPS_JSON
  e5_ft_report.py train  RUN_DIR RESULT_DIR TASK_ID
  e5_ft_report.py eval   RESULT_DIR TASK_ID VARIANT
"""
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def w(path, text):
    open(path + ".tmp", "w").write(text); os.replace(path + ".tmp", path)


def render(D, R, tid, expected):
    S = json.load(open(os.path.join(D, "scene_summary.json"))) if os.path.exists(os.path.join(D, "scene_summary.json")) else []
    ok = [r for r in S if "error" not in r]
    by = {}
    for r in ok:
        by.setdefault(r["split"], []).append(r)
    seeds = sorted(int(r["scene"].rsplit("_s", 1)[1]) for r in S)
    fails = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(D, "logs", "*.txt"))
                   if not os.path.exists(os.path.join(D, "out", os.path.basename(p)[:-4], "e5_meta.json")))
    L = [f"# {tid}", "", f"Training scenes for item 10 (fine-tuning), rendered with the approved generator settings into `{D}`.", "",
         f"- Rendered: {len(ok)} of {expected} planned scenes; render failures: {len(fails)} ({', '.join(fails[:10])}).",
         f"- Seeds {seeds[0] if seeds else '-'} to {seeds[-1] if seeds else '-'} (all at or above 2001, disjoint from every evaluation set, whose seeds are at most 720).",
         f"- GT flow self-check (geometric GT vs Kubric optical flow, frame 0 to 1): median EPE "
         f"{np.median([r['flow_epe_px'] for r in ok]) if ok else float('nan'):.3f} px, max {max([r['flow_epe_px'] for r in ok]) if ok else float('nan'):.3f} px.", "",
         "| split | scenes | median sep (px) | median SNR at 1 px |", "|---|---|---|---|"]
    for sp in sorted(by):
        L.append(f"| {sp} | {len(by[sp])} | {np.nanmedian([r['sep_px'] for r in by[sp]]):.1f} | {np.nanmedian([r['snr_1px'] for r in by[sp]]):.1f} |")
    w(os.path.join(R, f"{tid}_summary.md"), "\n".join(L) + "\n")
    sheets = sorted(glob.glob(os.path.join(D, "sheets", "*.png")))
    pick = []
    for sp in sorted(by):
        c = [s for s in sheets if os.path.basename(s).startswith(sp + "_s")]
        pick += c[:1]
    if pick:
        from PIL import Image
        ims = [Image.open(p).convert("RGB") for p in pick[:8]]
        wd = max(i.width for i in ims); ht = sum(i.height for i in ims)
        canvas = Image.new("RGB", (wd, ht), "white"); y = 0
        for i in ims:
            canvas.paste(i, (0, y)); y += i.height
        canvas = canvas.resize((wd // 2, ht // 2))
        canvas.save(os.path.join(R, f"{tid}_contact_sheets.png"))
    json.dump(dict(n_ok=len(ok), expected=int(expected), failures=fails, per_split={k: len(v) for k, v in by.items()}),
              open(os.path.join(R, f"{tid}_counts.json"), "w"), indent=1)


def curves(H, title, out):
    st = [h["step"] for h in H]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3))
    ax[0].plot(st, [h["loss"] for h in H], "o-", ms=3, label="total"); ax[0].plot(st, [h["sf"] for h in H], "s--", ms=3, label="scene flow")
    if "geo" in H[0]:
        ax[0].plot(st, [h["geo"] for h in H], "^:", ms=3, label="distillation")
    ax[0].set_title("validation loss"); ax[0].legend(frameon=False)
    ax[1].plot(st, [h["val_slope"] for h in H], "o-", ms=3); ax[1].axhline(1, color="0.6", lw=0.8); ax[1].set_title("slope (mean)")
    ax[2].plot(st, [h["val_epe_moving"] for h in H], "o-", ms=3, label="moving"); ax[2].plot(st, [h["val_epe_static"] for h in H], "s-", ms=3, label="static")
    ax[2].set_title("EPE, median px"); ax[2].legend(frameon=False)
    ax[3].plot(st, [h["val_mask_iou"] for h in H], "o-", ms=3); ax[3].set_title("motion-mask IoU (median)")
    for a_ in ax:
        a_.set_xlabel("optimizer step")
    fig.suptitle(title, x=0.01, ha="left"); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def pilot(run, R, tid, steps_json):
    T = json.load(open(os.path.join(run, "train_history.json"))); H = T["history"]
    loss = np.array([h["loss"] for h in H]); st = np.array([h["step"] for h in H])
    gain = loss[0] - loss.min()
    s90 = int(st[np.argmax(loss <= loss[0] - 0.9 * gain)]) if gain > 0 else int(st[-1])
    sec = 3600 * H[-1]["hours"] / max(1, st[-1])
    cap = int(3.0 * 3600 / sec) // 100 * 100
    chosen = int(min(max(int(math.ceil(2 * s90 / 100.0)) * 100, 400), 2500, max(400, cap)))
    json.dump(dict(max_steps=chosen, s90=s90, best_step=T["best_step"], sec_per_step=sec, time_cap_steps=cap,
                   rule="2 x (first validation step reaching 90% of the pilot's loss reduction), rounded up to 100, "
                        "clipped to [400, 2500] and to 3 h at the pilot's speed"), open(steps_json, "w"), indent=1)
    json.dump(json.load(open(steps_json)), open(os.path.join(R, "chosen_steps.json"), "w"), indent=1)
    curves(H, f"{tid}: pilot (variant heads_sf), validation on held-out training scenes", os.path.join(R, f"{tid}_curves.png"))
    L = [f"# {tid}", "", "Pilot run to choose the number of fine-tuning steps (variant heads_sf = five-head LoRA + scene-flow head, the largest variant).", "",
         f"- Validation every {st[1] - st[0] if len(st) > 1 else '-'} steps on {len(json.load(open(os.path.join(run, 'split.json')))['val'])} held-out training scenes (10%, never used for gradients).",
         f"- Validation loss {loss[0]:.4f} at step 0, minimum {loss.min():.4f} at step {int(st[loss.argmin()])}; 90% of the reduction reached at step {s90}.",
         f"- Speed: {sec:.2f} s per optimizer step ({T['config']['accum']} samples of 7 views).",
         f"- Chosen max_steps for every variant: **{chosen}** (rule: 2 x step-at-90%, rounded up to 100, within [400, 2500] and 3 h). Each run still early-stops on its own validation loss.", "",
         "| step | loss | slope | EPE moving | EPE static | mask IoU |", "|---|---|---|---|---|---|"]
    for h in H:
        L.append(f"| {h['step']} | {h['loss']:.4f} | {h['val_slope']:.3f} | {h['val_epe_moving']:.1f} | {h['val_epe_static']:.2f} | {h['val_mask_iou']:.3f} |")
    w(os.path.join(R, f"{tid}_summary.md"), "\n".join(L) + "\n")


def train(run, R, tid):
    T = json.load(open(os.path.join(run, "train_history.json"))); H = T["history"]; C = T["config"]
    curves(H, f"{tid}: variant {C['variant']}, validation on held-out training scenes", os.path.join(R, f"{tid}_curves.png"))
    mc = json.load(open(os.path.join(run, "merge_check.json"))) if os.path.exists(os.path.join(run, "merge_check.json")) else {}
    json.dump(T, open(os.path.join(R, "train_history.json"), "w"), indent=1, default=str)
    b = next(h for h in H if h["step"] == T["best_step"]); h0 = H[0]
    L = [f"# {tid}", "", f"Fine-tuning variant **{C['variant']}** of Any4D from any4d_4v_combined.pth.", "",
         f"- Trainable parameters: {C['n_trainable']:,}; heads: {C['heads'] or 'none (scene-flow head only)'}; LoRA rank {C['rank']}, alpha {C['alpha']}.",
         f"- Learning rates: LoRA {C['lr_lora']}, scene-flow head {C['lr_sf']}, gain {C['lr_gain']}; AdamW, warmup {C['warmup']}, cosine to 10%; bf16, activation checkpointing.",
         f"- Loss: Any4D scene-flow loss (scale-invariant, log-norm, x{C['dyn_w']} on moving pixels)" +
         (f" + {C['lam_geo']} x distillation to the frozen model (pointmaps, depth, pose, metric scale)." if C.get("need_teacher") else " (no distillation needed: transformer and geometry heads are frozen)."),
         f"- Stopped: {T['stopped']}. Best validation step: {T['best_step']}. Merge check (adapter vs merged weights): {mc}.", "",
         "| validation (held-out training scenes, frames 0 to 23) | step 0 (base) | best step |", "|---|---|---|"]
    for k, lab in [("loss", "loss"), ("val_slope", "slope (mean)"), ("val_epe_moving", "EPE moving (median px)"),
                   ("val_epe_static", "EPE static (median px)"), ("val_mask_iou", "motion-mask IoU (median)")]:
        L.append(f"| {lab} | {h0[k]:.4f} | {b[k]:.4f} |")
    L += ["", "These are training-distribution numbers; the held-out evaluation is in the matching ft_eval task."]
    w(os.path.join(R, f"{tid}_summary.md"), "\n".join(L) + "\n")


def eval_(R, tid, variant):
    lines = [f"# {tid}", "", f"Evaluation of **{variant}** with the round-1 metric code (output metrics: e5_causal / e5_step3 definitions).", "",
             "| dataset | n | slope (mean) | EPE static (median of scenes) | EPE moving | sep ratio (median) | mask IoU | mask IoU, epi_aligned |",
             "|---|---|---|---|---|---|---|---|"]
    vals = {}
    for ds in ("v2", "v3float", "v3far"):
        p = os.path.join(R, f"kubric_{ds}.json")
        if not os.path.exists(p):
            continue
        rows = [r for r in json.load(open(p)) if "error" not in r]
        g = lambda k, rr=rows: np.array([r.get(k, np.nan) for r in rr], float)  # noqa: E731
        epi = [r for r in rows if r["split"] == "epi_aligned"]
        vals[ds] = (np.nanmean(g("slope")), np.nanmedian(g("epe_moving")))
        lines.append(f"| {ds} | {len(rows)} | {np.nanmean(g('slope')):.3f} | {np.nanmedian(g('epe_static')):.2f} | {np.nanmedian(g('epe_moving')):.1f} | "
                     f"{np.nanmedian(g('sep_ratio')):.3f} | {np.nanmedian(g('mask_iou')):.3f} | {np.nanmedian(g('mask_iou', epi)) if epi else float('nan'):.3f} |")
    import csv
    for p in sorted(glob.glob(os.path.join(R, "*_wide.csv"))):
        rows = list(csv.DictReader(open(p)))
        f = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan for r in rows])  # noqa: E731
        nm = os.path.basename(p)[:-4]
        vals[nm] = (np.nanmean(f("slope")), np.nanmedian(f("epe_moving")))
        lines.append(f"| {nm} | {len(rows)} | {np.nanmean(f('slope')):.3f} | {np.nanmedian(f('epe_static')):.2f} | {np.nanmedian(f('epe_moving')):.1f} | "
                     f"{np.nanmedian(f('sep_ratio')):.3f} | - | - |")
    pp = os.path.join(R, "probeP_v3float_results.json")
    if os.path.exists(pp):
        P = json.load(open(pp))["P"]["rows"]
        lines += ["", "Internal motion decodability (view-0 linear probe, 5-fold by scene, floating set):", "",
                  "| layer | motion AUC, all splits | motion AUC, epi_aligned |", "|---|---|---|"]
        for l in sorted({r["layer"] for r in P}):
            a_ = [r.get("motion_auc_v0_objects", np.nan) for r in P if r["layer"] == l]
            e_ = [r.get("motion_auc_v0_objects", np.nan) for r in P if r["layer"] == l and r["split"] == "epi_aligned"]
            lines.append(f"| {l} | {np.nanmean(np.array(a_, float)):.3f} | {np.nanmean(np.array(e_, float)) if e_ else float('nan'):.3f} |")
    w(os.path.join(R, f"{tid}_summary.md"), "\n".join(lines) + "\n")
    if vals:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ks = list(vals)
        ax[0].bar(range(len(ks)), [vals[k][0] for k in ks], color="#4C72B0"); ax[0].axhline(1, color="0.5", lw=0.8)
        ax[1].bar(range(len(ks)), [vals[k][1] for k in ks], color="#DD8452")
        for a_, t in zip(ax, ["slope (1 = full object motion)", "EPE moving, median px"]):
            a_.set_xticks(range(len(ks))); a_.set_xticklabels(ks, rotation=30, ha="right"); a_.set_title(t)
        fig.suptitle(f"{tid}: {variant}", x=0.01, ha="left"); fig.tight_layout()
        fig.savefig(os.path.join(R, f"{tid}_metrics.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "render":
        render(*sys.argv[2:5], int(sys.argv[5]))
    elif m == "pilot":
        pilot(*sys.argv[2:6])
    elif m == "train":
        train(*sys.argv[2:5])
    elif m == "eval":
        eval_(*sys.argv[2:5])

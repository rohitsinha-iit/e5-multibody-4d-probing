"""Round 2, review W4: sanity summary of the control-model internals written by e5_probe_extract.py.
Usage: python e5_probe_extract_summary.py <probe_ctrl/feats> <out_dir> <task_id>"""
import glob, json, os, sys
import numpy as np

root, outd, tid = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outd, exist_ok=True)
L = [f"# {tid}: control-model internals for the probe controls (review W4)\n",
     "Variants: randtf = public DINOv2 ViT-L/14 encoder + Any4D projection, fusion norm, transformer and heads at their own seeded random init; "
     "randall = everything random (encoder re-initialised with the dinov2 constructor, pretrained=False); "
     "enc_any4d / enc_dino = raw encoder patch tokens of the trained Any4D encoder / the public DINOv2 encoder (1024-d, before fusion norm, "
     "projection and view PE); trained = round-1 Any4D internals (symlinks). Layers 0,4,...,24 for the transformer variants (0 = transformer input).\n",
     "| dataset | source | scenes | stored layers | non-finite values | encoder max abs param diff to public DINOv2 |", "|---|---|---|---|---|---|"]
curves = {}
for ds in sorted(os.listdir(root)):
    for src in ["trained", "randtf", "randall", "enc_any4d", "enc_dino"]:
        d = os.path.join(root, ds, src)
        if not os.path.isdir(d):
            continue
        toks = sorted(glob.glob(os.path.join(d, "*_tok.npy")))
        st = json.load(open(os.path.join(d, "stored_layers.json"))) if os.path.exists(os.path.join(d, "stored_layers.json")) else []
        info = json.load(open(os.path.join(d, "variant_info.json"))) if os.path.exists(os.path.join(d, "variant_info.json")) else {}
        lay = [0, 4, 8, 12, 16, 20, 24] if src in ("trained", "randtf", "randall") else [0]
        nf = 0; norms = []
        for f in toks[:20]:
            a = np.load(f, mmap_mode="r")
            arr = np.asarray(a[[st.index(l) for l in lay]]).astype(np.float32)
            nf += int((~np.isfinite(arr)).sum()); norms.append(np.linalg.norm(arr[:, 0], axis=-1).mean(-1))
        if norms:
            curves[(ds, src)] = (lay, np.mean(norms, 0))
        dif = info.get("encoder_max_abs_diff_to_public_dinov2", "")
        L.append(f"| {ds} | {src} | {len(toks)} | {','.join(map(str, st))} | {nf} (first 20 scenes) | {dif if dif == '' else f'{dif:.4g}'} |")
L.append("\nThe trained Any4D encoder differs from the public DINOv2 weights (it was fine-tuned), so enc_any4d and enc_dino are two different baselines.\n")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=False)
for ax, ds in zip(axs, sorted({k[0] for k in curves})):
    for src, col in [("trained", "#1f4e79"), ("randtf", "#e67e22"), ("randall", "#c0392b")]:
        if (ds, src) in curves:
            x, y = curves[(ds, src)]; ax.plot(x, y, "-o", ms=3, color=col, label=src)
    for src, mk in [("enc_any4d", "s"), ("enc_dino", "D")]:
        if (ds, src) in curves:
            ax.plot([-1], curves[(ds, src)][1], mk, color="0.3", mfc="white", label=src + " (1024-d)")
    ax.set_title(ds, fontsize=9); ax.set_xlabel("layer"); ax.set_ylabel("mean view-0 token L2 norm"); ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(outd, "fig_token_norms.png"), dpi=150)
open(os.path.join(outd, f"{tid}_summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

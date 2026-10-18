"""E5 v4 (review item W3): ground-truth checks of the new epipolar scene types.

Usage: python e5_v4_epicheck.py <out_dir_with_scenes> <result_prefix> [title]
Writes <prefix>.json (per-scene rows), <prefix>.png (figure) and <prefix>_summary.md.

Per scene (all from Kubric GT: depth, segmentation, per-frame camera and object matrices):
  sep_gt_0_23      two-view multibody separation of GT correspondences, frames 0 to 23 (same code as step 3:
                   300 points per motion group, median cross-motion Sampson residual, px). About 0 means the pair
                   is two-view ambiguous.
  sep_gt_0_12      the same for frames 0 to 12.
  resid3v_mov[i]   3-view static-consistency residual on moving pixels: triangulate each frame-0 pixel as a static
                   point from its GT matches in frames 0 and 23, project into frame i, median distance to its GT
                   match in frame i (px). i = 4, 8, 12, 16, 20. Large = a third view exposes the motion.
  resid3v_bg[i]    the same on static background pixels (should be about 0).
  pair_*           epi_accel vs epi_const with the same seed: pixel differences of frames 0, 12 and 23.
  sizes            moving size, k (object speed / camera speed) and implied static size = size / (1 + k).
"""
import glob, json, os, sys
import numpy as np
import e5_kubric_post as kp
import e5_step3_analysis as s3

MIDV = (4, 8, 12, 16, 20)


def rays(meta, x, fi):
    W, H = meta["resolution"]; K = np.array(meta["K"])
    C = np.array(meta["camera_matrix_world"][str(meta["frames"][fi])])
    d = np.stack([(x[:, 0] + 0.5 - W / 2) / K[0, 0], -(x[:, 1] + 0.5 - H / 2) / K[1, 1], -np.ones(len(x))], -1)
    d = d @ C[:3, :3].T
    return np.repeat(C[None, :3, 3], len(x), 0), d


def triangulate(meta, xa, fa, xb, fb):
    Ca, da = rays(meta, xa, fa); Cb, db = rays(meta, xb, fb)
    w0 = Ca - Cb
    a = (da * da).sum(1); b = (da * db).sum(1); c = (db * db).sum(1); d = (da * w0).sum(1); e = (db * w0).sum(1)
    den = a * c - b * b
    den = np.where(np.abs(den) < 1e-12, np.nan, den)
    s = (b * e - c * d) / den; u = (a * e - b * d) / den
    return 0.5 * ((Ca + s[:, None] * da) + (Cb + u[:, None] * db))


def centre_track(meta, k):
    out = []
    for fi in range(len(meta["frames"])):
        M = np.array(meta["objects"][k]["matrix_world"][str(meta["frames"][fi])])
        p, _ = kp.project(meta, M[None, :3, 3], fi)
        out.append(p[0])
    return np.array(out)


def scene_rows(d, rng):
    meta, gt, rgb = kp.load_scene(d)
    nF = len(meta["frames"]); last = nF - 1
    row = dict(scene=os.path.basename(d), split=meta["split"], seed=int(meta["seed"]))
    c23 = kp.correspondences(meta, gt, 0, last)
    for tag, t in (("0_23", last), ("0_12", int(round(12 * last / 23)))):
        c = c23 if t == last else kp.correspondences(meta, gt, 0, t)
        idx = s3.sample_by_group(c["group"], 300, np.random.default_rng(0))
        gi = c["group"][idx]
        row[f"sep_gt_{tag}"] = float(s3.sep_and_noise(c["x1"][idx], c["x2"][idx], gi)[0]) if len(set(gi)) >= 2 else float("nan")
    H, W = c23["dense_valid"].shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x0 = np.stack([u, v], -1).astype(float)
    row["resid3v_mov"], row["resid3v_bg"] = {}, {}
    for j in MIDV:
        i = int(round(j * last / 23))
        if i in (0, last):
            continue
        ct = kp.correspondences(meta, gt, 0, i)
        ok = c23["dense_valid"] & ct["dense_valid"]
        for name, m in (("mov", ok & (c23["dense_group"] > 0)), ("bg", ok & (c23["dense_group"] == 0))):
            yy, xx = np.nonzero(m)
            if len(yy) < 20:
                row[f"resid3v_{name}"][str(i)] = float("nan"); continue
            sel = rng.choice(len(yy), min(3000, len(yy)), replace=False); yy, xx = yy[sel], xx[sel]
            Q = triangulate(meta, x0[yy, xx], 0, c23["dense_x2"][yy, xx], last)
            q, _ = kp.project(meta, Q, i)
            row[f"resid3v_{name}"][str(i)] = float(np.nanmedian(np.linalg.norm(q - ct["dense_x2"][yy, xx], axis=1)))
    mov = [k for k, o in enumerate(meta["objects"]) if o["group"] > 0]
    if mov:
        tr = centre_track(meta, mov[0])
        Qc = triangulate(meta, tr[:1], 0, tr[-1:], last)  # static reading of the centre from frames 0 and 23
        lin = np.array([kp.project(meta, Qc, fi)[0][0] for fi in range(nF)])  # where that static point appears
        row["centre_track_x"] = tr[:, 0].tolist(); row["centre_static_hyp_x"] = lin[:, 0].tolist()
        row["centre_gap_max_px"] = float(np.max(np.linalg.norm(tr - lin, axis=1)))
    mv = meta["motions"].get("1", {})
    for key in ("k", "mov_size", "implied_static_size", "static_scale_ratio", "analytic_sep2_px", "accel_center_gap_mid_px",
                "analytic_resid3v_px", "analytic_sep_pairs_px", "profile"):
        if key in mv and mv[key] is not None:
            row[key] = mv[key]
    row["static_sizes"] = [o["size"] for o in meta["objects"] if o["group"] == 0]
    row["n_static"] = len(row["static_sizes"])
    row["_rgb"] = rgb
    return row


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (float(np.median(xs)), float(np.min(xs)), float(np.max(xs)), len(xs)) if xs else (float("nan"),) * 3 + (0,)


SPLITS = ["epi_accel", "epi_const", "epi_sizerand", "epi_sizematch", "epi_aligned"]
COL = {"epi_accel": "#d95f02", "epi_const": "#1b9e77", "epi_sizerand": "#7570b3", "epi_sizematch": "#e7298a", "epi_aligned": "#666666"}


def figure(rows, png, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    for r in rows:
        if r["split"] in ("epi_accel", "epi_const"):
            ks = sorted(r["resid3v_mov"], key=int)
            ax[0].plot([int(k) for k in ks], [r["resid3v_mov"][k] for k in ks], "o-", color=COL[r["split"]], alpha=0.6, lw=1)
            ax[0].plot([int(k) for k in ks], [r["resid3v_bg"][k] for k in ks], ".:", color="0.5", alpha=0.6, lw=0.8)
    for s in ("epi_accel", "epi_const"):
        ax[0].plot([], [], "o-", color=COL[s], label=f"{s}, moving pixels")
    ax[0].plot([], [], ".:", color="0.5", label="static background")
    ax[0].set_xlabel("third view (frame index)"); ax[0].set_ylabel("3-view static-consistency residual (px)")
    ax[0].set_title("(a) static fit from frames 0 and 23, error in frame i", fontsize=9); ax[0].legend(fontsize=7)
    sp = [s for s in SPLITS if any(r["split"] == s for r in rows)]
    for j, s in enumerate(sp):
        y = [r.get("sep_gt_0_23", np.nan) for r in rows if r["split"] == s]
        x = j + np.linspace(-0.12, 0.12, len(y)) if len(y) > 1 else [j]
        ax[1].plot(x, y, "o", color=COL[s])
    ax[1].set_xticks(range(len(sp))); ax[1].set_xticklabels([s.replace("epi_", "") for s in sp], fontsize=8)
    ax[1].set_ylabel("GT multibody separation, frames 0 to 23 (px)"); ax[1].set_ylim(bottom=0, top=max(1.0, ax[1].get_ylim()[1]))
    ax[1].set_title("(b) two-view separation (about 0 = ambiguous)", fontsize=9)
    ex = [r for r in rows if r["split"] == "epi_accel" and "centre_track_x" in r]
    if ex:
        r = ex[0]; o = next((q for q in rows if q["scene"] == f"epi_const_s{r['seed']}"), None)
        t = np.arange(len(r["centre_track_x"]))
        ax[2].plot(t, r["centre_track_x"], "o-", ms=3, color=COL["epi_accel"], label=f"accelerated ({r.get('profile', {}).get('mode', '')})")
        if o is not None:
            ax[2].plot(t, o["centre_track_x"], "s-", ms=3, color=COL["epi_const"], label="constant velocity (paired)")
        ax[2].plot(t, r["centre_static_hyp_x"], "k--", lw=1, label="static reading from frames 0 and 23")
        for f in (0, 4, 8, 12, 16, 20, 23):
            if f < len(t):
                ax[2].axvline(f, color="0.85", lw=0.6, zorder=0)
        ax[2].set_xlabel("frame"); ax[2].set_ylabel("object centre, image x (px)"); ax[2].legend(fontsize=7)
        ax[2].set_title(f"(c) {r['scene']}: image track (grey = 7-view frames)", fontsize=9)
    for j, s in enumerate(sp):
        rs = [r for r in rows if r["split"] == s and "mov_size" in r]
        if not rs:
            continue
        xs = j + np.linspace(-0.12, 0.12, len(rs)) if len(rs) > 1 else [j]
        ax[3].plot(xs, [r["mov_size"] for r in rs], "o", color=COL[s], mfc="none")
        ax[3].plot(xs, [r["implied_static_size"] for r in rs], "o", color=COL[s])
    ax[3].axhspan(1.4, 2.2, color="0.9", zorder=0, label="static-object sizes (1.4 to 2.2 m)")
    ax[3].plot([], [], "ko", mfc="none", label="true size of moving object"); ax[3].plot([], [], "ko", label="size if read as static")
    ax[3].set_xticks(range(len(sp))); ax[3].set_xticklabels([s.replace("epi_", "") for s in sp], fontsize=8)
    ax[3].set_ylabel("size (m)"); ax[3].legend(fontsize=7); ax[3].set_title("(d) moving-object size cue", fontsize=9)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)


def summary(rows, allrows, md, title, png):
    L = [f"# {title}", "", f"Scenes checked: {len(rows)}; scenes with errors: {len(allrows) - len(rows)}.", "",
         "All numbers come from Kubric ground truth (depth, segmentation, camera and object matrices).", "",
         "| split | n | sep GT 0 to 23 (px), median [min, max] | 3-view residual, moving, frame 12 (px), median [min, max] | 3-view residual, background, frame 12 (px) | max centre gap from static reading (px) | static objects per scene, mean |",
         "|---|---|---|---|---|---|---|"]
    for s in SPLITS:
        rs = [r for r in rows if r["split"] == s]
        if not rs:
            continue
        a = med([r.get("sep_gt_0_23") for r in rs]); b = med([r["resid3v_mov"].get("12", np.nan) for r in rs])
        c = med([r["resid3v_bg"].get("12", np.nan) for r in rs]); e = med([r.get("centre_gap_max_px", np.nan) for r in rs])
        L.append(f"| {s} | {len(rs)} | {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] | {b[0]:.1f} [{b[1]:.1f}, {b[2]:.1f}] | {c[0]:.2f} | {e[0]:.1f} | {np.mean([r['n_static'] for r in rs]):.2f} |")
    pr = [r for r in rows if "pair_maxdiff_f0" in r]
    L += ["", "## Pairing check (epi_accel vs epi_const, same seed)", ""]
    if pr:
        L.append(f"- Pairs: {len(pr)}. Largest pixel difference (0 to 255) in frame 0: {max(r['pair_maxdiff_f0'] for r in pr)}; "
                 f"in frame 23: {max(r['pair_maxdiff_flast'] for r in pr)}. Mean absolute difference in frame 12: "
                 f"{np.mean([r['pair_meandiff_fmid'] for r in pr]):.2f} (only the moving object differs there).")
        L.append("- A difference of 0 in frames 0 and 23 means the two-view input (frames 0 and 23) is identical within a pair.")
    else:
        L.append("- No complete pairs found.")
    L += ["", "## Size cue", "", "| split | n | true moving size (m), median [min, max] | size if read as static (m), median [min, max] | max abs error of static scale vs 1/(1+k) |", "|---|---|---|---|---|"]
    for s in SPLITS:
        rs = [r for r in rows if r["split"] == s and "mov_size" in r]
        if not rs:
            continue
        a = med([r["mov_size"] for r in rs]); b = med([r["implied_static_size"] for r in rs])
        err = max(abs(r["static_scale_ratio"] - 1 / (1 + r["k"])) for r in rs)
        L.append(f"| {s} | {len(rs)} | {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] | {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}] | {err:.1e} |")
    L += ["", "Static objects in every split are 1.4 to 2.2 m (the v3float range).", "", f"![GT checks]({png})", "",
          "How to read the figure: (a) for epi_accel the static fit from frames 0 and 23 misses the object in a third view by tens of pixels, "
          "while for epi_const and for the background it fits to about 0 px. (b) Two-view separation is about 0 in every split, so frames 0 and 23 alone are ambiguous. "
          "(c) The accelerated object's image track leaves the track of its static reading (the constant-velocity object stays on it). (d) Size randomisation decouples apparent size from the static interpretation."]
    open(md, "w").write("\n".join(L) + "\n")


def main():
    root, prefix = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "v4 epipolar set: GT checks"
    rng = np.random.default_rng(0)
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.exists(os.path.join(d, "e5_meta.json")):
            continue
        try:
            rows.append(scene_rows(d, rng))
        except Exception as e:
            rows.append(dict(scene=os.path.basename(d), error=repr(e)))
        print(rows[-1]["scene"], rows[-1].get("error", ""), flush=True)
    by = {r["scene"]: r for r in rows if "error" not in r}
    for r in by.values():
        if r["split"] == "epi_accel":
            o = by.get(f"epi_const_s{r['seed']}")
            if o is not None and o["_rgb"].shape == r["_rgb"].shape:
                A, B = r["_rgb"].astype(int), o["_rgb"].astype(int); last = len(A) - 1; mid = int(round(12 * last / 23))
                r["pair_maxdiff_f0"] = int(np.abs(A[0] - B[0]).max()); r["pair_maxdiff_flast"] = int(np.abs(A[last] - B[last]).max())
                r["pair_meandiff_fmid"] = float(np.abs(A[mid] - B[mid]).mean())
    for r in rows:
        r.pop("_rgb", None)
    json.dump(rows, open(prefix + ".json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    good = [r for r in rows if "error" not in r]
    figure(good, prefix + ".png", title)
    summary(good, rows, prefix + "_summary.md", title, os.path.basename(prefix + ".png"))


if __name__ == "__main__":
    main()

"""Post-processing of E5 Kubric scenes: GT correspondences, flow self-check, multibody SNR, contact sheets."""
import json, glob, os
import numpy as np
from PIL import Image
import e5_multibody as mb


def load_scene(d):
    meta = json.load(open(os.path.join(d, "e5_meta.json")))
    gt = np.load(os.path.join(d, "gt.npz"))
    frames = meta["frames"]
    rgb = np.stack([np.array(Image.open(os.path.join(d, f"rgba_{i:05d}.png")))[..., :3] for i in range(len(frames))])
    return meta, gt, rgb


def seg_to_group(meta):
    """Map segmentation value -> (motion group, object index). Background (0) -> group 0, obj -1."""
    m = {0: (0, -1)}
    for sid, k in meta["seg_index_map"].items():
        if k == "dome":
            m[int(sid)] = (0, -1)
        else:
            m[int(sid)] = (meta["objects"][int(k)]["group"], int(k))
    return m


def backproject(meta, depth, fi):
    W, H = meta["resolution"]; K = np.array(meta["K"])
    C = np.array(meta["camera_matrix_world"][str(meta["frames"][fi])])
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - W / 2) / K[0, 0], -(v - H / 2) / K[1, 1], -np.ones_like(u)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    Pc = d * depth[..., None]
    return Pc @ C[:3, :3].T + C[:3, 3]


def project(meta, X, fi):
    W, H = meta["resolution"]; K = np.array(meta["K"])
    C = np.array(meta["camera_matrix_world"][str(meta["frames"][fi])])
    Pc = (X - C[:3, 3]) @ C[:3, :3]
    z = -Pc[..., 2]
    u = K[0, 0] * Pc[..., 0] / z + W / 2 - 0.5
    v = -K[1, 1] * Pc[..., 1] / z + H / 2 - 0.5
    return np.stack([u, v], -1), np.linalg.norm(Pc, axis=-1)


def correspondences(meta, gt, a, b, occl_tol=0.02):
    """Dense GT correspondences frame a -> frame b. Returns dict of flat arrays (pixel centres)."""
    W, H = meta["resolution"]
    depth = gt["depth"][..., 0]; seg = gt["segmentation"][..., 0]
    X = backproject(meta, depth[a], a)
    s2g = seg_to_group(meta)
    grp = np.zeros((H, W), int); Xb = X.copy()
    fa, fb = str(meta["frames"][a]), str(meta["frames"][b])
    for sid, (g, k) in s2g.items():
        msk = seg[a] == sid
        grp[msk] = g
        if k >= 0 and msk.any():
            Ma = np.array(meta["objects"][k]["matrix_world"][fa]); Mb = np.array(meta["objects"][k]["matrix_world"][fb])
            T = Mb @ np.linalg.inv(Ma)
            Xb[msk] = X[msk] @ T[:3, :3].T + T[:3, 3]
    pb, rb = project(meta, Xb, b)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    inside = (pb[..., 0] >= 0) & (pb[..., 0] < W - 1) & (pb[..., 1] >= 0) & (pb[..., 1] < H - 1)
    ui = np.clip(np.round(pb[..., 0]).astype(int), 0, W - 1); vi = np.clip(np.round(pb[..., 1]).astype(int), 0, H - 1)
    vis = inside & (np.abs(depth[b][vi, ui] - rb) < occl_tol * rb)
    known = np.isin(seg[a], list(s2g.keys()))
    keep = vis & known
    return dict(x1=np.stack([u[keep], v[keep]], 1).astype(float), x2=pb[keep], group=grp[keep],
                seg=seg[a][keep], dense_x2=pb, dense_valid=keep, dense_group=grp)


def flow_check(meta, gt):
    """Median endpoint error of geometric GT (frame0->1) vs Kubric forward_flow, both channel orders."""
    c = correspondences(meta, gt, 0, 1)
    fl = gt["forward_flow"][0]
    xi = c["x1"].astype(int)
    f = fl[xi[:, 1], xi[:, 0]]
    d = c["x2"] - c["x1"]
    e_xy = np.median(np.linalg.norm(f - d, axis=1)); e_yx = np.median(np.linalg.norm(f[:, ::-1] - d, axis=1))
    return float(min(e_xy, e_yx)), ("xy" if e_xy <= e_yx else "yx"), float(np.median(np.linalg.norm(d, axis=1)))


def multibody_snr(meta, gt, a=0, b=-1, n_per_group=400, rng=None, noise_px=1.0, patch_px=14 * 512 / 518):
    rng = rng or np.random.default_rng(0)
    b = b % len(meta["frames"])
    c = correspondences(meta, gt, a, b)
    groups = sorted(set(c["group"]))
    pts = {}
    for g in groups:
        idx = np.where(c["group"] == g)[0]
        if len(idx) >= 16:
            pts[g] = rng.choice(idx, min(n_per_group, len(idx)), replace=False)
    Fs = {}
    for g, idx in pts.items():
        a_, Ta = mb.hartley_normalize(c["x1"][idx]); b_, Tb = mb.hartley_normalize(c["x2"][idx])
        Fs[g] = Tb.T @ mb.eight_point(a_, b_) @ Ta
    P1 = {g: np.c_[c["x1"][i], np.ones(len(i))] for g, i in pts.items()}
    P2 = {g: np.c_[c["x2"][i], np.ones(len(i))] for g, i in pts.items()}
    self_res = {g: float(np.median(np.sqrt(mb.sampson(Fs[g], P1[g], P2[g])))) for g in pts}
    seps = {}
    for g in pts:
        for h in pts:
            if g != h:
                seps[f"{g}|{h}"] = float(np.median(np.sqrt(mb.sampson(Fs[h], P1[g], P2[g]))))
    sep = min(seps.values()) if seps else float("nan")
    disp = {g: float(np.median(np.linalg.norm(c["x2"][i] - c["x1"][i], axis=1))) for g, i in pts.items()}
    return dict(n_groups_visible=len(pts), n_pts={g: int(len(i)) for g, i in pts.items()},
                sep_px=sep, pair_sep_px=seps, self_residual_px=self_res, median_disp_px=disp,
                snr_1px=sep / (noise_px * np.sqrt(2)), snr_patch14=sep / (patch_px / np.sqrt(6)))


GROUP_COLORS = np.array([[110, 110, 110], [31, 95, 168], [230, 140, 20], [40, 160, 90]], np.uint8)


def contact_sheet(meta, gt, rgb, out_png, title=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nF = len(meta["frames"]); idx = [0, nF // 3, 2 * nF // 3, nF - 1]
    c = correspondences(meta, gt, 0, nF - 1)
    seg = gt["segmentation"][..., 0]; s2g = seg_to_group(meta)
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.4))
    for j, fi in enumerate(idx):
        axs[0, j].imshow(rgb[fi]); axs[0, j].set_title(f"frame {fi}", fontsize=9)
    for j, fi in enumerate(idx[:2]):
        g = np.zeros(seg[fi].shape, int)
        for sid, (gg, _) in s2g.items():
            g[seg[fi] == sid] = gg
        ov = (0.45 * rgb[fi] + 0.55 * GROUP_COLORS[np.clip(g, 0, 3)]).astype(np.uint8)
        axs[1, j].imshow(ov); axs[1, j].set_title(f"motion groups, frame {fi}", fontsize=9)
    ax = axs[1, 2]; ax.imshow(rgb[0])
    rng = np.random.default_rng(0)
    sel = rng.choice(len(c["x1"]), min(500, len(c["x1"])), replace=False)
    for g in sorted(set(c["group"][sel])):
        s = sel[c["group"][sel] == g]
        col = GROUP_COLORS[min(g, 3)] / 255
        ax.plot(np.c_[c["x1"][s, 0], c["x2"][s, 0]].T, np.c_[c["x1"][s, 1], c["x2"][s, 1]].T, "-", color=col, lw=0.6)
    ax.set_title(f"GT correspondences 0 to {nF-1}", fontsize=9)
    ax.set_xlim(0, meta["resolution"][0]); ax.set_ylim(meta["resolution"][1], 0)
    ax = axs[1, 3]; ax.imshow(gt["depth"][0, ..., 0], cmap="magma_r"); ax.set_title("depth, frame 0", fontsize=9)
    for a in axs.flat:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

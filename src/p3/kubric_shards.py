"""CoTracker3_Kubric (facebook/CoTracker3_Kubric, MOVi-F) loader for P3 X2 (unified plan U3).

Shard layout (verified on shards 0000 to 0009, 2026-10-19):
  <root>/<id>/<id>.npy            pickled dict: coords (N,120,2) px, coords_depth (N,120) RAY RANGE (not z),
                                  visibility (N,120) bool which is TRUE WHEN OCCLUDED (checked: the complement
                                  matches the depth maps to 2 % for 95 % of in-image points), depth (120,512,512,1)
                                  uint16 range + metadata.depth_range, segmentations (120,512,512,1) uint8,
                                  camera {positions, quaternions, focal_length, sensor_width, field_of_view}
  <root>/<id>/<id>_with_rank.npz  extrinsics (120,3,4) OpenCV world-to-camera, shared_intrinsics (3,3)
  <root>/<id>/frames/NNN.png      RGB 512x512;  <root>/<id>/depths/NNN.npy  float64 z-depth
3D tracks: X_cam(t) = range * normalize(K^-1 [u,v,1]); X_world = E_t^-1 X_cam(t).
Windows: 25 frames at stride 1 (starts 0,25,50,75) and stride 2 (starts 0,49). Split by sequence: id % 10 == 9 -> val.
Each clip is written to <clips>/<id>_s<stride>_<start>/ as rgb.npy (25,512,512,3) uint8 and labels.npz.
"""
import argparse, glob, json, os, sys
import numpy as np

E5_BIG = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")
SHARDS = os.path.join(E5_BIG, "data", "cotracker3_kubric")
CLIPS = os.path.join(E5_BIG, "p3", "clips")
L = 25
EDGE_PX = 8


def list_sequences(root=SHARDS):
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(root, "*.done"))
                  if os.path.isdir(os.path.join(root, os.path.basename(p)[:-5])))


def split_of(seq_id):
    return "val" if int(seq_id) % 10 == 9 else "train"


def window_starts(T=120, stride=1):
    span = (L - 1) * stride + 1
    return list(range(0, T - span + 1, span))


def load_sequence(seq_id, root=SHARDS):
    d = os.path.join(root, seq_id)
    s = np.load(os.path.join(d, seq_id + ".npy"), allow_pickle=True).item()
    z = np.load(os.path.join(d, seq_id + "_with_rank.npz"))
    s["E"] = z["extrinsics"].astype(np.float64); s["K"] = z["shared_intrinsics"].astype(np.float64)
    s["frame_paths"] = [os.path.join(d, "frames", n) for n in z["names"]]
    s["dir"] = d
    return s


def rays(uv, K):
    x = np.stack([(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1], np.ones(uv.shape[:-1])], -1)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cam_points(coords, rng, K):
    """(N,T,2) px + (N,T) ray range -> (N,T,3) OpenCV camera coordinates of the camera at each frame."""
    return rays(coords.astype(np.float64), K) * rng[..., None].astype(np.float64)


def to_world(Xc, E):
    """Xc (N,T,3) in camera t; E (T,3,4) world-to-camera -> world points."""
    R, t = E[:, :, :3], E[:, :, 3]
    return np.einsum("tji,ntj->nti", R, Xc - t[None])


def to_cam(Xw, E):
    R, t = E[:, :, :3], E[:, :, 3]
    return np.einsum("tij,ntj->nti", R, Xw) + t[None]


def project(X, K):
    z = np.maximum(X[..., 2], 1e-6)
    return np.stack([X[..., 0] / z * K[0, 0] + K[0, 2], X[..., 1] / z * K[1, 1] + K[1, 2]], -1)


def edge_maps(seg):
    """seg (T,H,W) -> bool (T,H,W): within EDGE_PX of a segmentation boundary."""
    from scipy import ndimage
    out = np.zeros(seg.shape, bool)
    for i, s in enumerate(seg):
        b = np.zeros(s.shape, bool)
        b[:, 1:] |= s[:, 1:] != s[:, :-1]; b[1:, :] |= s[1:, :] != s[:-1, :]
        out[i] = ndimage.distance_transform_edt(~b) <= EDGE_PX
    return out


def build_clip(s, start, stride):
    from PIL import Image
    fr = np.arange(start, start + (L - 1) * stride + 1, stride)
    co = s["coords"][:, fr].astype(np.float32); rg = s["coords_depth"][:, fr].astype(np.float32)
    occ = s["visibility"][:, fr].astype(bool)
    E = s["E"][fr]; K = s["K"]
    seg = s["segmentations"][fr, ..., 0]
    H, W = seg.shape[1:]
    inb = (co[..., 0] >= 0) & (co[..., 0] <= W - 1) & (co[..., 1] >= 0) & (co[..., 1] <= H - 1)
    vis = (~occ) & inb
    em = edge_maps(seg)
    ui = np.clip(np.round(co[..., 0]).astype(int), 0, W - 1); vi = np.clip(np.round(co[..., 1]).astype(int), 0, H - 1)
    tt = np.broadcast_to(np.arange(len(fr))[None], ui.shape)
    near_edge = em[tt, vi, ui] & vis
    segid = seg[tt, vi, ui].astype(np.uint8)
    Xc = cam_points(co, rg, K)
    Xw = to_world(Xc, E).astype(np.float32)
    rgb = np.stack([np.asarray(Image.open(s["frame_paths"][f]).convert("RGB")) for f in fr])
    lab = dict(coords=co, Xw=Xw, vis=vis, near_edge=near_edge, segid=segid, E=E.astype(np.float64), K=K,
               frames=fr, stride=np.int32(stride), start=np.int32(start))
    return rgb, lab


def reproj_check(lab, a=0, b=L - 1):
    """G0e: lift static (segmentation 0) tracks visible at a and b, map through world, reproject into b."""
    m = lab["vis"][:, a] & lab["vis"][:, b] & (lab["segid"][:, a] == 0)
    Xb = to_cam(lab["Xw"][m][:, [a]].astype(np.float64), lab["E"][[b]])[:, 0]
    err = np.linalg.norm(project(Xb, lab["K"]) - lab["coords"][m, b], axis=1)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=SHARDS); ap.add_argument("--clips", default=CLIPS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default=os.path.join(E5_BIG, "results", "p3_g0e.json"))
    a = ap.parse_args()
    seqs = list_sequences(a.shards)[: a.limit or None]
    os.makedirs(a.clips, exist_ok=True)
    errs = []; n_new = 0
    for sid in seqs:
        s = None
        for stride in (1, 2):
            for st in window_starts(120, stride):
                name = f"{sid}_s{stride}_{st:03d}"; od = os.path.join(a.clips, name)
                if os.path.exists(os.path.join(od, "labels.npz")):
                    continue
                if s is None:
                    s = load_sequence(sid, a.shards)
                rgb, lab = build_clip(s, st, stride)
                e = reproj_check(lab); errs.append(dict(clip=name, n=int(len(e)), median=float(np.median(e)) if len(e) else None,
                                                        p90=float(np.percentile(e, 90)) if len(e) else None))
                os.makedirs(od + ".tmp", exist_ok=True)
                np.save(os.path.join(od + ".tmp", "rgb.npy"), rgb)
                np.savez(os.path.join(od + ".tmp", "labels.npz"), **lab)
                json.dump(dict(seq=sid, split=split_of(sid), stride=stride, start=st), open(os.path.join(od + ".tmp", "meta.json"), "w"))
                os.replace(od + ".tmp", od); n_new += 1
        print(sid, "done", flush=True)
    # G0e summary over all clips (recompute for existing clips without a record)
    rows = {r["clip"]: r for r in (json.load(open(a.report))["clips"] if os.path.exists(a.report) else [])}
    for r in errs:
        rows[r["clip"]] = r
    med = [r["median"] for r in rows.values() if r["median"] is not None]
    rep = dict(gate="G0e", rgb_present=True, criterion="median static reprojection error <= 0.5 px",
               n_clips=len(rows), worst_clip_median_px=float(max(med)) if med else None,
               overall_median_px=float(np.median(med)) if med else None,
               passed=bool(med) and max(med) <= 0.5, clips=sorted(rows.values(), key=lambda r: r["clip"]))
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=1)
    print("G0e", rep["passed"], rep["worst_clip_median_px"], "new clips", n_new, flush=True)


if __name__ == "__main__":
    main()

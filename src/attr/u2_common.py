"""U2 behaviour track (unified plan U2): shared paths, splits, scene lists, query-point sampling and IO.

Splits and exclusions are read from the FROZEN pre-registration ($E5_BIG/results/attr_preregistration.json):
  fit      = v2 even seeds, test = v2 odd seeds, transfer = v3float; excluded scenes as listed there.
Factorial layout: $E5_BIG/attr/factorial/<set>/<seed>/{A,B,C,D(symlink to the dataset render),D_gt}.
GT per variant: gt_factorial.npz (src/attr/gt_factorial.py) in <V>/ (A, B, C) or D_gt/ (D).
Outputs: $E5_BIG/results/u2/ (owned by the U2 track).
"""
import glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.dirname(HERE)
for p in (HERE, SRC):            # appended, never prepended: src/attr would shadow the attrs package
    if p not in sys.path:
        sys.path.append(p)

E5_BIG = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")
U2 = os.environ.get("U2_RES", os.path.join(E5_BIG, "results", "u2"))
PREREG = os.path.join(E5_BIG, "results", "attr_preregistration.json")
FACT = os.path.join(E5_BIG, "attr", "factorial")
REAL_GT = os.path.join(E5_BIG, "realdata", "gt")
POSESETS = os.path.join(E5_BIG, "realdata", "posesets")
VIEWS7 = [0, 4, 8, 12, 16, 20, 23]
TC3R_FRAMES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]
SPLIT_SET = {"fit": "v2", "test": "v2", "transfer": "v3float"}
BOUNDARY_PX = 8.0
SINTEL14 = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5", "market_6",
            "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
BONN5 = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]   # MonST3R / 4RC relpose metadata.py


def prereg():
    return json.load(open(PREREG))


def excluded():
    ex = prereg()["excluded_scenes"]
    return {k: set(v) for k, v in ex.items() if isinstance(v, list)}


def factorial_scenes(split):
    """[(seed, name)] of usable base scenes of a split, in seed order. A scene is usable when it is not excluded by
    the pre-registration and all four variants and their GT exist."""
    ds = SPLIT_SET[split]; ex = excluded().get(ds, set()); out = []
    for d in sorted(glob.glob(os.path.join(FACT, ds, "*")), key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 10 ** 9):
        b = os.path.basename(d)
        if not b.isdigit():
            continue
        seed = int(b)
        if ds == "v2" and ((split == "fit") != (seed % 2 == 0)):
            continue
        name = os.path.basename(os.path.realpath(os.path.join(d, "D")))
        if name in ex:
            continue
        ok = all(os.path.exists(os.path.join(d, v, "e5_meta.json")) for v in "ABCD") and \
            all(os.path.exists(gt_path(ds, seed, v)) for v in "ABCD")
        if ok:
            out.append((seed, name))
    return out


def vdir(ds, seed, v):
    return os.path.join(FACT, ds, str(seed), v)


def gt_path(ds, seed, v):
    return os.path.join(FACT, ds, str(seed), "D_gt" if v == "D" else v, "gt_factorial.npz")


def load_gtf(ds, seed, v):
    z = np.load(gt_path(ds, seed, v))
    return {k: z[k] for k in z.files}


def frame_paths(ds, seed, v, frames):
    return [os.path.join(vdir(ds, seed, v), f"rgba_{f:05d}.png") for f in frames]


def save_npz(path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"; np.savez_compressed(tmp, **kw); os.replace(tmp, path)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1, default=float); os.replace(tmp, path)


def sample_nearest(arr, g, x, y):
    """arr (h,w,...) on the grid described by geom g (e5_models layout); values at native pixels (x, y) by the nearest
    grid pixel (the e5_xm_eval to518 rule for square 518 inputs). Returns values and an inside mask."""
    import e5_models as em
    um, vm = em.native_to_model(g, np.asarray(x, float), np.asarray(y, float))
    ui = np.clip(np.round(um).astype(int), 0, arr.shape[1] - 1); vi = np.clip(np.round(vm).astype(int), 0, arr.shape[0] - 1)
    inside = (um > -0.5) & (um < arr.shape[1] - 0.5) & (vm > -0.5) & (vm < arr.shape[0] - 0.5)
    return arr[vi, ui], inside


def to_native(g, xm):
    import e5_models as em
    return np.stack(em.model_to_native(g, xm[..., 0], xm[..., 1]), -1)


def K_native(K, g):
    """Intrinsics at model resolution (pixel-index convention) -> native pixels, for the affine geom g."""
    ox = 0.5 * g["sx"] - 0.5 - g["left"]; oy = 0.5 * g["sy"] - 0.5 - g["top"]
    A = np.array([[g["sx"], 0, ox], [0, g["sy"], oy], [0, 0, 1.0]])
    return np.linalg.inv(A) @ np.asarray(K, float)


def gpu_stats():
    import torch
    return dict(peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30, peak_reserved_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


# ------------------------------------------------------------------ pose-set windows (written once, shared with U5b)
def _tum_assoc(seq_dir, rgb_sub, gt_file):
    imgs = sorted(glob.glob(os.path.join(seq_dir, rgb_sub, "*.png")))
    L = [l.split() for l in open(os.path.join(seq_dir, gt_file)) if l.strip() and not l.startswith("#")]
    ts = np.array([float(l[0]) for l in L])
    if len(L) == len(imgs):
        return imgs, list(range(len(L)))
    its = np.array([float(os.path.basename(f)[:-4]) for f in imgs])
    return imgs, [int(k) for k in np.abs(its[:, None] - ts[None]).argmin(1)]


def pose_sequences_spec():
    """Evaluation frames of each pose-set sequence (pre-registration: Sintel 14-sequence MonST3R subset, final pass;
    TUM-dynamics first 90 frames at stride 3 (the rgb_90 folders, 0.1 s spacing); Bonn MonST3R protocol (5 sequences,
    rgb_110))."""
    seqs = []
    root = os.path.join(POSESETS, "sintel", "training")
    for s in SINTEL14:
        imgs = sorted(glob.glob(os.path.join(root, "final", s, "*.png")))
        cams = sorted(glob.glob(os.path.join(root, "camdata_left", s, "*.cam")))
        assert imgs and len(imgs) == len(cams), s
        seqs.append(dict(set="sintel", seq=s, frames=imgs, gt=dict(kind="sintel_cam", files=cams)))
    for d in sorted(glob.glob(os.path.join(POSESETS, "tum", "rgbd_dataset_freiburg3_*"))):
        if os.path.isdir(os.path.join(d, "rgb_90")):
            imgs, idx = _tum_assoc(d, "rgb_90", "groundtruth_90.txt")
            seqs.append(dict(set="tum", seq=os.path.basename(d), frames=imgs,
                             gt=dict(kind="tum", file=os.path.join(d, "groundtruth_90.txt"), rows=idx)))
    for s in BONN5:
        d = os.path.join(POSESETS, "bonn", "rgbd_bonn_dataset", f"rgbd_bonn_{s}")
        imgs, idx = _tum_assoc(d, "rgb_110", "groundtruth_110.txt")
        seqs.append(dict(set="bonn", seq=f"rgbd_bonn_{s}", frames=imgs,
                         gt=dict(kind="tum", file=os.path.join(d, "groundtruth_110.txt"), rows=idx)))
    return seqs


def make_pose_windows(path=None, length=8):
    """Non-overlapping windows of `length` consecutive evaluation frames (pose.windows); the remainder at the end of a
    sequence is dropped. Written once; an existing file is never overwritten."""
    import pose as pm
    path = path or os.path.join(U2, "pose_windows.json")
    if os.path.exists(path):
        return json.load(open(path))
    wins = []
    for s in pose_sequences_spec():
        for k, w in enumerate(pm.windows(len(s["frames"]), length, 1)):
            g = dict(s["gt"]); 
            if g["kind"] == "sintel_cam":
                g = dict(kind="sintel_cam", files=[s["gt"]["files"][i] for i in w])
            else:
                g = dict(kind="tum", file=s["gt"]["file"], rows=[s["gt"]["rows"][i] for i in w])
            wins.append(dict(id=f"{s['set']}/{s['seq']}/w{k:02d}", set=s["set"], seq=s["seq"], frame_idx=w,
                             frames=[s["frames"][i] for i in w], gt=g))
    obj = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), length=length, rule="non-overlapping windows of 8 consecutive "
               "evaluation frames, remainder dropped; Sim(3) alignment per window", n_windows=len(wins),
               per_set={k: sum(w["set"] == k for w in wins) for k in ("sintel", "tum", "bonn")}, windows=wins)
    save_json(path, obj)
    return obj


def _quat_pose(t, q):
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def _sintel_cam(f):
    with open(f, "rb") as fh:
        fh.read(4); fh.read(72); N = np.frombuffer(fh.read(96), "<f8").reshape(3, 4)
    w2c = np.eye(4); w2c[:3] = N
    return np.linalg.inv(w2c)


def window_gt_c2w(w):
    g = w["gt"]
    if g["kind"] == "sintel_cam":
        return np.stack([_sintel_cam(f) for f in g["files"]])
    L = [l.split() for l in open(g["file"]) if l.strip() and not l.startswith("#")]
    return np.stack([_quat_pose(np.array(L[i][1:4], float), np.array(L[i][4:8], float)) for i in g["rows"]])

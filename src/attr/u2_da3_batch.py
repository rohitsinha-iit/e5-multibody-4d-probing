#!/usr/bin/env python3
"""DA3 inputs for TrackCraft3R condition (ii), batched: the model is loaded once and run per frame list exactly as
TrackCraft3R's scripts/preprocess_da3.py (DA3NESTED-GIANT-LARGE, process_res 504, upper_bound_resize; depth resized to
the native resolution with scipy zoom order 1; W2C padded to 4x4; intrinsics rescaled to the native resolution).
Kubric: frames 0,2,...,20,23 of variants C and D -> results/u2/da3/<set>/<seed>/<V>/{depth,extrinsics,intrinsics}.npy
KITTI:  offsets 0..10 plus 10 repeated (12 frames) -> results/u2/da3/real/kitti15_rgb_w12/<sample>/. Resumable."""
import argparse, glob, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.append(HERE); sys.path.append(os.path.dirname(HERE))
import u2_common as u  # noqa: E402


def jobs(kub, real, scenes=None, limit=0):
    J = []
    for it in [x for x in kub.split(",") if x]:
        sp, v = it.split(":"); ds = u.SPLIT_SET[sp]; sc = u.factorial_scenes(sp)
        if scenes:
            sc = [s for s in sc if str(s[0]) in scenes]
        for seed, _ in sc[: limit or None]:
            J.append((u.frame_paths(ds, seed, v, u.TC3R_FRAMES), os.path.join(u.U2, "da3", ds, str(seed), v)))
    for it in [x for x in real.split(",") if x]:
        dsn, pas, cfg = it.split(":"); offs = list(range(11)) + [10]
        files = sorted(f for f in glob.glob(os.path.join(u.REAL_GT, dsn, "*.npz")) if not f.endswith(".tmp.npz"))
        if limit:
            files = files[:: max(1, len(files) // limit)][:limit]
        for f in files:
            meta = json.loads(str(np.load(f)["meta"])); fr = meta["frames"].get(pas, {})
            if all(str(o) in fr for o in offs):
                J.append(([fr[str(o)] for o in offs], os.path.join(u.U2, "da3", "real", f"{dsn}_{pas}_{cfg}", os.path.basename(f)[:-4])))
    return J


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kubric", default=""); ap.add_argument("--real", default="")
    ap.add_argument("--scenes", default=""); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    J = [j for j in jobs(a.kubric, a.real, set(a.scenes.split(",")) if a.scenes else None, a.limit)
         if not os.path.exists(os.path.join(j[1], "intrinsics.npy"))]
    print("DA3 jobs to run", len(J), flush=True)
    if not J:
        return
    import torch
    from PIL import Image
    from scipy.ndimage import zoom
    sys.path.insert(0, os.path.join(u.E5_BIG, "models/src/depth-anything-3", "src"))
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to("cuda").eval()
    t0 = time.time()
    for k, (paths, od) in enumerate(J):
        ims = [Image.open(p).convert("RGB") for p in paths]; W, H = ims[0].size; T = len(ims)
        with torch.no_grad():
            pred = model.inference(image=ims, process_res=504, process_res_method="upper_bound_resize", export_dir=None)
        dp = pred.depth; Hp, Wp = dp.shape[1:]
        depth = (np.stack([zoom(dp[t], (H / Hp, W / Wp), order=1) for t in range(T)]) if (Hp, Wp) != (H, W) else dp).astype(np.float32)
        E = np.zeros((T, 4, 4), np.float32); E[:, :3, :] = pred.extrinsics; E[:, 3, 3] = 1.0
        K = pred.intrinsics.copy().astype(np.float32); sx, sy = W / Wp, H / Hp
        K[:, 0, 0] *= sx; K[:, 1, 1] *= sy; K[:, 0, 2] *= sx; K[:, 1, 2] *= sy
        os.makedirs(od, exist_ok=True)
        np.save(os.path.join(od, "depth.npy"), depth); np.save(os.path.join(od, "extrinsics.npy"), E)
        np.save(os.path.join(od, "intrinsics.tmp.npy"), K); os.replace(os.path.join(od, "intrinsics.tmp.npy"), os.path.join(od, "intrinsics.npy"))
        json.dump(dict(frames=paths), open(os.path.join(od, "frames.json"), "w"))
        if k % 25 == 0:
            print(f"[{time.time() - t0:.0f}s] {k + 1}/{len(J)} {od}", flush=True)
    print("DA3 done", len(J), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

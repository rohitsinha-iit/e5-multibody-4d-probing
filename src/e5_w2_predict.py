"""Round 2, review item 1 (W2): non-reconstruction baselines (CoTracker3, RAFT) on E5 Kubric scenes.

Every model writes <out>/<scene>.npz in the e5_xm_eval.py track format, so it is scored on the SAME query set as
Any4D and VGGT (query_set(meta, gt, si), si = index in the sorted scene list of the dataset):
  q_idx (N,), q_xy (N,2)  query set (518 grid, as in e5_xm_eval.py)
  x2_q   (N,2)  predicted location of each query pixel in frame 23, on the 518 grid (pixel-index convention)
  x2_q0  (N,2)  predicted location in frame 0 (echo of the query; should equal the query)
  x2_q512 (N,2) the same prediction on the native 512 grid
  vis_q, conf_q (N,)  model visibility / confidence where the model has one (CoTracker3: raw visibility and
                      confidence probabilities; RAFT direct: conf_q = -forward-backward error in px)
  frames      frames used
Queries are the exact GT query pixels x1[q] (native 512 grid, pixel-index convention: pixel i has centre i).

Models (--model):
  cotracker3   CoTracker3 offline (public facebook/cotracker3 scaled_offline.pth), queries at frame 0,
               --frames all (0..23) or 0,4,8,12,16,20,23. Same call as CoTrackerPredictor._compute_sparse_tracks
               (support grid of 6x6 points, 6 iterations), but keeping the continuous visibility and confidence.
  raft         torchvision raft_large, weights Raft_Large_Weights.C_T_V2 (FlyingChairs + FlyingThings only, so it has
               never seen KITTI or Sintel), 24 flow updates. --raft_mode direct: flow frame 0 -> 23;
               --raft_mode chain: flows between consecutive entries of --frames, composed by bilinear sampling at
               the propagated point positions.
Resumable: scenes whose npz exists are skipped; files are written atomically.
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--e5", required=True)
ap.add_argument("--model", required=True, choices=["cotracker3", "raft"])
ap.add_argument("--frames", default="0,4,8,12,16,20,23", help="comma list or 'all'")
ap.add_argument("--raft_mode", default="direct", choices=["direct", "chain"])
ap.add_argument("--raft_iters", type=int, default=24)
ap.add_argument("--cotracker_src", default=""); ap.add_argument("--cotracker_ckpt", default="")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--scenes", default="", help="comma list of scene names (smoke tests)")
a = ap.parse_args()
sys.path.insert(0, a.e5)
import e5_xm_eval as xe  # noqa: E402
import e5_kubric_post as kp  # noqa: E402

torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cuda")
os.makedirs(a.out, exist_ok=True)


def save_npz(path, **kw):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **kw); os.replace(tmp, path)


def to518c(x512, W=512):
    """Continuous 512 -> 518 grid mapping (pixel-index convention), inverse of e5_xm_eval's (x + 0.5) / s - 0.5."""
    return (x512 + 0.5) * (xe.M518 / W) - 0.5


# ------------------------------------------------------------------ models
if a.model == "cotracker3":
    sys.path.insert(0, a.cotracker_src)
    from cotracker.models.build_cotracker import build_cotracker  # noqa: E402
    from cotracker.models.core.model_utils import get_points_on_a_grid  # noqa: E402
    CT = build_cotracker(a.cotracker_ckpt, offline=True, window_len=60).to(dev).eval()
    IS = CT.model_resolution

    @torch.no_grad()
    def track(video, qxy):
        """video (T,3,H,W) float 0..255, qxy (N,2) native px at frame 0 -> tracks (T,N,2) native px, vis, conf (T,N)."""
        T, _, H, W = video.shape
        v = F.interpolate(video, tuple(IS), mode="bilinear", align_corners=True)[None]
        q = torch.zeros(1, len(qxy), 3, device=dev); q[0, :, 1:] = torch.as_tensor(qxy, device=dev, dtype=torch.float32)
        q[:, :, 1:] *= q.new_tensor([(IS[1] - 1) / (W - 1), (IS[0] - 1) / (H - 1)])
        g = get_points_on_a_grid(6, IS, device=dev); g = torch.cat([torch.zeros_like(g[:, :, :1]), g], 2)
        q = torch.cat([q, g], 1)
        out = CT(video=v, queries=q, iters=6)
        tr, vis = out[0], out[1]; cf = out[2] if len(out) > 2 and torch.is_tensor(out[2]) else vis
        n = len(qxy); tr = tr[0, :, :n] * tr.new_tensor([(W - 1) / (IS[1] - 1), (H - 1) / (IS[0] - 1)])
        return tr.cpu().numpy(), vis[0, :, :n].float().cpu().numpy(), cf[0, :, :n].float().cpu().numpy()
else:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights  # noqa: E402
    RW = Raft_Large_Weights.C_T_V2
    RAFT = raft_large(weights=RW, progress=False).to(dev).eval()

    @torch.no_grad()
    def flow(i0, i1):
        """i0, i1 (3,H,W) float 0..255 -> flow (2,H,W) in px (dx, dy), padded to a multiple of 8 internally."""
        H, W = i0.shape[-2:]; ph, pw = (-H) % 8, (-W) % 8
        x = torch.stack([i0, i1]) / 127.5 - 1.0                      # RAFT expects [-1, 1]
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        fl = RAFT(x[:1], x[1:], num_flow_updates=a.raft_iters)[-1][0]
        return fl[:, :H, :W]

    def sample(fl, pts):
        """bilinear sample of flow (2,H,W) at native px pts (N,2) (pixel-index convention, border clamp)."""
        H, W = fl.shape[-2:]
        g = torch.as_tensor(pts, device=dev, dtype=torch.float32).clone()
        g[:, 0] = 2 * g[:, 0] / (W - 1) - 1; g[:, 1] = 2 * g[:, 1] / (H - 1) - 1
        s = F.grid_sample(fl[None], g[None, None], mode="bilinear", padding_mode="border", align_corners=True)
        return s[0, :, 0].T.cpu().numpy().astype(np.float64)


def load_video(d, frames):
    ims = [np.asarray(Image.open(os.path.join(d, f"rgba_{f:05d}.png")).convert("RGB")) for f in frames]
    return torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().to(dev)


if __name__ == "__main__":
    scenes = xe.scene_dirs(a.data, a.limit)
    want = set(a.scenes.split(",")) if a.scenes else None
    t_all = time.time(); n_done = 0
    for si, d in enumerate(scenes):
        n = os.path.basename(d); pz = os.path.join(a.out, n + ".npz")
        if (want and n not in want) or os.path.exists(pz):
            continue
        t0 = time.time()
        meta, gt, _ = kp.load_scene(d)
        nF = len(meta["frames"]); W = meta["resolution"][0]
        FR = list(range(nF)) if a.frames == "all" else [int(x) for x in a.frames.split(",")]
        assert FR[0] == 0 and FR[-1] == nF - 1, FR
        q_idx, q_xy, c, _ = xe.query_set(meta, gt, si)
        x1 = c["x1"][q_idx].astype(np.float64)                          # exact native query pixels
        vid = load_video(d, FR)
        extra = {}
        if a.model == "cotracker3":
            tr, vis, cf = track(vid, x1)
            x2, x20 = tr[-1].astype(np.float64), tr[0].astype(np.float64)
            extra = dict(vis_q=vis[-1].astype(np.float32), conf_q=cf[-1].astype(np.float32),
                         vis_q_all=vis.astype(np.float16), conf_q_all=cf.astype(np.float16))
        else:
            if a.raft_mode == "direct":
                fl = flow(vid[0], vid[-1]); x2 = x1 + sample(fl, x1)
                bl = flow(vid[-1], vid[0]); fb = np.linalg.norm(x2 + sample(bl, x2) - x1, axis=1)
                extra = dict(fb_err_q=fb.astype(np.float32), conf_q=(-fb).astype(np.float32))
            else:
                x2 = x1.copy()
                for k in range(len(FR) - 1):
                    x2 = x2 + sample(flow(vid[k], vid[k + 1]), x2)
            x20 = x1.copy()
        save_npz(pz, q_idx=q_idx, q_xy=q_xy, frames=np.array(FR), x2_q512=x2.astype(np.float32),
                 x2_q=to518c(x2, W).astype(np.float32), x2_q0=to518c(x20, W).astype(np.float32), **extra)
        n_done += 1
        print(n, f"{time.time() - t0:.1f}s", flush=True)
    print("scenes", n_done, f"total {time.time() - t_all:.0f}s", "peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

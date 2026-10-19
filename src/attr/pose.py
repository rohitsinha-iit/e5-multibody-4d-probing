"""Camera pose metrics (unified plan section 6 and 5.4).

Poses are camera-to-world 4x4 (any consistent convention shared by pred and GT).
ATE      RMSE of camera centres after Sim(3) Umeyama alignment of predicted to GT centres.
RPE-T    mean translation error of consecutive relative poses, after applying the ATE Sim(3) scale to the prediction.
RPE-R    mean rotation error (deg) of consecutive relative poses.
Spurious ego-motion (frozen-camera variants A and C): per view t, the rotation angle (deg) of the predicted relative
pose of view t to view 0, and its translation norm divided by the median predicted frame-0 depth. Report C against A
(spurious_vs_A) to remove the model's baseline jitter on a static video.
"""
import numpy as np


def umeyama_sim3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    var = (A ** 2).sum(1).mean()
    s = float(np.trace(np.diag(S) @ D) / var) if var > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def align_sim3(pred, gt):
    """Returns (s, R, t, aligned pred poses) with the Sim(3) fit on camera centres."""
    P, G = np.asarray(pred, float), np.asarray(gt, float)
    s, R, t = umeyama_sim3(P[:, :3, 3], G[:, :3, 3])
    A = P.copy()
    A[:, :3, :3] = R @ P[:, :3, :3]
    A[:, :3, 3] = (s * (R @ P[:, :3, 3].T)).T + t
    return s, R, t, A


def ate(pred, gt):
    _, _, _, A = align_sim3(pred, gt)
    return float(np.sqrt(np.mean(np.sum((A[:, :3, 3] - np.asarray(gt)[:, :3, 3]) ** 2, 1))))


def rpe(pred, gt, aligned=True):
    """Consecutive-view RPE. Returns (rpe_t, rpe_r_deg)."""
    P = align_sim3(pred, gt)[3] if aligned else np.asarray(pred, float)
    G = np.asarray(gt, float)
    et, er = [], []
    for i in range(len(P) - 1):
        rp = np.linalg.inv(P[i]) @ P[i + 1]; rg = np.linalg.inv(G[i]) @ G[i + 1]
        E = np.linalg.inv(rg) @ rp
        et.append(np.linalg.norm(E[:3, 3])); er.append(rot_angle_deg(E[:3, :3]))
    return float(np.mean(et)), float(np.mean(er))


def pose_metrics(pred, gt):
    t, r = rpe(pred, gt)
    return dict(ate=ate(pred, gt), rpe_t=t, rpe_r=r)


def spurious_ego_motion(pred, median_depth0):
    """Per view t >= 1: (rotation deg, translation / median frame-0 depth) of inv(P_0) P_t."""
    P = np.asarray(pred, float); rel = np.linalg.inv(P[0])[None] @ P
    rot = np.array([rot_angle_deg(r[:3, :3]) for r in rel[1:]])
    tr = np.linalg.norm(rel[1:, :3, 3], axis=1) / float(median_depth0)
    return dict(rot_deg=rot, trans_rel=tr, rot_deg_mean=float(rot.mean()), trans_rel_mean=float(tr.mean()))


def spurious_vs_A(pred_C, depth_C, pred_A, depth_A):
    """Spurious ego-motion of variant C minus that of variant A (same base scene), per view and mean."""
    c = spurious_ego_motion(pred_C, depth_C); a = spurious_ego_motion(pred_A, depth_A)
    return dict(rot_deg=c["rot_deg"] - a["rot_deg"], trans_rel=c["trans_rel"] - a["trans_rel"],
                rot_deg_mean=c["rot_deg_mean"] - a["rot_deg_mean"], trans_rel_mean=c["trans_rel_mean"] - a["trans_rel_mean"],
                C=c, A=a)


def windows(n_frames, length=8, stride=1, step=None):
    """Indices of consecutive-frame windows (5.4: 8 frames at each set's stride). step = window start spacing in
    frames (default: non-overlapping windows, step = (length - 1) * stride + 1)."""
    span = (length - 1) * stride
    step = step or span + 1
    return [list(range(s, s + span + 1, stride)) for s in range(0, n_frames - span, step)]

"""G0k cross-check: re-score 4RC's saved pred_traj.txt with src/attr/pose.py against the repo's own GT loader."""
import os, sys, glob, json
import numpy as np
REPO = os.path.join(os.environ["E5_BIG"], "models/src/4RC"); sys.path[:0] = [REPO]
sys.path.append("/data2/DATA/rohit_sinha/e5/repo/src/attr")
os.chdir(REPO)
import pose as pm
from eval.relpose.metadata import dataset_metadata
from eval.relpose.evo_utils import load_traj
from scipy.spatial.transform import Rotation
out = {}
for ds in sys.argv[2:]:
    md = dataset_metadata[ds]; res_dir = os.path.join(sys.argv[1], ds); rows = {}
    for d in sorted(glob.glob(os.path.join(res_dir, "*", "pred_traj.txt"))):
        seq = os.path.basename(os.path.dirname(d))
        try:
            gt_file = md["gt_traj_func"](md["img_path"], md.get("anno_path"), seq)
            gt = load_traj(gt_traj_file=gt_file, traj_format=md.get("traj_format") or "sintel", stride=1)
            gtt = gt[0] if isinstance(gt, tuple) else gt
            G = np.stack([np.eye(4)] * len(gtt.positions_xyz)); G[:, :3, 3] = gtt.positions_xyz
            G[:, :3, :3] = Rotation.from_quat(gtt.orientations_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
            p = np.loadtxt(d); Pp = np.stack([np.eye(4)] * len(p)); Pp[:, :3, 3] = p[:, 1:4]
            Pp[:, :3, :3] = Rotation.from_quat(p[:, 4:8]).as_matrix()
            n = min(len(G), len(Pp)); rows[seq] = pm.pose_metrics(Pp[:n], G[:n]); rows[seq]["n"] = n
        except Exception as ex:  # noqa
            rows[seq] = dict(error=repr(ex)[:300])
    ok = [r for r in rows.values() if "ate" in r]
    out[ds] = dict(per_seq=rows, mean_ate=float(np.mean([r["ate"] for r in ok])) if ok else None,
                   mean_rpe_t=float(np.mean([r["rpe_t"] for r in ok])) if ok else None,
                   mean_rpe_r=float(np.mean([r["rpe_r"] for r in ok])) if ok else None)
json.dump(out, open(os.path.join(sys.argv[1], "g0k_pose_py_crosscheck.json"), "w"), indent=1)
print(json.dumps({k: {kk: v[kk] for kk in ("mean_ate", "mean_rpe_t", "mean_rpe_r")} for k, v in out.items()}))

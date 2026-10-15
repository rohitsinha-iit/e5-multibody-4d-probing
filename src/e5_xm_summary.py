"""Item 7: cross-model comparison table (Any4D, MapAnything, VGGT, PAGE-4D).
Usage: python e5_xm_summary.py <queue_results_dir> <out_prefix>
Reads <results>/t7_<model>_v2/xm_output_<model>.json and xm_internal_<model>_results.json.
Columns (scene means unless noted; n = scenes with a finite value):
  slope3d          3D object-motion slope from pointmaps (GT correspondence, similarity-aligned); 1 = full motion
  slope3d_track    same with the model's own 2D correspondence (scene flow or tracks) instead of the GT one
  slope3d_flow     Any4D only: from the scene-flow head
  slope2d          2D slope (e5_causal.py definition; Any4D on all pixels, VGGT tracks on the query set)
  sep_ratio        predicted / GT multibody separation (300 points per group)
  epe_static / epe_moving   median px
  motion_auc_in / _last     moving-vs-static linear probe AUC on the view-0 token, input layer and last layer
  pair_auc_last / pair_auc_best   same-rigid-body bilinear probe AUC, last layer and best layer
  copies_sep_last / _best   confound_same: share of moving-copy vs static-copy pairs predicted different
  att_track / att_rigid     attention hit rate on moving-distinguishable patches of the head selected on static hits
                            (cross-fitted), at the global block with the highest track hit; att_track_head_max = best single head
"""
import glob, json, os, sys
import numpy as np
R, outp = sys.argv[1], sys.argv[2]
MODELS = [("any4d", "Any4D (dynamic fine-tune of MapAnything)"), ("mapanything", "MapAnything (static)"),
          ("vggt", "VGGT-1B (static)"), ("page4d", "PAGE-4D (dynamic fine-tune of VGGT)")]


def mean(v):
    v = np.array([x for x in v if x is not None], float); v = v[np.isfinite(v)]
    return (float(v.mean()), int(len(v))) if len(v) else (float("nan"), 0)


rows, by_split = [], []
for m, label in MODELS:
    rec = dict(model=m, label=label)
    fo = os.path.join(R, f"t7_{m}_v2", f"xm_output_{m}.json")
    if os.path.exists(fo):
        orows = [r for r in json.load(open(fo))["rows"] if "error" not in r]
        rec["n_scenes_output"] = len(orows)
        for k in ["slope3d", "slope3d_track", "slope3d_flow", "slope3d_head", "epe3d_static", "align_rmse"]:
            rec[k], _ = mean([r.get(k) for r in orows])
        full = any("slope2d" in r for r in orows); t = "" if full else "_q"
        rec["slope2d_set"] = "all pixels" if full else ("query set" if any("slope2d_q" in r for r in orows) else "n/a")
        for k in ["slope2d", "sep_ratio", "epe_static", "epe_moving"]:
            rec[k], _ = mean([r.get(k + t) for r in orows])
        for sp in sorted({r["split"] for r in orows}):
            sr = [r for r in orows if r["split"] == sp]
            by_split.append(dict(model=m, split=sp, n=len(sr), slope3d=mean([r.get("slope3d") for r in sr])[0],
                                 slope2d=mean([r.get("slope2d" + t) for r in sr])[0], sep_ratio=mean([r.get("sep_ratio" + t) for r in sr])[0]))
    fi = os.path.join(R, f"t7_{m}_v2", f"xm_internal_{m}_results.json")
    if os.path.exists(fi):
        J = json.load(open(fi)); P = J["P"]["rows"]; L = sorted(set(r["layer"] for r in P)) if P else []
        if L:
            per = {l: [r for r in P if r["layer"] == l] for l in L}
            rec["layers"] = L
            rec["motion_auc_in"], _ = mean([r.get("motion_auc_v0") for r in per[L[0]]])
            rec["motion_auc_last"], _ = mean([r.get("motion_auc_v0") for r in per[L[-1]]])
            pa = {l: mean([r.get("pair_auc") for r in per[l]])[0] for l in L}
            cs = {l: mean([r.get("copies_separated") for r in per[l]])[0] for l in L}
            rec["pair_auc_last"] = pa[L[-1]]; lb = max(L, key=lambda l: pa[l] if np.isfinite(pa[l]) else -1)
            rec["pair_auc_best"], rec["pair_auc_best_layer"] = pa[lb], lb
            rec["copies_sep_last"] = cs[L[-1]]; lc = max(L, key=lambda l: cs[l] if np.isfinite(cs[l]) else -1)
            rec["copies_sep_best"], rec["copies_sep_best_layer"] = cs[lc], lc
            rec["probe_motion_auc_by_layer"] = {str(l): mean([r.get("motion_auc_v0") for r in per[l]])[0] for l in L}
            rec["probe_pair_auc_by_layer"] = {str(l): pa[l] for l in L}
        A = J.get("A", {})
        if A.get("sel_track"):
            st_, sr_ = np.array(A["sel_track"], float), np.array(A["sel_rigid"], float)
            gi = int(np.nanargmax(st_)); rec["att_track"], rec["att_rigid"] = float(st_[gi]), float(sr_[gi])
            rec["att_best_block"] = A["layers"][gi]; rec["att_n_global_blocks"] = len(st_)
            rec["att_track_head_max"] = float(np.nanmax(A["track_max_heads"]))
            rec["att_track_minus_rigid_mean"] = float(np.nanmean(st_ - sr_))
    rows.append(rec)
json.dump(dict(models=rows, by_split=by_split), open(outp + ".json", "w"), indent=1)
cols = ["slope3d", "slope3d_track", "slope2d", "sep_ratio", "epe_static", "epe_moving", "motion_auc_in", "motion_auc_last",
        "pair_auc_last", "pair_auc_best", "copies_sep_last", "copies_sep_best", "att_track", "att_rigid", "att_track_head_max"]
f = lambda v: "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
with open(outp + ".csv", "w") as fh:
    fh.write("model," + ",".join(cols) + ",n_scenes_output,slope2d_set,pair_auc_best_layer,att_best_block\n")
    for r in rows:
        fh.write(r["model"] + "," + ",".join(f(r.get(c)) for c in cols) + f",{r.get('n_scenes_output','')},{r.get('slope2d_set','')},{r.get('pair_auc_best_layer','')},{r.get('att_best_block','')}\n")
with open(outp + ".md", "w") as fh:
    ns = sorted({r.get("n_scenes_output", 0) for r in rows})
    fh.write(f"# Item 7: cross-model comparison on v2 (frames 0 to 23), scenes with output metrics per model: {ns}\n\n")
    if max(ns + [0]) < 100:
        fh.write("PROVISIONAL: fewer than the 100 v2 scenes were available for at least one model.\n\n")
    fh.write("| model | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n")
    for r in rows:
        fh.write(f"| {r['label']} | " + " | ".join(f(r.get(c)) for c in cols) + " |\n")
    fh.write("\nslope2d and sep_ratio: Any4D on all pixels (scene-flow head); VGGT-1B on the query set (tracks from the public "
             "facebook/VGGT_tracker_fixed head, whose aggregator weights are identical to VGGT-1B). MapAnything and PAGE-4D have no "
             "2D motion output (n/a; the PAGE-4D checkpoint ships no tracking head). slope3d is defined for all models. Layers are "
             "relative depths 0, 1/6, ..., 1 of each multi-view transformer (Any4D and MapAnything: 24 blocks; VGGT and PAGE-4D: 48).\n\n")
    fh.write("## By split\n\n| model | split | n | slope3d | slope2d | sep_ratio |\n|---|---|---|---|---|---|\n")
    for b in by_split:
        fh.write(f"| {b['model']} | {b['split']} | {b['n']} | {f(b['slope3d'])} | {f(b['slope2d'])} | {f(b['sep_ratio'])} |\n")
print(open(outp + ".md").read())

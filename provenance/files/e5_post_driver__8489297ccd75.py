"""Run post-processing on every scene dir under ./out: flow check, multibody SNR, contact sheet."""
import glob, json, os, sys, traceback
import numpy as np
import e5_kubric_post as kp

root = sys.argv[1] if len(sys.argv) > 1 else "out"
rows = []
os.makedirs("sheets", exist_ok=True)
for d in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.exists(os.path.join(d, "e5_meta.json")):
        rows.append(dict(scene=os.path.basename(d), error="missing e5_meta.json")); continue
    try:
        meta, gt, rgb = kp.load_scene(d)
        epe, order, disp1 = kp.flow_check(meta, gt)
        snr = kp.multibody_snr(meta, gt)
        name = os.path.basename(d)
        n_groups = 1 + len(meta["motions"])
        title = (f"{name}  |  split {meta['split']}  |  GT motions {n_groups}  |  "
                 f"sep {snr['sep_px']:.1f} px  |  SNR@1px {snr['snr_1px']:.1f}  |  SNR@patch {snr['snr_patch14']:.1f}  |  "
                 f"flow check EPE {epe:.2f} px")
        kp.contact_sheet(meta, gt, rgb, os.path.join("sheets", f"{name}.png"), title)
        from PIL import Image
        fr = [Image.fromarray(x).resize((256, 256)) for x in rgb]
        fr[0].save(os.path.join("sheets", f"{name}.gif"), save_all=True, append_images=fr[1:], duration=120, loop=0)
        analytic = {k: v.get("analytic_sep_px") for k, v in meta["motions"].items()}
        rows.append(dict(scene=name, split=meta["split"], gt_motions=n_groups, flow_epe_px=epe, flow_order=order,
                         median_flow_px=disp1, analytic_sep_px=analytic, **{k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in snr.items()}))
    except Exception as e:
        rows.append(dict(scene=os.path.basename(d), error=repr(e), tb=traceback.format_exc()[-800:]))
json.dump(rows, open("scene_summary.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
for r in rows:
    print(r.get("scene"), r.get("error") or f"EPE {r['flow_epe_px']:.2f} sep {r['sep_px']:.1f} snr1 {r['snr_1px']:.1f}")

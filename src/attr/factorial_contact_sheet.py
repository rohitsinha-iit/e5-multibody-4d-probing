#!/usr/bin/env python3
"""Contact sheet for one factorial base scene: rows = variants A, B, C, D; columns = frames; plus GT motion mask overlay."""
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_factorial as rf
ds, split, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cols = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "0,8,16,23").split(",")]
TS = 256; pad = 4; lw = 150; top = 46
names = {"A": "A: camera + objects frozen", "B": "B: objects frozen", "C": "C: camera frozen", "D": "D: original"}
Wd = lw + len(cols) * (TS + pad); Hd = top + 4 * (TS + pad) + 30
img = Image.new("RGB", (Wd, Hd), "white"); dr = ImageDraw.Draw(img)
dr.text((6, 6), f"MotionFactorial {ds} {split}_s{seed}: rows = variants, columns = frames (frame 0 of A, B, C is D's copied frame 0)", fill="black")
for j, f in enumerate(cols):
    dr.text((lw + j * (TS + pad) + TS // 2 - 20, 28), f"frame {f}", fill="black")
info = []
for i, v in enumerate("ABCD"):
    d = rf.target(ds, split, seed, v); y = top + i * (TS + pad)
    dr.text((6, y + TS // 2 - 12), names[v], fill="black")
    gt = np.load(os.path.join(d, "gt.npz")); meta = json.load(open(os.path.join(d, "e5_meta.json")))
    seg = gt["segmentation"][..., 0]
    mov = set(int(s) for s, k in meta["seg_index_map"].items() if k != "dome" and meta["objects"][k]["group"] > 0)
    for j, f in enumerate(cols):
        a = np.array(Image.open(os.path.join(d, f"rgba_{f:05d}.png")))[..., :3].astype(float)
        m = np.isin(seg[f], list(mov))
        # thin outline of moving-object GT mask
        e = m ^ (np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        a[e] = [255, 0, 0]
        img.paste(Image.fromarray(a.astype(np.uint8)).resize((TS, TS), Image.BILINEAR), (lw + j * (TS + pad), y))
    p = json.load(open(os.path.join(d, "frame0_copy.json")))["psnr_rgb_db"] if v != "D" else None
    info.append(f"{v}: frame-0 PSNR rendered vs copied = {p:.1f} dB" if p is not None else "")
dr.text((6, Hd - 22), "red outline = GT mask of moving objects.  " + "   ".join(x for x in info if x), fill="black")
img.save(out); print("saved", out, img.size)

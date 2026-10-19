#!/usr/bin/env python3
"""U1 MotionFactorial renderer (A6000, native Kubric; no docker).

Variants per base scene (unified plan 2.1):
  D  original motion          -> $E5_BIG/data/<set>/out/<split>_s<seed>   (native port of the dataset's generator)
  C  camera frozen            -> $E5_BIG/attr/factorial/<set>/<seed>/C     (kubric_e5_scene_factorial.py --freeze_camera 1)
  B  objects frozen           -> $E5_BIG/attr/factorial/<set>/<seed>/B     (--freeze_objects 1)
  A  both frozen              -> $E5_BIG/attr/factorial/<set>/<seed>/A     (--freeze_camera 1 --freeze_objects 1)
  $E5_BIG/attr/factorial/<set>/<seed>/D is a symlink to the D render.
Frame-0 identity: after A/B/C and D exist, the rendered rgba_00000.png of A/B/C is kept as rgba_00000_rendered.png and
replaced by a byte copy of D's rgba_00000.png; the RGB PSNR between the two is written to frame0_copy.json.
Resumable: a render whose e5_meta.json exists is skipped; a copy whose frame0_copy.json exists is skipped.
Usage:
  render_factorial.py --sets v2,v3float,v4epi --variants D,C,B,A [--only n2_generic_s401,...] [--limit N]
                      [--workers 3] [--cores 28-39] [--res 512] [--frames 24] [--spp 32] [--dry]
"""
import argparse, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E5_BIG = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")
KPY = os.environ.get("E5_KUBRIC_PY", "/data1/rohit_sinha/miniconda3/envs/h4-kubric/bin/python")
GEN = {"native": os.path.join(HERE, "kubric_e5_scene_native.py"),        # f65cfd0b port: v2, v3float
       "v4": os.path.join(HERE, "kubric_e5_scene_v4_native.py"),         # c6e4d25a port: v4epi
       "factorial": os.path.join(HERE, "kubric_e5_scene_factorial.py")}
FLAGS_V2 = ["--grounded", "1", "--size_lo", "1.4", "--size_hi", "2.2"]
FLAGS_FLOAT = FLAGS_V2 + ["--float_height", "0.6", "--no_shadows", "1"]
VFLAGS = {"D": [], "C": ["--freeze_camera", "1"], "B": ["--freeze_objects", "1"],
          "A": ["--freeze_camera", "1", "--freeze_objects", "1"]}


def scene_list(ds):
    """(split, seed) lists exactly as in the 4090 queue (render_v2_pilot/scale.sh, common_extra.sh, v4_common.sh)."""
    out = []
    if ds == "v2":  # out100: pilot s301-s314 (2 per split) + scale s401-s486
        seed = 300
        for S in ["n2_generic", "n3_generic", "camrot", "epi_aligned", "planar", "confound_same", "confound_glued"]:
            for _ in range(2):
                seed += 1; out.append((S, seed))
        extra = dict(n2_generic=8, n3_generic=18, camrot=8, epi_aligned=8, planar=8, confound_same=18, confound_glued=18)
        seed = 400
        for S in ["n2_generic", "n3_generic", "camrot", "epi_aligned", "planar", "confound_same", "confound_glued"]:
            for _ in range(extra[S]):
                seed += 1; out.append((S, seed))
    elif ds == "v3float":
        out += [("epi_aligned", s) for s in range(531, 561)] + [("n2_generic", s) for s in range(561, 571)]
        out += [("confound_same", s) for s in range(571, 581)] + [("n3_generic", s) for s in range(581, 591)]
        out += [("confound_glued", s) for s in range(591, 601)]
    elif ds == "v4epi":  # round 2 rows: epi_accel and epi_const share seeds 801-830
        for s in range(801, 831):
            out += [("epi_accel", s), ("epi_const", s)]
    else:
        raise ValueError(ds)
    return out


def flags_for(ds):
    return FLAGS_V2 if ds == "v2" else FLAGS_FLOAT


def d_dir(ds, split, seed):
    return os.path.join(E5_BIG, "data", ds, "out", f"{split}_s{seed}")


def f_dir(ds, seed, v):
    return os.path.join(E5_BIG, "attr", "factorial", ds, str(seed), v)


def target(ds, split, seed, v):
    return d_dir(ds, split, seed) if v == "D" else f_dir(ds, seed, v)


def render(ds, split, seed, v, a, cores):
    out = target(ds, split, seed, v)
    if os.path.exists(os.path.join(out, "e5_meta.json")):
        return dict(ds=ds, scene=f"{split}_s{seed}", v=v, status="skip")
    gen = GEN["v4"] if (ds == "v4epi") else (GEN["native"] if v == "D" else GEN["factorial"])
    logd = os.path.join(E5_BIG, "logs", "data", "render", ds); os.makedirs(logd, exist_ok=True)
    tmp = out + ".partial"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(os.path.dirname(out), exist_ok=True)
    scratch = os.path.join(E5_BIG, "scratch", "kubric", f"{ds}_{split}_s{seed}_{v}"); shutil.rmtree(scratch, ignore_errors=True)
    cmd = ["taskset", "-c", cores, KPY, gen, "--split", split, "--seed", str(seed),
           "--resolution", f"{a.res}x{a.res}", "--frame_end", str(a.frames), "--samples", str(a.spp),
           "--min_sep_px", "25"] + flags_for(ds) + VFLAGS[v] + ["--job-dir", tmp, "--scratch_dir", scratch]
    log = os.path.join(logd, f"{split}_s{seed}_{v}.txt")
    if a.dry:
        return dict(ds=ds, scene=f"{split}_s{seed}", v=v, status="dry", cmd=" ".join(cmd))
    t0 = time.time(); r0 = os.times()
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=os.path.dirname(gen))
    r1 = os.times(); wall = time.time() - t0; cpu = (r1.children_user - r0.children_user) + (r1.children_system - r0.children_system)
    ok = os.path.exists(os.path.join(tmp, "e5_meta.json"))
    shutil.rmtree(scratch, ignore_errors=True)
    if ok:
        with open(os.path.join(tmp, "render_info.json"), "w") as fh:
            json.dump(dict(cmd=cmd, generator=os.path.basename(gen), wall_s=wall, rc=rc, host="aicoe-a6000", cores=cores,
                           time=time.strftime("%Y-%m-%d %H:%M:%S")), fh)
        shutil.rmtree(out, ignore_errors=True); os.rename(tmp, out)
    return dict(ds=ds, scene=f"{split}_s{seed}", v=v, status="ok" if ok else f"FAILED rc={rc}", wall_s=round(wall, 1), log=log)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def frame0_copy(ds, split, seed):
    """Link D into the factorial tree and copy D's frame 0 RGB into A, B, C. Returns per-variant PSNR dict."""
    from PIL import Image
    D = d_dir(ds, split, seed); res = {}
    if not os.path.exists(os.path.join(D, "e5_meta.json")):
        return res
    link = f_dir(ds, seed, "D"); os.makedirs(os.path.dirname(link), exist_ok=True)
    if not os.path.lexists(link):
        os.symlink(D, link)
    for v in "ABC":
        Vd = f_dir(ds, seed, v); js = os.path.join(Vd, "frame0_copy.json")
        if not os.path.exists(os.path.join(Vd, "e5_meta.json")):
            continue
        if os.path.exists(js):
            res[v] = json.load(open(js))["psnr_rgb_db"]; continue
        f0, keep = os.path.join(Vd, "rgba_00000.png"), os.path.join(Vd, "rgba_00000_rendered.png")
        if not os.path.exists(keep):
            os.rename(f0, keep)
        a = np.array(Image.open(keep))[..., :3]; b = np.array(Image.open(os.path.join(D, "rgba_00000.png")))[..., :3]
        shutil.copyfile(os.path.join(D, "rgba_00000.png"), f0)
        p = psnr(a, b)
        json.dump(dict(scene=f"{split}_s{seed}", variant=v, psnr_rgb_db=p, max_abs_diff=int(np.abs(a.astype(int) - b).max()),
                       source=os.path.join(D, "rgba_00000.png"), rendered_kept_as="rgba_00000_rendered.png"), open(js, "w"))
        res[v] = p
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="v2,v3float,v4epi"); ap.add_argument("--variants", default="D,C,B,A")
    ap.add_argument("--only", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--cores", default="28-39")
    ap.add_argument("--res", type=int, default=512); ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--spp", type=int, default=32); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--status_json", default=os.path.join(E5_BIG, "logs", "data", "render_status.jsonl"))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.cores.split("-")); allc = list(range(lo, hi + 1)); per = len(allc) // a.workers
    slots = [f"{allc[i*per]}-{allc[i*per+per-1]}" for i in range(a.workers)]
    only = set(x for x in a.only.split(",") if x)
    tasks = []
    for ds in a.sets.split(","):
        sl = scene_list(ds)
        if only:
            sl = [x for x in sl if f"{x[0]}_s{x[1]}" in only]
        if a.limit:
            sl = sl[: a.limit]
        for split, seed in sl:
            for v in a.variants.split(","):
                if ds == "v4epi" and v != "D":
                    continue
                tasks.append((ds, split, seed, v))
    print(f"{len(tasks)} renders, {a.workers} workers x {per} cores ({slots})", flush=True)
    import queue, threading
    free = queue.Queue(); [free.put(s) for s in slots]

    def run(t):
        cores = free.get()
        try:
            return render(*t, a, cores)
        finally:
            free.put(cores)

    with ThreadPoolExecutor(a.workers) as ex, open(a.status_json, "a") as st:
        futs = {ex.submit(run, t): t for t in tasks}
        for fu in as_completed(futs):
            r = fu.result(); ds, split, seed, v = futs[fu]
            if not a.dry:
                r["frame0_psnr"] = frame0_copy(ds, split, seed) if ds != "v4epi" else {}
            r["t"] = time.strftime("%Y-%m-%d %H:%M:%S"); print(json.dumps(r), flush=True); st.write(json.dumps(r) + "\n"); st.flush()
    if not a.dry:  # final pass: copies whose D finished after A/B/C
        for ds in a.sets.split(","):
            if ds != "v4epi":
                for split, seed in scene_list(ds):
                    frame0_copy(ds, split, seed)
    print("RENDER_FACTORIAL_DONE", flush=True)


if __name__ == "__main__":
    main()
